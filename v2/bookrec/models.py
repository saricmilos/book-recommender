"""The models phase 1 compares. Every function returns a users x books matrix: a predicted
rating, or a score to rank books by.

- Baselines: global, user and book means, the bias baseline, random and most popular.
- v1 as shipped: v1's user- and item-based CF, run by v1's own code.
- v1 corrected: the same two models with only the averaging fixed (see rated_means).
- Similarity rankers: a user's books scored through a book-to-book similarity table, the way
  v1's API recommends; used for the API's own table and for the cosine table v1 measured.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, load_npz

from bookrec.data import MIN_BOOK_RATINGS, MIN_USER_RATINGS, SEED, TEST_FRAC, Split, v1_module
from bookrec.paths import V1_API_MODEL

V1_NEIGHBOURS = 10  # k in both v1 models
V1_API_TOP_N = 50  # similar books kept per book in v1's API table


def rated_means(matrix: csr_matrix) -> np.ndarray:
    """Each row's mean over the ratings it holds.

    v1 used `matrix.mean(axis=1)`, which divides by the full row length and so counts every
    unrated cell as a 0: a book rated 8 and 10 by two of 2,272 users gets a "mean" of 0.008
    instead of 9. That bug is the whole difference between v1 as shipped and v1 corrected.
    """
    m = csr_matrix(matrix)
    counts = np.diff(m.indptr)
    sums = np.asarray(m.sum(axis=1)).ravel()
    return np.divide(sums, counts, out=np.zeros(m.shape[0]), where=counts > 0)


# --- Baselines --------------------------------------------------------------------------


def global_mean(split: Split) -> np.ndarray:
    return np.full((split.n_users, split.n_items), split.train["book_rating"].mean())


def user_mean(split: Split) -> np.ndarray:
    means = split.train.groupby("user_idx")["book_rating"].mean().reindex(range(split.n_users))
    return np.repeat(means.to_numpy()[:, None], split.n_items, axis=1)


def book_mean(split: Split) -> np.ndarray:
    means = split.train.groupby("item_idx")["book_rating"].mean().reindex(range(split.n_items))
    return np.repeat(means.to_numpy()[None, :], split.n_users, axis=0)


def bias_baseline(split: Split, reg_book: float = 25.0, reg_user: float = 10.0) -> np.ndarray:
    """Overall mean + how far each book and each user sits from it, shrunk towards 0 when they
    have few ratings. The shrinkage values are Koren's textbook ones, not tuned on the test set."""
    t = split.train
    mu = t["book_rating"].mean()
    deviation = t["book_rating"] - mu
    by_book = deviation.groupby(t["item_idx"])
    b_book = (by_book.sum() / (reg_book + by_book.count())).reindex(range(split.n_items), fill_value=0.0)
    residual = deviation - b_book.to_numpy()[t["item_idx"].to_numpy()]
    by_user = residual.groupby(t["user_idx"])
    b_user = (by_user.sum() / (reg_user + by_user.count())).reindex(range(split.n_users), fill_value=0.0)
    return np.clip(mu + b_user.to_numpy()[:, None] + b_book.to_numpy()[None, :], 1, 10)


def random_scores(split: Split, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((split.n_users, split.n_items))


def popularity_scores(split: Split) -> np.ndarray:
    return np.broadcast_to(split.popularity().astype(float), (split.n_users, split.n_items))


# --- v1 as shipped ------------------------------------------------------------------------


@dataclass
class V1UserCF:
    predictions: np.ndarray
    model: object  # v1's UserBasedCF, kept for its own precision_recall_at_n


def v1_user_cf(split: Split) -> V1UserCF:
    """v1's UserBasedCF class, run by v1's code from the split onwards."""
    user_cf = v1_module("user_cf")
    model = user_cf.UserBasedCF(
        min_user_ratings=MIN_USER_RATINGS,
        min_book_ratings=MIN_BOOK_RATINGS,
        k_neighbors=V1_NEIGHBOURS,
        random_seed=SEED,
    )
    model.train_test_split(split.filtered.copy(), test_frac=TEST_FRAC)
    _check_same_split(split, model.test_df["user_idx"], model.test_df["book_idx"])
    model.build_train_matrix()
    model.compute_user_similarity()
    model.predict_ratings()
    return V1UserCF(model.pred_matrix, model)


@dataclass
class V1ItemCF:
    predictions: np.ndarray
    matrix: csr_matrix  # books x users
    similarity: np.ndarray  # cosine, books x books
    neighbours: np.ndarray  # each book's 10 most similar books


def v1_item_cf(split: Split) -> V1ItemCF:
    """v1's item-based functions from v1/src/item_cf.py, called as the notebook calls them."""
    item_cf = v1_module("item_cf")
    matrix = item_cf.build_item_user_matrix(
        split.train, rating_col="book_rating", num_items=split.n_items, num_users=split.n_users
    )
    similarity = item_cf.compute_item_similarity(matrix)
    neighbours = item_cf.get_top_k_items(similarity, k=V1_NEIGHBOURS)
    with np.errstate(divide="ignore", invalid="ignore"):  # v1 divides before masking 0/0
        predictions = item_cf.predict_ratings(matrix, similarity, neighbours)
    return V1ItemCF(predictions, matrix, similarity, neighbours)


def _check_same_split(split: Split, user_idx, item_idx) -> None:
    if not (
        np.array_equal(np.asarray(user_idx), split.test["user_idx"].to_numpy())
        and np.array_equal(np.asarray(item_idx), split.test["item_idx"].to_numpy())
    ):
        raise RuntimeError("v1's UserBasedCF split differs from the shared split")


# --- v1 corrected -----------------------------------------------------------------------


def user_cf_corrected(split: Split, k: int = V1_NEIGHBOURS) -> np.ndarray:
    """v1's user-based CF with two fixes and nothing else changed: means over the ratings a
    user gave (rated_means), and only neighbours who rated a book count towards its prediction,
    as v1's item-based model already did. Without the second fix, correct means would turn
    every unrated cell into a rating of -mean."""
    from sklearn.metrics.pairwise import cosine_similarity

    ratings = split.train_matrix()
    similarity = cosine_similarity(ratings)
    np.fill_diagonal(similarity, 0)
    neighbours = np.argsort(-similarity, axis=1)[:, :k]  # v1's choice of neighbours

    means = rated_means(ratings)
    dense = ratings.toarray()
    rated = dense > 0
    centred = np.where(rated, dense - means[:, None], 0.0)
    predictions = np.empty_like(dense)
    for u in range(split.n_users):
        nb = neighbours[u]
        s = similarity[u, nb]
        num = s @ centred[nb]
        den = np.abs(s) @ rated[nb]
        predictions[u] = means[u] + np.divide(num, den, out=np.zeros(split.n_items), where=den > 0)
    return predictions


def item_cf_corrected(v1: V1ItemCF) -> np.ndarray:
    """v1's item-based prediction (item_cf.predict_ratings) with book means over the ratings
    each book received; the same matrix, similarities, neighbours and formula as v1."""
    means = rated_means(v1.matrix)
    dense = v1.matrix.toarray()
    rated = dense > 0
    n_items, n_users = dense.shape
    predictions = np.empty((n_users, n_items))
    for i in range(n_items):
        nb = v1.neighbours[i]
        s = v1.similarity[i, nb]
        num = (s[:, None] * (dense[nb] - means[nb][:, None]) * rated[nb]).sum(axis=0)
        den = (np.abs(s)[:, None] * rated[nb]).sum(axis=0)
        predictions[:, i] = means[i] + np.divide(num, den, out=np.zeros(n_users), where=den > 0)
    return predictions


# --- Similarity rankers -------------------------------------------------------------------


def v1_api_similarity() -> csr_matrix:
    """The table v1's API serves: row j holds the 50 books shown next to book j."""
    return load_npz(V1_API_MODEL).tocsr()


def rebuild_v1_api_similarity(split: Split, top_n: int = V1_API_TOP_N) -> csr_matrix:
    """Cell [112] of v1's CF notebook, which wrote the API's table: for each book, the top_n
    books by raw dot product of their rating columns. (The notebook evaluated cosine; the API
    serves dot products, which favour books with many high ratings.)"""
    books_users = split.train_matrix().T.tocsr()
    rows, cols, data = [], [], []
    for i in range(split.n_items):
        sims = books_users[i].dot(books_users.T).toarray().ravel()
        sims[i] = 0
        top = np.argpartition(-sims, top_n)[:top_n]
        top = top[np.argsort(-sims[top])]
        rows.extend([i] * len(top))
        cols.extend(top)
        data.extend(sims[top])
    return csr_matrix((data, (rows, cols)), shape=(split.n_items, split.n_items))


def keep_top_n(similarity: np.ndarray, top_n: int = V1_API_TOP_N) -> csr_matrix:
    """A dense book-to-book table cut to each book's top_n, as v1 cut the API's table."""
    n = similarity.shape[0]
    top = np.argpartition(-similarity, top_n, axis=1)[:, :top_n]
    rows = np.repeat(np.arange(n), top_n)
    return csr_matrix((similarity[rows, top.ravel()], (rows, top.ravel())), shape=(n, n))


def similarity_scores(split: Split, similarity: csr_matrix) -> np.ndarray:
    """Score every book by adding up its similarity to each book the user rated in training:
    score(u, i) = sum over the user's books j of similarity[j, i], i.e. how often and how
    strongly i appears in the "similar books" lists of the books the user read."""
    read = csr_matrix((split.train_matrix() > 0).astype(float))
    return (read @ similarity).toarray()
