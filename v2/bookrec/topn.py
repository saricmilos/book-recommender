"""Stage 3's models: recommenders learned from interactions (rated or not), scored for the
benchmark's users. Every model takes an Interactions and returns a benchmark users x books
score matrix; books the user already has in training are left out when ranking.

- popularity: how many users have the book.
- item_knn: a book-to-book table (cosine of the users who have the books, optionally shrunk
  towards 0 for books with few users), keeping each book's k nearest; a user's score for a
  book adds up its similarity to the user's books. With raw co-occurrence counts instead of
  cosine and k = 50 it is v1's API table.
- user_knn: the k most similar users (cosine), adding up the books they have, weighted by
  similarity; with k = 10 it is what v1's user-based model, as shipped, amounted to.
- rp3beta: a random walk book -> user -> book, with each step's probability raised to alpha
  and each target book's score divided by its popularity to the power beta, which turns
  recommendations away from bestsellers (Paudel et al., 2017).
- ease: the book-to-book weights that best rebuild each user's books from their other books,
  in closed form with a ridge penalty lambda (Steck, 2019).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, diags

from bookrec.interactions import Interactions


def _benchmark_rows(data: Interactions) -> csr_matrix:
    return data.matrix()[: data.n_benchmark_users]


def _keep_top_k(weights: np.ndarray, k: int) -> np.ndarray:
    """Each row's k largest weights, the rest set to 0 (rows are "from" books)."""
    if k >= weights.shape[1]:
        return weights
    keep = np.argpartition(-weights, k, axis=1)[:, :k]
    out = np.zeros_like(weights)
    rows = np.arange(weights.shape[0])[:, None]
    out[rows, keep] = weights[rows, keep]
    return out


def popularity(data: Interactions) -> np.ndarray:
    return np.broadcast_to(data.popularity().astype(float), (data.n_benchmark_users, data.n_items))


def co_occurrence(data: Interactions) -> np.ndarray:
    x = data.matrix()
    return (x.T @ x).toarray()


def item_knn(data: Interactions, k: int = 50, shrink: float = 0.0, cosine: bool = True, counts: np.ndarray | None = None) -> np.ndarray:
    """`counts` can pass a precomputed co_occurrence(data) when trying several settings."""
    c = co_occurrence(data) if counts is None else counts
    users = np.diag(c).copy()
    sim = c / (np.sqrt(np.outer(users, users)) + shrink + 1e-12) if cosine else c.astype(float)
    np.fill_diagonal(sim, 0)
    return (_benchmark_rows(data) @ _keep_top_k(sim, k)).astype(float)


def user_neighbours(data: Interactions, k: int, chunk: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """For each benchmark user, the k most similar users among all users (cosine), and those
    similarities; a user is never their own neighbour."""
    x = data.matrix()
    norms = np.sqrt(np.asarray(x.sum(axis=1)).ravel()) + 1e-12
    xn = diags(1 / norms) @ x
    idx = np.empty((data.n_benchmark_users, k), dtype=int)
    sim = np.empty((data.n_benchmark_users, k))
    for start in range(0, data.n_benchmark_users, chunk):
        stop = min(start + chunk, data.n_benchmark_users)
        s = (xn[start:stop] @ xn.T).toarray()
        s[np.arange(stop - start), np.arange(start, stop)] = -np.inf  # not oneself
        top = np.argpartition(-s, k, axis=1)[:, :k]
        rows = np.arange(stop - start)[:, None]
        order = np.argsort(-s[rows, top], axis=1)
        idx[start:stop] = top[rows, order]
        sim[start:stop] = s[rows, top][rows, order]
    return idx, sim


def user_knn(data: Interactions, k: int = 10, neighbours: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """`neighbours` can pass user_neighbours(data, k_max) for any k <= k_max."""
    idx, sim = user_neighbours(data, k) if neighbours is None else neighbours
    idx, sim = idx[:, :k], np.maximum(sim[:, :k], 0)
    x = data.matrix()
    scores = np.zeros((data.n_benchmark_users, data.n_items))
    for u in range(data.n_benchmark_users):
        scores[u] = sim[u] @ x[idx[u]].toarray()
    return scores


def rp3beta_walk(data: Interactions, alpha: float) -> np.ndarray:
    """Book -> user -> book transition probabilities, each step raised to alpha."""
    x = data.matrix()
    user_to_book = diags(1 / np.maximum(np.asarray(x.sum(axis=1)).ravel(), 1)) @ x
    book_to_user = diags(1 / np.maximum(np.asarray(x.sum(axis=0)).ravel(), 1)) @ x.T
    return (book_to_user.power(alpha) @ user_to_book.power(alpha)).toarray()


def rp3beta(data: Interactions, alpha: float = 1.0, beta: float = 0.5, k: int = 100, walk: np.ndarray | None = None) -> np.ndarray:
    """`walk` can pass rp3beta_walk(data, alpha) when trying several beta and k."""
    w = rp3beta_walk(data, alpha) if walk is None else walk.copy()
    w /= np.power(np.maximum(data.popularity(), 1), beta)[None, :]
    np.fill_diagonal(w, 0)
    return (_benchmark_rows(data) @ _keep_top_k(w, k)).astype(float)


def ease(data: Interactions, lam: float = 500.0, counts: np.ndarray | None = None) -> np.ndarray:
    g = (co_occurrence(data) if counts is None else counts).astype(float)
    g[np.diag_indices_from(g)] += lam
    p = np.linalg.inv(g)
    b = -p / np.diag(p)[None, :]
    np.fill_diagonal(b, 0)
    return (_benchmark_rows(data) @ b).astype(float)
