"""Stage 3's training data: every Book-Crossing interaction with stage 2's books.

The benchmark stays fixed: v1's 2,272 users, stage 2's 2,405 books and the same held-out
ratings. What models may learn from grows, by one rule:

    Training may use any interaction in Book-Crossing with one of the 2,405 books, by any
    user, rated (1-10) or not (0: the user had the book but gave no rating), except one that
    covers a user and book whose rating is held out, through any edition of the book.

Editions: an ISBN the benchmark already uses keeps the book stage 2 gave it. Any other ISBN in
Books.csv belongs to a book when its title has that book's title key and its author loosely
matches the book's; a title shared by several books goes to the one whose author matches, and
an edition matching none, or more than one, is left out.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from bookrec.books import book_rows, clean_split, same_author, title_key
from bookrec.data import Split, load_split
from bookrec.paths import DATASETS


@dataclass(frozen=True)
class Interactions:
    """Training interactions, one row per user and book: user_id, user_idx, item_idx, rating
    (the mean of the user's 1-10 ratings of the book, NaN if none) and explicit. The
    benchmark's users come first, with the split's user_idx, so row u < split.n_users of
    any users x books matrix is the split's user u."""

    split: Split
    train: pd.DataFrame
    user_ids: np.ndarray
    editions: dict[str, int]  # ISBN -> item_idx
    counts: dict

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return self.split.n_items

    @property
    def n_benchmark_users(self) -> int:
        return self.split.n_users

    def matrix(self) -> csr_matrix:
        """Users x books, 1 for every interaction, rated or not."""
        t = self.train
        return csr_matrix(
            (np.ones(len(t)), (t["user_idx"], t["item_idx"])), shape=(self.n_users, self.n_items)
        )

    def train_items(self) -> list[np.ndarray]:
        """Every book each benchmark user interacted with in training, by user_idx: none of
        them is recommended to that user."""
        bench = self.train[self.train["user_idx"] < self.n_benchmark_users]
        by_user = bench.groupby("user_idx")["item_idx"].unique()
        return [np.asarray(by_user.get(u, []), dtype=int) for u in range(self.n_benchmark_users)]

    def popularity(self) -> np.ndarray:
        """Users who interacted with each book in training."""
        return np.bincount(self.train["item_idx"], minlength=self.n_items)

    def without(self, pairs: pd.DataFrame) -> Interactions:
        """The same data without the given (user_idx, item_idx) pairs: for tuning on a
        validation half carved out of training."""
        drop = self.train.merge(pairs[["user_idx", "item_idx"]], on=["user_idx", "item_idx"], how="left", indicator=True)
        keep = (drop["_merge"] == "left_only").to_numpy()
        return Interactions(self.split, self.train[keep].reset_index(drop=True), self.user_ids, self.editions, self.counts)


def benchmark_only(split: Split) -> Interactions:
    """The benchmark's own training ratings as Interactions, nothing added: to measure what the
    extra interactions do, with everything else the same."""
    train = split.train[["user_id", "user_idx", "item_idx", "book_rating"]].rename(columns={"book_rating": "rating"})
    train = train.assign(explicit=True).sort_values(["user_idx", "item_idx"]).reset_index(drop=True)
    return Interactions(split, train, split.user_ids, {}, {})


def validation_split(data: Interactions, frac: float = 0.2, seed: int = 7) -> tuple[Interactions, pd.DataFrame]:
    """For tuning: per benchmark user, `frac` of their benchmark training ratings (at least one)
    set aside as a validation set, and the training data without them. Settings are chosen on
    this set, so the held-out ratings are used once, for the final score."""
    rng = np.random.RandomState(seed)
    bench = data.split.train
    picked: list[int] = []
    for _, group in bench.groupby("user_idx"):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        picked.extend(idx[: max(1, int(round(frac * len(idx))))])
    validation = bench.loc[sorted(picked), ["user_idx", "item_idx", "book_rating"]].reset_index(drop=True)
    return data.without(validation), validation


def map_editions(books: pd.DataFrame, rows: pd.DataFrame, catalogue: pd.DataFrame) -> dict[str, int]:
    """ISBN -> item_idx for every edition of stage 2's books (see the module's docstring).
    books: stage 2's book table; rows: stage 2's rated rows (isbn, book_key); catalogue:
    Books.csv's isbn, title and author."""
    item_idx = {k: i for i, k in enumerate(books["book_key"])}
    editions = {isbn: item_idx[key] for isbn, key in zip(rows["isbn"], rows["book_key"])}
    by_title: dict[str, list[int]] = {}
    for i, key in enumerate(books["book_key"]):
        by_title.setdefault(key.rsplit(" | ", 1)[0], []).append(i)
    authors = books["author"].to_numpy()
    new = catalogue[~catalogue["isbn"].isin(editions)]
    keys = new["title"].fillna("").map(title_key)
    for isbn, tkey, author in zip(new["isbn"], keys, new["author"].fillna("")):
        candidates = by_title.get(tkey)
        if candidates:
            matching = [i for i in candidates if same_author(authors[i], author)]
            if len(matching) == 1:
                editions[isbn] = matching[0]
    return editions


def build(split: Split, books: pd.DataFrame, rows: pd.DataFrame, catalogue: pd.DataFrame, ratings: pd.DataFrame) -> Interactions:
    """split, books and rows from stage 2; catalogue: Books.csv (isbn, title, author); ratings:
    Ratings.csv (user_id, isbn, rating)."""
    editions = map_editions(books, rows, catalogue)
    r = ratings[ratings["isbn"].isin(editions)].assign(item_idx=lambda f: f["isbn"].map(editions))
    rated = r[r["rating"] > 0]
    pairs = r.groupby(["user_id", "item_idx"], as_index=False).size()[["user_id", "item_idx"]]
    pairs = pairs.merge(
        rated.groupby(["user_id", "item_idx"], as_index=False)["rating"].mean(), on=["user_id", "item_idx"], how="left"
    )

    def pair_set(frame: pd.DataFrame) -> pd.Series:
        return frame["user_id"].astype(str) + "|" + frame["item_idx"].astype(str)

    held_out = set(pair_set(split.test))
    benchmark_train = set(pair_set(split.train))
    key = pair_set(pairs)
    is_held_out = key.isin(held_out).to_numpy()
    extra = pairs[~is_held_out & ~key.isin(benchmark_train).to_numpy()]

    bench = split.train[["user_id", "item_idx", "book_rating"]].rename(columns={"book_rating": "rating"})
    train = pd.concat([bench, extra], ignore_index=True)
    train["explicit"] = train["rating"].notna()
    others = np.setdiff1d(np.unique(extra["user_id"]), split.user_ids)
    user_ids = np.concatenate([split.user_ids, others])
    train["user_idx"] = train["user_id"].map({u: i for i, u in enumerate(user_ids)})
    train = train[["user_id", "user_idx", "item_idx", "rating", "explicit"]].sort_values(["user_idx", "item_idx"])

    is_bench_user = extra["user_id"].isin(set(split.user_ids))
    counts = {
        "editions": len(editions),
        "editions_in_benchmark": int(rows["isbn"].nunique()),
        "benchmark_training_ratings": len(bench),
        "extra_benchmark_users_rated": int((is_bench_user & extra["rating"].notna()).sum()),
        "extra_benchmark_users_unrated": int((is_bench_user & extra["rating"].isna()).sum()),
        "extra_other_users_rated": int((~is_bench_user & extra["rating"].notna()).sum()),
        "extra_other_users_unrated": int((~is_bench_user & extra["rating"].isna()).sum()),
        "other_users": len(others),
        "left_out_as_held_out": int(is_held_out.sum()),
        "held_out_ratings": len(split.test),
        "training_interactions": len(train),
    }
    return Interactions(split, train.reset_index(drop=True), user_ids, editions, counts)


@lru_cache(maxsize=1)
def load_interactions() -> Interactions:
    v1 = load_split()
    split, books, _ = clean_split(v1)
    catalogue = pd.read_csv(
        DATASETS / "Books.csv", usecols=["ISBN", "Book-Title", "Book-Author"], dtype=str, low_memory=False
    ).rename(columns={"ISBN": "isbn", "Book-Title": "title", "Book-Author": "author"})
    ratings = pd.read_csv(DATASETS / "Ratings.csv", dtype={"ISBN": str}).rename(
        columns={"User-ID": "user_id", "ISBN": "isbn", "Book-Rating": "rating"}
    )
    return build(split, books, book_rows(v1), catalogue, ratings)
