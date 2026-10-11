"""The ratings every v2 model is measured on: v1's cleaned data and v1's train/test split,
produced by v1's own code so that v2's numbers stay comparable with v1's notebooks."""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import sys
from dataclasses import dataclass
from functools import lru_cache
from types import ModuleType

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from bookrec.paths import DATASETS, V1

# v1's settings, from v1/notebooks/collaborative_filtering.ipynb
MIN_USER_RATINGS = 10
MIN_BOOK_RATINGS = 10
TEST_FRAC = 0.2
SEED = 42


@contextlib.contextmanager
def _v1_on_path():
    sys.path.insert(0, str(V1))
    try:
        yield
    finally:
        sys.path.remove(str(V1))


def v1_module(name: str) -> ModuleType:
    """Import v1/src/<name>.py as v1 imports it (`src.<name>`), reading v1/ and never writing to it."""
    with _v1_on_path():
        return importlib.import_module(f"src.{name}")


@dataclass(frozen=True)
class Split:
    """v1's split. Rows carry user_id, book_title, book_rating, user_idx and item_idx; the indices
    are v1's label encoders, so item_idx i is the i-th title in sorted order, as in v1's API.
    Shared by every caller: treat the frames as read-only."""

    filtered: pd.DataFrame  # v1's filtered ratings, index kept: v1's split shuffles these labels
    train: pd.DataFrame
    test: pd.DataFrame
    titles: np.ndarray  # item_idx -> title
    user_ids: np.ndarray  # user_idx -> user_id
    cleaned_rows: int  # ratings left by v1's cleaning, before the 10-and-10 filter

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.titles)

    def train_matrix(self) -> csr_matrix:
        """Users x books, built as v1 builds it: when a user rated two editions with the same
        title, the two ratings add up in one cell."""
        t = self.train
        return csr_matrix(
            (t["book_rating"].to_numpy(dtype=float), (t["user_idx"], t["item_idx"])),
            shape=(self.n_users, self.n_items),
        )

    def train_items(self) -> list[np.ndarray]:
        """The books each user rated in training, by user_idx."""
        by_user = self.train.groupby("user_idx")["item_idx"].unique()
        return [np.asarray(by_user.get(u, []), dtype=int) for u in range(self.n_users)]

    def popularity(self) -> np.ndarray:
        """Training ratings per book."""
        return np.bincount(self.train["item_idx"], minlength=self.n_items)

    def fingerprint(self) -> str:
        """A short hash of the split's rows. If it changes, results are no longer comparable."""
        h = hashlib.sha256()
        for name, frame in (("train", self.train), ("test", self.test)):
            cols = ["user_id", "book_title", "book_rating"]
            rows = frame[cols].sort_values(cols).to_csv(index=False)
            h.update(name.encode())
            h.update(rows.encode("utf-8"))
        return h.hexdigest()[:16]


@lru_cache(maxsize=1)
def load_split() -> Split:
    """Read datasets/, then clean, filter, split and encode with v1's own functions."""
    dataloader = v1_module("dataloader")
    preprocess = v1_module("preprocess_user_books_ratings")
    item_cf = v1_module("item_cf")

    csvs = dataloader.load_all_csvs_from_folder(DATASETS, low_memory=False)
    cleaned = preprocess.preprocess_books_ratings_users(csvs["Books"], csvs["Ratings"], csvs["Users"])
    filtered = item_cf.filter_users_items(
        cleaned, min_user_ratings=MIN_USER_RATINGS, min_item_ratings=MIN_BOOK_RATINGS
    )
    train, test = item_cf.train_test_split_by_user(filtered, test_frac=TEST_FRAC, random_seed=SEED)
    encoders = item_cf.CFEncoders()
    train = encoders.fit(train)
    test = encoders.transform(test)
    return Split(
        filtered=filtered,
        train=train,
        test=test,
        titles=encoders.item_encoder.classes_,
        user_ids=encoders.user_encoder.classes_,
        cleaned_rows=len(cleaned),
    )
