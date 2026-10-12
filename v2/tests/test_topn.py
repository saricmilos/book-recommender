"""Stage 3's models on small examples worked by hand."""

import numpy as np
import pandas as pd
import pytest

from bookrec import topn
from bookrec.data import Split
from bookrec.interactions import Interactions


def toy(rows: list[tuple[int, int]], n_users: int, n_items: int, n_benchmark: int | None = None) -> Interactions:
    """Interactions from (user_idx, item_idx) pairs; the first n_benchmark users are scored."""
    n_benchmark = n_users if n_benchmark is None else n_benchmark
    train = pd.DataFrame(rows, columns=["user_idx", "item_idx"]).assign(user_id=lambda f: f["user_idx"], rating=np.nan, explicit=False)
    empty = pd.DataFrame(columns=["user_id", "user_idx", "item_idx", "book_rating"])
    split = Split(filtered=empty, train=empty, test=empty, titles=np.array([f"b{i}" for i in range(n_items)]),
                  user_ids=np.arange(n_benchmark), cleaned_rows=0)
    return Interactions(split, train, np.arange(n_users), {}, {})


# Books 0 and 1 always go together; book 2 is popular with everyone else.
PAIRS = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 2), (4, 2), (5, 2), (6, 2), (2, 2), (7, 3)]


def test_popularity_counts_users():
    assert topn.popularity(toy(PAIRS, 8, 4))[0].tolist() == [3, 3, 5, 1]


def test_item_knn_cosine_scores_by_the_users_books():
    data = toy([(0, 0)] + PAIRS[2:], 8, 4)  # user 0 has only book 0
    scores = topn.item_knn(data, k=3)
    c = topn.co_occurrence(data)
    expected = c[0, 1] / np.sqrt(c[0, 0] * c[1, 1])  # cosine of books 0 and 1
    assert scores[0, 1] == pytest.approx(expected)
    assert scores[0, 1] > scores[0, 2] and scores[0, 3] == 0  # book 3's only reader has nothing else


def test_item_knn_keeps_each_books_k_nearest_and_never_itself():
    data = toy(PAIRS, 8, 4)
    one = topn.item_knn(data, k=1)
    assert one[0, 0] > 0 and one[0, 1] > 0  # books 0 and 1 are each other's nearest
    assert one[0, 2] == 0  # book 2 is nobody's nearest from user 0's books
    assert topn.item_knn(data, k=1, cosine=False)[0, 0] == 3  # raw count: 3 users have books 0 and 1


def test_user_knn_never_counts_the_user_and_weights_by_similarity():
    data = toy(PAIRS, 8, 4)
    idx, sim = topn.user_neighbours(data, k=3)
    assert all(u not in idx[u] for u in range(8))
    scores = topn.user_knn(data, k=1)
    assert scores[0, 1] > 0 and scores[0, 3] == 0  # user 0's nearest (1 or 2) has book 1, nobody near has 3


def test_rp3beta_beta_turns_scores_away_from_popular_books():
    data = toy([(0, 0)] + PAIRS[2:] + [(3, 0)], 8, 4)
    plain, penalised = topn.rp3beta(data, alpha=1.0, beta=0.0, k=4), topn.rp3beta(data, alpha=1.0, beta=1.0, k=4)
    ratio = lambda s: s[0, 2] / s[0, 1]  # popular book 2 against book 1 for user 0  # noqa: E731
    assert ratio(penalised) < ratio(plain)


def test_standardise_z_scores_the_books_a_user_could_get_and_zeroes_the_rest():
    z = topn.standardise(np.array([[10.0, 1.0, 2.0, 3.0]]), [np.array([0])])
    assert z[0, 0] == 0  # a training book
    assert z[0, 1:].mean() == pytest.approx(0) and z[0, 1:].std() == pytest.approx(1)
    assert topn.standardise(np.array([[5.0, 5.0]]), [np.array([], dtype=int)]).tolist() == [[0.0, 0.0]]


def test_blend_adds_the_weighted_parts():
    from bookrec.stage3 import BLEND_WEIGHTS, blend

    parts = {"ease": np.array([[1.0]]), "rp3beta": np.array([[2.0]]), "text 5": np.array([[4.0]])}
    assert blend(parts, {"ease": 0.5, "rp3beta": 0.3, "text": 0.2, "text_k": 5})[0, 0] == pytest.approx(0.5 + 0.6 + 0.8)
    assert all(abs(sum(w.values()) - 1) < 1e-9 and w["text"] <= 0.5 for w in BLEND_WEIGHTS)
    assert {"ease": 1.0, "rp3beta": 0.0, "text": 0.0} in BLEND_WEIGHTS  # EASE alone is among the choices


def test_ease_ranks_the_constant_companion_first_and_shrinks_with_lambda():
    data = toy([(0, 0)] + PAIRS[2:], 8, 4)
    small, large = topn.ease(data, lam=0.5), topn.ease(data, lam=1e6)
    assert np.argmax(small[0, 1:]) + 1 == 1  # book 1, which every reader of book 0 has
    assert np.abs(large).max() < 1e-4  # a huge penalty leaves almost no weight
