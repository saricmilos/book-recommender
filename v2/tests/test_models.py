import numpy as np
import pytest
from scipy.sparse import csr_matrix

from bookrec import models
from bookrec.data import v1_module

from conftest import toy_split


def test_rated_means_ignore_missing_cells_where_scipy_mean_counts_them_as_zero():
    m = csr_matrix(np.array([[8, 0, 0, 10], [0, 0, 0, 0], [0, 6, 0, 0]], dtype=float))
    assert models.rated_means(m).tolist() == [9.0, 0.0, 6.0]
    # v1's mean, the bug: every missing rating is averaged in as a 0.
    assert np.asarray(m.mean(axis=1)).ravel().tolist() == [4.5, 0.0, 1.5]


# Books x users, every rating an 8. Any sensible model predicts 8 everywhere.
ALL_EIGHTS = np.array(
    [
        [8, 8, 0, 0],
        [8, 0, 8, 0],
        [0, 8, 8, 8],
    ],
    dtype=float,
)


def _v1_item_cf_on(books_users: np.ndarray) -> models.V1ItemCF:
    item_cf = v1_module("item_cf")
    matrix = csr_matrix(books_users)
    similarity = item_cf.compute_item_similarity(matrix)
    neighbours = item_cf.get_top_k_items(similarity, k=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        predictions = item_cf.predict_ratings(matrix, similarity, neighbours)
    return models.V1ItemCF(predictions, matrix, similarity, neighbours)


def test_v1_item_cf_as_shipped_is_pulled_towards_zero():
    v1 = _v1_item_cf_on(ALL_EIGHTS)
    # User 3 rated only book 2. v1's means: book 0 = 16/4 = 4, book 2 = 24/4 = 6, so book 0
    # is predicted 4 + (8 - 6) = 6 instead of 8.
    assert v1.predictions[3, 0] == pytest.approx(6.0)


def test_item_cf_corrected_predicts_eight_everywhere():
    assert np.allclose(models.item_cf_corrected(_v1_item_cf_on(ALL_EIGHTS)), 8.0)


def test_user_cf_corrected_predicts_eight_everywhere():
    # Counting a neighbour who did not rate a book would centre that missing 0 to -8.
    rows = [(u, b, 8.0) for b, u in zip(*np.nonzero(ALL_EIGHTS))]
    split = toy_split(rows, n_users=4, n_items=3)
    assert np.allclose(models.user_cf_corrected(split, k=2), 8.0)


def test_bias_baseline_worked_by_hand():
    # Mean 8. Book offsets: book 0 (+2, 0) -> +1, book 1 (-2) -> -2.
    # User offsets after the book offsets: user 0 (+1, 0) -> +0.5, user 1 (-1) -> -1.
    split = toy_split([(0, 0, 10.0), (0, 1, 6.0), (1, 0, 8.0)], n_users=2, n_items=2)
    predictions = models.bias_baseline(split, reg_book=0.0, reg_user=0.0)
    assert predictions[0, 0] == pytest.approx(9.5)
    assert predictions[1, 1] == pytest.approx(5.0)


def test_similarity_scores_add_up_the_lists_of_the_books_a_user_read():
    split = toy_split([(0, 0, 9.0), (0, 1, 3.0)], n_users=1, n_items=4)
    table = csr_matrix(np.array([[0, 0, 0.5, 0], [0, 0, 0.25, 1.0], [0, 0, 0, 0], [0, 0, 0, 0]]))
    assert models.similarity_scores(split, table).tolist() == [[0.0, 0.0, 0.75, 1.0]]


def test_keep_top_n_keeps_each_rows_largest():
    dense = np.array([[0, 3, 1, 2], [5, 0, 4, 1], [1, 2, 0, 9], [2, 7, 3, 0]], dtype=float)
    kept = models.keep_top_n(dense, top_n=2).toarray()
    assert kept.tolist() == [[0, 3, 0, 2], [5, 0, 4, 0], [0, 2, 0, 9], [0, 7, 3, 0]]
