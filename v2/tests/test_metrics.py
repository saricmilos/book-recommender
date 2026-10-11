import numpy as np
import pandas as pd
import pytest

from bookrec.metrics import mae, ranking_metrics, relevant_items, rmse, tied_at_k, top_k


def test_mae_and_rmse():
    assert mae([1, 2, 3], [2, 2, 5]) == pytest.approx(1.0)
    assert rmse([1, 2, 3], [2, 2, 5]) == pytest.approx(np.sqrt(5 / 3))


def test_top_k_leaves_out_training_books():
    scores = np.array([5.0, 9.0, 9.5, 1.0, 7.0])
    assert top_k(scores, exclude=np.array([2]), k=3).tolist() == [1, 4, 0]


def test_top_k_gives_ties_to_the_lower_index():
    assert top_k(np.array([1.0, 3.0, 1.0, 1.0]), exclude=np.array([], dtype=int), k=3).tolist() == [1, 0, 2]


def test_tied_at_k_counts_users_whose_last_place_is_a_tie():
    scores = np.array([[3.0, 2.0, 2.0, 1.0], [3.0, 2.0, 1.0, 0.0], [9.0, 2.0, 2.0, 1.0]])
    train_items = [np.array([], dtype=int), np.array([], dtype=int), np.array([2])]
    # User 0: second place (2.0) ties with book 2. User 1: no tie. User 2: book 2 is a training book.
    assert tied_at_k(scores, train_items, users=[0, 1, 2], k=2) == 1


def test_relevant_items_applies_the_threshold_and_drops_users_with_none():
    test = pd.DataFrame({"user_idx": [0, 0, 1], "item_idx": [3, 4, 5], "book_rating": [8, 3, 2]})
    assert relevant_items(test, threshold=4) == {0: {3}}


def test_ranking_metrics_worked_by_hand():
    # User 0 trained on book 0; scores rank books 1, 3, 2, 4; top 2 = [1, 3]; relevant {3, 4}.
    # User 1 trained on nothing; top 2 = [4, 0] (tie between 0..3 goes to book 0); relevant {4}.
    scores = np.array([[9.0, 8.0, 1.0, 7.0, 0.0], [0.0, 0.0, 0.0, 0.0, 5.0]])
    train_items = [np.array([0]), np.array([], dtype=int)]
    relevant = {0: {3, 4}, 1: {4}}
    popularity = np.array([10, 20, 30, 40, 50])

    m = ranking_metrics(scores, train_items, relevant, popularity, k=2)

    ndcg_user0 = (1 / np.log2(3)) / (1 + 1 / np.log2(3))  # hit at rank 2, two relevant books
    assert m["precision"] == pytest.approx((1 / 2 + 1 / 2) / 2)
    assert m["recall"] == pytest.approx((1 / 2 + 1) / 2)
    assert m["ndcg"] == pytest.approx((ndcg_user0 + 1) / 2)
    assert m["hit_rate"] == pytest.approx(1.0)
    assert m["coverage"] == pytest.approx(4 / 5)  # books 0, 1, 3, 4 were shown
    assert m["mean_popularity"] == pytest.approx(30.0)  # (20+40)/2 and (50+10)/2
    assert m["users"] == 2


def test_a_perfect_ranking_scores_one():
    scores = np.array([[3.0, 2.0, 1.0, 0.0]])
    m = ranking_metrics(scores, [np.array([], dtype=int)], {0: {0, 1}}, np.ones(4), k=2)
    assert m["precision"] == m["recall"] == m["ndcg"] == pytest.approx(1.0)
