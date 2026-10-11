"""Rating error and top-k metrics. tests/test_metrics.py checks each one by hand on small cases."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2)))


def rating_errors(test: pd.DataFrame, predictions: np.ndarray) -> dict[str, float]:
    """MAE and RMSE of a users x books prediction matrix on the held-out ratings."""
    y_pred = predictions[test["user_idx"].to_numpy(), test["item_idx"].to_numpy()]
    y_true = test["book_rating"].to_numpy()
    return {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred)}


def top_k(scores: np.ndarray, exclude: np.ndarray, k: int) -> np.ndarray:
    """The k best-scored books, leaving out `exclude` (the user's training books). Equal scores
    go to the lower index, i.e. title order, which says nothing about relevance; a stable sort
    keeps that the same on every machine."""
    s = np.array(scores, dtype=float)
    s[exclude] = -np.inf
    return np.argsort(-s, kind="stable")[:k]


def tied_at_k(scores: np.ndarray, train_items: list[np.ndarray], users, k: int) -> int:
    """How many of `users` have their k-th recommendation tied with a book ranked below it,
    so that the order of ties decides what they are shown."""
    tied = 0
    for user in users:
        s = np.array(scores[user], dtype=float)
        s[train_items[user]] = -np.inf
        kth = s[top_k(scores[user], train_items[user], k)[-1]]
        tied += int((s == kth).sum() > 1)
    return tied


def relevant_items(test: pd.DataFrame, threshold: float) -> dict[int, set[int]]:
    """Held-out books each user rated at least `threshold`; users with none are left out."""
    liked = test[test["book_rating"] >= threshold]
    return {int(u): set(map(int, items)) for u, items in liked.groupby("user_idx")["item_idx"]}


def ranking_metrics(
    scores: np.ndarray,
    train_items: list[np.ndarray],
    relevant: dict[int, set[int]],
    popularity: np.ndarray,
    k: int = 10,
) -> dict[str, float]:
    """Top-k metrics, averaged over the users in `relevant`.

    precision: share of the k recommendations that are relevant
    recall:    share of the user's relevant books that made the top k
    ndcg:      like recall, but a hit counts more the higher it ranks (binary gains)
    hit_rate:  share of users with at least one relevant book in their top k
    coverage:  share of the catalogue recommended to at least one user
    mean_popularity: training ratings of the average recommended book
    """
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    shown = np.zeros(scores.shape[1], dtype=bool)
    precision, recall, ndcg, hits, popular = [], [], [], [], []
    for user, rel in relevant.items():
        recs = top_k(scores[user], train_items[user], k)
        gains = np.fromiter((i in rel for i in recs), dtype=float, count=len(recs))
        n_hits = gains.sum()
        precision.append(n_hits / k)
        recall.append(n_hits / len(rel))
        hits.append(n_hits > 0)
        ndcg.append((gains * discounts[: len(recs)]).sum() / discounts[: min(len(rel), k)].sum())
        shown[recs] = True
        popular.append(popularity[recs].mean())
    return {
        "precision": float(np.mean(precision)),
        "recall": float(np.mean(recall)),
        "ndcg": float(np.mean(ndcg)),
        "hit_rate": float(np.mean(hits)),
        "coverage": float(shown.mean()),
        "mean_popularity": float(np.mean(popular)),
        "users": len(relevant),
    }
