"""Phase 1: check that v1's code still gives v1's published numbers, then measure baselines and
v1's models on the same split. Writes reports/phase1-evaluation.md and .json.

    python -m bookrec.evaluate
"""

from __future__ import annotations

import json
import platform
from collections.abc import Callable

import numpy as np
import pandas as pd
import scipy
import sklearn

from bookrec import models
from bookrec.data import Split, load_split
from bookrec.metrics import ranking_metrics, rating_errors, relevant_items, tied_at_k
from bookrec.paths import REPORTS

K = 10
RELEVANT_FROM = 4  # v1's rule: a held-out rating of 4 or more makes a book relevant

# v1's published numbers, from v1/notebooks/collaborative_filtering.ipynb, where the metrics
# are printed to 4 decimal places.
V1_PUBLISHED = {
    "Ratings after v1's cleaning": 383_839,
    "Ratings after the 10-and-10 filter": 56_967,
    "Users": 2_272,
    "Books": 2_508,
    "Training ratings": 45_592,
    "Held-out ratings": 11_375,
    "User-based CF: MAE": 7.3627,
    "User-based CF: RMSE": 7.6219,
    "User-based CF: Precision@10": 0.0383,
    "User-based CF: Recall@10": 0.0848,
    "Item-based CF: MAE": 5.2947,
    "Item-based CF: RMSE": 6.4017,
}


def run() -> dict:
    split = load_split()
    train_items = split.train_items()
    popularity = split.popularity()
    relevant = relevant_items(split.test, RELEVANT_FROM)

    rating_rows: list[dict] = []
    ranking_rows: list[dict] = []

    def rate(name: str, predictions: np.ndarray, rank: bool = True) -> None:
        rating_rows.append({"model": name, **rating_errors(split.test, predictions)})
        if rank:
            score(name, predictions)

    def score(name: str, scores: np.ndarray) -> None:
        ranking_rows.append({"model": name, **ranking_metrics(scores, train_items, relevant, popularity, K)})

    score("Random", models.random_scores(split))
    score("Most popular", models.popularity_scores(split))
    rate("Global mean", models.global_mean(split), rank=False)  # one score for every book
    rate("User mean", models.user_mean(split), rank=False)  # same score for all of a user's books
    rate("Book mean", models.book_mean(split))
    rate("Bias baseline", models.bias_baseline(split))

    user_cf = models.v1_user_cf(split)
    rate("v1 user-based CF, as shipped", user_cf.predictions)
    v1_user_precision, v1_user_recall = user_cf.model.precision_recall_at_n(N=K, threshold=RELEVANT_FROM)
    v1_user_ties = tied_at_k(user_cf.predictions, train_items, relevant, K)
    del user_cf
    rate("v1 user-based CF, corrected", models.user_cf_corrected(split))

    item_cf = models.v1_item_cf(split)
    rate("v1 item-based CF, as shipped", item_cf.predictions)
    rate("v1 item-based CF, corrected", models.item_cf_corrected(item_cf))
    cosine_top_n = models.keep_top_n(item_cf.similarity)
    del item_cf

    api = models.v1_api_similarity()
    api_rebuilt_identical = _same_sparse(api, models.rebuild_v1_api_similarity(split))
    score("v1 API table: dot product, top 50", models.similarity_scores(split, api))
    score("Cosine table, top 50 (what v1 measured)", models.similarity_scores(split, cosine_top_n))

    by_name = {r["model"]: r for r in rating_rows}
    rerun = {
        "Ratings after v1's cleaning": split.cleaned_rows,
        "Ratings after the 10-and-10 filter": len(split.filtered),
        "Users": split.n_users,
        "Books": split.n_items,
        "Training ratings": len(split.train),
        "Held-out ratings": len(split.test),
        "User-based CF: MAE": by_name["v1 user-based CF, as shipped"]["mae"],
        "User-based CF: RMSE": by_name["v1 user-based CF, as shipped"]["rmse"],
        "User-based CF: Precision@10": float(v1_user_precision),
        "User-based CF: Recall@10": float(v1_user_recall),
        "Item-based CF: MAE": by_name["v1 item-based CF, as shipped"]["mae"],
        "Item-based CF: RMSE": by_name["v1 item-based CF, as shipped"]["rmse"],
    }
    reproduction = [
        {"what": what, "published": published, "rerun": rerun[what], "match": _matches(published, rerun[what])}
        for what, published in V1_PUBLISHED.items()
    ]
    reproduction.append(
        {
            "what": "API table rebuilt from v1's code equals v1/models/topk_item_sim.npz",
            "published": None,
            "rerun": None,
            "match": api_rebuilt_identical,
        }
    )

    return {
        "split": {
            "fingerprint": split.fingerprint(),
            "users": split.n_users,
            "books": split.n_items,
            "train": len(split.train),
            "test": len(split.test),
        },
        "reproduction": reproduction,
        "rating": rating_rows,
        "ranking": {
            "k": K,
            "relevant_from": RELEVANT_FROM,
            "users": len(relevant),
            "rows": ranking_rows,
            "v1_user_cf_users_tied_at_k": v1_user_ties,
        },
        "data_notes": _data_notes(split),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }


def _matches(published: float, rerun: float) -> bool:
    # Published metrics are rounded to 4 places; counts must be equal.
    if isinstance(published, int):
        return published == rerun
    return round(rerun, 4) == published


def _same_sparse(a, b) -> bool:
    return a.shape == b.shape and bool(abs(a - b).max() == 0)


def _data_notes(split: Split) -> dict:
    pairs = split.filtered.groupby(["user_id", "book_title"]).size()
    train_pairs = split.train[["user_idx", "item_idx"]].drop_duplicates()
    in_both = split.test.merge(train_pairs, on=["user_idx", "item_idx"]).shape[0]
    return {
        "duplicate_user_title_pairs": int((pairs > 1).sum()),
        "largest_training_cell": float(split.train_matrix().max()),
        "held_out_titles_also_in_training": int(in_both),
    }


# --- Report -------------------------------------------------------------------------------


def write_report(results: dict) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "phase1-evaluation.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (REPORTS / "phase1-evaluation.md").write_text(render_markdown(results), encoding="utf-8")


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _fmt(value, spec: str) -> str:
    return "" if value is None else format(value, spec)


def render_markdown(r: dict) -> str:
    s, rk, notes, env = r["split"], r["ranking"], r["data_notes"], r["environment"]
    count = lambda v: _fmt(v, ",") if isinstance(v, int) else _fmt(v, ".4f")  # noqa: E731

    reproduction = _table(
        ["", "v1 notebook", "Re-run now", "Same"],
        [[x["what"], count(x["published"]), count(x["rerun"]), "yes" if x["match"] else "**no**"] for x in r["reproduction"]],
    )
    rating = _table(
        ["Model", "MAE", "RMSE"],
        [[x["model"], f"{x['mae']:.3f}", f"{x['rmse']:.3f}"] for x in r["rating"]],
    )
    ranking = _table(
        ["Model", "Precision@10", "Recall@10", "NDCG@10", "Hit rate", "Coverage", "Avg. popularity"],
        [
            [
                x["model"],
                f"{x['precision']:.4f}",
                f"{x['recall']:.4f}",
                f"{x['ndcg']:.4f}",
                f"{x['hit_rate']:.1%}",
                f"{x['coverage']:.1%}",
                f"{x['mean_popularity']:.1f}",
            ]
            for x in rk["rows"]
        ],
    )
    versions = ", ".join(f"{k} {v}" for k, v in env.items())

    return f"""# Phase 1: evaluation

Generated by `python -m bookrec.evaluate`; edit the code, not this file.

Every model here is measured on the same data and the same split as v1: v1's cleaned
Book-Crossing ratings, keeping users and books with at least 10 ratings, with 20% of each
user's ratings held out (seed 42). The split is produced by v1's own code.

| | |
|---|---|
| Users | {s['users']:,} |
| Books | {s['books']:,} |
| Training ratings | {s['train']:,} |
| Held-out ratings | {s['test']:,} |
| Split fingerprint | `{s['fingerprint']}` |

## v1's numbers, reproduced

v1's own functions, re-run on the data in `datasets/`, against what its notebook printed.

{reproduction}

## Predicting a rating

Error on the held-out ratings, on Book-Crossing's 1-10 scale. Lower is better.

{rating}

"As shipped" is v1's code unchanged. "Corrected" changes only how averages are taken: over
the ratings a user or book has, where v1 divided by every user or book in the matrix and so
counted each missing rating as a 0. The user-based model also stops counting neighbours who
didn't rate the book, which v1's item-based model already did.

## Recommending 10 books

Each of the {rk['users']:,} users with a held-out rating of {rk['relevant_from']} or more gets
{rk['k']} books they did not rate in training. A recommendation is a hit when it is one of
those held-out books (v1's rule). Rating models rank books by predicted rating; the two
table models score each book by its similarity to the user's training books. Equal scores
go to title order. Higher is better, except that a higher average popularity (training
ratings per recommended book) means more of the same few bestsellers.

{ranking}

v1's own function gives its user-based CF the slightly different 0.0383 and 0.0848 above
because it orders tied scores differently, and ties matter: for
{rk['v1_user_cf_users_tied_at_k']:,} of the {rk['users']:,} users, the tenth book is tied
with at least one book ranked below it.

## Notes on the data

- {notes['duplicate_user_title_pairs']:,} users rated two or more editions that share a title.
  v1 adds those ratings together in one cell, so a cell can reach
  {notes['largest_training_cell']:.0f} on a 1-10 scale.
- {notes['held_out_titles_also_in_training']:,} held-out ratings are for a title the same user
  also rated in training. No model can recommend those, since training books are excluded.
- Both are left as v1 had them here, so that v1's numbers reproduce. Phase 2 cleans them up.

Environment: {versions}.
"""


def main() -> None:
    results = run()
    write_report(results)
    failed = [x["what"] for x in results["reproduction"] if not x["match"]]
    print(f"Wrote {REPORTS / 'phase1-evaluation.md'}")
    if failed:
        raise SystemExit("v1 did not reproduce: " + "; ".join(failed))


if __name__ == "__main__":
    main()
