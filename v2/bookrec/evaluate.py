"""Measure every model and write the reports.

Stage 1: check that v1's code still gives v1's published numbers, then score baselines and
v1's models on v1's split (reports/phase1-evaluation.md).
Stage 2: the same models on the same ratings, with books identified properly
(reports/phase2-books.md, and data/books.csv listing what every book merged).
Stage 3: more training data and tuned models, on the same held-out ratings
(reports/phase3-models.md; the code is in bookrec/stage3.py).

    python -m bookrec.evaluate
"""

from __future__ import annotations

import json
import platform

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.sparse import csr_matrix

from bookrec import models, openlibrary
from bookrec.books import MIN_TRAINING_RATINGS_TO_SPLIT, clean_split
from bookrec.data import Split, load_split
from bookrec.metrics import ranking_metrics, rating_errors, relevant_items, tied_at_k
from bookrec.paths import DATA, REPORTS
from bookrec.stage3 import render_stage3, run_stage3

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

API_TABLE = "v1 API table: dot product, top 50"
COSINE_TABLE = "Cosine table, top 50 (what v1 measured)"


def score_models(split: Split, api_table: csr_matrix) -> dict:
    """Every model on one split: rating errors, and top-10 metrics for those that can rank.
    `api_table` is the dot-product table to score: v1's own file on v1's split, the same
    method rebuilt on any other split."""
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

    score(API_TABLE, models.similarity_scores(split, api_table))
    score(COSINE_TABLE, models.similarity_scores(split, cosine_top_n))

    return {
        "split": {
            "fingerprint": split.fingerprint(),
            "users": split.n_users,
            "books": split.n_items,
            "train": len(split.train),
            "test": len(split.test),
        },
        "rating": rating_rows,
        "ranking": {
            "k": K,
            "relevant_from": RELEVANT_FROM,
            "users": len(relevant),
            "rows": ranking_rows,
            "v1_user_cf_users_tied_at_k": v1_user_ties,
        },
        "v1_user_cf_precision_recall": [float(v1_user_precision), float(v1_user_recall)],
    }


def run() -> dict:
    """Stage 1, on v1's split."""
    split = load_split()
    api = models.v1_api_similarity()
    results = score_models(split, api)

    by_name = {r["model"]: r for r in results["rating"]}
    precision, recall = results.pop("v1_user_cf_precision_recall")
    rerun = {
        "Ratings after v1's cleaning": split.cleaned_rows,
        "Ratings after the 10-and-10 filter": len(split.filtered),
        "Users": split.n_users,
        "Books": split.n_items,
        "Training ratings": len(split.train),
        "Held-out ratings": len(split.test),
        "User-based CF: MAE": by_name["v1 user-based CF, as shipped"]["mae"],
        "User-based CF: RMSE": by_name["v1 user-based CF, as shipped"]["rmse"],
        "User-based CF: Precision@10": precision,
        "User-based CF: Recall@10": recall,
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
            "match": _same_sparse(api, models.rebuild_v1_api_similarity(split)),
        }
    )
    return {
        "split": results["split"],
        "reproduction": reproduction,
        "rating": results["rating"],
        "ranking": results["ranking"],
        "data_notes": _data_notes(split),
        "environment": _environment(),
    }


def run_stage2() -> tuple[dict, pd.DataFrame]:
    """Stage 2: v1's ratings and halves, with books identified by bookrec.books."""
    cleaned, books, changes = clean_split(load_split())
    results = score_models(cleaned, models.rebuild_v1_api_similarity(cleaned))
    results.pop("v1_user_cf_precision_recall")
    records = openlibrary.read_details()
    details = openlibrary.summarise(records) if records else None
    return {**results, "changes": changes, "details": details, "environment": _environment()}, books


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


def _environment() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
    }


# --- Reports ------------------------------------------------------------------------------


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _fmt(value, spec: str) -> str:
    return "" if value is None else format(value, spec)


def _rating_table(rows: list[dict]) -> str:
    return _table(["Model", "MAE", "RMSE"], [[x["model"], f"{x['mae']:.3f}", f"{x['rmse']:.3f}"] for x in rows])


def _ranking_table(rows: list[dict]) -> str:
    return _table(
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
            for x in rows
        ],
    )


def _versions(env: dict) -> str:
    return ", ".join(f"{k} {v}" for k, v in env.items())


RANKING_METHOD = """A recommendation is a hit when it is one of the user's held-out books rated
{relevant_from} or more (v1's rule); books the user rated in training are never recommended.
Rating models rank books by predicted rating; the two table models score each book by its
similarity to the user's training books. Equal scores go to title order. Higher is better,
except that a higher average popularity (training ratings per recommended book) means more of
the same few bestsellers."""


def render_stage1(r: dict) -> str:
    s, rk, notes = r["split"], r["ranking"], r["data_notes"]
    count = lambda v: _fmt(v, ",") if isinstance(v, int) else _fmt(v, ".4f")  # noqa: E731
    reproduction = _table(
        ["", "v1 notebook", "Re-run now", "Same"],
        [[x["what"], count(x["published"]), count(x["rerun"]), "yes" if x["match"] else "**no**"] for x in r["reproduction"]],
    )
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

{_rating_table(r['rating'])}

"As shipped" is v1's code unchanged. "Corrected" changes only how averages are taken: over
the ratings a user or book has, where v1 divided by every user or book in the matrix and so
counted each missing rating as a 0. The user-based model also stops counting neighbours who
didn't rate the book, which v1's item-based model already did.

## Recommending 10 books

Each of the {rk['users']:,} users with a held-out rating of {rk['relevant_from']} or more gets
{rk['k']} books. {RANKING_METHOD.format(relevant_from=rk['relevant_from'])}

{_ranking_table(rk['rows'])}

v1's own function gives its user-based CF the slightly different 0.0383 and 0.0848 above
because it orders tied scores differently, and ties matter: for
{rk['v1_user_cf_users_tied_at_k']:,} of the {rk['users']:,} users, the tenth book is tied
with at least one book ranked below it.

## Notes on the data

- {notes['duplicate_user_title_pairs']:,} times, a user rated two or more editions that share a
  title. v1 adds those ratings together in one cell, so a cell can reach
  {notes['largest_training_cell']:.0f} on a 1-10 scale.
- {notes['held_out_titles_also_in_training']:,} held-out ratings are for a title the same user
  also rated in training. No model can recommend those, since training books are excluded.
- Both are left as v1 had them here, so that v1's numbers reproduce. Stage 2
  ([phase2-books.md](phase2-books.md)) cleans them up.

Environment: {_versions(r['environment'])}.
"""


def render_stage2(r: dict, stage1: dict) -> str:
    s, rk, c = r["split"], r["ranking"], r["changes"]
    held_out_fewer = stage1["split"]["test"] - s["test"]
    users_before = stage1["ranking"]["users"]
    users_scored = (
        f"the same {rk['users']:,} users are scored"
        if users_before == rk["users"]
        else f"{users_before:,} users were scored before, {rk['users']:,} now"
    )
    before_rank = {x["model"]: x for x in stage1["ranking"]["rows"]}
    before_rate = {x["model"]: x for x in stage1["rating"]}
    arrow = lambda a, b, spec: f"{format(a, spec)} → {format(b, spec)}"  # noqa: E731
    comparison = _table(
        ["Model", "Hit rate", "NDCG@10", "Coverage", "Avg. popularity"],
        [
            [
                x["model"],
                arrow(before_rank[x["model"]]["hit_rate"], x["hit_rate"], ".1%"),
                arrow(before_rank[x["model"]]["ndcg"], x["ndcg"], ".4f"),
                arrow(before_rank[x["model"]]["coverage"], x["coverage"], ".1%"),
                arrow(before_rank[x["model"]]["mean_popularity"], x["mean_popularity"], ".1f"),
            ]
            for x in rk["rows"]
        ],
    )
    rating_comparison = _table(
        ["Model", "MAE", "RMSE"],
        [
            [
                x["model"],
                arrow(before_rate[x["model"]]["mae"], x["mae"], ".3f"),
                arrow(before_rate[x["model"]]["rmse"], x["rmse"], ".3f"),
            ]
            for x in r["rating"]
        ],
    )
    return f"""# Stage 2: which ratings are about the same book

Generated by `python -m bookrec.evaluate`; edit the code, not this file.

v1 identified a book by its exact title. Editions with different titles were different
books ("Bel Canto" and "Bel Canto: A Novel"), different books with the same title were one
("Mortal Fear" by Greg Iles and by Robin Cook, "Lightning" by Danielle Steel and by Dean
Koontz), and HTML codes stayed in titles (`Angels &amp; Demons`).

## The rules

`bookrec/books.py` gives every rating a book:

- Titles are compared without case, punctuation, a leading "The", "A" or "An", bracketed
  edition notes ("(Movie Tie-In)") and generic subtitles (": A Novel"). Numbers in brackets are
  kept, so the volumes of a series stay apart, and real subtitles are kept, so "Dune" and
  "Dune: House Atreides" stay apart.
- Within a title, authors whose names loosely match are one author: misspellings, reversed
  names, initials, garbled accents and two pen names (Richard Bachman, Phyllis White).
- A second author makes a separate book only with at least {MIN_TRAINING_RATINGS_TO_SPLIT} training ratings; smaller groups
  are usually a publisher or editor in the author field ("Golden Books" for *Jurassic Park*)
  and stay with the main book. Only training ratings decide, so the held-out ratings never
  shape the books.
- Books with the same main title and author are then one book when Open Library files an
  edition of each under the same work ([`data/isbn_works.csv`](../data/isbn_works.csv)). This
  joins {c['books_joined_by_open_library_work']:,} books' titles, among them *The Hobbit*'s three ("The Hobbit: or There and
  Back Again", "The Hobbit: The Enchanting Prelude to The Lord of the Rings"), *Harry Potter
  and the Sorcerer's Stone* with and without "(Book 1)", and *Pride and Prejudice* with "(The
  Penguin English Library, El 72)". Books that only share a main title, such as two *Dilbert*
  collections, stay apart.
- Titles have their HTML codes decoded, accents garbled by a wrong encoding repaired, quotes
  that Books.csv escaped with a backslash put back (`\\Surely You're Joking, Mr. Feynman!\\"`),
  and a bookseller's "Tag: Author of ..." dropped.
- A user's ratings of the same book are averaged, and a held-out rating of a book the user
  also rated in training is dropped: no recommender would suggest a book the user has read.

Every rule leans towards changing nothing. Known misses: translations of *The Odyssey* stay
apart, listed under their translators, and *The Diary of Ellen Rimbauer* splits between its
two credited names. [`data/books.csv`](../data/books.csv) lists every book with the v1 titles
and ISBNs it merges.

## What changed

| | |
|---|---|
| v1's books (distinct titles) | {c['v1_titles']:,} |
| Books now | {c['books']:,} |
| Books that merge two or more v1 titles | {c['books_merging_several_v1_titles']:,} |
| of which joined because Open Library files them as one work | {c['books_joined_by_open_library_work']:,} |
| v1 titles split between different authors' books | {c['v1_titles_split_by_author']:,} |
| Titles with HTML codes, decoded | {c['titles_with_html_codes']:,} |
| Repeat ratings of one book by one user, averaged | {c['repeat_ratings_averaged']:,} |
| Held-out ratings dropped as already read in training | {c['held_out_dropped_as_already_read']:,} |
| Training / held-out ratings now | {s['train']:,} / {s['test']:,} |

The users and the train/held-out halves are v1's, so the numbers below compare directly with
[stage 1](phase1-evaluation.md).

## Recommending 10 books: stage 1 → stage 2

The held-out set is {held_out_fewer:,} ratings smaller: {c['held_out_dropped_as_already_read']:,} were of a book the user had
already rated in training, which no model could recommend, and {held_out_fewer - c['held_out_dropped_as_already_read']:,} were repeats, now
averaged. Dropping ratings no model could hit leaves the hits unchanged, so hit rate and
precision compare directly ({users_scored}); recall and NDCG, which divide by the number of
held-out books, rise slightly for every model.

{comparison}

## Predicting a rating: stage 1 → stage 2

{rating_comparison}

## Stage 2 in full

{rk['users']:,} users have a held-out rating of {rk['relevant_from']} or more. {RANKING_METHOD.format(relevant_from=rk['relevant_from'])}
The dot-product and cosine tables are rebuilt with v1's method on the cleaned books.

{_ranking_table(rk['rows'])}

{_rating_table(r['rating'])}

{_render_details(r['details'])}

Environment: {_versions(r['environment'])}.
"""


def _render_details(d: dict | None) -> str:
    if d is None:
        return "## Book details from Open Library\n\nNot fetched yet: run `python -m bookrec.openlibrary`."
    share = lambda n: f"{n:,} ({n / d['books']:.1%})"  # noqa: E731
    facts = _table(
        ["", "Books"],
        [
            ["Found by an edition's ISBN", share(d["by_isbn"])],
            ["Found by a title and author search", share(d["by_title_and_author"])],
            ["Not found", share(d["unmatched"])],
            ["With a cover", share(d["with_cover"])],
            ["With a description", share(d["with_description"])],
            ["With subjects", share(d["with_subjects"])],
        ],
    )
    work_titles = _table(
        ["Book", "Author", "Open Library's title for its work"],
        [[x["title"], x["author"], x["open_library"]] for x in d["work_title_examples"]],
    )
    titles = _table(
        ["Book", "Author", "Other Open Library titles of its editions"],
        [[x["title"], x["author"], "; ".join(x["others"])] for x in d["other_title_examples"]],
    )
    authors = _table(
        ["Book", "Author (from the rated edition)", "Open Library's author"],
        [[x["title"], x["author"], ", ".join(x["open_library"])] for x in d["author_examples"]],
    )
    return f"""## Book details from Open Library

`python -m bookrec.openlibrary` fetched every book's Open Library work once: its title,
authors, first year, cover, subjects and description, saved in
[`data/book_details.jsonl`](../data/book_details.jsonl). Later stages read that file, never the
network. A book takes the work most of its ratings' editions belong to, unless another of its
works has the book's title: an edition filed under an omnibus ("The Anne of Green Gables
novels [8 novels]") or under a wrong work then gives way. A book none of whose ISBNs Open
Library knows is searched by title and author, and kept only if both match. The work of every
rated edition's ISBN is saved first, in [`data/isbn_works.csv`](../data/isbn_works.csv), for
the rule above that joins books which are one work.

{facts}

Three checks, each with its full list:

- For {d['work_title_differs']:,} books the work's title doesn't match ours. Mostly that is the
  original-language title ("Le petit prince") or an earlier title of the same book, and the
  work is right; a few are wrong works with no better one among the book's editions ("House
  Harkonnen" under "Dune"), so a description shown for these books should be checked first.

<details><summary>The {d['work_title_differs']:,} books</summary>

{work_titles}

</details>

- {d['with_other_works']:,} books have editions in more than one Open Library work. For
  {d['other_works_with_another_title']:,} of them another work has a different title: mostly an omnibus or a
  translation Open Library files apart, sometimes a sign that one title covers several books
  ("Chobits", several volumes under one title in Book-Crossing).

<details><summary>The {d['other_works_with_another_title']:,} books</summary>

{titles}

</details>

- For {d['open_library_author_differs']:,} books, Open Library credits an author who doesn't match the one on
  the rated edition: usually a pen name (J. D. Robb is Nora Roberts) or Open Library's mistake
  (it credits Robin Cook's novels to Lisa Jackson). v2 keeps the edition's author.

<details><summary>The {d['open_library_author_differs']:,} books</summary>

{authors}

</details>"""


def write_reports(stage1: dict, stage2: dict, books: pd.DataFrame, stage3: dict) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    for name, results, text in (
        ("phase1-evaluation", stage1, render_stage1(stage1)),
        ("phase2-books", stage2, render_stage2(stage2, stage1)),
        ("phase3-models", stage3, render_stage3(stage3, stage2, _versions(_environment()))),
    ):
        (REPORTS / f"{name}.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        (REPORTS / f"{name}.md").write_text(text, encoding="utf-8")
    columns = ["title", "author", "ratings", "v1_titles", "isbns", "book_key"]
    books.rename_axis("book_id")[columns].to_csv(DATA / "books.csv", encoding="utf-8", lineterminator="\n")


def main() -> None:
    stage1 = run()
    stage2, books = run_stage2()
    stage3 = run_stage3(progress=lambda s: print(s, flush=True))
    write_reports(stage1, stage2, books, stage3)
    print(f"Wrote the reports in {REPORTS} and {DATA / 'books.csv'}")
    failed = [x["what"] for x in stage1["reproduction"] if not x["match"]]
    if failed:
        raise SystemExit("v1 did not reproduce: " + "; ".join(failed))


if __name__ == "__main__":
    main()
