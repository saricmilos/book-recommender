"""Stage 4: the numbers behind "What the data showed" on the demo's page, four facts about
Book-Crossing that each changed how v2 is built. export.py writes them into the site as
findings.json, with books named as the demo names them.

1. Most interactions are a 0: the reader had the book and gave no rating. v1 dropped them;
   from stage 3 they are training data.
2. Most editions have one reader. v1 kept the titles with at least 10 ratings from readers with
   at least 10; stage 3 gives those books every reader who has them.
3. Wild Animus has the most readers and the lowest mean rating of the most-read books: the demo
   never recommends it.
4. A title isn't a book: stage 2's editions, books that share a title, and HTML codes.

Books are given by book ID (data/books.csv, the same as item_idx).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from bookrec.interactions import Interactions, load_interactions
from bookrec.paths import DATA, DATASETS, REPORTS

# Readers per edition, as (from, to) with to=None for "or more"
READER_BINS = [(1, 1), (2, 4), (5, 9), (10, 49), (50, 99), (100, None)]
MOST_READ = 8

# The page's examples of stage 2, checked against data/books.csv on every export
MERGED_EXAMPLE = "Bel Canto"  # with "Bel Canto: A Novel", one book
SPLIT_EXAMPLE = "Lightning"  # Dean Koontz's and Danielle Steel's
HTML_EXAMPLE = "Angels &amp; Demons"


def read_ratings() -> pd.DataFrame:
    return pd.read_csv(DATASETS / "Ratings.csv", dtype={"ISBN": str}).rename(
        columns={"User-ID": "user_id", "ISBN": "isbn", "Book-Rating": "rating"}
    )


def rating_counts(ratings: pd.DataFrame) -> list[int]:
    """Interactions by rating, 0 (had the book, gave no rating) to 10."""
    return np.bincount(ratings["rating"], minlength=11).tolist()


def readers_per_edition(ratings: pd.DataFrame, catalogue: set[str]) -> list[dict]:
    """The editions in Books.csv that anyone has, counted by how many readers have them."""
    readers = ratings[ratings["isbn"].isin(catalogue)].groupby("isbn")["user_id"].nunique()
    return [
        {"from": lo, "to": hi, "editions": int(((readers >= lo) & (readers <= (hi or np.inf))).sum())}
        for lo, hi in READER_BINS
    ]


def most_read(data: Interactions, n: int = MOST_READ) -> list[dict]:
    """The n books with the most readers in the training data (the demo's count of readers),
    with how many of them rated it and their mean rating."""
    g = data.train.groupby("item_idx").agg(
        readers=("user_idx", "size"), rated=("explicit", "sum"), mean_rating=("rating", "mean")
    )
    top = g.sort_values("readers", ascending=False, kind="stable").head(n)
    return [
        {"book": int(i), "readers": int(r.readers), "rated": int(r.rated), "mean_rating": round(float(r.mean_rating), 2)}
        for i, r in top.iterrows()
    ]


def title_examples(books: pd.DataFrame) -> dict:
    """The page's examples, found in data/books.csv: a book under several v1 titles, a v1 title
    that is several authors' books, and a title with an HTML code. Raises if one no longer holds,
    so a change to stage 2's rules can't leave the page with a wrong example."""
    v1_titles = books["v1_titles"].str.split(" | ", regex=False)
    merged = books[v1_titles.map(lambda ts: MERGED_EXAMPLE in ts and len(ts) > 1)]
    split = books[v1_titles.map(lambda ts: SPLIT_EXAMPLE in ts)]
    escaped = books[v1_titles.map(lambda ts: HTML_EXAMPLE in ts)]
    if len(merged) != 1 or len(split) < 2 or len(escaped) != 1:
        raise RuntimeError("An example of stage 2 on the page no longer holds: choose another in findings.py")
    return {
        "merged": {"book": int(merged["book_id"].iloc[0]), "v1_titles": v1_titles[merged.index[0]]},
        "split": {"v1_title": SPLIT_EXAMPLE, "books": split["book_id"].astype(int).tolist()},
        "escaped": {"book": int(escaped["book_id"].iloc[0]), "v1_title": HTML_EXAMPLE},
    }


def findings(data: Interactions | None = None) -> dict:
    data = data or load_interactions()
    ratings = read_ratings()
    catalogue = set(pd.read_csv(DATASETS / "Books.csv", usecols=["ISBN"], dtype=str)["ISBN"])
    books = pd.read_csv(DATA / "books.csv")
    stage2 = json.loads((REPORTS / "phase2-books.json").read_text(encoding="utf-8"))["changes"]
    stage3 = json.loads((REPORTS / "phase3-models.json").read_text(encoding="utf-8"))
    ease = {x["data"]: x["ndcg"] for x in stage3["extra_data"] if x["model"] == "EASE"}
    v1_ratings = np.bincount(data.split.train["item_idx"], minlength=data.n_items)
    return {
        "interactions": len(ratings),
        "readers": int(ratings["user_id"].nunique()),
        "ratings": rating_counts(ratings),
        # What stage 3 found adding the 0s and other readers does, at fixed settings
        "ease_ndcg": {
            "ratings_only": ease["Benchmark ratings only"],
            "own_books": ease["+ users' own unrated books"],
            "all": ease["All interactions"],
        },
        "editions": {
            "listed": len(catalogue),
            "by_readers": readers_per_edition(ratings, catalogue),
        },
        "books": {
            "count": data.n_items,
            "editions": len(data.editions),
            "share_of_interactions": float(ratings["isbn"].isin(data.editions).mean()),
            "median_v1_ratings": float(np.median(v1_ratings)),
            "median_v2_readers": float(np.median(data.popularity())),
        },
        "most_read": most_read(data),
        "titles": {
            "v1_titles": stage2["v1_titles"],
            "books": stage2["books"],
            "merging_several": stage2["books_merging_several_v1_titles"],
            "split_by_author": stage2["v1_titles_split_by_author"],
            "html_codes": stage2["titles_with_html_codes"],
            "editions_in_v1": stage3["counts"]["editions_in_benchmark"],
            "examples": title_examples(books),
        },
    }
