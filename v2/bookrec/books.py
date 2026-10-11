"""Which ratings are about the same book.

v1 used the exact title string as a book's identity. So editions with different titles were
different books ("Bel Canto" and "Bel Canto: A Novel"), different books with the same title
were one ("A Heart of Stone" by Renate Dorrestein and by Barbara Cartland), and HTML codes
stayed in titles ("Angels &amp; Demons").

Here titles that reduce to the same title_key are one book unless their authors differ. The
author field is noisy (misspellings, reversed names, publishers and editors in place of the
author), so an author counts as different only if the names don't loosely match, and a title
splits only if the second author has at least MIN_TRAINING_RATINGS_TO_SPLIT training ratings;
smaller groups stay with the main book, as in v1. Both rules lean towards changing nothing:
a missed merge or split leaves things as v1 had them, a wrong one would be a new error.
Only training ratings decide, so the held-out ratings never shape the books.
"""

from __future__ import annotations

import html
import re
import unicodedata
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from bookrec.data import Split

MIN_TRAINING_RATINGS_TO_SPLIT = 3

# A subtitle that names the kind of book rather than the book: ": A Novel", ": A Memoir",
# ": An Alex Delaware Novel". Other subtitles are kept, so "Dune" and "Dune: House Atreides"
# (both by a Herbert) stay two books.
_GENERIC_SUBTITLE = re.compile(
    r"^(a|an) (.+ )?(novel|novels|memoir|thriller|mystery|story|stories|novella|romance|tale|tales"
    r"|biography|fable)( of [a-z]+( [a-z]+)?)?$"
)
_ARTICLE = re.compile(r"^(the|a|an) ")
_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "phd", "md"}

# Pen names that split a book from itself in this data: Thinner and The Regulators appear
# under both Stephen King and Richard Bachman, Devices and Desires under P. D. James and her
# married name. Keys and values are as _author_words gives them.
PEN_NAMES = {"richard bachman": "stephen king", "phyllis white": "p d james"}


def _repair(text: str) -> str:
    """Undo UTF-8 that was read as Latin-1 ("Saint-ExupÃ©ry" -> "Saint-Exupéry"); text that
    isn't garbled that way comes back unchanged."""
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def _ascii(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def clean_title(title: str) -> str:
    """The title to show: HTML codes decoded, garbled accents repaired, spaces tidied."""
    return re.sub(r"\s+", " ", _repair(html.unescape(str(title)))).strip()


def _plain(text: str) -> str:
    text = re.sub(r"['’]", "", _ascii(text).lower())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip()


def title_key(title: str) -> str:
    """Lower case, no punctuation, no leading article, no bracketed edition notes and no
    generic subtitle. Numbers inside brackets are kept ("(Book 2)", "(Left Behind #6)"), so
    the volumes of a series never merge."""
    t = clean_title(title).lower()
    bracketed = []
    while (m := re.search(r"\([^()]*\)", t)) is not None:  # innermost first, for nested brackets
        bracketed.append(m.group(0))
        t = t[: m.start()] + " " + t[m.end() :]
    numbers = re.findall(r"\d+", " ".join(bracketed))

    head, _, subtitle = t.partition(":")
    if subtitle and _GENERIC_SUBTITLE.match(_plain(subtitle)):
        t = head
    key = _ARTICLE.sub("", _plain(t))
    return f"{key} #{'-'.join(numbers)}" if numbers else key


def _author_words(author: str) -> list[str]:
    name = re.sub(r"['’.]", "", _ascii(_repair(html.unescape(str(author)))).lower())
    words = [w for w in re.findall(r"[a-z]+", name) if w not in _NAME_SUFFIXES]
    return PEN_NAMES.get(" ".join(words), " ".join(words)).split()


def author_key(author: str) -> str:
    """The author's surname: "Jeff Deaver" and "Jeffery Deaver" give "deaver", "Tolkien, J.R.R."
    and "J. R. R. Tolkien" give "tolkien"."""
    name = _repair(html.unescape(str(author)))  # before lower(): "Ã" lower-cased can't be repaired
    before_comma = _author_words(name.split(",")[0]) if "," in name else []
    if len(before_comma) == 1:  # "Surname, First names"
        return before_comma[0]
    words = _author_words(name)
    return words[-1] if words else ""


def same_author(a: str, b: str) -> bool:
    """Loosely the same person: equal surnames, one surname among the other name's words
    ("O'Dell Scott" and "Scott O'Dell"), or surnames of 5+ letters that differ by a typo
    ("Sanford" and "Sandford")."""
    sa, sb = author_key(a), author_key(b)
    if sa == sb:
        return True
    if sa in _author_words(b) or sb in _author_words(a):
        return True
    return min(len(sa), len(sb)) >= 5 and SequenceMatcher(None, sa, sb).ratio() >= 0.8


def assign_books(rows: pd.DataFrame) -> pd.Series:
    """A book key for every row of `rows` (book_title, book_author, and part: "train" or "test").

    Rows are grouped by title_key, which merges editions; within a group, authors that loosely
    match are one author, taken in order of training ratings, most first. The first author's
    group keeps the book; another author's group becomes its own book only with at least
    MIN_TRAINING_RATINGS_TO_SPLIT training ratings, and otherwise stays with the first.
    """
    keys = pd.Series(index=rows.index, dtype=object)
    for tkey, group in rows.groupby(rows["book_title"].map(title_key), sort=True):
        training = group.loc[group["part"] == "train", "book_author"].value_counts()
        authors = sorted(group["book_author"].unique(), key=lambda a: (-training.get(a, 0), a))
        clusters: list[list[str]] = []  # each: [representative, *members]
        for author in authors:
            for cluster in clusters:
                if same_author(cluster[0], author):
                    cluster.append(author)
                    break
            else:
                clusters.append([author])
        main = clusters[0][0]
        book_of = {}
        for cluster in clusters:
            n_training = sum(training.get(a, 0) for a in cluster)
            owner = cluster[0] if cluster[0] == main or n_training >= MIN_TRAINING_RATINGS_TO_SPLIT else main
            for author in cluster:
                book_of[author] = f"{tkey} | {author_key(owner)}"
        keys[group.index] = group["book_author"].map(book_of)
    return keys


def book_rows(split: Split) -> pd.DataFrame:
    """Every rating of v1's split, train and held-out ("part"), with its ISBN and book_key."""
    cols = ["user_id", "isbn", "book_title", "book_author", "book_rating"]
    rows = pd.concat(
        [split.train[cols].assign(part="train"), split.test[cols].assign(part="test")], ignore_index=True
    )
    rows["book_key"] = assign_books(rows)
    return rows


def _most_common(values: pd.Series) -> str:
    return values.value_counts().index[0]


def clean_split(split: Split) -> tuple[Split, pd.DataFrame, dict]:
    """The same ratings, in the same train and held-out halves, with books identified by
    assign_books. A user's ratings of the same book are averaged; a held-out rating of a book
    the user also rated in training is dropped, since no recommender would suggest it again.

    Returns the new split, a table of books (what each one merged), and counts of what changed.
    """
    rows = book_rows(split)
    frames = {part: rows[rows["part"] == part] for part in ("train", "test")}

    books = (
        rows.groupby("book_key")
        .agg(
            title=("book_title", lambda s: clean_title(_most_common(s))),
            author=("book_author", lambda s: clean_title(_most_common(s))),
            v1_titles=("book_title", lambda s: " | ".join(sorted(set(s)))),
            isbns=("isbn", lambda s: " ".join(sorted(set(s)))),
            ratings=("book_rating", "size"),
        )
        .reset_index()
    )
    missing = set(frames["test"]["book_key"]) - set(frames["train"]["book_key"])
    if missing:
        raise RuntimeError(f"{len(missing)} held-out books have no training ratings")
    item_idx = {k: i for i, k in enumerate(books["book_key"])}
    user_ids = np.unique(rows["user_id"])
    user_idx = {u: i for i, u in enumerate(user_ids)}

    def collapse(f: pd.DataFrame) -> pd.DataFrame:
        out = f.groupby(["user_id", "book_key"], as_index=False)["book_rating"].mean()
        out["user_idx"] = out["user_id"].map(user_idx)
        out["item_idx"] = out["book_key"].map(item_idx)
        out["book_title"] = books["title"].to_numpy()[out["item_idx"]]
        return out

    train = collapse(frames["train"])
    test = collapse(frames["test"])
    seen = test.merge(train[["user_id", "book_key"]], on=["user_id", "book_key"], how="left", indicator=True)
    keep = (seen["_merge"] == "left_only").to_numpy()
    test = test[keep].reset_index(drop=True)

    v1_titles_per_book = books["v1_titles"].str.count(r" \| ") + 1
    books_per_v1_title = rows.groupby("book_title")["book_key"].nunique()
    changes = {
        "v1_titles": int(rows["book_title"].nunique()),
        "books": len(books),
        "titles_with_html_codes": int(sum(html.unescape(t) != t for t in rows["book_title"].unique())),
        "books_merging_several_v1_titles": int((v1_titles_per_book > 1).sum()),
        "v1_titles_split_by_author": int((books_per_v1_title > 1).sum()),
        "repeat_ratings_averaged": int(len(rows) - len(train) - len(collapse(frames["test"]))),
        "held_out_dropped_as_already_read": int((~keep).sum()),
    }

    cleaned = Split(
        filtered=pd.concat([train, test], ignore_index=True),
        train=train,
        test=test,
        titles=books["title"].to_numpy(),
        user_ids=user_ids,
        cleaned_rows=split.cleaned_rows,
    )
    return cleaned, books, changes
