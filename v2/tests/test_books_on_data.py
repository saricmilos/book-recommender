"""The book rules on the real data. If a rule changes, these counts change, and the stage 2
report and data/books.csv must be regenerated and reviewed."""

import pytest

from bookrec import openlibrary
from bookrec.books import clean_split
from bookrec.data import load_split


@pytest.fixture(scope="module")
def cleaned():
    return clean_split(load_split())


def test_counts_are_the_reviewed_ones(cleaned):
    split, books, changes = cleaned
    assert changes == {
        "v1_titles": 2_508,
        "books": 2_405,
        "titles_with_html_codes": 27,
        "books_merging_several_v1_titles": 173,
        "v1_titles_split_by_author": 75,
        "repeat_ratings_averaged": 478,
        "held_out_dropped_as_already_read": 253,
    }
    assert (len(split.train), len(split.test)) == (45_145, 11_091)


def test_every_book_has_training_ratings_and_no_user_holds_out_a_book_they_trained_on(cleaned):
    split, _, _ = cleaned
    assert set(split.test["item_idx"]) <= set(split.train["item_idx"])
    overlap = split.test.merge(split.train, on=["user_idx", "item_idx"])
    assert overlap.empty


def test_book_details_belong_to_these_books(cleaned):
    """data/book_details.jsonl was fetched for these books; if the rules change the books,
    re-run `python -m bookrec.openlibrary` (it answers from its cache where it can)."""
    _, books, _ = cleaned
    records = openlibrary.read_details()
    assert records is not None, "data/book_details.jsonl is missing"
    assert [(r["book_id"], r["title"]) for r in records] == list(enumerate(books["title"]))


def test_known_splits_and_merges(cleaned):
    _, books, _ = cleaned
    by_title = books.groupby("title")["author"].apply(set)
    assert {"Greg Iles", "Robin Cook"} <= by_title["Mortal Fear"]
    assert len(by_title["Thinner"]) == 1  # Stephen King and Richard Bachman are one author
    bel_canto = books[books["title"].str.startswith("Bel Canto")]
    assert len(bel_canto) == 1 and "Bel Canto: A Novel" in bel_canto["v1_titles"].iloc[0]
