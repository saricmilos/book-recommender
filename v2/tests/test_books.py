import pandas as pd
import pytest

from bookrec.books import assign_books, author_key, clean_split, clean_title, same_author, title_key

from conftest import toy_split


@pytest.mark.parametrize(
    "a, b",
    [
        ("Bel Canto", "Bel Canto: A Novel"),
        ("BODY FARM", "The Body Farm"),
        ("Angela's Ashes: A Memoir", "Angelas Ashes"),
        ("A Cold Heart (Alex Delaware Novels (Paperback))", "A Cold Heart: An Alex Delaware Novel"),
        ("The Mark (Left Behind #8)", "The Mark (Left Behind No. 8)"),
        ("Bridget Jones : The Edge of Reason", "Bridget Jones: The Edge of Reason"),
    ],
)
def test_editions_share_a_title_key(a, b):
    assert title_key(a) == title_key(b)


@pytest.mark.parametrize(
    "a, b",
    [
        ("Dune", "Dune: House Atreides"),  # a real subtitle is kept
        ("Harry Potter and the Chamber of Secrets (Book 2)", "Harry Potter and the Chamber of Secrets Postcard Book"),
        ("The Fellowship of the Ring (Part 1)", "The Fellowship of the Ring (Part 2)"),  # volumes stay apart
    ],
)
def test_different_books_keep_different_title_keys(a, b):
    assert title_key(a) != title_key(b)


def test_clean_title_decodes_html_and_repairs_garbled_accents():
    assert clean_title("Angels &amp; Demons") == "Angels & Demons"
    assert clean_title("Le Petit Prince par Saint-ExupÃ©ry") == "Le Petit Prince par Saint-Exupéry"
    assert clean_title("Saint-Exupéry") == "Saint-Exupéry"  # already right, left alone


@pytest.mark.parametrize(
    "author, surname",
    [
        ("Jeffery Deaver", "deaver"),
        ("J. R. R. Tolkien", "tolkien"),
        ("Tolkien, J.R.R.", "tolkien"),
        ("Frank B., Jr. Gilbreth", "gilbreth"),
        ("Antoine de Saint-ExupÃ©ry", "exupery"),
        ("Richard Bachman", "king"),
    ],
)
def test_author_key_is_the_surname(author, surname):
    assert author_key(author) == surname


@pytest.mark.parametrize(
    "a, b",
    [
        ("Jeff Deaver", "Jeffery Deaver"),
        ("SCOTT O'DELL", "O'Dell Scott"),
        ("John Sandford", "John Sanford"),
        ("Antoine De Saint-Exupery", "Antoine de Saint-ExupÃ©ry"),
        ("Stephen King", "Richard Bachman"),
    ],
)
def test_same_author_despite_noise(a, b):
    assert same_author(a, b)


def test_different_authors():
    assert not same_author("Danielle Steel", "Dean R. Koontz")
    assert not same_author("Robin Cook", "Greg Iles")


def _rows(*entries):
    """(title, author, part) entries, one rating each."""
    return pd.DataFrame(entries, columns=["book_title", "book_author", "part"])


def test_a_title_splits_when_the_second_author_has_enough_training_ratings():
    rows = _rows(*[("Mortal Fear", "Greg Iles", "train")] * 5, *[("Mortal Fear", "Robin Cook", "train")] * 3)
    assert set(assign_books(rows)) == {"mortal fear | iles", "mortal fear | cook"}


def test_a_small_second_author_stays_with_the_main_book():
    rows = _rows(*[("Jurassic Park", "Michael Crichton", "train")] * 5, ("Jurassic Park", "Golden Books", "train"))
    assert set(assign_books(rows)) == {"jurassic park | crichton"}


def test_held_out_ratings_never_decide_a_split():
    rows = _rows(*[("Mortal Fear", "Greg Iles", "train")] * 5, *[("Mortal Fear", "Robin Cook", "test")] * 9)
    assert set(assign_books(rows)) == {"mortal fear | iles"}


def test_clean_split_merges_editions_averages_repeats_and_drops_books_already_read():
    split = toy_split([], n_users=0, n_items=0)
    train = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3],
            "isbn": ["a", "b", "a", "c", "c"],
            "book_title": ["Bel Canto", "Bel Canto: A Novel", "Bel Canto", "Emma", "Emma"],
            "book_author": ["Ann Patchett", "Ann Patchett", "Ann Patchett", "Jane Austen", "Jane Austen"],
            "book_rating": [8, 10, 6, 7, 9],
        }
    )
    test = pd.DataFrame(
        {
            "user_id": [1, 3],
            "isbn": ["a", "a"],
            "book_title": ["Bel Canto", "Bel Canto: A Novel"],
            "book_author": ["Ann Patchett", "Ann Patchett"],
            "book_rating": [9, 5],
        }
    )
    split = split.__class__(
        filtered=train, train=train, test=test, titles=split.titles, user_ids=split.user_ids, cleaned_rows=7
    )

    cleaned, books, changes = clean_split(split)

    assert cleaned.n_items == 2  # Bel Canto (two titles, one book) and Emma
    user1 = cleaned.train[cleaned.train["user_id"] == 1]
    assert user1["book_rating"].tolist() == [9.0]  # 8 and 10 for two editions, averaged
    assert cleaned.test["user_id"].tolist() == [3]  # user 1 already rated Bel Canto in training
    assert changes["books_merging_several_v1_titles"] == 1
    assert changes["held_out_dropped_as_already_read"] == 1
    assert changes["repeat_ratings_averaged"] == 1
