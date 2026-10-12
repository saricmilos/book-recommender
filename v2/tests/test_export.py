import pytest

from bookrec import export
from bookrec.interactions import load_interactions


@pytest.mark.parametrize(
    "raw, shown",
    [
        ("The Bell Jar : A Novel (Perennial Classics)", "The Bell Jar"),
        ("Harry Potter and the Chamber of Secrets (Book 2)", "Harry Potter and the Chamber of Secrets (Book 2)"),
        ("Dune: House Atreides", "Dune: House Atreides"),
        ("Cruel &amp; Unusual (Kay Scarpetta Mysteries (Paperback))", "Cruel & Unusual"),
        ("BAG OF BONES : A NOVEL", "Bag of Bones"),
        ("Bridget Jones : The Edge of Reason", "Bridget Jones: The Edge of Reason"),
    ],
)
def test_display_title(raw, shown):
    assert export.display_title(raw) == shown


@pytest.mark.parametrize(
    "raw, shown",
    [("DANIELLE STEEL", "Danielle Steel"), ("SCOTT O'DELL", "Scott O'Dell"), ("V.C. ANDREWS", "V.C. Andrews"),
     ("ANGELA'S ASHES", "Angela's Ashes"), ("Dan Brown", "Dan Brown")],
)
def test_tidy_case(raw, shown):
    assert export.tidy_case(raw) == shown


def test_short_description_ends_on_a_sentence():
    text = "One. " + "A sentence of some length goes here. " * 20
    short = export.short_description(text, limit=100)
    assert len(short) <= 100 and short.endswith(".")
    assert export.short_description("Short.") == "Short."
    assert export.short_description(None) is None


def test_first_distinct_keeps_one_per_key_and_never_the_book_itself():
    keys = ["a", "b", "b", "c", "a", "d"]
    assert export.first_distinct([4, 1, 2, 3, 5], keys, own=0, n=3) == [1, 3, 5]


def test_never_recommended_is_wild_animus_alone():
    data = load_interactions()
    assert [data.split.titles[i] for i in export.never_recommended(data)] == ["Wild Animus"]
