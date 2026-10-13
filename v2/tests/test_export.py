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
    text = "One. " + " ".join(f"The sentence {n} of this text is here." for n in range(20))
    short = export.short_description(text, limit=100)
    assert len(short) <= 100 and short.endswith(".")
    assert export.short_description("A short one for the shelf.") == "A short one for the shelf."
    assert export.short_description(None) is None


def test_is_english_compares_common_words():
    assert export.is_english("The story of a boy and his dog, who live by the sea.")
    assert not export.is_english("Le Petit Prince est une œuvre de langue française, la plus connue de Saint-Exupéry.")
    assert not export.is_english("Die erfolgreiche Fotografin hat ihr Leben im Griff, und sie ist auf der Insel.")


def test_plain_text_drops_markdown_and_lists_of_other_books():
    raw = "**Book Two** in the [Dune Chronicles][1].\r\n\r\nPaul *continues*.\r\n----------\r\nAlso contained in: [Dune](/works/OL1W)"
    assert export.plain_text(raw) == "Book Two in the Dune Chronicles.\n\nPaul continues."


def test_short_description_keeps_english_paragraphs_and_drops_what_the_shops_left():
    rose = "It is the year 1327. Monks are dying.\n\nUltima settimana del novembre 1327. Il novizio Adso da Melk accompagna frate Guglielmo."
    assert export.short_description(rose) == "It is the year 1327. Monks are dying."
    grisham = "John Grisham's bestselling backlist repackaged with fantastic new coversRay Atlee is a professor of law.He has a father."
    assert export.short_description(grisham) == "Ray Atlee is a professor of law. He has a father."
    assert export.short_description("A man reads to her. Until he stops. A man reads to her. Until he stops.") == (
        "A man reads to her. Until he stops."
    )
    assert export.short_description("Kurzbeschreibung\n\nDie Tochter des Magiers ist eine Novelle von Nora Roberts.") is None
    assert export.short_description("Awesome") is None


def test_first_distinct_keeps_one_per_key_and_never_the_book_itself():
    keys = ["a", "b", "b", "c", "a", "d"]
    assert export.first_distinct([4, 1, 2, 3, 5], keys, own=0, n=3) == [1, 3, 5]


def test_never_recommended_is_wild_animus_alone():
    data = load_interactions()
    assert [data.split.titles[i] for i in export.never_recommended(data)] == ["Wild Animus"]


def test_same_book_keys_spell_out_abbreviations():
    records = [
        {"title": "Into Thin Air : A Personal Account of the Mt. Everest Disaster", "author": "JON KRAKAUER"},
        {"title": "Into Thin Air : A Personal Account of the Mount Everest Disaster", "author": "Jon Krakauer"},
        {"title": "Dune", "author": "Frank Herbert"},
    ]
    keys = export.same_book_keys(records)
    assert keys[0] == keys[1] != keys[2]


def test_wilson_interval():
    lo, hi = export.wilson(0.5, 100)
    assert lo == pytest.approx(0.4038, abs=1e-4) and hi == pytest.approx(0.5962, abs=1e-4)


def test_list_stats():
    similar = {"v1": [[2, 1], [2, 0], [2, 1]], "v2": [[1, 2], [0, 2], [0, 1]]}
    stats = export.list_stats(similar, ["a", "b", "c"])
    assert stats["v1"] == {"most_listed": "c", "most_listed_share": 1.0, "books_listed": 3}
    assert stats["v2"]["most_listed_share"] == pytest.approx(2 / 3)
