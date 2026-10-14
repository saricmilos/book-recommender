"""The Open Library fetch, with a fake network: requests are answered from a dict of URLs."""

import urllib.parse

import pandas as pd
import pytest

from bookrec import openlibrary as ol


def fake_client(answers: dict, tmp_path=None) -> ol.Client:
    """A Client whose network is `answers`: (path or search query) -> JSON body."""
    seen = []

    def fetch(url):
        seen.append(url)
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query)
        key = parsed.path if parsed.path != "/search.json" else (query.get("q") or query["title"])[0]
        return (200, answers[key]) if key in answers else (404, None)

    client = ol.Client(cache=tmp_path, rate=1e9, fetch=fetch)
    client.seen = seen
    return client


def doc(key, title, authors, isbns, **more):
    return {"key": key, "title": title, "author_name": authors, "isbn": isbns, **more}


@pytest.mark.parametrize(
    "raw, clean",
    [("0451524934", "0451524934"), ("080410526x", "080410526X"), ("0-451-52493-4", "0451524934"),
     ("342310538", None), ("X123456789", None), ("9780451524935", "9780451524935")],
)
def test_clean_isbn(raw, clean):
    assert ol.clean_isbn(raw) == clean


def test_works_by_isbn_maps_each_isbn_to_its_work():
    orwell = doc("/works/OL1W", "Nineteen Eighty-Four", ["George Orwell"], ["0451524934", "9780451524935"])
    client = fake_client({"isbn:(0316666009 OR 0451524934)": {"docs": [orwell]}})
    assert ol.works_by_isbn(client, ["0451524934", "0316666009"]) == {"0451524934": orwell}


def test_works_by_isbn_keeps_the_first_work_and_reads_every_page(monkeypatch):
    monkeypatch.setattr(ol, "ISBNS_PER_SEARCH", 1)  # limit = 2 results a page
    first = doc("/works/FIRST", "Dune", ["Frank Herbert"], ["0441172717"])
    duplicate = doc("/works/DUP", "Dune", ["Frank Herbert"], ["0441172717"])
    later = doc("/works/LATER", "Dune", ["Frank Herbert"], ["0441172717"])
    pages = {1: [first, duplicate], 2: [later]}

    def fetch(url):
        page = int(urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get("page", ["1"])[0])
        return 200, {"numFound": 3, "docs": pages.get(page, [])}

    client = ol.Client(cache=None, rate=1e9, fetch=fetch)
    assert ol.works_by_isbn(client, ["0441172717"])["0441172717"]["key"] == "/works/FIRST"
    assert client.requests == 2


def test_choose_work_follows_the_ratings_not_the_number_of_editions():
    a, b = doc("/works/A", "A", [], []), doc("/works/B", "B", [], [])
    by_isbn = {"1111111111": a, "2222222222": b, "3333333333": b}
    work, others = ol.choose_work([("1111111111", 10), ("2222222222", 3), ("3333333333", 2)], by_isbn)
    assert (work["key"], [o["key"] for o in others]) == ("/works/A", ["/works/B"])


@pytest.mark.parametrize(
    "a, b, same",
    [
        ("Cat and Mouse", "Cat & Mouse", True),
        ("Blood Lure", "Blood Lure (Anna Pigeon Mysteries, 9)", True),
        ("Lies and the Lying Liars Who Tell Them: A Fair and Balanced Look", "Lies and the Lying Liars Who Tell Them", True),
        ("Longitude: The True Story of a Lone Genius", "Longitude", True),
        ("The Purpose-Driven Life", "Purpose-Driven Life, The", True),
        ("The Hobbit, or There and Back Again", "Hobbit", False),  # one word, with no colon, is too little to go on
        ("Critical Judgment", "Critical judgement", True),
        ("The Darwin Awards II: Unnatural Selection", "The Darwin Awards", False),
        ("The Darwin Awards II: Unnatural Selection", "The Darwin Awards II", True),
        ("House Harkonnen (Dune: House Trilogy, Book 2)", "Dune", False),
        ("The Call of the Wild", "Wuthering Heights", False),
    ],
)
def test_titles_match(a, b, same):
    assert ol.titles_match(a, b) is same


def test_choose_work_prefers_a_work_with_the_books_title_over_more_ratings():
    omnibus = doc("/works/OMNI", "The Anne of Green Gables novels [8 novels]", [], [])
    novel = doc("/works/NOVEL", "Anne of the Island", [], [])
    by_isbn = {"1111111111": omnibus, "2222222222": novel}
    pairs = [("1111111111", 9), ("2222222222", 2)]
    assert ol.choose_work(pairs, by_isbn)[0]["key"] == "/works/OMNI"  # without a title: most ratings
    work, others = ol.choose_work(pairs, by_isbn, title="Anne of the Island")
    assert (work["key"], [o["key"] for o in others]) == ("/works/NOVEL", ["/works/OMNI"])


def test_choose_work_ignores_a_same_titled_work_by_another_author():
    # Danielle Steel's Bittersweet: her editions point to the Spanish-titled work; a one-rating
    # edition by Leslie Li, folded into the book in stage 2, points to Li's own "Bittersweet".
    steel = doc("/works/STEEL", "Dulce Y Amargo", ["Danielle Steel"], [])
    li = doc("/works/LI", "Bittersweet", ["Leslie Li"], [])
    work, _ = ol.choose_work([("1111111111", 13), ("2222222222", 1)], {"1111111111": steel, "2222222222": li},
                             title="Bittersweet", author="DANIELLE STEEL")
    assert work["key"] == "/works/STEEL"


def test_choose_work_keeps_the_most_rated_work_when_no_title_matches():
    original = doc("/works/FR", "Le petit prince", [], [])
    work, _ = ol.choose_work([("1111111111", 3)], {"1111111111": original}, title="The Little Prince")
    assert work["key"] == "/works/FR"


def test_choose_work_breaks_ties_by_the_most_rated_edition():
    a, b = doc("/works/A", "A", [], []), doc("/works/B", "B", [], [])
    work, _ = ol.choose_work([("1111111111", 2), ("2222222222", 5), ("3333333333", 3)],
                             {"1111111111": a, "2222222222": b, "3333333333": a})
    assert work["key"] == "/works/B"  # 5 against 2 + 3: a tie, and B holds the most-rated edition


def test_search_by_title_needs_the_same_title_and_author():
    wrong_author = doc("/works/W", "Mortal Fear", ["Robin Cook"], [])
    right = doc("/works/R", "Mortal Fear: A Novel", ["Greg Iles"], [])
    client = fake_client({"Mortal Fear": {"docs": [wrong_author, right]}})
    assert ol.search_by_title(client, "Mortal Fear", "Greg Iles") == right
    client = fake_client({"Mortal Fear": {"docs": [wrong_author]}})
    assert ol.search_by_title(client, "Mortal Fear", "Greg Iles") is None


def test_clean_description_keeps_the_text_and_drops_link_targets_and_edition_lists():
    raw = {"value": "A dystopian novel. ([source][1])\n\n[1]: https://example.org\n----------\nAlso in: Omnibus"}
    assert ol.clean_description(raw) == "A dystopian novel."
    assert ol.clean_description("Plain.") == "Plain."
    assert ol.clean_description(None) is None
    assert ol.clean_description("See work: https://openlibrary.org/works/OL64225W") is None


def test_details_survive_a_line_separator_inside_a_description(tmp_path):
    records = [{"book_id": 0, "description": "First part. Second part."}, {"book_id": 1, "description": None}]
    ol.write_details(records, tmp_path / "details.jsonl")
    assert ol.read_details(tmp_path / "details.jsonl") == records


def _record(title, author, ol_title, ol_authors, others=(), matched_by="isbn", **more):
    return {
        "title": title, "author": author, "matched_by": matched_by, "ol_work": "/works/X" if matched_by else None,
        "ol_title": ol_title, "ol_authors": list(ol_authors), "cover_id": 1, "description": "d", "subjects": ["s"],
        "other_works": [{"ol_work": "/works/O", "ol_title": t, "ol_authors": []} for t in others], **more,
    }


def test_summarise_flags_other_titles_and_other_authors():
    records = [
        _record("1984", "George Orwell", "Nineteen Eighty-Four", ["George Orwell"], others=["Nineteen eighty-four"]),
        _record("Bel Canto", "Ann Patchett", "Bel Canto", ["Ann Patchett"], others=["The Magician's Assistant"]),
        _record("Mortal Fear", "Robin Cook", "Mortal Fear", ["Lisa Jackson"]),
        _record("Unknown", "Nobody", None, [], matched_by=None),
    ]
    s = ol.summarise(records)
    assert (s["books"], s["by_isbn"], s["unmatched"]) == (4, 3, 1)
    assert s["work_title_differs"] == 1  # 1984 is Nineteen Eighty-Four at Open Library
    assert s["with_other_works"] == 2
    assert s["other_works_with_another_title"] == 1  # a duplicate work with the same title is no warning
    assert s["other_title_examples"][0]["title"] == "Bel Canto"
    assert s["open_library_author_differs"] == 1
    assert s["author_examples"] == [{"title": "Mortal Fear", "author": "Robin Cook", "open_library": ["Lisa Jackson"]}]


def test_details_end_to_end(tmp_path):
    orwell = doc("/works/OL1W", "Nineteen Eighty-Four", ["George Orwell"], ["0451524934"],
                 first_publish_year=1949, cover_i=9267242, subject=["Dystopias", "Totalitarianism"])
    answers = {
        "isbn:(0451524934 OR 9999999999)": {"docs": [orwell]},
        "/works/OL1W.json": {"description": "Big Brother is watching.", "covers": [1]},
        "Unknown Book": {"docs": []},
    }
    books = pd.DataFrame({"title": ["1984", "Unknown Book"], "author": ["George Orwell", "Nobody"]})
    client = fake_client(answers, tmp_path)

    records = ol.details(client, books, {0: [("0451524934", 59)], 1: [("9999999999", 3)]})

    assert records[0] | {} == {
        "book_id": 0, "title": "1984", "author": "George Orwell", "matched_by": "isbn", "other_works": [],
        "ol_work": "/works/OL1W", "ol_title": "Nineteen Eighty-Four", "ol_authors": ["George Orwell"],
        "first_publish_year": 1949, "cover_id": 9267242, "subjects": ["Dystopias", "Totalitarianism"],
        "description": "Big Brother is watching.",
    }
    assert records[1]["matched_by"] is None and records[1]["ol_work"] is None

    # A second run answers from the cache and makes no requests.
    again = fake_client({}, tmp_path)
    assert ol.details(again, books, {0: [("0451524934", 59)], 1: [("9999999999", 3)]}) == records
    assert again.seen == []


def test_fetch_covers_keeps_each_answer_and_asks_again_only_for_failures(tmp_path):
    asked = []

    def fetch(cover_id):
        asked.append(cover_id)
        if cover_id == 2:
            return None  # Open Library has no such cover
        if cover_id == 3:
            raise RuntimeError("502")
        return b"jpeg"

    counts = ol.fetch_covers([1, 2, 3, 1], directory=tmp_path, rate=1000, fetch=fetch)
    assert counts == {"fetched": 1, "missing": 1, "failed": 1, "cached": 0}
    assert (tmp_path / "1.jpg").read_bytes() == b"jpeg" and (tmp_path / "2.none").exists()
    assert sorted(asked) == [1, 2, 3]  # 1 once, though listed twice

    asked.clear()
    counts = ol.fetch_covers([1, 2, 3], directory=tmp_path, rate=1000, fetch=fetch)
    assert asked == [3] and counts["cached"] == 2


def test_fetch_covers_starts_requests_at_the_rate(tmp_path):
    import time

    started = []
    ol.fetch_covers(range(4), directory=tmp_path, rate=20, fetch=lambda i: started.append(time.monotonic()) or b"x")
    gaps = [b - a for a, b in zip(sorted(started), sorted(started)[1:])]
    assert min(gaps) >= 0.04  # 20 a second is 0.05 apart, give or take the clock
