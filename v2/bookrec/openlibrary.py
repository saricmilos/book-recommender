"""Book details from Open Library, fetched once.

For every book: the Open Library work its rated editions belong to (by ISBN, or else by a
title and author search that must match), the work's title, authors, first year, cover,
subjects and description. Every response is cached in data/openlibrary-cache/ (not committed)
so a stopped fetch resumes; the result is data/book_details.jsonl, which is committed. Later
stages read that file and never the network.

    python -m bookrec.openlibrary        # about 45 minutes the first time, at 1 request/second
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from bookrec.books import author_key, book_rows, clean_split, same_author, title_key
from bookrec.data import load_split
from bookrec.paths import DATA

# Identifies this project to Open Library, as its API guidelines ask.
USER_AGENT = "book-recommender-v2 (+https://github.com/saricmilos/book-recommender)"
REQUESTS_PER_SECOND = 1.0
ISBNS_PER_SEARCH = 50
MAX_SUBJECTS = 50
SEARCH = "https://openlibrary.org/search.json"
SEARCH_FIELDS = "key,title,author_name,first_publish_year,cover_i,subject,isbn"
CACHE = DATA / "openlibrary-cache"
DETAILS = DATA / "book_details.jsonl"

Fetch = Callable[[str], tuple[int, object]]  # url -> (HTTP status, parsed JSON or None)


class Client:
    """GETs JSON, at most `rate` requests a second, retrying rate limits and server errors, and
    caching every answer (404s included) on disk. `fetch` replaces the network in tests."""

    def __init__(self, cache: Path | None = CACHE, rate: float = REQUESTS_PER_SECOND, fetch: Fetch | None = None):
        self.cache = cache
        self.interval = 1.0 / rate
        self.fetch = fetch or self._http
        self.requests = 0
        self._last = 0.0

    def get(self, url: str) -> object | None:
        path = self.cache / f"{hashlib.sha1(url.encode()).hexdigest()[:24]}.json" if self.cache else None
        if path and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["body"]
        wait = self.interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        status, body = self.fetch(url)
        self._last = time.monotonic()
        self.requests += 1
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"url": url, "status": status, "body": body}), encoding="utf-8")
        return body

    @staticmethod
    def _http(url: str, attempts: int = 6) -> tuple[int, object]:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    return response.status, json.load(response)
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return 404, None
                if e.code != 429 and e.code < 500:
                    raise
                delay = float(e.headers.get("Retry-After") or 2 ** (attempt + 1))
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                delay = 2 ** (attempt + 1)
            time.sleep(min(delay, 120))
        raise RuntimeError(f"Open Library did not answer after {attempts} attempts: {url}")


def clean_isbn(isbn: str) -> str | None:
    """Book-Crossing's ISBNs as Open Library stores them: digits and a final X, 10 or 13 long."""
    s = re.sub(r"[^0-9Xx]", "", str(isbn)).upper()
    return s if len(s) in (10, 13) and "X" not in s[:-1] else None


def _batches(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def works_by_isbn(client: Client, isbns: Iterable[str]) -> dict[str, dict]:
    """The search result (a work) for each ISBN Open Library knows, ISBNS_PER_SEARCH at a time.
    An ISBN can sit in more than one work (Open Library has duplicate works); the first result,
    Open Library's best match, wins."""
    wanted = sorted({i for i in map(clean_isbn, isbns) if i})
    found: dict[str, dict] = {}
    for batch in _batches(wanted, ISBNS_PER_SEARCH):
        page, seen, total = 1, 0, None
        while total is None or seen < total:
            query = {"q": f"isbn:({' OR '.join(batch)})", "fields": SEARCH_FIELDS, "limit": 2 * len(batch)}
            if page > 1:
                query["page"] = page
            body = client.get(f"{SEARCH}?{urllib.parse.urlencode(query)}") or {}
            docs = body.get("docs", [])
            total = body.get("numFound", 0) if docs else seen  # an empty page ends the batch
            seen += len(docs)
            page += 1
            for doc in docs:
                for isbn in set(doc.get("isbn", [])) & set(batch):
                    found.setdefault(isbn, doc)
    return found


def _title_words(title: str) -> list[str]:
    """title_key without bracketed volume numbers or "and", for comparing titles across sources
    ("Cat & Mouse" and "Cat and Mouse", "Blood Lure (Anna Pigeon Mysteries, 9)" and "Blood Lure")."""
    key = re.sub(r" #[0-9-]+$", "", title_key(title or ""))
    words = [w for w in key.split() if w != "and"]
    if len(words) > 1 and words[-1] in ("the", "a", "an"):  # "Purpose-Driven Life, The"
        words = words[:-1]
    return words


def titles_match(a: str, b: str) -> bool:
    """Loosely the same title: equal words, one title equal to the other's part before a colon
    ("Longitude: The True Story of..." and "Longitude"), the words of a title of 3+ words all in
    the other, or nearly the same spelling."""
    x, y = _title_words(a), _title_words(b)
    if not x or not y:
        return False
    if x == y or _title_words(a.split(":")[0]) == y or x == _title_words(b.split(":")[0]):
        return True
    short, long = sorted((set(x), set(y)), key=len)
    if len(short) >= 3 and short <= long:
        return True
    return SequenceMatcher(None, " ".join(x), " ".join(y)).ratio() >= 0.85


def choose_work(
    isbn_ratings: list[tuple[str, int]], by_isbn: dict[str, dict], title: str | None = None, author: str | None = None
) -> tuple[dict | None, list[dict]]:
    """The work a book's ratings point to through their editions' ISBNs, and the other works
    they point to, most-rated first.

    Works are ranked by the ratings behind them; ties go to the work of the most-rated edition.
    Given the book's title and author, the best-ranked work with a matching title and author
    (or no author) wins, so that an edition filed under an omnibus ("The Anne of Green Gables
    novels [8 novels]") or a wrong work ("The Call of the Wild" under "Wuthering Heights") gives
    way to the right one. The author must match too: a small edition by another author can be
    folded into a book (stage 2), and its same-titled work is another book. With no match the
    best-ranked work stays, since it is usually the original-language title ("Le petit prince")."""
    votes: dict[str, int] = {}
    docs: dict[str, dict] = {}
    for isbn, ratings in sorted(isbn_ratings, key=lambda x: -x[1]):
        doc = by_isbn.get(clean_isbn(isbn) or "")
        if doc:
            votes[doc["key"]] = votes.get(doc["key"], 0) + ratings
            docs.setdefault(doc["key"], doc)
    if not votes:
        return None, []
    ranked = sorted(votes, key=lambda k: -votes[k])  # stable: equals keep the most-rated edition's order
    if title is not None:

        def fits(doc: dict) -> bool:
            names = doc.get("author_name") or []
            by_author = not names or author is None or any(same_author(n, author) for n in names)
            return by_author and titles_match(title, doc.get("title", ""))

        ranked.sort(key=lambda k: not fits(docs[k]))  # stable again: among fits, most-rated first
    return docs[ranked[0]], [docs[k] for k in ranked[1:]]


def search_by_title(client: Client, title: str, author: str) -> dict | None:
    """For a book none of whose ISBNs Open Library knows: the first search result with the same
    title key and a loosely matching author, else nothing."""
    query = {"title": title, "author": author_key(author), "fields": SEARCH_FIELDS, "limit": 10}
    body = client.get(f"{SEARCH}?{urllib.parse.urlencode(query)}") or {}
    for doc in body.get("docs", []):
        if title_key(doc.get("title", "")) == title_key(title) and any(
            same_author(name, author) for name in doc.get("author_name", [])
        ):
            return doc
    return None


def clean_description(text: object) -> str | None:
    """A work's description as plain text: Open Library stores either a string or {"value": ...},
    often ending in source links or a "----------" list of other editions; those go."""
    if isinstance(text, dict):
        text = text.get("value")
    if not isinstance(text, str):
        return None
    text = re.split(r"\n\s*-{5,}", text)[0]  # an edition list after a rule
    text = re.sub(r"^\s*\[\d+\]:\s*\S+\s*$", "", text, flags=re.M)  # markdown link targets
    text = re.sub(r"\(\[[^\]]*\]\[\d+\]\)", "", text)  # "([source][1])"
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if re.fullmatch(r"(see( also)?( work)?:?\s*)?https?://\S+", text, flags=re.I):  # a pointer, not a description
        return None
    return text or None


def details(client: Client, books: pd.DataFrame, isbn_ratings: dict[int, list[tuple[str, int]]]) -> list[dict]:
    """One record per book (books: book_id index, title, author)."""
    by_isbn = works_by_isbn(client, (i for pairs in isbn_ratings.values() for i, _ in pairs))
    records = []
    for book_id, book in books.iterrows():
        doc, others = choose_work(isbn_ratings.get(book_id, []), by_isbn, book["title"], book["author"])
        matched_by = "isbn" if doc else None
        if doc is None:
            doc = search_by_title(client, book["title"], book["author"])
            matched_by = "title and author" if doc else None
        record = {
            "book_id": int(book_id),
            "title": book["title"],
            "author": book["author"],
            "matched_by": matched_by,
            "other_works": [
                {"ol_work": o["key"], "ol_title": o.get("title"), "ol_authors": o.get("author_name", [])}
                for o in others
            ],
            "ol_work": None,
            "ol_title": None,
            "ol_authors": [],
            "first_publish_year": None,
            "cover_id": None,
            "subjects": [],
            "description": None,
        }
        if doc:
            work = client.get(f"https://openlibrary.org{doc['key']}.json") or {}
            covers = [c for c in work.get("covers", []) if isinstance(c, int) and c > 0]
            record.update(
                ol_work=doc["key"],
                ol_title=doc.get("title"),
                ol_authors=doc.get("author_name", []),
                first_publish_year=doc.get("first_publish_year"),
                cover_id=doc.get("cover_i") or (covers[0] if covers else None),
                subjects=(doc.get("subject") or work.get("subjects") or [])[:MAX_SUBJECTS],
                description=clean_description(work.get("description")),
            )
        records.append(record)
        if len(records) % 100 == 0:
            print(f"{len(records):,} of {len(books):,} books, {client.requests:,} requests so far", flush=True)
    return records


def write_details(records: list[dict], path: Path = DETAILS) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


def read_details(path: Path = DETAILS) -> list[dict] | None:
    if not path.exists():
        return None
    # Split on "\n" only: splitlines() would also break at separators such as U+2028, which
    # descriptions contain and JSON keeps inside strings.
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


def summarise(records: list[dict], examples: int | None = None) -> dict:
    """What the fetch found, and three checks: books whose work has another title than ours
    (mostly the original-language title, sometimes a wrong work: their descriptions need care),
    books with editions in a work of another title (a possible wrong merge in stage 2, or a
    translation Open Library files apart), and books Open Library credits to an author who
    doesn't match ours (often a pen name or Open Library's mistake; ours comes from the edition)."""
    matched = [r for r in records if r["ol_work"]]
    work_title_differs = [r for r in matched if not titles_match(r["title"], r["ol_title"] or "")]
    other_titles = [
        r for r in matched
        if any(
            not titles_match(o["ol_title"] or "", r["ol_title"] or "") and not titles_match(o["ol_title"] or "", r["title"])
            for o in r["other_works"]
        )
    ]
    other_authors = [
        r for r in matched if r["ol_authors"] and not any(same_author(a, r["author"]) for a in r["ol_authors"])
    ]
    count = lambda pred: sum(1 for r in records if pred(r))  # noqa: E731
    return {
        "books": len(records),
        "by_isbn": count(lambda r: r["matched_by"] == "isbn"),
        "by_title_and_author": count(lambda r: r["matched_by"] == "title and author"),
        "unmatched": count(lambda r: r["matched_by"] is None),
        "with_cover": count(lambda r: r["cover_id"]),
        "with_description": count(lambda r: r["description"]),
        "with_subjects": count(lambda r: r["subjects"]),
        "work_title_differs": len(work_title_differs),
        "work_title_examples": [
            {"title": r["title"], "author": r["author"], "open_library": r["ol_title"]} for r in work_title_differs[:examples]
        ],
        "with_other_works": count(lambda r: r["other_works"]),
        "other_works_with_another_title": len(other_titles),
        "other_title_examples": [
            {
                "title": r["title"],
                "author": r["author"],
                "others": sorted(
                    {o["ol_title"] for o in r["other_works"]
                     if not titles_match(o["ol_title"] or "", r["ol_title"] or "")
                     and not titles_match(o["ol_title"] or "", r["title"])}
                ),
            }
            for r in other_titles[:examples]
        ],
        "open_library_author_differs": len(other_authors),
        "author_examples": [
            {"title": r["title"], "author": r["author"], "open_library": r["ol_authors"]} for r in other_authors[:examples]
        ],
    }


def isbn_ratings_by_book() -> tuple[pd.DataFrame, dict[int, list[tuple[str, int]]]]:
    """Stage 2's books, and for each book its editions' ISBNs with how many ratings each has."""
    split = load_split()
    _, books, _ = clean_split(split)
    book_id = {k: i for i, k in enumerate(books["book_key"])}
    counts = book_rows(split).groupby(["book_key", "isbn"]).size()
    pairs: dict[int, list[tuple[str, int]]] = {}
    for (key, isbn), n in counts.items():
        pairs.setdefault(book_id[key], []).append((isbn, int(n)))
    return books[["title", "author"]], pairs


def main() -> None:
    books, pairs = isbn_ratings_by_book()
    client = Client()
    records = details(client, books, pairs)
    write_details(records)
    print(f"Wrote {DETAILS} ({len(records):,} books, {client.requests:,} requests)")


if __name__ == "__main__":
    sys.exit(main())
