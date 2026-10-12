"""Stage 4: the data behind the demo on saricmilos.com, which runs with no server.

For each of the 2,405 books, the ten books v2 recommends to a reader of that one book (the
stage 3 blend, at the settings its report chose, trained on all interactions), and the ten
v1's API returns for it (its own table, v1/models/topk_item_sim.npz), plus each book's title,
author, year, cover and a short description. Written as JSON into the site's
src/content/book-recommender/, which the page loads on demand. Also the page's table of results
and the numbers behind its "What the data showed" (findings.py).

Two display rules, not model changes: a list shows one edition of a book (same_book_keys),
and v2 never recommends Wild Animus (never_recommended says why); the evaluation is as the
reports have it.

    python -m bookrec.export        # BOOKREC_SITE names the site's folder, if not ../../sites/saricmilos.com
"""

from __future__ import annotations

import html
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from bookrec import findings, openlibrary, text, topn
from bookrec.books import author_key, book_rows, clean_split, clean_title, title_key
from bookrec.data import load_split
from bookrec.interactions import load_interactions
from bookrec.models import v1_api_similarity
from bookrec.paths import REPO, REPORTS

SITE = Path(os.environ.get("BOOKREC_SITE", REPO.parents[1] / "sites" / "saricmilos.com"))
OUT = SITE / "src" / "content" / "book-recommender"
N = 10
DESCRIPTION_CHARS = 420

# Books whose Open Library work is a different book, found by reading the stage 2 report's list
# of works with another title (the rest of that list is original-language titles, earlier
# titles of the same book, or omnibus editions): no cover or description for these.
WRONG_WORKS = {
    "House Harkonnen (Dune: House Trilogy, Book 2)",  # under "Dune"
    "Jewels",  # Danielle Steel's, under her "Loving"
    "Palomino",  # also under "Loving"
    "Prehistory of the Far Side",  # under "The Far Side Gallery 5"
    "Chobits (Chobits)",  # several volumes under one title in Book-Crossing
    "Dilbert Fugitive From The Cubicle Police",  # under a generic "Dilbert"
}


_SMALL_WORDS = {"a", "an", "the", "and", "of", "in", "on", "at", "to", "for", "or", "but", "by", "with", "from"}


def _capitalise(word: str) -> str:
    # A capital to start, after a full stop or hyphen ("V.C.", "Jean-Paul"), and after an
    # apostrophe that starts a name ("O'Dell") but not a possessive ("Angela's")
    word = re.sub(r"(^|[.\-])([a-z])", lambda m: m.group(1) + m.group(2).upper(), word)
    return re.sub(r"'([a-z])(?=[a-z]{2})", lambda m: "'" + m.group(1).upper(), word)


def tidy_case(text_: str) -> str:
    """A title or name typed in capitals ("BAG OF BONES", "DANIELLE STEEL") in ordinary case;
    anything else unchanged."""
    if not re.search(r"[A-Za-z]", text_) or text_ != text_.upper():
        return text_
    words = text_.lower().split(" ")
    return " ".join(w if (i and w in _SMALL_WORDS) else _capitalise(w) for i, w in enumerate(words))


def display_title(title: str) -> str:
    """The title to show: without bracketed edition notes ("(Perennial Classics)") unless they
    number a volume ("(Book 2)"), and without a generic subtitle (": A Novel")."""
    t = clean_title(title)
    t = re.sub(r"\s*\((?:[^()]|\([^()]*\))*\)", lambda m: m.group(0) if re.search(r"\d", m.group(0)) else "", t)
    head, sep, rest = t.partition(":")
    if sep and title_key(head) == title_key(t):  # the subtitle is generic: dropping it changes nothing
        t = head
    t = re.sub(r"\s+:", ":", t)
    return tidy_case(re.sub(r"\s+", " ", t).strip(" :"))


def short_description(text_: str | None, limit: int = DESCRIPTION_CHARS) -> str | None:
    """The description's opening sentences, up to about `limit` characters."""
    if not text_:
        return None
    t = re.sub(r"\s+", " ", html.unescape(text_)).strip()
    if len(t) <= limit:
        return t
    cut = t[:limit]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return cut[: end + 1] if end > limit // 3 else cut.rsplit(" ", 1)[0] + "…"


def never_recommended(data) -> set[int]:
    """Books many have and few like: more than 1,000 readers and a mean rating below 5. Only
    Wild Animus meets this. Its publisher gave copies away through BookCrossing by the
    thousand, so having it says who received one, not who chose it, and it would top every
    list. Models still learn from it; the demo never recommends it."""
    t = data.train
    g = t.groupby("item_idx").agg(readers=("user_idx", "size"), mean_rating=("rating", "mean"))
    return set(g[(g["readers"] > 1000) & (g["mean_rating"] < 5)].index.tolist())


# Abbreviations an edition's title may spell out ("Mt. Everest", "Mount Everest")
_SPELLED_OUT = {"mt": "mount", "st": "saint", "dr": "doctor", "mr": "mister"}


def same_book_keys(records: list[dict]) -> list[str]:
    """A key per book that two editions stage 2 kept apart share: the title key without its
    bracketed volume number and with abbreviations spelled out, and the author's surname
    ("Harry Potter and the Sorcerer's Stone" and "... (Book 1)"; "Into Thin Air: ... Mt.
    Everest Disaster" and "... Mount Everest Disaster"). Lists show one book of each key."""
    keys = []
    for r in records:
        title = re.sub(r" #[0-9-]+$", "", title_key(r["title"]))
        keys.append(" ".join(_SPELLED_OUT.get(w, w) for w in title.split(" ")) + " | " + author_key(r["author"]))
    return keys


def first_distinct(candidates, keys: list[str], own: int, n: int) -> list[int]:
    picks, seen = [], {keys[own]}
    for j in candidates:
        j = int(j)
        if keys[j] not in seen:
            picks.append(j)
            seen.add(keys[j])
            if len(picks) == n:
                break
    return picks


def blend_settings() -> dict:
    """The stage 3 report's chosen settings for EASE, RP3beta, text and the blend."""
    tuned = {x["model"]: x["params"] for x in json.loads((REPORTS / "phase3-models.json").read_text(encoding="utf-8"))["tuned"]}
    return {"ease": tuned["EASE"], "rp3beta": tuned["RP3beta"], "blend": tuned["Blend: EASE + RP3beta + text"]}


def v2_similar(keys: list[str], n: int = N) -> list[list[int]]:
    """For each book, the n books the blend recommends to a reader of that book alone, one per
    same_book_keys key, never one of never_recommended."""
    data = load_interactions()
    s = blend_settings()
    counts = topn.co_occurrence(data).astype(float)
    g = counts.copy()
    g[np.diag_indices_from(g)] += s["ease"]["lam"]
    p = np.linalg.inv(g)
    ease = -p / np.diag(p)[None, :]
    r = s["rp3beta"]
    rp3 = topn.rp3beta_walk(data, r["alpha"])
    rp3 /= np.power(np.maximum(data.popularity(), 1), r["beta"])[None, :]
    np.fill_diagonal(rp3, 0)
    rp3 = topn._keep_top_k(rp3, r["k"])
    words = topn._keep_top_k(text.text_similarity(), s["blend"]["text_k"])
    own = [np.array([i]) for i in range(data.n_items)]  # a reader of book i has book i
    w = s["blend"]
    scores = (
        w["ease"] * topn.standardise(ease, own)
        + w["rp3beta"] * topn.standardise(rp3, own)
        + w["text"] * topn.standardise(words, own)
    )
    np.fill_diagonal(scores, -np.inf)
    scores[:, sorted(never_recommended(data))] = -np.inf
    ranked = np.argsort(-scores, axis=1, kind="stable")[:, : 4 * n]
    return [first_distinct(ranked[i], keys, i, n) for i in range(data.n_items)]


def v1_similar(keys: list[str], n: int = N) -> list[list[int]]:
    """For each book, what v1's API returns for it, as v2 books: the API's table row for the
    book's most-rated v1 title, in its order, each v1 title taken to the v2 book most of its
    ratings went to, one per same_book_keys key; a v1 title that is this book again is skipped."""
    v1 = load_split()
    rows = book_rows(v1)
    _, books, _ = clean_split(v1)
    book_id = {k: i for i, k in enumerate(books["book_key"])}
    by_title = rows.groupby(["book_title", "book_key"]).size().reset_index(name="n")
    title_to_book = by_title.sort_values("n", ascending=False).drop_duplicates("book_title").set_index("book_title")["book_key"]
    book_to_title = by_title.sort_values("n", ascending=False).drop_duplicates("book_key").set_index("book_key")["book_title"]
    v1_index = {t: i for i, t in enumerate(v1.titles)}
    table = v1_api_similarity()
    out = []
    for key in books["book_key"]:
        row = table[v1_index[book_to_title[key]]].toarray().ravel().astype(float)  # the table holds whole counts
        row[v1_index[book_to_title[key]]] = -np.inf
        listed = [j for j in np.argsort(-row, kind="stable") if row[j] > 0]  # its top 50, best first
        out.append(first_distinct((book_id[title_to_book[v1.titles[j]]] for j in listed), keys, book_id[key], n))
    return out


def export(out: Path = OUT) -> dict:
    data = load_interactions()
    records = openlibrary.read_details()
    readers = data.popularity()
    books = []
    descriptions = {}
    for r in records:
        trusted = r["title"] not in WRONG_WORKS
        books.append(
            [
                display_title(r["title"]),
                tidy_case(clean_title(r["author"])),
                r["first_publish_year"] if trusted else None,
                r["cover_id"] if trusted else None,
                int(readers[r["book_id"]]),
            ]
        )
        d = short_description(r["description"]) if trusted else None
        if d:
            descriptions[str(r["book_id"])] = d

    keys = same_book_keys(records)
    similar = {"v2": v2_similar(keys), "v1": v1_similar(keys)}
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "books.json": {"fields": ["title", "author", "year", "cover", "readers"], "books": books},
        "similar.json": similar,
        "descriptions.json": descriptions,
        "results.json": {**results(), "lists": list_stats(similar, [b[0] for b in books])},
        "findings.json": named_findings(findings.findings(data), books),
    }
    for name, content in files.items():
        (out / name).write_text(json.dumps(content, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    return {name: (out / name).stat().st_size for name in files}


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """The 95% Wilson interval of a proportion p over n."""
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


def list_stats(similar: dict[str, list[list[int]]], titles: list[str]) -> dict:
    """How much each version's lists repeat themselves: the book in the most lists and the share
    of lists it is in, and how many different books the lists hold between them."""
    out = {}
    for version, lists in similar.items():
        counts = np.bincount([j for row in lists for j in row], minlength=len(titles))
        top = int(np.argmax(counts))
        out[version] = {
            "most_listed": titles[top],
            "most_listed_share": float(counts[top] / len(lists)),
            "books_listed": int((counts > 0).sum()),
        }
    return out


def results() -> dict:
    """The numbers the page shows, from the reports: every row on the same held-out ratings,
    with a 95% interval for its hit rate over the users scored."""
    stage2 = json.loads((REPORTS / "phase2-books.json").read_text(encoding="utf-8"))
    stage3 = json.loads((REPORTS / "phase3-models.json").read_text(encoding="utf-8"))
    pick = lambda rows, name: next(x for x in rows if x["model"] == name)  # noqa: E731
    users = stage3["users_scored"]
    keep = lambda x: {  # noqa: E731
        **{k: x[k] for k in ("ndcg", "hit_rate", "coverage", "mean_popularity")},
        "hit_rate_ci": wilson(x["hit_rate"], users),
    }
    rows2, rows3 = stage2["ranking"]["rows"], stage3["tuned"]
    return {
        "users": users,
        "training": stage3["counts"]["training_interactions"],
        "rows": [
            {"label": "Most popular books", **keep(pick(rows3, "Most popular"))},
            {"label": "v1 as shipped (user-based CF)", **keep(pick(rows2, "v1 user-based CF, as shipped"))},
            {"label": "v1's API table (dot product)", **keep(pick(rows2, "v1 API table: dot product, top 50"))},
            {"label": "Stage 2: cosine table, clean books", **keep(pick(rows2, "Cosine table, top 50 (what v1 measured)"))},
            {"label": "Stage 3: EASE, all interactions", **keep(pick(rows3, "EASE"))},
            {"label": "v2: blend of EASE, RP3beta and text", **keep(pick(rows3, "Blend: EASE + RP3beta + text"))},
        ],
    }


def named_findings(found: dict, books: list[list]) -> dict:
    """findings.findings() with each book's title and author as the demo shows them (books:
    books.json's rows, by book ID)."""
    title = lambda i: books[i][0]  # noqa: E731
    author = lambda i: books[i][1]  # noqa: E731
    examples = found["titles"]["examples"]
    return {
        **found,
        "most_read": [{**x, "title": title(x["book"]), "author": author(x["book"])} for x in found["most_read"]],
        "titles": {
            **found["titles"],
            "examples": {
                "merged": {**examples["merged"], "author": author(examples["merged"]["book"])},
                "split": {**examples["split"], "authors": [author(i) for i in examples["split"]["books"]]},
                "escaped": {**examples["escaped"], "title": title(examples["escaped"]["book"])},
            },
        },
    }


def main() -> None:
    sizes = export()
    print(f"Wrote {OUT}: " + ", ".join(f"{k} {v / 1024:.0f} kB" for k, v in sizes.items()))


if __name__ == "__main__":
    main()
