"""Stage 4: the data behind the demo on saricmilos.com, which runs with no server.

For each of stage 2's books, the ten books v2 recommends to a reader of that one book (the
stage 3 blend, at the settings its report chose, trained on all interactions), and the ten
v1's API returns for it (its own table, v1/models/topk_item_sim.npz), plus each book's title,
author, year, cover and a short description. Written as JSON into the site's
src/content/book-recommender/, which the page loads on demand. Also the page's table of results
and the numbers behind its "What the data showed" (findings.py), and the covers, as small
pictures in the site's public/book-recommender/covers/, so the site serves them itself instead
of archive.org, which is slow (openlibrary.py fetches them once).

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
from PIL import Image

from bookrec import findings, openlibrary, text, topn
from bookrec.books import SPELLED_OUT, author_key, book_rows, clean_split, clean_title, title_key
from bookrec.data import load_split
from bookrec.interactions import load_interactions
from bookrec.models import v1_api_similarity
from bookrec.paths import REPO, REPORTS

SITE = Path(os.environ.get("BOOKREC_SITE", REPO.parents[1] / "sites" / "saricmilos.com"))
OUT = SITE / "src" / "content" / "book-recommender"
COVERS_OUT = SITE / "public" / "book-recommender" / "covers"
COVER_WIDTH = 180  # the page shows a cover at most about 90 pixels wide: sharp at twice that
COVER_QUALITY = 72
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

# Descriptions left out, found by reading those of the 200 most-read books and of the page's
# examples: written by a reader with mistakes in them, or about another edition.
BAD_DESCRIPTIONS = {
    "A Wrinkle in Time",  # a school report: "It is about Meg And Charles Walence ... It also has a movie!"
    "The Cat in the Hat",  # the Latin translation's: "Includes a Latin-English glossary"
    "The Catcher in the Rye",  # "Story of Holden Caufield"
    "Fast Food Nation: The Dark Side of the All-American Meal",  # "the story of postwar Amerca", "has has"
    "Tara Road",  # "her husband completetely"
    "Good in Bed",  # "writes an artilce"
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


# Common short words of English, and of the other languages Open Library's descriptions come in
# (Spanish, French, German, Italian, Portuguese, Dutch, Czech), leaving out words both share
# ("in", "is", "a", "die", "do", "no", "son", "non").
_ENGLISH = set(
    "the and of to he for his with as her by she at they from this be are have which their one but not has had who"
    " were it that when into after about would what there its been more him them can".split()
)
_OTHER = set(
    "el los las del una por con para que es su sus como más pero al le les des une est du dans qui pour pas sur au"
    " par il elle avec ce sa ses der das und ist ein eine zu den von mit sich auf für nicht im dem er sie auch di"
    " che è della gli nel os da um em não na het een en van dat op te zijn voor niet hij je se že".split()
)


def _votes(text_: str) -> tuple[int, int]:
    words = re.findall(r"[^\W\d_]+", text_.lower())
    return sum(w in _ENGLISH for w in words), sum(w in _OTHER for w in words)


def is_english(text_: str) -> bool:
    """Whether a text is in English: more of its words are common English words than common
    words of the other languages."""
    english, other = _votes(text_)
    return english > other


def plain_text(text_: str) -> str:
    """A description's Markdown as plain text: links and emphasis become their words, and what
    follows a rule or an "Also contained in:" (lists of other books) and "Preceded by:" lines go."""
    t = re.split(r"\n[ \t]*-{3,}[ \t]*(?:\n|$)", text_.replace("\r\n", "\n").replace("\r", "\n"))[0]
    t = re.split(r"\n\s*(?:Also contained in|Contains|See also)\s*:", t)[0]
    t = re.sub(r"^(?:(?:Preceded|Followed) by|Sequel to):.*$|^\s*\(?(?:Main )?Sources?\b.*$", "", t, flags=re.M)
    t = re.sub(r"\[([^\]]*)\](?:\[\d*\]|\([^)]*\))", r"\1", t)  # [words][1] and [words](url)
    t = re.sub(r"(\*{1,3}|_{1,3})(?=\S)(.+?)(?<=\S)\1", r"\2", t)  # *words*, **words**, _words_
    t = re.sub(r"^#+\s*|`|\*{2,}", "", t, flags=re.M)
    return t.replace("\\*", "*").strip()


# A first sentence about the edition rather than the book ("This deluxe trade paperback edition
# ... features French flaps", "All Random House Large Print editions are published in a
# 16-point typeface", "PerfectBound e-book exclusive: ...")
_ABOUT_THE_EDITION = re.compile(
    r"\b(?:e-?books?|large print|repackaged|typeface|now (?:available )?in (?:paperback|hardcover))\b"
    r"|^(?:This|These)\b[^.!?]*\beditions?\b",
    re.I,
)


def _tidy(text_: str) -> str:
    """Leftovers of the shops the descriptions were copied from: a source's label to start,
    first sentences about the edition rather than the book, ISBNs, "More..." links, sentences
    run together ("the law.Michael", "new coversRay"), and a passage given twice."""
    t = re.sub(r"^(?:LibraryThing|Publisher's summary|From the publisher|From Wikipedia|Product description|Synopsis)\s*:\s*",
               "", text_, flags=re.I)
    t = re.sub(r"\s*\(ISBN[^)]*\)|\bMore\.\.\.(?=[a-z])", "", t)
    t = re.sub(r"\b(new covers|point typeface)\s*(?=[A-Z])", r"\1. ", t)  # "...fantastic new coversRay Atlee is"
    t = re.sub(r"([a-z][.!?]|[.!?]\))(?=[A-Z](?:[a-z]| [a-z]))|([A-Z]\.)(?=[A-Z][a-z]{2})", r"\1\2 ", t)  # "the law.Michael", "Rose.A family", "D.C.The"
    sentences = re.split(r"(?<=[.!?])\s+", t)
    while len(sentences) > 1 and _ABOUT_THE_EDITION.search(sentences[0]):
        sentences.pop(0)
    seen: set[str] = set()
    kept = [s for s in sentences if not (s in seen or seen.add(s))]  # a passage pasted twice, once
    return " ".join(kept)


def short_description(text_: str | None, limit: int = DESCRIPTION_CHARS) -> str | None:
    """The description's opening sentences, up to about `limit` characters, as plain text and
    from its English paragraphs only (Open Library's are often in another language, or in
    English and then another); None if none is English."""
    if not text_:
        return None
    paragraphs = re.split(r"\n\s*\n", html.unescape(plain_text(text_)))
    whole = is_english(" ".join(paragraphs))
    # A paragraph with no common word of any language ("Hammond Cross is a man caught in a moral
    # dilemma.") goes with the description as a whole
    kept = [p for p in paragraphs if (is_english(p) if any(_votes(p)) else whole)]
    t = _tidy(re.sub(r"\s+", " ", " ".join(kept)).strip())
    if not t:
        return None
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


def same_book_keys(records: list[dict]) -> list[str]:
    """A key per book that two editions stage 2 kept apart would share: the title key without
    its bracketed volume number and with abbreviations spelled out, and the author's surname
    ("Harry Potter and the Sorcerer's Stone" and "... (Book 1)"; "Into Thin Air: ... Mt.
    Everest Disaster" and "... Mount Everest Disaster"). Lists show one book of each key. Since
    stage 2 joins books that are one Open Library work, this catches only editions Open Library
    doesn't know."""
    keys = []
    for r in records:
        title = re.sub(r" #[0-9-]+$", "", title_key(r["title"]))
        keys.append(" ".join(SPELLED_OUT.get(w, w) for w in title.split(" ")) + " | " + author_key(r["author"]))
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


def write_covers(cover_ids: set[int], out: Path = COVERS_OUT, cache: Path = openlibrary.COVERS) -> set[int]:
    """Each cover in the Open Library cache as a WebP at most COVER_WIDTH wide, named by its
    cover ID; a cover the cache lacks keeps the copy the site has, and covers no book uses any
    more are removed. Returns the cover IDs the site has, so a book without one shows the cover
    set in type at once instead of asking for a picture that isn't there."""
    out.mkdir(parents=True, exist_ok=True)
    have = set()
    for cover_id in sorted(cover_ids):
        source, target = cache / f"{cover_id}.jpg", out / f"{cover_id}.webp"
        if source.exists():
            try:
                with Image.open(source) as picture:
                    picture = picture.convert("RGB")
                    if picture.width < 10:  # a placeholder, not a cover
                        continue
                    if picture.width > COVER_WIDTH:
                        height = round(picture.height * COVER_WIDTH / picture.width)
                        picture = picture.resize((COVER_WIDTH, height), Image.Resampling.LANCZOS)
                    picture.save(target, "WEBP", quality=COVER_QUALITY, method=6)
            except OSError:  # not a picture
                continue
        if target.exists():
            have.add(cover_id)
    for old in out.glob("*.webp"):
        if old.stem.isdigit() and int(old.stem) not in cover_ids:
            old.unlink()
    return have


def export(out: Path = OUT, covers_out: Path = COVERS_OUT) -> dict:
    data = load_interactions()
    records = openlibrary.read_details()
    readers = data.popularity()
    on_site = write_covers({r["cover_id"] for r in records if r["cover_id"] and r["title"] not in WRONG_WORKS}, covers_out)
    books = []
    descriptions = {}
    for r in records:
        trusted = r["title"] not in WRONG_WORKS
        books.append(
            [
                display_title(r["title"]),
                tidy_case(clean_title(r["author"])),
                r["first_publish_year"] if trusted else None,
                r["cover_id"] if trusted and r["cover_id"] in on_site else None,
                int(readers[r["book_id"]]),
            ]
        )
        d = short_description(r["description"]) if trusted and r["title"] not in BAD_DESCRIPTIONS else None
        if d:
            descriptions[str(r["book_id"])] = d

    keys = same_book_keys(records)
    similar = {"v2": v2_similar(keys), "v1": v1_similar(keys)}
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "books.json": {"fields": ["title", "author", "year", "cover", "readers"], "books": books},
        # The site's demo shows the blend's lists only; v1's are still counted in results.json's "lists"
        "similar.json": {"v2": similar["v2"]},
        "descriptions.json": descriptions,
        "results.json": {**results(), "lists": list_stats(similar, [b[0] for b in books])},
        "findings.json": named_findings(findings.findings(data), books),
    }
    for name, content in files.items():
        (out / name).write_text(json.dumps(content, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    sizes = {name: (out / name).stat().st_size for name in files}
    sizes[f"{len(on_site):,} covers"] = sum(f.stat().st_size for f in covers_out.glob("*.webp"))
    return sizes


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
    with a 95% interval for its hit rate over the users scored; and, from stage 1 on v1's own
    split, what fixing v1's averaging bug does to its user-based model."""
    stage1 = json.loads((REPORTS / "phase1-evaluation.json").read_text(encoding="utf-8"))
    stage2 = json.loads((REPORTS / "phase2-books.json").read_text(encoding="utf-8"))
    stage3 = json.loads((REPORTS / "phase3-models.json").read_text(encoding="utf-8"))
    pick = lambda rows, name: next(x for x in rows if x["model"] == name)  # noqa: E731

    def user_cf(version: str) -> dict:
        name = f"v1 user-based CF, {version}"
        return {"hit_rate": pick(stage1["ranking"]["rows"], name)["hit_rate"], "mae": pick(stage1["rating"], name)["mae"]}

    users = stage3["users_scored"]
    keep = lambda x: {  # noqa: E731
        **{k: x[k] for k in ("ndcg", "hit_rate", "coverage", "mean_popularity")},
        "hit_rate_ci": wilson(x["hit_rate"], users),
    }
    rows2, rows3 = stage2["ranking"]["rows"], stage3["tuned"]
    return {
        "users": users,
        "training": stage3["counts"]["training_interactions"],
        "bug": {"as_shipped": user_cf("as shipped"), "corrected": user_cf("corrected")},
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
