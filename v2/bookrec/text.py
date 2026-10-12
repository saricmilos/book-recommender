"""A text model: books are similar when their words are, in the details fetched from Open
Library (description and subjects) and in their own title and author.

Each book is one document: title, author, subjects and description. TF-IDF weighs each word
by how rare it is across the books (English stop words out, single words and pairs, kept when
in at least 2 books and at most half of them), and books are compared by cosine. A user's
score for a book adds up its similarity to the user's books, keeping each book's k nearest,
as in item kNN. The text settings are fixed; only k is tuned.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from bookrec import openlibrary
from bookrec.interactions import Interactions
from bookrec.topn import _benchmark_rows, _keep_top_k


def book_text(record: dict) -> str:
    return ". ".join([record["title"], record["author"], "; ".join(record["subjects"]), record["description"] or ""])


@lru_cache(maxsize=1)
def text_similarity() -> np.ndarray:
    """Books x books cosine similarity of their TF-IDF vectors, 0 on the diagonal."""
    records = openlibrary.read_details()
    if records is None:
        raise FileNotFoundError("data/book_details.jsonl is missing: run python -m bookrec.openlibrary")
    tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2, max_df=0.5, ngram_range=(1, 2))
    x = tfidf.fit_transform(book_text(r) for r in records)
    sim = (x @ x.T).toarray()
    np.fill_diagonal(sim, 0)
    return sim


def text_knn(data: Interactions, k: int = 10, similarity: np.ndarray | None = None) -> np.ndarray:
    sim = text_similarity() if similarity is None else similarity
    return (_benchmark_rows(data) @ _keep_top_k(sim, k)).astype(float)
