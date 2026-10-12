import numpy as np
import pandas as pd
import pytest

from bookrec import findings
from bookrec.interactions import Interactions
from conftest import toy_split


def test_rating_counts_include_the_zeros():
    ratings = pd.DataFrame({"rating": [0, 0, 0, 8, 10]})
    assert findings.rating_counts(ratings) == [3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]


def test_readers_per_edition_counts_readers_once_and_only_listed_editions():
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 3, 1, 2, 3, 4, 5, 1],
            "isbn": ["A", "A", "A", "B", "C", "C", "C", "C", "C", "X"],  # 1 has A twice; X isn't listed
        }
    )
    bins = findings.readers_per_edition(ratings, {"A", "B", "C"})
    counts = {(b["from"], b["to"]): b["editions"] for b in bins}
    assert counts[(1, 1)] == 1  # B
    assert counts[(2, 4)] == 1  # A: two readers
    assert counts[(5, 9)] == 1  # C
    assert sum(counts.values()) == 3


def test_most_read_counts_readers_and_rates_only_ratings():
    split = toy_split([(0, 0, 8.0)], n_users=3, n_items=2)
    train = pd.DataFrame(
        {
            "user_id": [0, 1, 2, 0],
            "user_idx": [0, 1, 2, 0],
            "item_idx": [0, 0, 0, 1],
            "rating": [8.0, np.nan, 4.0, 6.0],  # NaN: had it, no rating
            "explicit": [True, False, True, True],
        }
    )
    data = Interactions(split, train, np.arange(3), {}, {})
    top = findings.most_read(data, n=2)
    assert top[0] == {"book": 0, "readers": 3, "rated": 2, "mean_rating": 6.0}
    assert top[1]["book"] == 1


def _books(**change):
    rows = {
        "book_id": [0, 1, 2, 3],
        "v1_titles": ["Bel Canto | Bel Canto: A Novel", "Lightning", "Lightning", "Angels &amp; Demons"],
    }
    rows.update(change)
    return pd.DataFrame(rows)


def test_title_examples_are_found():
    examples = findings.title_examples(_books())
    assert examples["merged"] == {"book": 0, "v1_titles": ["Bel Canto", "Bel Canto: A Novel"]}
    assert examples["split"] == {"v1_title": "Lightning", "books": [1, 2]}
    assert examples["escaped"] == {"book": 3, "v1_title": "Angels &amp; Demons"}


def test_title_examples_raise_when_one_no_longer_holds():
    with pytest.raises(RuntimeError):
        findings.title_examples(_books(v1_titles=["Bel Canto", "Lightning", "Lightning", "Angels &amp; Demons"]))
