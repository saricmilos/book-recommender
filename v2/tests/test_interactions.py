import numpy as np
import pandas as pd

from bookrec.data import Split
from bookrec.interactions import build, map_editions


def _books():
    return pd.DataFrame(
        {
            "book_key": ["bel canto | patchett", "mortal fear | cook", "mortal fear | iles"],
            "title": ["Bel Canto", "Mortal Fear", "Mortal Fear"],
            "author": ["Ann Patchett", "Robin Cook", "Greg Iles"],
        }
    )


def _split():
    train = pd.DataFrame({"user_id": [1, 2], "user_idx": [0, 1], "item_idx": [0, 1], "book_rating": [8.0, 6.0]})
    test = pd.DataFrame({"user_id": [1], "user_idx": [0], "item_idx": [2], "book_rating": [9.0]})
    return Split(filtered=train, train=train, test=test, titles=_books()["title"].to_numpy(),
                 user_ids=np.array([1, 2]), cleaned_rows=3)


def test_map_editions_by_title_and_author():
    rows = pd.DataFrame({"isbn": ["A"], "book_key": ["bel canto | patchett"]})
    catalogue = pd.DataFrame(
        {
            "isbn": ["A", "B", "C", "D", "E"],
            "title": ["Bel Canto", "Bel Canto: A Novel", "Mortal Fear", "Mortal Fear", "Night"],
            "author": ["Ann Patchett", "ANN PATCHETT", "Greg Iles", "Lisa Jackson", "Elie Wiesel"],
        }
    )
    assert map_editions(_books(), rows, catalogue) == {"A": 0, "B": 0, "C": 2}  # D: no author matches; E: no book


def test_build_keeps_the_benchmark_adds_the_rest_and_never_trains_on_a_held_out_book():
    rows = pd.DataFrame({"isbn": ["A", "M1", "M2"], "book_key": ["bel canto | patchett", "mortal fear | cook", "mortal fear | iles"]})
    catalogue = pd.DataFrame({"isbn": ["A", "B", "M1", "M2", "M3"],
                              "title": ["Bel Canto", "Bel Canto: A Novel", "Mortal Fear", "Mortal Fear", "Mortal Fear"],
                              "author": ["Ann Patchett", "Ann Patchett", "Robin Cook", "Greg Iles", "Greg Iles"]})
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 3, 3, 3],
            "isbn": ["A", "B", "M3", "B", "B", "M2", "M2"],
            "rating": [8, 0, 0, 0, 0, 7, 9],
        }
    )
    data = build(_split(), _books(), rows, catalogue, ratings)
    t = data.train

    # User 1's held-out book (Iles's Mortal Fear) never reaches training, not even through
    # another edition (M3); their extra unrated copy of Bel Canto (B) is the same book as their
    # training rating, so the benchmark's rating stands.
    assert not ((t["user_id"] == 1) & (t["item_idx"] == 2)).any()
    assert t[t["user_id"] == 1]["rating"].tolist() == [8.0]
    # User 2 gains an unrated Bel Canto; user 3, new, gets the next user_idx.
    assert t[(t["user_id"] == 2) & (t["item_idx"] == 0)]["explicit"].tolist() == [False]
    assert data.user_ids.tolist() == [1, 2, 3] and set(t[t["user_id"] == 3]["user_idx"]) == {2}
    # User 3 rated Iles's book twice through one edition: one interaction, rating averaged.
    assert t[(t["user_id"] == 3) & (t["item_idx"] == 2)]["rating"].tolist() == [8.0]
    assert data.counts["left_out_as_held_out"] == 1
    assert data.train_items()[0].tolist() == [0]  # benchmark users only
    assert data.popularity().tolist() == [3, 1, 1]
