import numpy as np
import pandas as pd

from bookrec.data import Split


def toy_split(ratings: list[tuple[int, int, float]], n_users: int, n_items: int) -> Split:
    """A Split from (user_idx, item_idx, rating) training rows, with no held-out rows."""
    train = pd.DataFrame(ratings, columns=["user_idx", "item_idx", "book_rating"])
    train["user_id"] = train["user_idx"]
    train["book_title"] = "book " + train["item_idx"].astype(str)
    return Split(
        filtered=train,
        train=train,
        test=train.iloc[:0],
        titles=np.array([f"book {i}" for i in range(n_items)]),
        user_ids=np.arange(n_users),
        cleaned_rows=len(train),
    )
