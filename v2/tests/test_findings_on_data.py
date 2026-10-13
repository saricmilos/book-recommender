"""The page's findings on the real data: the numbers its text is written around."""

import pytest

from bookrec import findings
from bookrec.interactions import load_interactions


@pytest.fixture(scope="module")
def found():
    return findings.findings()


def test_most_interactions_are_a_zero(found):
    assert found["interactions"] == 1_149_780
    assert found["ratings"][0] == 716_109
    assert max(found["ratings"][1:]) == found["ratings"][8]  # of the ratings given, 8 is the most common


def test_the_zeros_lift_ease(found):
    lift = found["ease_ndcg"]
    assert lift["ratings_only"] < lift["own_books"] < lift["all"]


def test_most_editions_have_one_reader(found):
    bins = found["editions"]["by_readers"]
    with_readers = sum(b["editions"] for b in bins)
    assert with_readers == 270_151
    assert bins[0]["editions"] / with_readers > 0.5


def test_the_books_gain_readers(found):
    books = found["books"]
    assert books["count"] == 2_366
    assert books["median_v2_readers"] > 5 * books["median_v1_ratings"]


def test_wild_animus_is_the_most_read_and_the_least_liked_of_them(found):
    data = load_interactions()
    top = found["most_read"]
    assert data.split.titles[top[0]["book"]] == "Wild Animus"
    assert top[0]["readers"] > 1.5 * top[1]["readers"]
    assert top[0]["mean_rating"] < 5 < min(x["mean_rating"] for x in top[1:])
    assert top[0]["readers"] == data.popularity()[top[0]["book"]]  # as the demo counts readers


def test_title_examples_hold(found):
    examples = found["titles"]["examples"]
    assert len(examples["merged"]["v1_titles"]) == 2
    assert len(examples["split"]["books"]) == 2
