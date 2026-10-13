"""Stage 3's training-data rule on the real data. The counts are the reviewed ones: if the
rule or stage 2's books change, they change, and the stage 3 report must be regenerated."""

import numpy as np
import pytest

from bookrec.interactions import load_interactions, validation_split
from bookrec.stage3 import data_settings


@pytest.fixture(scope="module")
def data():
    return load_interactions()


def test_counts_are_the_reviewed_ones(data):
    assert data.counts == {
        "editions": 8_884,
        "editions_in_benchmark": 5_160,
        "benchmark_training_ratings": 45_047,
        "extra_benchmark_users_rated": 2_870,
        "extra_benchmark_users_unrated": 94_449,
        "extra_other_users_rated": 61_950,
        "extra_other_users_unrated": 75_889,
        "other_users": 45_960,
        "left_out_as_held_out": 11_052,
        "held_out_ratings": 11_052,
        "training_interactions": 280_205,
    }


def test_no_training_interaction_covers_a_held_out_user_and_book(data):
    held_out = set(zip(data.split.test["user_id"], data.split.test["item_idx"]))
    assert held_out.isdisjoint(zip(data.train["user_id"], data.train["item_idx"]))


def test_the_benchmarks_training_ratings_are_all_kept_unchanged(data):
    bench = data.train.merge(data.split.train, on=["user_id", "item_idx"])
    assert len(bench) == len(data.split.train)
    assert (bench["rating"] == bench["book_rating"]).all()


def test_benchmark_users_keep_their_indices(data):
    first = data.user_ids[: data.n_benchmark_users]
    assert (first == data.split.user_ids).all()


def test_the_validation_set_comes_from_training_and_leaves_it(data):
    tuning, validation = validation_split(data)
    pairs = set(zip(validation["user_idx"], validation["item_idx"]))
    assert pairs <= set(zip(data.split.train["user_idx"], data.split.train["item_idx"]))
    assert pairs.isdisjoint(zip(tuning.train["user_idx"], tuning.train["item_idx"]))
    assert len(tuning.train) == len(data.train) - len(validation)
    assert validation["user_idx"].nunique() == data.n_benchmark_users  # everyone has at least one
    assert 0.19 < len(validation) / len(data.split.train) < 0.22


def test_text_similarity_compares_every_book_with_every_other(data):
    from bookrec.text import text_similarity

    sim = text_similarity()
    assert sim.shape == (data.n_items, data.n_items)
    assert np.allclose(sim, sim.T) and (np.diag(sim) == 0).all()
    assert sim.min() >= 0 and sim.max() <= 1 + 1e-9
    # The two Harry Potter books of the benchmark are nearer each other than to 1984.
    titles = list(data.split.titles)
    hp = [i for i, t in enumerate(titles) if t.startswith("Harry Potter and the Chamber")][0]
    hp2 = [i for i, t in enumerate(titles) if t.startswith("Harry Potter and the Prisoner")][0]
    orwell = titles.index("1984")
    assert sim[hp, hp2] > sim[hp, orwell]


def test_the_four_data_settings_nest(data):
    sizes = {name: len(d.train) for name, d in data_settings(data).items()}
    assert sizes["Benchmark ratings only"] == len(data.split.train)
    assert sizes["+ other readers"] + sizes["+ users' own unrated books"] - sizes["Benchmark ratings only"] == sizes["All interactions"]
