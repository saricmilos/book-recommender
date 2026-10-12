"""Stage 3: what the extra training data does, and tuned models on all of it.

1. Every model, with fixed settings, on four versions of the training data, from the
   benchmark's ratings alone to every interaction, all scored on the same held-out ratings.
2. Each model's settings chosen on a validation set carved out of training (never the
   held-out ratings), then the chosen model trained on everything and scored once.
3. A blend of EASE, RP3beta and the text model, its weights chosen on the same validation set.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from bookrec import text, topn
from bookrec.interactions import Interactions, benchmark_only, load_interactions, validation_split
from bookrec.metrics import ranking_metrics, recall_among, relevant_items

K = 10
RELEVANT_FROM = 4
PRIMARY = "ndcg"  # what tuning maximises

# Fixed settings for part 1: v1's where v1 had one (50 books per table, 10 neighbours).
DEFAULTS = {
    "Most popular": {},
    "Dot-product table, top 50 (v1's API)": {"k": 50, "cosine": False},
    "Cosine table, top 50 (v1's notebook)": {"k": 50},
    "User kNN, 10 neighbours (v1's user-based)": {"k": 10},
    "RP3beta": {"alpha": 1.0, "beta": 0.5, "k": 100},
    "EASE": {"lam": 500.0},
}

# Grids reach past each setting's best on the validation set, so that no best sits at an edge.
GRIDS = {
    "Item kNN (cosine)": [{"k": k, "shrink": s} for k in (2, 3, 5, 8, 10, 20, 50, 100, 200, 400) for s in (0, 10, 50, 100)],
    "User kNN": [{"k": k} for k in (5, 10, 25, 50, 100, 200, 400, 800)],
    "RP3beta": [
        {"alpha": a, "beta": b, "k": k}
        for a in (0.4, 0.6, 0.8, 1.0)
        for b in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        for k in (50, 200, 500)
    ],
    "EASE": [{"lam": lam} for lam in (10.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0)],
    "Text (TF-IDF)": [{"k": k} for k in (3, 5, 10, 20, 50, 100, 300)],
}

# The blend: per-user z-scores of EASE, RP3beta (each at its tuned settings) and the text
# model, added with weights that sum to 1, in steps of 0.1, text at most 0.5.
BLEND = "Blend: EASE + RP3beta + text"
BLEND_TEXT_K = (5, 10, 20, 50)
BLEND_WEIGHTS = [
    {"ease": a / 10, "rp3beta": (10 - a - c) / 10, "text": c / 10}
    for a in range(0, 11)
    for c in range(0, 6)
    if a + c <= 10
]


class Scorer:
    """Scores one Interactions under many settings, computing the shared parts once."""

    def __init__(self, data: Interactions):
        self.data = data
        self._counts: np.ndarray | None = None
        self._walks: dict[float, np.ndarray] = {}
        self._neighbours: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def counts(self) -> np.ndarray:
        if self._counts is None:
            self._counts = topn.co_occurrence(self.data)
        return self._counts

    def __call__(self, model: str, params: dict) -> np.ndarray:
        d = self.data
        if model == "Most popular":
            return topn.popularity(d)
        if model.startswith(("Dot-product", "Cosine", "Item kNN")):
            return topn.item_knn(d, counts=self.counts, **params)
        if model.startswith("User kNN"):
            k_max = max(200, params["k"])
            if self._neighbours is None or self._neighbours[0].shape[1] < k_max:
                self._neighbours = topn.user_neighbours(d, k_max)
            return topn.user_knn(d, params["k"], neighbours=self._neighbours)
        if model == "RP3beta":
            a = params["alpha"]
            if a not in self._walks:
                self._walks[a] = topn.rp3beta_walk(d, a)
            return topn.rp3beta(d, a, params["beta"], params["k"], walk=self._walks[a])
        if model == "EASE":
            return topn.ease(d, params["lam"], counts=self.counts)
        if model == "Text (TF-IDF)":
            return text.text_knn(d, params["k"])
        raise KeyError(model)

    def blend_parts(self, settings: dict[str, dict], text_ks: tuple[int, ...]) -> dict[str, np.ndarray]:
        """Per-user z-scores of EASE and RP3beta at their tuned settings, and of the text model
        for each k in text_ks."""
        train_items = self.data.train_items()
        parts = {
            "ease": topn.standardise(self("EASE", settings["EASE"]), train_items),
            "rp3beta": topn.standardise(self("RP3beta", settings["RP3beta"]), train_items),
        }
        for k in text_ks:
            parts[f"text {k}"] = topn.standardise(self("Text (TF-IDF)", {"k": k}), train_items)
        return parts


def blend(parts: dict[str, np.ndarray], params: dict) -> np.ndarray:
    return (
        params["ease"] * parts["ease"]
        + params["rp3beta"] * parts["rp3beta"]
        + params["text"] * parts[f"text {params['text_k']}"]
    )


def data_settings(data: Interactions) -> dict[str, Interactions]:
    """The four versions of the training data in part 1."""
    t, nb = data.train, data.n_benchmark_users
    bench_pairs = set(zip(data.split.train["user_idx"], data.split.train["item_idx"]))
    is_rating = np.fromiter((p in bench_pairs for p in zip(t["user_idx"], t["item_idx"])), dtype=bool, count=len(t))
    is_benchmark_user = t["user_idx"].to_numpy() < nb

    def subset(mask: np.ndarray) -> Interactions:
        return Interactions(data.split, t[mask].reset_index(drop=True), data.user_ids, data.editions, {})

    return {
        "Benchmark ratings only": benchmark_only(data.split),
        "+ other readers": subset(is_rating | ~is_benchmark_user),
        "+ users' own unrated books": subset(is_benchmark_user),
        "All interactions": data,
    }


def _metrics(scores: np.ndarray, data: Interactions, relevant: dict, popularity: np.ndarray) -> dict:
    return ranking_metrics(scores, data.train_items(), relevant, popularity, K)


def run_stage3(progress: Callable[[str], None] = lambda s: None) -> dict:
    data = load_interactions()
    split = data.split
    test_relevant = relevant_items(split.test, RELEVANT_FROM)
    popularity = split.popularity()  # benchmark training ratings, as in every stage's "avg. popularity"

    # Part 1: fixed settings, four versions of the training data.
    extra_data = []
    for setting, version in data_settings(data).items():
        progress(f"part 1: {setting}")
        score = Scorer(version)
        for model, params in DEFAULTS.items():
            m = _metrics(score(model, params), version, test_relevant, popularity)
            extra_data.append({"data": setting, "model": model, "params": params, **m})

    # Part 2: settings chosen on a validation set, then one score on the held-out ratings.
    tuning_data, validation = validation_split(data)
    validation_relevant = relevant_items(validation, RELEVANT_FROM)
    tune = Scorer(tuning_data)
    final = Scorer(data)
    # The less popular half of the books, by training interactions: where text should help.
    counts = data.popularity()
    less_popular = set(np.flatnonzero(counts < np.median(counts)).tolist())
    train_items = data.train_items()

    def scored(scores: np.ndarray) -> dict:
        m = _metrics(scores, data, test_relevant, popularity)
        return {**m, "less_popular_recall": recall_among(scores, train_items, test_relevant, less_popular, K)}

    def trial(scores: np.ndarray, params: dict) -> dict:
        m = _metrics(scores, tuning_data, validation_relevant, popularity)
        return {"params": params, "ndcg": m["ndcg"], "hit_rate": m["hit_rate"], "coverage": m["coverage"]}

    tuned = [{"model": "Most popular", "params": {}, "validation_ndcg": None, "trials": [], **scored(final("Most popular", {}))}]
    for model, grid in GRIDS.items():
        progress(f"part 2: tuning {model}")
        trials = [trial(tune(model, params), params) for params in grid]
        best = max(trials, key=lambda x: x[PRIMARY])  # the first of equals: the grid's order
        tuned.append(
            {"model": model, "params": best["params"], "validation_ndcg": best["ndcg"], "trials": trials,
             **scored(final(model, best["params"]))}
        )

    # Part 3: the blend, its weights chosen on the validation set.
    progress("part 3: blending")
    settings = {x["model"]: x["params"] for x in tuned}
    parts = tune.blend_parts(settings, BLEND_TEXT_K)
    trials = [trial(blend(parts, {**w, "text_k": k}), {**w, "text_k": k}) for k in BLEND_TEXT_K for w in BLEND_WEIGHTS]
    best = max(trials, key=lambda x: x[PRIMARY])
    final_parts = final.blend_parts(settings, (best["params"]["text_k"],))
    tuned.append(
        {"model": BLEND, "params": best["params"], "validation_ndcg": best["ndcg"], "trials": trials,
         **scored(blend(final_parts, best["params"]))}
    )

    return {
        "counts": data.counts,
        "users_scored": len(test_relevant),
        "validation": {"ratings": len(validation), "users": len(validation_relevant)},
        "less_popular": {"books": len(less_popular), "below": float(np.median(counts))},
        "extra_data": extra_data,
        "tuned": tuned,
    }


# --- Report -------------------------------------------------------------------------------


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in rows])


def _settings(params: dict) -> str:
    names = {"k": "k", "shrink": "shrink", "alpha": "α", "beta": "β", "lam": "λ", "cosine": "cosine",
             "ease": "EASE", "rp3beta": "RP3beta", "text": "text", "text_k": "text k"}
    return ", ".join(f"{names[k]} = {v:g}" if isinstance(v, float) else f"{names[k]} = {v}" for k, v in params.items()) or "none"


def render_stage3(r: dict, stage2: dict, versions: str) -> str:
    c = r["counts"]
    sources = _table(
        ["", "Interactions"],
        [
            ["The benchmark's training ratings (stages 1 and 2)", f"{c['benchmark_training_ratings']:,}"],
            ["Benchmark users' further ratings of these books", f"{c['extra_benchmark_users_rated']:,}"],
            ["Benchmark users' unrated books", f"{c['extra_benchmark_users_unrated']:,}"],
            [f"Other readers' ratings ({c['other_users']:,} readers)", f"{c['extra_other_users_rated']:,}"],
            ["Other readers' unrated books", f"{c['extra_other_users_unrated']:,}"],
            ["**Training interactions now**", f"**{c['training_interactions']:,}**"],
            ["Left out: a user and book whose rating is held out", f"{c['left_out_as_held_out']:,}"],
        ],
    )
    settings = list(dict.fromkeys(x["data"] for x in r["extra_data"]))
    models = list(dict.fromkeys(x["model"] for x in r["extra_data"]))
    cell = {(x["model"], x["data"]): x for x in r["extra_data"]}

    def grid(metric: str, spec: str) -> str:
        return _table(["Model"] + settings, [[m] + [format(cell[(m, s)][metric], spec) for s in settings] for m in models])

    best_stage2 = max(stage2["ranking"]["rows"], key=lambda x: x["ndcg"])
    top_hit_stage2 = max(stage2["ranking"]["rows"], key=lambda x: x["hit_rate"])
    tuned = _table(
        ["Model", "Settings chosen", "Validation NDCG@10", "Precision@10", "Recall@10", "NDCG@10", "Hit rate",
         "Recall, less popular half", "Coverage", "Avg. popularity"],
        [
            [
                x["model"],
                _settings(x["params"]),
                "" if x["validation_ndcg"] is None else f"{x['validation_ndcg']:.4f}",
                f"{x['precision']:.4f}",
                f"{x['recall']:.4f}",
                f"**{x['ndcg']:.4f}**",
                f"{x['hit_rate']:.1%}",
                f"{x['less_popular_recall']:.1%}",
                f"{x['coverage']:.1%}",
                f"{x['mean_popularity']:.1f}",
            ]
            for x in r["tuned"]
        ],
    )

    def trial_rows(x: dict) -> list[dict]:
        if x["model"] == BLEND:  # 200-odd weightings: the best 15
            return sorted(x["trials"], key=lambda t: -t["ndcg"])[:15]
        return x["trials"]

    trials = "\n\n".join(
        f"<details><summary>{x['model']}: {len(x['trials'])} settings tried"
        + (", the best 15 shown" if x["model"] == BLEND else "")
        + "</summary>\n\n"
        + _table(
            ["Settings", "Validation NDCG@10", "Validation hit rate", "Coverage"],
            [[_settings(t["params"]), f"{t['ndcg']:.4f}", f"{t['hit_rate']:.1%}", f"{t['coverage']:.1%}"] for t in trial_rows(x)],
        )
        + "\n\n</details>"
        for x in r["tuned"]
        if x["trials"]
    )
    v, lp = r["validation"], r["less_popular"]
    return f"""# Stage 3: models

Generated by `python -m bookrec.evaluate`; edit the code, not this file.

The benchmark is unchanged: the same {r['users_scored']:,} users and the same held-out ratings as
stage 2, with a hit meaning a held-out book rated {RELEVANT_FROM} or more in the top {K}. What changes
is what the models may learn from, and the models.

## The training data

Stages 1 and 2 trained on the benchmark's own ratings: 1-10 ratings of the 2,405 books by the
2,272 users, after v1 dropped every 0 ("had the book, gave no rating") and every user and book
with fewer than 10 ratings. From stage 3, one rule (`bookrec/interactions.py`):

> Training may use any interaction in Book-Crossing with one of the 2,405 books, by any user,
> rated (1-10) or not (0: the user had the book but gave no rating), except one that covers a
> user and book whose rating is held out, through any edition of the book.

The books' editions are now {c['editions']:,} ISBNs (the benchmark's ratings use {c['editions_in_benchmark']:,}): an ISBN
whose title has a book's title key and whose author loosely matches the book's belongs to it.
A book a user has in training, rated or not, is never recommended to them.

{sources}

## What the extra data does

Every model with fixed settings (v1's where v1 had one: 50 books per table, 10 neighbours) on
four versions of the training data, all scored on the same held-out ratings. Models learn
from whether a user has a book, not from the rating. "+ other readers" adds the ratings and
unrated books of readers outside the benchmark; "+ users' own unrated books" adds the
benchmark users' own further interactions instead; "All interactions" adds both.

NDCG@10:

{grid('ndcg', '.4f')}

Hit rate:

{grid('hit_rate', '.1%')}

Coverage (share of the books recommended to anyone):

{grid('coverage', '.1%')}

## Tuned models, on all interactions

Each model's settings were chosen on a validation set, never on the held-out ratings: then the
model was trained on all interactions and scored once. Stage 2's best were NDCG@10
{best_stage2['ndcg']:.4f} ({best_stage2['model']}) and a hit rate of {top_hit_stage2['hit_rate']:.1%} ({top_hit_stage2['model']}).

{tuned}

"Recall, less popular half" is the share of the held-out ratings of 4 or more for the
{lp['books']:,} books with fewer than {lp['below']:g} training interactions that made the user's top 10:
how well a model finds the books collaborative data knows least about.

The text model (`bookrec/text.py`) compares books by the words of their title, author and
Open Library subjects and description, weighted by TF-IDF; it learns nothing from who reads
what, beyond the user's own books. The blend turns each user's scores from EASE, RP3beta (at
their tuned settings) and the text model into z-scores over the books the user could be
recommended, and adds them with weights that sum to 1, in steps of 0.1 with text at most 0.5,
for text k of {', '.join(str(k) for k in BLEND_TEXT_K)}; the weights too were chosen on the validation set.

## How the settings were chosen

The validation set is {v['ratings']:,} ratings, 20% of each benchmark user's training ratings (seed 7),
taken out of training while tuning; {v['users']:,} users have one rated {RELEVANT_FROM} or more. Every setting
in each model's grid was trained without them and ranked by NDCG@10 on them; the best was then
trained on all interactions.

{trials}

Environment: {versions}.
"""
