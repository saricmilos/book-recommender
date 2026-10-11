# book-recommender

One book recommender in two versions, on the Book-Crossing ratings in `datasets/`.

| Folder | What it is | Rule |
|---|---|---|
| `v1/` | The version Miloš built by hand, November 2025 to early 2026: EDA, user- and item-based CF, content-based filtering, the FastAPI app, the report. | Frozen. Never edit, move, reformat or fix anything under `v1/`, not even a typo or a broken path. |
| `v2/` | The version built with Claude Code, from October 2026. | All new work goes here. |
| `datasets/` | `Books.csv`, `Ratings.csv`, `Users.csv`, shared by both versions. | Read-only. |

Tag `v1.0` is v1 in its original layout, before it moved into `v1/`. A problem found in v1 is
written down and fixed in v2, never in v1. `.claude/settings.json` denies the file tools on
`v1/` and `datasets/`; scripts can still write there, so this rule is what holds.

## v1 facts that matter for v2

- The live API (`what-else-should-i-read.fly.dev`, Fly app `what-else-should-i-read`) serves the
  book demo on cassiopeiai.com. It needs `item_encoder.pkl`, which is not in `v1/models/`
  (`*.pkl` is gitignored); the only copy is inside the deployed image. Do not redeploy v1 from
  this checkout.
- The notebooks load the data from the absolute path of the machine they were written on, so
  they don't run as they are. v2 reads `datasets/` by a relative path.

## Commits

A subject line in plain words saying what changed (the history's style), a short body when the
why isn't obvious, and the `Co-Authored-By` trailer.
