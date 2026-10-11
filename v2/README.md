# v2

The second version of the book recommender, built with Claude Code from October 2026: I set
the goals, review every change and check the results. Every number below comes from code in
this folder, and the tests check that code, including a re-run of v1.

## Run it

```bash
cd v2
python -m venv .venv
.venv/Scripts/activate          # Windows; on macOS or Linux: source .venv/bin/activate
pip install -e ".[test]"
pytest                          # 16 tests, about 30 seconds, v1's re-run included
python -m bookrec.evaluate      # writes reports/phase1-evaluation.md and .json
```

## Stages

Every stage is measured on the same split as v1, so the numbers stay comparable.

| Stage | What | Status |
|---|---|---|
| 1. Evaluation | Re-run v1, add baselines, score v1 again under the same rules | Done: [report](reports/phase1-evaluation.md) |
| 2. Data quality | Clean titles, merge editions of the same book, fetch book details once | Next |
| 3. Models | Use the 0 ratings as signals of interest, add text embeddings, combine both | |
| 4. Demo | Recommendations on saricmilos.com, with no server | |

## What stage 1 found

- **v1 reproduces exactly.** Every count and metric its notebook printed comes out the same
  from v1's own code, and the API's model file is rebuilt with identical values.
- **v1's rating errors came from an averaging bug.** v1 averaged each user's and book's
  ratings over every user or book in the matrix, so each missing rating counted as a 0 and
  every average sat near 0. That gave MAE 7.36 (user-based) and 5.29 (item-based) on a 1-10
  scale. Averaged correctly, both models reach about 1.31, but the simple bias baseline (the
  overall mean plus each user's and each book's offset) does better still, at 1.17.
- **For recommending, v1's user-based model as shipped is the best so far:** 27.7% of users
  find one of their held-out books in its top 10, against 14.9% for the most popular books
  (NDCG@10 0.082 against 0.030). With every average near 0, its "predicted rating" becomes
  a similarity-weighted sum of the ratings of the 10 most similar readers, so books more of
  them read rank higher. That is a good signal for what someone reads next. Fixing the
  averaging gives sensible ratings and weaker recommendations (7.2%).
- **The API serves a different model from the one v1 measured.** v1's notebook evaluated
  cosine similarity; the API's table uses raw dot products. The dot-product table is about
  as accurate (26.3%) but recommends from 15.8% of the books, mostly bestsellers (99
  training ratings per recommended book, against 18 for cosine). The cosine table is a
  little less accurate (21.4%) and much more varied (58.1% of the books).

**The bar for v2:** beat a 27.7% hit rate and NDCG@10 of 0.082 without narrowing the
recommendations to bestsellers.

## Code

- `bookrec/data.py`: the shared split, produced by v1's own functions (v1 is imported, never changed).
- `bookrec/metrics.py`: rating errors and top-10 metrics.
- `bookrec/models.py`: baselines, v1 as shipped, v1 corrected, and the similarity tables.
- `bookrec/evaluate.py`: runs everything and writes the report.
- `tests/`: small worked examples for every metric and fix, and the v1 re-run that guards the rest.

The library versions are pinned to v1's. `tests/test_v1_reproduction.py` must still pass
before they move.
