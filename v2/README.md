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
pytest                          # 107 tests, under 3 minutes, v1's re-run included
python -m bookrec.evaluate      # about 12 minutes: writes the reports in reports/ and data/books.csv
python -m bookrec.export        # about 2 minutes: writes the demo's data into the site
```

Book details were fetched once from [Open Library](https://openlibrary.org) with
`python -m bookrec.openlibrary` (about 45 minutes at 1 request a second) and are committed in
`data/book_details.jsonl`, so nothing above needs the network.

## Stages

Every stage is measured on the same split as v1, so the numbers stay comparable.

| Stage | What | Status |
|---|---|---|
| 1. Evaluation | Re-run v1, add baselines, score v1 again under the same rules | Done: [report](reports/phase1-evaluation.md) |
| 2. Data quality | Identify books properly: merge editions, split different books that share a title; fetch book details once | Done: [report](reports/phase2-books.md), [books](data/books.csv), [details](data/book_details.jsonl) |
| 3. Models | Train on every interaction with the books, rated or not; tune collaborative models and a text model on a validation set; blend them | Done: [report](reports/phase3-models.md) |
| 4. Demo | Recommendations on saricmilos.com, with no server: each book's ten worked out ahead of time | Done: [the demo](https://www.saricmilos.com/projects/book-recommender) |

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

## What stage 2 found

- **v1's 2,508 books are 2,405.** v1 identified a book by its exact title. 173 books had been
  split across several titles, mostly editions ("Bel Canto" and "Bel Canto: A Novel"), and 75
  titles mixed different authors' books ("Mortal Fear" by Greg Iles and by Robin Cook,
  "Lightning" by Danielle Steel and by Dean Koontz). The rules lean towards changing nothing
  and are listed in the [report](reports/phase2-books.md).
- **The similarity tables gain most.** On the same users and the same held-out ratings, the
  cosine table v1 measured goes from a 21.4% to a 28.0% hit rate, and its NDCG@10 from 0.070
  to 0.092, the best of any model, while recommending from 70% of the books. The dot-product
  table v1's API serves has the highest hit rate (30.7%) but still keeps to bestsellers: 19% of
  the books, 109 training ratings per recommended book.
- **Rating error improves too** (about 1.26 MAE for both corrected v1 models), but the bias
  baseline is still better at 1.17.
- **Every book is on Open Library**, found through one of its editions' ISBNs: 99.9% have a
  cover, 87% a description and 99% subjects, for stage 3's text model and stage 4's demo.
  Open Library has errors of its own (it credits Robin Cook's novels to Lisa Jackson and files
  some editions under the wrong work), so v2 keeps the authors from the rated editions,
  prefers a work whose title and author match the book's, and lists the 121 books whose work
  still has another title: mostly the original-language title, a handful of wrong works.

**The bar for v2's models:** NDCG@10 above 0.092 and a hit rate above 30.7%, with the
cosine table's variety rather than the dot-product table's bestsellers.

## What stage 3 found

- **One rule gives six times the training data.** Training may now use any interaction in
  Book-Crossing with the 2,405 books, by any user, rated or not, except one that covers a
  held-out user and book through any edition. That takes training from 45,145 interactions to
  280,669 (a median of 81 per book instead of 13), and the held-out ratings stay exactly as
  they were. A book a user already has, rated or not, is never recommended to them.
- **More data helps the similarity models; raw counts get worse.** With fixed settings, EASE
  goes from NDCG@10 0.108 to 0.148, RP3beta from 0.096 to 0.152 and the cosine table v1
  measured from 0.095 to 0.108. Both sources help on their own: other readers' interactions,
  which can't carry anything about the held-out ratings, and the benchmark users' own unrated
  books. The raw dot-product table v1's API serves gets worse (0.088 to 0.045) and the most
  popular books stay flat: unrated books lean even more towards bestsellers.
- **Tuned on a validation set, EASE recommends best:** NDCG@10 0.162 and a 46.5% hit rate,
  against stage 2's 0.092 and 30.7%, with settings chosen without ever looking at the
  held-out ratings. RP3beta is nearly as accurate (0.158, 45.4%) and more varied: it
  recommends 67% of the books (EASE 55%) and fewer bestsellers (58 training ratings per
  recommended book against 65). Stage 2's cosine table was more varied still (70%, 23), so
  more accuracy has cost some variety.
- **Text alone is weak, and a good partner.** A TF-IDF model of each book's title, author and
  Open Library subjects and description reaches NDCG@10 0.099, about v1's level, but
  recommends little-read books (25 training ratings per recommended book) and finds 12.7% of
  the held-out ratings among the less popular half of the books, where EASE finds 8.3%.
- **The blend recommends best:** EASE 0.6, RP3beta 0.3 and text 0.1, weights chosen on the
  validation set, reach NDCG@10 0.173 and a 48.4% hit rate, against 0.162 and 46.5% for EASE
  alone and 0.092 and 30.7% after stage 2. It recommends 60% of the books and finds 12.2% of
  the less popular half's held-out ratings; RP3beta alone finds the most of those (14.8%).

**The model for the demo:** the blend.

## Stage 4: the demo

[saricmilos.com/projects/book-recommender](https://www.saricmilos.com/projects/book-recommender)
runs with no server. `bookrec/export.py` works out, for each of the 2,405 books, the ten books
the blend recommends to a reader of that book (trained on all interactions, at the settings
the stage 3 report chose) and the ten v1's API returned for it, and writes them, with each
book's title, author, year, Open Library cover and a short description, into the site, which
loads them on demand. A visitor searches for a book and switches between v2 and v1.

Two rules shape what the demo shows, and neither changes the model or its scores:

- **One edition per list.** Editions stage 2 kept apart, such as "Harry Potter and the
  Sorcerer's Stone" with and without "(Book 1)", appear once.
- **Wild Animus is never recommended.** Its publisher gave copies away through BookCrossing by
  the thousand, so its 2,480 readers (twice the next book's, with a mean rating of 4.4 against
  a typical 7.7) say who received a copy, not who chose it, and it would top every list. The
  rule, more than 1,000 readers and a mean rating below 5, picks no other book.

## Code

- `bookrec/data.py`: the shared split, produced by v1's own functions (v1 is imported, never changed).
- `bookrec/books.py`: which ratings are about the same book.
- `bookrec/openlibrary.py`: book details from Open Library, fetched once and cached.
- `bookrec/interactions.py`: stage 3's training data, the rule above, and the validation set.
- `bookrec/topn.py`: popularity, item and user kNN, RP3beta and EASE.
- `bookrec/text.py`: the text model.
- `bookrec/stage3.py`: the extra-data comparison, the tuning, the blend, and the stage 3 report.
- `bookrec/export.py`: the demo's data, written into saricmilos.com.
- `bookrec/metrics.py`: rating errors and top-10 metrics.
- `bookrec/models.py`: baselines, v1 as shipped, v1 corrected, and the similarity tables.
- `bookrec/evaluate.py`: runs everything and writes the report.
- `tests/`: small worked examples for every metric, fix and book rule; the v1 re-run that
  guards the rest; and the reviewed counts of the book rules on the real data.

The library versions are pinned to v1's. `tests/test_v1_reproduction.py` must still pass
before they move.
