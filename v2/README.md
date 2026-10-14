# v2

The second version of the book recommender, built from October 2026. Every number below comes
from code in this folder, and the tests check that code, including a re-run of v1.

## Run it

```bash
cd v2
python -m venv .venv
.venv/Scripts/activate          # Windows; on macOS or Linux: source .venv/bin/activate
pip install -e ".[test]"
pytest                          # 132 tests, under 3 minutes, v1's re-run included
python -m bookrec.evaluate      # about 25 minutes: writes the reports in reports/ and data/books.csv
python -m bookrec.export        # about 2 minutes: writes the demo's data into the site
```

Book details were fetched once from [Open Library](https://openlibrary.org) with
`python -m bookrec.openlibrary` (about 45 minutes at 1 request a second) and are committed in
`data/book_details.jsonl`, with the Open Library work of every rated edition in
`data/isbn_works.csv`, so nothing above needs the network. The same run keeps each book's
cover picture in the gitignored `data/openlibrary-cache/covers/`, from which the export makes
the small copies the site serves; without that folder the export keeps the site's copies.

## Stages

Every stage is measured on the same split as v1, so the numbers stay comparable.

| Stage | What | Status |
|---|---|---|
| 1. Evaluation | Re-run v1, add baselines, score v1 again under the same rules | Done: [report](reports/phase1-evaluation.md) |
| 2. Data quality | Identify books properly: merge editions, split different books that share a title, join what Open Library files as one work; fetch book details once | Done: [report](reports/phase2-books.md), [books](data/books.csv), [details](data/book_details.jsonl) |
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

- **v1's 2,508 books are 2,366.** v1 identified a book by its exact title. 196 books had been
  split across several titles, mostly editions ("Bel Canto" and "Bel Canto: A Novel"), and 75
  titles mixed different authors' books ("Mortal Fear" by Greg Iles and by Robin Cook,
  "Lightning" by Danielle Steel and by Dean Koontz). Titles keep their real subtitles and
  volume numbers apart, so "Dune" and "Dune: House Atreides" stay two books; 29 books are then
  joined because Open Library files all their editions as one work: *The Hobbit*'s three titles,
  *Harry Potter and the Sorcerer's Stone* with and without "(Book 1)", the six serial parts of
  *The Green Mile*. Two *Dilbert* collections share a main title but not a work, and stay apart.
  The rules lean towards changing nothing and are listed in the [report](reports/phase2-books.md).
- **Duplicate editions had flattered the bestsellers.** With every edition of a book joined,
  282 held-out ratings are of a book the reader already had in training, against 253 before,
  and are dropped: no recommender should get credit for suggesting a book again. Every model's
  hit rate rises a little with the cleaner books except one: the dot-product table v1's API
  serves, which recommends the same bestsellers to everyone, falls from 30.7% to 29.6%.
- **The similarity tables gain most.** On the same users and the same held-out ratings, the
  cosine table v1 measured goes from a 21.4% to a 28.8% hit rate, and its NDCG@10 from 0.070
  to 0.093, the best of any model, while recommending from 70% of the books. The dot-product
  table v1's API serves has the highest hit rate (29.6%) but still keeps to bestsellers: 19% of
  the books, 117 training ratings per recommended book.
- **Rating error improves too** (about 1.26 MAE for both corrected v1 models), but the bias
  baseline is still better at 1.17.
- **Every book is on Open Library**, found through one of its editions' ISBNs: 99.9% have a
  cover, 87% a description and 99% subjects, for stage 3's text model and stage 4's demo.
  Open Library has errors of its own (it credits Robin Cook's novels to Lisa Jackson and files
  some editions under the wrong work), so v2 keeps the authors from the rated editions,
  prefers a work whose title and author match the book's, and lists the 118 books whose work
  still has another title: mostly the original-language title, a handful of wrong works.

**The bar for v2's models:** NDCG@10 above 0.093 and a hit rate above 29.6%, with the
cosine table's variety rather than the dot-product table's bestsellers.

## What stage 3 found

- **One rule gives six times the training data.** Training may now use any interaction in
  Book-Crossing with the 2,366 books, by any user, rated or not, except one that covers a
  held-out user and book through any edition. That takes training from 45,047 interactions to
  280,205 (a median of 81 per book instead of 12), and the held-out ratings stay exactly as
  they were. A book a user already has, rated or not, is never recommended to them.
- **More data helps the similarity models; raw counts get worse.** With fixed settings, EASE
  goes from NDCG@10 0.108 to 0.150, RP3beta from 0.096 to 0.152 and the cosine table v1
  measured from 0.095 to 0.108. Both sources help on their own: other readers' interactions,
  which can't carry anything about the held-out ratings, and the benchmark users' own unrated
  books. The raw dot-product table v1's API serves gets worse (0.086 to 0.046) and the most
  popular books stay flat: unrated books lean even more towards bestsellers.
- **Tuned on a validation set, EASE recommends best of the single models:** NDCG@10 0.161 and
  a 46.4% hit rate, against stage 2's 0.093 and 29.6%, with settings chosen without ever looking
  at the held-out ratings. RP3beta is nearly as accurate (0.160, 46.1%) and recommends a few
  more of the books (62%, EASE 59%). Stage 2's cosine table was more varied still (70%, with 25
  training interactions per recommended book against EASE's 64), so more accuracy has cost
  some variety.
- **Text alone is weak, and a good partner.** A TF-IDF model of each book's title, author and
  Open Library subjects and description reaches NDCG@10 0.107, a little above v1's best, but
  recommends little-read books (25 training interactions per recommended book) and finds 12.7%
  of the held-out ratings among the less popular half of the books, where EASE finds 8.8%.
- **The blend recommends best:** EASE 0.2, RP3beta 0.7 and text 0.1, weights chosen on the
  validation set, reach NDCG@10 0.172 and a 48.2% hit rate, against 0.161 and 46.4% for EASE
  alone and 0.093 and 29.6% after stage 2. It recommends 63% of the books and finds 12.8% of
  the less popular half's held-out ratings, more than any model alone.

**The model for the demo:** the blend.

## Stage 4: the demo

[saricmilos.com/projects/book-recommender](https://www.saricmilos.com/projects/book-recommender)
runs with no server. `bookrec/export.py` works out, for each of the 2,366 books, the ten books
the blend recommends to a reader of that book (trained on all interactions, at the settings
the stage 3 report chose) and the ten v1's API returned for it, and writes them, with each
book's title, author, year, Open Library cover (a 180-pixel WebP copy the site serves
itself, since archive.org, which serves Open Library's, is slow) and a short description, into
the site. The page
opens on both lists for a reader of *The Cat in the Hat*, where v1's ten hold Stephen King's
*It* and *Misery* and v2's open with five Dr. Seuss books, and a visitor can search for any
other book; the site fails to build if v1's ten for it ever stop holding those two. The export
also writes the page's results, with a 95% interval for each hit rate, how much each version's
lists repeat themselves (v1 puts *Harry Potter and the Chamber of Secrets* in 30% of its lists,
v2 no book in more than 1.4%), and what fixing v1's averaging bug did to its user-based model.

Three rules shape what the demo shows, and none changes the model or its scores:

- **One edition per list.** Should two books stage 2 kept apart still be editions of one (the
  same title and author but for a volume number in brackets or an abbreviation spelled out),
  a list shows only one of them. Since stage 2 joins the editions Open Library files as one
  work, this is a safety net.
- **Wild Animus is never recommended.** Its publisher gave copies away through BookCrossing by
  the thousand, so its 2,480 readers (twice the next book's, with a mean rating of 4.4 against
  a typical 7.7) say who received a copy, not who chose it, and it would top every list. The
  rule, more than 1,000 readers and a mean rating below 5, picks no other book.
- **Descriptions in English, as plain text.** Open Library's descriptions come in seven
  languages and in Markdown, some with a shop's leftovers ("This deluxe trade paperback
  edition…", "All Random House Large Print editions…") or a passage given twice. The export keeps the English
  paragraphs as plain text, without the leftovers, and leaves out six it lists as bad, found by
  reading those of the 200 most-read books (a reader's school report with misspellings, the
  Latin edition's text for *The Cat in the Hat*).

The page also shows what the data showed: four findings, each with what it changed in v2 (most
interactions are a 0, most editions have one reader, Wild Animus, a title isn't a book).
`bookrec/findings.py` works out their numbers from `datasets/` and the reports, and the export
writes them into the site as `findings.json`, so no number on the page is typed by hand. Its
examples of stage 2 ("Bel Canto", "Lightning", "Angels &amp; Demons") are checked against
`data/books.csv` on every export.

## Code

- `bookrec/data.py`: the shared split, produced by v1's own functions (v1 is imported, never changed).
- `bookrec/books.py`: which ratings are about the same book.
- `bookrec/openlibrary.py`: book details from Open Library, and each rated edition's work, fetched once and cached.
- `bookrec/interactions.py`: stage 3's training data, the rule above, and the validation set.
- `bookrec/topn.py`: popularity, item and user kNN, RP3beta and EASE.
- `bookrec/text.py`: the text model.
- `bookrec/stage3.py`: the extra-data comparison, the tuning, the blend, and the stage 3 report.
- `bookrec/export.py`: the demo's data, written into saricmilos.com.
- `bookrec/findings.py`: the numbers behind the page's "What the data showed".
- `bookrec/metrics.py`: rating errors and top-10 metrics.
- `bookrec/models.py`: baselines, v1 as shipped, v1 corrected, and the similarity tables.
- `bookrec/evaluate.py`: runs everything and writes the report.
- `tests/`: small worked examples for every metric, fix and book rule; the v1 re-run that
  guards the rest; and the reviewed counts of the book rules on the real data.

The library versions are pinned to v1's. `tests/test_v1_reproduction.py` must still pass
before they move.
