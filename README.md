# Book Recommender

Book recommendations learned from the Book-Crossing ratings: 1,149,780 ratings by 278,858 users
on 271,360 books (Kaggle, *Book Recommendation Dataset*).

The repository holds two versions of the project. **v1** is kept exactly as I built it by hand.
**v2** is being built beside it, on the same data, and will be measured against
v1 on the same test split.

| | v1 | v2 |
|---|---|---|
| Built | by hand, November 2025 to early 2026 | from October 2026 |
| Code | [`v1/`](v1/), and tag [`v1.0`](https://github.com/saricmilos/book-recommender/tree/v1.0) for the original layout | [`v2/`](v2/) |
| Status | frozen | in progress |

## v1

- [`v1/notebooks/`](v1/notebooks/): exploratory data analysis, collaborative filtering (user- and
  item-based) and content-based filtering.
- [`v1/src/`](v1/src/): data loading, cleaning and the collaborative filtering models.
- [`v1/app/`](v1/app/): a FastAPI service that finds books similar to a given title, with covers
  and descriptions from Open Library. It runs on Fly.io at
  <https://what-else-should-i-read.fly.dev>; the machine sleeps when idle, so the first request
  can take up to 20 seconds.
- The report: [*Book Recommendation System: Collaborative & Content-Based Filtering
  Approaches*](v1/Book%20Recommendation%20System%20Collaborative%20%26%20Content-Based%20Filtering%20Approaches.pdf)
  (30 pages, LaTeX).

v1's files are as they were; they only moved into `v1/`. The last three v1 commits (October
2026) only tidied the repository: committing local work, limiting
CORS and updating paths. The notebooks read the data from an absolute path on the machine they
were written on; to run them, point `dataset_folder` at `datasets/`.

## v2

Built in stages, each measured on the same test split as v1. [`v2/README.md`](v2/README.md)
says how to run it and what each stage found.

1. Evaluation, done ([report](v2/reports/phase1-evaluation.md)): v1 reproduces exactly from its
   own code, and v1 is scored again next to simple baselines. v1's user-based model, as
   shipped, recommends best so far: 27.7% of users find one of their held-out books in its
   top 10, against 14.9% for the most popular books.
2. Data quality, done ([report](v2/reports/phase2-books.md)): v1 identified a book by its
   exact title, so editions counted as different books and different books with one title as
   one. Identified properly, with Open Library saying which editions are one work, v1's 2,508
   books are 2,366, and the cosine similarity table v1 measured now recommends best (NDCG@10
   0.070 → 0.093). Covers, descriptions and subjects come from
   [Open Library](https://openlibrary.org), fetched once and kept in the repository.
3. Models, done ([report](v2/reports/phase3-models.md)): training now uses every interaction
   in Book-Crossing with the 2,366 books, rated or not (62.3% of Book-Crossing's ratings are 0,
   a book seen but not rated), except any that covers a held-out user and book; that is six
   times the data. A blend of EASE, RP3beta and a text model of the book descriptions, tuned
   on a validation set, reaches NDCG@10 0.172 and a 48.2% hit rate, against 0.093 and 29.6%
   after stage 2 and 0.089 and 28.7% for v1's user-based model as shipped, on the same
   held-out ratings.
4. Demo, done: [saricmilos.com/projects/book-recommender](https://www.saricmilos.com/projects/book-recommender),
   with no server. It opens on what v1 recommended to readers of *The Cat in the Hat* (Stephen
   King's *It* and *Misery* among them) beside what v2 does; pick any book to compare the two.

## Data

[`datasets/`](datasets/) holds `Books.csv`, `Ratings.csv` and `Users.csv`. Both versions read
them and neither changes them.

## License

[MIT](LICENSE)
