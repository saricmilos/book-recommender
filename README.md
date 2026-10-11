# Book Recommender

Book recommendations learned from the Book-Crossing ratings: 1,149,780 ratings by 278,858 users
on 271,360 books (Kaggle, *Book Recommendation Dataset*).

The repository holds two versions of the project. **v1** is kept exactly as I built it by hand.
**v2** is being built beside it with Claude Code, on the same data, and will be measured against
v1 on the same test split.

| | v1 | v2 |
|---|---|---|
| Built | by hand, November 2025 to early 2026 | with Claude Code, from October 2026: I set the goals, review every change and check the evaluation |
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
2026) were made with Claude Code while tidying the repository: committing local work, limiting
CORS and updating paths. The notebooks read the data from an absolute path on the machine they
were written on; to run them, point `dataset_folder` at `datasets/`.

## v2

Built in stages, each measured on the same test split as v1. [`v2/README.md`](v2/README.md)
says how to run it and what each stage found.

1. Evaluation, done ([report](v2/reports/phase1-evaluation.md)): v1 reproduces exactly from its
   own code, and v1 is scored again next to simple baselines. v1's user-based model, as
   shipped, recommends best so far: 27.7% of users find one of their held-out books in its
   top 10, against 14.9% for the most popular books.
2. Data quality, books done ([report](v2/reports/phase2-books.md)): v1 identified a book by its
   exact title, so editions counted as different books and different books with one title as
   one. Identified properly, v1's 2,508 books are 2,405, and the cosine similarity table v1
   measured now recommends best (NDCG@10 0.070 → 0.092). Next: fetch book details once.
3. Models: use the implicit interactions (62.3% of the ratings are 0, meaning a book was seen
   but not rated), add content embeddings for books with few ratings, and combine both.
4. A demo on [saricmilos.com](https://www.saricmilos.com) that needs no server.

## Data

[`datasets/`](datasets/) holds `Books.csv`, `Ratings.csv` and `Users.csv`. Both versions read
them and neither changes them.

## License

[MIT](LICENSE)
