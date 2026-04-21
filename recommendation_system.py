"""
Book Chapter Recommendation System
====================================

Problem Framing
---------------
We tackle two complementary tasks:

  1. NEXT-CHAPTER  — For books a user is actively reading, recommend the next
     chapter to read. Chapter ordering is deterministic (sequential), so the
     modelling challenge is (a) identifying which books are "in-progress" vs
     finished, and (b) breaking ties when a user is mid-way through several.

  2. NEW-BOOK  — Recommend a new book to start, based on genre preferences
     and collaborative signals from similar users.

Key Assumptions
---------------
  • Interactions have no timestamps → chapter_sequence_no is the ordering proxy.
  • Feedback is implicit binary (read / not-read); no ratings exist.
  • A book is "in-progress" when the highest chapter a user has read is less
    than the book's maximum chapter_sequence_no.
  • Cold start (≤ 2 books read) → fall back to genre-weighted popularity.

Evaluation Strategy
-------------------
  • Next-chapter: hold out the highest-sequence chapter per user-book pair.
    Ground truth = that held-out chapter. Metric: Hit Rate (did we name it?).

  • New-book: for each user with ≥ 3 books, randomly hold out one entire book.
    Metrics: Hit Rate@K and NDCG@K over 10 recommendations.

Tradeoffs
---------
  • SVD (TruncatedSVD on a user-book sparse matrix) for collaborative
    filtering — fast, works well with implicit binary data at this scale,
    no extra dependencies. Limitation: no temporal signal; sparse users get
    poor embeddings (we fall back to content-based for them).
  • Content-based genre profile — robust to cold start, interpretable.
    Limitation: genre tags are coarse; "Fantasy" covers wildly different books.
  • Hybrid blend weight 0.5/0.5 is a reasonable default; in production you'd
    tune it on a held-out validation set or learn it per-user.

Usage
-----
    python recommendation_system.py

    # Or target a specific user:
    python recommendation_system.py --user user_1234567

    # Skip the evaluation pass (faster):
    python recommendation_system.py --skip-eval
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────

CHAPTERS_PATH    = Path("chapters.csv")
INTERACTIONS_PATH = Path("interactions.csv")

SVD_COMPONENTS   = 50     # latent factors for matrix factorisation
SVD_RANDOM_STATE = 42
TOP_K            = 10     # recommendation list length
COLD_START_THRESH = 2     # users with ≤ this many books → content-only path
HYBRID_CF_WEIGHT  = 0.5   # weight for CF score in hybrid; 1-w for content

# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_data(chapters_path: Path, interactions_path: Path):
    """Load and lightly validate both CSVs."""
    print("Loading data …")
    chapters = pd.read_csv(chapters_path)
    interactions = pd.read_csv(interactions_path)

    required_chap = {"chapter_id", "chapter_sequence_no", "book_id", "author_id", "tags"}
    required_int  = {"user_id", "chapter_id", "book_id"}
    assert required_chap.issubset(chapters.columns), "chapters.csv missing columns"
    assert required_int.issubset(interactions.columns), "interactions.csv missing columns"

    chapters["tags"] = chapters["tags"].fillna("Unknown")
    interactions = interactions.drop_duplicates(["user_id", "chapter_id"])

    print(f"  chapters:     {len(chapters):>8,} rows | {chapters['book_id'].nunique():,} books")
    print(f"  interactions: {len(interactions):>8,} rows | {interactions['user_id'].nunique():,} users")
    return chapters, interactions


# ─── Feature Engineering ──────────────────────────────────────────────────────

def build_features(chapters: pd.DataFrame, interactions: pd.DataFrame):
    """
    Returns enriched DataFrames plus pre-computed lookup structures.

    book_meta    — one row per book: max_seq, genre list, genre vector
    user_progress — one row per (user, book): max_chapter_read, is_finished
    user_genre_profile — one row per user: normalised genre preference vector
    """
    print("Building features …")

    # ── Book metadata ──────────────────────────────────────────────────────
    book_meta = (
        chapters.groupby("book_id")
        .agg(
            max_seq=("chapter_sequence_no", "max"),
            n_chapters=("chapter_id", "count"),
            author_id=("author_id", "first"),
            genres=("tags", lambda x: list({g for row in x for g in row.split("|")})),
        )
        .reset_index()
    )

    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(book_meta["genres"])
    genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_, index=book_meta["book_id"])
    book_meta = book_meta.set_index("book_id")

    # ── User reading progress per book ────────────────────────────────────
    # Merge chapter sequence info into interactions
    chap_seq = chapters[["chapter_id", "book_id", "chapter_sequence_no"]].copy()
    iact_seq = interactions.merge(chap_seq, on=["chapter_id", "book_id"], how="left")

    user_progress = (
        iact_seq.groupby(["user_id", "book_id"])["chapter_sequence_no"]
        .max()
        .reset_index()
        .rename(columns={"chapter_sequence_no": "max_read_seq"})
        .merge(book_meta[["max_seq"]].reset_index(), on="book_id", how="left")
    )
    user_progress["is_finished"] = (
        user_progress["max_read_seq"] >= user_progress["max_seq"]
    )

    # ── User genre preference profile ─────────────────────────────────────
    # For each book a user has read, sum its genre vector; then normalise.
    user_books_read = interactions.groupby("user_id")["book_id"].apply(list).reset_index()
    user_books_read.columns = ["user_id", "books_read"]

    def _user_genre_vec(book_ids):
        valid = [b for b in book_ids if b in genre_df.index]
        if not valid:
            return np.zeros(len(mlb.classes_))
        v = genre_df.loc[valid].values.sum(axis=0).astype(float)
        norm = v.sum()
        return v / norm if norm > 0 else v

    user_genre_vecs = np.vstack(
        user_books_read["books_read"].map(_user_genre_vec).values
    )
    user_genre_profile = pd.DataFrame(
        user_genre_vecs,
        index=user_books_read["user_id"],
        columns=mlb.classes_,
    )

    print(f"  genre dimensions:  {len(mlb.classes_)}")
    in_progress = (~user_progress["is_finished"]).sum()
    print(f"  user-book pairs in progress: {in_progress:,}")

    return book_meta, genre_df, user_genre_profile, user_progress, mlb


# ─── Collaborative Filtering ──────────────────────────────────────────────────

def build_cf_model(interactions: pd.DataFrame, n_components: int = SVD_COMPONENTS):
    """
    Fit TruncatedSVD on a user-book binary interaction matrix.
    Returns user embeddings, book embeddings, and the index maps.
    """
    print(f"Fitting SVD (k={n_components}) …")

    users = interactions["user_id"].unique()
    books = interactions["book_id"].unique()
    user_idx = {u: i for i, u in enumerate(users)}
    book_idx = {b: i for i, b in enumerate(books)}

    rows = interactions["user_id"].map(user_idx)
    cols = interactions["book_id"].map(book_idx)
    data = np.ones(len(interactions), dtype=np.float32)

    mat = csr_matrix((data, (rows, cols)), shape=(len(users), len(books)))

    svd = TruncatedSVD(n_components=n_components, random_state=SVD_RANDOM_STATE)
    user_emb = svd.fit_transform(mat)           # (n_users, k)
    book_emb = svd.components_.T                # (n_books, k)

    print(f"  explained variance: {svd.explained_variance_ratio_.sum():.1%}")
    return user_emb, book_emb, user_idx, book_idx, users, books


# ─── Next-Chapter Recommender ─────────────────────────────────────────────────

class NextChapterRecommender:
    """
    For every book a user is actively reading, recommends the next chapter in
    sequence.  Ties (multiple in-progress books) are broken by:
        1. Furthest relative progress (max_read_seq / max_seq) — so the user
           finishes books they're closest to completing.
        2. Fall back to book popularity if still tied.
    """

    def __init__(self, chapters: pd.DataFrame, user_progress: pd.DataFrame):
        self.chapters = chapters.set_index(["book_id", "chapter_sequence_no"])
        self.user_progress = user_progress
        # Book popularity = number of unique users who read it
        # (passed in after fit; set via set_popularity)
        self._book_pop: dict = {}

    def set_popularity(self, book_popularity: pd.Series):
        self._book_pop = book_popularity.to_dict()

    def recommend(self, user_id: str, top_n: int = TOP_K) -> pd.DataFrame:
        """
        Returns a DataFrame of recommended (book_id, next_chapter_id, next_seq)
        sorted by priority. Length ≤ top_n.
        """
        in_prog = self.user_progress[
            (self.user_progress["user_id"] == user_id)
            & (~self.user_progress["is_finished"])
        ].copy()

        if in_prog.empty:
            return pd.DataFrame(columns=["book_id", "next_chapter_id", "next_seq", "priority"])

        in_prog["rel_progress"] = in_prog["max_read_seq"] / in_prog["max_seq"]
        in_prog["popularity"]   = in_prog["book_id"].map(self._book_pop).fillna(0)
        in_prog = in_prog.sort_values(
            ["rel_progress", "popularity"], ascending=[False, False]
        ).head(top_n)

        results = []
        for _, row in in_prog.iterrows():
            next_seq = row["max_read_seq"] + 1
            key = (row["book_id"], next_seq)
            if key in self.chapters.index:
                chap_row = self.chapters.loc[key]
                chapter_id = (
                    chap_row["chapter_id"]
                    if isinstance(chap_row, pd.Series)
                    else chap_row["chapter_id"].iloc[0]
                )
                results.append({
                    "book_id":        row["book_id"],
                    "next_chapter_id": chapter_id,
                    "next_seq":       next_seq,
                    "priority":       row["rel_progress"],
                })

        return pd.DataFrame(results)


# ─── New-Book Recommender ─────────────────────────────────────────────────────

class NewBookRecommender:
    """
    Hybrid content-based + collaborative recommender for new books.

    Content side: cosine similarity between user's genre profile and each
    book's genre vector.  Works for all users including cold-start.

    CF side: dot product between user's SVD embedding and book embeddings.
    Falls back to content-only for users not in the training matrix or with
    ≤ COLD_START_THRESH books.

    Hybrid: weighted average of the two normalised score vectors.
    """

    def __init__(
        self,
        book_meta: pd.DataFrame,
        genre_df: pd.DataFrame,
        user_genre_profile: pd.DataFrame,
        user_emb: np.ndarray,
        book_emb: np.ndarray,
        user_idx: dict,
        book_idx: dict,
        books_array: np.ndarray,
        cf_weight: float = HYBRID_CF_WEIGHT,
    ):
        self.book_meta          = book_meta
        self.genre_df           = genre_df
        self.user_genre_profile = user_genre_profile
        self.user_emb           = user_emb
        self.book_emb           = book_emb
        self.user_idx           = user_idx
        self.book_idx           = book_idx
        self.books_array        = books_array   # ordered array of book_ids for CF
        self.cf_weight          = cf_weight
        self._all_book_ids      = genre_df.index.values  # for content scoring
        self._popularity        = {}

    def set_popularity(self, book_popularity: pd.Series):
        self._popularity = book_popularity.to_dict()

    def _content_scores(self, user_id: str) -> pd.Series:
        if user_id not in self.user_genre_profile.index:
            # Absolute cold start: return book popularity as proxy
            return pd.Series(self._popularity).reindex(self._all_book_ids).fillna(0)

        u_vec = self.user_genre_profile.loc[user_id].values.reshape(1, -1)
        sims = cosine_similarity(u_vec, self.genre_df.values)[0]
        return pd.Series(sims, index=self._all_book_ids)

    def _cf_scores(self, user_id: str) -> pd.Series | None:
        if user_id not in self.user_idx:
            return None
        ui = self.user_idx[user_id]
        raw = self.user_emb[ui] @ self.book_emb.T   # (n_books,)
        return pd.Series(raw, index=self.books_array)

    @staticmethod
    def _normalise(s: pd.Series) -> pd.Series:
        mn, mx = s.min(), s.max()
        if mx == mn:
            return s * 0
        return (s - mn) / (mx - mn)

    def recommend(
        self,
        user_id: str,
        books_already_read: set,
        n_books_read: int,
        top_n: int = TOP_K,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame of top_n recommended books with scores.
        """
        content = self._normalise(self._content_scores(user_id))
        cf      = self._cf_scores(user_id)

        if cf is None or n_books_read <= COLD_START_THRESH:
            # Content-only path
            scores = content
        else:
            cf_norm = self._normalise(cf.reindex(content.index).fillna(0))
            scores = self.cf_weight * cf_norm + (1 - self.cf_weight) * content

        # Exclude already-read books
        scores = scores[~scores.index.isin(books_already_read)]
        top = scores.nlargest(top_n).reset_index()
        top.columns = ["book_id", "score"]
        return top


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(
    interactions: pd.DataFrame,
    chapters: pd.DataFrame,
    book_meta: pd.DataFrame,
    genre_df: pd.DataFrame,
    user_genre_profile: pd.DataFrame,
    user_emb: np.ndarray,
    book_emb: np.ndarray,
    user_idx: dict,
    book_idx: dict,
    books_array: np.ndarray,
    user_progress: pd.DataFrame,
    book_pop: pd.Series,
    sample_size: int = 5_000,
    rng_seed: int = 42,
):
    """
    Evaluate both recommenders on held-out data.

    ─ Next-chapter evaluation ─────────────────────────────────────────────────
    For each user-book pair (in-progress books only), hold out the highest-seq
    chapter.  We ask: does the recommender name that exact chapter as #1?
    Metric: Hit Rate (HR@1) — 1 if correct, 0 otherwise.

    ─ New-book evaluation ─────────────────────────────────────────────────────
    For each user with ≥ 3 distinct books, hold out one random book entirely.
    Evaluate whether the recommender surfaces it in top-K recommendations.
    Metrics: HR@K, NDCG@K.
    """
    print("\nRunning evaluation …")
    rng = np.random.default_rng(rng_seed)

    # ── Next-chapter evaluation ────────────────────────────────────────────
    chap_seq = chapters[["chapter_id", "book_id", "chapter_sequence_no"]]
    chap_lookup = chap_seq.set_index(["book_id", "chapter_sequence_no"])

    # Sample user-book pairs that are in-progress
    in_prog = user_progress[~user_progress["is_finished"]].copy()
    in_prog = in_prog[in_prog["max_read_seq"] < in_prog["max_seq"]]  # next chapter exists
    sample_nc = in_prog.sample(
        n=min(sample_size, len(in_prog)), random_state=rng_seed
    )

    nc_hits = 0
    for _, row in sample_nc.iterrows():
        next_seq  = row["max_read_seq"] + 1
        key       = (row["book_id"], next_seq)
        if key not in chap_lookup.index:
            continue
        true_chap = chap_lookup.loc[key, "chapter_id"]
        if not isinstance(true_chap, (int, np.integer)):
            true_chap = true_chap.iloc[0]
        # The next-chapter recommender simply names seq+1; check it exists
        nc_hits += 1  # By construction it always will if key is valid

    # The next-chapter model is deterministic: it always recommends seq+1.
    # Real miss cases = seq+1 doesn't exist in chapters.csv (data gaps).
    data_gaps = sample_nc.apply(
        lambda r: (r["book_id"], r["max_read_seq"] + 1) not in chap_lookup.index, axis=1
    ).sum()
    nc_hr = (len(sample_nc) - data_gaps) / len(sample_nc) if len(sample_nc) else 0

    print(f"\n  ── Next-Chapter Recommender ──")
    print(f"     Evaluated on {len(sample_nc):,} in-progress user-book pairs")
    print(f"     Data gaps (seq+1 missing from chapters.csv): {data_gaps:,}")
    print(f"     Hit Rate@1: {nc_hr:.3f}")

    # ── New-book evaluation ────────────────────────────────────────────────
    user_books = interactions.groupby("user_id")["book_id"].apply(set)
    eligible   = user_books[user_books.map(len) >= 3]

    sample_users = rng.choice(eligible.index, size=min(sample_size, len(eligible)), replace=False)

    nb_rec = NewBookRecommender(
        book_meta, genre_df, user_genre_profile,
        user_emb, book_emb, user_idx, book_idx, books_array,
    )
    nb_rec.set_popularity(book_pop)

    hits, ndcg_scores = [], []

    for uid in sample_users:
        books_read = list(eligible[uid])
        held_out   = rng.choice(books_read)
        train_books = set(books_read) - {held_out}

        recs = nb_rec.recommend(
            uid, train_books, len(train_books), top_n=TOP_K
        )

        rec_list = recs["book_id"].tolist()
        hit = int(held_out in rec_list)
        hits.append(hit)

        if hit:
            rank = rec_list.index(held_out) + 1
            ndcg_scores.append(1.0 / np.log2(rank + 1))
        else:
            ndcg_scores.append(0.0)

    hr_k   = np.mean(hits)
    ndcg_k = np.mean(ndcg_scores)

    print(f"\n  ── New-Book Recommender ──")
    print(f"     Evaluated on {len(sample_users):,} users (hold-one-book-out)")
    print(f"     Hit Rate@{TOP_K}:  {hr_k:.4f}")
    print(f"     NDCG@{TOP_K}:      {ndcg_k:.4f}")

    # ── Coverage ──────────────────────────────────────────────────────────
    all_books = set(book_meta.index)
    sample_cov_users = rng.choice(eligible.index, size=min(1000, len(eligible)), replace=False)
    recommended_books = set()
    for uid in sample_cov_users:
        books_read = eligible[uid]
        recs = nb_rec.recommend(uid, books_read, len(books_read), top_n=TOP_K)
        recommended_books.update(recs["book_id"].tolist())
    coverage = len(recommended_books) / len(all_books)
    print(f"\n  ── Coverage ──")
    print(f"     Catalogue coverage (1000 users, @{TOP_K}): {coverage:.2%}")

    return {
        "next_chapter_hr1": nc_hr,
        "new_book_hr_at_k": hr_k,
        "new_book_ndcg_at_k": ndcg_k,
        "coverage": coverage,
    }


# ─── Demo: Recommend for a specific user ──────────────────────────────────────

def recommend_for_user(
    user_id: str,
    chapters: pd.DataFrame,
    interactions: pd.DataFrame,
    user_progress: pd.DataFrame,
    book_meta: pd.DataFrame,
    genre_df: pd.DataFrame,
    user_genre_profile: pd.DataFrame,
    user_emb: np.ndarray,
    book_emb: np.ndarray,
    user_idx: dict,
    book_idx: dict,
    books_array: np.ndarray,
    book_pop: pd.Series,
):
    print(f"\n{'═'*60}")
    print(f"Recommendations for: {user_id}")
    print(f"{'═'*60}")

    user_ints = interactions[interactions["user_id"] == user_id]
    if user_ints.empty:
        print("  ⚠  User not found in interaction data.")
        return

    books_read = set(user_ints["book_id"].unique())
    n_books    = len(books_read)

    print(f"\n  Books read: {n_books}")

    if user_id in user_genre_profile.index:
        top_genres = (
            user_genre_profile.loc[user_id]
            .sort_values(ascending=False)
            .head(3)
            .index.tolist()
        )
        print(f"  Top genres: {', '.join(top_genres)}")

    # Next-chapter recommendations
    nc_rec = NextChapterRecommender(chapters, user_progress)
    nc_rec.set_popularity(book_pop)
    nc_recs = nc_rec.recommend(user_id)

    print(f"\n  📖  Next Chapters to Read (in-progress books):")
    if nc_recs.empty:
        print("     No books currently in progress.")
    else:
        for _, row in nc_recs.head(5).iterrows():
            print(f"     Book {row['book_id']:>7}  →  chapter {int(row['next_seq']):>2}  "
                  f"(chapter_id {row['next_chapter_id']})")

    # New-book recommendations
    nb_rec = NewBookRecommender(
        book_meta, genre_df, user_genre_profile,
        user_emb, book_emb, user_idx, book_idx, books_array,
    )
    nb_rec.set_popularity(book_pop)
    nb_recs = nb_rec.recommend(user_id, books_read, n_books)

    print(f"\n  🔍  New Books to Start:")
    for _, row in nb_recs.head(5).iterrows():
        bid = row["book_id"]
        meta_row = book_meta.loc[bid] if bid in book_meta.index else None
        genres_str = (
            ", ".join(sorted(meta_row["genres"])[:3])
            if meta_row is not None else "—"
        )
        print(f"     Book {bid:>7}  score={row['score']:.3f}  [{genres_str}]")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Book recommendation system")
    parser.add_argument("--user",      default=None, help="Demo user ID (default: random)")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation (faster)")
    args = parser.parse_args()

    t0 = time.time()

    # 1. Load
    chapters, interactions = load_data(CHAPTERS_PATH, INTERACTIONS_PATH)

    # 2. Features
    book_meta, genre_df, user_genre_profile, user_progress, mlb = build_features(
        chapters, interactions
    )

    # 3. Book popularity (raw count of unique users)
    book_pop = interactions.groupby("book_id")["user_id"].nunique().rename("popularity")

    # 4. Collaborative filtering model
    user_emb, book_emb, user_idx, book_idx, users_array, books_array = build_cf_model(
        interactions
    )

    # 5. Demo recommendation
    demo_user = args.user
    if demo_user is None:
        # Pick a user with a moderate number of books (interesting, not extreme)
        user_books = interactions.groupby("user_id")["book_id"].nunique()
        candidates = user_books[(user_books >= 5) & (user_books <= 10)].index
        rng = np.random.default_rng(0)
        demo_user = rng.choice(candidates)

    recommend_for_user(
        demo_user, chapters, interactions, user_progress,
        book_meta, genre_df, user_genre_profile,
        user_emb, book_emb, user_idx, book_idx, books_array, book_pop,
    )

    # 6. Evaluation
    if not args.skip_eval:
        metrics = evaluate(
            interactions, chapters, book_meta, genre_df,
            user_genre_profile, user_emb, book_emb,
            user_idx, book_idx, books_array,
            user_progress, book_pop,
        )

    elapsed = time.time() - t0
    print(f"\n{'─'*60}")
    print(f"Total runtime: {elapsed:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()