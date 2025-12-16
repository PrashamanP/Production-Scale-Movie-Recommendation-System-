import os
from typing import List, Iterable, Dict, Set
import pandas as pd
import numpy as np

ART_DIR = os.path.join("artifacts")
POPULARITY = os.path.join(ART_DIR, "popgen_popularity.parquet")
USER_GENRES = os.path.join(ART_DIR, "popgen_user_genre_prefs.parquet")
CATALOG = os.path.join(ART_DIR, "movie_catalog.json")

class PopGenreRecommender:
    """
    Popularity baseline + light personalization:
      score(movie) = global_pop[movie] * (1 + alpha * user_genre_boost(movie))
    where user_genre_boost(movie) is derived from the user's top genres.
    """
    def __init__(self, alpha: float = 0.3, top_k_pool: int = 2000):
        self.alpha = float(alpha)
        self.top_k_pool = int(top_k_pool)

        # ---- Load popularity and normalize schema -> ('movie_id','score') ----
        pop_df = pd.read_parquet(POPULARITY)

        # If score column missing, rename sensible alternatives
        if "score" not in pop_df.columns:
            if 0 in pop_df.columns:
                pop_df = pop_df.rename(columns={0: "score"})
            elif "value" in pop_df.columns:
                pop_df = pop_df.rename(columns={"value": "score"})
            elif len(pop_df.columns) == 1:
                only = pop_df.columns[0]
                pop_df = pop_df.rename(columns={only: "score"})

        # If movie_id not a column, move index to column
        if "movie_id" not in pop_df.columns:
            if pop_df.index.name:
                pop_df = pop_df.reset_index().rename(columns={pop_df.index.name: "movie_id"})
            else:
                pop_df = pop_df.reset_index().rename(columns={"index": "movie_id"})

        pop_df["movie_id"] = pop_df["movie_id"].astype(str)
        pop_df["score"] = pd.to_numeric(pop_df["score"], errors="coerce").fillna(0.0)

        # Keep a pool of top movies for fast inference
        pop_df = pop_df.sort_values("score", ascending=False)
        self.pop_list = pop_df[["movie_id", "score"]].head(self.top_k_pool).to_numpy()
        # Also keep as dict for quick lookup
        self.pop = dict(zip(pop_df["movie_id"], pop_df["score"]))

        # ---- Load user genre preferences (wide format) ----
        ug = pd.read_parquet(USER_GENRES)
        # Force numeric types for genre weights; keep user_id as int
        if "user_id" not in ug.columns:
            raise RuntimeError(f"'user_id' column missing in {USER_GENRES}")
        ug["user_id"] = pd.to_numeric(ug["user_id"], errors="coerce").astype("Int64")
        # Coerce all others to numeric (genre columns)
        for col in ug.columns:
            if col != "user_id":
                ug[col] = pd.to_numeric(ug[col], errors="coerce").fillna(0.0)

        # Convert to dict: uid -> {genre: weight}
        self.user_genres: Dict[int, Dict[str, float]] = {}
        genre_cols = [c for c in ug.columns if c != "user_id"]
        for row in ug.itertuples(index=False):
            uid = int(getattr(row, "user_id"))
            prefs = {}
            for g in genre_cols:
                w = float(getattr(row, g))
                if w > 0.0:
                    prefs[g] = w
            if prefs:
                self.user_genres[uid] = prefs

        # ---- Load catalog to map movie -> genres ----
        # { movie_id: {"genres": ["Action","Comedy",...], ...}, ... }
        self.catalog = pd.read_json(CATALOG, typ="series").to_dict()

    def _user_boost(self, user_id: int, movie_id: str) -> float:
        """Compute normalized boost for a user on a movie based on shared genres."""
        info = self.catalog.get(movie_id)
        if not info:
            return 0.0
        movie_genres = info.get("genres", [])
        if not movie_genres:
            return 0.0

        prefs = self.user_genres.get(int(user_id))
        if not prefs:
            return 0.0

        # Sum the user's weights for the movie's genres
        score = 0.0
        for g in movie_genres:
            # allow dict-like entries in catalog where genre could be dict with "name"
            if isinstance(g, dict) and "name" in g:
                g = g["name"]
            if not isinstance(g, str):
                continue
            score += float(prefs.get(g, 0.0))
        if score <= 0.0:
            return 0.0

        # Normalize by total weight for stability
        total = float(sum(prefs.values())) or 1.0
        return score / total

    def recommend(
        self,
        user_id: int,
        k: int = 20,
        exclude: Iterable[str] | None = None
    ) -> List[str]:
        """
        Rank top-K from a global-popularity pool with a user-genre multiplicative boost.
        """
        banned: Set[str] = set(map(str, exclude)) if exclude else set()

        # Compute personalized score on the top pool only (fast)
        scores = []
        for movie_id, base in self.pop_list:
            if movie_id in banned:
                continue
            boost = self._user_boost(user_id, movie_id)
            final = float(base) * (1.0 + self.alpha * boost)
            scores.append((final, movie_id))

        # Arg-sort and take top-k
        scores.sort(reverse=True, key=lambda x: x[0])
        return [m for _, m in scores[:k]]
