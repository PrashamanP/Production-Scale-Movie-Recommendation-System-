import os, json
import pandas as pd
from collections import defaultdict, Counter

DATA = os.path.join("data/interactions.csv")
CAT  = os.path.join("artifacts/movie_catalog.json")
OUT_POP = os.path.join("artifacts/popgen_popularity.parquet")
OUT_UP  = os.path.join("artifacts/popgen_user_genre_prefs.parquet")

WATCH_WEIGHT  = 1.0
RATING_WEIGHT = 1.0
CHUNKSIZE = 2_000_000

def load_catalog():
    with open(CAT, "r") as f:
        raw = json.load(f)
    # normalize to: movie_id -> {"genres": [str, ...], ...}
    norm = {}
    for mid, info in raw.items():
        gs = info.get("genres", [])
        norm_genres = []
        for g in gs:
            if isinstance(g, str):
                norm_genres.append(g)
            elif isinstance(g, dict):
                # common TMDB/metadata shapes
                name = g.get("name") or g.get("genre") or g.get("Genre") or g.get("label")
                if name:
                    norm_genres.append(str(name))
        info2 = dict(info)
        info2["genres"] = norm_genres
        norm[mid] = info2
    return norm

def main():
    os.makedirs(os.path.dirname(OUT_POP), exist_ok=True)
    catalog = load_catalog()

    movie_score = defaultdict(float)   # movie_id -> score
    user_genres = defaultdict(Counter) # user_id -> Counter(genre -> weight)

    usecols = ["user_id", "movie_id", "rating"]
    dtypes = {"user_id":"int64", "movie_id":"string", "rating":"int8"}

    print("Aggregating in chunks...")
    for i, chunk in enumerate(pd.read_csv(DATA, usecols=usecols, dtype=dtypes, chunksize=CHUNKSIZE)):
        chunk = chunk.dropna(subset=["user_id","movie_id","rating"])
        for uid, mid, r in chunk.itertuples(index=False):
            r = int(r)
            # base weight: watch = 1, rating adds stars
            w = WATCH_WEIGHT if r == 1 else (WATCH_WEIGHT + RATING_WEIGHT * r)

            movie_score[mid] += w

            info = catalog.get(mid)
            if info:
                for gname in info.get("genres", []):
                    if gname:  # string after normalization
                        user_genres[int(uid)][gname] += w

        if (i+1) % 5 == 0:
            print(f"Processed ~{(i+1)*CHUNKSIZE:,} rows...")

    # popularity table
    pop_df = (
        pd.Series(movie_score, name="pop")
        .rename_axis("movie_id")
        .sort_values(ascending=False)
        .to_frame()
    )
    pop_df.to_parquet(OUT_POP)
    print(f"Wrote popularity to {OUT_POP} ({len(pop_df)} movies)")

    # top-10 genres per user
    records = []
    for uid, cnt in user_genres.items():
        top_items = cnt.most_common(10)
        records.append({"user_id": int(uid), "top_genres": [g for g, _ in top_items]})
    up_df = pd.DataFrame(records)
    up_df.to_parquet(OUT_UP, index=False)
    print(f"Wrote user genre prefs to {OUT_UP} ({len(up_df)} users)")

if __name__ == "__main__":
    main()
