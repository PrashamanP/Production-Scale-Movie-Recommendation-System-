import os
import sys
import time
import random
from pathlib import Path
from typing import Dict, Set, Tuple, List

import pandas as pd
import numpy as np

from .infer_popgen import PopGenreRecommender

INTERACTIONS = os.path.join("data/interactions.csv")
ART_DIR = os.path.join("artifacts")

# How many users to evaluate for hit@20
EVAL_USERS = 3000
# How many users to time for latency/throughput
TIMING_USERS = 1000

def sample_holdouts(n_users: int = EVAL_USERS) -> List[Tuple[int, str]]:
    """
    Stream through interactions.csv and collect one (user, holdout_movie) pair per user.
    We take the LAST seen movie for that user as the holdout.
    """
    usecols = ["user_id", "movie_id"]
    dtypes = {"user_id": "int64", "movie_id": "string"}

    holdout: Dict[int, str] = {}
    seen: Set[int] = set()

    print("Sampling user holdouts from interactions.csv (streaming)...")
    for chunk in pd.read_csv(INTERACTIONS, usecols=usecols, dtype=dtypes, chunksize=2_000_000):
        # keep last occurrence per user in this chunk
        # (order within chunk follows file order; we just overwrite)
        for uid, mid in chunk[["user_id", "movie_id"]].itertuples(index=False):
            holdout[int(uid)] = str(mid)
            seen.add(int(uid))
        if len(seen) >= n_users * 3:  # gather a buffer to allow random sampling
            break

    users = list(seen)
    if len(users) > n_users:
        random.shuffle(users)
        users = users[:n_users]

    pairs = [(u, holdout[u]) for u in users if u in holdout]
    print(f"Prepared {len(pairs)} user holdouts.")
    return pairs

def evaluate_hit_at_20(reco: PopGenreRecommender, pairs: List[Tuple[int, str]]) -> float:
    """
    For each (user, holdout_movie), check if holdout appears in top-20.
    We do not filter 'seen' here (fast proxy, sufficient for M1).
    """
    hits = 0
    for uid, target_mid in pairs:
        recs = reco.recommend(uid, k=20)
        if target_mid in recs:
            hits += 1
    return hits / max(1, len(pairs))

def measure_latency_throughput(reco: PopGenreRecommender, users: List[int]) -> Tuple[float, float, float]:
    """
    Time recommend(user,k=20) for a sample of users.
    Return (avg_ms, p95_ms, throughput_qps).
    """
    if len(users) > TIMING_USERS:
        users = random.sample(users, TIMING_USERS)

    lat_ms = []
    start_all = time.perf_counter()
    for uid in users:
        t0 = time.perf_counter()
        _ = reco.recommend(uid, k=20)
        lat_ms.append((time.perf_counter() - t0) * 1000.0)
    total_s = time.perf_counter() - start_all

    avg_ms = float(np.mean(lat_ms)) if lat_ms else 0.0
    p95_ms = float(np.percentile(lat_ms, 95)) if lat_ms else 0.0
    qps = len(users) / total_s if total_s > 0 else 0.0
    return avg_ms, p95_ms, qps

def model_size_bytes() -> int:
    """
    Sum sizes of the baseline artifacts.
    """
    files = [
        os.path.join(ART_DIR, "popgen_popularity.parquet"),
        os.path.join(ART_DIR, "popgen_user_genre_prefs.parquet"),
        os.path.join(ART_DIR, "movie_catalog.json"),
    ]
    total = 0
    for f in files:
        if os.path.exists(f):
            total += os.path.getsize(f)
    return total

def main():
    print("Loading recommender artifacts...")
    reco = PopGenreRecommender(alpha=0.3, top_k_pool=2000)

    # 1) Accuracy proxy: Hit@20
    pairs = sample_holdouts(EVAL_USERS)
    hit20 = evaluate_hit_at_20(reco, pairs)
    print(f"Hit@20 (proxy accuracy) over {len(pairs)} users: {hit20:.4f}")

    # 2) Inference cost (latency/throughput)
    user_ids = [u for (u, _) in pairs]
    avg_ms, p95_ms, qps = measure_latency_throughput(reco, user_ids)
    print(f"Inference latency: avg={avg_ms:.2f} ms, p95={p95_ms:.2f} ms; throughput≈{qps:.2f} q/s")

    # 3) Model size
    sz = model_size_bytes()
    print(f"Model size (artifacts total): {sz / (1024*1024):.2f} MiB")

    # 4) Training cost
   
    print("Training cost: measure with `/usr/bin/time -v python3 scripts/train_baseline_popgenre.py`")

if __name__ == "__main__":
    main()
