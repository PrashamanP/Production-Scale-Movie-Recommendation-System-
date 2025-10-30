import os
import time
import random
from pathlib import Path
from typing import Dict, Set, Tuple, List

import pandas as pd
import numpy as np

# Import the refactored model class
from .model import ALSRecommender

# Constants for evaluation
EVAL_USERS = 3000
TIMING_USERS = 1000

def sample_holdouts(interactions_path: str, n_users: int = EVAL_USERS) -> List[Tuple[int, str]]:
    """Streams through interactions and collects one (user, holdout_movie) pair per user."""
    # (This function is the same as before, but now accepts a path)
    holdout: Dict[int, str] = {}
    seen: Set[int] = set()

    print(f"Sampling user holdouts from {interactions_path} (streaming)...")
    for chunk in pd.read_csv(interactions_path, usecols=["user_id", "movie_id"], chunksize=2_000_000):
        for uid, mid in chunk.itertuples(index=False):
            holdout[uid] = mid
            seen.add(uid)
        if len(seen) >= n_users * 3:
            break
    
    users = random.sample(list(seen), min(len(seen), n_users))
    pairs = [(u, holdout[u]) for u in users if u in holdout]
    print(f"Prepared {len(pairs)} user holdouts.")
    return pairs

def evaluate_hit_at_20(reco: ALSRecommender, pairs: List[Tuple[int, str]]) -> float:
    """For each (user, holdout_movie), check if holdout appears in top-20."""
    hits = sum(1 for uid, target_mid in pairs if target_mid in reco.recommend(uid, k=20))
    return hits / max(1, len(pairs))

def measure_latency_throughput(reco: ALSRecommender, users: List[int]) -> Tuple[float, float, float]:
    """Times recommend() for a sample of users."""
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

def model_size_bytes(art_dir: str) -> int:
    """Sums the sizes of the ALS model artifacts."""
    files = [
        os.path.join(art_dir, "als_model.npz"),
        os.path.join(art_dir, "user_map.json"),
        os.path.join(art_dir, "movie_map.json"),
    ]
    return sum(os.path.getsize(f) for f in files if os.path.exists(f))

def main():
    reco = ALSRecommender.load(artifacts_dir="artifacts")

    # 1) Accuracy proxy: Hit@20
    pairs = sample_holdouts("data/interactions.csv", EVAL_USERS)
    hit20 = evaluate_hit_at_20(reco, pairs)
    print(f"Hit@20 (proxy accuracy) over {len(pairs)} users: {hit20:.4f}")

    # 2) Inference cost (latency/throughput)
    user_ids = [u for (u, _) in pairs]
    avg_ms, p95_ms, qps = measure_latency_throughput(reco, user_ids)
    print(f"Inference latency: avg={avg_ms:.2f} ms, p95={p95_ms:.2f} ms; throughput≈{qps:.2f} q/s")

    # 3) Model size
    sz = model_size_bytes("artifacts")
    print(f"Model size (artifacts total): {sz / (1024*1024):.2f} MiB")

if __name__ == "__main__":
    main()