"""
Offline evaluation for the ALS recommender.

It applies a chronological leave-last-N-out split per user, retrains an
ALS model on the training split, and reports several ranking metrics on the
validation split. It is designed to be lightweight enough to run on subsamples,
which makes it suitable for CI smoke tests as well as larger offline analysis
runs.

Example:
    PYTHONPATH=src python -m evaluation.offline_als \
        --interactions data/interactions_sample.csv \
        --sample-users 2000 --holdout-per-user 1 --k 10 --k 20 
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Optional

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix, csr_matrix

from models.als.model import ALSRecommender


DEFAULT_ALS_PARAMS = {
    "factors": 50,
    "regularization": 0.01,
    "iterations": 20,
    "alpha": 40.0,
    "random_state": 42,
}


@dataclass
class DatasetSplits:
    train: pd.DataFrame
    validation: pd.DataFrame


def _read_interactions(path: str, dtype_override: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Read the interactions CSV, preserving row order for chronological fallback."""
    header = pd.read_csv(path, nrows=0)
    available_cols = header.columns.tolist()

    required_cols = ["user_id", "movie_id"]
    missing = [c for c in required_cols if c not in available_cols]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    optional_cols = ["rating", "timestamp"]
    usecols = required_cols + [c for c in optional_cols if c in available_cols]

    dtype = {"user_id": "string", "movie_id": "string", "rating": "float32"}
    if dtype_override:
        dtype.update(dtype_override)

    df = pd.read_csv(path, usecols=usecols, dtype=dtype)

    if "rating" not in df.columns:
        df["rating"] = 1.0

    df["user_id"] = df["user_id"].astype("string")
    df["movie_id"] = df["movie_id"].astype("string")
    df["rating"] = df["rating"].fillna(0.0).clip(lower=0.0)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df["_row_order"] = np.arange(len(df))
    return df


def _train_validation_split(
    interactions: pd.DataFrame,
    holdout_per_user: int,
    min_history: int,
) -> DatasetSplits:
    """Leave-last-N-out split per user with optional minimum history filtering."""
    if "timestamp" in interactions.columns:
        ordered = interactions.sort_values(["user_id", "timestamp", "_row_order"])
    else:
        ordered = interactions.sort_values(["user_id", "_row_order"])

    counts = ordered.groupby("user_id")["movie_id"].count()
    required_total = holdout_per_user + max(min_history, 1)
    eligible_users = counts[counts >= required_total].index

    filtered = ordered[ordered["user_id"].isin(eligible_users)].copy()

    val_idx = filtered.groupby("user_id").tail(holdout_per_user).index
    validation = filtered.loc[val_idx]
    train = filtered.drop(index=val_idx)

    return DatasetSplits(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
    )


def _build_matrix(
    interactions: pd.DataFrame,
    alpha: float,
) -> tuple[csr_matrix, Dict[str, int], Dict[int, str]]:
    """Construct a CSR matrix and mapping dictionaries from interactions."""
    user_codes, user_uniques = pd.factorize(interactions["user_id"], sort=True)
    movie_codes, movie_uniques = pd.factorize(interactions["movie_id"], sort=True)

    confidence = 1.0 + alpha * interactions["rating"].to_numpy(dtype=np.float32)

    matrix = coo_matrix(
        (confidence, (user_codes, movie_codes)),
        shape=(len(user_uniques), len(movie_uniques)),
    ).tocsr()

    user_map = {str(user_id): int(idx) for idx, user_id in enumerate(user_uniques)}
    item_inv_map = {int(idx): str(movie_id) for idx, movie_id in enumerate(movie_uniques)}
    return matrix, user_map, item_inv_map


def recommend_items(
    recommender: ALSRecommender,
    user_index: int,
    item_inv_map: Dict[int, str],
    seen_item_indices: set[int],
    top_k: int,
) -> List[str]:
    """Computes top-k recommendations via raw factor scores."""
    model = recommender.model
    if user_index >= model.user_factors.shape[0]:
        return []

    user_vec = model.user_factors[user_index]
    if user_vec.ndim == 0 or not user_vec.any():
        return []

    scores = user_vec @ model.item_factors.T
    ranked = np.argsort(scores)[::-1]

    recs: List[str] = []
    for idx in ranked:
        if idx in seen_item_indices:
            continue
        movie_id = item_inv_map.get(int(idx))
        if movie_id is not None:
            recs.append(movie_id)
        if len(recs) >= top_k:
            break
    return recs


def _dcg_at_k(recommended: Sequence[str], relevant: Sequence[str], k: int) -> float:
    score = 0.0
    relevant_set = set(relevant)
    for rank, movie_id in enumerate(recommended[:k], start=1):
        if movie_id in relevant_set:
            score += 1.0 / math.log2(rank + 1)
    return score


def _ideal_dcg(num_relevant: int, k: int) -> float:
    use = min(num_relevant, k)
    if use <= 0:
        return 0.0
    return sum(1.0 / math.log2(rank + 1) for rank in range(1, use + 1))


def evaluate_model(
    recommender: ALSRecommender,
    train_matrix: csr_matrix,
    splits: DatasetSplits,
    user_map: Dict[str, int],
    item_inv_map: Dict[int, str],
    k_values: Sequence[int],
) -> Dict[str, float]:
    """Compute ranking metrics for each user in the validation split."""
    metrics = {f"hit@{k}": 0.0 for k in k_values}
    metrics.update({f"precision@{k}": 0.0 for k in k_values})
    metrics.update({f"recall@{k}": 0.0 for k in k_values})
    metrics.update({f"ndcg@{k}": 0.0 for k in k_values})

    evaluated_users = 0
    grouped = splits.validation.groupby("user_id")["movie_id"].apply(list)
    max_k = max(k_values)
    train_csr = train_matrix.tocsr()

    for user_id, targets in grouped.items():
        user_key = str(user_id)
        if user_key not in user_map:
            continue
        user_idx = user_map[user_key]
        seen = set(map(int, train_csr.getrow(user_idx).indices))
        recommended = recommend_items(recommender, user_idx, item_inv_map, seen, max_k)
        if not recommended:
            continue

        evaluated_users += 1
        target_set = set(targets)

        for k in k_values:
            top_k = recommended[:k]
            hits = sum(1 for mid in top_k if mid in target_set)
            metrics[f"hit@{k}"] += 1.0 if hits > 0 else 0.0

            denom_prec = min(k, len(top_k)) or 1.0
            denom_rec = len(target_set) or 1.0
            metrics[f"precision@{k}"] += hits / denom_prec
            metrics[f"recall@{k}"] += hits / denom_rec

            dcg = _dcg_at_k(top_k, targets, k)
            idcg = _ideal_dcg(len(target_set), k)
            metrics[f"ndcg@{k}"] += (dcg / idcg) if idcg > 0 else 0.0

    results: Dict[str, float] = {}
    denom = evaluated_users or 1
    for k in k_values:
        results[f"hit@{k}"] = metrics[f"hit@{k}"] / denom
        results[f"precision@{k}"] = metrics[f"precision@{k}"] / denom
        results[f"recall@{k}"] = metrics[f"recall@{k}"] / denom
        results[f"ndcg@{k}"] = metrics[f"ndcg@{k}"] / denom

    results["evaluated_users"] = float(evaluated_users)
    return results


def run_offline_evaluation(args: argparse.Namespace) -> Dict[str, float]:
    """Convenience wrapper used by the CLI and potential unit tests."""
    interactions = _read_interactions(args.interactions)

    if args.sample_users:
        unique_users = interactions["user_id"].drop_duplicates()
        n = min(int(args.sample_users), len(unique_users))
        sampled_users = unique_users.sample(n=n, random_state=args.random_state, replace=False)
        interactions = interactions[interactions["user_id"].isin(sampled_users)]

    splits = _train_validation_split(
        interactions,
        holdout_per_user=args.holdout_per_user,
        min_history=args.min_history,
    )

    matrix, user_map, item_inv_map = _build_matrix(
        splits.train,
        alpha=args.alpha,
    )

    model = AlternatingLeastSquares(
        factors=args.factors,
        regularization=args.regularization,
        iterations=args.iterations,
        random_state=args.random_state,
    )
    model.fit(matrix)

    recommender = ALSRecommender(
        user_factors=model.user_factors,
        item_factors=model.item_factors,
        user_map=user_map,
        movie_inv_map=item_inv_map,
    )
    recommender.model = model

    metrics = evaluate_model(
        recommender=recommender,
        train_matrix=matrix,
        splits=splits,
        user_map=user_map,
        item_inv_map=item_inv_map,
        k_values=args.k_values,
    )

    metrics["train_interactions"] = float(len(splits.train))
    metrics["holdout_interactions"] = float(len(splits.validation))

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline evaluation for the ALS recommender.")
    parser.add_argument(
        "--interactions",
        type=str,
        required=True,
        help="Path to the interactions CSV (needs user_id, movie_id, optional rating/timestamp).",
    )
    parser.add_argument(
        "--sample-users",
        type=int,
        default=0,
        help="If >0, randomly sample this many users for a faster smoke run.",
    )
    parser.add_argument(
        "--holdout-per-user",
        type=int,
        default=1,
        help="Number of latest interactions per user reserved for validation.",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=1,
        help="Minimum number of training interactions per user after the holdout.",
    )
    parser.add_argument(
        "--k",
        dest="k_values",
        type=int,
        action="append",
        default=[10, 20],
        help="Ranking cutoffs to evaluate (repeat flag for multiple values).",
    )
    parser.add_argument(
        "--factors",
        type=int,
        default=DEFAULT_ALS_PARAMS["factors"],
        help="Number of latent factors for ALS.",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=DEFAULT_ALS_PARAMS["regularization"],
        help="Regularization strength for ALS.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ALS_PARAMS["iterations"],
        help="Number of ALS training iterations.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALS_PARAMS["alpha"],
        help="Confidence scaling for implicit feedback (confidence = 1 + alpha * rating).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_ALS_PARAMS["random_state"],
        help="Random seed to make sampling and ALS deterministic.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        help="Optional path to dump the resulting metrics as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    metrics = run_offline_evaluation(args)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
