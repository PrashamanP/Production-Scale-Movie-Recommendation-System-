import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix, csr_matrix

# Ensure project modules are importable when run as a script
sys.path.insert(0, "/app")
from src.provenance.manifest import build_model_manifest, persist_manifest

# Group hyperparameters for clarity
HPARAMS = {
    "factors": 50,
    "regularization": 0.01,
    "iterations": 20,
    "alpha": 40,
    "random_state": 42,
}

EMPTY_PROFILE = {"row_count": 0, "unique_users": 0, "unique_movies": 0}


def _load_and_prepare_data(data_path: str) -> Tuple[csr_matrix, dict, dict, Dict[str, int]]:
    """Loads interaction data and prepares it for the model."""
    print("Reading and preparing interaction data...")
    
    usecols = ["user_id", "movie_id", "rating"]
    dtypes = {"user_id": "string", "movie_id": "string", "rating": "int8"}
    
    # Handle missing or empty files early
    if not os.path.exists(data_path) or os.stat(data_path).st_size == 0:
        print("Empty or missing data file — returning empty outputs.")
        return csr_matrix((0, 0)), {}, {}, EMPTY_PROFILE.copy()

    try:
        df = pd.read_csv(data_path, usecols=usecols, dtype=dtypes).dropna()
    except pd.errors.EmptyDataError:
        print("EmptyDataError: returning empty outputs.")
        return csr_matrix((0, 0)), {}, {}, EMPTY_PROFILE.copy()

    # Handle CSVs with headers but no rows
    if df.empty:
        print("No interaction data found — returning empty outputs.")
        return csr_matrix((0, 0)), {}, {}, EMPTY_PROFILE.copy()

    data_profile = {
        "row_count": int(len(df)),
        "unique_users": int(df["user_id"].nunique()),
        "unique_movies": int(df["movie_id"].nunique()),
    }

    # Normal processing
    df['user_idx'] = df['user_id'].astype('category').cat.codes
    df['movie_idx'] = df['movie_id'].astype('category').cat.codes

    confidence = 1 + HPARAMS["alpha"] * df['rating'].to_numpy()

    interaction_matrix = coo_matrix(
        (confidence, (df['user_idx'], df['movie_idx']))
    ).tocsr()

    print(f"Built a {interaction_matrix.shape[0]}x{interaction_matrix.shape[1]} sparse matrix.")

    user_map = dict(zip(df['user_id'].astype(str), df['user_idx']))
    movie_map = dict(zip(df['movie_id'], df['movie_idx']))

    return interaction_matrix, user_map, movie_map, data_profile

def _train_model(interaction_matrix: csr_matrix) -> AlternatingLeastSquares:
    """Trains the ALS model."""
    print("Training ALS model...")
    model = AlternatingLeastSquares(
        factors=HPARAMS["factors"],
        regularization=HPARAMS["regularization"],
        iterations=HPARAMS["iterations"],
        random_state=HPARAMS["random_state"],
    )
    model.fit(interaction_matrix)
    print("Training complete.")
    return model

def _save_artifacts(
    model: AlternatingLeastSquares, user_map: dict, movie_map: dict, art_dir: str
) -> Dict[str, str]:
    """Saves model factors and mappings to disk."""
    os.makedirs(art_dir, exist_ok=True)
    
    # 1. Save model factors
    out_model = os.path.join(art_dir, "als_model.npz")
    np.savez(
        out_model,
        user_factors=model.user_factors,
        item_factors=model.item_factors
    )
    print(f"Wrote model factors to {out_model}")

    # 2. Save user map
    out_user_map = os.path.join(art_dir, "user_map.json")
    with open(out_user_map, "w") as f:
        json.dump(user_map, f)
    print(f"Wrote user ID map to {out_user_map}")
    
    # 3. Save movie map
    out_movie_map = os.path.join(art_dir, "movie_map.json")
    with open(out_movie_map, "w") as f:
        json.dump(movie_map, f)
    print(f"Wrote movie ID map to {out_movie_map}")

    return {
        "als_model": out_model,
        "user_map": out_user_map,
        "movie_map": out_movie_map,
    }

def main():
    parser = argparse.ArgumentParser(description="Train an ALS recommendation model.")
    parser.add_argument("--data_path", type=str, default="data/interactions.csv", help="Path to the interactions data.")
    parser.add_argument("--artifact_dir", type=str, default="artifacts", help="Directory to save model artifacts.")
    parser.add_argument(
        "--manifest_dir",
        type=str,
        default=None,
        help="Directory to store provenance manifests (defaults to artifact_dir).",
    )
    parser.add_argument(
        "--model_version",
        type=str,
        default=None,
        help="Optional explicit model version identifier.",
    )
    parser.add_argument(
        "--data_version",
        type=str,
        default=None,
        help="Optional identifier describing the training dataset snapshot.",
    )
    args = parser.parse_args()

    matrix, user_map, movie_map, data_profile = _load_and_prepare_data(args.data_path)
    model = _train_model(matrix)
    artifact_paths = _save_artifacts(model, user_map, movie_map, args.artifact_dir)

    manifest = build_model_manifest(
        model_family="als",
        artifact_paths=artifact_paths,
        hyperparameters=HPARAMS,
        data_profile=data_profile,
        data_path=os.path.abspath(args.data_path),
        data_version=args.data_version,
        model_version=args.model_version,
        extra_metadata={
            "trained_at_host": platform.node(),
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )

    manifest_dir = args.manifest_dir or args.artifact_dir
    manifest_path = persist_manifest(manifest, manifest_dir, args.model_version)
    print(f"Wrote provenance manifest to {manifest_path}")

if __name__ == "__main__":
    main()
