# Provenance & Versioning


1. **Training-time manifests** that record immutable metadata for each model build.
2. **Serving-time propagation** of that metadata through structured logs and HTTP headers.

## Training-Time Manifests

Running the ALS training CLI (`PYTHONPATH=src python -m models.als.train`) now produces:

- `artifacts/als_model.npz`, `user_map.json`, and `movie_map.json` (unchanged artifacts).
- A manifest per model version under `artifacts/manifests/<model_version>.json`.
- A canonical pointer `artifacts/model_manifest.json` that mirrors the most recent manifest.

Key CLI options:

| Flag | Purpose |
| --- | --- |
| `--model_version <id>` | Override the auto-generated `als-YYYYMMDDHHMMSS` identifier. Useful when shipping coordinated releases. |
| `--data_version <id>` | Tag the dataset snapshot (e.g., Kafka offset or S3 version). Defaults to the SHA256 checksum of `--data_path`. |
| `--manifest_dir <path>` | Store manifests outside of `--artifact_dir` if needed. Defaults to the artifact directory. |

Every manifest captures:

```jsonc
{
  "model_family": "als",
  "model_version": "als-20250304",
  "trained_at": "2025-03-04T18:12:55.123456Z",
  "pipeline": {
    "git_commit": "80d8e56f5c1b...",
    "git_branch": "main",
    "git_dirty": "0"
  },
  "data": {
    "dataset_path": "data/interactions_dedup.csv",
    "dataset_version": "kafka-20250301",
    "dataset_sha256": "e7d9...",
    "row_count": 2_445_982,
    "unique_users": 973_112,
    "unique_movies": 25_913,
    "dataset_size_bytes": 1_942_614_271
  },
  "hyperparameters": { "...": "..." },
  "artifacts": {
    "als_model": { "path": "artifacts/als_model.npz", "sha256": "...", "size_bytes": 4723920 },
    "user_map": { "path": "artifacts/user_map.json", "sha256": "...", "size_bytes": 318049 },
    "movie_map": { "path": "artifacts/movie_map.json", "sha256": "...", "size_bytes": 219873 }
  },
  "manifest_sha256": "51ac..."
}
```

The manifest SHA is logged so we can prove which manifest was used later. If a manifest ever needs to be inspected, grab it via:

```bash
jq '.' artifacts/model_manifest.json
```

## Serving-Time Propagation

`src/serve_als_model.py` loads the active manifest on startup. If the manifest exists:

- Every recommendation log line (the ones forwarded to Kafka) now carries provenance fields: `model_version`, `pipeline_commit`, `pipeline_branch`, `data_version`, and `manifest_sha256`.
- HTTP responses include headers so operators can trace individual requests without digging through logs:

| Header | Description |
| --- | --- |
| `X-Model-Version` | Manifest `model_version`. |
| `X-Pipeline-Commit` | Short git SHA that produced the model. |
| `X-Data-Version` | Dataset identifier (explicit flag or SHA). |
| `X-Model-Manifest` | Manifest SHA, which can be matched against files in `artifacts/manifests/`. |

Example log excerpt:

```
2025-03-04T18:21:03.219384Z,39157,recommendation request /recommend/39157, status 200, variant=als_model, bucket=0.392104, result: ..., 42 ms, model_version=als-20250304, pipeline_commit=80d8e56, data_version=kafka-20250301, manifest_sha256=51ac...
```

Given the log line (or response headers), we can:

1. Find the matching manifest (`manifest_sha256`).
2. Read the manifest to identify the model artifacts (`als_model.npz`, `user_map.json`).
3. Confirm the git commit and dataset SHA that produced the model.
