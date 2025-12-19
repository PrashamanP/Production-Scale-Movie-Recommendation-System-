# Production-Scale-Movie-Recommendation-Service

This repo contains 2 models for recommending movies to users:
1. Model 1: Alternating Least Squares
2. Model 2: Popularity ranking + user-genre boost (Baseline model)

A lightweight sample CSV is included for fast tests; full data can be regenerated via Kafka

## Getting Started
Instructions on setting up the project locally.

### Prerequisites
* Python (tested on v3.12.3)

* From *requirements.txt*:
    ```
    pandas==2.2.2
    numpy==1.26.4
    pyarrow==21.0.0
    flask==3.1.2
    gunicorn==23.0.0
    confluent-kafka==2.11.1
    requests==2.32.5
    implicit==0.7.2
    ```

### Installation
* Clone the repository (main branch)
    ```sh
    git clone https://github.com/cmu-seai/group-project-f25-blockbusters.git
    cd group-project-f25-blockbusters
    ```
* Create virtual environment
    ```sh
    python3 -m venv .venv
    source .venv/bin/activate
    ```
* Install dependencies
    ```sh
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

### Data Setup
1. Fetch data from Kafka stream into a log file *[data/raw.log]*

    Open the tunnel in a new terminal (Terminal A):
    ```sh
    ssh -L 9092:localhost:9092 tunnel@128.2.220.241 -NT
    # password: mlip-kafka
    ```

    Consume events in a new terminal (Terminal B)
    ```bash
    mkdir -p data
    kcat -b localhost:9092 -t movielog13 -C -o beginning -q > data/raw.log
    ```

    Terminate both processes (Ctrl+C) when you have enough data.\
    We terminated when *raw.log* was approximately 2.2GB.

2. Convert Raw logs → CSV *[data/interactions.csv]*

    Run the following code in your virtual environment
    ```sh
    python3 - << 'PY'
    import re,csv,time,os
    src="data/raw.log"
    dst="data/interactions.csv"
    USER=re.compile(r",(\d+),"); WATCH=re.compile(r"GET /data/m/([^/]+)/\d+\.mpg"); RATE=re.compile(r"GET /rate/([^=]+)=(\d)")
    os.makedirs("data", exist_ok=True)
    with open(dst,"w",newline="") as out:
        w=csv.writer(out); w.writerow(["user_id","movie_id","rating","timestamp"])
        for line in open(src, encoding="utf-8", errors="ignore"):
            u=USER.search(line)
            if not u: continue
            uid=u.group(1); ts=int(time.time())
            m=WATCH.search(line)
            if m: w.writerow([uid,m.group(1),1,ts]); continue
            m=RATE.search(line)
            if m: w.writerow([uid,m.group(1),int(m.group(2)),ts])
    print("Wrote",dst)
    PY
    ```

3. Deduplicate interactions → CSV *[data/interactions_dedup.csv]*

    To prevent duplicate `(user_id, movie_id)` pairs from leaking into validation splits, deduplicate the freshly generated interactions file:
    ```sh
    python - <<'PY'
    import pandas as pd
    src = "data/interactions.csv"
    dst = "data/interactions_dedup.csv"
    df = (
        pd.read_csv(src)
        .sort_values(["user_id", "timestamp"], kind="mergesort")
        .assign(_row_order=lambda frame: range(len(frame)))
        .sort_values(["user_id", "timestamp", "_row_order"])
        .drop_duplicates(["user_id", "movie_id"], keep="last")
        .drop(columns="_row_order")
    )
    df.to_csv(dst, index=False)
    print(f"Wrote {dst} with {len(df)} unique interactions.")
    PY
    ```
    Treat `data/interactions_dedup.csv` as the canonical dataset for downstream training and evaluation jobs.

3. Build Movie Catalog *[artifacts/movie_catalog.json]*
    ```bash
    PYTHONPATH=src python3 -m build_movie_catalog
    ```

## Training
1. Alternating Least Squares Model (Model 1)
    ```sh
    PYTHONPATH=src python3 -m models.als.train
    # or with time (prints training cost: CPU, wall time, memory):
    /usr/bin/time -v PYTHONPATH=src python3 -m models.als.train
    ```
    Artifacts produced:
    - *artifacts/als_model.npz*: Model's user and item factor matrices
    - *artifacts/user_map.json*: User id to index mappings for inference
    - *artifacts/movie_map.json*: Movie id to index mappings for inference

2. Baseline: Popularity based General Model (Model 2)
    ```sh
    PYTHONPATH=src python3 -m models.pop-gen.train_popgen
    # or with time (prints training cost: CPU, wall time, memory):
    /usr/bin/time -v PYTHONPATH=src python3 -m models.pop-gen.train_popgen
    ```
    Artifacts produced:
    - *artifacts/popgen_popularity.parquet*: Popularity table
    - *artifacts/popgen_user_genre_prefs.parquet*: Top 10 genre per user

## Evaluation
1. Alternating Least Squares Model (Model 1)
    ```sh
    PYTHONPATH=src python3 -m models.als.evaluate
    ```
    Example output:
    ```
    Loading ALS model artifacts...
    Sampling user holdouts from interactions.csv (streaming)...
    Prepared 3000 user holdouts.
    Hit@20 (proxy accuracy) over 3000 users: 0.3813
    Inference latency: avg=0.13 ms, p95=0.15 ms; throughput≈7774.03 q/s
    Model size (artifacts total): 4.50 MiB
    Training cost: measure with `/usr/bin/time -v python3 als/train.py`
    ```

2. Baseline: Popularity based General Model (Model 2)
    ```sh
    PYTHONPATH=src python3 -m models.pop-gen.eval_popgen
    ```
    Example output:
    ```
    Loading recommender artifacts...
    Sampling user holdouts from interactions.csv (streaming)...
    Prepared 3000 user holdouts.
    Hit@20 (proxy accuracy) over 3000 users: 0.0783
    Inference latency: avg=2.69 ms, p95=2.81 ms; throughput≈371.36 q/s
    Model size (artifacts total): 1.53 MiB
    Training cost: measure with `/usr/bin/time -v python3 scripts/train_baseline_popgenre.py`
    ```

Report the four required properties:
- Proxy accuracy (Hit@20)
- Training cost (from `/usr/bin/time -v`)
- Inference cost (latency/throughput from eval)
- Model size (sum of artifacts used at inference)

## Experimentation & A/B Testing

Traffic splitting, telemetry, and per-variant analysis live in `src/serve_als_model.py`, `src/experimentation/`, and `src/evaluation/online_als.py`. To adjust variant weights or run the watchdog CLI, see [docs/experimentation.md](docs/experimentation.md).

## Versioning & Provenance

- Every call to `models.als.train` now emits a structured manifest that captures the model version, git commit, hyperparameters, and exact dataset snapshot (row counts + SHA256). Manifests live in `artifacts/manifests/<model_version>.json`, and the active manifest is mirrored at `artifacts/model_manifest.json`.
- Provide optional flags during training to control provenance metadata:
  ```bash
  PYTHONPATH=src python -m models.als.train \
    --data_path data/interactions_dedup.csv \
    --model_version als-20250304 \
    --data_version kafka-20250301 \
    --artifact_dir artifacts \
    --manifest_dir artifacts
  ```
- The Flask/Gunicorn service automatically loads the manifest on startup, attaches provenance headers to every `/recommend/<uid>` response, and appends provenance fields to its structured Kafka-compatible logs. See [docs/provenance.md](docs/provenance.md) for details on the fields and operational guidance.

### Offline evaluation (ALS)
Our offline analysis deduplicates raw interactions to avoid temporal leakage, applies a chronological leave-last-out split per user, trains a fresh ALS model on the training slice, and scores hit-based metrics on the held-out interactions.

**1. Prepare evaluation data**
- Activate the project environment and ensure `PYTHONPATH=src`.
- Use the upstream deduplicated dataset (`data/interactions_dedup.csv`) generated during *Data Setup*. For quick smoke tests the bundled sample (`data/interactions_sample.csv`) is available.
- (Recommended for samples) If you rely on the sample CSV, deduplicate each `(user_id, movie_id)` pair so the hold-out interaction is novel for that user. This mirrors the preprocessing used for the numbers we report.
    ```sh
    python - <<'PY'
    import pandas as pd
    source = "data/interactions_sample.csv"  # replace with data/interactions.csv for full runs
    df = pd.read_csv(source)
    dedup = df.sort_values("timestamp").drop_duplicates(["user_id", "movie_id"], keep="last")
    dedup.to_csv("data/interactions_sample_dedup.csv", index=False)
    print("Wrote data/interactions_sample_dedup.csv with", len(dedup), "rows")
    PY
    ```
- The evaluation script ingests the CSV directly, validates schema (`user_id`, `movie_id`, optional `rating`, `timestamp`), and preserves row order via `_row_order` to break ties.

**2. Run the evaluator**
```sh
PYTHONPATH=src python -m evaluation.offline_als \
  --interactions data/interactions_dedup.csv \
  --holdout-per-user 1 \
  --min-history 1 \
  --k 10 --k 20 \
  --output-json artifacts/offline_eval_latest.json
```
- For sample-based smoke runs, swap `data/interactions_dedup.csv` for `data/interactions_sample_dedup.csv` (produced with the snippet above).
- Flags can be adjusted (e.g., add `--sample-users 2000` for faster smoke runs or extend `--k` values). The CLI lives in `src/evaluation/offline_als.py`.
- The command prints metrics to stdout and writes the same dictionary to `artifacts/offline_eval_latest.json` for traceability.

**3. Inspect results**
- The JSON contains per-cutoff hit, precision, recall, ndcg, plus counts of evaluated, training, and hold-out interactions. Example (sample data with deduplication):
    ```json
    {
      "hit@10": 0.0526,
      "hit@20": 0.0789,
      "precision@10": 0.0053,
      "recall@20": 0.0789,
      "ndcg@20": 0.0395,
      "evaluated_users": 152.0,
      "train_interactions": 1672.0,
      "holdout_interactions": 152.0
    }
    ```
- These metrics align with the “metric / data / operationalization” pattern: metric = hit@k and related ranking metrics, data = chronological leave-last-out validation slice, operationalization = `evaluation.offline_als` CLI.

**4. Reference implementation**
- Core logic: `src/evaluation/offline_als.py`
- Recommended preprocessing script (above): place in your notebook/automation or adapt into a helper module if you need to share it across pipelines.
### Online Evaluation (Milestone 2)
This script evaluates the production model's performance by parsing Kafka logs for recommendation events and subsequent watch events.

```sh
PYTHONPATH=src python3 src/evaluate_online.py --log_file data/raw.log
```
This command processes the specified log file and creates a JSON report.

Example output file (`artifacts/online_evaluation.json`):
```json
{
  "metric": "Recommendation Watch-Through Rate (WTR)",
  "log_file_processed": "data/raw.log",
  "total_log_lines": 33203894,
  "attribution_window_hours": 1,
  "total_recommendation_events": 241470,
  "converted_recommendation_events": 119582,
  "watch_through_rate": 0.49522507972004803,
  "calculation_timestamp": "2025-10-27T22:03:48.947788Z"
}
```

## Serving (HTTP service)

- Local Flask development
    ```sh
    python3 -m src.serve_als_model
    ```

- Production (gunicorn)
    ```sh
    gunicorn -b 0.0.0.0:8082 "src.serve_als_model:create_app()" --workers 2 --threads 4 --timeout 30
    ```

Endpoint:
```
GET http://<vm>:8082/recommend/<userid>
```
Returns plain text: a single line of up to 20 comma-separated movie IDs.

* Example:
    ```sh
    wget -qO- http://localhost:8082/recommend/39157

    # Output:
    office+space+1999,crouching+tiger_+hidden+dragon+2000,true+lies+1994,the+lord+of+the+rings+the+fellowship+of+the+ring+2001,charlies+angels+2000,eternal+sunshine+of+the+spotless+mind+2004,dr.+strangelove+or+how+i+learned+to+stop+worrying+and+love+the+bomb+1964,being+john+malkovich+1999,the+lord+of+the+rings+the+return+of+the+king+2003,ace+ventura+when+nature+calls+1995,dumb+and+dumber+1994,little+miss+sunshine+2006,memento+2000,star+wars+episode+ii+-+attack+of+the+clones+2002,mission+impossible+ii+2000,high+fidelity+2000,gattaca+1997,best+in+show+2000,donnie+darko+2001,about+a+boy+2002
    ```
## Data Quality

This section ensures that both *current* (`interactions.csv`) and *reference* (`interactions_original.csv`) datasets meet schema and drift standards before model training or retraining.

Typical workflow:
- Validate incoming *current* and *reference* data files.
- Clean and write → `interactions_clean_cur.parquet` and `interactions_clean_ref.parquet`.
- Run drift comparison between the two to detect behavioral or popularity shifts that might require retraining.

---

1. **Schema validation — CURRENT dataset**
    ```sh
    PYTHONPATH=src python3 -m data_quality.validate_data \
      --in data/interactions.csv \
      --schema src/data_quality/schemas/interactions.schema.json \
      --out data/interactions_clean_cur.parquet \
      --report reports/dq_schema_report_cur.json \
      --chunksize 250000 \
      --verbose
    ```
    *Quick summary of the report:*
    ```sh
    jq '{input_rows, clean_rows, reject_rows, reject_rate}' reports/dq_schema_report_cur.json
    ```
    *Strict gate (CI or local): add to the validate command to fail on any reject*
    ```sh
    --max_reject_rate 0.0
    ```

2. **Schema validation — REFERENCE dataset**
    ```sh
    PYTHONPATH=src python3 -m data_quality.validate_data \
      --in data/interactions_original.csv \
      --schema src/data_quality/schemas/interactions.schema.json \
      --out data/interactions_clean_ref.parquet \
      --report reports/dq_schema_report_ref.json \
      --chunksize 250000 \
      --verbose
    ```
    *Quick summary of the report:*
    ```sh
    jq '{input_rows, clean_rows, reject_rows, reject_rate}' reports/dq_schema_report_ref.json
    ```

    The *reference* dataset (`interactions_original.csv`) represents a previously verified, stable snapshot used to benchmark drift against new data.

---

3. **Drift check — Reference vs Current**
    ```sh
    PYTHONPATH=src python3 -m data_quality.check_drift \
      --ref data/interactions_clean_ref.parquet \
      --cur data/interactions_clean_cur.parquet \
      --out reports/dq_drift_report.json \
      --top100_overlap_threshold 0.40 \
      --pop_top100_delta_moderate 0.10 --pop_top100_delta_strong 0.15 \
      --pop_top10_delta_moderate 0.05  --pop_top10_delta_strong 0.08 \
      --pop_gini_delta_moderate 0.06  --pop_gini_delta_strong 0.10
    ```
    *Inspect drift results:*
    ```sh
    jq '{drift_detected, metrics: {rating_psi, movie_popularity, top100_overlap, user_activity}}' \
      reports/dq_drift_report.json
    ```
    If `"drift_detected": true`, retraining is recommended.  
    Drift can occur when:
    - The top 100 popular movies overlap drops below 0.4.
    - Gini coefficients or popularity deltas exceed defined thresholds.
    - User engagement patterns shift significantly.

---

4. **Drift check — Single snapshot (no reference available)**
    ```sh
    PYTHONPATH=src python3 -m data_quality.check_drift \
      --cur data/interactions_clean_cur.parquet \
      --out reports/dq_drift_report_single.json
    ```
    Used when the reference dataset does not yet exist — typically before the first model training baseline is established.

---

5. **Sanity test — Validate schema enforcement (tiny CSV)**
    ```sh
    cat > data/interactions_tiny.csv << 'EOF'
    user_id,movie_id,rating,timestamp
    123,the+matrix+1999,5,1700000000
    -3,illegal!id,4,1700000000
    789,good+id,seven,1700000000
    456,also+good+id,6,1700000000
    999,bad id with spaces,3,1700000000
    EOF

    PYTHONPATH=src python3 -m data_quality.validate_data \
      --in data/interactions_tiny.csv \
      --schema src/data_quality/schemas/interactions.schema.json \
      --out data/interactions_tiny_clean.parquet \
      --report reports/dq_schema_report_tiny.json \
      --chunksize 100 \
      --verbose \
      --max_reject_rate 0.0

    jq '{input_rows, clean_rows, reject_rows, reject_rate}' reports/dq_schema_report_tiny.json
    ```
    This example shows how invalid rows (negative IDs, malformed movie IDs, non-integer ratings) are automatically rejected under the schema.
