# Experimentation & A/B Testing Guide

This project now serves recommendations through an experiment-aware gateway that can split traffic across multiple model variants. Use the steps below to configure splits, deploy changes, and analyze results.

## 1. Configure Variant Splits

Experiment configs live under `artifacts/experiments/`. Each JSON file defines:

```json
{
  "experiment_id": "als_vs_popgen",
  "salt": "als-popgen-v1",
  "variants": [
    { "name": "als_model", "weight": 0.6 },
    { "name": "popgen_model", "weight": 0.4 }
  ]
}
```

- `experiment_id`: label emitted in logs and HTTP headers.
- `salt`: optional string to reshuffle bucket assignments.
- `variants`: ordered list with `name` and relative `weight`. Weights are normalized to percentages.

To adjust traffic, edit the JSON and redeploy. The gateway reads the path from `EXPERIMENT_CONFIG_PATH`; by default it loads `artifacts/experiments/als_vs_pop.json`.

## 2. Serving Behavior & Telemetry

- `src/serve_als_model.py` now instantiates both ALS and PopGen recommenders and routes every `/recommend/<user_id>` through the `HashExperimentRouter`.
- Each response sets `X-Experiment-Id` and `X-Experiment-Variant` headers and logs the chosen variant plus the recommendation list. Those logs should be forwarded to Kafka to power monitoring.
- Prometheus-format metrics are exposed at `/metrics`:
  - `experiment_reco_requests_total{variant=...}`
  - `experiment_reco_latency_ms_bucket{variant=...}`

Scrape these metrics (e.g., via Prometheus or Grafana Agent) to visualize per-variant traffic and latency in near real time.

## 3. Analyze Watch-Through Rate (WTR) Per Variant

Use the updated CLI to compute online quality metrics and statistical confidence:

```bash
PYTHONPATH=src python src/evaluation/online_als.py \
  /path/to/kafka.log \
  --output_file artifacts/wtr_latest.json \
  --variant-a als_model \
  --variant-b popgen_model \
  --window 1 \
  --alpha 0.05
```

The script outputs:
- Overall WTR
- Per-variant totals/conversions
- Two-proportion z-test comparing the requested variants (effect size, p-value, 95% CI)

Archive the resulting JSON (and dashboard screenshots) for Milestone 3 reporting.

## 4. Reporting Checklist

When running an experiment:
1. Commit the experiment config JSON used for the split.
2. Capture Prometheus/Grafana panels showing per-variant availability, latency, WTR, and drift.
3. Run `online_als.py` on the corresponding Kafka log and store the JSON under `artifacts/`.
4. Summarize the experiment (traffic share, runtime, z-test results) in your Gradescope report.
