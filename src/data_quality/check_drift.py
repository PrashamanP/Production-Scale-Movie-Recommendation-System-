"""
Data drift & health checks (lean version; no NaN-prone metrics).

Modes:
1) Reference-vs-Current (when --ref provided)
   - Rating PSI (moderate/strong thresholds)
   - Movie popularity: top10/top100 shares, Gini (levels) + ABS deltas with moderate/strong flags
   - Top-100 overlap (threshold)
   - User activity deltas (active users, one-and-done share, p95 events/user)

2) Single-snapshot (when --ref omitted)
   - Label degeneracy: rating entropy (bits) + top rating share
   - Movie popularity: top10/top100 shares, Gini (level flags)
   - User activity: active users, one-and-done share, p95 events/user

Outputs a JSON report with raw metrics, thresholds, flags, and an overall 'drift_detected' boolean.
Never exits non-zero; for M2 it only reports.
"""

from __future__ import annotations
import argparse
import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# -------------------- Utilities --------------------

def _read_any(path: str) -> pd.DataFrame:
    lower = path.lower()
    if lower.endswith(".parquet"):
        return pd.read_parquet(path)
    if lower.endswith(".json") or lower.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)

def _ensure_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def psi_from_series(ref: pd.Series, cur: pd.Series, bins: int = 10) -> Tuple[float, List[float]]:
    """Population Stability Index using reference quantile bins."""
    ref = ref.dropna()
    cur = cur.dropna()
    if len(ref) == 0 or len(cur) == 0:
        return float("nan"), []
    qs = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(ref, qs))
    if len(edges) < 3:
        lo = float(ref.min())
        hi = float(ref.max()) if ref.max() > lo else lo + 1.0
        edges = np.linspace(lo, hi, bins + 1)
    r_counts, edges = np.histogram(ref, bins=edges)
    c_counts, _ = np.histogram(cur, bins=edges)
    eps = 1e-6
    r_prop = np.clip(r_counts / max(1, len(ref)), eps, 1.0)
    c_prop = np.clip(c_counts / max(1, len(cur)), eps, 1.0)
    psi_vals = (r_prop - c_prop) * np.log(r_prop / c_prop)
    psi = float(np.sum(psi_vals))
    return psi, edges.tolist()

def rating_entropy_bits(series: pd.Series) -> float:
    counts = series.value_counts(dropna=True)
    total = counts.sum()
    if total == 0:
        return float("nan")
    p = (counts / total).values
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if len(p) else float("nan")

def gini_from_counts(counts: np.ndarray) -> float:
    """Gini coefficient for non-negative counts."""
    x = np.asarray(counts, dtype=float)
    x = x[x >= 0]
    if x.size == 0 or x.sum() == 0:
        return 0.0
    x.sort()
    n = x.size
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))

def movie_popularity_metrics(df: pd.DataFrame) -> Dict[str, float]:
    mc = df["movie_id"].value_counts()
    t10 = float(mc.head(10).sum() / len(df)) if len(df) else float("nan")
    t100 = float(mc.head(100).sum() / len(df)) if len(df) else float("nan")
    gini = gini_from_counts(mc.values)
    return {"top10_share": t10, "top100_share": t100, "gini_movies": gini}

def user_activity_metrics(df: pd.DataFrame) -> Dict[str, float]:
    by_user = df.groupby("user_id").size()
    active_users = int(by_user.size)
    one_and_done_share = float((by_user == 1).mean()) if by_user.size else float("nan")
    p95 = float(by_user.quantile(0.95)) if by_user.size else float("nan")
    return {"active_users": active_users, "one_and_done_share": one_and_done_share, "p95_events_per_user": p95}


# -------------------- CLI --------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Data drift & health checks (lean, NaN-free).")
    ap.add_argument("--ref", required=False, help="Reference interactions (csv|parquet|jsonl)")
    ap.add_argument("--cur", required=True, help="Current interactions (csv|parquet|jsonl)")
    ap.add_argument("--out", required=True, help="Output JSON report")

    # Rating PSI thresholds
    ap.add_argument("--psi_moderate", type=float, default=0.10)
    ap.add_argument("--psi_strong", type=float, default=0.25)

    # Single-snapshot label degeneracy thresholds
    ap.add_argument("--min_rating_entropy", type=float, default=0.10)
    ap.add_argument("--max_top_rating_share", type=float, default=0.99)

    # Popularity level thresholds (absolute levels)
    ap.add_argument("--top10_share_threshold", type=float, default=0.50)
    ap.add_argument("--gini_movies_threshold", type=float, default=0.90)

    # Popularity delta thresholds (ABSOLUTE DELTAS) to trigger drift in ref-vs-cur
    ap.add_argument("--pop_top10_delta_moderate", type=float, default=0.05)
    ap.add_argument("--pop_top10_delta_strong",   type=float, default=0.08)
    ap.add_argument("--pop_top100_delta_moderate", type=float, default=0.10)
    ap.add_argument("--pop_top100_delta_strong",   type=float, default=0.15)
    ap.add_argument("--pop_gini_delta_moderate", type=float, default=0.06)
    ap.add_argument("--pop_gini_delta_strong",   type=float, default=0.10)

    # Overlap & user thresholds
    ap.add_argument("--top100_overlap_threshold", type=float, default=0.40)  # suggest 0.35–0.40
    ap.add_argument("--one_and_done_threshold", type=float, default=0.9999)

    args = ap.parse_args()

    # Load current
    cur = _read_any(args.cur)
    _ensure_columns(cur, ["user_id", "movie_id", "rating"])
    cur = cur.copy()
    cur["movie_id"] = cur["movie_id"].astype(str)

    report: Dict[str, any] = {
        "mode": "ref_vs_cur" if args.ref else "single_snapshot",
        "current_path": args.cur,
        "reference_path": args.ref if args.ref else None,
        "thresholds": {
            "psi_moderate": args.psi_moderate,
            "psi_strong": args.psi_strong,
            "min_rating_entropy": args.min_rating_entropy,
            "max_top_rating_share": args.max_top_rating_share,
            "top10_share_threshold": args.top10_share_threshold,
            "gini_movies_threshold": args.gini_movies_threshold,
            "top100_overlap_threshold": args.top100_overlap_threshold,
            "one_and_done_threshold": args.one_and_done_threshold,
            "pop_top10_delta_moderate": args.pop_top10_delta_moderate,
            "pop_top10_delta_strong": args.pop_top10_delta_strong,
            "pop_top100_delta_moderate": args.pop_top100_delta_moderate,
            "pop_top100_delta_strong": args.pop_top100_delta_strong,
            "pop_gini_delta_moderate": args.pop_gini_delta_moderate,
            "pop_gini_delta_strong": args.pop_gini_delta_strong,
        },
        "metrics": {},
    }

    flags: List[bool] = []

    # ---------- Single-snapshot ----------
    if not args.ref:
        # Label degeneracy
        ent = rating_entropy_bits(cur["rating"])
        vc = cur["rating"].value_counts(dropna=True)
        top_share_val = float((vc.iloc[0] / vc.sum())) if len(vc) else float("nan")
        single_snapshot = {
            "rating_entropy_bits": ent,
            "unique_ratings": int(cur["rating"].nunique(dropna=True)),
            "top_rating_share": top_share_val,
        }
        report["metrics"]["label_degeneracy"] = single_snapshot

        deg_flags = {
            "low_rating_entropy": (not math.isnan(ent)) and ent < args.min_rating_entropy,
            "single_class_ratings": (not math.isnan(top_share_val)) and top_share_val >= args.max_top_rating_share,
        }
        report["degeneracy_flags"] = deg_flags
        flags.extend(deg_flags.values())

        # Popularity concentration (absolute level)
        pop = movie_popularity_metrics(cur)
        pop_flags = {
            "top10_concentration": (not math.isnan(pop["top10_share"])) and pop["top10_share"] > args.top10_share_threshold,
            "gini_high": (not math.isnan(pop["gini_movies"])) and pop["gini_movies"] > args.gini_movies_threshold,
        }
        report["metrics"]["movie_popularity"] = pop | {"flags": pop_flags}
        flags.extend(pop_flags.values())

        # User activity
        ua = user_activity_metrics(cur)
        ua_flags = {
            "too_many_one_and_done": (not math.isnan(ua["one_and_done_share"])) and ua["one_and_done_share"] > args.one_and_done_threshold
        }
        report["metrics"]["user_activity"] = ua | {"flags": ua_flags}
        flags.extend(ua_flags.values())

        report["drift_detected"] = any(flags)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return 0

    # ---------- Reference-vs-Current ----------
    ref = _read_any(args.ref)
    _ensure_columns(ref, ["user_id", "movie_id", "rating"])
    ref = ref.copy()
    ref["movie_id"] = ref["movie_id"].astype(str)

    # PSI on rating
    psi_rating, rating_edges = psi_from_series(ref["rating"], cur["rating"], bins=10)
    rating_severity = "strong" if (not math.isnan(psi_rating) and psi_rating >= args.psi_strong) else \
                      ("moderate" if (not math.isnan(psi_rating) and psi_rating >= args.psi_moderate) else "none")
    psi_flag = (rating_severity == "strong")
    report["metrics"]["rating_psi"] = {
        "psi": psi_rating,
        "edges": rating_edges,
        "severity": rating_severity,
        "drift_flag": psi_flag
    }
    flags.append(psi_flag)

    # Movie popularity (levels + deltas + delta flags)
    pop_ref = movie_popularity_metrics(ref)
    pop_cur = movie_popularity_metrics(cur)
    del_top10  = float(pop_cur["top10_share"]  - pop_ref.get("top10_share",  float("nan")))
    del_top100 = float(pop_cur["top100_share"] - pop_ref.get("top100_share", float("nan")))
    del_gini   = float(pop_cur["gini_movies"]  - pop_ref.get("gini_movies",  float("nan")))

    pop_delta_flags = {
        "top10_delta_moderate":  (not any(map(math.isnan, [del_top10])))  and abs(del_top10)  >= args.pop_top10_delta_moderate,
        "top10_delta_strong":    (not any(map(math.isnan, [del_top10])))  and abs(del_top10)  >= args.pop_top10_delta_strong,
        "top100_delta_moderate": (not any(map(math.isnan, [del_top100]))) and abs(del_top100) >= args.pop_top100_delta_moderate,
        "top100_delta_strong":   (not any(map(math.isnan, [del_top100]))) and abs(del_top100) >= args.pop_top100_delta_strong,
        "gini_delta_moderate":   (not any(map(math.isnan, [del_gini])))   and abs(del_gini)   >= args.pop_gini_delta_moderate,
        "gini_delta_strong":     (not any(map(math.isnan, [del_gini])))   and abs(del_gini)   >= args.pop_gini_delta_strong,
    }
    pop_level_flags = {
        "top10_concentration": (not math.isnan(pop_cur["top10_share"])) and pop_cur["top10_share"] > args.top10_share_threshold,
        "gini_high":           (not math.isnan(pop_cur["gini_movies"])) and pop_cur["gini_movies"] > args.gini_movies_threshold,
    }

    report["metrics"]["movie_popularity"] = {
        "ref": pop_ref,
        "cur": pop_cur,
        "delta": {
            "top10_share_delta": del_top10,
            "top100_share_delta": del_top100,
            "gini_movies_delta": del_gini,
        },
        "flags": {**pop_level_flags, **pop_delta_flags},
    }
    # Only STRONG deltas or level flags contribute to overall drift
    flags.extend([
        pop_level_flags["top10_concentration"],
        pop_level_flags["gini_high"],
        pop_delta_flags["top10_delta_strong"],
        pop_delta_flags["top100_delta_strong"],
        pop_delta_flags["gini_delta_strong"],
    ])

    # Top-100 overlap
    ref_top = set(ref["movie_id"].value_counts().head(100).index)
    cur_top = set(cur["movie_id"].value_counts().head(100).index)
    overlap = float(len(ref_top & cur_top) / 100.0) if (len(ref_top) >= 100 and len(cur_top) >= 100) else float("nan")
    overlap_flag = (not math.isnan(overlap)) and overlap < args.top100_overlap_threshold
    report["metrics"]["top100_overlap"] = {"overlap": overlap, "flag_low_overlap": overlap_flag}
    flags.append(overlap_flag)

    # User activity deltas
    ua_ref = user_activity_metrics(ref)
    ua_cur = user_activity_metrics(cur)
    ua_delta = {
        "active_users_delta": float(ua_cur["active_users"] - ua_ref["active_users"]),
        "one_and_done_share_delta": float(ua_cur["one_and_done_share"] - ua_ref["one_and_done_share"]),
        "p95_events_per_user_delta": float(ua_cur["p95_events_per_user"] - ua_ref["p95_events_per_user"]),
    }
    ua_flags = {
        "too_many_one_and_done": (not math.isnan(ua_cur["one_and_done_share"])) and ua_cur["one_and_done_share"] > args.one_and_done_threshold
    }
    report["metrics"]["user_activity"] = {"ref": ua_ref, "cur": ua_cur, "delta": ua_delta, "flags": ua_flags}
    flags.extend(ua_flags.values())

    report["drift_detected"] = any(flags)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
