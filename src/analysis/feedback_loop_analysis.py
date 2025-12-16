"""
Detect signs of popularity-based feedback loops in recommendation logs.

The script buckets recommendation and watch events into fixed windows,
then measures whether items heavily recommended in one window gain watch
share in the following window (relative to the prior window). A large
positive delta suggests reinforcement of popularity.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from src.evaluation.online_als import (
    RECO_LOG_PATTERN,
    WATCH_LOG_PATTERN,
    parse_timestamp,
)


@dataclass
class WindowStats:
    start: datetime
    end: datetime
    coverage: int
    gini: float
    rec_count: int
    watch_count: int


def gini_from_counter(counter: Counter) -> float:
    """Compute the Gini coefficient for a Counter of counts."""
    if not counter:
        return 0.0
    values = sorted(counter.values())
    n = len(values)
    total = float(sum(values))
    if total == 0:
        return 0.0
    cum_weighted = sum((i + 1) * v for i, v in enumerate(values))
    return (2 * cum_weighted) / (n * total) - (n + 1) / n


def stream_events(log_file: Path) -> Tuple[List[Tuple[datetime, List[str]]], List[Tuple[datetime, str]]]:
    """Parse the raw log file into recommendation and watch event timelines."""
    rec_events: List[Tuple[datetime, List[str]]] = []
    watch_events: List[Tuple[datetime, str]] = []

    with log_file.open("r", encoding="utf-8") as f:
        for line in f:
            rec_match = RECO_LOG_PATTERN.search(line)
            if rec_match:
                ts = parse_timestamp(rec_match.group("timestamp"))
                movie_csv = rec_match.group("movies")
                if ts and movie_csv:
                    movies = [m.strip() for m in movie_csv.split(",") if m.strip()]
                    rec_events.append((ts, movies))
                continue

            watch_match = WATCH_LOG_PATTERN.search(line)
            if watch_match:
                ts_str, _, movie_id = watch_match.groups()
                ts = parse_timestamp(ts_str)
                if ts:
                    watch_events.append((ts, movie_id))

    return rec_events, watch_events


def build_windows(
    rec_events: Sequence[Tuple[datetime, List[str]]],
    watch_events: Sequence[Tuple[datetime, str]],
    window_hours: int,
) -> Tuple[List[Counter], List[Counter], List[datetime]]:
    """Bucket recommendation and watch events into fixed windows."""
    if not rec_events and not watch_events:
        return [], [], []

    all_times = [ts for ts, _ in rec_events] + [ts for ts, _ in watch_events]
    start = min(all_times)
    end = max(all_times)

    # Normalize to UTC to avoid timezone math surprises.
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    window_delta = timedelta(hours=window_hours)
    windows: List[datetime] = []
    current = start
    while current <= end:
        windows.append(current)
        current += window_delta

    rec_counts = [Counter() for _ in range(len(windows))]
    watch_counts = [Counter() for _ in range(len(windows))]

    def window_index(ts: datetime) -> int:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return int((ts - start) / window_delta)

    for ts, movies in rec_events:
        idx = window_index(ts)
        if 0 <= idx < len(rec_counts):
            rec_counts[idx].update(movies)

    for ts, movie_id in watch_events:
        idx = window_index(ts)
        if 0 <= idx < len(watch_counts):
            watch_counts[idx][movie_id] += 1

    return rec_counts, watch_counts, windows


def analyze_popularity_drift(
    rec_counts: List[Counter],
    watch_counts: List[Counter],
    windows: List[datetime],
    top_k: int,
) -> Dict:
    """
    For each middle window, compare watch share of top-K recommended movies
    in the following window versus the previous window.
    """
    per_window = []
    deltas: List[float] = []
    for i in range(1, len(windows) - 1):
        rec_counter = rec_counts[i]
        if not rec_counter:
            continue

        top_movies = [m for m, _ in rec_counter.most_common(top_k)]
        prev_total_watches = sum(watch_counts[i - 1].values())
        next_total_watches = sum(watch_counts[i + 1].values())

        prev_share = (
            sum(watch_counts[i - 1][m] for m in top_movies) / prev_total_watches
            if prev_total_watches
            else 0.0
        )
        next_share = (
            sum(watch_counts[i + 1][m] for m in top_movies) / next_total_watches
            if next_total_watches
            else 0.0
        )
        delta = next_share - prev_share
        deltas.append(delta)

        per_window.append(
            {
                "window_index": i,
                "window_start": windows[i].isoformat(),
                "prev_watch_share_topk": prev_share,
                "next_watch_share_topk": next_share,
                "delta_watch_share": delta,
                "topk": top_k,
                "unique_recs": len(rec_counter),
            }
        )

    avg_delta = sum(deltas) / len(deltas) if deltas else 0.0
    return {
        "avg_delta_watch_share": avg_delta,
        "window_results": per_window,
    }


def summarize_windows(
    rec_counts: List[Counter],
    watch_counts: List[Counter],
    windows: List[datetime],
) -> List[WindowStats]:
    """Produce per-window summary stats for coverage and concentration."""
    stats: List[WindowStats] = []
    for i, start in enumerate(windows):
        end = windows[i + 1] if i + 1 < len(windows) else start
        rec_counter = rec_counts[i] if i < len(rec_counts) else Counter()
        watch_counter = watch_counts[i] if i < len(watch_counts) else Counter()
        stats.append(
            WindowStats(
                start=start,
                end=end,
                coverage=len(rec_counter),
                gini=gini_from_counter(rec_counter),
                rec_count=sum(rec_counter.values()),
                watch_count=sum(watch_counter.values()),
            )
        )
    return stats


def run_analysis(
    log_file: Path,
    window_hours: int,
    top_k: int,
) -> Dict:
    rec_events, watch_events = stream_events(log_file)
    rec_counts, watch_counts, windows = build_windows(rec_events, watch_events, window_hours)
    summary = summarize_windows(rec_counts, watch_counts, windows)
    popularity = analyze_popularity_drift(rec_counts, watch_counts, windows, top_k=top_k)

    mean_coverage = (
        sum(stat.coverage for stat in summary) / len(summary) if summary else 0.0
    )
    mean_gini = sum(stat.gini for stat in summary) / len(summary) if summary else 0.0

    return {
        "log_file": str(log_file),
        "window_hours": window_hours,
        "top_k": top_k,
        "num_windows": len(windows),
        "mean_unique_recs": mean_coverage,
        "mean_gini_recs": mean_gini,
        "popularity_reinforcement": popularity,
        "window_summaries": [asdict(stat) for stat in summary],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze feedback loops by measuring popularity reinforcement across time windows."
    )
    parser.add_argument(
        "--log_file",
        required=True,
        type=Path,
        help="Path to Kafka-style combined log (contains recommendation and watch events).",
    )
    parser.add_argument(
        "--window_hours",
        type=float,
        default=12,
        help="Size of the time window used to bucket events.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-K recommended items considered when computing watch share deltas.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/feedback_loop_analysis.json"),
        help="Where to write the JSON report.",
    )

    args = parser.parse_args()

    if args.window_hours <= 0:
        raise ValueError(f"window_hours must be > 0 (got {args.window_hours})")

    if not args.log_file.exists():
        raise FileNotFoundError(f"Log file not found: {args.log_file}")

    report = run_analysis(
        log_file=args.log_file,
        window_hours=args.window_hours,
        top_k=args.top_k,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"Wrote feedback loop analysis to {args.out}")
    print(json.dumps(report["popularity_reinforcement"], indent=2))


if __name__ == "__main__":
    main()
