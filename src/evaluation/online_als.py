import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Dict, List, Optional, Set, Tuple

# --- Log Parsing Constants ---

# Regex to capture successful recommendation events (status 200)
# Format: <time>,<userid>,recommendation request <server>, status 200[, variant=<name>], result: <movie_ids_csv>, <responsetime>
# Example: 2024-10-25T10:00:00Z,12345,recommendation request /... status 200, variant=als_model, result: m101,m202,m303, 55 ms
RECO_LOG_PATTERN = re.compile(
    r"^(?P<timestamp>\S+),"
    r"(?P<user_id>\d+),"
    r"recommendation request.*"
    r"status 200"
    r"(?:, variant=(?P<variant>[A-Za-z0-9_\-\.]+))?"
    r"(?:, bucket=(?P<bucket>[0-9]*\.?[0-9]+))?"
    r", result: "
    r"(?P<movies>.*),"
    r" (?P<response_time>\d+) ms$"
)

# Regex to capture movie watch events
# Format: <time>,<userid>,GET /data/m/<movieid>/<minute>.mpg
# Example: 2024-10-25T10:05:00Z,12345,GET /data/m/m202/1.mpg
WATCH_LOG_PATTERN = re.compile(
    r"(^\S+),"                          # 1: Timestamp (ISO 8601)
    r"(\d+),"                          # 2: User ID
    r"GET /data/m/"                    # ...
    r"([^/]+)/"                        # 3: Movie ID (match everything until next '/')
)

# --- Type Definitions ---

# A (timestamp, set_of_movie_ids, variant_name) tuple
RecoEvent = Tuple[datetime, Set[str], str]
# A (timestamp, movie_id) tuple
WatchEvent = Tuple[datetime, str]

# {user_id: [RecoEvent, ...]}
RecsByUser = Dict[str, List[RecoEvent]]
# {user_id: [WatchEvent, ...]}
WatchesByUser = Dict[str, List[WatchEvent]]


def parse_timestamp(ts_str: str) -> Optional[datetime]:
    """
    Parses an ISO 8601 timestamp string.
    Handles potential 'Z' suffix and high-precision fractional seconds.
    """
    try:
        # 1. Truncate fractional seconds (e.g., .146597063)
        if '.' in ts_str:
             ts_str = ts_str.split('.')[0]
        
        # 2. Handle 'Z' suffix if it's now at the end
        if ts_str.endswith('Z'):
            ts_str = ts_str[:-1] + "+00:00"
        
        # 3. Handle cases where there's no timezone info (like the new log)
        #    This will be parsed as a "naive" datetime object.
        ts = datetime.fromisoformat(ts_str)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts
    except ValueError:
        # Attempt to salvage obvious typos (extra letters/digits) before giving up.
        cleaned = re.sub(r"[^0-9T:\-]", "", ts_str)
        match = re.match(
            r"(?P<year>\d{4})-?(?P<month>\d{2})-?(?P<day>\d{2})T?(?P<hour>\d{2}):?(?P<minute>\d{2}):?(?P<second>\d{2})",
            cleaned,
        )
        if match:
            fixed = "{year}-{month}-{day}T{hour}:{minute}:{second}".format(**match.groupdict())
            try:
                ts = datetime.fromisoformat(fixed)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                return ts
            except ValueError:
                pass

        print(f"Warning: Could not parse timestamp '{ts_str}'")
        return None

def parse_log_file(
    log_file_path: Path,
) -> Tuple[RecsByUser, WatchesByUser, int]:
    """
    Parses a log file line-by-line and extracts recommendation/watch events.
    
    Returns:
        - recs_by_user: A dict mapping user IDs to their recommendation events.
        - watches_by_user: A dict mapping user IDs to their watch events.
        - total_lines: Total lines processed for context.
    """
    recs_by_user: RecsByUser = defaultdict(list)
    watches_by_user: WatchesByUser = defaultdict(list)
    
    total_lines = 0
    parsed_recos = 0
    parsed_watches = 0

    print(f"Parsing log file: {log_file_path}...")
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            total_lines += 1
            
            # 1. Check for a Recommendation Event
            reco_match = RECO_LOG_PATTERN.search(line)
            if reco_match:
                ts_str = reco_match.group("timestamp")
                user_id = reco_match.group("user_id")
                movie_csv = reco_match.group("movies")
                variant = reco_match.group("variant") or "unknown"
                ts = parse_timestamp(ts_str)
                if not ts or not movie_csv:
                    continue  # Skip if timestamp is bad or result list is empty
                
                # Split by comma and strip whitespace from each movie ID
                movie_set = set(m.strip() for m in movie_csv.split(','))
                recs_by_user[user_id].append((ts, movie_set, variant))
                parsed_recos += 1
                continue # A line can only be one type of event
            
            # 2. Check for a Watch Event
            watch_match = WATCH_LOG_PATTERN.search(line)
            if watch_match:
                ts_str, user_id, movie_id = watch_match.groups()
                ts = parse_timestamp(ts_str)
                if not ts:
                    continue # Skip if timestamp is bad
                
                watches_by_user[user_id].append((ts, movie_id))
                parsed_watches += 1

    print(f"Done parsing. Processed {total_lines} lines.")
    print(f"Found {parsed_recos} recommendation events for {len(recs_by_user)} users.")
    print(f"Found {parsed_watches} watch events for {len(watches_by_user)} users.")
    return recs_by_user, watches_by_user, total_lines


def calculate_watch_through_rate(
    recs_by_user: RecsByUser,
    watches_by_user: WatchesByUser,
    attribution_window_hours: int = 1,
) -> Tuple[int, int, float, Dict[str, Dict[str, float]]]:
    """
    Calculates the Recommendation Watch-Through Rate (WTR).

    Metric Definition:
    - Numerator: Number of successful recommendation events that were "converted".
    - Denominator: Total number of successful recommendation events.
    - Conversion: A recommendation event is "converted" if the user watches
                  *any* movie from the recommended list within the
                  attribution window (e.g., 1 hour) *after* the
                  recommendation was served.
    """
    print(f"Calculating WTR with a {attribution_window_hours}-hour attribution window...")
    
    attribution_window = timedelta(hours=attribution_window_hours)
    
    total_recommendation_events = 0
    converted_recommendation_events = 0
    variant_counters: Dict[str, List[int]] = defaultdict(lambda: [0, 0])

    # Iterate through every single recommendation event we logged
    for user_id, reco_events in recs_by_user.items():
        if user_id not in watches_by_user:
            # This user received recs but watched nothing.
            total_recommendation_events += len(reco_events)
            continue
            
        user_watches = watches_by_user[user_id]
        
        for reco_ts, reco_movie_set, variant_name in reco_events:
            total_recommendation_events += 1
            variant_counters[variant_name][0] += 1
            is_converted = False
            
            # Check all of this user's watches against this single reco event
            for watch_ts, watched_movie_id in user_watches:
                
                # Check 1: Did the user watch a movie from the recommended set?
                if watched_movie_id in reco_movie_set:
                    
                    # Check 2: Did the watch happen *after* the recommendation?
                    if watch_ts > reco_ts:
                        
                        # Check 3: Did it happen within the attribution window?
                        if (watch_ts - reco_ts) <= attribution_window:
                            is_converted = True
                            break # This reco event is converted, move to the next reco event
            
            if is_converted:
                converted_recommendation_events += 1
                variant_counters[variant_name][1] += 1

    variant_breakdown: Dict[str, Dict[str, float]] = {}
    for variant_name, (total, converted) in variant_counters.items():
        rate = converted / total if total else 0.0
        variant_breakdown[variant_name] = {
            "total_recommendation_events": total,
            "converted_recommendation_events": converted,
            "watch_through_rate": rate,
        }

    print(f"Total recommendation events: {total_recommendation_events}")
    print(f"Converted recommendation events: {converted_recommendation_events}")
    
    wtr = 0.0
    if total_recommendation_events > 0:
        wtr = converted_recommendation_events / total_recommendation_events
        
    print(f"Online WTR@{attribution_window_hours}h: {wtr:.4f}")
    for variant_name, stats in variant_breakdown.items():
        print(
            f"  Variant '{variant_name}': "
            f"{int(stats['converted_recommendation_events'])}/"
            f"{int(stats['total_recommendation_events'])} "
            f"({stats['watch_through_rate']:.4f})"
        )

    return total_recommendation_events, converted_recommendation_events, wtr, variant_breakdown


def two_proportion_z_test(
    success_a: int,
    total_a: int,
    success_b: int,
    total_b: int,
    alpha: float = 0.05,
) -> Optional[Dict[str, float]]:
    """
    Performs a two-proportion z-test comparing success rates of two variants.

    Returns None if the test cannot be computed (e.g., zero totals).
    """
    if total_a <= 0 or total_b <= 0:
        return None

    p1 = success_a / total_a
    p2 = success_b / total_b
    pooled = (success_a + success_b) / (total_a + total_b)
    standard_error = (pooled * (1 - pooled) * (1 / total_a + 1 / total_b)) ** 0.5

    if standard_error == 0:
        return None

    z_score = (p1 - p2) / standard_error
    normal = NormalDist()
    p_value = 2 * (1 - normal.cdf(abs(z_score)))

    z_critical = normal.inv_cdf(1 - alpha / 2)
    margin_of_error = z_critical * ((p1 * (1 - p1) / total_a + p2 * (1 - p2) / total_b) ** 0.5)
    diff = p1 - p2

    return {
        "z_score": z_score,
        "p_value": p_value,
        "effect_size": diff,
        "confidence_interval": [diff - margin_of_error, diff + margin_of_error],
        "alpha": alpha,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate online WTR from production logs."
    )
    parser.add_argument(
        "log_file",
        type=Path,
        help="Path to the Kafka log file (e.g., /logs/movielog-2024-10-25.log)",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=Path("artifacts/online_evaluation.json"),
        help="Path to save the JSON evaluation results.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=1,
        help="Attribution window in hours.",
    )
    parser.add_argument(
        "--variant-a",
        type=str,
        help="Name of the first variant to compare in the z-test.",
    )
    parser.add_argument(
        "--variant-b",
        type=str,
        help="Name of the second variant to compare in the z-test.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level used for the z-test confidence interval.",
    )
    args = parser.parse_args()

    if not args.log_file.exists():
        print(f"Error: Log file not found at {args.log_file}")
        return

    # 1. Parse all relevant events from the log file
    recs_by_user, watches_by_user, total_lines = parse_log_file(args.log_file)
    
    # 2. Calculate the metric
    total_recos, converted_recos, wtr, variant_details = calculate_watch_through_rate(
        recs_by_user,
        watches_by_user,
        attribution_window_hours=args.window,
    )

    z_test_payload = None
    chosen_variants: Optional[Tuple[str, str]] = None
    if len(variant_details) >= 2:
        if args.variant_a and args.variant_b:
            chosen_variants = (args.variant_a, args.variant_b)
        else:
            # Pick the two variants with the most traffic
            sorted_variants = sorted(
                variant_details.items(),
                key=lambda item: item[1]["total_recommendation_events"],
                reverse=True,
            )
            chosen_variants = (sorted_variants[0][0], sorted_variants[1][0])

    if chosen_variants:
        variant_a, variant_b = chosen_variants
        stats_a = variant_details.get(variant_a)
        stats_b = variant_details.get(variant_b)
        if stats_a and stats_b:
            z_test_payload = two_proportion_z_test(
                success_a=int(stats_a["converted_recommendation_events"]),
                total_a=int(stats_a["total_recommendation_events"]),
                success_b=int(stats_b["converted_recommendation_events"]),
                total_b=int(stats_b["total_recommendation_events"]),
                alpha=args.alpha,
            )
            if z_test_payload:
                z_test_payload["variant_a"] = variant_a
                z_test_payload["variant_b"] = variant_b
                print(
                    f"\nZ-test ({variant_a} vs {variant_b}): "
                    f"z={z_test_payload['z_score']:.3f}, "
                    f"p={z_test_payload['p_value']:.4f}, "
                    f"effect={z_test_payload['effect_size']:.4f}"
                )
            else:
                print(f"\nInsufficient data to compute z-test for {variant_a} vs {variant_b}.")
    
    # 3. Save results to a JSON file
    results = {
        "metric": "Recommendation Watch-Through Rate (WTR)",
        "log_file_processed": str(args.log_file),
        "total_log_lines": total_lines,
        "attribution_window_hours": args.window,
        "total_recommendation_events": total_recos,
        "converted_recommendation_events": converted_recos,
        "watch_through_rate": wtr,
        "variant_breakdown": variant_details,
        "calculation_timestamp": datetime.utcnow().isoformat() + "Z",
    }

    if z_test_payload:
        results["z_test"] = z_test_payload
    
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\nSuccessfully saved online evaluation results to {args.output_file}")


if __name__ == "__main__":
    main()
