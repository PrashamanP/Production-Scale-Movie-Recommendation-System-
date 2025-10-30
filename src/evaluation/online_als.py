import re
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

# --- Log Parsing Constants ---

# Regex to capture successful recommendation events (status 200)
# Format: <time>,<userid>,recommendation request <server>, status 200, result: <movie_ids_csv>, <responsetime>
# Example: 2024-10-25T10:00:00Z,12345,recommendation request /... status 200, result: m101,m202,m303, 55 ms
RECO_LOG_PATTERN = re.compile(
    r"^(\S+),"                          # 1: Timestamp (ISO 8601)
    r"(\d+),"                          # 2: User ID
    r"recommendation request.*"        # ...
    r"status 200, result: "            # ...
    r"(.*),"                           # 3: Movie IDs list (greedy capture)
    r" (\d+) ms$"                      # 4: Response time (captured but unused)
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

# A (timestamp, set_of_movie_ids) tuple
RecoEvent = Tuple[datetime, Set[str]]
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
        return datetime.fromisoformat(ts_str)
    except ValueError:
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
    with open(log_file_path, 'r') as f:
        for line in f:
            total_lines += 1
            
            # 1. Check for a Recommendation Event
            reco_match = RECO_LOG_PATTERN.search(line)
            if reco_match:
                # We capture 4 groups: ts, user_id, movie_csv, response_time
                ts_str, user_id, movie_csv, _ = reco_match.groups()
                ts = parse_timestamp(ts_str)
                if not ts or not movie_csv:
                    continue  # Skip if timestamp is bad or result list is empty
                
                # Split by comma and strip whitespace from each movie ID
                movie_set = set(m.strip() for m in movie_csv.split(','))
                recs_by_user[user_id].append((ts, movie_set))
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
) -> Tuple[int, int, float]:
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

    # Iterate through every single recommendation event we logged
    for user_id, reco_events in recs_by_user.items():
        if user_id not in watches_by_user:
            # This user received recs but watched nothing.
            total_recommendation_events += len(reco_events)
            continue
            
        user_watches = watches_by_user[user_id]
        
        for reco_ts, reco_movie_set in reco_events:
            total_recommendation_events += 1
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

    print(f"Total recommendation events: {total_recommendation_events}")
    print(f"Converted recommendation events: {converted_recommendation_events}")
    
    wtr = 0.0
    if total_recommendation_events > 0:
        wtr = converted_recommendation_events / total_recommendation_events
        
    print(f"Online WTR@{attribution_window_hours}h: {wtr:.4f}")
    return total_recommendation_events, converted_recommendation_events, wtr


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
    args = parser.parse_args()

    if not args.log_file.exists():
        print(f"Error: Log file not found at {args.log_file}")
        return

    # 1. Parse all relevant events from the log file
    recs_by_user, watches_by_user, total_lines = parse_log_file(args.log_file)
    
    # 2. Calculate the metric
    total_recos, converted_recos, wtr = calculate_watch_through_rate(
        recs_by_user,
        watches_by_user,
        attribution_window_hours=args.window,
    )
    
    # 3. Save results to a JSON file
    results = {
        "metric": "Recommendation Watch-Through Rate (WTR)",
        "log_file_processed": str(args.log_file),
        "total_log_lines": total_lines,
        "attribution_window_hours": args.window,
        "total_recommendation_events": total_recos,
        "converted_recommendation_events": converted_recos,
        "watch_through_rate": wtr,
        "calculation_timestamp": datetime.utcnow().isoformat() + "Z",
    }
    
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\nSuccessfully saved online evaluation results to {args.output_file}")


if __name__ == "__main__":
    main()
