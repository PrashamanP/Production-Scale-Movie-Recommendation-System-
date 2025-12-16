from __future__ import annotations
import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import pandas as pd
from jsonschema import Draft202012Validator
import pyarrow as pa
import pyarrow.parquet as pq


# --------------------------- Helpers ---------------------------

def _load_schema(schema_path: str) -> Dict[str, Any]:
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _coerce_and_repair_row(row: Dict[str, Any]) -> Dict[str, Any]:
    # user_id
    v = row.get("user_id")
    if v is not None and v != "":
        try:
            row["user_id"] = int(v)
        except Exception:
            pass
    # movie_id is a slug/string (do not coerce to int)
    v = row.get("movie_id")
    if v is not None:
        row["movie_id"] = str(v)

    # rating (clip to [1,5])
    v = row.get("rating")
    if v is not None and v != "":
        try:
            row["rating"] = int(float(v))
            if row["rating"] < 1:
                row["rating"] = 1
            if row["rating"] > 5:
                row["rating"] = 5
        except Exception:
            pass

    # timestamp (integer seconds; allow missing if not in schema required)
    v = row.get("timestamp")
    if v is not None and v != "":
        try:
            row["timestamp"] = int(float(v))
            if row["timestamp"] < 0:
                row["timestamp"] = 0
        except Exception:
            pass
    return row

def _validate_chunk(
    df: pd.DataFrame,
    schema: Dict[str, Any],
    cols: List[str],
    validator: Draft202012Validator
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    good: List[Dict[str, Any]] = []
    rejects: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        row = _coerce_and_repair_row({k: r.get(k) for k in df.columns})
        cand = {k: row.get(k) for k in cols}
        errs = list(validator.iter_errors(cand))
        if errs:
            rejects.append({"row": row, "errors": [e.message for e in errs]})
        else:
            good.append(cand)
    return good, rejects

def _to_table(rows: List[Dict[str, Any]], cols: List[str]) -> pa.Table:
    if not rows:
        return pa.Table.from_pydict({c: [] for c in cols})
    cols_dict = {c: [r.get(c) for r in rows] for c in cols}
    return pa.Table.from_pydict(cols_dict)


# ------------------ Dedupe engines (post-validate) ------------------

def _run_duckdb_dedupe(src_parquet: str, dst_parquet: str, keys: List[str], strategy: str) -> Tuple[int, int]:
    import duckdb
    con = duckdb.connect()
    # Register parquet as a view without loading to RAM
    con.execute(f"CREATE VIEW v AS SELECT * FROM parquet_scan('{src_parquet}')")
    # Does it have a timestamp?
    has_ts = bool(
        con.execute(
            "SELECT COUNT(*) FROM information_schema.columns "
            "WHERE table_name='v' AND column_name='timestamp'"
        ).fetchone()[0]
    )
    if strategy == "latest_by_timestamp" and not has_ts:
        strategy = "last_seen"  # degrade gracefully

    key_cols = ", ".join(keys)
    if strategy == "latest_by_timestamp":
        order = "timestamp DESC NULLS LAST"
    elif strategy == "first_seen":
        order = "1"  # input order; acceptable stand-in
    elif strategy == "last_seen":
        order = "1 DESC"
    else:
        raise ValueError(f"Unknown dedupe_strategy: {strategy}")

    con.execute(f"""
        CREATE TABLE _dedup AS
        SELECT * FROM (
            SELECT *,
                   ROW_NUMBER() OVER (PARTITION BY {key_cols} ORDER BY {order}) AS _rn
            FROM v
        )
        WHERE _rn = 1
    """)
    in_rows = con.execute("SELECT COUNT(*) FROM v").fetchone()[0]
    out_rows = con.execute("SELECT COUNT(*) FROM _dedup").fetchone()[0]
    # Write parquet
    con.execute(f"COPY (SELECT * FROM _dedup) TO '{dst_parquet}' (FORMAT PARQUET)")
    con.close()
    return int(in_rows), int(out_rows)

def _run_pandas_dedupe(src_parquet: str, dst_parquet: str, keys: List[str], strategy: str) -> Tuple[int, int]:
    df = pd.read_parquet(src_parquet)
    in_rows = len(df)
    if strategy == "latest_by_timestamp" and "timestamp" in df.columns:
        df = df.sort_values("timestamp").drop_duplicates(subset=keys, keep="last")
    elif strategy == "first_seen":
        df = df.drop_duplicates(subset=keys, keep="first")
    elif strategy == "last_seen":
        df = df.drop_duplicates(subset=keys, keep="last")
    else:
        df = df.drop_duplicates(subset=keys, keep="last")
    df.to_parquet(dst_parquet, index=False)
    out_rows = len(df)
    return int(in_rows), int(out_rows)


# ------------------------------ Main ------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Chunked schema validation + optional de-dup for interactions")
    ap.add_argument("--in", dest="src", required=True, help="Input interactions file (csv|parquet|jsonl)")
    ap.add_argument("--schema", required=True, help="JSON schema for rows")
    ap.add_argument("--out", required=True, help="Output CLEAN parquet path")
    ap.add_argument("--report", required=True, help="JSON report path")
    ap.add_argument("--rejects_out", default=None, help="Optional parquet for rejected rows")
    ap.add_argument("--chunksize", type=int, default=250_000, help="Rows per chunk for CSV/JSON inputs")
    ap.add_argument("--max_reject_rate", type=float, default=0.05, help="Exit code 1 if rejects/input > this")
    ap.add_argument("--verbose", action="store_true", help="Print per-chunk progress")

    # De-dup options
    ap.add_argument("--dedupe", action="store_true", help="Run de-duplication AFTER validation")
    ap.add_argument("--dedupe_out", default=None, help="Output parquet for deduped data (required with --dedupe)")
    ap.add_argument("--dedupe_keys", default="user_id,movie_id", help="Comma-separated keys (default: user_id,movie_id)")
    ap.add_argument("--dedupe_strategy", default="latest_by_timestamp",
                    choices=["latest_by_timestamp", "first_seen", "last_seen"],
                    help="Which row to keep per key")
    args = ap.parse_args()

    # Ensure dirs
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    if args.rejects_out:
        os.makedirs(os.path.dirname(args.rejects_out), exist_ok=True)
    if args.dedupe:
        if not args.dedupe_out:
            print("ERROR: --dedupe_out is required when --dedupe is set", file=sys.stderr)
            return 2
        os.makedirs(os.path.dirname(args.dedupe_out), exist_ok=True)

    # Load schema & validator
    try:
        schema = _load_schema(args.schema)
        cols = list(schema.get("properties", {}).keys())
        validator = Draft202012Validator(schema)
    except Exception as e:
        print(f"ERROR: failed to load/parse schema: {e}", file=sys.stderr)
        return 2

    # Choose reader (chunked for CSV/JSON)
    lower = args.src.lower()
    try:
        if lower.endswith(".parquet"):
            chunks = [pd.read_parquet(args.src)]  # single chunk
        elif lower.endswith(".json") or lower.endswith(".jsonl"):
            chunks = pd.read_json(args.src, lines=True, chunksize=args.chunksize)
        else:
            chunks = pd.read_csv(args.src, chunksize=args.chunksize)
    except Exception as e:
        print(f"ERROR: failed to open input: {e}", file=sys.stderr)
        return 2

    # Prepare parquet writers
    clean_writer: pq.ParquetWriter | None = None
    rejects_writer: pq.ParquetWriter | None = None

    def _open_clean_writer(table: pa.Table):
        nonlocal clean_writer
        if clean_writer is None:
            clean_writer = pq.ParquetWriter(args.out, table.schema)

    def _open_rejects_writer(table: pa.Table):
        nonlocal rejects_writer
        if args.rejects_out and rejects_writer is None:
            rejects_writer = pq.ParquetWriter(args.rejects_out, table.schema)

    # Stats
    total_in = total_clean = total_reject = 0
    t0 = time.time()

    # Process chunks
    chunk_idx = 0
    for chunk in chunks:
        chunk_idx += 1
        in_rows = len(chunk)
        total_in += in_rows

        good, rejects = _validate_chunk(chunk, schema, cols, validator)

        tbl = _to_table(good, cols)
        if tbl.num_rows > 0:
            _open_clean_writer(tbl)
            clean_writer.write_table(tbl)
        total_clean += tbl.num_rows

        if args.rejects_out and rejects:
            # Build a superset of columns present in rejected rows
            all_keys = set().union(*[r["row"].keys() for r in rejects])
            rej_rows = [{k: r["row"].get(k) for k in all_keys} for r in rejects]
            rej_tbl = pa.Table.from_pydict({k: [r.get(k) for r in rej_rows] for k in sorted(all_keys)})
            _open_rejects_writer(rej_tbl)
            rejects_writer.write_table(rej_tbl)
        total_reject += len(rejects)

        if args.verbose:
            elapsed = time.time() - t0
            rate = total_in / max(1.0, elapsed)
            print(
                f"[chunk {chunk_idx}] in={in_rows}  clean+={tbl.num_rows}  reject+={len(rejects)}  "
                f"totals in={total_in} clean={total_clean} reject={total_reject}  {rate:,.0f} rows/s"
            )

    # Close writers
    if clean_writer is not None:
        clean_writer.close()
    if rejects_writer is not None:
        rejects_writer.close()

    # Base report
    report = {
        "input_path": args.src,
        "output_path": args.out,
        "rejects_path": args.rejects_out,
        "schema_path": args.schema,
        "input_rows": int(total_in),
        "clean_rows": int(total_clean),
        "reject_rows": int(total_reject),
        "reject_rate": float(total_reject / max(1, total_in)) if total_in else 0.0,
        "elapsed_sec": round(time.time() - t0, 2),
        "chunksize": args.chunksize
    }

    # Optional de-duplication
    if args.dedupe:
        keys = [k.strip() for k in args.dedupe_keys.split(",") if k.strip()]
        try:
            try:
                import duckdb  # noqa: F401
                engine = "duckdb"
                in_rows, out_rows = _run_duckdb_dedupe(args.out, args.dedupe_out, keys, args.dedupe_strategy)
            except ImportError:
                engine = "pandas"
                in_rows, out_rows = _run_pandas_dedupe(args.out, args.dedupe_out, keys, args.dedupe_strategy)

            report.update({
                "dedupe": True,
                "dedupe_engine": engine,
                "dedupe_keys": keys,
                "dedupe_strategy": args.dedupe_strategy,
                "dedupe_rows_in": in_rows,
                "dedupe_rows_out": out_rows,
                "dedupe_drop_fraction": float((in_rows - out_rows) / max(1, in_rows)),
                "dedupe_output_path": args.dedupe_out
            })
        except Exception as e:
            report.update({"dedupe": True, "dedupe_error": str(e)})

    # Exit code policy: only based on schema reject rate (M2 does not require failing on drift)
    rc = 1 if (total_in > 0 and (total_reject / total_in) > args.max_reject_rate) else 0

    # Write report
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
