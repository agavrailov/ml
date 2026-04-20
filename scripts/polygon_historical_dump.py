"""One-time Polygon.io historical minute-bar dump for all portfolio symbols.

Deletes existing raw CSVs for the requested symbols before downloading so
the result is a clean, broadcast-free baseline (no YF contamination).

Usage:
    python scripts/polygon_historical_dump.py
    python scripts/polygon_historical_dump.py --symbols NVDA JPM
    python scripts/polygon_historical_dump.py --start 2022-01-01 --end 2026-04-20

Environment variables (required):
    POLYGON_API_KEY  — Polygon.io API key (free Stocks Starter plan is enough)
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config_resolver import get_configured_symbols
from src.config import get_raw_data_csv_path
from src.ingestion.polygon_historical import download_minute_history


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download full minute-bar history from Polygon.io for all portfolio symbols."
    )
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="Tickers to download. Defaults to configs/portfolio.json.",
    )
    parser.add_argument("--start", default="2022-01-01", help="Start date YYYY-MM-DD.")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD. Default: today.")
    parser.add_argument(
        "--method", default=None, choices=["api", "flatfiles"],
        help=(
            "api: REST endpoint, works on free plan (default). "
            "flatfiles: S3 direct download, much faster, requires paid plan."
        ),
    )
    parser.add_argument(
        "--unadjusted", action="store_true",
        help="Fetch raw prices instead of split-adjusted (not recommended).",
    )
    parser.add_argument(
        "--no-delete", action="store_true",
        help="Skip deleting existing raw CSVs (appends instead of replacing).",
    )
    args = parser.parse_args()

    api_key = os.environ.get("POLYGON_API_KEY", "91ca9428-893d-47de-aa49-9ef70f014c66")
    if not api_key:
        print("ERROR: Set POLYGON_API_KEY environment variable first.", file=sys.stderr)
        sys.exit(1)

    symbols = args.symbols or get_configured_symbols()
    start_d = date.fromisoformat(args.start)
    end_d = date.fromisoformat(args.end) if args.end else None
    adjusted = not args.unadjusted
    method = args.method or os.environ.get("POLYGON_METHOD", "api")

    print(f"Symbols    : {symbols}")
    print(f"Date range : {args.start} -> {args.end or 'today'}")
    print(f"Adjusted   : {adjusted}")
    print(f"Method     : {method}")
    print()

    for sym in symbols:
        raw_csv = get_raw_data_csv_path(sym)
        if not args.no_delete and os.path.exists(raw_csv):
            os.remove(raw_csv)
            print(f"[dump] Deleted old raw CSV: {raw_csv}")

        try:
            download_minute_history(
                sym, start=start_d, end=end_d,
                adjusted=adjusted, method=method, api_key=api_key,
            )
        except Exception as exc:
            print(f"[dump] ERROR for {sym}: {exc}", file=sys.stderr)

    print()
    print("Done. Next: rebuild processed CSVs:")
    print("  python src/daily_data_agent.py --skip-ingestion")


if __name__ == "__main__":
    main()
