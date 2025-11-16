"""Utility to remove previously forward-filled ranges from RAW_DATA_CSV.

This is intended for one-off cleanup when you want to re-attempt filling long
weekday gaps from TWS instead of relying on synthetic forward-filled data.

Usage (from repo root):

    python cleanup_forward_filled_ranges.py \
        --gaps-json data/processed/missing_trading_days.json

By default it uses RAW_DATA_CSV and PROCESSED_DATA_DIR from ``src.config`` and
expects the gaps JSON to contain entries of the form::

    {
        "start": "2024-06-19T02:59:00",
        "end": "2024-06-20T11:00:00",
        "duration": "1 days 08:01:00"
    }

For each gap, all rows with ``DateTime`` strictly between ``start`` and ``end``
are removed from the raw CSV. The original file is backed up with a timestamped
suffix in the same directory.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from typing import List, Dict

import pandas as pd

from src.config import RAW_DATA_CSV, PROCESSED_DATA_DIR  # type: ignore


DEFAULT_GAPS_JSON = os.path.join(PROCESSED_DATA_DIR, "missing_trading_days.json")


def remove_forward_filled_ranges(
    raw_csv_path: str = RAW_DATA_CSV,
    gaps_json_path: str = DEFAULT_GAPS_JSON,
    backup: bool = True,
) -> None:
    """Remove rows in RAW_DATA_CSV that correspond to previously filled gaps.

    Args:
        raw_csv_path: Path to the raw minute CSV (defaults to config.RAW_DATA_CSV).
        gaps_json_path: Path to a JSON file containing gap definitions.
        backup: When True, write a timestamped backup of the original CSV.
    """
    if not os.path.exists(raw_csv_path):
        print(f"[cleanup] Raw CSV not found at {raw_csv_path}; nothing to do.")
        return

    if not os.path.exists(gaps_json_path):
        print(f"[cleanup] Gaps JSON not found at {gaps_json_path}; nothing to remove.")
        return

    with open(gaps_json_path, "r") as f:
        try:
            gaps: List[Dict] = json.load(f)
        except json.JSONDecodeError:
            print(f"[cleanup] Could not decode JSON from {gaps_json_path}; aborting.")
            return

    if not gaps:
        print(f"[cleanup] Gaps JSON at {gaps_json_path} is empty; nothing to remove.")
        return

    print(f"[cleanup] Loading raw CSV from {raw_csv_path}...")
    df = pd.read_csv(raw_csv_path, parse_dates=["DateTime"])

    if backup:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.dirname(raw_csv_path)
        base_name = os.path.basename(raw_csv_path)
        backup_name = os.path.join(base_dir, f"backup_{ts}_{base_name}")
        df.to_csv(backup_name, index=False)
        print(f"[cleanup] Backed up original raw CSV to {backup_name}")

    original_len = len(df)

    for gap in gaps:
        try:
            start = pd.to_datetime(gap["start"])
            end = pd.to_datetime(gap["end"])
        except Exception as e:
            print(f"[cleanup] Warning: could not parse gap entry {gap}: {e}")
            continue

        if start >= end:
            print(f"[cleanup] Skipping gap with non-positive range: {gap}")
            continue

        mask_inside = (df["DateTime"] > start) & (df["DateTime"] < end)
        removed_count = int(mask_inside.sum())
        if removed_count > 0:
            print(
                f"[cleanup] Removing {removed_count} rows in interval "
                f"({start}, {end}) from raw CSV."
            )
        df = df.loc[~mask_inside].copy()

    final_len = len(df)
    print(
        f"[cleanup] Finished. Rows before: {original_len}, after: {final_len}, "
        f"removed: {original_len - final_len}."
    )

    df.to_csv(raw_csv_path, index=False)
    print(f"[cleanup] Wrote cleaned raw CSV back to {raw_csv_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove forward-filled ranges from RAW_DATA_CSV based on a gaps JSON "
            "(e.g., missing_trading_days.json)."
        )
    )
    parser.add_argument(
        "--raw-csv",
        default=RAW_DATA_CSV,
        help=f"Path to raw CSV (default: {RAW_DATA_CSV})",
    )
    parser.add_argument(
        "--gaps-json",
        default=DEFAULT_GAPS_JSON,
        help=f"Path to gaps JSON (default: {DEFAULT_GAPS_JSON})",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not write a backup copy of the original raw CSV.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    remove_forward_filled_ranges(
        raw_csv_path=args.raw_csv,
        gaps_json_path=args.gaps_json,
        backup=not args.no_backup,
    )