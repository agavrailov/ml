"""Curated minute-bar transform and simple access helpers.

This module keeps operations simple and atomic:

- Raw ingestion writes minute bars to ``RAW_DATA_CSV``.
- ``run_transform_minute_bars`` cleans and normalizes raw bars and writes a
  curated-minute CSV under ``PROCESSED_DATA_DIR``.
- ``get_raw_bars`` / ``get_curated_bars`` provide small read helpers for
  downstream code.

Gap analysis/filling remains a separate concern (e.g., via ``daily_data_agent``)
so that each step can be re-run independently.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import pandas as pd

from src.config import PATHS, PROCESSED_DATA_DIR  # type: ignore
from src.data_processing import clean_raw_minute_data  # type: ignore

# Legacy constant for NVDA (kept for backwards-compatible imports).
CURATED_MINUTE_PATH = os.path.join(PROCESSED_DATA_DIR, "nvda_minute_curated.csv")


def run_transform_minute_bars(symbol: str = "NVDA") -> str:
    """Clean raw minute data for ``symbol`` and write a curated-minute CSV.

    Uses symbol-parameterized paths so any registered symbol can be processed.
    Returns the path to the curated-minute CSV.
    """
    raw_csv = PATHS.raw_data_csv(symbol)
    curated_path = os.path.join(PROCESSED_DATA_DIR, f"{symbol.lower()}_minute_curated.csv")

    if not os.path.exists(raw_csv):
        raise FileNotFoundError(
            f"Raw minute data not found at {raw_csv}. Run ingestion first."
        )

    # Clean in-place (dedupe, sort, ensure DateTime column).
    clean_raw_minute_data(raw_csv)

    # Load cleaned raw data and write to curated path.
    df = pd.read_csv(raw_csv)
    # Keep the canonical columns used elsewhere.
    expected_cols = [
        col
        for col in ["DateTime", "Open", "High", "Low", "Close"]
        if col in df.columns
    ]
    df_out = df[expected_cols].copy()

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df_out.to_csv(curated_path, index=False)

    print(f"Curated minute data written to {curated_path}")
    return curated_path


def _load_bars(path: str, start: Optional[datetime], end: Optional[datetime]) -> pd.DataFrame:
    """Internal helper to load a bars CSV and optionally filter by time range."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path, parse_dates=["DateTime"])
    if start is not None:
        df = df[df["DateTime"] >= start]
    if end is not None:
        df = df[df["DateTime"] < end]
    return df


def get_raw_bars(symbol: str, start: Optional[datetime] = None, end: Optional[datetime] = None) -> pd.DataFrame:
    """Return raw minute bars for ``symbol`` between ``start`` and ``end``."""
    raw_csv = PATHS.raw_data_csv(symbol)
    return _load_bars(raw_csv, start, end)


def get_curated_bars(symbol: str, start: Optional[datetime] = None, end: Optional[datetime] = None) -> pd.DataFrame:
    """Return curated minute bars for ``symbol`` between ``start`` and ``end``.

    Call ``run_transform_minute_bars(symbol)`` first if the file does not exist.
    """
    curated_path = os.path.join(PROCESSED_DATA_DIR, f"{symbol.lower()}_minute_curated.csv")
    return _load_bars(curated_path, start, end)


def main() -> None:
    """CLI entrypoint for curated-minute transform.

    Usage (from repo root):

        python -m src.ingestion.curated_minute --symbol NVDA

    The symbol argument is currently accepted for future compatibility but only
    NVDA is supported.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Run the raw→curated minute-bar transform. This cleans RAW_DATA_CSV "
            "and writes a curated-minute snapshot under PROCESSED_DATA_DIR."
        )
    )
    parser.add_argument(
        "--symbol",
        default="NVDA",
        help="Symbol to transform (currently NVDA-only).",
    )

    args = parser.parse_args()
    path = run_transform_minute_bars(args.symbol)
    print(f"Curated minute data written to {path}")


if __name__ == "__main__":
    main()
