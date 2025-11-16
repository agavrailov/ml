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

from src.config import RAW_DATA_CSV, PROCESSED_DATA_DIR  # type: ignore
from src.data_processing import clean_raw_minute_data  # type: ignore

# Single curated-minute file for now; simple and explicit.
CURATED_MINUTE_PATH = os.path.join(PROCESSED_DATA_DIR, "nvda_minute_curated.csv")


def run_transform_minute_bars(symbol: str = "NVDA") -> str:
    """Clean raw minute data and write a curated-minute CSV.

    For now this is a thin wrapper around ``clean_raw_minute_data`` followed by
    copying the cleaned raw data into a curated file. Gap analysis/filling is
    intentionally left to other components to keep this step simple and atomic.

    Returns the path to the curated-minute CSV.
    """
    # v1 is NVDA-only; symbol is accepted for future compatibility.
    if symbol.upper() != "NVDA":
        print(
            f"run_transform_minute_bars currently only supports NVDA; "
            f"received symbol={symbol}. Proceeding with RAW_DATA_CSV={RAW_DATA_CSV}."
        )

    if not os.path.exists(RAW_DATA_CSV):
        raise FileNotFoundError(
            f"Raw minute data not found at {RAW_DATA_CSV}. Run ingestion first."
        )

    # Clean in-place (dedupe, sort, ensure DateTime column).
    clean_raw_minute_data(RAW_DATA_CSV)

    # Load cleaned raw data and write to curated path.
    df = pd.read_csv(RAW_DATA_CSV)
    # Keep the canonical columns used elsewhere.
    expected_cols = [
        col
        for col in ["DateTime", "Open", "High", "Low", "Close"]
        if col in df.columns
    ]
    df_out = df[expected_cols].copy()

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df_out.to_csv(CURATED_MINUTE_PATH, index=False)

    print(f"Curated minute data written to {CURATED_MINUTE_PATH}")
    return CURATED_MINUTE_PATH


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
    """Return raw minute bars for ``symbol`` between ``start`` and ``end``.

    Currently scoped to NVDA and backed by ``RAW_DATA_CSV``.
    """
    # symbol is accepted for future use; we only support NVDA today.
    return _load_bars(RAW_DATA_CSV, start, end)


def get_curated_bars(symbol: str, start: Optional[datetime] = None, end: Optional[datetime] = None) -> pd.DataFrame:
    """Return curated minute bars for ``symbol`` between ``start`` and ``end``.

    Backed by ``CURATED_MINUTE_PATH``; call ``run_transform_minute_bars`` first
    if the file does not exist.
    """
    # symbol is accepted for future use; we only support NVDA today.
    return _load_bars(CURATED_MINUTE_PATH, start, end)