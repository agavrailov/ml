"""Regression guardrail for the 2026-04 NVDA daily-broadcast corruption.

Captures the bug where a Yahoo **daily**-interval feed was concatenated into
``data/raw/nvda_minute.csv``. A daily OHLC row ended up repeated across every
minute slot of ~21% of 2023 trading days. After resampling to 60min every hour
of those days carried the full daily range, which caused the strategy to blow
up from $10,000 to $0 inside the first ~50 bars of 2023 during backtests.

These tests pin the ingestion-boundary defense at
:func:`src.data_processing.clean_raw_minute_data`: daily-broadcast rows must be
dropped BEFORE the DateTime dedup so that co-located real minute bars survive.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.data_processing import (
    DAILY_BROADCAST_MIN_REPEAT,
    clean_raw_minute_data,
    identify_daily_broadcast_rows,
)


def _real_minute_rows(date: datetime, n: int = 390) -> list[dict]:
    """Build ``n`` realistic one-minute bars for ``date`` starting 09:30."""
    rows = []
    price = 100.0
    for i in range(n):
        ts = date.replace(hour=9, minute=30) + timedelta(minutes=i)
        o = price + 0.001 * i
        h = o + 0.02
        lo = o - 0.02
        c = o + 0.005
        rows.append({"DateTime": ts.isoformat(), "Open": o, "High": h, "Low": lo, "Close": c})
        price = c
    return rows


def _daily_broadcast_rows(date: datetime, n: int, ohlc: tuple[float, float, float, float]) -> list[dict]:
    """Build ``n`` minute rows on ``date`` that all carry the same daily OHLC."""
    o, h, lo, c = ohlc
    rows = []
    for i in range(n):
        ts = date.replace(hour=9, minute=30) + timedelta(minutes=i)
        rows.append({"DateTime": ts.isoformat(), "Open": o, "High": h, "Low": lo, "Close": c})
    return rows


def test_identify_daily_broadcast_rows_flags_repeated_daily_ohlc_only() -> None:
    date = datetime(2023, 1, 3)
    df = pd.DataFrame(
        _real_minute_rows(date, n=390)
        + _daily_broadcast_rows(date, n=DAILY_BROADCAST_MIN_REPEAT + 5,
                                ohlc=(14.8353, 14.9802, 14.0811, 14.2999))
    )
    mask = identify_daily_broadcast_rows(df)

    # Exactly the broadcast rows are flagged.
    assert mask.sum() == DAILY_BROADCAST_MIN_REPEAT + 5
    # No real minute row is flagged.
    assert not mask.iloc[:390].any()
    # All broadcast rows are flagged.
    assert mask.iloc[390:].all()


def test_identify_daily_broadcast_rows_spares_legitimate_flat_prints() -> None:
    """A single illiquid minute with O==H==L==C repeated briefly must NOT be
    misclassified as a daily broadcast."""
    date = datetime(2023, 6, 15)
    # 120 minute rows where price freezes at 50.0 (O==H==L==C, zero range).
    frozen = []
    for i in range(120):
        ts = date.replace(hour=9, minute=30) + timedelta(minutes=i)
        frozen.append({"DateTime": ts.isoformat(), "Open": 50.0, "High": 50.0, "Low": 50.0, "Close": 50.0})
    df = pd.DataFrame(frozen)

    mask = identify_daily_broadcast_rows(df)

    assert mask.sum() == 0, "Zero-range flat prints must not be flagged as broadcast"


def test_identify_daily_broadcast_rows_respects_min_repeat_threshold() -> None:
    """Small handful of coincidentally-repeated OHLC values must not be flagged
    as broadcast — only a full-day-worth repeat counts."""
    date = datetime(2023, 3, 1)
    rows = _real_minute_rows(date, n=390)
    # Add only 10 repeats of a specific OHLC — below threshold.
    rows += _daily_broadcast_rows(date, n=10, ohlc=(123.45, 124.00, 123.00, 123.80))

    df = pd.DataFrame(rows)
    mask = identify_daily_broadcast_rows(df)

    assert mask.sum() == 0


def test_clean_raw_minute_data_drops_daily_broadcast_rows(tmp_path) -> None:
    """End-to-end: concatenated feed (real minute bars + daily-broadcast rows
    at the same timestamps) must leave only the real minute bars after
    cleaning, with DateTime dedup NOT keeping the broadcast row."""
    date = datetime(2023, 1, 3)
    real = _real_minute_rows(date, n=200)
    broadcast = _daily_broadcast_rows(
        date,
        n=200,  # shares timestamps with the real minute rows
        ohlc=(14.8353, 14.9802, 14.0811, 14.2999),
    )
    # Simulate the upstream concatenation order: broadcast rows first (Yahoo
    # daily feed), then real minute rows (IBKR). Current dedup keeps the first
    # row per DateTime — so without the guardrail the BROADCAST rows win.
    csv_path = tmp_path / "nvda_minute.csv"
    pd.DataFrame(broadcast + real).to_csv(csv_path, index=False)

    clean_raw_minute_data(str(csv_path))

    cleaned = pd.read_csv(csv_path)
    cleaned["DateTime"] = pd.to_datetime(cleaned["DateTime"])

    # One row per minute on the date (real minute bars only).
    assert len(cleaned) == 200
    # No row carries the daily-broadcast OHLC signature.
    bad_sig = (
        (cleaned["Open"] == 14.8353)
        & (cleaned["High"] == 14.9802)
        & (cleaned["Low"] == 14.0811)
        & (cleaned["Close"] == 14.2999)
    )
    assert not bad_sig.any(), "Daily-broadcast rows survived clean_raw_minute_data"


def test_clean_raw_minute_data_removes_true_duplicate_timestamps(tmp_path, capsys) -> None:
    """True duplicates (identical timestamps from two ingestion runs concatenated
    into the same CSV) must be removed, keeping one row per DateTime.

    This is distinct from the broadcast-row bug: here both rows have realistic
    varying OHLC (not daily-broadcast signatures), so identify_daily_broadcast_rows
    will NOT flag them.  The plain drop_duplicates call at the end of
    clean_raw_minute_data must catch them instead.

    Regression risk: if someone wraps the dedup in a conditional that skips it
    for non-broadcast data, true duplicates would silently survive into the
    hourly resample and produce doubled-weight bars.
    """
    date = datetime(2024, 3, 5)
    real = _real_minute_rows(date, n=100)
    # Simulate a second ingestion run appended to the same CSV — exact copies.
    csv_path = tmp_path / "nvda_minute.csv"
    pd.DataFrame(real + real).to_csv(csv_path, index=False)

    assert len(pd.read_csv(csv_path)) == 200  # pre-condition: 100 dups present

    clean_raw_minute_data(str(csv_path))

    cleaned = pd.read_csv(csv_path)

    # Exactly 100 rows — duplicates removed.
    assert len(cleaned) == 100, (
        f"Expected 100 rows after dedup, got {len(cleaned)}"
    )
    # No duplicate timestamps remain.
    assert cleaned["DateTime"].duplicated().sum() == 0

    # The cleaner should print a message about removed rows.
    captured = capsys.readouterr()
    assert "duplicate" in captured.out.lower() or "removed" in captured.out.lower(), (
        "clean_raw_minute_data should log how many duplicate rows were dropped"
    )


def test_clean_raw_minute_data_drops_broadcast_even_when_alone(tmp_path) -> None:
    """When a date has ONLY broadcast rows (no surviving real minute data), the
    cleaner must still drop them — a half-contaminated day must not pollute
    the downstream hourly resample."""
    date = datetime(2023, 1, 17)
    broadcast = _daily_broadcast_rows(
        date,
        n=300,
        ohlc=(14.50, 14.90, 14.10, 14.30),
    )
    csv_path = tmp_path / "nvda_minute.csv"
    pd.DataFrame(broadcast).to_csv(csv_path, index=False)

    clean_raw_minute_data(str(csv_path))

    cleaned = pd.read_csv(csv_path)
    assert len(cleaned) == 0
