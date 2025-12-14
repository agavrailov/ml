"""Formatting and filtering helpers for UI display."""

from __future__ import annotations

import pandas as pd


def format_timestamp(ts: str | None) -> str | None:
    """Return an ISO timestamp truncated to seconds (YYYY-MM-DDTHH:MM:SS)."""
    if not ts:
        return None
    s = str(ts)

    # Fast-path: keep only the first 19 chars, which correspond to seconds.
    # Examples:
    # - 2025-12-12T10:58:23.123456+00:00 -> 2025-12-12T10:58:23
    # - 2025-12-12T10:58:23 -> 2025-12-12T10:58:23
    if len(s) >= 19 and s[4] == "-" and s[10] == "T":
        return s[:19]

    try:
        t = pd.to_datetime(s, utc=True, errors="coerce")
        if pd.isna(t):
            return None
        return t.strftime("%Y-%m-%dT%H:%M:%S")
    except Exception:
        return None


def filter_training_history(history: list[dict], *, frequency: str, tsteps: int) -> list[dict]:
    """Return training history rows matching a specific (frequency, tsteps)."""
    rows = [
        r
        for r in (history or [])
        if isinstance(r, dict)
        and r.get("frequency") == frequency
        and int(r.get("tsteps", -1)) == int(tsteps)
    ]

    # Most-recent first.
    return sorted(rows, key=lambda r: r.get("timestamp", ""), reverse=True)


def filter_backtest_history(history: list[dict], *, frequency: str) -> list[dict]:
    """Return backtest history rows matching a specific frequency."""
    rows = [
        r
        for r in (history or [])
        if isinstance(r, dict) and r.get("frequency") == frequency
    ]

    # Most-recent first.
    return sorted(rows, key=lambda r: r.get("timestamp", ""), reverse=True)


def filter_optimization_history(history: list[dict], *, frequency: str) -> list[dict]:
    """Return optimization history rows matching a specific frequency."""
    rows = [
        r
        for r in (history or [])
        if isinstance(r, dict) and r.get("frequency") == frequency
    ]

    # Most-recent first.
    return sorted(rows, key=lambda r: r.get("timestamp", ""), reverse=True)


def get_best_training_row(rows: list[dict]) -> dict | None:
    """Return the row with minimal validation_loss, or None."""
    best: dict | None = None
    best_loss = float("inf")

    for r in rows or []:
        try:
            loss = float(r.get("validation_loss"))
        except Exception:
            continue

        if loss < best_loss:
            best_loss = loss
            best = r

    return best
