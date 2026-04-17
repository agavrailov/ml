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


def _matches_symbol(row: dict, symbol: str | None) -> bool:
    """Return True if ``row`` matches ``symbol``.

    Rows without a ``symbol`` field are treated as legacy NVDA rows so the
    filter doesn't hide pre-existing history when the user has been working
    on NVDA.  Pass ``symbol=None`` to disable filtering.
    """
    if symbol is None:
        return True
    row_sym = str(row.get("symbol", "NVDA")).upper()
    return row_sym == str(symbol).upper()


def filter_training_history(
    history: list[dict],
    *,
    frequency: str,
    tsteps: int,
    symbol: str | None = None,
) -> list[dict]:
    """Return training history rows matching (frequency, tsteps, [symbol])."""
    rows = [
        r
        for r in (history or [])
        if isinstance(r, dict)
        and r.get("frequency") == frequency
        and int(r.get("tsteps", -1)) == int(tsteps)
        and _matches_symbol(r, symbol)
    ]

    # Most-recent first.
    return sorted(rows, key=lambda r: r.get("timestamp", ""), reverse=True)


def filter_backtest_history(
    history: list[dict],
    *,
    frequency: str,
    symbol: str | None = None,
) -> list[dict]:
    """Return backtest history rows matching (frequency, [symbol])."""
    rows = [
        r
        for r in (history or [])
        if isinstance(r, dict)
        and r.get("frequency") == frequency
        and _matches_symbol(r, symbol)
    ]

    # Most-recent first.
    return sorted(rows, key=lambda r: r.get("timestamp", ""), reverse=True)


def filter_optimization_history(
    history: list[dict],
    *,
    frequency: str,
    symbol: str | None = None,
) -> list[dict]:
    """Return optimization history rows matching (frequency, [symbol])."""
    rows = [
        r
        for r in (history or [])
        if isinstance(r, dict)
        and r.get("frequency") == frequency
        and _matches_symbol(r, symbol)
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
