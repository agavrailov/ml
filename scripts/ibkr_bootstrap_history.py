"""One-time IBKR historical minute-bar bootstrap for all portfolio symbols.

Fetches 1-month chunks of 1-min bars per request (vs the daily incremental
updater which uses 1-day chunks). This cuts the request count from ~6,000
to ~288 for a 4-year window across 6 symbols, completing in ~2-3 hours.

IB pacing rules (enforced server-side):
  - Max 60 historical requests per 10 min rolling window
  - Requests returning large bar counts (>10k) trigger extra scrutiny
  - After a cancel/timeout, IB expects a longer back-off before retrying

Requirements:
    - TWS or IB Gateway must be running and accepting connections.
    - The client-id used here (default 99) must not be in use by any other
      session. Stop the live daemon or paper-trader before running.

Usage:
    python scripts/ibkr_bootstrap_history.py
    python scripts/ibkr_bootstrap_history.py --symbols NVDA JPM --start 2022-01-01
    python scripts/ibkr_bootstrap_history.py --client-id 9
    python scripts/ibkr_bootstrap_history.py --resume  # skip symbols with existing CSVs

After it finishes run:
    python -m src.daily_data_agent --skip-ingestion
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from ib_insync import IB, Stock, util  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    TWS_HOST,
    TWS_PORT,
    CONTRACT_REGISTRY,
    get_raw_data_csv_path,
)
from src.core.config_resolver import get_configured_symbols
from src.data_processing import clean_raw_minute_data

# ---------------------------------------------------------------------------
# Pacing constants — tuned to avoid IB server-side cancels
# ---------------------------------------------------------------------------
_DURATION = "1 M"
_PACING_SLEEP = 30.0       # 2 req/min  (IB allows 6/min but large bars need slack)
_COOLDOWN_AFTER_ERROR = 180.0  # 3-min pause after a cancel/timeout
_MAX_RETRIES = 4           # attempts per month-chunk before giving up
_RETRY_DELAYS = [60, 120, 180, 300]  # back-off ladder (seconds)
_CLIENT_ID = 99            # dedicated bootstrap client-id; change if in use

# "1 M" in IBKR durationStr returns one calendar month of bars ending at
# ``endDateTime``. We therefore step the cursor back to the last day of the
# previous calendar month: consecutive chunks touch exactly at month
# boundaries, which also makes the log lines read "month ending 2026-03-31,
# 2026-02-28, 2026-01-31, …" (matching human expectations). A 1-day overlap
# is tolerated by ``drop_duplicates`` downstream.
#
# The first iteration is special — ``end`` is usually ``datetime.now()`` mid
# month, so its log line will show today's date rather than a month-end. The
# bug this guards against (see tests/test_ibkr_bootstrap_stride.py) is the
# previous ``days=32 + replace(day=1)`` logic, which produced a 2-month
# stride from a mid-month start and never requested odd calendar months.
_MAX_CHUNK_COVERAGE = timedelta(days=31)


def step_cursor_back(cursor: datetime) -> datetime:
    """Return the endDateTime for the next (older) 1-month IBKR chunk.

    Snaps to the last day of the previous calendar month so that chunk
    boundaries align with calendar months. Pure function for unit testing.
    """
    first_of_month = cursor.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return first_of_month - timedelta(days=1)


def iter_chunk_endings(start: datetime, end: datetime) -> list[datetime]:
    """Enumerate the chunk-end datetimes produced by the walk-back loop.

    Each endDateTime will be passed to IBKR with ``durationStr="1 M"``, which
    returns bars in roughly ``[endDateTime - 30 days, endDateTime]``. Returned
    list is ordered newest-first, matching the loop order in
    :func:`bootstrap_symbol`.
    """
    endings: list[datetime] = []
    cursor = end
    while cursor > start:
        endings.append(cursor)
        cursor = step_cursor_back(cursor)
    return endings


async def _fetch_month(ib: IB, contract, end_dt: datetime) -> pd.DataFrame:
    """Request 1 month of 1-min bars ending at ``end_dt`` with retry logic."""
    end_str = end_dt.strftime("%Y%m%d %H:%M:%S")

    for attempt in range(_MAX_RETRIES):
        try:
            bars = await ib.reqHistoricalDataAsync(
                contract,
                endDateTime=end_str,
                durationStr=_DURATION,
                barSizeSetting="1 min",
                whatToShow="TRADES",
                useRTH=False,
                formatDate=1,
                timeout=120,  # seconds; ib_insync default is 60
            )
            if not bars:
                return pd.DataFrame()

            df = util.df(bars)[["date", "open", "high", "low", "close"]].copy()
            df.columns = ["DateTime", "Open", "High", "Low", "Close"]
            df["DateTime"] = pd.to_datetime(df["DateTime"]).dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            return df

        except Exception as exc:
            delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
            print(
                f"  [retry {attempt + 1}/{_MAX_RETRIES}] error: {exc}  "
                f"-> sleeping {delay}s before retry ...",
                flush=True,
            )
            await asyncio.sleep(delay)

    # All retries exhausted
    print(f"  [skip] giving up on {end_str} after {_MAX_RETRIES} attempts.")
    return pd.DataFrame()


async def bootstrap_symbol(
    symbol: str,
    start: datetime,
    end: datetime,
    host: str,
    port: int,
    client_id: int,
    resume: bool = False,
) -> None:
    """Fetch full minute history for one symbol and write to raw CSV."""
    if symbol not in CONTRACT_REGISTRY:
        print(f"[bootstrap] {symbol}: not in CONTRACT_REGISTRY, skipping.")
        return

    csv_path = get_raw_data_csv_path(symbol)

    if resume and Path(csv_path).exists():
        print(f"[bootstrap] {symbol}: raw CSV already exists, skipping (--resume).")
        return

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    ib = IB()
    await ib.connectAsync(host, port, client_id)
    print(f"[bootstrap] Connected to IBKR {host}:{port} (client {client_id})")

    details = CONTRACT_REGISTRY[symbol]
    contract = Stock(
        symbol=details["symbol"],
        exchange=details["exchange"],
        currency=details["currency"],
    )
    await ib.qualifyContractsAsync(contract)
    print(f"[bootstrap] {symbol}: qualified contract.")

    # Walk backward month-by-month from end -> start.
    frames: list[pd.DataFrame] = []
    cursor = end
    consecutive_errors = 0

    while cursor > start:
        print(
            f"[bootstrap] {symbol}: requesting month ending "
            f"{cursor.strftime('%Y-%m-%d')} ...",
            flush=True,
        )

        df = await _fetch_month(ib, contract, cursor)

        if not df.empty:
            frames.append(df)
            consecutive_errors = 0
            print(f"[bootstrap] {symbol}: got {len(df):,} bars.")
        else:
            consecutive_errors += 1
            print(
                f"[bootstrap] {symbol}: empty/failed "
                f"(consecutive failures: {consecutive_errors})."
            )
            if consecutive_errors >= 2:
                print(
                    f"[bootstrap] {symbol}: {consecutive_errors} consecutive "
                    f"failures — cooldown {_COOLDOWN_AFTER_ERROR:.0f}s ...",
                    flush=True,
                )
                await asyncio.sleep(_COOLDOWN_AFTER_ERROR)
                consecutive_errors = 0

        cursor = step_cursor_back(cursor)

        await asyncio.sleep(_PACING_SLEEP)

    ib.disconnect()

    if not frames:
        print(f"[bootstrap] {symbol}: no data fetched.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("DateTime").drop_duplicates(subset=["DateTime"])
    combined = combined[combined["DateTime"] >= start.strftime("%Y-%m-%d")]
    combined.to_csv(csv_path, index=False)
    print(f"[bootstrap] {symbol}: wrote {len(combined):,} raw bars -> {csv_path}")

    # Run the broadcast-row guardrail.
    clean_raw_minute_data(csv_path)
    print(f"[bootstrap] {symbol}: cleaned.")


async def main_async(
    symbols: list[str],
    start: datetime,
    end: datetime,
    host: str,
    port: int,
    client_id: int,
    resume: bool,
) -> None:
    for sym in symbols:
        print(f"\n{'='*60}")
        print(f"  {sym}  ({start.date()} -> {end.date()})")
        print(f"{'='*60}")
        try:
            await bootstrap_symbol(sym, start, end, host, port, client_id, resume)
        except Exception as exc:
            print(f"[bootstrap] {sym}: FAILED — {exc}")
        # Extra pause between symbols to let IB's 10-min window breathe.
        print(f"[bootstrap] Pausing 60s before next symbol ...", flush=True)
        await asyncio.sleep(60)

    print("\nBootstrap complete. Next:")
    print("  python -m src.daily_data_agent --skip-ingestion")
    print("  python scripts/generate_predictions_csv.py")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-time IBKR historical minute-bar bootstrap (1-month chunks)."
    )
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--host", default=TWS_HOST)
    parser.add_argument("--port", default=TWS_PORT, type=int)
    parser.add_argument("--client-id", default=_CLIENT_ID, type=int)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip symbols that already have a raw CSV on disk.",
    )
    args = parser.parse_args()

    symbols = args.symbols or get_configured_symbols()
    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = (
        datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.now()
    )

    # ~48 months/symbol × 30s pacing + retries budget
    est_min = len(symbols) * 48 * _PACING_SLEEP / 60
    print(f"Symbols    : {symbols}")
    print(f"Date range : {args.start} -> {args.end or 'today'}")
    print(f"IBKR       : {args.host}:{args.port} client-id={args.client_id}")
    print(f"Pacing     : {_PACING_SLEEP:.0f}s between requests")
    print(f"Est. time  : ~{est_min:.0f} min (no retries)")
    print(f"Resume     : {args.resume}")
    print()

    asyncio.run(
        main_async(symbols, start_dt, end_dt, args.host, args.port, args.client_id, args.resume)
    )


if __name__ == "__main__":
    main()
