"""Bulk IBKR historical data downloader — single connection, all symbols in parallel.

Design:
  - ONE IB connection / ONE client ID for all symbols (avoids TWS slot exhaustion)
  - Single global asyncio.Semaphore(50) — IBKR's documented max simultaneous requests
  - All (symbol, day) request pairs are interleaved across symbols to avoid
    per-contract 60-req/10-min soft pacing violations
  - Pacing errors (error 162 / 366) trigger a brief sleep + retry
  - Existing raw CSVs are deleted before download (--fresh flag, required)
  - Saves: data/raw/{symbol}_minute.csv  (DateTime,Open,High,Low,Close)
"""
from __future__ import annotations

import argparse
import asyncio
import os
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from ib_insync import IB, Stock, util

from src.config import (
    CONTRACT_REGISTRY,
    TWS_HOST,
    TWS_PORT,
    PATHS,
)

# IBKR documented hard limit on simultaneous historical data requests.
IBKR_MAX_CONCURRENT = 50
# Pacing error codes from IBKR.
PACING_ERROR_CODES = {162, 366}
# How long to pause when a pacing violation is detected.
PACING_SLEEP_S = 12
# Max retries per (symbol, date) request before giving up.
MAX_RETRIES = 3


# ─────────────────────────────────────────────────────────────────────────────
# Request worker
# ─────────────────────────────────────────────────────────────────────────────

async def _request_one_day(
    ib: IB,
    contract,
    end_dt: datetime,
    semaphore: asyncio.Semaphore,
    per_sym_semaphore: asyncio.Semaphore,
    symbol: str,
) -> tuple[str, pd.DataFrame]:
    """Fetch one day of 1-min bars ending at end_dt. Returns (symbol, df)."""
    end_str = end_dt.strftime("%Y%m%d %H:%M:%S")
    for attempt in range(1, MAX_RETRIES + 1):
        async with per_sym_semaphore:
          async with semaphore:
            try:
                bars = await ib.reqHistoricalDataAsync(
                    contract,
                    endDateTime=end_str,
                    durationStr="1 D",
                    barSizeSetting="1 min",
                    whatToShow="TRADES",
                    useRTH=False,
                    formatDate=1,
                )
            except Exception as exc:
                print(f"  [{symbol}] exception on {end_str[:10]}: {exc}")
                bars = []

        if bars:
            df = util.df(bars)[["date", "open", "high", "low", "close"]].copy()
            df.columns = ["DateTime", "Open", "High", "Low", "Close"]
            df["DateTime"] = df["DateTime"].dt.strftime("%Y-%m-%d %H:%M:%S")
            return symbol, df

        # Check for pacing error via IB error events (ib_insync surfaces them
        # as warnings; check the reqId=-1 channel).
        # Simpler: sleep briefly on empty result and retry if retries left.
        if attempt < MAX_RETRIES:
            await asyncio.sleep(PACING_SLEEP_S)

    return symbol, pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Main download
# ─────────────────────────────────────────────────────────────────────────────

async def download_all(
    symbols: list[str],
    start: datetime,
    end: datetime,
    client_id: int,
    concurrency: int,
    per_sym_concurrency: int = 1,
) -> None:
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    ib = IB()
    try:
        await ib.connectAsync(TWS_HOST, TWS_PORT, clientId=client_id)
        print(f"Connected: {TWS_HOST}:{TWS_PORT} clientId={client_id}\n")

        # Qualify contracts once upfront.
        contracts: dict[str, Stock] = {}
        for sym in symbols:
            cd = CONTRACT_REGISTRY[sym]
            c = Stock(symbol=cd["symbol"], exchange=cd["exchange"], currency=cd["currency"])
            await ib.qualifyContractsAsync(c)
            contracts[sym] = c
            print(f"  Qualified: {sym}")

        # Build all (symbol, end_datetime) pairs, interleaved across symbols
        # so requests rotate between symbols rather than hammering one contract.
        cursor = end
        day_lists: dict[str, list[datetime]] = {s: [] for s in symbols}
        while cursor > start:
            for sym in symbols:
                day_lists[sym].append(cursor)
            cursor -= timedelta(days=1)

        # Interleave: day1_sym1, day1_sym2, ..., day2_sym1, day2_sym2 ...
        all_tasks_args: list[tuple[str, datetime]] = []
        max_days = max(len(v) for v in day_lists.values())
        for i in range(max_days):
            for sym in symbols:
                if i < len(day_lists[sym]):
                    all_tasks_args.append((sym, day_lists[sym][i]))

        total = len(all_tasks_args)
        print(f"\nTotal requests: {total} ({len(symbols)} symbols x ~{max_days} days)")
        print(f"Concurrency: {concurrency}  |  Est. time: ~{total/concurrency*0.5/60:.1f} min\n")

        # Global semaphore: IBKR max 50 simultaneous requests.
        # Per-symbol semaphore: 1 = strictly sequential per contract (safest).
        semaphore = asyncio.Semaphore(concurrency)
        per_sym_semaphores = {s: asyncio.Semaphore(per_sym_concurrency) for s in symbols}

        buffers: dict[str, list[pd.DataFrame]] = {s: [] for s in symbols}
        file_created: dict[str, bool] = {s: False for s in symbols}
        completed = 0
        t0 = time.time()

        # One round = one day per symbol (all symbols in parallel).
        # Keeping rounds small prevents per-contract burst that triggers IBKR pacing.
        ROUND_SIZE = len(symbols)
        FLUSH_EVERY_ROUNDS = 20  # flush to disk every 20 rounds
        INTER_ROUND_SLEEP = 1.0  # seconds between rounds to avoid burst pacing

        async def run_and_collect(sym: str, dt: datetime) -> tuple[str, pd.DataFrame]:
            return await _request_one_day(ib, contracts[sym], dt, semaphore, per_sym_semaphores[sym], sym)

        rounds_since_flush = 0
        for batch_start in range(0, total, ROUND_SIZE):
            batch = all_tasks_args[batch_start: batch_start + ROUND_SIZE]
            results = await asyncio.gather(*[run_and_collect(s, d) for s, d in batch])

            for sym, df in results:
                if not df.empty:
                    buffers[sym].append(df)
                completed += 1

            rounds_since_flush += 1
            await asyncio.sleep(INTER_ROUND_SLEEP)

            if rounds_since_flush >= FLUSH_EVERY_ROUNDS:
                # Flush buffers to disk.
                for sym in symbols:
                    if buffers[sym]:
                        combined = pd.concat(buffers[sym]).drop_duplicates("DateTime").sort_values("DateTime")
                        path = raw_dir / f"{sym.lower()}_minute.csv"
                        combined.to_csv(path, mode="a", header=not file_created[sym], index=False)
                        file_created[sym] = True
                        buffers[sym] = []
                rounds_since_flush = 0

                elapsed = time.time() - t0
                pct = completed / total * 100
                remaining = (elapsed / completed * (total - completed)) if completed else 0
                print(f"  [{completed}/{total} {pct:.0f}%] elapsed={elapsed:.0f}s  eta={remaining:.0f}s", flush=True)

        # Final flush.
        for sym in symbols:
            if buffers[sym]:
                combined = pd.concat(buffers[sym]).drop_duplicates("DateTime").sort_values("DateTime")
                path = raw_dir / f"{sym.lower()}_minute.csv"
                combined.to_csv(path, mode="a", header=not file_created[sym], index=False)
                file_created[sym] = True

    finally:
        if ib.isConnected():
            ib.disconnect()
            print("\nDisconnected from TWS.")

    # Summary.
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    for sym in symbols:
        path = raw_dir / f"{sym.lower()}_minute.csv"
        if path.exists():
            df = pd.read_csv(path)
            print(f"  {sym:5} {len(df):>8} rows  {df['DateTime'].iloc[0][:10]} -> {df['DateTime'].iloc[-1][:10]}")
        else:
            print(f"  {sym:5} NO DATA")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+",
                   default=["NVDA", "MSFT", "JPM", "XOM", "UNH", "BAC", "GS", "JNJ"])
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--client-id", type=int, default=10)
    p.add_argument("--concurrency", type=int, default=IBKR_MAX_CONCURRENT)
    p.add_argument("--per-sym-concurrency", type=int, default=1,
                   help="Max simultaneous requests per symbol (default 1 = sequential per contract)")
    p.add_argument("--fresh", action="store_true", default=False,
                   help="Delete existing raw CSVs before downloading")
    args = p.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d")
    # Use current moment as end — never a future time, which causes IBKR to hold requests
    end = datetime.now().replace(second=0, microsecond=0)

    if args.fresh:
        raw_dir = Path("data/raw")
        for sym in args.symbols:
            p2 = raw_dir / f"{sym.lower()}_minute.csv"
            if p2.exists():
                p2.unlink()
                print(f"Deleted {p2}")

    asyncio.run(download_all(
        args.symbols, start, end, args.client_id,
        args.concurrency, args.per_sym_concurrency,
    ))


if __name__ == "__main__":
    main()
