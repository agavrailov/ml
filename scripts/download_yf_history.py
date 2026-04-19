"""Download full price history from Yahoo Finance for specified symbols.

Strategy:
  - 1-minute bars: max 7 days per request -> chunk from start_date to now
  - Falls back to 1-hour bars if 1m chunk returns nothing (throttle/limit)
  - Merges new YF data with any existing raw CSV (IBKR data preserved)
  - Saves in DateTime,Open,High,Low,Close format matching existing raw CSVs
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

RAW_DIR = Path("data/raw")
CHUNK_DAYS_1M = 7
CHUNK_DAYS_1H = 729
START_DATE = datetime(2022, 1, 1)
SLEEP_BETWEEN_CHUNKS = 1.5  # seconds — avoid YF throttle


def _fetch_chunk(ticker: str, start: datetime, end: datetime, interval: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    df = t.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), interval=interval)
    if df.empty:
        return pd.DataFrame()
    df = df[["Open", "High", "Low", "Close"]].copy()
    df.index = pd.DatetimeIndex(df.index).tz_localize(None) if df.index.tzinfo is not None else df.index
    df.index.name = "DateTime"
    return df.reset_index()


def download_1m_history(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Download 1-minute bars in 7-day chunks."""
    chunks = []
    cursor = start
    total_days = (end - start).days
    done = 0
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=CHUNK_DAYS_1M), end)
        df = _fetch_chunk(ticker, cursor, chunk_end, "1m")
        if not df.empty:
            chunks.append(df)
        done += (chunk_end - cursor).days
        pct = done / total_days * 100
        print(f"  [{ticker} 1m] {cursor.date()} -> {chunk_end.date()} | {len(df)} rows | {pct:.0f}%")
        cursor = chunk_end
        time.sleep(SLEEP_BETWEEN_CHUNKS)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def download_1h_history(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Download 1-hour bars in 729-day chunks (YF limit)."""
    chunks = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=CHUNK_DAYS_1H), end)
        df = _fetch_chunk(ticker, cursor, chunk_end, "1h")
        if not df.empty:
            chunks.append(df)
        print(f"  [{ticker} 1h] {cursor.date()} -> {chunk_end.date()} | {len(df)} rows")
        cursor = chunk_end
        time.sleep(SLEEP_BETWEEN_CHUNKS)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def download_1d_history(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Download daily bars (fallback for very old data, sets timestamp to 09:30)."""
    df = _fetch_chunk(ticker, start, end, "1d")
    if df.empty:
        return df
    df["DateTime"] = pd.to_datetime(df["DateTime"]).dt.normalize() + timedelta(hours=9, minutes=30)
    print(f"  [{ticker} 1d] {start.date()} -> {end.date()} | {len(df)} rows (daily, expanded to 09:30)")
    return df


def merge_with_existing(new_df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return new_df
    try:
        existing = pd.read_csv(csv_path, parse_dates=["DateTime"])
    except Exception:
        return new_df
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["DateTime"]).sort_values("DateTime").reset_index(drop=True)
    print(f"  Merged: {len(existing)} existing + {len(new_df)} new = {len(combined)} total rows")
    return combined


def save(df: pd.DataFrame, csv_path: Path) -> None:
    df["DateTime"] = pd.to_datetime(df["DateTime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    df[["DateTime", "Open", "High", "Low", "Close"]].to_csv(csv_path, index=False)
    print(f"  Saved {len(df)} rows -> {csv_path}")


def process_symbol(ticker: str, start: datetime, end: datetime, mode: str) -> None:
    csv_path = RAW_DIR / f"{ticker.lower()}_minute.csv"
    print(f"\n{'='*60}")
    print(f"Downloading {ticker} | {start.date()} -> {end.date()} | mode={mode}")
    print(f"{'='*60}")

    if mode == "1m":
        df = download_1m_history(ticker, start, end)
    elif mode == "1h":
        df = download_1h_history(ticker, start, end)
    elif mode == "1h+1d":
        # Hourly for last 730 days, daily for everything before
        hourly_start = max(start, end - timedelta(days=729))
        df_daily = pd.DataFrame()
        if hourly_start > start:
            df_daily = download_1d_history(ticker, start, hourly_start)
        df_hourly = download_1h_history(ticker, hourly_start, end)
        parts = [p for p in [df_daily, df_hourly] if not p.empty]
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if df.empty:
        print(f"  WARNING: No data returned for {ticker}")
        return

    df = df.drop_duplicates(subset=["DateTime"]).sort_values("DateTime").reset_index(drop=True)
    print(f"  Downloaded {len(df)} rows | {df['DateTime'].iloc[0]} -> {df['DateTime'].iloc[-1]}")

    merged = merge_with_existing(df, csv_path)
    save(merged, csv_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=["GS", "BAC", "UNH", "JNJ"])
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--mode", default="1h+1d",
                   choices=["1m", "1h", "1h+1d"],
                   help="1m=minute chunks (slow), 1h=hourly (fast), 1h+1d=hourly+daily hybrid")
    args = p.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.now().replace(hour=23, minute=59, second=0)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for sym in args.symbols:
        process_symbol(sym, start, end, args.mode)

    print("\nDownload complete. Run daily_data_agent.py to process features.")


if __name__ == "__main__":
    main()
