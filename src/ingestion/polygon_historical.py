"""Historical minute-bar download via Polygon.io — REST API or S3 Flat Files.

Two access modes (set via ``method`` arg or ``POLYGON_METHOD`` env var):

  ``api``        — REST ``/v2/aggs`` endpoint, auto-paginated.
                   Works on the free Stocks Starter plan.
                   ~10 API calls per symbol for 4 years of data.

  ``flatfiles``  — Direct S3 download of pre-built day-partition gzip CSV files.
                   Dramatically faster for bulk pulls (no pagination, parallel
                   downloads possible). Requires a paid Polygon plan with S3
                   Flat Files access enabled.
                   S3 endpoint: https://files.polygon.io
                   Bucket:      flatfiles.polygon.io
                   Path:        us_stocks_sip/minute_aggs_v1/{year}/{month}/{date}.csv.gz
                   Columns:     ticker,window_start,open,high,low,close,volume,vwap,transactions
                   (window_start is nanoseconds UTC)

Environment variables:
    POLYGON_API_KEY   — required for both modes
    POLYGON_METHOD    — ``api`` (default) or ``flatfiles``

Output CSV format (both modes):
    DateTime, Open, High, Low, Close
    DateTime in "%Y-%m-%d %H:%M:%S" (naive UTC), matching IBKR format.
"""
from __future__ import annotations

import io
import os
import gzip
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import get_raw_data_csv_path


_DEFAULT_START = date(2022, 1, 1)
_API_RATE_SLEEP = 0.22          # ~4.5 req/s — under free 5 req/s cap
_FLATFILES_WORKERS = 8          # parallel day-file downloads
_S3_ENDPOINT = "https://files.massive.com"
_S3_BUCKET = "flatfiles"
_S3_PREFIX = "us_stocks_sip/minute_aggs_v1"


# ---------------------------------------------------------------------------
# API mode
# ---------------------------------------------------------------------------

def _download_via_api(
    symbol: str,
    start: date,
    end: date,
    adjusted: bool,
    api_key: str,
) -> pd.DataFrame:
    from polygon import RESTClient  # type: ignore
    client = RESTClient(api_key=api_key)

    rows: list[dict] = []
    for agg in client.list_aggs(
        ticker=symbol,
        multiplier=1,
        timespan="minute",
        from_=str(start),
        to=str(end),
        adjusted=adjusted,
        sort="asc",
        limit=50_000,
    ):
        dt = datetime.fromtimestamp(agg.timestamp / 1000, tz=timezone.utc)
        rows.append({
            "DateTime": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "Open":  agg.open,
            "High":  agg.high,
            "Low":   agg.low,
            "Close": agg.close,
        })
    time.sleep(_API_RATE_SLEEP)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Flat files mode
# ---------------------------------------------------------------------------

def _make_s3_client(api_key: str):
    import boto3  # type: ignore
    return boto3.client(
        "s3",
        endpoint_url=_S3_ENDPOINT,
        aws_access_key_id=api_key,
        aws_secret_access_key=api_key,  # Polygon uses the API key for both
    )


def _trading_days(start: date, end: date) -> list[date]:
    """Return calendar dates between start and end inclusive.

    We include all weekdays and filter empty files at download time rather
    than importing a calendar library — Polygon simply has no file for
    holidays/weekends.
    """
    days = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:  # Mon–Fri
            days.append(cur)
        cur += timedelta(days=1)
    return days


def _fetch_day_file(s3, day: date, symbol: str) -> pd.DataFrame:
    """Download one day-partition file and filter to ``symbol``."""
    key = f"{_S3_PREFIX}/{day.year}/{day.month:02d}/{day}.csv.gz"
    try:
        obj = s3.get_object(Bucket=_S3_BUCKET, Key=key)
        raw = obj["Body"].read()
    except Exception:
        return pd.DataFrame()   # holiday / weekend / not yet uploaded

    with gzip.open(io.BytesIO(raw), "rt") as f:
        df = pd.read_csv(f)

    if df.empty or "ticker" not in df.columns:
        return pd.DataFrame()

    df = df[df["ticker"] == symbol].copy()
    if df.empty:
        return pd.DataFrame()

    # window_start is nanoseconds UTC -> datetime
    df["DateTime"] = (
        pd.to_datetime(df["window_start"], unit="ns", utc=True)
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )
    return df[["DateTime", "open", "high", "low", "close"]].rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}
    )


def _download_via_flatfiles(
    symbol: str,
    start: date,
    end: date,
    api_key: str,
) -> pd.DataFrame:
    s3 = _make_s3_client(api_key)
    days = _trading_days(start, end)
    print(f"[polygon/flatfiles] {symbol}: {len(days)} trading days to fetch", flush=True)

    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=_FLATFILES_WORKERS) as pool:
        futures = {pool.submit(_fetch_day_file, s3, d, symbol): d for d in days}
        done = 0
        for fut in as_completed(futures):
            df = fut.result()
            if not df.empty:
                frames.append(df)
            done += 1
            if done % 50 == 0:
                print(f"[polygon/flatfiles] {symbol}: {done}/{len(days)} days done", flush=True)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_minute_history(
    symbol: str,
    start: date | datetime = _DEFAULT_START,
    end: date | datetime | None = None,
    adjusted: bool = True,
    method: Optional[str] = None,
    api_key: Optional[str] = None,
    out_path: Optional[str] = None,
) -> str:
    """Download full minute-bar history for ``symbol`` and write CSV.

    Args:
        symbol:   Ticker (e.g. ``"NVDA"``).
        start:    Inclusive start. Default 2022-01-01.
        end:      Inclusive end. Defaults to today (UTC).
        adjusted: Return split-adjusted prices (API mode only; flat files are
                  always adjusted by Polygon before writing to S3).
        method:   ``"api"`` or ``"flatfiles"``. Defaults to
                  ``POLYGON_METHOD`` env var, then ``"api"``.
        api_key:  Polygon API key. Falls back to ``POLYGON_API_KEY`` env var.
        out_path: Override output CSV path.

    Returns:
        Absolute path to the written CSV.
    """
    key = api_key or os.environ.get("POLYGON_API_KEY", "")
    if not key:
        raise RuntimeError(
            "Polygon API key not found. Set the POLYGON_API_KEY environment variable."
        )

    method = method or os.environ.get("POLYGON_METHOD", "api")
    end = end or datetime.now(timezone.utc).date()
    if isinstance(start, datetime):
        start = start.date()
    if isinstance(end, datetime):
        end = end.date()

    csv_path = out_path or get_raw_data_csv_path(symbol)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"[polygon/{method}] {symbol}: {start} -> {end}", flush=True)

    if method == "flatfiles":
        df = _download_via_flatfiles(symbol, start, end, key)
    else:
        df = _download_via_api(symbol, start, end, adjusted, key)

    if df.empty:
        print(f"[polygon] {symbol}: no bars returned.")
        return csv_path

    df = df.sort_values("DateTime").drop_duplicates(subset=["DateTime"])
    df.to_csv(csv_path, index=False)
    print(f"[polygon] {symbol}: wrote {len(df):,} bars -> {csv_path}")
    return csv_path


def download_all_symbols(
    symbols: list[str],
    start: date | datetime = _DEFAULT_START,
    end: date | datetime | None = None,
    adjusted: bool = True,
    method: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict[str, str]:
    """Download minute history for each symbol. Returns ``{symbol: csv_path}``."""
    key = api_key or os.environ.get("POLYGON_API_KEY", "")
    results: dict[str, str] = {}
    for sym in symbols:
        try:
            path = download_minute_history(
                sym, start=start, end=end, adjusted=adjusted,
                method=method, api_key=key,
            )
            results[sym] = path
        except Exception as exc:
            print(f"[polygon] ERROR for {sym}: {exc}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--method", default=None, choices=["api", "flatfiles"])
    parser.add_argument("--unadjusted", action="store_true")
    args = parser.parse_args()

    download_all_symbols(
        symbols=args.symbols,
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end) if args.end else None,
        adjusted=not args.unadjusted,
        method=args.method,
    )
