import os
import sys
import json
import asyncio
from datetime import datetime

import pandas as pd

from src.config import (
    PROCESSED_DATA_DIR,
    FREQUENCY,
    CONTRACT_REGISTRY,
    get_raw_data_csv_path,
)
from src.ingestion import (
    fetch_historical_data,
    CURATED_MINUTE_PATH,
    run_transform_minute_bars,
)
from src.data_processing import (
    clean_raw_minute_data,
    convert_minute_to_timeframe,
    prepare_keras_input_data,
    fill_gaps,
)
from src.data_quality import analyze_raw_minute_data, compute_quality_kpi


GAP_ANALYSIS_OUTPUT_JSON = os.path.join(PROCESSED_DATA_DIR, "missing_trading_days.json")


async def _run_subprocess(cmd: list[str]) -> dict:
    """Async helper to run a subprocess and capture stdout/stderr.

    Note: ``asyncio.create_subprocess_exec`` does not support ``text=True``.
    We work with bytes and decode to UTF-8.
    """
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_b, stderr_b = await process.communicate()
    stdout = stdout_b.decode("utf-8", errors="replace") if stdout_b is not None else ""
    stderr = stderr_b.decode("utf-8", errors="replace") if stderr_b is not None else ""
    return {"stdout": stdout, "stderr": stderr}


def run_gap_analysis(raw_csv_path: str) -> list[dict]:
    """Run analyze_gaps.py on *raw_csv_path* and return the parsed gaps list."""
    analyze_gaps_script_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "analyze_gaps.py")
    )

    cmd = [sys.executable, analyze_gaps_script_path, raw_csv_path, GAP_ANALYSIS_OUTPUT_JSON]
    print(f"[agent] Running gap analysis: {' '.join(cmd)}")

    try:
        result = asyncio.run(_run_subprocess(cmd))
        if result["stdout"]:
            print("[agent] analyze_gaps.py stdout:")
            print(result["stdout"])
        if result["stderr"]:
            print("[agent] analyze_gaps.py stderr:")
            print(result["stderr"])
    except Exception as e:
        print(f"[agent] Error running analyze_gaps.py: {e}")
        return []

    if not os.path.exists(GAP_ANALYSIS_OUTPUT_JSON):
        print("[agent] No gap analysis JSON found; assuming no gaps.")
        return []

    try:
        with open(GAP_ANALYSIS_OUTPUT_JSON, "r") as f:
            gaps = json.load(f)
    except json.JSONDecodeError:
        print(f"[agent] Could not decode {GAP_ANALYSIS_OUTPUT_JSON}; treating as no gaps.")
        return []

    print(f"[agent] Loaded {len(gaps)} gaps from {GAP_ANALYSIS_OUTPUT_JSON}")
    return gaps


async def ingest_new_data(symbol: str) -> None:
    """Fetch new minute data for *symbol* and append to its raw CSV.

    Bootstrap strategy:
      - Raw CSV absent → one-time Alpaca historical dump (2022-01-01 → now).
        Requires ALPACA_API_KEY and ALPACA_SECRET_KEY env vars.
      - Raw CSV present → IBKR incremental update (appends bars beyond last row).

    The IBKR path requires TWS/Gateway to be running.
    """
    if symbol not in CONTRACT_REGISTRY:
        raise ValueError(f"Symbol '{symbol}' not found in CONTRACT_REGISTRY. Add it to src/config.py.")

    raw_csv = get_raw_data_csv_path(symbol)
    end_date = datetime.now()

    if not os.path.exists(raw_csv):
        print(
            f"[agent] No raw CSV for {symbol}; running one-time Alpaca historical dump."
        )
        from src.ingestion.polygon_historical import download_minute_history
        from datetime import date as _date
        download_minute_history(
            symbol,
            start=_date(2022, 1, 1),
            end=None,  # today
        )
        return

    print(
        f"[agent] Raw CSV found for {symbol}; running IBKR incremental update → {raw_csv}"
    )
    default_start_date = datetime(2024, 1, 1)
    await fetch_historical_data(
        contract_details=CONTRACT_REGISTRY[symbol],
        end_date=end_date,
        start_date=default_start_date,
        file_path=raw_csv,
        strict_range=False,
    )


async def _backfill_gaps_from_tws(gaps: list[dict], symbol: str) -> None:
    """Attempt to backfill each identified gap using TWS historical data for *symbol*.

    For each gap, we request historical minute bars from IB/TWS using the
    ingestion core (`fetch_historical_data`) with `strict_range=True` so that
    only the missing window is fetched. Any failures are logged by the
    underlying ingestion code.
    """
    if not gaps:
        return

    raw_csv = get_raw_data_csv_path(symbol)
    contract = CONTRACT_REGISTRY[symbol]
    print(f"[agent] Attempting TWS backfill for {len(gaps)} gaps ({symbol})...")

    for gap in gaps:
        try:
            start = pd.to_datetime(gap["start"])
            end = pd.to_datetime(gap["end"])
        except Exception as e:
            print(f"[agent] Warning: could not parse gap entry {gap}: {e}")
            continue

        # Guard against obviously invalid ranges.
        if start >= end:
            print(f"[agent] Skipping gap with non-positive range: {gap}")
            continue

        print(
            f"[agent] TWS backfill for gap from {start} to {end} "
            f"(duration {pd.to_timedelta(gap.get('duration', 'unknown'))})."
        )

        await fetch_historical_data(
            contract_details=contract,
            end_date=end,
            start_date=start,
            file_path=raw_csv,
            strict_range=True,
        )


def smart_fill_gaps(symbol: str, skip_tws: bool = False) -> None:
    """Apply TWS-based backfill + forward-fill gap handling for *symbol*'s raw CSV.

    Strategy:
        1. Run `analyze_gaps.py` to detect long weekday gaps.
        2. For each gap, attempt to backfill from TWS using historical
           ingestion with `strict_range=True` (skipped when skip_tws=True).
        3. Re-run gap analysis; any remaining gaps are assumed to be genuine
           upstream holes and are filled via `fill_gaps` (forward-fill).
    """
    raw_csv = get_raw_data_csv_path(symbol)

    if not os.path.exists(raw_csv):
        print(f"[agent] Raw CSV not found at {raw_csv}; skipping gap filling for {symbol}.")
        return

    # 1) Initial gap detection.
    gaps = run_gap_analysis(raw_csv)
    if not gaps:
        print(f"[agent] No long weekday gaps detected for {symbol}; skipping backfill/fill.")
        return

    if not os.path.exists(raw_csv):
        print(f"[agent] Raw CSV not found at {raw_csv} after gap analysis; skipping.")
        return

    # 2) Attempt TWS backfill for identified gaps (skip when offline / skip_tws).
    if skip_tws:
        print(f"[agent] Skipping TWS gap backfill for {symbol} (skip_tws=True).")
    else:
        try:
            asyncio.run(_backfill_gaps_from_tws(gaps, symbol))
        except RuntimeError as e:
            # In case we're already inside an event loop (unlikely here), just log.
            print(f"[agent] Warning: could not run TWS backfill coroutine: {e}")

    # 3) Re-run gap analysis to see what remains after TWS backfill.
    remaining_gaps = run_gap_analysis(raw_csv)

    initial_gap_count = len(gaps)
    remaining_gap_count = len(remaining_gaps)
    tws_fixed_count = max(initial_gap_count - remaining_gap_count, 0)

    print(
        "[agent] Gap summary: total_detected={initial}, "
        "fixed_by_tws={fixed}, forward_filled={ff}".format(
            initial=initial_gap_count,
            fixed=tws_fixed_count,
            ff=remaining_gap_count,
        )
    )

    if not remaining_gaps:
        print(f"[agent] No remaining long weekday gaps after TWS backfill for {symbol}; skipping forward-fill.")
        return

    # 4) Forward-fill remaining gaps as a fallback.
    if not os.path.exists(raw_csv):
        print(f"[agent] Raw CSV not found at {raw_csv} before forward-fill; skipping.")
        return

    df = pd.read_csv(raw_csv, parse_dates=["DateTime"])
    df_filled = fill_gaps(df, remaining_gaps)
    df_filled.to_csv(raw_csv, index=False)
    print(f"[agent] Saved gap-filled raw data back to {raw_csv}.")


def resample_and_add_features(symbol: str = "NVDA") -> None:
    """Convert curated minute data to hourly and add features for FREQUENCY."""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    curated_path = os.path.join(PROCESSED_DATA_DIR, f"{symbol.lower()}_minute_curated.csv")
    print(f"[agent] Converting curated minute data to {FREQUENCY} and saving to processed dir.")
    hourly_path = convert_minute_to_timeframe(curated_path, FREQUENCY, PROCESSED_DATA_DIR, symbol=symbol)

    from src.config import FEATURES_TO_USE_OPTIONS

    features = FEATURES_TO_USE_OPTIONS[0]

    if not os.path.exists(hourly_path):
        print(f"[agent] Hourly data CSV not found at {hourly_path}; skipping feature engineering.")
        return

    df_featured, feature_cols = prepare_keras_input_data(hourly_path, features)
    print(f"[agent] Prepared Keras input data with features: {feature_cols}")
    # Optional: persist featured data for inspection
    # featured_path = os.path.join(PROCESSED_DATA_DIR, "featured_hourly_data.csv")
    # df_featured.to_csv(featured_path, index=False)


def run_daily_pipeline(
    skip_ingestion: bool = False,
    symbols: list[str] | None = None,
) -> None:
    """Top-level entry point to be run once per day.

    Args:
        skip_ingestion: When True, skip IB/TWS data ingestion. Useful for dry runs.
        symbols: List of symbols to process. Defaults to the portfolio symbol list
                 from configs/portfolio.json (via get_configured_symbols()).
    """
    from src.core.config_resolver import get_configured_symbols
    if symbols is None:
        symbols = get_configured_symbols()

    print(f"[agent] --- Daily Data Pipeline Agent start | symbols={symbols} ---")

    # 1) Ingest + clean + gap-fill per symbol
    for sym in symbols:
        raw_csv = get_raw_data_csv_path(sym)
        os.makedirs(os.path.dirname(raw_csv), exist_ok=True)

        if skip_ingestion:
            print(f"[agent] Skipping IB data ingestion for {sym} (dry run mode).")
        else:
            asyncio.run(ingest_new_data(sym))

        # 2) Clean and deduplicate
        if os.path.exists(raw_csv):
            print(f"[agent] Cleaning raw minute data for {sym}...")
            clean_raw_minute_data(raw_csv)
        else:
            print(f"[agent] Raw CSV not found at {raw_csv}; skipping cleaning for {sym}.")

        # 3) Analyze and fill gaps
        print(f"[agent] Analyzing and filling gaps for {sym}...")
        smart_fill_gaps(sym, skip_tws=skip_ingestion)

        # 3b) Raw data quality snapshot
        if os.path.exists(raw_csv):
            print(f"[agent] Running raw minute data quality checks for {sym}...")
            checks = analyze_raw_minute_data(raw_csv)
            kpi = compute_quality_kpi(checks)
            print(
                "[agent] Data quality snapshot | symbol={sym} | score={score:.1f} / 100 | "
                "pass={n_pass} warn={n_warn} fail={n_fail}".format(
                    sym=sym,
                    score=kpi.get("score_0_100", 0.0),
                    n_pass=kpi.get("n_pass", 0),
                    n_warn=kpi.get("n_warn", 0),
                    n_fail=kpi.get("n_fail", 0),
                )
            )

    # 4) Create curated-minute snapshot for each symbol
    for sym in symbols:
        print(f"[agent] Processing curated-minute snapshot for {sym}...")
        run_transform_minute_bars(sym)

    # 5) Resample to hourly and engineer features
    for sym in symbols:
        resample_and_add_features(sym)

    print("[agent] --- Daily Data Pipeline Agent completed ---")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Daily data pipeline agent.")
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Skip IB/TWS data ingestion (dry run).",
    )
    parser.add_argument(
        "--symbol",
        default="NVDA",
        help="Ticker symbol to process (default: NVDA).",
    )
    args = parser.parse_args()
    run_daily_pipeline(skip_ingestion=args.skip_ingestion, symbols=[args.symbol])
