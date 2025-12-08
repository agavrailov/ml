import os
import sys
import json
import asyncio
from datetime import datetime

import pandas as pd

from src.config import (
    RAW_DATA_CSV,
    PROCESSED_DATA_DIR,
    FREQUENCY,
    NVDA_CONTRACT_DETAILS,
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


def run_gap_analysis() -> list[dict]:
    """Run analyze_gaps.py and return the parsed gaps list.

    This function acts as the "gap analysis tool" for the daily agent.
    """
    analyze_gaps_script_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "analyze_gaps.py")
    )

    cmd = [sys.executable, analyze_gaps_script_path, RAW_DATA_CSV, GAP_ANALYSIS_OUTPUT_JSON]
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


async def ingest_new_data() -> None:
    """Fetch new minute data from IB and append to RAW_DATA_CSV.

    The underlying fetch_historical_data function will only pull data beyond the
    last timestamp present in RAW_DATA_CSV when strict_range=False.
    It already uses the configured TWS_MAX_CONCURRENT_REQUESTS semaphore to push
    IB concurrency to your configured limit safely.
    """
    end_date = datetime.now()
    default_start_date = datetime(2024, 1, 1)

    print(
        f"[agent] Starting ingestion from {default_start_date} to {end_date} "
        f"into {RAW_DATA_CSV}"
    )
    await fetch_historical_data(
        contract_details=NVDA_CONTRACT_DETAILS,
        end_date=end_date,
        start_date=default_start_date,
        file_path=RAW_DATA_CSV,
        strict_range=False,
    )


async def _backfill_gaps_from_tws(gaps: list[dict]) -> None:
    """Attempt to backfill each identified gap using TWS historical data.

    For each gap, we request historical minute bars from IB/TWS using the
    ingestion core (`fetch_historical_data`) with `strict_range=True` so that
    only the missing window is fetched. Any failures are logged by the
    underlying ingestion code.
    """
    if not gaps:
        return

    print(f"[agent] Attempting TWS backfill for {len(gaps)} gaps...")

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
            contract_details=NVDA_CONTRACT_DETAILS,
            end_date=end,
            start_date=start,
            file_path=RAW_DATA_CSV,
            strict_range=True,
        )


def smart_fill_gaps() -> None:
    """Apply TWS-based backfill + forward-fill gap handling on RAW_DATA_CSV.

    Strategy:
        1. Run `analyze_gaps.py` to detect long weekday gaps.
        2. For each gap, attempt to backfill from TWS using historical
           ingestion with `strict_range=True`.
        3. Re-run gap analysis; any remaining gaps are assumed to be genuine
           upstream holes and are filled via `fill_gaps` (forward-fill).
    """
    if not os.path.exists(RAW_DATA_CSV):
        print(f"[agent] RAW_DATA_CSV not found at {RAW_DATA_CSV}; skipping gap filling.")
        return

    # 1) Initial gap detection.
    gaps = run_gap_analysis()
    if not gaps:
        print("[agent] No long weekday gaps detected; skipping backfill/fill.")
        return

    if not os.path.exists(RAW_DATA_CSV):
        print(f"[agent] RAW_DATA_CSV not found at {RAW_DATA_CSV} after gap analysis; skipping.")
        return

    # 2) Attempt TWS backfill for identified gaps.
    try:
        asyncio.run(_backfill_gaps_from_tws(gaps))
    except RuntimeError as e:
        # In case we're already inside an event loop (unlikely here), just log.
        print(f"[agent] Warning: could not run TWS backfill coroutine: {e}")

    # 3) Re-run gap analysis to see what remains after TWS backfill.
    remaining_gaps = run_gap_analysis()

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
        print("[agent] No remaining long weekday gaps after TWS backfill; skipping forward-fill.")
        return

    # 4) Forward-fill remaining gaps as a fallback.
    if not os.path.exists(RAW_DATA_CSV):
        print(f"[agent] RAW_DATA_CSV not found at {RAW_DATA_CSV} before forward-fill; skipping.")
        return

    df = pd.read_csv(RAW_DATA_CSV, parse_dates=["DateTime"])
    df_filled = fill_gaps(df, remaining_gaps)
    df_filled.to_csv(RAW_DATA_CSV, index=False)
    print("[agent] Saved gap-filled raw data back to RAW_DATA_CSV.")


def resample_and_add_features() -> None:
    """Convert curated minute data to hourly and add features for FREQUENCY."""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    print(f"[agent] Converting curated minute data to {FREQUENCY} and saving to processed dir.")
    convert_minute_to_timeframe(CURATED_MINUTE_PATH, FREQUENCY, PROCESSED_DATA_DIR)

    from src.config import FEATURES_TO_USE_OPTIONS, get_hourly_data_csv_path

    features = FEATURES_TO_USE_OPTIONS[0]
    hourly_path = get_hourly_data_csv_path(FREQUENCY, PROCESSED_DATA_DIR)

    if not os.path.exists(hourly_path):
        print(f"[agent] Hourly data CSV not found at {hourly_path}; skipping feature engineering.")
        return

    df_featured, feature_cols = prepare_keras_input_data(hourly_path, features)
    print(f"[agent] Prepared Keras input data with features: {feature_cols}")
    # Optional: persist featured data for inspection
    # featured_path = os.path.join(PROCESSED_DATA_DIR, "featured_hourly_data.csv")
    # df_featured.to_csv(featured_path, index=False)


def run_daily_pipeline(skip_ingestion: bool = False) -> None:
    """Top-level entry point to be run once per day.

    Args:
        skip_ingestion: When True, skip IB/TWS data ingestion. Useful for dry runs.
    """
    print("[agent] --- Daily Data Pipeline Agent start ---")

    os.makedirs(os.path.dirname(RAW_DATA_CSV), exist_ok=True)

    # 1) Ingest new raw minute data
    if skip_ingestion:
        print("[agent] Skipping IB data ingestion (dry run mode).")
    else:
        asyncio.run(ingest_new_data())

    # 2) Clean and deduplicate
    if os.path.exists(RAW_DATA_CSV):
        print("[agent] Cleaning raw minute data...")
        clean_raw_minute_data(RAW_DATA_CSV)
    else:
        print(f"[agent] RAW_DATA_CSV not found at {RAW_DATA_CSV}; skipping cleaning.")

    # 3) Analyze and fill gaps
    print("[agent] Analyzing and filling gaps...")
    smart_fill_gaps()

    # 3b) Run a raw data quality snapshot so that any severe issues are
    # visible in logs even when no UI is running. We do *not* fail the daily
    # pipeline on warnings or failures here; that decision can be made by the
    # caller based on the summary.
    if os.path.exists(RAW_DATA_CSV):
        print("[agent] Running raw minute data quality checks (snapshot)...")
        checks = analyze_raw_minute_data(RAW_DATA_CSV)
        kpi = compute_quality_kpi(checks)
        print(
            "[agent] Data quality snapshot | score={score:.1f} / 100 | "
            "pass={n_pass} warn={n_warn} fail={n_fail}".format(
                score=kpi.get("score_0_100", 0.0),
                n_pass=kpi.get("n_pass", 0),
                n_warn=kpi.get("n_warn", 0),
                n_fail=kpi.get("n_fail", 0),
            )
        )

    # 4) Create curated-minute snapshot from cleaned, gap-filled raw data
    print("[agent] Creating curated-minute snapshot...")
    run_transform_minute_bars("NVDA")

    # 5) Resample to hourly and engineer features
    resample_and_add_features()

    print("[agent] --- Daily Data Pipeline Agent completed ---")


if __name__ == "__main__":
    # Simple CLI: allow `--skip-ingestion` for dry runs
    skip = "--skip-ingestion" in sys.argv
    run_daily_pipeline(skip_ingestion=skip)
