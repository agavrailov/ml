"""Update historical data for live trading warmup.

Run this before starting a live session to ensure the predictor has fresh data
for warmup. This is especially important for large timeframes (e.g. 240min)
where waiting for 24+ live bars would take days.

Usage:
    python -m src.update_live_data --frequency 240min
"""

import argparse
import asyncio
import os
import sys

from src.config import (
    RAW_DATA_CSV,
    PROCESSED_DATA_DIR,
    NVDA_CONTRACT_DETAILS,
)
from src.daily_data_agent import ingest_new_data
from src.data_processing import clean_raw_minute_data, convert_minute_to_timeframe
from src.ingestion import CURATED_MINUTE_PATH, run_transform_minute_bars


def update_data_for_frequency(frequency: str) -> None:
    """Update raw minute data and resample to target frequency.
    
    Args:
        frequency: Target bar frequency (e.g. "240min", "60min")
    """
    print(f"[update_live_data] Updating data for frequency={frequency}")
    
    # 1. Ingest new minute data from IBKR
    print("[update_live_data] Step 1: Fetching new minute data from IBKR...")
    asyncio.run(ingest_new_data())
    
    # 2. Clean raw data
    if os.path.exists(RAW_DATA_CSV):
        print("[update_live_data] Step 2: Cleaning raw minute data...")
        clean_raw_minute_data(RAW_DATA_CSV)
    else:
        print(f"[update_live_data] Warning: RAW_DATA_CSV not found at {RAW_DATA_CSV}")
    
    # 3. Create curated minute snapshot
    print("[update_live_data] Step 3: Creating curated minute snapshot...")
    run_transform_minute_bars("NVDA")
    
    # 4. Resample to target frequency
    print(f"[update_live_data] Step 4: Resampling to {frequency}...")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    convert_minute_to_timeframe(CURATED_MINUTE_PATH, frequency, PROCESSED_DATA_DIR)
    
    from src.config import get_hourly_data_csv_path
    output_path = get_hourly_data_csv_path(frequency)
    
    if os.path.exists(output_path):
        print(f"[update_live_data] ✓ Data updated successfully: {output_path}")
    else:
        print(f"[update_live_data] ✗ Failed to create {output_path}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update historical data for live trading warmup"
    )
    parser.add_argument(
        "--frequency",
        type=str,
        default="60min",
        help="Target bar frequency (e.g. 240min, 60min)",
    )
    args = parser.parse_args()
    
    update_data_for_frequency(args.frequency)


if __name__ == "__main__":
    main()
