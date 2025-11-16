"""Backwards-compatible CLI and wrapper for TWS historical ingestion.

The core ingestion logic now lives in :mod:`src.ingestion.tws_historical`.
This module is kept as a thin wrapper to preserve existing imports and
command-line usage.
"""

from datetime import datetime

from src.ingestion.tws_historical import (  # type: ignore
    _get_latest_timestamp_from_csv,
    fetch_historical_data,
    trigger_historical_ingestion,
)
from src.config import NVDA_CONTRACT_DETAILS  # type: ignore


__all__ = [
    "_get_latest_timestamp_from_csv",
    "fetch_historical_data",
    "trigger_historical_ingestion",
]


if __name__ == "__main__":
    # Preserve the previous behavior: run a historical ingestion for NVDA from
    # 2024-01-01 until now into the default RAW_DATA_CSV.
    print("Running TWS data ingestion for NVDA via trigger_historical_ingestion()...")

    end_date = datetime.now()
    default_start_date = datetime(2024, 1, 1)

    # Use the new high-level API; this currently still delegates to the same
    # underlying implementation as before, but is a stable hook for future
    # refactors.
    trigger_historical_ingestion(
        symbols=[NVDA_CONTRACT_DETAILS["symbol"]],
        start=default_start_date,
        end=end_date,
    )
