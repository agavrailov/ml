# src/ingestion/__init__.py

"""Ingestion-related building blocks.

This package hosts the refactored ingestion core (e.g. TWS historical
ingestion, raw/curated minute bars, and simple access helpers). Existing
scripts like ``src/data_ingestion.py`` and ``daily_data_agent`` should
prefer delegating to these modules over time.
"""

from .tws_historical import (  # noqa: F401
    _get_latest_timestamp_from_csv,
    fetch_historical_data,
    trigger_historical_ingestion,
)
from .curated_minute import (  # noqa: F401
    CURATED_MINUTE_PATH,
    get_curated_bars,
    get_raw_bars,
    run_transform_minute_bars,
)

__all__ = [
    "_get_latest_timestamp_from_csv",
    "fetch_historical_data",
    "trigger_historical_ingestion",
    "CURATED_MINUTE_PATH",
    "run_transform_minute_bars",
    "get_raw_bars",
    "get_curated_bars",
]
