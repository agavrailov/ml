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


def _parse_args() -> tuple[list[str], datetime, datetime, str, bool, str | None]:
    """Parse CLI arguments for historical ingestion.

    This provides a minimal but flexible interface while preserving the previous
    default behavior when no arguments are supplied.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Trigger TWS historical ingestion into the raw NVDA CSV. "
            "Defaults match the previous hard-coded behavior if no arguments "
            "are provided."
        )
    )
    parser.add_argument(
        "--symbol",
        default=NVDA_CONTRACT_DETAILS["symbol"],
        help=(
            "Symbol to ingest (currently only NVDA is supported; other "
            "symbols are accepted but ignored by the underlying implementation"
        ),
    )
    parser.add_argument(
        "--start",
        help="Start date (YYYY-MM-DD). Default: 2024-01-01.",
    )
    parser.add_argument(
        "--end",
        help="End date (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--bar-size",
        default="1 min",
        help="IB barSizeSetting, e.g. '1 min', '5 mins'. Default: '1 min'.",
    )
    parser.add_argument(
        "--strict-range",
        action="store_true",
        help=(
            "If set, always fetch exactly the [start, end) range, ignoring "
            "existing data in the CSV."
        ),
    )
    parser.add_argument(
        "--file-path",
        default=None,
        help=(
            "Optional override for the raw CSV path. When omitted, the "
            "configured RAW_DATA_CSV is used."
        ),
    )

    args = parser.parse_args()

    # Defaults to preserve previous behavior when flags are omitted.
    symbols = [args.symbol]
    start = (
        datetime.strptime(args.start, "%Y-%m-%d")
        if args.start
        else datetime(2024, 1, 1)
    )
    end = (
        datetime.strptime(args.end, "%Y-%m-%d")
        if args.end
        else datetime.now()
    )
    bar_size = args.bar_size
    strict_range = bool(args.strict_range)
    file_path = args.file_path

    return symbols, start, end, bar_size, strict_range, file_path


def main() -> None:
    """CLI entrypoint for historical ingestion.

    This replaces the previous hard-coded __main__ behavior with an argparse-
    based interface while keeping identical defaults when called without
    arguments.
    """
    symbols, start, end, bar_size, strict_range, file_path = _parse_args()

    print(
        "Running TWS historical ingestion via trigger_historical_ingestion() "
        f"for symbols={symbols}, start={start}, end={end}, bar_size='{bar_size}'."
    )

    trigger_historical_ingestion(
        symbols=symbols,
        start=start,
        end=end,
        bar_size=bar_size,
        strict_range=strict_range,
        file_path=file_path,
    )


if __name__ == "__main__":
    main()
