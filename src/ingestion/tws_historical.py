"""TWS historical ingestion core.

This module contains the IB/TWS-specific logic for fetching historical bars
and writing them to the raw CSV used by the rest of the project.

Existing callers should prefer the public functions exposed here instead of
reaching into CLI scripts. `src.data_ingestion` re-exports key APIs for
backwards compatibility.
"""

import os
import sys
from datetime import datetime, timedelta
import asyncio
from typing import Iterable, Optional

import pandas as pd
from ib_insync import IB, Stock, util, Forex, CFD, Future

# Maximum simultaneous historical data requests as per IB docs.
# See: "The maximum number of simultaneous open historical data requests
# from the API is 50."
_IB_HIST_MAX_SIMULT_REQUESTS = 50

# Bar sizes considered "small" for pacing purposes (30 seconds or less).
_SMALL_BAR_SIZES = {
    "1 secs",
    "2 secs",
    "3 secs",
    "5 secs",
    "10 secs",
    "15 secs",
    "30 secs",
}

# Ensure we can import src.config when this module is used standalone.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (  # type: ignore  # noqa: E402
    RAW_DATA_CSV,
    TWS_HOST,
    TWS_PORT,
    TWS_CLIENT_ID,
    NVDA_CONTRACT_DETAILS,
    TWS_MAX_CONCURRENT_REQUESTS,
    DATA_BATCH_SAVE_SIZE,
)


def _get_latest_timestamp_from_csv(file_path: str) -> Optional[datetime]:
    """Return the latest DateTime from the CSV file, or ``None``.

    This is a low-level helper that the higher-level ingestion APIs use to
    implement incremental fetching.
    """
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return None

    try:
        # Read only the last line to get the latest timestamp efficiently
        with open(file_path, "rb") as f:
            f.seek(-2, os.SEEK_END)  # Jump to the second last byte
            while f.read(1) != b"\n":  # Until newline is found
                f.seek(-2, os.SEEK_CUR)  # Jump back two bytes
            last_line = f.readline().decode().strip()

        print(f"DEBUG: Last line read from {file_path}: '{last_line}'")
        # Parse the DateTime from the last line
        latest_dt_str = last_line.split(",")[0]
        print(f"DEBUG: Extracted DateTime string: '{latest_dt_str}'")
        return datetime.strptime(latest_dt_str, "%Y-%m-%d %H:%M:%S")
    except Exception as e:  # pragma: no cover - defensive
        print(f"Error reading latest timestamp from {file_path}: {e}")
        return None


async def _fetch_single_day_data(ib: IB,
                                 contract,
                                 end_date_str: str,
                                 barSizeSetting: str,
                                 semaphore: asyncio.Semaphore) -> pd.DataFrame:
    """Fetch data for a single day with rate limiting and return as DataFrame.

    Pacing considerations:
        * IB limits simultaneous historical requests to 50.
        * For small bar sizes (<= 30 seconds), additional pacing rules apply
          (e.g. no more than 6 requests for the same contract within 2 seconds).

    We rely on a shared semaphore to cap concurrent requests below IB's hard
    limit and keep effective concurrency modest for small bars.
    """
    async with semaphore:
        print(f"Requesting data ending {end_date_str} for duration '1 D'...")
        bars = await ib.reqHistoricalDataAsync(
            contract,
            endDateTime=end_date_str,
            durationStr="1 D",
            barSizeSetting=barSizeSetting,
            whatToShow="TRADES",
            useRTH=False,
            formatDate=1,
        )
        if bars:
            print(f"Fetched {len(bars)} bars ending {end_date_str}.")
            df = util.df(bars)
            # Rename columns to match expected format (DateTime, Open, High, Low, Close)
            df = df[["date", "open", "high", "low", "close"]]
            df.columns = ["DateTime", "Open", "High", "Low", "Close"]
            # Ensure DateTime is in the correct format
            df["DateTime"] = df["DateTime"].dt.strftime("%Y-%m-%d %H:%M:%S")
            return df
        else:
            print(f"No bars received for period ending {end_date_str}.")
            return pd.DataFrame()  # Return empty DataFrame if no bars


async def fetch_historical_data(
    contract_details: dict,
    end_date: datetime,
    start_date: datetime,
    barSizeSetting: str = "1 min",
    file_path: str = RAW_DATA_CSV,
    strict_range: bool = False,
) -> None:
    """Fetch historical data from IB and append it to a CSV file.

    Data is requested in daily chunks, optionally respecting existing data in
    ``file_path`` to avoid re-downloading overlapping periods.

    Args:
        contract_details: Contract description (symbol, secType, exchange, currency).
        end_date: End of the requested period (exclusive).
        start_date: Start of the requested period if no prior data exists, or
            when ``strict_range=True``.
        barSizeSetting: IB bar size (e.g. ``"1 min"``, ``"5 mins"``, ``"1 hour"``).
        file_path: CSV path where fetched data will be appended.
        strict_range: When ``True``, ignore existing CSV contents and always use
            ``start_date``/``end_date`` as the effective range.
    """
    ib = IB()

    try:
        await ib.connectAsync(TWS_HOST, TWS_PORT, TWS_CLIENT_ID)
        print(
            f"Connected to IB TWS/Gateway on {TWS_HOST}:{TWS_PORT} "
            f"with Client ID {TWS_CLIENT_ID}"
        )

        # Create contract based on secType
        sec_type = contract_details.get("secType", "STK")
        if sec_type == "STK":
            contract = Stock(
                symbol=contract_details["symbol"],
                exchange=contract_details["exchange"],
                currency=contract_details["currency"],
            )
        elif sec_type == "CASH":
            contract = Forex(
                pair=f"{contract_details['symbol']}{contract_details['currency']}"
            )
        elif sec_type == "CFD":
            contract = CFD(
                symbol=contract_details["symbol"],
                exchange=contract_details["exchange"],
                currency=contract_details["currency"],
            )
        elif sec_type == "FUT":
            contract = Future(
                symbol=contract_details["symbol"],
                lastTradeDateOrContractMonth=contract_details.get(
                    "lastTradeDateOrContractMonth"
                ),
                exchange=contract_details["exchange"],
                currency=contract_details["currency"],
            )
        else:
            raise ValueError(f"Unsupported security type: {sec_type}")

        # Ensure the contract details are resolved
        print(f"Qualifying contract: {contract.symbol}...")
        await ib.qualifyContractsAsync(contract)

        # Determine the actual start date for fetching
        actual_fetch_start_date = start_date
        if not strict_range:
            latest_timestamp_in_file = _get_latest_timestamp_from_csv(file_path)
            if latest_timestamp_in_file:
                # Start fetching from 1 minute after the latest timestamp in the file
                actual_fetch_start_date = latest_timestamp_in_file + timedelta(minutes=1)
                print(
                    "Latest data in file is "
                    f"{latest_timestamp_in_file}. Fetching new data from "
                    f"{actual_fetch_start_date}."
                )
            else:
                print(
                    f"No existing data found in {file_path}. "
                    f"Fetching from initial start date {start_date}."
                )
        else:
            print(f"Strict range fetching: from {start_date} to {end_date}.")

        if actual_fetch_start_date >= end_date:
            print("Data is already up to date for the specified range. No new data to fetch.")
            return

        print(
            "Fetching historical data for "
            f"{contract.symbol}/{contract.currency} from "
            f"{actual_fetch_start_date.strftime('%Y%m%d %H:%M:%S')} to "
            f"{end_date.strftime('%Y%m%d %H:%M:%S')}..."
        )

        # Determine effective concurrency based on IB pacing guidance.
        # - Hard cap at 50 simultaneous historical requests.
        # - For small bars (<= 30 secs) we keep concurrency very low to avoid
        #   6-requests-in-2-seconds pacing violations.
        if barSizeSetting in _SMALL_BAR_SIZES:
            effective_concurrency = min(TWS_MAX_CONCURRENT_REQUESTS, 2, _IB_HIST_MAX_SIMULT_REQUESTS)
        else:
            # For barSize >= 1 min, IB has lifted the hard pacing limits but
            # still implements soft throttling. We cap by both user config and
            # the documented 50-request ceiling.
            effective_concurrency = min(
                TWS_MAX_CONCURRENT_REQUESTS,
                _IB_HIST_MAX_SIMULT_REQUESTS,
            )

        if effective_concurrency <= 0:
            raise ValueError("TWS_MAX_CONCURRENT_REQUESTS must be >= 1")

        print(
            f"Using effective_concurrency={effective_concurrency} "
            f"(configured={TWS_MAX_CONCURRENT_REQUESTS}, barSize='{barSizeSetting}')"
        )

        # Generate all daily end dates for requests in reverse chronological order
        request_dates = []
        current_date = end_date
        while current_date > actual_fetch_start_date:
            request_dates.append(current_date)
            current_date -= timedelta(days=1)

        # Create a semaphore to limit concurrent requests according to
        # effective pacing-aware concurrency.
        semaphore = asyncio.Semaphore(effective_concurrency)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file_exists = os.path.exists(file_path)
        total_bars_fetched = 0

        # Process requests in batches for saving
        for i in range(0, len(request_dates), DATA_BATCH_SAVE_SIZE):
            batch_dates = request_dates[i : i + DATA_BATCH_SAVE_SIZE]

            tasks = []
            for req_date in batch_dates:
                end_date_str = req_date.strftime("%Y%m%d %H:%M:%S")
                tasks.append(
                    _fetch_single_day_data(
                        ib,
                        contract,
                        end_date_str,
                        barSizeSetting,
                        semaphore,
                    )
                )

            # Run tasks concurrently for the current batch
            results_batch = await asyncio.gather(*tasks)

            # Concatenate all DataFrames in the batch
            dfs_to_append = [df for df in results_batch if not df.empty]
            if dfs_to_append:
                combined_df_batch = pd.concat(dfs_to_append)
                combined_df_batch.sort_values("DateTime", inplace=True)  # Ensure chronological order

                combined_df_batch.to_csv(
                    file_path,
                    mode="a",
                    header=not file_exists,
                    index=False,
                )
                file_exists = True
                total_bars_fetched += len(combined_df_batch)
                print(
                    f"Appended {len(combined_df_batch)} bars (batch "
                    f"{i // DATA_BATCH_SAVE_SIZE + 1}) to {file_path}"
                )

        if total_bars_fetched == 0:
            print("No historical data received for the entire period.")
            return

        print(
            f"Successfully fetched a total of {total_bars_fetched} bars "
            f"and saved to {file_path}"
        )

    except Exception as e:  # pragma: no cover - defensive
        print(f"An error occurred: {e}")
    finally:
        if ib and hasattr(ib, "disconnect") and callable(ib.disconnect):
            try:
                await ib.disconnect()
                print("Disconnected from IB TWS/Gateway.")
            except TypeError:
                print(
                    "Warning: TypeError encountered during IB TWS/Gateway "
                    "disconnection. Connection might already be closed."
                )
        else:  # pragma: no cover - defensive
            print(
                "Could not disconnect from IB TWS/Gateway (ib object or "
                "disconnect method not available)."
            )


def trigger_historical_ingestion(
    symbols: Iterable[str],
    start: datetime,
    end: datetime,
    bar_size: str = "1 min",
    strict_range: bool = False,
    file_path: Optional[str] = None,
) -> None:
    """High-level entrypoint to fetch historical data for one or more symbols.

    For now, this is a thin convenience wrapper around :func:`fetch_historical_data`
    using the existing NVDA contract details from ``src.config``. It is scoped to
    the current single-symbol use case in this repo but exposes a future-friendly
    API.
    """
    # v1 implementation: only NVDA is supported; ignore symbols other than NVDA.
    # This keeps behavior aligned with the existing pipeline while giving us a
    # stable interface to hang future multi-symbol support on.
    symbols = list(symbols)
    if not symbols:
        raise ValueError("At least one symbol must be provided")

    if len(symbols) > 1 or symbols[0].upper() != "NVDA":
        print(
            "trigger_historical_ingestion currently only supports NVDA. "
            f"Received symbols={symbols}. Using NVDA contract details."
        )

    target_file = file_path or RAW_DATA_CSV
    asyncio.run(
        fetch_historical_data(
            contract_details=NVDA_CONTRACT_DETAILS,
            end_date=end,
            start_date=start,
            barSizeSetting=bar_size,
            file_path=target_file,
            strict_range=strict_range,
        )
    )