import sys
import os
import pandas as pd
from ib_insync import IB, Stock, util, Forex, CFD, Contract, Future
from datetime import datetime, timedelta, time
import asyncio
import exchange_calendars as xcals

# Add the project root to the Python path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import (
    RAW_DATA_CSV, TWS_HOST, TWS_PORT, TWS_CLIENT_ID, NVDA_CONTRACT_DETAILS,
    TWS_MAX_CONCURRENT_REQUESTS, INITIAL_START_DATE, MARKET_OPEN_TIME,
    MARKET_CLOSE_TIME, MARKET_TIMEZONE, EXCHANGE_CALENDAR_NAME
)
from src.data_ingestion import _get_latest_timestamp_from_csv, fetch_historical_data
from src.data_processing import clean_raw_minute_data

def _get_market_calendar(calendar_name):
    """Initializes and returns the specified market calendar."""
    try:
        return xcals.get_calendar(calendar_name)
    except ValueError:
        print(f"Error: Market calendar '{calendar_name}' not found. Using default NYSE.")
        return xcals.get_calendar('XNYS')

def _is_market_open(dt_obj, market_calendar):
    """
    Checks if a given datetime object falls within market hours on a trading day.
    Assumes dt_obj is timezone-aware or in the market's local timezone.
    """
    # Convert to market timezone if not already
    if dt_obj.tzinfo is None:
        market_tz = pd.Timestamp(dt_obj).tz_localize(MARKET_TIMEZONE)
    else:
        market_tz = dt_obj.astimezone(MARKET_TIMEZONE)

    date = market_tz.date()
    
    # Check if it's a trading day
    if not market_calendar.is_session(date):
        return False

    # Check if within market hours
    market_open_dt = market_tz.replace(hour=MARKET_OPEN_TIME.hour, minute=MARKET_OPEN_TIME.minute, second=0, microsecond=0)
    market_close_dt = market_tz.replace(hour=MARKET_CLOSE_TIME.hour, minute=MARKET_CLOSE_TIME.minute, second=0, microsecond=0)

    return market_open_dt <= market_tz < market_close_dt

async def update_historical_data():
    """
    Orchestrates the continuous update and gap filling of historical minute-level data.
    """
    print("Starting historical data update and gap filling process...")
    
    market_calendar = _get_market_calendar(EXCHANGE_CALENDAR_NAME)

    # 1. Load existing data
    existing_df = pd.DataFrame()
    if os.path.exists(RAW_DATA_CSV) and os.path.getsize(RAW_DATA_CSV) > 0:
        existing_df = pd.read_csv(RAW_DATA_CSV, parse_dates=['DateTime'])
        existing_df.set_index('DateTime', inplace=True)
        existing_df.sort_index(inplace=True)
        # Remove duplicates based on index (DateTime)
        existing_df = existing_df[~existing_df.index.duplicated(keep='first')]
        print(f"Loaded {len(existing_df)} existing records from {RAW_DATA_CSV}")
    else:
        print(f"No existing data found at {RAW_DATA_CSV}. Starting initial ingestion.")

    # Determine the actual start date for fetching new data
    latest_timestamp_in_file = _get_latest_timestamp_from_csv(RAW_DATA_CSV)
    fetch_start_date = latest_timestamp_in_file + timedelta(minutes=1) if latest_timestamp_in_file else INITIAL_START_DATE
    
    current_time = datetime.now()

    # 2. Fetch new data (from last record to now)
    print(f"Fetching new data from {fetch_start_date} to {current_time}...")
    await fetch_historical_data(
        contract_details=NVDA_CONTRACT_DETAILS,
        end_date=current_time,
        initial_start_date=fetch_start_date,
        file_path=RAW_DATA_CSV # fetch_historical_data will append to this file
    )
    
    # Reload data after fetching new data to get the most up-to-date dataset
    existing_df = pd.read_csv(RAW_DATA_CSV, parse_dates=['DateTime'])
    existing_df.set_index('DateTime', inplace=True)
    existing_df.sort_index(inplace=True)
    existing_df = existing_df[~existing_df.index.duplicated(keep='first')]
    print(f"Reloaded {len(existing_df)} records after fetching new data.")

    # 3. Identify and fill historical gaps
    print("Identifying and filling historical gaps...")
    
    # Generate a complete list of expected minute timestamps within market hours
    # from the earliest data point to the latest.
    if not existing_df.empty:
        min_date = existing_df.index.min().date()
        max_date = existing_df.index.max().date()
        
        all_expected_minutes = []
        for session_label in market_calendar.sessions_in_range(min_date, max_date):
            session_start = market_calendar.session_open(session_label).tz_convert(MARKET_TIMEZONE)
            session_end = market_calendar.session_close(session_label).tz_convert(MARKET_TIMEZONE)
            
            current_minute = session_start.replace(hour=MARKET_OPEN_TIME.hour, minute=MARKET_OPEN_TIME.minute)
            while current_minute < session_end.replace(hour=MARKET_CLOSE_TIME.hour, minute=MARKET_CLOSE_TIME.minute):
                if _is_market_open(current_minute, market_calendar):
                    all_expected_minutes.append(current_minute)
                current_minute += timedelta(minutes=1)
        
        expected_series = pd.Series(index=pd.to_datetime(all_expected_minutes))
        
        # Find missing timestamps
        missing_timestamps = expected_series.index.difference(existing_df.index)
        
        if not missing_timestamps.empty:
            print(f"Found {len(missing_timestamps)} missing minute bars. Attempting to fill gaps.")
            
            # Group missing timestamps into contiguous intervals
            gaps_to_fill = []
            if len(missing_timestamps) > 0:
                current_gap_start = missing_timestamps[0]
                current_gap_end = missing_timestamps[0]
                
                for i in range(1, len(missing_timestamps)):
                    if missing_timestamps[i] == current_gap_end + timedelta(minutes=1):
                        current_gap_end = missing_timestamps[i]
                    else:
                        gaps_to_fill.append((current_gap_start, current_gap_end))
                        current_gap_start = missing_timestamps[i]
                        current_gap_end = missing_timestamps[i]
                gaps_to_fill.append((current_gap_start, current_gap_end)) # Add the last gap

            for gap_start, gap_end in gaps_to_fill:
                print(f"Filling gap from {gap_start} to {gap_end}...")
                # Fetch data for the gap. Note: IB API endDateTime is exclusive, so we need to go one minute past the gap_end
                await fetch_historical_data(
                    contract_details=NVDA_CONTRACT_DETAILS,
                    end_date=gap_end + timedelta(minutes=1), # Fetch up to and including gap_end
                    initial_start_date=gap_start,
                    file_path=RAW_DATA_CSV,
                    strict_range=True # Ensure strict range for gap filling
                )
            
            # Reload data after filling gaps
            existing_df = pd.read_csv(RAW_DATA_CSV, parse_dates=['DateTime'])
            existing_df.set_index('DateTime', inplace=True)
            existing_df.sort_index(inplace=True)
            existing_df = existing_df[~existing_df.index.duplicated(keep='first')]
            print(f"Reloaded {len(existing_df)} records after filling gaps.")
        else:
            print("No historical gaps identified.")
    else:
        print("No existing data to check for historical gaps.")

    # 4. Final clean-up: sort and deduplicate the entire dataset
    print("Performing final sort and deduplication...")
    clean_raw_minute_data(RAW_DATA_CSV)
    
    print("Historical data update and gap filling process completed.")

if __name__ == "__main__":
    asyncio.run(update_historical_data())
