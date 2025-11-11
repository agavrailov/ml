import sys
import os
import pandas as pd
from ib_insync import IB, Stock, util, Forex, CFD, Contract, Future
from datetime import datetime, timedelta
import asyncio

def _get_latest_timestamp_from_csv(file_path):
    """
    Reads the latest DateTime from the CSV file.
    Returns None if the file does not exist or is empty.
    """
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return None
    
    try:
        # Read only the last line to get the latest timestamp efficiently
        with open(file_path, 'rb') as f:
            f.seek(-2, os.SEEK_END) # Jump to the second last byte
            while f.read(1) != b'\n': # Until newline is found
                f.seek(-2, os.SEEK_CUR) # Jump back two bytes
            last_line = f.readline().decode().strip()
        
        # Parse the DateTime from the last line
        latest_dt_str = last_line.split(',')[0]
        return datetime.strptime(latest_dt_str, '%Y-%m-%dT%H:%M')
    except Exception as e:
        print(f"Error reading latest timestamp from {file_path}: {e}")
        return None

# Add the project root to the Python path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import RAW_DATA_CSV, TWS_HOST, TWS_PORT, TWS_CLIENT_ID, NVDA_CONTRACT_DETAILS, TWS_MAX_CONCURRENT_REQUESTS, INITIAL_START_DATE, DATA_BATCH_SAVE_SIZE

async def _fetch_single_day_data(ib, contract, end_date_str, barSizeSetting, semaphore):
    """Helper function to fetch data for a single day with rate limiting and return as DataFrame."""
    async with semaphore:
        print(f"Requesting data ending {end_date_str} for duration '1 D'...")
        bars = await ib.reqHistoricalDataAsync(
            contract,
            endDateTime=end_date_str,
            durationStr='1 D',
            barSizeSetting=barSizeSetting,
            whatToShow='TRADES',
            useRTH=False,
            formatDate=1
        )
        if bars:
            print(f"Fetched {len(bars)} bars ending {end_date_str}.")
            df = util.df(bars)
            # Rename columns to match expected format (DateTime, Open, High, Low, Close)
            df = df[['date', 'open', 'high', 'low', 'close']]
            df.columns = ['DateTime', 'Open', 'High', 'Low', 'Close']
            # Ensure DateTime is in the correct format
            df['DateTime'] = df['DateTime'].dt.strftime('%Y-%m-%dT%H:%M')
            return df
        else:
            print(f"No bars received for period ending {end_date_str}.")
            return pd.DataFrame() # Return empty DataFrame if no bars

async def fetch_historical_data(
    contract_details: dict,
    end_date: datetime,
    barSizeSetting='1 min',
    file_path=RAW_DATA_CSV,
    initial_start_date: datetime = INITIAL_START_DATE # Use config for default
):
    """
    Connects to IB TWS/Gateway, fetches historical minute-level data for the specified contract
    between start_date and end_date, and saves it to a CSV file. Data is fetched concurrently
    and saved in batches.

    Args:
        contract_details (dict): A dictionary containing contract details (e.g., symbol, secType, exchange, currency).
        end_date (datetime): The end date for data fetching.
        barSizeSetting (str): Bar size (e.g., '1 min', '5 mins', '1 hour').
        file_path (str): Path to save the fetched data as a CSV.
        initial_start_date (datetime): The start date for data fetching if no existing data.
    """
    ib = IB()
    
    try:
        await ib.connectAsync(TWS_HOST, TWS_PORT, TWS_CLIENT_ID)
        print(f"Connected to IB TWS/Gateway on {TWS_HOST}:{TWS_PORT} with Client ID {TWS_CLIENT_ID}")

        # Create contract based on secType
        sec_type = contract_details.get('secType', 'STK')
        if sec_type == 'STK':
            contract = Stock(
                symbol=contract_details['symbol'],
                exchange=contract_details['exchange'],
                currency=contract_details['currency']
            )
        elif sec_type == 'CASH':
            contract = Forex(
                pair=f"{contract_details['symbol']}{contract_details['currency']}"
            )
        elif sec_type == 'CFD':
            contract = CFD(
                symbol=contract_details['symbol'],
                exchange=contract_details['exchange'],
                currency=contract_details['currency']
            )
        elif sec_type == 'FUT':
            contract = Future(
                symbol=contract_details['symbol'],
                lastTradeDateOrContractMonth=contract_details.get('lastTradeDateOrContractMonth'),
                exchange=contract_details['exchange'],
                currency=contract_details['currency']
            )
        else:
            raise ValueError(f"Unsupported security type: {sec_type}")
        
        # Ensure the contract details are resolved
        print(f"Qualifying contract: {contract.symbol}...")
        await ib.qualifyContractsAsync(contract)

        # Determine the actual start date for fetching
        latest_timestamp_in_file = _get_latest_timestamp_from_csv(file_path)
        actual_fetch_start_date = initial_start_date
        if latest_timestamp_in_file:
            # Start fetching from 1 minute after the latest timestamp in the file
            actual_fetch_start_date = latest_timestamp_in_file + timedelta(minutes=1)
            print(f"Latest data in file is {latest_timestamp_in_file}. Fetching new data from {actual_fetch_start_date}.")
        else:
            print(f"No existing data found in {file_path}. Fetching from initial start date {initial_start_date}.")

        if actual_fetch_start_date >= end_date:
            print("Data is already up to date. No new data to fetch.")
            return

        print(f"Fetching historical data for {contract.symbol}/{contract.currency} from {actual_fetch_start_date.strftime('%Y%m%d %H:%M:%S')} to {end_date.strftime('%Y%m%d %H:%M:%S')}...")

        # Generate all daily end dates for requests in reverse chronological order
        request_dates = []
        current_date = end_date
        while current_date > actual_fetch_start_date:
            request_dates.append(current_date)
            current_date -= timedelta(days=1)
        
        # Create a semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(TWS_MAX_CONCURRENT_REQUESTS)
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file_exists = os.path.exists(file_path)
        total_bars_fetched = 0
        
        # Process requests in batches for saving
        for i in range(0, len(request_dates), DATA_BATCH_SAVE_SIZE):
            batch_dates = request_dates[i:i + DATA_BATCH_SAVE_SIZE]
            
            tasks = []
            for req_date in batch_dates:
                end_date_str = req_date.strftime('%Y%m%d %H:%M:%S')
                tasks.append(_fetch_single_day_data(ib, contract, end_date_str, barSizeSetting, semaphore))
            
            # Run tasks concurrently for the current batch
            results_batch = await asyncio.gather(*tasks)
            
            # Concatenate all DataFrames in the batch
            dfs_to_append = [df for df in results_batch if not df.empty]
            if dfs_to_append:
                combined_df_batch = pd.concat(dfs_to_append)
                combined_df_batch.sort_values('DateTime', inplace=True) # Ensure chronological order
                
                combined_df_batch.to_csv(file_path, mode='a', header=not file_exists, index=False)
                file_exists = True
                total_bars_fetched += len(combined_df_batch)
                print(f"Appended {len(combined_df_batch)} bars (batch {i//DATA_BATCH_SAVE_SIZE + 1}) to {file_path}")

        if total_bars_fetched == 0:
            print("No historical data received for the entire period.")
            return

        print(f"Successfully fetched a total of {total_bars_fetched} bars and saved to {file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if ib and hasattr(ib, 'disconnect') and callable(ib.disconnect):
            try:
                await ib.disconnect()
                print("Disconnected from IB TWS/Gateway.")
            except TypeError:
                print("Warning: TypeError encountered during IB TWS/Gateway disconnection. Connection might already be closed.")
        else:
            print("Could not disconnect from IB TWS/Gateway (ib object or disconnect method not available).")

if __name__ == "__main__":
    print("Running TWS data ingestion for NVDA...")
    
    # Define the date range for data fetching (January 1, 2024, until now)
    end_date = datetime.now()
    # initial_start_date is now taken from config.py by default in fetch_historical_data

    asyncio.run(fetch_historical_data(
        contract_details=NVDA_CONTRACT_DETAILS,
        end_date=end_date
    ))