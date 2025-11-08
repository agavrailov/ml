import sys
import os
import pandas as pd
from ib_insync import IB, Stock, util, Forex, CFD, Contract, Future
from datetime import datetime, timedelta
import asyncio

# Add the project root to the Python path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import RAW_DATA_CSV, TWS_HOST, TWS_PORT, TWS_CLIENT_ID, NVDA_CONTRACT_DETAILS

async def fetch_historical_data(
    contract_details: dict,
    start_date: datetime,
    end_date: datetime,
    barSizeSetting='1 min',
    file_path=RAW_DATA_CSV
):
    """
    Connects to IB TWS/Gateway, fetches historical minute-level data for the specified contract
    between start_date and end_date, and saves it to a CSV file. Data is fetched in daily chunks.

    Args:
        contract_details (dict): A dictionary containing contract details (e.g., symbol, secType, exchange, currency).
        start_date (datetime): The start date for data fetching.
        end_date (datetime): The end date for data fetching.
        barSizeSetting (str): Bar size (e.g., '1 min', '5 mins', '1 hour').
        file_path (str): Path to save the fetched data as a CSV.
    """
    ib = IB()
    all_bars = []
    current_end_date = end_date

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

        print(f"Fetching historical data for {contract.symbol}/{contract.currency} from {start_date.strftime('%Y%m%d %H:%M:%S')} to {end_date.strftime('%Y%m%d %H:%M:%S')}...")

        while current_end_date > start_date:
            # Request data for one day at a time
            duration_str = '1 D'
            
            # TWS API endDateTime is exclusive, so we fetch up to the current_end_date
            # and then move current_end_date back by one day.
            end_date_str = current_end_date.strftime('%Y%m%d %H:%M:%S')
            
            print(f"Requesting data ending {end_date_str} for duration {duration_str}...")
            bars = await ib.reqHistoricalDataAsync(
                contract,
                endDateTime=end_date_str,
                durationStr=duration_str,
                barSizeSetting=barSizeSetting,
                whatToShow='TRADES',
                useRTH=False,
                formatDate=1
            )

            if bars:
                all_bars.extend(bars)
                print(f"Fetched {len(bars)} bars ending {end_date_str}.")
            else:
                print(f"No bars received for period ending {end_date_str}.")
            
            # Move to the previous day
            current_end_date -= timedelta(days=1)
            # Add a small delay to avoid hitting API rate limits
            await asyncio.sleep(1) 

        if not all_bars:
            print("No historical data received for the entire period.")
            return

        # Convert to DataFrame
        df = util.df(all_bars)
        
        # Sort by date to ensure chronological order
        df.sort_values('date', inplace=True)

        # Rename columns to match expected format (DateTime, Open, High, Low, Close)
        df = df[['date', 'open', 'high', 'low', 'close']]
        df.columns = ['DateTime', 'Open', 'High', 'Low', 'Close']
        
        # Ensure DateTime is in the correct format
        df['DateTime'] = df['DateTime'].dt.strftime('%Y-%m-%dT%H:%M')

        # Save to CSV
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # If file exists, append without header, otherwise write with header
        if os.path.exists(file_path):
            df.to_csv(file_path, mode='a', header=False, index=False)
        else:
            df.to_csv(file_path, index=False)
        print(f"Successfully fetched a total of {len(df)} bars and saved to {file_path}")

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
    
    # Define the date range for data fetching (e.g., last 3 months)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90) # Approximately 3 months

    asyncio.run(fetch_historical_data(
        contract_details=NVDA_CONTRACT_DETAILS,
        start_date=start_date,
        end_date=end_date
    ))