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
    durationStr='1 D', barSizeSetting='1 min',
    endDateTime='',
    file_path=RAW_DATA_CSV
):
    """
    Connects to IB TWS/Gateway, fetches historical minute-level data for the specified contract,
    and saves it to a CSV file.

    Args:
        contract_details (dict): A dictionary containing contract details (e.g., symbol, secType, exchange, currency).
        durationStr (str): Duration of data to fetch (e.g., '1 Y', '1 M', '1 W', '1 D').
        barSizeSetting (str): Bar size (e.g., '1 min', '5 mins', '1 hour').
        endDateTime (str): End date and time for the data fetch. If empty, current time is used.
        file_path (str): Path to save the fetched data as a CSV.
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

        print(f"Fetching historical data for {contract.symbol}/{contract.currency}...")
        bars = await ib.reqHistoricalDataAsync(
            contract,
            endDateTime=endDateTime,
            durationStr=durationStr,
            barSizeSetting=barSizeSetting,
            whatToShow='TRADES', # Changed from 'MIDPOINT' to 'TRADES'
            useRTH=False, # Use regular trading hours
            formatDate=1 # 1 for datetime objects, 2 for strings
        )

        if not bars:
            print("No historical data received.")
            return

        # Convert to DataFrame
        df = util.df(bars)
        
        # Rename columns to match expected format (DateTime, Open, High, Low, Close)
        df = df[['date', 'open', 'high', 'low', 'close']]
        df.columns = ['DateTime', 'Open', 'High', 'Low', 'Close']
        
        # Ensure DateTime is in the correct format
        df['DateTime'] = df['DateTime'].dt.strftime('%Y-%m-%dT%H:%M')

        # Save to CSV
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        print(f"Successfully fetched {len(df)} bars and saved to {file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if ib.isConnected():
            await ib.disconnect()
            print("Disconnected from IB TWS/Gateway.")

if __name__ == "__main__":
    print("Running TWS data ingestion for NVDA...")
    asyncio.run(fetch_historical_data(contract_details=NVDA_CONTRACT_DETAILS, durationStr='1 D'))