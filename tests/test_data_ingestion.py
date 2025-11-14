import pytest
import asyncio
from unittest.mock import AsyncMock, patch
import pandas as pd
from datetime import datetime, timedelta

from src.data_ingestion import fetch_historical_data
from src.config import NVDA_CONTRACT_DETAILS, TWS_HOST, TWS_PORT, TWS_CLIENT_ID, RAW_DATA_CSV

# Mock BarData class for reqHistoricalDataAsync return value
class MockBarData:
    def __init__(self, date, open, high, low, close):
        self.date = date
        self.open = open
        self.high = high
        self.low = low
        self.close = close

@pytest.fixture
def mock_ib():
    """Fixture to mock the IB object and its async methods."""
    with patch('src.data_ingestion.IB', autospec=True) as mock_ib_class:
        mock_ib_instance = mock_ib_class.return_value
        mock_ib_instance.connectAsync = AsyncMock()
        mock_ib_instance.qualifyContractsAsync = AsyncMock()
        mock_ib_instance.reqHistoricalDataAsync = AsyncMock()
        mock_ib_instance.isConnected.return_value = True
        mock_ib_instance.disconnect = AsyncMock()
        yield mock_ib_instance

@pytest.mark.asyncio
async def test_fetch_historical_data_nvda(mock_ib):
    """
    Test that fetch_historical_data connects to TWS, qualifies the contract,
    requests historical data for NVDA, and attempts to save it.
    """
    # Mock historical data response
    mock_bars = [
        MockBarData(datetime(2023, 1, 1, 9, 30), 100.0, 101.0, 99.0, 100.5),
        MockBarData(datetime(2023, 1, 1, 9, 31), 100.5, 102.0, 100.0, 101.5),
    ]
    mock_ib.reqHistoricalDataAsync.return_value = mock_bars

    # Mock pandas util.df to return a DataFrame from our mock bars
    with patch('src.data_ingestion.util.df') as mock_util_df:
        mock_util_df.return_value = pd.DataFrame([
            {'date': bar.date, 'open': bar.open, 'high': bar.high, 'low': bar.low, 'close': bar.close}
            for bar in mock_bars
        ])
        
        # Mock os.makedirs and df.to_csv to prevent actual file system operations
        with patch('os.makedirs') as mock_makedirs, \
             patch('pandas.DataFrame.to_csv') as mock_to_csv:

            # Define a date range for the test
            test_end_date = datetime(2023, 1, 1, 9, 31)
            test_start_date = test_end_date - timedelta(days=1)

            await fetch_historical_data(
                contract_details=NVDA_CONTRACT_DETAILS,
                start_date=test_start_date,
                end_date=test_end_date,
                strict_range=True,
            )

            # Assertions
            mock_ib.connectAsync.assert_called_once_with(TWS_HOST, TWS_PORT, TWS_CLIENT_ID)
            mock_ib.qualifyContractsAsync.assert_called_once() # Argument is a Contract object, harder to assert directly
            mock_ib.reqHistoricalDataAsync.assert_called()
            
            # Verify arguments for each call
            for call_args in mock_ib.reqHistoricalDataAsync.call_args_list:
                args, kwargs = call_args
                assert kwargs['durationStr'] == '1 D'
                assert kwargs['barSizeSetting'] == '1 min'
                assert kwargs['whatToShow'] == 'TRADES'
                assert kwargs['useRTH'] == False
                assert kwargs['formatDate'] == 1
                assert 'endDateTime' in kwargs and kwargs['endDateTime'] != ''
            mock_ib.disconnect.assert_called_once()
            mock_makedirs.assert_called_once()
            mock_to_csv.assert_called_once_with(RAW_DATA_CSV, mode='a', header=False, index=False)
            # Verify the contract type passed to qualifyContractsAsync
            contract_arg = mock_ib.qualifyContractsAsync.call_args[0][0]
            assert contract_arg.symbol == NVDA_CONTRACT_DETAILS['symbol']
            assert contract_arg.secType == NVDA_CONTRACT_DETAILS['secType']
            assert contract_arg.exchange == NVDA_CONTRACT_DETAILS['exchange']
            assert contract_arg.currency == NVDA_CONTRACT_DETAILS['currency']
