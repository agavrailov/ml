import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import json
from datetime import datetime, timedelta

from src.data_processing import convert_minute_to_timeframe, prepare_keras_input_data, add_features
from src.config import PROCESSED_DATA_DIR, RAW_DATA_CSV, FEATURES_TO_USE_OPTIONS

@pytest.fixture
def setup_teardown_data_processing_test(tmp_path):
    """
    Fixture to set up mock data files and clean them up after tests.
    Uses tmp_path for temporary file creation.
    """
    # Create a temporary directory for processed data
    temp_processed_dir = tmp_path / "data" / "processed"
    temp_processed_dir.mkdir(parents=True, exist_ok=True)

    # Mock RAW_DATA_CSV path
    mock_raw_data_csv = tmp_path / "data" / "raw" / "nvda_minute.csv"
    mock_raw_data_csv.parent.mkdir(parents=True, exist_ok=True)

    # Generate mock minute-level data for 2 hours
    start_time = datetime(2023, 1, 1, 9, 0)
    data = []
    for i in range(120): # 2 hours * 60 minutes
        current_time = start_time + timedelta(minutes=i)
        data.append({
            'DateTime': current_time.strftime('%Y-%m-%dT%H:%M'),
            'Open': 100 + i * 0.1,
            'High': 101 + i * 0.1,
            'Low': 99 + i * 0.1,
            'Close': 100.5 + i * 0.1
        })
    mock_df_minute = pd.DataFrame(data)
    mock_df_minute.to_csv(mock_raw_data_csv, index=False)

    yield mock_raw_data_csv, temp_processed_dir

    # Teardown: files are automatically removed by tmp_path fixture

def test_convert_minute_to_timeframe(setup_teardown_data_processing_test):
    mock_raw_data_csv, temp_processed_dir = setup_teardown_data_processing_test

    # Temporarily set FREQUENCY for this test
    original_frequency = os.environ.get('ML_LSTM_FREQUENCY', None)
    os.environ['ML_LSTM_FREQUENCY'] = '60min' # Test with 60min frequency

    try:
        test_frequency = '60min'
        convert_minute_to_timeframe(mock_raw_data_csv, test_frequency, temp_processed_dir)
        
        mock_hourly_data_csv = temp_processed_dir / f"nvda_{test_frequency}.csv"
        assert mock_hourly_data_csv.exists()
        df_hourly = pd.read_csv(mock_hourly_data_csv)

        # Expect 2 hourly data points (9:00 and 10:00)
        assert len(df_hourly) == 2
        assert df_hourly['Time'].iloc[0] == '2023-01-01 09:00:00'
        assert df_hourly['Time'].iloc[1] == '2023-01-01 10:00:00'

        # Verify OHLC values for the first hour (9:00-9:59)
        # Open should be the first minute's open
        assert df_hourly['Open'].iloc[0] == pytest.approx(100.0)
        # High should be the max of the first 60 minutes' highs
        assert df_hourly['High'].iloc[0] == pytest.approx(101 + 59 * 0.1)
        # Low should be the min of the first 60 minutes' lows
        assert df_hourly['Low'].iloc[0] == pytest.approx(99.0)
        # Close should be the last minute's close
        assert df_hourly['Close'].iloc[0] == pytest.approx(100.5 + 59 * 0.1)
    finally:
        # Restore original FREQUENCY
        if original_frequency is not None:
            os.environ['ML_LSTM_FREQUENCY'] = original_frequency
        else:
            if 'ML_LSTM_FREQUENCY' in os.environ:
                del os.environ['ML_LSTM_FREQUENCY']

def test_add_features():
    """
    Tests the add_features function with a sample DataFrame and selected features.
    """
    data = {
        'Time': pd.to_datetime(['2023-01-01 09:00:00', '2023-01-01 10:00:00', '2023-01-01 11:00:00',
                                '2023-01-01 12:00:00', '2023-01-01 13:00:00', '2023-01-01 14:00:00',
                                '2023-01-01 15:00:00', '2023-01-01 16:00:00', '2023-01-01 17:00:00',
                                '2023-01-01 18:00:00', '2023-01-01 19:00:00', '2023-01-01 20:00:00',
                                '2023-01-01 21:00:00', '2023-01-01 22:00:00', '2023-01-01 23:00:00',
                                '2023-01-02 00:00:00', '2023-01-02 01:00:00', '2023-01-02 02:00:00',
                                '2023-01-02 03:00:00', '2023-01-02 04:00:00', '2023-01-02 05:00:00',
                                '2023-01-02 06:00:00']),
        'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121],
        'High': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122],
        'Low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120],
        'Close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5, 110.5, 111.5, 112.5, 113.5, 114.5, 115.5, 116.5, 117.5, 118.5, 119.5, 120.5, 121.5]
    }
    df = pd.DataFrame(data)

    # Test with a subset of features
    features_to_generate = ['SMA_7', 'Hour']
    df_featured = add_features(df.copy(), features_to_generate)

    assert 'SMA_7' in df_featured.columns
    assert 'SMA_21' not in df_featured.columns # Should not be added
    assert 'RSI' not in df_featured.columns # Should not be added
    assert 'Hour' in df_featured.columns
    assert 'DayOfWeek' not in df_featured.columns # Should not be added

    # Check SMA_7 calculation (simple check for non-NaN values after 7 periods)
    assert not df_featured['SMA_7'].isnull().any()
    assert df_featured['SMA_7'].iloc[0] == pytest.approx(103.5) # (100.5 + ... + 106.5) / 7

    # Check Hour calculation
    assert df_featured['Hour'].iloc[0] == 15 # First non-NaN row after SMA_7 is 15:00

def test_prepare_keras_input_data(setup_teardown_data_processing_test):
    mock_raw_data_csv, temp_processed_dir = setup_teardown_data_processing_test

    # Temporarily set FREQUENCY for this test
    original_frequency = os.environ.get('ML_LSTM_FREQUENCY', None)
    os.environ['ML_LSTM_FREQUENCY'] = '60min' # Test with 60min frequency

    try:
        # First, convert minute to hourly to create the input for this test
        test_frequency = '60min'
        convert_minute_to_timeframe(mock_raw_data_csv, test_frequency, temp_processed_dir)
        mock_hourly_data_csv = temp_processed_dir / f"nvda_{test_frequency}.csv"

        # Test with a specific set of features
        features_to_use = ['Open', 'High', 'Low', 'Close', 'SMA_7', 'RSI', 'Hour']
        df_prepared, feature_cols = prepare_keras_input_data(mock_hourly_data_csv, features_to_use)

        assert 'Time' in df_prepared.columns
        assert all(f in df_prepared.columns for f in features_to_use)
        assert 'SMA_21' not in df_prepared.columns # Should not be in final df if not requested
        assert 'DayOfWeek' not in df_prepared.columns # Should not be in final df if not requested

        assert feature_cols == features_to_use
        assert len(feature_cols) == len(features_to_use)
        assert not df_prepared.isnull().any().any() # No NaN values after feature engineering
    finally:
        # Restore original FREQUENCY
        if original_frequency is not None:
            os.environ['ML_LSTM_FREQUENCY'] = original_frequency
        else:
            if 'ML_LSTM_FREQUENCY' in os.environ:
                del os.environ['ML_LSTM_FREQUENCY']