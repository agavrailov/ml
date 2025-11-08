import pytest
import pandas as pd
import os
import json
from datetime import datetime, timedelta

from src.data_processing import convert_minute_to_hourly, prepare_keras_input_data
from src.config import PROCESSED_DATA_DIR, HOURLY_DATA_CSV, TRAINING_DATA_CSV, SCALER_PARAMS_JSON

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

    # Mock HOURLY_DATA_CSV path
    mock_hourly_data_csv = temp_processed_dir / "nvda_hourly.csv"

    # Mock TRAINING_DATA_CSV path
    mock_training_data_csv = temp_processed_dir / "training_data.csv"

    # Mock SCALER_PARAMS_JSON path
    mock_scaler_params_json = temp_processed_dir / "scaler_params.json"

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

    yield mock_raw_data_csv, mock_hourly_data_csv, mock_training_data_csv, mock_scaler_params_json

    # Teardown: files are automatically removed by tmp_path fixture

def test_convert_minute_to_hourly(setup_teardown_data_processing_test):
    mock_raw_data_csv, mock_hourly_data_csv, _, _ = setup_teardown_data_processing_test

    convert_minute_to_hourly(mock_raw_data_csv, mock_hourly_data_csv)

    assert mock_hourly_data_csv.exists()
    df_hourly = pd.read_csv(mock_hourly_data_csv)

    # Expect 2 hourly data points (9:00 and 10:00)
    assert len(df_hourly) == 2
    assert df_hourly['Time'].iloc[0] == '2023-01-01T09:00'
    assert df_hourly['Time'].iloc[1] == '2023-01-01T10:00'

    # Verify OHLC values for the first hour (9:00-9:59)
    # Open should be the first minute's open
    assert df_hourly['Open'].iloc[0] == pytest.approx(100.0)
    # High should be the max of the first 60 minutes' highs
    assert df_hourly['High'].iloc[0] == pytest.approx(101 + 59 * 0.1)
    # Low should be the min of the first 60 minutes' lows
    assert df_hourly['Low'].iloc[0] == pytest.approx(99.0)
    # Close should be the last minute's close
    assert df_hourly['Close'].iloc[0] == pytest.approx(100.5 + 59 * 0.1)

def test_prepare_keras_input_data(setup_teardown_data_processing_test):
    mock_raw_data_csv, mock_hourly_data_csv, mock_training_data_csv, mock_scaler_params_json = setup_teardown_data_processing_test

    # First, convert minute to hourly to create the input for this test
    convert_minute_to_hourly(mock_raw_data_csv, mock_hourly_data_csv)

    prepare_keras_input_data(mock_hourly_data_csv, mock_training_data_csv, mock_scaler_params_json)

    assert mock_training_data_csv.exists()
    assert mock_scaler_params_json.exists()

    df_training = pd.read_csv(mock_training_data_csv)
    scaler_params = json.loads(mock_scaler_params_json.read_text())

    # Verify normalization
    ohlc_cols = ['Open', 'High', 'Low', 'Close']
    for col in ohlc_cols:
        # Check if normalized data has mean close to 0 and std close to 1
        # Due to small sample size, these won't be exactly 0 and 1, but should be close
        assert df_training[col].mean() == pytest.approx(0.0, abs=1e-6)
        assert df_training[col].std() == pytest.approx(1.0, abs=1e-6)

        # Check if scaler params are saved correctly
        assert col in scaler_params['mean']
        assert col in scaler_params['std']
