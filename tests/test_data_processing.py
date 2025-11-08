import pandas as pd
import json
import os
import pytest
from src.data_processing import convert_minute_to_hourly, prepare_keras_input_data

# Fixture to create a dummy minute-level CSV file
@pytest.fixture
def dummy_minute_csv(tmp_path):
    data = {
        'DateTime': pd.to_datetime(['2023-01-01T00:00', '2023-01-01T00:30', '2023-01-01T00:59',
                                    '2023-01-01T01:00', '2023-01-01T01:15', '2023-01-01T01:45']),
        'Open': [100, 101, 102, 103, 104, 105],
        'High': [102, 103, 104, 105, 106, 107],
        'Low': [99, 100, 101, 102, 103, 104],
        'Close': [101, 102, 103, 104, 105, 106]
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "xagusd_minute.csv"
    df.to_csv(file_path, index=False)
    return file_path

# Fixture to create a dummy hourly-level CSV file
@pytest.fixture
def dummy_hourly_csv(tmp_path):
    data = {
        'Time': pd.to_datetime(['2023-01-01T00:00', '2023-01-01T01:00', '2023-01-01T02:00']),
        'Open': [100, 103, 106],
        'High': [102, 105, 108],
        'Low': [99, 102, 105],
        'Close': [101, 104, 107]
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "xagusd_hourly.csv"
    df.to_csv(file_path, index=False)
    return file_path

def test_convert_minute_to_hourly(dummy_minute_csv, tmp_path):
    output_path = tmp_path / "xagusd_hourly.csv"
    convert_minute_to_hourly(dummy_minute_csv, output_path)

    assert os.path.exists(output_path)
    df_hourly = pd.read_csv(output_path, parse_dates=['Time'])

    # Expected hourly data based on dummy_minute_csv
    # First hour (00:00-00:59)
    # Open: 100 (from 00:00)
    # High: max(102, 103, 104) = 104
    # Low: min(99, 100, 101) = 99
    # Close: 103 (from 00:59)

    # Second hour (01:00-01:59)
    # Open: 103 (from 01:00)
    # High: max(105, 106, 107) = 107
    # Low: min(102, 103, 104) = 102
    # Close: 106 (from 01:45)

    assert len(df_hourly) == 2
    assert df_hourly.iloc[0]['Time'] == pd.Timestamp('2023-01-01 00:00:00')
    assert df_hourly.iloc[0]['Open'] == 100
    assert df_hourly.iloc[0]['High'] == 104
    assert df_hourly.iloc[0]['Low'] == 99
    assert df_hourly.iloc[0]['Close'] == 103

    assert df_hourly.iloc[1]['Time'] == pd.Timestamp('2023-01-01 01:00:00')
    assert df_hourly.iloc[1]['Open'] == 103
    assert df_hourly.iloc[1]['High'] == 107
    assert df_hourly.iloc[1]['Low'] == 102
    assert df_hourly.iloc[1]['Close'] == 106


def test_prepare_keras_input_data(dummy_hourly_csv, tmp_path):
    output_training_path = tmp_path / "training_data.csv"
    output_scaler_params_path = tmp_path / "scaler_params.json"
    prepare_keras_input_data(dummy_hourly_csv, output_training_path, output_scaler_params_path)

    assert os.path.exists(output_training_path)
    assert os.path.exists(output_scaler_params_path)

    df_training = pd.read_csv(output_training_path)
    with open(output_scaler_params_path, 'r') as f:
        scaler_params = json.load(f)

    # Check if normalization was applied (values should be different from original)
    original_df = pd.read_csv(dummy_hourly_csv)
    ohlc_cols = ['Open', 'High', 'Low', 'Close']
    for col in ohlc_cols:
        assert not df_training[col].equals(original_df[col])

    # Check if scaler params are correctly saved
    assert 'mean' in scaler_params
    assert 'std' in scaler_params
    assert len(scaler_params['mean']) == 4 # OHLC columns
    assert len(scaler_params['std']) == 4

    # Verify a normalized value (simple check)
    # For example, if original Open was 100, and mean was 100, std was 1, normalized should be 0
    # This requires more precise dummy data or calculation, for now, just check non-zero std
    for col in ohlc_cols:
        assert scaler_params['std'][col] > 0