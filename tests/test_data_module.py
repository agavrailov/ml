from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pandas as pd
import numpy as np

import src.data as data_mod


def test_load_hourly_ohlc_uses_config_path_and_parses_time(tmp_path):
    """load_hourly_ohlc should read from the config-resolved CSV and parse Time."""
    csv_path = tmp_path / "nvda_15min.csv"
    df_in = pd.DataFrame(
        {
            "Time": ["2023-01-01 09:00:00", "2023-01-01 10:00:00"],
            "Open": [100.0, 101.0],
            "High": [101.0, 102.0],
            "Low": [99.0, 100.0],
            "Close": [100.5, 101.5],
        }
    )
    df_in.to_csv(csv_path, index=False)

    with patch("src.data.get_hourly_data_csv_path", return_value=str(csv_path)) as mock_path:
        df_out = data_mod.load_hourly_ohlc("15min")

    mock_path.assert_called_once_with("15min")
    assert len(df_out) == 2
    assert pd.api.types.is_datetime64_any_dtype(df_out["Time"])
    assert float(df_out["Open"].iloc[0]) == 100.0


def test_load_hourly_features_delegates_to_prepare_keras_input_data():
    """load_hourly_features should call prepare_keras_input_data once with path+features."""
    dummy_df = pd.DataFrame({"Time": [datetime(2023, 1, 1)], "Open": [100.0]})
    dummy_features = ["Open"]

    with (
        patch("src.data.get_hourly_data_csv_path", return_value="/tmp/nvda_15min.csv") as mock_path,
        patch("src.data.prepare_keras_input_data", return_value=(dummy_df, dummy_features)) as mock_prepare,
    ):
        df, features = data_mod.load_hourly_features("15min", dummy_features)

    mock_path.assert_called_once_with("15min")
    mock_prepare.assert_called_once_with("/tmp/nvda_15min.csv", dummy_features)
    assert df is dummy_df
    assert features == dummy_features
