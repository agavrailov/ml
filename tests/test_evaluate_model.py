import json
from datetime import datetime
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

from src.evaluate_model import evaluate_model_performance
from src.config import FREQUENCY, TSTEPS


def test_evaluate_model_uses_last_twenty_percent_window(tmp_path):
    """ensure evaluate_model_performance evaluates on the last 20% of rows.

    We build a synthetic hourly dataset, then patch the filesystem helpers so
    that evaluate_model_performance reads this data. We intercept the call to
    ``apply_standard_scaler`` to inspect the evaluation window DataFrame.
    """

    # Build a simple hourly dataset with a known length
    total_rows = 100
    times = pd.date_range("2023-01-01", periods=total_rows, freq="H")
    df_hourly = pd.DataFrame({"Time": times, "Open": np.linspace(100, 120, total_rows)})

    hourly_csv_path = tmp_path / "nvda_15min.csv"
    df_hourly.to_csv(hourly_csv_path, index=False)

    # Minimal scaler params so that mean/std lookups succeed
    scaler_params_path = tmp_path / "scaler_params_15min.json"
    scaler_params = {"mean": {"Open": 0.0}, "std": {"Open": 1.0}}
    scaler_params_path.write_text(json.dumps(scaler_params))

    feature_cols = ["Open"]
    df_full_featured = df_hourly[["Time"] + feature_cols].copy()

    captured_eval_df = {}

    def _capture_apply_standard_scaler(df, cols, params):
        # capture the evaluation window dataframe that is passed in
        captured_eval_df["df"] = df.copy()
        return df

    with (
        patch("src.evaluate_model.get_hourly_data_csv_path", return_value=str(hourly_csv_path)),
        patch("src.evaluate_model.get_scaler_params_json_path", return_value=str(scaler_params_path)),
        patch("src.evaluate_model.load_hourly_features", return_value=(df_full_featured, feature_cols)),
        patch("src.evaluate_model.apply_standard_scaler", side_effect=_capture_apply_standard_scaler),
        patch(
            "src.evaluate_model.create_sequences_for_stateless_lstm",
            return_value=(
                np.ones((10, TSTEPS, len(feature_cols))),
                np.ones((10, 1)),
            ),
        ),
        patch("src.evaluate_model.load_model") as mock_load_model,
        patch("src.evaluate_model.build_lstm_model") as mock_build_model,
        patch("src.evaluate_model.load_stateful_weights_into_non_stateful_model"),
        patch("src.evaluate_model.plt.savefig"),
    ):
        # Configure fake models
        stateful_model = MagicMock()
        mock_load_model.return_value = stateful_model

        eval_model = MagicMock()
        eval_model.predict.return_value = np.ones((10, 1))
        mock_build_model.return_value = eval_model

        # Run evaluation (we only care about the data window, not metrics)
        mae, corr = evaluate_model_performance(
            model_path="dummy_model.keras",
            frequency=FREQUENCY,
            tsteps=TSTEPS,
            lstm_units=16,
            n_lstm_layers=1,
            stateful=True,
            features_to_use=feature_cols,
            bias_correction_path=None,
        )

    # Sanity check that evaluation returned metrics
    assert mae is not None
    assert corr is not None

    eval_df = captured_eval_df["df"]

    # Expect window size to be 20% of the full dataset
    expected_window = max(1, int(total_rows * 0.2))
    assert len(eval_df) == expected_window

    # And it should correspond to the *last* 20% of rows
    pd.testing.assert_series_equal(
        eval_df["Time"].reset_index(drop=True),
        df_full_featured["Time"].iloc[-expected_window:].reset_index(drop=True),
    )
