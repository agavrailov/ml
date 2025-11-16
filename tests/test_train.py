import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock, mock_open

from src.train import train_model
from src.config import TR_SPLIT


def _build_dummy_feature_df() -> pd.DataFrame:
    """Create a small feature DataFrame spanning 2022-12 to early 2023.

    We deliberately include rows before and after 2023-01-01 so we can verify
    that ``train_model`` filters to 2023+ and then applies the TR_SPLIT ratio
    on the remaining rows.
    """
    times = pd.date_range("2022-12-28", periods=10, freq="D")
    return pd.DataFrame(
        {
            "Time": times,
            "Open": np.linspace(100.0, 110.0, len(times)),
            "High": np.linspace(101.0, 111.0, len(times)),
        }
    )


def test_train_model_filters_to_2023_plus_and_respects_split(tmp_path):
    df = _build_dummy_feature_df()
    feature_cols = ["Open", "High"]

    # Compute the expected number of training rows after the 2023+ cutoff
    df_filtered = df[df["Time"] >= pd.Timestamp("2023-01-01")]
    expected_train_len = int(len(df_filtered) * TR_SPLIT)

    # Minimal hyperparameter set used by train_model via get_run_hyperparameters
    hps = {
        "lstm_units": 16,
        "learning_rate": 0.01,
        "epochs": 1,
        "batch_size": 2,
        "n_lstm_layers": 1,
        "stateful": True,
        "features_to_use": feature_cols,
        "frequency": "15min",
        "tsteps": 5,
        "optimizer_name": "rmsprop",
        "loss_function": "mae",
    }

    # Dummy paths for hourly data and scaler params
    hourly_csv_path = tmp_path / "nvda_15min.csv"
    hourly_csv_path.write_text("Time,Open,High\n")
    scaler_params_path = tmp_path / "scaler_params_15min.json"

    with (
        patch("src.train.get_run_hyperparameters", return_value=hps),
        patch("src.train.get_hourly_data_csv_path", return_value=str(hourly_csv_path)),
        patch("src.train.get_scaler_params_json_path", return_value=str(scaler_params_path)),
        patch("src.train.os.path.exists", return_value=True),
        patch("src.train.prepare_keras_input_data", return_value=(df.copy(), feature_cols)),
        patch("src.train.fit_standard_scaler") as mock_fit_scaler,
        patch("src.train.apply_standard_scaler", side_effect=lambda d, cols, sp: d),
        patch(
            "src.train.create_sequences_for_stateful_lstm",
            return_value=(
                np.ones((10, hps["tsteps"], len(feature_cols))),
                np.ones((10, 1)),
            ),
        ),
        patch("src.train.build_lstm_model") as mock_build_model,
        patch("src.train.load_stateful_weights_into_non_stateful_model"),
        patch("src.train.json.dump"),
        patch("src.train.open", mock_open(), create=True),
        patch("src.train.datetime") as mock_datetime,
    ):
        # Configure scaler mock to return mean/std and a JSON-serialisable dict
        mock_fit_scaler.return_value = (
            pd.Series({"Open": 0.0, "High": 0.0}),
            pd.Series({"Open": 1.0, "High": 1.0}),
            {"mean": {"Open": 0.0, "High": 0.0}, "std": {"Open": 1.0, "High": 1.0}},
        )

        # Configure fake model & history so training logic can run
        mock_datetime.now.return_value = datetime(2025, 1, 1, 0, 0, 0)

        model = MagicMock()
        history = MagicMock()
        history.history = {"val_loss": [0.5, 0.4]}
        model.fit.return_value = history
        model.predict.return_value = np.ones((10, 1))
        mock_build_model.return_value = model

        result = train_model(frequency="15min", tsteps=hps["tsteps"], features_to_use=feature_cols)

    # Training should have returned a result (we don't check exact values here)
    assert result is not None

    # ``fit_standard_scaler`` is called with the *training* slice only.
    fit_args, _ = mock_fit_scaler.call_args
    df_train_used = fit_args[0]

    # 1) No rows before 2023-01-01 should be present in the training set.
    assert (df_train_used["Time"] < pd.Timestamp("2023-01-01")).sum() == 0

    # 2) The length of the training set should match TR_SPLIT applied to the
    #    filtered (2023+) data.
    assert len(df_train_used) == expected_train_len
