import numpy as np
import pandas as pd

from src.data_utils import (
    fit_standard_scaler,
    apply_standard_scaler,
    get_effective_data_length,
    create_sequences_for_stateless_lstm,
    create_sequences_for_stateful_lstm,
)


def test_fit_and_apply_standard_scaler_roundtrip():
    # Simple 2-feature dataframe
    df = pd.DataFrame({
        "Time": pd.date_range("2023-01-01", periods=5, freq="H"),
        "Open": [1.0, 2.0, 3.0, 4.0, 5.0],
        "Close": [2.0, 4.0, 6.0, 8.0, 10.0],
    })
    feature_cols = ["Open", "Close"]

    mean_vals, std_vals, scaler_params = fit_standard_scaler(df, feature_cols)
    df_norm = apply_standard_scaler(df, feature_cols, scaler_params)

    # Means should be ~0; std should be non-zero and roughly similar across features
    np.testing.assert_allclose(df_norm[feature_cols].mean().values, np.zeros(2), atol=1e-7)
    stds = df_norm[feature_cols].std(ddof=0).values
    assert (stds > 0).all()


def _make_dummy_series(n: int = 20) -> pd.DataFrame:
    times = pd.date_range("2023-01-01", periods=n, freq="H")
    df = pd.DataFrame({
        "Time": times,
        "Open": np.linspace(100.0, 120.0, n),
        "High": np.linspace(101.0, 121.0, n),
        "Low": np.linspace(99.0, 119.0, n),
        "Close": np.linspace(100.5, 120.5, n),
    })
    return df


def test_get_effective_data_length_and_stateless_sequences():
    df = _make_dummy_series(20)
    seq_len = 5
    rows_ahead = 1

    eff_len = get_effective_data_length(df, seq_len, rows_ahead)
    assert eff_len > 0

    X, y = create_sequences_for_stateless_lstm(df, seq_len, rows_ahead)
    assert X.shape[0] == eff_len
    assert X.shape[1] == seq_len
    # n_features = all columns except Time
    assert X.shape[2] == (df.shape[1] - 1)
    assert y.shape == (eff_len, 1)


def test_create_sequences_for_stateful_lstm_respects_batch_size():
    df = _make_dummy_series(30)
    seq_len = 3
    rows_ahead = 1
    batch_size = 4

    X, y = create_sequences_for_stateful_lstm(df, seq_len, batch_size, rows_ahead)
    # number of sequences should be a multiple of batch_size
    assert X.shape[0] % batch_size == 0
    assert X.shape[1] == seq_len
    assert X.shape[2] == (df.shape[1] - 1)
    assert y.shape == (X.shape[0], 1)
