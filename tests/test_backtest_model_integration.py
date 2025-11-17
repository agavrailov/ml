"""Integration test for model-based backtesting using real artifacts.

This test is slower and depends on:
- A trained 60min model in the registry.
- A valid `best_hyperparameters.json` entry for (60min, TSTEPS).
- An hourly CSV at the path returned by `get_hourly_data_csv_path("60min")`.

It exercises the real `_make_model_prediction_provider` and backtest engine
end-to-end, and asserts that equity and trades are numeric and NaN-free.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from src.config import get_hourly_data_csv_path, TSTEPS
from src.backtest import run_backtest_on_dataframe


@pytest.mark.slow  # type: ignore[attr-defined]
def test_model_mode_integration_real_artifacts() -> None:
    freq = "60min"
    csv_path = get_hourly_data_csv_path(freq)
    if not os.path.exists(csv_path):
        pytest.skip(f"Hourly CSV for {freq} not found at {csv_path}; skipping integration test.")

    # Use a moderate tail to keep test time reasonable but still exercise
    # batched predictions and backtest logic.
    df = pd.read_csv(csv_path)
    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"])
    df = df.tail(2000)

    try:
        result = run_backtest_on_dataframe(
            df,
            initial_equity=10_000.0,
            frequency=freq,
            prediction_mode="model",
            commission_per_unit_per_leg=0.0,
            min_commission_per_order=0.0,
        )
    except FileNotFoundError:
        pytest.skip("No trained model/scaler found for 60min; skipping integration test.")

    # Equity curve should be same length as data and finite.
    assert len(result.equity_curve) == len(df)
    assert np.isfinite(result.final_equity)
    assert all(np.isfinite(e) for e in result.equity_curve)

    # Trades, if any, should have finite PnL.
    for trade in result.trades:
        assert np.isfinite(trade.pnl)
        assert np.isfinite(trade.gross_pnl)
