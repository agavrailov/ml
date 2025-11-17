"""Tests for model-based backtesting with batched predictions.

We ensure that:
- `run_backtest_on_dataframe` runs end-to-end in `prediction_mode="model"` on
  a small synthetic dataset when the prediction provider is mocked.
- The equity curve contains finite values and the final equity is numeric.
- No NaNs propagate into basic metrics when using the real engine.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from src.backtest import run_backtest_on_dataframe


def _dummy_ohlc_with_time() -> pd.DataFrame:
    times = pd.date_range("2023-01-01", periods=50, freq="H")
    return pd.DataFrame(
        {
            "Time": times,
            "Open": np.linspace(100.0, 102.0, len(times)),
            "High": np.linspace(100.5, 102.5, len(times)),
            "Low": np.linspace(99.5, 101.5, len(times)),
            "Close": np.linspace(100.2, 102.2, len(times)),
        }
    )


def test_run_backtest_model_mode_no_nans(monkeypatch: Any) -> None:
    data = _dummy_ohlc_with_time()

    # Mock the model-based prediction provider factory so that it returns a
    # simple deterministic provider, bypassing heavy ML and focusing on the
    # integration with the backtest engine.
    def _fake_model_provider(data_arg: pd.DataFrame, frequency: str):  # noqa: ARG001
        series = data_arg["Close"].to_numpy(dtype=float) * 1.01  # +1% edge

        def provider(i: int, row: pd.Series) -> float:  # noqa: ARG001
            return float(series[i])

        return provider

    with patch("src.backtest._make_model_prediction_provider", side_effect=_fake_model_provider):
        result = run_backtest_on_dataframe(
            data,
            initial_equity=10_000.0,
            frequency="60min",
            prediction_mode="model",
            commission_per_unit_per_leg=0.0,
            min_commission_per_order=0.0,
        )

    # Equity curve should have same length as data and contain finite values.
    assert len(result.equity_curve) == len(data)
    assert np.isfinite(result.final_equity)
    assert all(np.isfinite(e) for e in result.equity_curve)
