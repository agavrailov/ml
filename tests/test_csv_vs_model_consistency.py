from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from unittest.mock import patch

from src import backtest as backtest_mod


def _dummy_ohlc_with_time(n: int = 200) -> pd.DataFrame:
    times = pd.date_range("2023-01-01", periods=n, freq="H")
    return pd.DataFrame(
        {
            "Time": times,
            "Open": np.linspace(100.0, 110.0, n),
            "High": np.linspace(100.5, 110.5, n),
            "Low": np.linspace(99.5, 109.5, n),
            "Close": np.linspace(100.2, 110.2, n),
        }
    )


def test_model_vs_csv_equity_parity(tmp_path: Path) -> None:
    """Model-mode and CSV-mode backtests should produce the same equity curve
    when given the *same* per-bar predictions and sigma behavior.

    This is a synthetic, fast test that stubs out the heavy ML bits and
    residual sigma logic so we can focus purely on consistency.
    """

    data = _dummy_ohlc_with_time(200)

    # Synthetic predictions: +1% edge on Close.
    preds = pd.DataFrame(
        {
            "Time": data["Time"],
            "predicted_price": data["Close"] * 1.01,
        }
    )
    preds_path = tmp_path / "preds_parity.csv"
    preds.to_csv(preds_path, index=False)

    # Fake model provider: returns exactly the same series as in preds.
    def _fake_model_provider(data_arg: pd.DataFrame, frequency: str):  # noqa: ARG001
        series = (data_arg["Close"].to_numpy(dtype=float) * 1.01).astype(float)

        def provider(i: int, row: pd.Series) -> float:  # type: ignore[name-defined]
            return float(series[i])

        # Use zeros so that run_backtest_on_dataframe falls back to ATR via
        # its existing "if all zeros -> use atr_series" logic.
        sigma_series = np.zeros(len(data_arg), dtype=np.float32)
        return provider, sigma_series

    # Identity bias-correction and zero residual sigma so CSV-mode doesn't
    # distort predictions relative to model-mode in this test.
    def _identity_bias_correction(
        predictions: np.ndarray,
        actuals: np.ndarray,
        window: int,
        global_mean_residual: float,
        enable_amplitude: bool = False,
    ) -> np.ndarray:  # noqa: ARG001
        return predictions

    def _zero_residual_sigma(
        predictions: np.ndarray,
        actuals: np.ndarray,
        window: int,
    ) -> np.ndarray:  # noqa: ARG001
        return np.zeros(len(predictions), dtype=np.float32)

    with (
        patch("src.backtest._make_model_prediction_provider", side_effect=_fake_model_provider),
        patch(
            "src.backtest.apply_rolling_bias_and_amplitude_correction",
            side_effect=_identity_bias_correction,
        ),
        patch(
            "src.backtest.compute_rolling_residual_sigma",
            side_effect=_zero_residual_sigma,
        ),
    ):
        # MODEL mode
        result_model = backtest_mod.run_backtest_on_dataframe(
            data.copy(),
            initial_equity=10_000.0,
            frequency="60min",
            prediction_mode="model",
            commission_per_unit_per_leg=0.0,
            min_commission_per_order=0.0,
            # Keep sigma-related filters neutral for this test.
            k_sigma_err=0.0,
            k_atr_min_tp=0.1,
        )

        # CSV mode consuming the *same* predictions.
        result_csv = backtest_mod.run_backtest_on_dataframe(
            data.copy(),
            initial_equity=10_000.0,
            frequency="60min",
            prediction_mode="csv",
            predictions_csv=str(preds_path),
            commission_per_unit_per_leg=0.0,
            min_commission_per_order=0.0,
            k_sigma_err=0.0,
            k_atr_min_tp=0.1,
        )

    eq_model = np.asarray(result_model.equity_curve, dtype=float)
    eq_csv = np.asarray(result_csv.equity_curve, dtype=float)

    # Basic sanity check.
    assert len(eq_model) == len(eq_csv) > 0

    # Core invariant: equity curves should match up to tiny float noise.
    assert np.allclose(eq_model, eq_csv, atol=1e-9)

    # Optional: trades length parity to catch structure mismatches.
    assert len(result_model.trades) == len(result_csv.trades)
