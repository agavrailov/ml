import numpy as np
import pandas as pd

from src import backtest as backtest_mod
from src.backtest_engine import BacktestResult
from src.config import ROWS_AHEAD


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


def test_csv_mode_uses_residual_sigma_not_raw_atr(tmp_path, monkeypatch):
    data = _dummy_ohlc_with_time(200)

    # Synthetic predictions with a small edge to ensure non-zero residual sigma.
    preds = pd.DataFrame({
        "Time": data["Time"],
        "predicted_price": data["Open"] * 1.01,
    })
    preds_path = tmp_path / "preds.csv"
    preds.to_csv(preds_path, index=False)

    captured = {}

    def _fake_run_backtest(data_arg, provider, bt_cfg, atr_series, model_error_sigma_series):
        # Capture the series that backtest receives.
        captured["atr_series"] = atr_series
        captured["sigma_series"] = model_error_sigma_series
        # Touch provider once to ensure it is callable.
        _ = provider(0, data_arg.iloc[0])
        return BacktestResult(equity_curve=[bt_cfg.initial_equity] * len(data_arg), trades=[])

    monkeypatch.setattr(backtest_mod, "run_backtest", _fake_run_backtest)

    backtest_mod.run_backtest_on_dataframe(
        data,
        initial_equity=10_000.0,
        frequency="60min",
        prediction_mode="csv",
        predictions_csv=str(preds_path),
        commission_per_unit_per_leg=0.0,
        min_commission_per_order=0.0,
    )

    atr_series = captured["atr_series"]
    sigma_series = captured["sigma_series"]

    assert isinstance(sigma_series, pd.Series)
    assert len(sigma_series) == len(atr_series) == len(data)

    # With non-trivial residuals, sigma should not be identically equal to ATR.
    assert not sigma_series.equals(atr_series)
    # And should contain some positive finite values.
    assert np.isfinite(sigma_series).any()
    assert (sigma_series > 0).any()


def test_csv_mode_falls_back_to_atr_when_residual_sigma_zero(tmp_path, monkeypatch):
    data = _dummy_ohlc_with_time(200)

    # Predictions exactly equal to the future Open (shifted by ROWS_AHEAD)
    # so residuals are ~0 and residual_sigma should collapse to zeros,
    # triggering ATR fallback.
    future_open = data["Open"].shift(-ROWS_AHEAD)
    preds = pd.DataFrame({
        "Time": data["Time"],
        "predicted_price": future_open,
    })
    preds_path = tmp_path / "preds_zero_resid.csv"
    preds.to_csv(preds_path, index=False)

    captured = {}

    def _fake_run_backtest(data_arg, provider, bt_cfg, atr_series, model_error_sigma_series):
        captured["atr_series"] = atr_series
        captured["sigma_series"] = model_error_sigma_series
        _ = provider(0, data_arg.iloc[0])
        return BacktestResult(equity_curve=[bt_cfg.initial_equity] * len(data_arg), trades=[])

    monkeypatch.setattr(backtest_mod, "run_backtest", _fake_run_backtest)

    backtest_mod.run_backtest_on_dataframe(
        data,
        initial_equity=10_000.0,
        frequency="60min",
        prediction_mode="csv",
        predictions_csv=str(preds_path),
        commission_per_unit_per_leg=0.0,
        min_commission_per_order=0.0,
    )

    atr_series = captured["atr_series"]
    sigma_series = captured["sigma_series"]

    assert isinstance(sigma_series, pd.Series)
    assert len(sigma_series) == len(atr_series) == len(data)

    # When residual sigma collapses, we expect a fallback to ATR.
    assert sigma_series.equals(atr_series)
