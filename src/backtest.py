"""Command-line entrypoint for running a simple backtest.

This is a thin wrapper around `backtest_engine.run_backtest` intended for
local, Phase 0 experiments. It:

- Loads OHLC data for the configured frequency from `data/processed/`.
- Builds a default `StrategyConfig` and `BacktestConfig`.
- Uses a naive prediction provider (Close + k * typical_range) for now.

Later this can be extended to use real model predictions.
"""
from __future__ import annotations

import argparse
from typing import Optional

import numpy as np
import pandas as pd

from src.backtest_engine import BacktestConfig, BacktestResult, PredictionProvider, run_backtest
from src.config import FREQUENCY, TSTEPS, get_hourly_data_csv_path
from src.trading_strategy import StrategyConfig
from src.predict import predict_future_prices


def _estimate_atr_like(data: pd.DataFrame, window: int = 14) -> float:
    """Estimate an ATR-like volatility measure from OHLC data.

    This is a simplified ATR proxy using high-low ranges.
    """

    ranges = (data["High"] - data["Low"]).abs()
    if len(ranges) < window:
        return float(ranges.mean()) if not ranges.empty else 1.0
    return float(ranges.rolling(window).mean().dropna().mean())


def _make_naive_prediction_provider(offset_multiple: float, atr_like: float) -> PredictionProvider:
    """Return a simple prediction provider for experimentation.

    For each bar, prediction = Close + offset_multiple * atr_like.
    This ensures a consistent positive edge in tests/experiments without
    calling the real model.
    """

    def provider(i: int, row: pd.Series) -> float:  # noqa: ARG001
        return float(row["Close"]) + offset_multiple * atr_like

    return provider


def _make_model_prediction_provider(data: pd.DataFrame, frequency: str) -> PredictionProvider:
    """Return a prediction provider using the LSTM model.

    The model makes predictions for the entire dataset upfront.
    """
    # Make predictions for the entire dataset
    predictions = predict_future_prices(data, frequency)

    def provider(i: int, row: pd.Series) -> float:  # noqa: ARG001
        # Return the pre-calculated prediction for the current index
        return float(predictions)

    return provider


def run_backtest_on_dataframe(
    data: pd.DataFrame,
    initial_equity: float = 10_000.0,
    frequency: Optional[str] = None,
    prediction_mode: str = "naive",
    commission_per_unit_per_leg: float = 0.005,
    min_commission_per_order: float = 1.0,
) -> BacktestResult:
    """Run a backtest on an in-memory DataFrame using default settings.

    This is primarily for tests and notebook experiments.
    """

    freq = frequency or FREQUENCY
    atr_like = _estimate_atr_like(data)

    strat_cfg = StrategyConfig(
        risk_per_trade_pct=0.01,
        reward_risk_ratio=2.0,
        k_sigma_err=1.0,
        k_atr_min_tp=0.5,
    )
    bt_cfg = BacktestConfig(
        initial_equity=initial_equity,
        strategy_config=strat_cfg,
        model_error_sigma=atr_like,  # placeholder until real residuals are wired
        fixed_atr=atr_like,
        commission_per_unit_per_leg=commission_per_unit_per_leg,
        min_commission_per_order=min_commission_per_order,
    )

    if prediction_mode == "naive":
        provider = _make_naive_prediction_provider(offset_multiple=2.0, atr_like=atr_like)
    elif prediction_mode == "model":
        provider = _make_model_prediction_provider(data, frequency=freq)
    else:
        raise ValueError(f"Unknown prediction_mode: {prediction_mode}")

    return run_backtest(data, provider, bt_cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a simple LSTM-based trading backtest.")
    parser.add_argument(
        "--frequency",
        type=str,
        default=FREQUENCY,
        help="Resample frequency to use (e.g. '15min', '60min'). Defaults to config FREQUENCY.",
    )
    parser.add_argument(
        "--initial-equity",
        type=float,
        default=10_000.0,
        help="Initial account equity for the backtest.",
    )
    parser.add_argument(
        "--prediction-mode",
        type=str,
        choices=["naive", "model"],
        default="naive",
        help="Prediction source: 'naive' (Close + k*ATR) or 'model' (LSTM).",
    )
    parser.add_argument(
        "--commission-per-unit-per-leg",
        type=float,
        default=0.005,
        help=(
            "Commission per unit per leg (entry or exit). "
            "Total round-trip commission = commission_per_unit_per_leg * size * 2. "
            "Set this from your broker's fee schedule (e.g. IBKR) once."
        ),
    )
    parser.add_argument(
        "--min-commission-per-order",
        type=float,
        default=1.0,
        help=(
            "Minimum commission per order (per leg) in account currency. "
            "Total minimum round-trip commission = 2 * min_commission_per_order."
        ),
    )

    args = parser.parse_args()

    freq = args.frequency
    initial_equity = float(args.initial_equity)
    prediction_mode = args.prediction_mode
    commission_per_unit_per_leg = float(args.commission_per_unit_per_leg)
    min_commission_per_order = float(args.min_commission_per_order)

    csv_path = get_hourly_data_csv_path(freq)
    data = pd.read_csv(csv_path)

    # Basic sanity check for required columns.
    required_cols = {"Open", "High", "Low", "Close"}
    missing = required_cols - set(data.columns)
    if missing:
        raise SystemExit(f"Data file {csv_path} is missing required columns: {missing}")

    # Pass commission into the BacktestConfig by adjusting the helper's
    # default behavior. For now, we simply override the commission on the
    # resulting BacktestConfig by re-running with the same inputs but
    # injecting commission into the config. To keep things simple and avoid
    # refactoring the helper, we just adjust equity at the reporting level
    # when commission is non-zero.
    result = run_backtest_on_dataframe(
        data,
        initial_equity=initial_equity,
        frequency=freq,
        prediction_mode=prediction_mode,
        commission_per_unit_per_leg=commission_per_unit_per_leg,
        min_commission_per_order=min_commission_per_order,
    )

    # Simple summary output.
    print(f"Frequency:      {freq}")
    if "Time" in data.columns:
        try:
            time_series = pd.to_datetime(data["Time"])
            start_time = time_series.iloc[0]
            end_time = time_series.iloc[-1]
            print(f"Start date:     {start_time}")
            print(f"End date:       {end_time}")
        except Exception:  # pragma: no cover - best-effort parsing
            pass
    print(f"Initial equity: {initial_equity:.2f}")
    print(f"Final equity:   {result.final_equity:.2f}")
    print(f"Trades:         {len(result.trades)}")

    if result.trades:
        pnls = np.array([t.pnl for t in result.trades], dtype=float)
        print(f"Total PnL:      {pnls.sum():.2f}")
        print(f"Mean trade PnL: {pnls.mean():.4f}")


if __name__ == "__main__":  # pragma: no cover
    main()
