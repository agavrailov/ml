"""Phase 1 paper-trading scaffold.

This module provides a minimal paper-trading loop that *reuses* the
backtest engine and strategy configuration but runs over a streaming
sequence of bars.

For now, the primary entrypoint is a CLI that simulates a paper-trading
session over historical NVDA OHLC data (from ``data/processed/``) to
exercise the logic end-to-end. Later, the same engine can be wired to a
real-time data adapter (e.g. IB/TWS).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.backtest_engine import BacktestConfig, BacktestResult, run_backtest
from src.config import (
    FREQUENCY,
    RISK_PER_TRADE_PCT,
    REWARD_RISK_RATIO,
    K_SIGMA_ERR,
    K_ATR_MIN_TP,
    INITIAL_EQUITY,
    COMMISSION_PER_UNIT_PER_LEG,
    MIN_COMMISSION_PER_ORDER,
    BACKTEST_DEFAULT_START_DATE,
    BACKTEST_DEFAULT_END_DATE,
    get_hourly_data_csv_path,
)
from src.data import load_hourly_ohlc
from src.strategy import StrategyConfig
from src.backtest import (
    _compute_atr_series,
    _load_predictions_csv,
    _make_csv_prediction_provider,
    _plot_price_and_equity_with_trades,
    _apply_date_range,
)


@dataclass
class PaperTradingConfig:
    """Configuration for a paper-trading session.

    This mirrors ``BacktestConfig`` but is kept separate so we can evolve it
    independently (e.g. add logging destinations, real-time feed settings).
    """

    initial_equity: float = INITIAL_EQUITY
    frequency: str = FREQUENCY
    commission_per_unit_per_leg: float = COMMISSION_PER_UNIT_PER_LEG
    min_commission_per_order: float = MIN_COMMISSION_PER_ORDER

    # Strategy-specific parameters; kept flat for easier CLI mapping. Defaults
    # come from STRATEGY_DEFAULTS in config.py so they stay in sync with
    # backtest defaults, but can still be overridden per paper-trading run.
    risk_per_trade_pct: float = RISK_PER_TRADE_PCT
    reward_risk_ratio: float = REWARD_RISK_RATIO
    k_sigma_err: float = K_SIGMA_ERR
    k_atr_min_tp: float = K_ATR_MIN_TP


def _build_backtest_config(cfg: PaperTradingConfig, atr_like: float) -> BacktestConfig:
    """Translate a ``PaperTradingConfig`` into a ``BacktestConfig``.

    ``PaperTradingConfig`` still exposes shared noise filters
    (``k_sigma_err``, ``k_atr_min_tp``) for simplicity. Here we map them onto
    the long/short-specific fields expected by ``StrategyConfig`` so that paper
    trading stays aligned with the core strategy implementation.
    """

    strat_cfg = StrategyConfig(
        risk_per_trade_pct=cfg.risk_per_trade_pct,
        reward_risk_ratio=cfg.reward_risk_ratio,
        # Use the same shared filters for both long and short sides in the
        # paper-trading CLI; callers who need asymmetric behavior can switch to
        # the lower-level backtest API.
        k_sigma_long=cfg.k_sigma_err,
        k_sigma_short=cfg.k_sigma_err,
        k_atr_long=cfg.k_atr_min_tp,
        k_atr_short=cfg.k_atr_min_tp,
    )

    return BacktestConfig(
        initial_equity=cfg.initial_equity,
        strategy_config=strat_cfg,
        model_error_sigma=atr_like,
        fixed_atr=atr_like,
        commission_per_unit_per_leg=cfg.commission_per_unit_per_leg,
        min_commission_per_order=cfg.min_commission_per_order,
    )


def run_paper_trading_over_dataframe(
    data: pd.DataFrame,
    cfg: Optional[PaperTradingConfig] = None,
    predictions_csv: Optional[str] = None,
) -> BacktestResult:
    """Simulate a paper-trading session over a historical DataFrame.

    This is intentionally very close to ``run_backtest`` but structured as a
    separate function so it can be called from a future real-time loop by
    feeding bars incrementally.
    """

    if cfg is None:
        cfg = PaperTradingConfig()

    if data.empty:
        return BacktestResult(equity_curve=[cfg.initial_equity], trades=[])

    # Compute per-bar ATR series (same helper as used by the backtest CLI).
    atr_series = _compute_atr_series(data, window=14)
    atr_like = float(atr_series.dropna().mean()) if not atr_series.dropna().empty else 1.0
    bt_cfg = _build_backtest_config(cfg, atr_like=atr_like)

    if predictions_csv is None:
        raise ValueError(
            "run_paper_trading_over_dataframe requires predictions_csv path for CSV-based predictions",
        )

    preds_df = _load_predictions_csv(predictions_csv)
    provider = _make_csv_prediction_provider(preds_df, data)

    required_cols = {"Open", "High", "Low", "Close"}
    missing = required_cols - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns in data: {missing}")

    # Delegate to the core backtest engine so that paper trading uses the
    # exact same entry/exit logic and per-bar ATR handling.
    result = run_backtest(
        data,
        prediction_provider=provider,
        cfg=bt_cfg,
        atr_series=atr_series,
        model_error_sigma_series=atr_series,
    )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulated paper-trading loop over historical NVDA data.")
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
        help="Initial account equity for the paper-trading session.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional start date (YYYY-MM-DD) for the paper-trade window.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional end date (YYYY-MM-DD) for the paper-trade window.",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help=(
            "Optional override for the input OHLC CSV. When omitted, the NVDA "
            "file under data/processed/ for the chosen frequency is used."
        ),
    )
    parser.add_argument(
        "--predictions-csv",
        type=str,
        default="data/processed/predictions.csv",
        help=(
            "Path to a per-bar predictions CSV with at least 'predicted_price' "
            "(and optionally 'Time') columns. Used as the prediction source for "
            "paper trading."
        ),
    )

    args = parser.parse_args()

    freq = args.frequency
    initial_equity = float(args.initial_equity)
    csv_path = args.csv_path or get_hourly_data_csv_path(freq)

    if args.csv_path is None:
        # Use centralized loader so that all invariants for hourly OHLC data
        # (schema, Time monotonicity, NaNs) are enforced consistently.
        data_full = load_hourly_ohlc(freq)
    else:
        data_full = pd.read_csv(csv_path)
    data, date_from, date_to = _apply_date_range(
        data_full,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    cfg = PaperTradingConfig(initial_equity=initial_equity, frequency=freq)
    result = run_paper_trading_over_dataframe(data, cfg=cfg, predictions_csv=args.predictions_csv)

    print("[paper] Session summary")
    print(f"Frequency:      {freq}")
    if "Time" in data.columns and not data.empty:
        try:
            time_series = pd.to_datetime(data["Time"])
            start_time = time_series.iloc[0]
            end_time = time_series.iloc[-1]
            print(f"Start date:     {start_time}")
            print(f"End date:       {end_time}")
        except Exception:
            print(f"Start/End date: {date_from} -> {date_to}")
    else:
        print(f"Start/End date: {date_from} -> {date_to}")
    print(f"Initial equity: {initial_equity:.2f}")
    print(f"Final equity:   {result.final_equity:.2f}")
    print(f"Trades:         {len(result.trades)}")

    # Optional visualization: reuse the backtest helper to save a price +
    # equity + trades diagram under backtests/. This uses matplotlib if
    # available and otherwise no-ops with a message, so it is safe for CLI
    # usage and tests.
    _plot_price_and_equity_with_trades(
        data,
        result,
        symbol="NVDA",
        freq=freq,
        k_sigma_err=cfg.k_sigma_err,
        k_atr_min_tp=cfg.k_atr_min_tp,
        risk_per_trade_pct=cfg.risk_per_trade_pct,
        reward_risk_ratio=cfg.reward_risk_ratio,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
