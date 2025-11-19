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
from typing import Iterable, Optional, List

import numpy as np
import pandas as pd

from src.backtest_engine import BacktestConfig, BacktestResult, Position, Trade
from src.config import FREQUENCY, get_hourly_data_csv_path
from src.trading_strategy import StrategyConfig, StrategyState, TradePlan, compute_tp_sl_and_size
from src.backtest import _estimate_atr_like, _load_predictions_csv, _make_csv_prediction_provider


@dataclass
class PaperTradingConfig:
    """Configuration for a paper-trading session.

    This mirrors ``BacktestConfig`` but is kept separate so we can evolve it
    independently (e.g. add logging destinations, real-time feed settings).
    """

    initial_equity: float = 10_000.0
    frequency: str = FREQUENCY
    commission_per_unit_per_leg: float = 0.005
    min_commission_per_order: float = 1.0

    # Strategy-specific parameters; kept flat for easier CLI mapping.
    risk_per_trade_pct: float = 0.01
    reward_risk_ratio: float = 2.0
    k_sigma_err: float = 1.0
    k_atr_min_tp: float = 0.5


@dataclass
class PaperTradingState:
    """Mutable state for an ongoing paper-trading session."""

    equity: float
    position: Optional[Position]
    trades: List[Trade]
    equity_curve: List[float]


def _build_backtest_config(cfg: PaperTradingConfig, atr_like: float) -> BacktestConfig:
    """Translate a ``PaperTradingConfig`` into a ``BacktestConfig``."""

    strat_cfg = StrategyConfig(
        risk_per_trade_pct=cfg.risk_per_trade_pct,
        reward_risk_ratio=cfg.reward_risk_ratio,
        k_sigma_err=cfg.k_sigma_err,
        k_atr_min_tp=cfg.k_atr_min_tp,
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

    atr_like = _estimate_atr_like(data)
    bt_cfg = _build_backtest_config(cfg, atr_like=atr_like)

    if predictions_csv is None:
        raise ValueError(
            "run_paper_trading_over_dataframe requires predictions_csv path for CSV-based predictions",
        )

    preds_df = _load_predictions_csv(predictions_csv)
    provider = _make_csv_prediction_provider(preds_df, data)

    # Internal state mirrors the backtest engine.
    state = PaperTradingState(
        equity=bt_cfg.initial_equity,
        position=None,
        trades=[],
        equity_curve=[],
    )

    required_cols = {"Open", "High", "Low", "Close"}
    missing = required_cols - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns in data: {missing}")

    n = len(data)
    for i in range(n):
        row = data.iloc[i]

        # 1) Exit logic for any open position.
        if state.position is not None:
            high = float(row["High"])
            low = float(row["Low"])

            exit_price: Optional[float] = None
            if low <= state.position.sl_price:
                exit_price = state.position.sl_price
            elif high >= state.position.tp_price:
                exit_price = state.position.tp_price

            if exit_price is not None:
                per_leg_commission = max(
                    bt_cfg.commission_per_unit_per_leg * state.position.size,
                    bt_cfg.min_commission_per_order,
                )
                commission = per_leg_commission * 2.0
                trade = Trade(
                    entry_index=state.position.entry_index,
                    exit_index=i,
                    entry_price=state.position.entry_price,
                    exit_price=exit_price,
                    size=state.position.size,
                    commission=commission,
                )
                state.equity += trade.pnl
                state.trades.append(trade)
                state.position = None

        # 2) Entry logic if flat and not at the last bar.
        if state.position is None and i < n - 1:
            predicted_price = provider(i, row)
            decision_price = float(row["Close"])
            strat_state = StrategyState(
                current_price=decision_price,
                predicted_price=float(predicted_price),
                model_error_sigma=float(bt_cfg.model_error_sigma),
                atr=float(bt_cfg.fixed_atr),
                account_equity=float(state.equity),
                has_open_position=False,
            )

            plan: Optional[TradePlan] = compute_tp_sl_and_size(strat_state, bt_cfg.strategy_config)
            if plan is not None:
                next_row = data.iloc[i + 1]
                entry_price = float(next_row["Open"])
                tp_dist = float(plan.tp_price) - decision_price
                sl_dist = decision_price - float(plan.sl_price)

                state.position = Position(
                    entry_index=i + 1,
                    entry_price=entry_price,
                    size=float(plan.size),
                    tp_price=entry_price + tp_dist,
                    sl_price=entry_price - sl_dist,
                )

        state.equity_curve.append(state.equity)

    # 3) Close any remaining open position at the last Close.
    if state.position is not None:
        last_row = data.iloc[-1]
        exit_price = float(last_row["Close"])
        per_leg_commission = max(
            bt_cfg.commission_per_unit_per_leg * state.position.size,
            bt_cfg.min_commission_per_order,
        )
        commission = per_leg_commission * 2.0
        trade = Trade(
            entry_index=state.position.entry_index,
            exit_index=n - 1,
            entry_price=state.position.entry_price,
            exit_price=exit_price,
            size=state.position.size,
            commission=commission,
        )
        state.equity += trade.pnl
        state.trades.append(trade)
        state.equity_curve[-1] = state.equity

    return BacktestResult(equity_curve=state.equity_curve, trades=state.trades)


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

    data = pd.read_csv(csv_path)

    cfg = PaperTradingConfig(initial_equity=initial_equity, frequency=freq)
    result = run_paper_trading_over_dataframe(data, cfg=cfg, predictions_csv=args.predictions_csv)

    print("[paper] Session summary")
    print(f"Frequency:      {freq}")
    if "Time" in data.columns:
        try:
            time_series = pd.to_datetime(data["Time"])
            start_time = time_series.iloc[0]
            end_time = time_series.iloc[-1]
            print(f"Start date:     {start_time}")
            print(f"End date:       {end_time}")
        except Exception:
            pass
    print(f"Initial equity: {initial_equity:.2f}")
    print(f"Final equity:   {result.final_equity:.2f}")
    print(f"Trades:         {len(result.trades)}")


if __name__ == "__main__":  # pragma: no cover
    main()
