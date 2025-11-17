"""Minimal long-only backtest engine for the MVP trading system.

This is a Phase 0, offline-only backtester that:
- Iterates over historical OHLC bars.
- Uses a prediction provider + StrategyConfig to decide on entries.
- Simulates TP/SL exits and updates equity.

It is intentionally simple and focused on NVDA/hourly use cases but is
kept generic enough for extension.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import pandas as pd
import numpy as np

from src.trading_strategy import StrategyConfig, StrategyState, TradePlan, compute_tp_sl_and_size


@dataclass
class BacktestConfig:
    """Configuration for a single backtest run.

    For MVP we assume:
    - Long-only trades.
    - Single open position at a time.
    - Execution at the bar where the decision is made (using `Close`).
    - TP/SL evaluated against subsequent bars' High/Low.
    - Optional per-unit commission per leg (entry + exit).
    """

    initial_equity: float
    strategy_config: StrategyConfig

    # For now we treat model error sigma and ATR as scalars; they can be
    # extended to per-bar series via additional inputs.
    model_error_sigma: float
    fixed_atr: float

    # Commission per unit per leg. The total commission for a round-trip
    # trade is: commission_per_unit_per_leg * size * 2.
    # Default is 0.005 USD per share per leg, matching IBKR fixed pricing.
    commission_per_unit_per_leg: float = 0.005

    # Minimum commission per order (per leg). Total minimum round-trip
    # commission is 2 * min_commission_per_order.
    min_commission_per_order: float = 1.0


@dataclass
class Position:
    entry_index: int
    entry_price: float
    size: float
    tp_price: float
    sl_price: float


@dataclass
class Trade:
    entry_index: int
    exit_index: int
    entry_price: float
    exit_price: float
    size: float
    commission: float = 0.0

    @property
    def gross_pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.size

    @property
    def pnl(self) -> float:
        """Net PnL after commission."""
        return self.gross_pnl - self.commission


@dataclass
class BacktestResult:
    equity_curve: List[float]
    trades: List[Trade]

    @property
    def final_equity(self) -> float:
        return self.equity_curve[-1] if self.equity_curve else 0.0


    # Prediction provider type: given the current bar index and row, return a
# predicted price at the chosen horizon (e.g. next-bar close).
PredictionProvider = Callable[[int, pd.Series], float]


def run_backtest(
    data: pd.DataFrame,
    prediction_provider: PredictionProvider,
    cfg: BacktestConfig,
    atr_series: Optional[pd.Series] = None,
    model_error_sigma_series: Optional[pd.Series] = None,
) -> BacktestResult:
    """Run a simple long-only backtest over `data`.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain at least `Close`, `High`, `Low` columns. Index can be
        anything; we iterate positionally.
    prediction_provider : callable
        Function that takes (i, row) and returns a predicted price for the
        target horizon.
    cfg : BacktestConfig
        Backtest configuration including initial equity and strategy params.
    """

    required_cols = {"Open", "Close", "High", "Low"}
    missing = required_cols - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns in data: {missing}")

    equity = cfg.initial_equity
    equity_curve: List[float] = []
    trades: List[Trade] = []

    position: Optional[Position] = None

    n = len(data)
    if n == 0:
        return BacktestResult(equity_curve=[equity], trades=[])

    # Iterate over bars; for simplicity we:
    # - Make decisions on bar i using bar i's Close (no look-ahead).
    # - Open positions at the *next* bar's Open (i+1).
    # - Evaluate TP/SL on bars after entry (i+1, i+2, ...).
    #
    # For long runs, emit a lightweight textual progress indicator roughly
    # every 0.1% of bars so the CLI shows that work is progressing.
    progress_step = max(n // 1000, 1)  # 1000 steps ≈ 0.1% increments

    for i in range(n):
        if i % progress_step == 0:
            pct = (i / n) * 100.0
            # Print on a new line so progress is clearly visible in all terminals.
            print(f"Backtest progress: {pct:5.1f}% ({i}/{n} bars)", flush=True)

        row = data.iloc[i]
        # 1) If there is an open position, check for exits on this bar.
        if position is not None:
            high = float(row["High"])
            low = float(row["Low"])

            exit_price: Optional[float] = None

            # Simple rule: SL first, then TP if SL not hit.
            if low <= position.sl_price:
                exit_price = position.sl_price
            elif high >= position.tp_price:
                exit_price = position.tp_price

            if exit_price is not None:
                per_leg_commission = max(
                    cfg.commission_per_unit_per_leg * position.size,
                    cfg.min_commission_per_order,
                )
                commission = per_leg_commission * 2.0
                trade = Trade(
                    entry_index=position.entry_index,
                    exit_index=i,
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    size=position.size,
                    commission=commission,
                )
                equity += trade.pnl
                trades.append(trade)
                position = None

        # 2) If no open position, consider opening a new one.
        if position is None and i < n - 1:
            # Decision is made at bar i using bar i's Close. Entry happens at
            # bar i+1 Open to avoid the unrealistic assumption that we can
            # always fill at the same bar's Close.
            predicted_price = float(prediction_provider(i, row))

            decision_price = float(row["Close"])

            # Resolve per-bar risk inputs. If per-bar series are provided,
            # prefer them; otherwise fall back to the scalar config values.
            if model_error_sigma_series is not None:
                model_sigma = float(model_error_sigma_series.iloc[i])
            else:
                model_sigma = float(cfg.model_error_sigma)

            if atr_series is not None:
                atr_value = float(atr_series.iloc[i])
            else:
                atr_value = float(cfg.fixed_atr)

            # If any of the core decision inputs are NaN or non-positive where
            # positivity is required (ATR, sigma), skip opening a new trade for
            # this bar. This prevents NaNs from propagating into position sizing
            # and PnL.
            if not np.isfinite(predicted_price) or not np.isfinite(decision_price):
                equity_curve.append(equity)
                continue
            if not (np.isfinite(model_sigma) and np.isfinite(atr_value)):
                equity_curve.append(equity)
                continue
            if not (model_sigma > 0 and atr_value > 0):
                equity_curve.append(equity)
                continue

            state = StrategyState(
                current_price=decision_price,
                predicted_price=predicted_price,
                model_error_sigma=model_sigma,
                atr=atr_value,
                account_equity=float(equity),
                has_open_position=False,
            )

            plan: Optional[TradePlan] = compute_tp_sl_and_size(state, cfg.strategy_config)
            if plan is not None:
                # Use distances from the decision price to adjust TP/SL to the
                # actual entry price (next bar's Open).
                next_row = data.iloc[i + 1]
                entry_price = float(next_row["Open"])
                tp_dist = float(plan.tp_price) - decision_price
                sl_dist = decision_price - float(plan.sl_price)

                position = Position(
                    entry_index=i + 1,
                    entry_price=entry_price,
                    size=float(plan.size),
                    tp_price=entry_price + tp_dist,
                    sl_price=entry_price - sl_dist,
                )

        equity_curve.append(equity)

    # 3) If a position is still open at the end, close at last Close.
    if position is not None:
        last_row = data.iloc[-1]
        exit_price = float(last_row["Close"])
        per_leg_commission = max(
            cfg.commission_per_unit_per_leg * position.size,
            cfg.min_commission_per_order,
        )
        commission = per_leg_commission * 2.0
        trade = Trade(
            entry_index=position.entry_index,
            exit_index=n - 1,
            entry_price=position.entry_price,
            exit_price=exit_price,
            size=position.size,
            commission=commission,
        )
        equity += trade.pnl
        trades.append(trade)
        equity_curve[-1] = equity

    return BacktestResult(equity_curve=equity_curve, trades=trades)
