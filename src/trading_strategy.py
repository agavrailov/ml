"""Core trading strategy primitives for the MVP backtester.

This module implements the concrete strategy defined in
`docs/trading system/trading_system_strategy.md`.

It is intentionally self-contained and focuses only on:
- StrategyConfig
- StrategyState
- TradePlan
- compute_tp_sl_and_size

The backtest engine and data loading live in separate modules.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class StrategyConfig:
    """Configuration for the MVP trading strategy.

    All parameters are intended to be easily sweepable in backtests.
    """

    risk_per_trade_pct: float  # e.g. 0.01 for 1% of equity
    reward_risk_ratio: float  # e.g. 2.0 for 2:1 reward:risk
    k_sigma_err: float  # error margin multiplier on model residual std
    k_atr_min_tp: float  # minimum TP distance as multiple of ATR
    min_position_size: float = 1.0  # minimum size (e.g. 1 share)


@dataclass
class StrategyState:
    """Snapshot of inputs required for a single decision step.

    This is deliberately minimal for now and can be extended as needed.
    """

    current_price: float  # P0
    predicted_price: float  # P_pred
    model_error_sigma: float  # sigma_err (RMSE or residual std for this horizon)
    atr: float  # ATR_H (same timeframe as trades)
    account_equity: float  # current equity in the backtest account

    has_open_position: bool  # True if we already hold a long position


@dataclass
class TradePlan:
    """A concrete trade decision for a NEW long entry.

    For MVP we only support LONG entries; FLAT/NO-TRADE is represented by
    returning None from the decision function.
    """

    size: float  # number of shares/contracts
    tp_price: float  # take profit level
    sl_price: float  # stop loss level


def compute_tp_sl_and_size(state: StrategyState, cfg: StrategyConfig) -> Optional[TradePlan]:
    """Compute TP, SL, and size for a potential new long position.

    Returns a TradePlan if all entry conditions are satisfied; otherwise
    returns None to indicate "no trade".

    Logic follows `trading_system_strategy.md`:
    - Require predicted_move > 0.
    - Define usable_move = predicted_move - k_sigma_err * sigma_err and
      require it to be positive and sufficiently large compared to ATR.
    - Set tp_dist = usable_move and sl_dist = tp_dist / reward_risk_ratio.
    - Size the position so that max_risk_notional = risk_per_trade_pct * equity.
    """

    # Do not open new positions if one is already open.
    if state.has_open_position:
        return None

    # 1) Basic directional check.
    predicted_move = state.predicted_price - state.current_price
    if predicted_move <= 0:
        return None

    # 2) Error-adjusted usable move.
    usable_move = predicted_move - cfg.k_sigma_err * state.model_error_sigma
    if usable_move <= 0:
        return None

    # 3) ATR-based filter: avoid trading on noise.
    min_tp_dist_atr = cfg.k_atr_min_tp * state.atr
    if usable_move < min_tp_dist_atr:
        return None

    # 4) Define TP distance from usable move.
    tp_dist = usable_move
    if tp_dist <= 0:
        return None

    tp_price = state.current_price + tp_dist

    # 5) Define SL from reward:risk ratio.
    if cfg.reward_risk_ratio <= 0:
        return None

    stop_dist = tp_dist / cfg.reward_risk_ratio
    if stop_dist <= 0:
        return None

    sl_price = state.current_price - stop_dist

    # 6) Position sizing based on capital at risk.
    max_risk_notional = cfg.risk_per_trade_pct * state.account_equity
    if max_risk_notional <= 0:
        return None

    risk_per_unit = stop_dist  # long-only approximation
    if risk_per_unit <= 0:
        return None

    size = max_risk_notional / risk_per_unit

    # Enforce minimum size.
    if size < cfg.min_position_size:
        return None

    return TradePlan(size=size, tp_price=tp_price, sl_price=sl_price)
