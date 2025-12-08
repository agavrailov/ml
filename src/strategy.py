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

    We allow different filter strengths for long and short trades via
    ``k_sigma_long`` / ``k_sigma_short`` and ``k_atr_long`` / ``k_atr_short``.
    """

    risk_per_trade_pct: float  # e.g. 0.01 for 1% of equity
    reward_risk_ratio: float  # e.g. 2.0 for 2:1 reward:risk

    # Long/short-specific signal filters
    k_sigma_long: float  # error margin multiplier on model residual std (longs)
    k_sigma_short: float  # same, for shorts
    k_atr_long: float  # minimum SNR threshold on usable_return / sigma_return (longs)
    k_atr_short: float  # same, for shorts

    min_position_size: float = 1.0  # minimum size (e.g. 1 share)
    enable_longs: bool = True  # gate for enabling long entries
    allow_shorts: bool = False  # gate for enabling short entries


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

    has_open_position: bool  # True if we already hold a position (long or short)


@dataclass
class TradePlan:
    """A concrete trade decision for a NEW entry.

    ``direction`` is +1 for LONG and -1 for SHORT. FLAT/NO-TRADE is represented
    by returning ``None`` from the decision function.
    """

    direction: int  # +1 = long, -1 = short
    size: float  # number of shares/contracts (always positive)
    tp_price: float  # take profit level
    sl_price: float  # stop loss level


def _compute_trade_plan_for_long(price: float, predicted_return: float, sigma_return: float, atr_return: float, state: StrategyState, cfg: StrategyConfig) -> Optional[TradePlan]:
    """Internal helper: long-side trade plan.

    Logic matches the original long-only implementation but is factored out so
    that a symmetric short-side version can reuse the same structure.
    """

    # 2) Error-adjusted usable move (in returns).
    usable_return = predicted_return - cfg.k_sigma_long * sigma_return
    if usable_return <= 0.0:
        return None

    # 3) SNR-based filter (long side).
    if sigma_return > 0.0:
        snr = usable_return / sigma_return
        if snr < cfg.k_atr_long:
            return None

    # 4) Define TP distance from usable move and convert back to price.
    tp_dist = usable_return * price
    if tp_dist <= 0.0:
        return None

    tp_price = price + tp_dist

    # 5) Define SL from reward:risk ratio.
    if cfg.reward_risk_ratio <= 0.0:
        return None

    stop_dist = tp_dist / cfg.reward_risk_ratio
    if stop_dist <= 0.0:
        return None

    sl_price = price - stop_dist

    # 6) Position sizing based on capital at risk.
    max_risk_notional = cfg.risk_per_trade_pct * state.account_equity
    if max_risk_notional <= 0.0:
        return None

    risk_per_unit = stop_dist  # long-only approximation (price - SL)
    if risk_per_unit <= 0.0:
        return None

    size = max_risk_notional / risk_per_unit
    if size < cfg.min_position_size:
        return None

    return TradePlan(direction=+1, size=size, tp_price=tp_price, sl_price=sl_price)


def _compute_trade_plan_for_short(price: float, predicted_return: float, sigma_return: float, atr_return: float, state: StrategyState, cfg: StrategyConfig) -> Optional[TradePlan]:
    """Internal helper: short-side trade plan.

    Mirrors the long logic in return space, but with inverted direction.
    """

    # For shorts predicted_return is negative; we work with its magnitude.
    predicted_return_abs = -predicted_return

    usable_return = predicted_return_abs - cfg.k_sigma_short * sigma_return
    if usable_return <= 0.0:
        return None

    if sigma_return > 0.0:
        snr = usable_return / sigma_return
        if snr < cfg.k_atr_short:
            return None

    # TP for shorts is below current price.
    tp_dist = usable_return * price
    if tp_dist <= 0.0:
        return None

    tp_price = price - tp_dist

    if cfg.reward_risk_ratio <= 0.0:
        return None

    stop_dist = tp_dist / cfg.reward_risk_ratio
    if stop_dist <= 0.0:
        return None

    sl_price = price + stop_dist

    max_risk_notional = cfg.risk_per_trade_pct * state.account_equity
    if max_risk_notional <= 0.0:
        return None

    # For shorts, risk per unit is SL - entry price.
    risk_per_unit = stop_dist
    if risk_per_unit <= 0.0:
        return None

    size = max_risk_notional / risk_per_unit
    if size < cfg.min_position_size:
        return None

    return TradePlan(direction=-1, size=size, tp_price=tp_price, sl_price=sl_price)


def compute_tp_sl_and_size(state: StrategyState, cfg: StrategyConfig) -> Optional[TradePlan]:
    """Compute TP, SL, size, and direction for a potential new position.

    Returns a TradePlan if all entry conditions are satisfied; otherwise
    returns None to indicate "no trade".

    This version reasons *primarily in returns* rather than absolute price
    distances so that the logic is approximately scale-invariant:

    - predicted_return = (P_pred / P0) - 1
    - sigma_return     = model_error_sigma / P0
    - atr_return       = ATR / P0

    It supports both long and short entries by mirroring the long logic for
    negative predicted returns.
    """

    # Do not open new positions if one is already open.
    if state.has_open_position:
        return None

    # Guard against nonsensical prices.
    if state.current_price <= 0:
        return None

    price = float(state.current_price)

    # 1) Basic directional check, in return space.
    predicted_return = (state.predicted_price / price) - 1.0
    if predicted_return == 0.0:
        return None

    sigma_return = state.model_error_sigma / price if price > 0.0 else 0.0
    atr_return = state.atr / price if price > 0.0 else 0.0

    if predicted_return > 0.0:
        # Long signal: only act when longs are enabled.
        if not cfg.enable_longs:
            return None
        return _compute_trade_plan_for_long(price, predicted_return, sigma_return, atr_return, state, cfg)
    else:
        # Short signal: consider a short only when enabled.
        if not cfg.allow_shorts:
            return None
        return _compute_trade_plan_for_short(price, predicted_return, sigma_return, atr_return, state, cfg)
