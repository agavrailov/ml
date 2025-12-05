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

    This version reasons *primarily in returns* rather than absolute price
    distances so that the logic is approximately scale-invariant:

    - predicted_return = (P_pred / P0) - 1
    - sigma_return     = model_error_sigma / P0
    - atr_return       = ATR / P0

    We then apply the same structure as the original strategy, but in return
    space, and finally map TP/SL back to absolute prices.
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
    if predicted_return <= 0.0:
        return None

    sigma_return = state.model_error_sigma / price if price > 0.0 else 0.0
    atr_return = state.atr / price if price > 0.0 else 0.0

    # 2) Error-adjusted usable move (in returns).
    usable_return = predicted_return - cfg.k_sigma_err * sigma_return
    if usable_return <= 0.0:
        return None

    # 3) SNR-based filter: avoid trading on noise.
    #    We interpret ``k_atr_min_tp`` as a minimum signal-to-noise ratio
    #    threshold on usable_return / sigma_return. If ``sigma_return`` is not
    #    available (<= 0), we skip this filter and rely only on usable_return.
    if sigma_return > 0.0:
        snr = usable_return / sigma_return
        if snr < cfg.k_atr_min_tp:
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

    risk_per_unit = stop_dist  # long-only approximation
    if risk_per_unit <= 0.0:
        return None

    size = max_risk_notional / risk_per_unit

    # Enforce minimum size.
    if size < cfg.min_position_size:
        return None

    return TradePlan(size=size, tp_price=tp_price, sl_price=sl_price)
