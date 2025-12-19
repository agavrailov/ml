"""Trading session helpers.

This module wires together:
- Strategy (src.strategy)
- Broker abstraction (src.broker)
- Execution mapping (src.execution)

The goal is to provide a minimal "session" runner that can:
- Iterate over a sequence of OHLC bars and predictions.
- Use the MVP strategy to compute a TradePlan.
- Submit the TradePlan via a Broker (SimulatedBroker by default).

For now this is intended for experimentation and tests rather than a
production-grade live engine. It deliberately keeps the state model simple:
- At most one open position at a time.
- We track "has_open_position" locally instead of reconciling with broker
  positions on every step.

Later, this module can be extended to:
- Use broker-reported positions and account equity as source of truth.
- Support richer risk management and order lifecycle handling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd

from src.broker import Broker, SimulatedBroker
from src.config import (
    BROKER_BACKEND,
    K_ATR_LONG,
    K_ATR_SHORT,
    K_SIGMA_LONG,
    K_SIGMA_SHORT,
    RISK_PER_TRADE_PCT,
    REWARD_RISK_RATIO,
)
from src.execution import ExecutionContext, submit_trade_plan
from src.strategy import StrategyConfig, StrategyState, TradePlan, compute_tp_sl_and_size


@dataclass
class SessionConfig:
    """Configuration for a simple single-symbol trading session."""

    symbol: str = "NVDA"
    initial_equity: float = 10_000.0
    backend: Optional[str] = None  # When None, fall back to config.BROKER_BACKEND.


def make_strategy_config_from_defaults() -> StrategyConfig:
    """Build a StrategyConfig using global STRATEGY_DEFAULTS.

    For now we use the same filters for both long and short sides as indicated
    by the flat config aliases.
    """

    return StrategyConfig(
        risk_per_trade_pct=RISK_PER_TRADE_PCT,
        reward_risk_ratio=REWARD_RISK_RATIO,
        k_sigma_long=K_SIGMA_LONG,
        k_sigma_short=K_SIGMA_SHORT,
        k_atr_long=K_ATR_LONG,
        k_atr_short=K_ATR_SHORT,
    )


def make_broker(backend: Optional[str] = None, *, ib: object | None = None) -> Broker:
    """Construct a Broker instance based on backend string.

    Supported backends:
    - "SIM"      -> SimulatedBroker (in-memory).
    - "IBKR_TWS" -> IBKRBrokerTws (requires ib_insync and src.ibkr_broker).

    Parameters
    ----------
    ib : object, optional
        Optional injected ``ib_insync.IB`` instance. When provided and
        ``backend==IBKR_TWS``, the broker will reuse the same socket connection as
        the live engine (single-IB design).
    """

    backend_eff = (backend or BROKER_BACKEND).upper()

    if backend_eff == "SIM":
        return SimulatedBroker()
    elif backend_eff == "IBKR_TWS":  # pragma: no cover - relies on ib_insync at runtime
        from src.ibkr_broker import IBKRBrokerConfig, IBKRBrokerTws

        cfg = IBKRBrokerConfig.from_global_config()
        return IBKRBrokerTws(config=cfg, ib=ib)  # type: ignore[arg-type]
    else:
        raise ValueError(f"Unsupported broker backend: {backend_eff!r}")


def _make_strategy_state_for_bar(
    *,
    current_price: float,
    predicted_price: float,
    model_error_sigma: float,
    atr: float,
    account_equity: float,
    has_open_position: bool,
) -> StrategyState:
    return StrategyState(
        current_price=current_price,
        predicted_price=predicted_price,
        model_error_sigma=model_error_sigma,
        atr=atr,
        account_equity=account_equity,
        has_open_position=has_open_position,
    )


def run_session_over_dataframe(
    data: pd.DataFrame,
    predictions: pd.Series,
    *,
    cfg: Optional[SessionConfig] = None,
    broker: Optional[Broker] = None,
) -> Iterable[TradePlan]:
    """Iterate over bars, compute TradePlans, and submit them via a Broker.

    Parameters
    ----------
    data : DataFrame
        Must contain a ``Close`` column at minimum.
    predictions : Series
        Per-bar predicted price, aligned index-wise with ``data``.
    cfg : SessionConfig, optional
        Session settings (symbol, backend, initial equity).
    broker : Broker, optional
        If provided, use this broker instead of constructing one.

    Yields
    ------
    TradePlan
        Each non-None TradePlan that was submitted.

    Notes
    -----
    - This function is intentionally simple: it does not attempt to track
      fills or exits, and assumes at most one open position at a time based on
      local state.
    - It is primarily useful for plumbing tests and simple experiments where
      you want to see the strategy drive a Broker implementation.
    """

    if cfg is None:
        cfg = SessionConfig()

    if broker is None:
        broker = make_broker(cfg.backend)

    if data.empty:
        return []

    if len(predictions) != len(data):
        raise ValueError("predictions length must match data length for this simple session runner")

    strat_cfg = make_strategy_config_from_defaults()
    ctx = ExecutionContext(symbol=cfg.symbol)

    equity = cfg.initial_equity
    has_open_position = False

    for i in range(len(data)):
        row = data.iloc[i]
        current_price = float(row["Close"])
        predicted_price = float(predictions.iloc[i])

        # For now we use coarse placeholders for model_error_sigma and ATR.
        # They can be wired to real per-bar series later.
        model_error_sigma = max(1e-6, 0.5 * current_price * 0.01)  # ~0.5% of price
        atr = max(1e-6, current_price * 0.01)  # ~1% of price

        state = _make_strategy_state_for_bar(
            current_price=current_price,
            predicted_price=predicted_price,
            model_error_sigma=model_error_sigma,
            atr=atr,
            account_equity=equity,
            has_open_position=has_open_position,
        )

        plan = compute_tp_sl_and_size(state, strat_cfg)
        if plan is None:
            continue

        # Submit via Broker and mark that we "opened" a position.
        submit_trade_plan(broker, plan, ctx)
        has_open_position = True

        yield plan

        # Simple policy: stop after first trade in this demo runner.
        break

    # In the future this function can be extended to update equity based on
    # realised PnL and allow subsequent trades.
    return []
