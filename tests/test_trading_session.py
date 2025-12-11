"""Tests for src.trading_session.

These tests validate that the simple session runner can:
- Build a StrategyConfig from config defaults.
- Use a Broker (SimulatedBroker or a recording stub) to submit TradePlans.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.broker import Broker, OrderRequest
from src.trading_session import SessionConfig, make_broker, make_strategy_config_from_defaults, run_session_over_dataframe


def test_make_broker_sim_backend() -> None:
    broker = make_broker("SIM")
    # We cannot reliably use isinstance with Protocol without @runtime_checkable,
    # so just assert that the core Broker methods exist.
    assert hasattr(broker, "place_order")
    assert hasattr(broker, "get_open_orders")


def test_make_strategy_config_from_defaults_roundtrip() -> None:
    cfg = make_strategy_config_from_defaults()
    # Ensure core fields are present and reasonable.
    assert cfg.risk_per_trade_pct > 0
    assert cfg.reward_risk_ratio > 0


@dataclass
class _RecordingBroker(Broker):
    placed: list[OrderRequest] | None = None

    def __post_init__(self) -> None:  # type: ignore[override]
        if self.placed is None:
            self.placed = []

    def place_order(self, order: OrderRequest) -> str:  # type: ignore[override]
        assert self.placed is not None
        self.placed.append(order)
        return str(len(self.placed))

    def cancel_order(self, order_id: str) -> None:  # type: ignore[override]
        return None

    def get_open_orders(self):  # type: ignore[override]
        return []

    def get_positions(self):  # type: ignore[override]
        return []

    def get_account_summary(self):  # type: ignore[override]
        return {}


def test_run_session_over_dataframe_submits_at_most_one_order() -> None:
    # Simple 4-bar dataset with rising price and predictions above price so that
    # the strategy is likely to generate a long entry.
    data = pd.DataFrame(
        {
            "Close": [100.0, 101.0, 102.0, 103.0],
        }
    )
    preds = pd.Series([102.0, 103.0, 104.0, 105.0])

    broker = _RecordingBroker()
    cfg = SessionConfig(symbol="NVDA", initial_equity=10_000.0, backend="SIM")

    plans = list(run_session_over_dataframe(data, preds, cfg=cfg, broker=broker))

    # The demo runner should submit at most one trade plan.
    assert len(plans) <= 1

    if plans:
        # If we did get a trade plan, ensure the broker saw exactly one order.
        assert broker.placed is not None
        assert len(broker.placed) == 1
        order = broker.placed[0]
        assert order.symbol == cfg.symbol
        assert order.quantity > 0
