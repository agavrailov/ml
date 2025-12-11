"""Tests for src.execution helpers.

We validate that TradePlan objects are correctly translated into Broker
OrderRequest instances and that submit_trade_plan delegates to the broker
appropriately.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from src.broker import Broker, OrderRequest, OrderType, Side
from src.execution import ExecutionContext, submit_trade_plan, trade_plan_to_order_request
from src.strategy import TradePlan


@dataclass
class _RecordingBroker(Broker):
    """Minimal broker implementation for testing submit_trade_plan.

    It records the last order it was asked to place and returns a fixed id.
    """

    last_order: OrderRequest | None = None
    placed_ids: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:  # type: ignore[override]
        if self.placed_ids is None:
            self.placed_ids = []

    def place_order(self, order: OrderRequest) -> str:  # type: ignore[override]
        self.last_order = order
        oid = f"test-{len(self.placed_ids) + 1}"
        self.placed_ids.append(oid)
        return oid

    def cancel_order(self, order_id: str) -> None:  # type: ignore[override]
        return None

    def get_open_orders(self):  # type: ignore[override]
        return []

    def get_positions(self):  # type: ignore[override]
        return []

    def get_account_summary(self) -> Dict[str, float]:  # type: ignore[override]
        return {}


def test_trade_plan_to_order_request_long_direction() -> None:
    ctx = ExecutionContext(symbol="NVDA")
    plan = TradePlan(direction=+1, size=3.7, tp_price=110.0, sl_price=95.0)

    order = trade_plan_to_order_request(plan, ctx)

    assert order.symbol == "NVDA"
    assert order.side is Side.BUY
    assert order.order_type is OrderType.MARKET
    # Quantity should be rounded to nearest whole number and stored as float.
    assert order.quantity == 4.0
    assert order.limit_price is None
    assert order.time_in_force == "DAY"


def test_trade_plan_to_order_request_short_direction() -> None:
    ctx = ExecutionContext(symbol="NVDA")
    plan = TradePlan(direction=-1, size=1.2, tp_price=90.0, sl_price=105.0)

    order = trade_plan_to_order_request(plan, ctx)

    assert order.symbol == "NVDA"
    assert order.side is Side.SELL
    assert order.order_type is OrderType.MARKET
    assert order.quantity == 1.0


def test_submit_trade_plan_is_noop_for_none() -> None:
    ctx = ExecutionContext(symbol="NVDA")
    broker = _RecordingBroker()

    oid = submit_trade_plan(broker, None, ctx)

    assert oid is None
    assert broker.last_order is None


def test_submit_trade_plan_places_order_via_broker() -> None:
    ctx = ExecutionContext(symbol="NVDA")
    broker = _RecordingBroker()

    plan = TradePlan(direction=+1, size=2.0, tp_price=110.0, sl_price=95.0)

    oid = submit_trade_plan(broker, plan, ctx)

    assert oid is not None
    assert broker.last_order is not None
    assert broker.last_order.symbol == "NVDA"
    assert broker.last_order.side is Side.BUY
    assert broker.last_order.quantity == 2.0
