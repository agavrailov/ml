"""Tests for src.broker.SimulatedBroker.

These tests validate the minimal contract of the in-memory simulated broker:
- Order id allocation.
- Tracking of open vs cancelled orders.
- Basic position and account summary accessors.
"""
from __future__ import annotations

from src.broker import Broker, OrderRequest, OrderType, Side, SimulatedBroker


def test_simulated_broker_assigns_incrementing_order_ids() -> None:
    broker: Broker = SimulatedBroker()

    order1 = OrderRequest(symbol="NVDA", side=Side.BUY, quantity=1.0, order_type=OrderType.MARKET)
    order2 = OrderRequest(symbol="NVDA", side=Side.BUY, quantity=2.0, order_type=OrderType.MARKET)

    oid1 = broker.place_order(order1)
    oid2 = broker.place_order(order2)

    assert oid1 != oid2
    # IDs should be non-empty strings.
    assert isinstance(oid1, str) and oid1
    assert isinstance(oid2, str) and oid2


def test_simulated_broker_tracks_open_and_cancelled_orders() -> None:
    broker: Broker = SimulatedBroker()

    order = OrderRequest(symbol="NVDA", side=Side.BUY, quantity=1.0, order_type=OrderType.MARKET)
    oid = broker.place_order(order)

    open_orders = broker.get_open_orders()
    assert len(open_orders) == 1
    assert open_orders[0].order_id == oid
    assert open_orders[0].status == "NEW"

    broker.cancel_order(oid)

    open_after_cancel = broker.get_open_orders()
    assert open_after_cancel == []


def test_simulated_broker_positions_and_account_summary_are_safe_defaults() -> None:
    broker: Broker = SimulatedBroker()

    # With no trades, there should be no positions and a minimal account summary
    # dict (currently empty; future fields can be added without breaking this test).
    positions = broker.get_positions()
    summary = broker.get_account_summary()

    assert positions == []
    assert isinstance(summary, dict)
