"""Unit tests for bracket order checking."""
from __future__ import annotations

from dataclasses import dataclass

from src.ibkr_live_session import check_positions_have_brackets


@dataclass
class MockPosition:
    """Mock position for testing."""

    symbol: str
    quantity: float


@dataclass
class MockOrder:
    """Mock order for testing."""

    order_id: str
    symbol: str
    side: str


@dataclass
class MockIBOrder:
    """Mock IB order with type."""

    orderId: str  # noqa: N815
    orderType: str  # noqa: N815


@dataclass
class MockIBTrade:
    """Mock IB trade."""

    order: MockIBOrder


def test_position_with_complete_brackets():
    """Test that positions with TP and SL orders don't generate warnings."""
    positions = [MockPosition(symbol="NVDA", quantity=10.0)]
    open_orders = [
        MockOrder(order_id="1", symbol="NVDA", side="SELL"),  # TP
        MockOrder(order_id="2", symbol="NVDA", side="SELL"),  # SL
    ]
    ib_trades = [
        MockIBTrade(order=MockIBOrder(orderId="1", orderType="LMT")),
        MockIBTrade(order=MockIBOrder(orderId="2", orderType="STP")),
    ]

    warnings = check_positions_have_brackets(
        positions, open_orders, run_id="test", symbol="NVDA", ib_trades=ib_trades
    )

    assert len(warnings) == 0


def test_position_missing_sl():
    """Test that position with only TP generates warning."""
    positions = [MockPosition(symbol="NVDA", quantity=10.0)]
    open_orders = [
        MockOrder(order_id="1", symbol="NVDA", side="SELL"),  # TP only
    ]
    ib_trades = [
        MockIBTrade(order=MockIBOrder(orderId="1", orderType="LMT")),
    ]

    warnings = check_positions_have_brackets(
        positions, open_orders, run_id="test", symbol="NVDA", ib_trades=ib_trades
    )

    assert len(warnings) == 1
    assert warnings[0]["type"] == "position_missing_brackets"
    assert warnings[0]["symbol"] == "NVDA"
    assert warnings[0]["has_tp"] is True
    assert warnings[0]["has_sl"] is False


def test_position_missing_tp():
    """Test that position with only SL generates warning."""
    positions = [MockPosition(symbol="NVDA", quantity=10.0)]
    open_orders = [
        MockOrder(order_id="2", symbol="NVDA", side="SELL"),  # SL only
    ]
    ib_trades = [
        MockIBTrade(order=MockIBOrder(orderId="2", orderType="STP")),
    ]

    warnings = check_positions_have_brackets(
        positions, open_orders, run_id="test", symbol="NVDA", ib_trades=ib_trades
    )

    assert len(warnings) == 1
    assert warnings[0]["has_tp"] is False
    assert warnings[0]["has_sl"] is True


def test_position_no_orders():
    """Test that position with no orders generates warning."""
    positions = [MockPosition(symbol="NVDA", quantity=10.0)]
    open_orders = []
    ib_trades = []

    warnings = check_positions_have_brackets(
        positions, open_orders, run_id="test", symbol="NVDA", ib_trades=ib_trades
    )

    assert len(warnings) == 1
    assert warnings[0]["has_tp"] is False
    assert warnings[0]["has_sl"] is False
    assert warnings[0]["tp_orders_count"] == 0
    assert warnings[0]["sl_orders_count"] == 0


def test_short_position_with_brackets():
    """Test short position with correct bracket orders (BUY side)."""
    positions = [MockPosition(symbol="NVDA", quantity=-10.0)]
    open_orders = [
        MockOrder(order_id="1", symbol="NVDA", side="BUY"),  # TP
        MockOrder(order_id="2", symbol="NVDA", side="BUY"),  # SL
    ]
    ib_trades = [
        MockIBTrade(order=MockIBOrder(orderId="1", orderType="LMT")),
        MockIBTrade(order=MockIBOrder(orderId="2", orderType="STP")),
    ]

    warnings = check_positions_have_brackets(
        positions, open_orders, run_id="test", symbol="NVDA", ib_trades=ib_trades
    )

    assert len(warnings) == 0


def test_ignores_entry_orders():
    """Test that entry orders (same side as position) are ignored."""
    positions = [MockPosition(symbol="NVDA", quantity=10.0)]
    open_orders = [
        MockOrder(order_id="1", symbol="NVDA", side="BUY"),  # Entry (ignored)
        MockOrder(order_id="2", symbol="NVDA", side="SELL"),  # TP
        MockOrder(order_id="3", symbol="NVDA", side="SELL"),  # SL
    ]
    ib_trades = [
        MockIBTrade(order=MockIBOrder(orderId="1", orderType="MKT")),
        MockIBTrade(order=MockIBOrder(orderId="2", orderType="LMT")),
        MockIBTrade(order=MockIBOrder(orderId="3", orderType="STP")),
    ]

    warnings = check_positions_have_brackets(
        positions, open_orders, run_id="test", symbol="NVDA", ib_trades=ib_trades
    )

    assert len(warnings) == 0


def test_no_positions_no_warnings():
    """Test that no positions means no warnings."""
    positions = []
    open_orders = [MockOrder(order_id="1", symbol="NVDA", side="SELL")]
    ib_trades = []

    warnings = check_positions_have_brackets(
        positions, open_orders, run_id="test", symbol="NVDA", ib_trades=ib_trades
    )

    assert len(warnings) == 0


def test_multiple_positions():
    """Test checking multiple positions at once."""
    positions = [
        MockPosition(symbol="NVDA", quantity=10.0),
        MockPosition(symbol="TSLA", quantity=5.0),
    ]
    # NVDA has complete brackets, TSLA missing SL
    open_orders = [
        MockOrder(order_id="1", symbol="NVDA", side="SELL"),
        MockOrder(order_id="2", symbol="NVDA", side="SELL"),
        MockOrder(order_id="3", symbol="TSLA", side="SELL"),  # TP only
    ]
    ib_trades = [
        MockIBTrade(order=MockIBOrder(orderId="1", orderType="LMT")),
        MockIBTrade(order=MockIBOrder(orderId="2", orderType="STP")),
        MockIBTrade(order=MockIBOrder(orderId="3", orderType="LMT")),
    ]

    warnings = check_positions_have_brackets(
        positions, open_orders, run_id="test", symbol="TEST", ib_trades=ib_trades
    )

    assert len(warnings) == 1
    assert warnings[0]["symbol"] == "TSLA"
    assert warnings[0]["has_tp"] is True
    assert warnings[0]["has_sl"] is False
