"""Unit tests for bracket order execution."""
from __future__ import annotations

import pytest

from src.broker import BracketOrderRequest, OrderType, Side
from src.execution import ExecutionContext, trade_plan_to_bracket_order
from src.strategy import TradePlan


def test_trade_plan_to_bracket_order_long():
    """Test conversion of long TradePlan to BracketOrderRequest."""
    plan = TradePlan(direction=+1, size=10.5, tp_price=460.0, sl_price=440.0)
    ctx = ExecutionContext(symbol="NVDA")

    bracket = trade_plan_to_bracket_order(plan, ctx)

    assert bracket.symbol == "NVDA"
    assert bracket.side == Side.BUY
    assert bracket.quantity == 10.0  # Python's round() uses banker's rounding (round half to even)
    assert bracket.entry_order_type == OrderType.MARKET
    assert bracket.entry_limit_price is None
    assert bracket.tp_price == 460.0
    assert bracket.sl_price == 440.0
    assert bracket.time_in_force == "DAY"


def test_trade_plan_to_bracket_order_short():
    """Test conversion of short TradePlan to BracketOrderRequest."""
    plan = TradePlan(direction=-1, size=5.2, tp_price=430.0, sl_price=460.0)
    ctx = ExecutionContext(symbol="TSLA")

    bracket = trade_plan_to_bracket_order(plan, ctx)

    assert bracket.symbol == "TSLA"
    assert bracket.side == Side.SELL
    assert bracket.quantity == 5.0  # rounded from 5.2
    assert bracket.entry_order_type == OrderType.MARKET
    assert bracket.entry_limit_price is None
    assert bracket.tp_price == 430.0
    assert bracket.sl_price == 460.0


def test_trade_plan_to_bracket_order_minimum_size():
    """Test that size below 1 is clamped to 1."""
    plan = TradePlan(direction=+1, size=0.3, tp_price=460.0, sl_price=440.0)
    ctx = ExecutionContext(symbol="NVDA")

    bracket = trade_plan_to_bracket_order(plan, ctx)

    assert bracket.quantity == 1.0


def test_trade_plan_to_bracket_order_invalid_direction():
    """Test that invalid direction raises ValueError."""
    plan = TradePlan(direction=0, size=10.0, tp_price=460.0, sl_price=440.0)
    ctx = ExecutionContext(symbol="NVDA")

    with pytest.raises(ValueError, match="Unsupported TradePlan.direction"):
        trade_plan_to_bracket_order(plan, ctx)


def test_trade_plan_to_bracket_order_preserves_tp_sl():
    """Test that TP/SL prices are preserved exactly."""
    plan = TradePlan(direction=+1, size=15.0, tp_price=500.123, sl_price=450.789)
    ctx = ExecutionContext(symbol="AAPL")

    bracket = trade_plan_to_bracket_order(plan, ctx)

    assert bracket.tp_price == 500.123
    assert bracket.sl_price == 450.789
