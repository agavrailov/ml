"""Execution helpers for turning strategy decisions into broker orders.

This module sits between the strategy/backtest layer (which produces TradePlan
objects) and the Broker abstraction (which expects OrderRequest instances).

For now it only provides a very small mapping helper for single-symbol equity
trading. It is intentionally simple so that both simulated and IBKR-backed
execution can share the same conversion logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.broker import Broker, BracketOrderIds, BracketOrderRequest, OrderRequest, OrderType, Side
from src.strategy import TradePlan


@dataclass
class ExecutionContext:
    """Static execution context for a given strategy run.

    MVP assumes a single symbol per engine instance (e.g. NVDA). If/when we
    trade multiple symbols, this can be extended or replaced with a richer
    routing mechanism.
    """

    symbol: str


def trade_plan_to_order_request(plan: TradePlan, ctx: ExecutionContext) -> OrderRequest:
    """Convert a TradePlan into a single MARKET OrderRequest.

    The mapping is deliberately conservative:
    - LONG (direction = +1) -> BUY
    - SHORT (direction = -1) -> SELL
    - Quantity is rounded to the nearest whole unit and clipped at 1.
    - Orders are DAY time-in-force by default.

    TP/SL levels remain in the TradePlan; higher-level components (e.g., a
    risk manager or OCO-order implementation) can use them to place linked
    orders if desired. For MVP, we only send the entry order and keep TP/SL
    in our own state/backtest logic.
    """

    if plan.direction not in (+1, -1):
        raise ValueError(f"Unsupported TradePlan.direction={plan.direction!r}; expected +1 or -1")

    side = Side.BUY if plan.direction > 0 else Side.SELL

    # Round to whole units; ensure at least 1 unit since TradePlan already
    # passed min_position_size checks.
    qty = int(round(plan.size))
    if qty <= 0:
        qty = 1

    return OrderRequest(
        symbol=ctx.symbol,
        side=side,
        quantity=float(qty),
        order_type=OrderType.MARKET,
        limit_price=None,
        time_in_force="DAY",
    )


def trade_plan_to_bracket_order(plan: TradePlan, ctx: ExecutionContext) -> BracketOrderRequest:
    """Convert a TradePlan into a BracketOrderRequest (entry + TP + SL).

    This creates a bracket order that includes:
    - Entry order (MARKET, same as trade_plan_to_order_request)
    - Take-profit order (LIMIT at tp_price)
    - Stop-loss order (STOP at sl_price)

    The TP and SL orders are automatically linked to the entry order by the
    broker, ensuring they only activate after entry fills and one cancels
    the other (OCO behavior).
    """

    if plan.direction not in (+1, -1):
        raise ValueError(f"Unsupported TradePlan.direction={plan.direction!r}; expected +1 or -1")

    side = Side.BUY if plan.direction > 0 else Side.SELL

    # Round to whole units; ensure at least 1 unit.
    qty = int(round(plan.size))
    if qty <= 0:
        qty = 1

    return BracketOrderRequest(
        symbol=ctx.symbol,
        side=side,
        quantity=float(qty),
        entry_order_type=OrderType.MARKET,
        entry_limit_price=None,
        tp_price=float(plan.tp_price),
        sl_price=float(plan.sl_price),
        time_in_force="DAY",
    )


def submit_trade_plan(broker: Broker, plan: Optional[TradePlan], ctx: ExecutionContext) -> Optional[str]:
    """Submit a TradePlan via a Broker and return the order id.

    If ``plan`` is None ("no trade"), this is a no-op and returns None.

    NOTE: This only submits the entry order without TP/SL. For bracket orders
    with automatic TP/SL enforcement by the broker, use submit_trade_plan_bracket.
    """

    if plan is None:
        return None

    order = trade_plan_to_order_request(plan, ctx)
    return broker.place_order(order)


def submit_trade_plan_bracket(
    broker: Broker, plan: Optional[TradePlan], ctx: ExecutionContext
) -> Optional[BracketOrderIds]:
    """Submit a TradePlan as a bracket order (entry + TP + SL).

    If ``plan`` is None ("no trade"), this is a no-op and returns None.

    Returns BracketOrderIds with entry_id, tp_id, and sl_id on success.
    """

    if plan is None:
        return None

    bracket = trade_plan_to_bracket_order(plan, ctx)
    return broker.place_bracket_order(bracket)
