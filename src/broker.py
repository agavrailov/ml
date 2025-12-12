"""Broker abstractions for trading system integration.

This module introduces a minimal, repo-specific broker interface that can be
used by:
- Offline/simulated execution ("SimulatedBroker").
- IBKR-backed execution ("IBKRBroker", to be implemented later).

The goal is to keep the interface small and explicit so that the strategy and
backtest code can depend on it without pulling in any broker-specific details.

Design is aligned with docs in:
- docs/trading system/trading_system_requirements.md
- docs/trading system/trading_system_hld.md
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Protocol


class Side(str, Enum):
    """Order side.

    We keep this as a simple string enum so it is easy to serialize/log.
    """

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Supported order types for MVP.

    Additional types (STOP, STOP_LIMIT, etc.) can be added later if needed.
    """

    MARKET = "MKT"
    LIMIT = "LMT"


@dataclass
class OrderRequest:
    """Order request submitted by the strategy or execution engine.

    This is intentionally minimal for now and focused on single-instrument
    equity trades (e.g. NVDA). It can be extended with more fields later
    (e.g. exchange, currency, account id) without breaking the basic pattern.
    """

    symbol: str
    side: Side
    quantity: float  # positive number of shares/contracts
    order_type: OrderType
    limit_price: Optional[float] = None
    time_in_force: str = "DAY"  # DAY, GTC, etc.
    idempotency_key: Optional[str] = None  # optional external correlation id


@dataclass
class OrderStatus:
    """Lightweight view of order state.

    Status values are kept as free-form strings to avoid over-engineering;
    callers can treat them as opaque except for simple checks like "FILLED".
    """

    order_id: str
    symbol: str
    side: Side
    quantity: float
    filled_quantity: float
    avg_fill_price: Optional[float]
    status: str  # e.g. NEW, PARTIALLY_FILLED, FILLED, CANCELLED, REJECTED


@dataclass
class PositionInfo:
    """Snapshot of a position as seen by the broker.

    For MVP we assume a single long/short position per symbol.
    """

    symbol: str
    quantity: float  # positive for long, negative for short
    avg_price: float


class Broker(Protocol):
    """Abstract trading interface used by the strategy/execution engine.

    Implementations are expected to be side-effectful (talk to a broker or
    maintain internal state). The interface is deliberately small so that
    unit tests can easily mock it.
    """

    def place_order(self, order: OrderRequest) -> str:
        """Submit an order and return the broker-assigned order id."""

    def cancel_order(self, order_id: str) -> None:
        """Request cancellation of an existing order (best-effort)."""

    def get_all_orders(self) -> List[OrderStatus]:
        """Return a snapshot of all known orders (including filled/cancelled).

        Not all brokers can provide historical orders; for those cases, a best-effort
        subset is acceptable.
        """

    def get_open_orders(self) -> List[OrderStatus]:
        """Return a snapshot of currently open orders."""

    def get_positions(self) -> List[PositionInfo]:
        """Return current positions as seen by the broker."""

    def get_account_summary(self) -> Dict[str, float]:
        """Return simple account metrics (e.g. cash, equity, buying power)."""


class SimulatedBroker:
    """In-memory, single-process broker implementation.

    This is a placeholder for a richer simulated execution model. It is useful
    for tests and for a Phase 1 paper-trading engine that does not talk to a
    real broker yet.

    For now, the implementation only tracks orders and positions in memory and
    does not attempt to model fills; higher-level components may mark orders as
    filled by updating positions directly or by replacing this class with a
    more detailed simulator.
    """

    def __init__(self) -> None:
        self._next_id: int = 1
        self._orders: Dict[str, OrderStatus] = {}
        self._positions: Dict[str, PositionInfo] = {}

    def _allocate_order_id(self) -> str:
        oid = str(self._next_id)
        self._next_id += 1
        return oid

    def place_order(self, order: OrderRequest) -> str:
        order_id = self._allocate_order_id()
        status = OrderStatus(
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            filled_quantity=0.0,
            avg_fill_price=None,
            status="NEW",
        )
        self._orders[order_id] = status
        return order_id

    def cancel_order(self, order_id: str) -> None:
        status = self._orders.get(order_id)
        if status is None:
            return
        if status.status in {"FILLED", "CANCELLED"}:
            return
        status.status = "CANCELLED"

    def get_all_orders(self) -> List[OrderStatus]:
        return list(self._orders.values())

    def get_open_orders(self) -> List[OrderStatus]:
        return [o for o in self._orders.values() if o.status not in {"FILLED", "CANCELLED"}]

    def get_positions(self) -> List[PositionInfo]:
        return list(self._positions.values())

    def get_account_summary(self) -> Dict[str, float]:
        # Placeholder summary: real implementations will compute equity, etc.
        return {}
