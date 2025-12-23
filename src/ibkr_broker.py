"""Interactive Brokers (IBKR) broker implementation using the TWS API.

This module provides an ``IBKRBrokerTws`` class that implements the generic
``Broker`` interface defined in ``src.broker`` using the IBKR TWS API via
``ib_insync``.

Design goals:
- Keep the interface small and aligned with ``Broker``.
- Make ``ib_insync`` an optional dependency: importing this module must *not*
  fail if the library is missing; instead, attempting to construct a broker
  without ``ib_insync`` installed will raise a clear RuntimeError.
- Allow injecting a pre-configured ``IB`` instance for tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from src.broker import (
    Broker,
    BracketOrderIds,
    BracketOrderRequest,
    OrderRequest,
    OrderStatus,
    OrderType,
    PositionInfo,
    Side,
)
from src.config import IB as IbConfig

try:  # Optional dependency guard
    from ib_insync import IB, Stock, MarketOrder, LimitOrder, Order, Trade  # type: ignore[import]

    _HAVE_IB_INSYNC = True
except Exception:  # pragma: no cover - environment without ib_insync
    IB = object  # type: ignore[assignment]
    Stock = object  # type: ignore[assignment]
    MarketOrder = object  # type: ignore[assignment]
    LimitOrder = object  # type: ignore[assignment]
    Order = object  # type: ignore[assignment]
    Trade = object  # type: ignore[assignment]
    _HAVE_IB_INSYNC = False


@dataclass
class IBKRBrokerConfig:
    """Configuration for ``IBKRBrokerTws``.

    Defaults come from ``config.IB`` but can be overridden.
    """

    host: str
    port: int
    client_id: int
    account: Optional[str] = None  # optional explicit account id (for multi-account setups)

    @classmethod
    def from_global_config(cls) -> "IBKRBrokerConfig":
        ib_cfg = IbConfig
        return cls(host=ib_cfg.host, port=ib_cfg.port, client_id=ib_cfg.client_id, account=ib_cfg.account)


class IBKRBrokerTws(Broker):
    """IBKR broker implementation backed by the TWS API via ``ib_insync``.

    This class is intentionally minimal and focused on the subset of features
    required for the MVP trading engine:
    - Stock trading (e.g. NVDA) on SMART/US exchanges.
    - Market and limit orders with DAY time-in-force.
    - Read-only access to positions and a simple account summary.

    The class can either manage its own ``IB`` connection or reuse an injected
    instance (useful for tests).
    """

    def __init__(
        self,
        config: Optional[IBKRBrokerConfig] = None,
        ib: Optional[IB] = None,
    ) -> None:
        if ib is None and not _HAVE_IB_INSYNC:
            raise RuntimeError(
                "ib_insync is required to use IBKRBrokerTws; install it with 'pip install ib-insync'",
            )

        self._config = config or IBKRBrokerConfig.from_global_config()
        self._ib: IB = ib if ib is not None else IB()  # type: ignore[assignment]
        self._owns_ib = ib is None

        # For safety we do not auto-connect in __init__; callers should
        # explicitly call ``connect`` or rely on the lazy `_ensure_connected`.

    # -------------
    # Connectivity
    # -------------

    def connect(self) -> None:
        """Establish a connection to TWS/IB Gateway if not already connected."""

        # ``isConnected`` attribute is provided by ib_insync.IB; for tests a
        # duck-typed object may omit it, in which case we attempt to connect
        # unconditionally.
        is_connected = getattr(self._ib, "isConnected", None)
        if callable(is_connected):
            try:
                if is_connected():  # type: ignore[func-returns-value]
                    return
            except Exception:  # pragma: no cover - defensive
                pass

        self._ib.connect(self._config.host, self._config.port, clientId=self._config.client_id)

    def disconnect(self) -> None:
        """Disconnect if we own the IB instance."""

        if not self._owns_ib:
            return
        try:
            disconnect = getattr(self._ib, "disconnect", None)
            if callable(disconnect):
                disconnect()
        except Exception:  # pragma: no cover - defensive
            pass

    def _ensure_connected(self) -> None:
        self.connect()

    # -------------
    # Helpers
    # -------------

    def _make_stock_contract(self, symbol: str):
        """Create a basic SMART/NYSE/NASDAQ stock contract.

        For MVP we hard-code SMART/"USD" routing; this can be extended or made
        configurable later.
        """

        return Stock(symbol, "SMART", "USD")  # type: ignore[call-arg]

    def _make_ib_order(self, order: OrderRequest):
        action = "BUY" if order.side is Side.BUY else "SELL"
        qty = float(order.quantity)

        if order.order_type is OrderType.MARKET:
            ib_order = MarketOrder(action, qty)  # type: ignore[call-arg]
        elif order.order_type is OrderType.LIMIT:
            if order.limit_price is None:
                raise ValueError("Limit order requires limit_price")
            ib_order = LimitOrder(action, qty, float(order.limit_price))  # type: ignore[call-arg]
        else:  # pragma: no cover - unreachable with current enum
            raise ValueError(f"Unsupported OrderType: {order.order_type!r}")

        # IBKR error 435 ("You must specify an account") happens in multi-account
        # setups when the Order.account field is missing. If the user configured an
        # explicit account (e.g. via IBKR_ACCOUNT env var), always set it.
        if self._config.account:
            setattr(ib_order, "account", self._config.account)

        return ib_order

    # -------------
    # Broker interface
    # -------------

    def place_order(self, order: OrderRequest) -> str:
        self._ensure_connected()

        contract = self._make_stock_contract(order.symbol)
        ib_order = self._make_ib_order(order)

        trade: Trade = self._ib.placeOrder(contract, ib_order)  # type: ignore[assignment]

        # ib_insync attaches the orderId to the underlying ``Order`` object.
        order_id = getattr(trade.order, "orderId", None)
        if order_id is None:
            # Fallback: some IBAPI flows use ``trade.orderId`` directly.
            order_id = getattr(trade, "orderId", None)

        return str(order_id) if order_id is not None else "unknown"

    def place_bracket_order(self, bracket: BracketOrderRequest) -> BracketOrderIds:
        """Submit a bracket order (entry + TP + SL) using IBKR's parent-child linking.

        The implementation follows the standard IBKR bracket pattern:
        1. Entry order (parent) with transmit=False
        2. TP order (child) with transmit=False, parentId=entry.orderId
        3. SL order (child) with transmit=True, parentId=entry.orderId

        Setting transmit=True only on the last order sends all three atomically.
        IBKR enforces OCO behavior: when one child fills, the other is cancelled.
        """
        self._ensure_connected()

        contract = self._make_stock_contract(bracket.symbol)

        # Entry order (parent)
        entry_action = "BUY" if bracket.side is Side.BUY else "SELL"
        if bracket.entry_order_type is OrderType.MARKET:
            entry_order = MarketOrder(entry_action, float(bracket.quantity))  # type: ignore[call-arg]
        elif bracket.entry_order_type is OrderType.LIMIT:
            if bracket.entry_limit_price is None:
                raise ValueError("Limit entry order requires entry_limit_price")
            entry_order = LimitOrder(  # type: ignore[call-arg]
                entry_action, float(bracket.quantity), float(bracket.entry_limit_price)
            )
        else:
            raise ValueError(f"Unsupported entry order type: {bracket.entry_order_type!r}")

        # Apply account if configured (multi-account setups)
        if self._config.account:
            setattr(entry_order, "account", self._config.account)

        # Do not transmit entry yet; we need to attach children
        setattr(entry_order, "transmit", False)

        # Submit entry to get orderId
        entry_trade: Trade = self._ib.placeOrder(contract, entry_order)  # type: ignore[assignment]
        entry_id = getattr(entry_trade.order, "orderId", None)
        if entry_id is None:
            raise RuntimeError("Failed to get orderId for entry order")

        # TP order (child): opposite side, LIMIT at tp_price
        tp_action = "SELL" if bracket.side is Side.BUY else "BUY"
        tp_order = LimitOrder(tp_action, float(bracket.quantity), float(bracket.tp_price))  # type: ignore[call-arg]
        if self._config.account:
            setattr(tp_order, "account", self._config.account)
        setattr(tp_order, "parentId", int(entry_id))
        setattr(tp_order, "transmit", False)

        tp_trade: Trade = self._ib.placeOrder(contract, tp_order)  # type: ignore[assignment]
        tp_id = getattr(tp_trade.order, "orderId", None)
        if tp_id is None:
            raise RuntimeError("Failed to get orderId for TP order")

        # SL order (child): opposite side, STOP at sl_price
        # For IBKR, we use Order() directly with orderType="STP" and auxPrice
        sl_order = Order()  # type: ignore[call-arg]
        setattr(sl_order, "action", tp_action)  # same as TP
        setattr(sl_order, "orderType", "STP")  # stop order
        setattr(sl_order, "totalQuantity", float(bracket.quantity))
        setattr(sl_order, "auxPrice", float(bracket.sl_price))  # stop trigger price
        if self._config.account:
            setattr(sl_order, "account", self._config.account)
        setattr(sl_order, "parentId", int(entry_id))
        setattr(sl_order, "transmit", True)  # transmit all three now

        sl_trade: Trade = self._ib.placeOrder(contract, sl_order)  # type: ignore[assignment]
        sl_id = getattr(sl_trade.order, "orderId", None)
        if sl_id is None:
            raise RuntimeError("Failed to get orderId for SL order")

        return BracketOrderIds(entry_id=str(entry_id), tp_id=str(tp_id), sl_id=str(sl_id))

    def cancel_order(self, order_id: str) -> None:
        self._ensure_connected()

        # Best-effort: find the matching trade and cancel its order.
        trades = getattr(self._ib, "trades", None)
        if not callable(trades):  # pragma: no cover - defensive
            return

        for t in trades():
            tid = getattr(getattr(t, "order", t), "orderId", None)
            if tid is not None and str(tid) == str(order_id):
                cancel_fn = getattr(self._ib, "cancelOrder", None)
                if callable(cancel_fn):
                    cancel_fn(t.order)
                break

    def get_all_orders(self) -> List[OrderStatus]:
        self._ensure_connected()

        result: List[OrderStatus] = []
        trades = getattr(self._ib, "trades", None)
        if not callable(trades):
            return result

        for t in trades():
            order = getattr(t, "order", None)
            status_obj = getattr(t, "orderStatus", None)
            if order is None or status_obj is None:
                continue

            status = getattr(status_obj, "status", "")

            side = Side.BUY if getattr(order, "action", "BUY") == "BUY" else Side.SELL
            qty = float(getattr(order, "totalQuantity", 0.0))
            filled_qty = float(getattr(status_obj, "filled", 0.0))
            avg_fill_price = getattr(status_obj, "avgFillPrice", None)
            oid = getattr(order, "orderId", None)

            result.append(
                OrderStatus(
                    order_id=str(oid) if oid is not None else "unknown",
                    symbol=getattr(getattr(t, "contract", None), "symbol", ""),
                    side=side,
                    quantity=qty,
                    filled_quantity=filled_qty,
                    avg_fill_price=float(avg_fill_price) if avg_fill_price is not None else None,
                    status=str(status),
                ),
            )

        return result

    def get_open_orders(self) -> List[OrderStatus]:
        orders = self.get_all_orders()
        return [o for o in orders if o.status not in {"Filled", "Cancelled"}]

    def get_positions(self) -> List[PositionInfo]:
        self._ensure_connected()

        positions_fn = getattr(self._ib, "positions", None)
        if not callable(positions_fn):
            return []

        infos: List[PositionInfo] = []
        for p in positions_fn():
            contract = getattr(p, "contract", None)
            symbol = getattr(contract, "symbol", "") if contract is not None else ""
            qty = float(getattr(p, "position", 0.0))
            avg_price = float(getattr(p, "avgCost", 0.0))
            infos.append(PositionInfo(symbol=symbol, quantity=qty, avg_price=avg_price))

        return infos

    def get_account_summary(self) -> Dict[str, float]:
        self._ensure_connected()

        summary_fn = getattr(self._ib, "accountSummary", None)
        if not callable(summary_fn):
            return {}

        summary_items = summary_fn()
        result: Dict[str, float] = {}

        for item in summary_items:
            tag = getattr(item, "tag", None)
            val = getattr(item, "value", None)
            if not tag:
                continue
            try:
                result[str(tag)] = float(val)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                continue

        return result

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.disconnect()
        except Exception:
            pass
