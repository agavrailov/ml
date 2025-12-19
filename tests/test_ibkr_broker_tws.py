"""Tests for src.ibkr_broker.IBKRBrokerTws.

These tests focus on the *mapping logic* from our internal OrderRequest/
PositionInfo structures to/from the ib_insync / TWS API. They do not attempt
live connectivity.

Tests are skipped entirely if ``ib_insync`` is not installed.
"""
from __future__ import annotations

import pytest

ib_insync = pytest.importorskip("ib_insync")  # type: ignore[assignment]

from src.broker import OrderRequest, OrderType, Side
from src.ibkr_broker import IBKRBrokerConfig, IBKRBrokerTws


class _FakeOrderStatus:
    def __init__(self, status: str, filled: float = 0.0, avg_fill_price: float | None = None) -> None:
        self.status = status
        self.filled = filled
        self.avgFillPrice = avg_fill_price


class _FakeContract:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol


class _FakeTrade:
    def __init__(self, contract, order, status: _FakeOrderStatus) -> None:
        self.contract = contract
        self.order = order
        self.orderStatus = status


class _FakeIB:
    """Minimal duck-typed stand-in for ``ib_insync.IB``.

    It records ``placeOrder`` calls and returns fake trades.
    """

    def __init__(self) -> None:
        self.placed = []
        self._trades = []

    # Connectivity API expected by IBKRBrokerTws
    def connect(self, host: str, port: int, clientId: int) -> None:  # noqa: N803
        self._host = host
        self._port = port
        self._clientId = clientId

    def isConnected(self) -> bool:  # noqa: N802
        return True

    def disconnect(self) -> None:
        return None

    # Trading API used by IBKRBrokerTws
    def placeOrder(self, contract, order):  # noqa: N802
        # Mimic TWS behavior: attach an orderId to the order.
        order_id = len(self.placed) + 1
        order.orderId = order_id
        trade = _FakeTrade(contract, order, _FakeOrderStatus(status="Submitted"))
        self.placed.append((contract, order))
        self._trades.append(trade)
        return trade

    def trades(self):
        return list(self._trades)

    def positions(self):
        class P:
            def __init__(self, symbol: str, position: float, avg_cost: float) -> None:
                self.contract = _FakeContract(symbol)
                self.position = position
                self.avgCost = avg_cost

        return [P("NVDA", 10.0, 100.0)]

    def accountSummary(self):
        class S:
            def __init__(self, tag: str, value: str) -> None:
                self.tag = tag
                self.value = value

        return [S("NetLiquidation", "12345.67"), S("BuyingPower", "9999")]


def test_place_order_maps_to_ib_order_and_returns_order_id():
    fake_ib = _FakeIB()
    cfg = IBKRBrokerConfig(host="127.0.0.1", port=7497, client_id=1)
    broker = IBKRBrokerTws(config=cfg, ib=fake_ib)

    req = OrderRequest(
        symbol="NVDA",
        side=Side.BUY,
        quantity=5.0,
        order_type=OrderType.MARKET,
    )

    oid = broker.place_order(req)

    assert oid == "1"  # first order id assigned by FakeIB
    assert len(fake_ib.placed) == 1
    contract, order = fake_ib.placed[0]

    assert contract.symbol == "NVDA"
    assert order.action == "BUY"
    assert float(order.totalQuantity) == 5.0

    # get_all_orders should include submitted orders.
    all_orders = broker.get_all_orders()
    assert len(all_orders) == 1
    assert all_orders[0].order_id == "1"
    assert all_orders[0].status == "Submitted"

    # open orders should include it as well.
    open_orders = broker.get_open_orders()
    assert len(open_orders) == 1
    assert open_orders[0].order_id == "1"


def test_place_order_sets_account_when_configured():
    fake_ib = _FakeIB()
    cfg = IBKRBrokerConfig(host="127.0.0.1", port=7497, client_id=1, account="U16442949")
    broker = IBKRBrokerTws(config=cfg, ib=fake_ib)

    req = OrderRequest(
        symbol="NVDA",
        side=Side.BUY,
        quantity=1.0,
        order_type=OrderType.MARKET,
    )

    broker.place_order(req)

    assert len(fake_ib.placed) == 1
    _contract, order = fake_ib.placed[0]
    assert getattr(order, "account", None) == "U16442949"


def test_get_positions_maps_ib_positions_to_position_info():
    fake_ib = _FakeIB()
    cfg = IBKRBrokerConfig(host="127.0.0.1", port=7497, client_id=1)
    broker = IBKRBrokerTws(config=cfg, ib=fake_ib)

    positions = broker.get_positions()

    assert len(positions) == 1
    pos = positions[0]
    assert pos.symbol == "NVDA"
    assert pos.quantity == 10.0
    assert pos.avg_price == 100.0


def test_get_account_summary_maps_tags_to_dict():
    fake_ib = _FakeIB()
    cfg = IBKRBrokerConfig(host="127.0.0.1", port=7497, client_id=1)
    broker = IBKRBrokerTws(config=cfg, ib=fake_ib)

    summary = broker.get_account_summary()

    assert summary["NetLiquidation"] == pytest.approx(12345.67)
    assert summary["BuyingPower"] == pytest.approx(9999.0)
