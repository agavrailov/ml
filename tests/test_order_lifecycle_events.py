"""Tests for live order lifecycle event derivation.

These tests cover the pure diff logic used by src.ibkr_live_session to emit:
- order_status events when state changes
- fill events when filled_quantity increases
"""

from __future__ import annotations

from dataclasses import dataclass

from src.ibkr_live_session import derive_order_lifecycle_events


@dataclass
class _O:
    order_id: str
    status: str
    quantity: float = 0.0
    filled_quantity: float = 0.0
    avg_fill_price: float | None = None
    side: str = "BUY"


def test_order_status_emitted_on_first_seen_order() -> None:
    prev = {}
    nxt, events = derive_order_lifecycle_events(
        prev,
        [_O(order_id="1", status="Submitted", quantity=5.0, filled_quantity=0.0)],
        run_id="r1",
        symbol="NVDA",
        frequency="60min",
        where="snap",
        bar_time="t",
    )

    assert "1" in nxt
    assert [e["type"] for e in events] == ["order_status"]
    assert events[0]["order_id"] == "1"
    assert events[0]["status"] == "SUBMITTED"


def test_fill_emitted_when_filled_quantity_increases() -> None:
    prev, _ = derive_order_lifecycle_events(
        {},
        [_O(order_id="1", status="Submitted", quantity=5.0, filled_quantity=0.0)],
        run_id="r1",
        symbol="NVDA",
        frequency="60min",
        where="snap1",
        bar_time="t1",
    )

    nxt, events = derive_order_lifecycle_events(
        prev,
        [_O(order_id="1", status="Filled", quantity=5.0, filled_quantity=5.0, avg_fill_price=100.0)],
        run_id="r1",
        symbol="NVDA",
        frequency="60min",
        where="snap2",
        bar_time="t2",
    )

    assert nxt["1"]["status"] == "FILLED"

    types = [e["type"] for e in events]
    assert "order_status" in types
    assert "fill" in types

    fill = [e for e in events if e["type"] == "fill"][0]
    assert fill["order_id"] == "1"
    assert fill["fill_quantity"] == 5.0
    assert fill["filled_quantity_total"] == 5.0
    assert fill["avg_fill_price"] == 100.0


def test_no_events_when_snapshot_is_unchanged() -> None:
    prev, _ = derive_order_lifecycle_events(
        {},
        [_O(order_id="1", status="Submitted", quantity=5.0, filled_quantity=0.0)],
        run_id="r1",
        symbol="NVDA",
        frequency="60min",
        where="snap1",
        bar_time="t1",
    )

    nxt, events = derive_order_lifecycle_events(
        prev,
        [_O(order_id="1", status="Submitted", quantity=5.0, filled_quantity=0.0)],
        run_id="r1",
        symbol="NVDA",
        frequency="60min",
        where="snap2",
        bar_time="t2",
    )

    assert nxt == prev
    assert events == []


def test_partial_fills_emit_multiple_fill_events() -> None:
    prev, _ = derive_order_lifecycle_events(
        {},
        [_O(order_id="1", status="Submitted", quantity=5.0, filled_quantity=0.0)],
        run_id="r1",
        symbol="NVDA",
        frequency="60min",
        where="snap1",
        bar_time="t1",
    )

    prev, events1 = derive_order_lifecycle_events(
        prev,
        [_O(order_id="1", status="Submitted", quantity=5.0, filled_quantity=2.0, avg_fill_price=99.0)],
        run_id="r1",
        symbol="NVDA",
        frequency="60min",
        where="snap2",
        bar_time="t2",
    )
    assert any(e["type"] == "fill" and e["fill_quantity"] == 2.0 for e in events1)

    prev, events2 = derive_order_lifecycle_events(
        prev,
        [_O(order_id="1", status="Filled", quantity=5.0, filled_quantity=5.0, avg_fill_price=100.0)],
        run_id="r1",
        symbol="NVDA",
        frequency="60min",
        where="snap3",
        bar_time="t3",
    )
    assert any(e["type"] == "fill" and e["fill_quantity"] == 3.0 for e in events2)
    assert any(e["type"] == "order_status" and e["status"] == "FILLED" for e in events2)
