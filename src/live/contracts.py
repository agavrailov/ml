"""Contracts for live trading events.

Provides stable, typed event schemas for the live trading daemon.
All events are written to JSONL logs for observability and post-hoc analysis.

Event Types:
- RunStartEvent: Session initialization
- RunEndEvent: Session termination
- BarEvent: New OHLC bar received
- DecisionEvent: Strategy decision (TRADE or NO_TRADE)
- OrderSubmittedEvent: Order submitted to broker
- OrderStatusEvent: Order status update
- FillEvent: Partial or full fill
- OrderRejectedEvent: Order submission failed
- BrokerSnapshotEvent: Periodic snapshot of broker state
- BrokerConnectedEvent: Broker connection established
- IbErrorEvent: IB error notification
- NoteEvent: Manual annotation
- ErrorEvent: Internal error/exception
- HeartbeatEvent: Liveness indicator
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.live_log import append_event as _append_event_raw


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunStartEvent:
    """Live session started."""

    run_id: str
    symbol: str
    frequency: str
    tsteps: int
    backend: str
    initial_equity: float
    kill_switch_enabled: bool

    def to_dict(self) -> dict[str, Any]:
        return {"type": "run_start", **asdict(self)}


@dataclass
class RunEndEvent:
    """Live session ended."""

    run_id: str
    symbol: str
    frequency: str
    kill_switch_enabled: bool

    def to_dict(self) -> dict[str, Any]:
        return {"type": "run_end", **asdict(self)}


@dataclass
class BarEvent:
    """New OHLC bar received."""

    run_id: str
    symbol: str
    frequency: str
    bar_time: str | None
    open: float
    high: float
    low: float
    close: float

    def to_dict(self) -> dict[str, Any]:
        return {"type": "bar", **asdict(self)}


@dataclass
class DecisionEvent:
    """Strategy decision made."""

    run_id: str
    symbol: str
    frequency: str
    bar_time: str | None
    close: float
    predicted_price: float
    model_error_sigma: float
    atr: float
    has_open_position_before: bool
    action: str  # "TRADE" or "NO_TRADE"
    direction: int | None = None  # +1 (long), -1 (short), None if NO_TRADE
    size: float | None = None
    tp_price: float | None = None
    sl_price: float | None = None
    blocked_by_kill_switch: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"type": "decision", **asdict(self)}


@dataclass
class OrderSubmittedEvent:
    """Order successfully submitted to broker."""

    run_id: str
    symbol: str
    frequency: str
    bar_time: str | None
    order_id: str
    direction: int
    size: float
    tp_price: float
    sl_price: float

    def to_dict(self) -> dict[str, Any]:
        return {"type": "order_submitted", **asdict(self)}


@dataclass
class OrderStatusEvent:
    """Order status update from broker."""

    run_id: str
    symbol: str
    frequency: str
    bar_time: str | None
    where: str
    order_id: str
    status: str
    filled_quantity: float
    avg_fill_price: float | None
    side: str | None = None
    quantity: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"type": "order_status", **asdict(self)}


@dataclass
class FillEvent:
    """Partial or full fill notification."""

    run_id: str
    symbol: str
    frequency: str
    bar_time: str | None
    where: str
    order_id: str
    fill_quantity: float
    filled_quantity_total: float
    avg_fill_price: float | None
    side: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"type": "fill", **asdict(self)}


@dataclass
class OrderRejectedEvent:
    """Order submission failed."""

    run_id: str
    symbol: str
    frequency: str
    bar_time: str | None
    direction: int
    size: float
    tp_price: float
    sl_price: float
    error: str
    traceback: str

    def to_dict(self) -> dict[str, Any]:
        return {"type": "order_rejected", **asdict(self)}


@dataclass
class BrokerSnapshotEvent:
    """Periodic snapshot of broker state (positions, orders, account)."""

    run_id: str
    symbol: str
    frequency: str
    bar_time: str | None
    where: str
    kill_switch_enabled: bool
    positions: list[dict]
    open_orders: list[dict]
    account_summary: dict

    def to_dict(self) -> dict[str, Any]:
        return {"type": "broker_snapshot", **asdict(self)}


@dataclass
class BrokerConnectedEvent:
    """Broker connection established."""

    run_id: str
    host: str
    port: int
    client_id: int

    def to_dict(self) -> dict[str, Any]:
        return {"type": "broker_connected", **asdict(self)}


@dataclass
class IbErrorEvent:
    """IB error notification."""

    run_id: str
    symbol: str
    frequency: str
    req_id: int | None
    error_code: int | None
    error: str

    def to_dict(self) -> dict[str, Any]:
        return {"type": "ib_error", **asdict(self)}


@dataclass
class NoteEvent:
    """Manual annotation (typically from UI)."""

    run_id: str
    symbol: str
    frequency: str
    source: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"type": "note", **asdict(self)}


@dataclass
class ErrorEvent:
    """Internal error/exception."""

    run_id: str
    where: str
    error: str
    traceback: str

    def to_dict(self) -> dict[str, Any]:
        return {"type": "error", **asdict(self)}


@dataclass
class HeartbeatEvent:
    """Liveness indicator."""

    run_id: str
    symbol: str
    frequency: str
    bar_count: int
    seconds_since_last_update: float
    last_has_new_bar: bool | None
    last_bar_time: str | None

    def to_dict(self) -> dict[str, Any]:
        return {"type": "heartbeat", **asdict(self)}


@dataclass
class DataInitialEvent:
    """Initial historical bars received."""

    run_id: str
    symbol: str
    frequency: str
    n_bars: int
    last_bar_time: str | None

    def to_dict(self) -> dict[str, Any]:
        return {"type": "data_initial", **asdict(self)}


@dataclass
class DataSubscribedEvent:
    """Data subscription established."""

    run_id: str
    bar_size_setting: str
    duration: str
    what_to_show: str
    use_rth: bool

    def to_dict(self) -> dict[str, Any]:
        return {"type": "data_subscribed", **asdict(self)}


# Helper to write any event to log
def write_event(
    log_path: Path,
    event: (
        RunStartEvent
        | RunEndEvent
        | BarEvent
        | DecisionEvent
        | OrderSubmittedEvent
        | OrderStatusEvent
        | FillEvent
        | OrderRejectedEvent
        | BrokerSnapshotEvent
        | BrokerConnectedEvent
        | IbErrorEvent
        | NoteEvent
        | ErrorEvent
        | HeartbeatEvent
        | DataInitialEvent
        | DataSubscribedEvent
    ),
) -> None:
    """Write a typed event to the live log."""
    _append_event_raw(log_path, event.to_dict())
