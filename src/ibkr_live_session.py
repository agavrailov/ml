"""IBKR/TWS live session runner (CLI shim).

This module provides a CLI interface to the live trading engine.
It parses command-line arguments and delegates to src.live.engine.

The core logic is in src.live.engine, which:
- Connects to IBKR TWS / IB Gateway via ib_insync
- Subscribes to a keepUpToDate historical bar stream
- For each new completed bar, calls LivePredictor.update_and_predict()
- Feeds the result into the strategy/execution pipeline
- Writes typed events to JSONL log (see src.live.contracts)

Observability:
- Writes an append-only JSONL event log under ui_state/live/ by default
- Supports a file-based kill switch (ui_state/live/KILL_SWITCH) that disables
  new order submissions without stopping the daemon

Usage example:
    python -m src.ibkr_live_session --symbol NVDA --frequency 60min --backend IBKR_TWS

Note: requires ib_insync installed and a running TWS/IB Gateway.
"""

from __future__ import annotations

import argparse
import os
import threading
import time
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from src.config import IB as IbConfig
from src.execution import ExecutionContext, submit_trade_plan_bracket
from src.live.alerts import AlertManager, AlertSeverity
from src.live.persistence import PersistentBarTracker, compute_bar_hash
from src.live.reconnect import ConnectionManager, ReconnectConfig, ReconnectController
from src.live.state import StateMachine, SystemState
from src.live.status import StatusManager
from src.live_log import (
    append_event,
    create_run_id,
    is_kill_switch_enabled,
    make_log_path,
)
from src.utils.timestamped_print import ts_print


def _canonical_order_status(status: object) -> str:
    s = str(status or "").strip()
    low = s.lower()
    if "reject" in low:
        return "REJECTED"
    if "cancel" in low:
        return "CANCELLED"
    if "fill" in low:
        return "FILLED"
    if "submit" in low:
        return "SUBMITTED"
    if not s:
        return "UNKNOWN"
    return s.upper()


def check_positions_have_brackets(
    positions: list[object],
    open_orders: list[object],
    *,
    run_id: str,
    symbol: str,
    ib_trades: list[object] | None = None,
) -> list[dict[str, object]]:
    """Check if all positions have associated TP/SL bracket orders.

    Returns a list of warning events for positions that lack bracket orders.

    A position is considered to have brackets if there are exit orders for both
    TP (LIMIT) and SL (STOP) on the opposite side of the position.

    Args:
        positions: List of PositionInfo objects from broker
        open_orders: List of OrderStatus objects from broker
        run_id: Current run ID for logging
        symbol: Symbol to check (for logging)
        ib_trades: Optional list of raw IB trades (for more detailed order type info)
    """
    warnings: list[dict[str, object]] = []

    if not positions:
        return warnings

    # Build order type map from IB trades if available
    order_types: dict[str, str] = {}
    if ib_trades:
        for trade in ib_trades:
            order = getattr(trade, "order", None)
            if order:
                order_id = str(getattr(order, "orderId", ""))
                order_type = str(getattr(order, "orderType", "")).upper()
                if order_id and order_type:
                    order_types[order_id] = order_type

    # Count open orders per symbol and categorize by type
    orders_by_symbol: dict[str, dict[str, list[object]]] = {}
    for order in open_orders:
        order_symbol = str(getattr(order, "symbol", ""))
        if not order_symbol:
            continue

        order_id = str(getattr(order, "order_id", ""))
        side = str(getattr(order, "side", "")).upper()

        # Get order type from IB trades if available, otherwise use heuristic
        if order_id in order_types:
            otype = order_types[order_id]
        else:
            # Fallback: can't reliably determine without IB trades
            otype = "UNKNOWN"

        if order_symbol not in orders_by_symbol:
            orders_by_symbol[order_symbol] = {"LIMIT": [], "STOP": [], "OTHER": []}

        if "LMT" in otype or "LIMIT" in otype:
            orders_by_symbol[order_symbol]["LIMIT"].append(order)
        elif "STP" in otype or "STOP" in otype:
            orders_by_symbol[order_symbol]["STOP"].append(order)
        else:
            orders_by_symbol[order_symbol]["OTHER"].append(order)

    # Check each position
    for pos in positions:
        pos_symbol = str(getattr(pos, "symbol", ""))
        pos_qty = float(getattr(pos, "quantity", 0.0))

        if not pos_symbol or abs(pos_qty) < 0.01:
            continue

        # Get orders for this symbol
        symbol_order_groups = orders_by_symbol.get(pos_symbol, {"LIMIT": [], "STOP": [], "OTHER": []})

        # Look for exit orders (opposite side of position)
        exit_limit_orders = []
        exit_stop_orders = []

        for order in symbol_order_groups["LIMIT"]:
            side = str(getattr(order, "side", "")).upper()
            is_exit = (pos_qty > 0 and side == "SELL") or (pos_qty < 0 and side == "BUY")
            if is_exit:
                exit_limit_orders.append(order)

        for order in symbol_order_groups["STOP"]:
            side = str(getattr(order, "side", "")).upper()
            is_exit = (pos_qty > 0 and side == "SELL") or (pos_qty < 0 and side == "BUY")
            if is_exit:
                exit_stop_orders.append(order)

        has_tp = len(exit_limit_orders) > 0
        has_sl = len(exit_stop_orders) > 0

        # If position lacks both TP and SL, warn
        if not (has_tp and has_sl):
            warnings.append(
                {
                    "type": "position_missing_brackets",
                    "run_id": run_id,
                    "symbol": pos_symbol,
                    "quantity": pos_qty,
                    "has_tp": has_tp,
                    "has_sl": has_sl,
                    "tp_orders_count": len(exit_limit_orders),
                    "sl_orders_count": len(exit_stop_orders),
                    "total_open_orders": sum(len(v) for v in symbol_order_groups.values()),
                }
            )

    return warnings


def derive_order_lifecycle_events(
    prev_state: dict[str, dict[str, object]],
    orders: list[object],
    *,
    run_id: str,
    symbol: str,
    frequency: str,
    where: str,
    bar_time: str | None,
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    """Derive order_status/fill events by diffing broker order snapshots.

    Minimal by design:
    - Emit an `order_status` event whenever (status, filled_qty, avg_fill_price) changes.
    - Emit a `fill` event whenever filled_quantity increases.
    """

    next_state: dict[str, dict[str, object]] = {}
    out: list[dict[str, object]] = []

    for o in orders:
        oid = str(getattr(o, "order_id", "") or "")
        if not oid:
            continue

        status_raw = getattr(o, "status", None)
        status = _canonical_order_status(status_raw)
        filled_qty = float(getattr(o, "filled_quantity", 0.0) or 0.0)
        avg_fill_price = getattr(o, "avg_fill_price", None)
        try:
            avg_fill_price_f = float(avg_fill_price) if avg_fill_price is not None else None
        except Exception:
            avg_fill_price_f = None

        prev = prev_state.get(oid) or {}
        prev_status = str(prev.get("status", ""))
        prev_filled = float(prev.get("filled_quantity", 0.0) or 0.0)

        next_state[oid] = {
            "status": status,
            "filled_quantity": filled_qty,
            "avg_fill_price": avg_fill_price_f,
        }

        if status != prev_status or abs(filled_qty - prev_filled) > 1e-9 or (avg_fill_price_f != prev.get("avg_fill_price")):
            out.append(
                {
                    "type": "order_status",
                    "run_id": run_id,
                    "symbol": symbol,
                    "frequency": frequency,
                    "bar_time": bar_time,
                    "where": where,
                    "order_id": oid,
                    "status": status,
                    "filled_quantity": filled_qty,
                    "avg_fill_price": avg_fill_price_f,
                    # Optional passthrough metadata if present on the OrderStatus object.
                    "side": str(getattr(o, "side", "")),
                    "quantity": float(getattr(o, "quantity", 0.0) or 0.0),
                }
            )

        if filled_qty > prev_filled + 1e-9:
            out.append(
                {
                    "type": "fill",
                    "run_id": run_id,
                    "symbol": symbol,
                    "frequency": frequency,
                    "bar_time": bar_time,
                    "where": where,
                    "order_id": oid,
                    "fill_quantity": float(filled_qty - prev_filled),
                    "filled_quantity_total": filled_qty,
                    "avg_fill_price": avg_fill_price_f,
                    "side": str(getattr(o, "side", "")),
                }
            )

    return next_state, out
from src.live_predictor import LivePredictor, LivePredictorConfig
from src.strategy import StrategyState, compute_tp_sl_and_size
from src.trading_session import make_strategy_config_from_defaults, make_broker

try:  # optional dependency
    from ib_insync import IB, Stock  # type: ignore[import]

    _HAVE_IB = True
except Exception:  # pragma: no cover
    IB = object  # type: ignore[assignment]
    Stock = object  # type: ignore[assignment]
    _HAVE_IB = False


@dataclass
class LiveSessionConfig:
    symbol: str = "NVDA"
    frequency: str = "60min"  # must match trained model frequency
    tsteps: int = 5
    backend: str = "IBKR_TWS"
    initial_equity: float = 10_000.0
    stop_after_first_trade: bool = False

    # Optional override for IBKR/TWS clientId. If None, uses src.config.IB.client_id.
    client_id: int | None = None

    # Observability / UX wiring
    run_id: str | None = None
    log_to_disk: bool = True
    # When None, defaults to repo_root/ui_state/live.
    log_dir: str | None = None

    # Broker snapshot cadence (Step 2). Use 1 for every bar, larger to reduce noise.
    snapshot_every_n_bars: int = 1


def _frequency_to_bar_size_setting(freq: str) -> str:
    # Normalize common spellings used across the repo (e.g. "5min") and tolerate
    # whitespace (e.g. "5 min").
    f = freq.lower().strip().replace(" ", "")

    if f in {"1min", "1m"}:
        return "1 min"
    if f in {"5min", "5m"}:
        return "5 mins"
    if f in {"15min", "15m"}:
        return "15 mins"
    if f in {"30min", "30m"}:
        return "30 mins"
    if f in {"60min", "60m", "1h", "1hr", "1hour"}:
        return "1 hour"
    if f in {"240min", "240m", "4h", "4hr", "4hour"}:
        return "4 hours"
    raise ValueError(f"Unsupported frequency for live bars: {freq!r}")


def _frequency_to_minutes(freq: str) -> int | None:
    f = str(freq or "").lower().strip().replace(" ", "")
    if f.endswith("min"):
        n = f[: -len("min")]
        return int(n) if n.isdigit() else None
    if f in {"1h", "1hr", "1hour"}:
        return 60
    if f in {"4h", "4hr", "4hour"}:
        return 240
    return None


def _no_update_warning_threshold_seconds(freq: str) -> float:
    """Return how long to wait before warning about *no updateEvent callbacks*.

    keepUpToDate updates are expected at least once per bar (often more frequently),
    but for larger bar sizes it's normal to see long quiet periods.

    After reconnection, bars should arrive quickly if HMDS is active. We cap the
    threshold at 10 minutes to detect HMDS inactivity faster.
    """
    base = 300.0  # 5 minutes minimum
    mins = _frequency_to_minutes(freq)
    if mins is None or mins <= 0:
        return base
    # Use 0.5x bar period, capped at 10 minutes
    bar_half_period = float(mins * 30)  # 30 seconds per minute (mins * 60 / 2)
    return min(600.0, max(base, bar_half_period))


def _is_market_hours(*, premarket: bool = True) -> bool:
    """Check if current time is within US market trading hours (EST/EDT).

    Args:
        premarket: If True, includes pre-market hours (4:00 AM - 9:30 AM EST).
                  Regular market: 9:30 AM - 4:00 PM EST.

    Returns True during trading hours (Mon-Fri only), False otherwise.
    """
    try:
        from datetime import datetime, timezone
        import zoneinfo
    except ImportError:
        # Fallback if zoneinfo not available (Python < 3.9)
        try:
            from datetime import datetime, timezone
            from backports.zoneinfo import ZoneInfo as zoneinfo  # type: ignore[import]
        except ImportError:
            # If we can't determine timezone, conservatively assume market hours
            return True

    try:
        # Get current time in US Eastern timezone
        eastern = zoneinfo.ZoneInfo("America/New_York")
        now = datetime.now(timezone.utc).astimezone(eastern)

        # Skip weekends
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        # Get time as minutes since midnight
        current_minutes = now.hour * 60 + now.minute

        if premarket:
            # Pre-market: 4:00 AM - 9:30 AM, Regular: 9:30 AM - 4:00 PM
            premarket_start = 4 * 60  # 4:00 AM
            market_close = 16 * 60  # 4:00 PM
            return premarket_start <= current_minutes < market_close
        else:
            # Regular market hours only: 9:30 AM - 4:00 PM
            market_open = 9 * 60 + 30  # 9:30 AM
            market_close = 16 * 60  # 4:00 PM
            return market_open <= current_minutes < market_close

    except Exception:
        # If timezone detection fails, conservatively assume market hours
        return True


def _is_trading_day(date_to_check: datetime | None = None) -> bool:
    """Check if a given date is a trading day using market calendar.

    Uses pandas_market_calendars to check against official exchange holidays.
    Falls back to simple weekday check if calendar not available.

    Args:
        date_to_check: Date to check. If None, uses current date in US Eastern time.

    Returns:
        True if the date is a valid trading day, False otherwise.
    """
    try:
        from datetime import datetime, timezone
        import zoneinfo
    except ImportError:
        try:
            from datetime import datetime, timezone
            from backports.zoneinfo import ZoneInfo as zoneinfo  # type: ignore[import]
        except ImportError:
            # Conservative fallback: assume it's a trading day
            return True

    try:
        if date_to_check is None:
            eastern = zoneinfo.ZoneInfo("America/New_York")
            date_to_check = datetime.now(timezone.utc).astimezone(eastern)

        # Quick weekend check
        if date_to_check.weekday() >= 5:
            return False

        # Try to use market calendar for holiday awareness
        try:
            import pandas as pd
            import pandas_market_calendars as mcal  # type: ignore[import]

            cal = mcal.get_calendar("NASDAQ")
            check_date = pd.Timestamp(date_to_check.date())
            schedule = cal.schedule(start_date=check_date, end_date=check_date)
            return not schedule.empty
        except Exception:
            # Fallback: if calendar fails, just check weekday (already done above)
            return True

    except Exception:
        # Conservative fallback
        return True


def _connect_with_unique_client_id(
    ib,  # noqa: ANN001
    *,
    host: str,
    port: int,
    preferred_client_id: int,
    max_tries: int = 25,
) -> int:
    """Connect to TWS/IB Gateway, retrying with different clientIds if needed.

    IB enforces unique clientIds per (host, port). There's no reliable "check"
    call; the pragmatic way is to attempt a connection and, on the
    "client id is already in use" failure, retry with a new id.

    Returns the clientId that successfully connected.
    """

    last_exc: Exception | None = None

    for offset in range(max_tries):
        cid = int(preferred_client_id) + offset
        try:
            # Capture async errors raised immediately after connect (notably error 326).
            connect_errors: list[tuple[int, str]] = []

            def _on_error(reqId, errorCode, errorString, contract=None):  # noqa: ANN001,N803
                try:
                    connect_errors.append((int(errorCode), str(errorString)))
                except Exception:
                    connect_errors.append((-1, str(errorString)))

            err_event = getattr(ib, "errorEvent", None)
            if err_event is not None:
                try:
                    err_event += _on_error
                except Exception:
                    err_event = None

            ib.connect(host, port, clientId=cid)

            # Give IB a moment to deliver immediate rejection errors (e.g. 326).
            # Always use time.sleep() instead of ib.sleep() to avoid event loop issues
            # when called from background threads.
            time.sleep(0.35)

            if any(code == 326 for code, _ in connect_errors):
                raise TimeoutError("Unable to connect as the client id is already in use")

            # Some failure modes don't raise from connect() immediately; they show
            # up as error events and the socket gets closed. Probe the connection
            # so we only treat it as "connected" if it's actually usable.
            is_connected = getattr(ib, "isConnected", None)
            if callable(is_connected) and not is_connected():
                raise TimeoutError("connect() returned but IB reports not connected")

            probe = getattr(ib, "reqCurrentTime", None)
            if callable(probe):
                probe()  # raises on timeout / closed socket

            return cid
        except Exception as exc:  # noqa: BLE001
            last_exc = exc

            # Best-effort cleanup before retrying.
            try:
                disc = getattr(ib, "disconnect", None)
                if callable(disc):
                    disc()
            except Exception:
                pass

            # Remove error handler if we attached one.
            try:
                err_event = getattr(ib, "errorEvent", None)
                if err_event is not None and "_on_error" in locals():
                    err_event -= _on_error
            except Exception:
                pass

            msg = str(exc).lower()
            # ib_insync often raises TimeoutError() after TWS rejects the clientId.
            # The underlying TWS error (326) is printed on stderr, so we treat a
            # timeout as potentially "clientId in use" and keep trying.
            if "client id" in msg and "already" in msg and "use" in msg:
                continue
            if isinstance(exc, TimeoutError):
                # Common case when TWS rejects connect for a duplicate clientId.
                # Retry with a new id.
                continue

            # Unknown connect failure - stop early.
            raise

    raise RuntimeError(
        f"Unable to connect to TWS at {host}:{port} with a free clientId; "
        f"tried {preferred_client_id}..{preferred_client_id + max_tries - 1}. "
        f"Last error: {last_exc!r}",
    )


def _activate_hmds(
    ib,  # noqa: ANN001
    contract,  # noqa: ANN001
    bar_size_setting: str,
    log_fn: Callable[[dict], None],
    run_id: str,
    symbol: str,
) -> bool:
    """Force HMDS data farm activation with one-shot historical data request.

    After reconnection or sleep/wake, IBKR's HMDS farm may go inactive (error 2107).
    This causes keepUpToDate subscriptions to stop delivering bars until HMDS activates.
    
    This function triggers HMDS activation by requesting a small amount of historical data.

    Args:
        ib: ib_insync.IB instance
        contract: Qualified contract for the symbol
        bar_size_setting: Bar size (e.g., "4 hours")
        log_fn: Logging function
        run_id: Current run ID
        symbol: Symbol name for logging

    Returns:
        True if activation succeeded, False otherwise
    """
    try:
        # Request 1 day of data to wake up HMDS
        wake_bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr="1 D",
            barSizeSetting=bar_size_setting,
            whatToShow="TRADES",
            useRTH=False,
            formatDate=1,
            keepUpToDate=False,  # One-shot request
        )
        
        log_fn({
            "type": "hmds_activation_request",
            "run_id": run_id,
            "symbol": symbol,
            "bars_received": len(wake_bars) if wake_bars else 0,
        })
        
        # Brief delay to let HMDS fully activate
        time.sleep(2.0)
        return True
        
    except Exception as exc:  # noqa: BLE001
        log_fn({
            "type": "hmds_activation_failed",
            "run_id": run_id,
            "symbol": symbol,
            "error": repr(exc),
        })
        return False


def _attach_ib_error_logging(
    ib,  # noqa: ANN001
    *,
    log_fn: Callable[[dict], None],
    run_id: str,
    symbol: str,
    frequency: str,
) -> Callable[[], None]:
    """Attach a best-effort IB error handler for observability.

    In practice, "no data" situations (permissions, data farms, pacing) often show up
    only via errorEvent callbacks.

    Returns a detach function.
    """

    err_event = getattr(ib, "errorEvent", None)
    if err_event is None:
        return lambda: None

    def _on_error(reqId, errorCode, errorString, contract=None):  # noqa: ANN001,N803
        ev = {
            "type": "ib_error",
            "run_id": run_id,
            "symbol": symbol,
            "frequency": frequency,
            "req_id": int(reqId) if reqId is not None else None,
            "error_code": int(errorCode) if errorCode is not None else None,
            "error": str(errorString),
        }
        log_fn(ev)
        # Keep console printing lightweight but actionable.
        ts_print(f"[live][ib_error] code={ev['error_code']} reqId={ev['req_id']} {ev['error']}")

    try:
        err_event += _on_error
    except Exception:
        return lambda: None

    def _detach() -> None:
        # Avoid augmented-assignment to a closure variable (would require `nonlocal`).
        try:
            ib.errorEvent -= _on_error
        except Exception:
            pass

    return _detach


def run_live_session(cfg: LiveSessionConfig) -> None:
    if not _HAVE_IB:
        raise RuntimeError("ib_insync is required; install it with 'pip install ib-insync'")
    
    # Suppress RuntimeWarning for unawaited coroutines from ib_insync
    # ib_insync's reqHistoricalData creates coroutines internally but handles them
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*coroutine.*was never awaited.*")

    live_dir = Path(cfg.log_dir) if cfg.log_dir else None

    log_path: Path | None = None
    run_id = cfg.run_id or create_run_id()
    if cfg.log_to_disk:
        log_path = make_log_path(symbol=cfg.symbol, frequency=cfg.frequency, run_id=run_id, live_dir=live_dir)

    # IMPORTANT: single-IB design (robustness):
    # We reuse the *same* ib_insync.IB instance for both:
    # - market data / event loop
    # - order execution via the Broker abstraction
    # This reduces reconnection complexity and avoids clientId coordination.
    broker: object | None = None

    def _log(event: dict) -> None:  # noqa: ANN001
        if log_path is None:
            return
        try:
            append_event(log_path, event)
        except Exception:
            # Logging must never crash the trading loop.
            pass

    _log(
        {
            "type": "run_start",
            "run_id": run_id,
            "symbol": cfg.symbol,
            "frequency": cfg.frequency,
            "tsteps": int(cfg.tsteps),
            "backend": cfg.backend,
            "initial_equity": float(cfg.initial_equity),
            "kill_switch_enabled": bool(is_kill_switch_enabled(live_dir)),
        }
    )

    ib = IB()  # type: ignore[call-arg]

    preferred_client_id = IbConfig.client_id if cfg.client_id is None else int(cfg.client_id)

    # Initialize observability components (before state machine/reconnect controller)
    live_dir_path = Path(live_dir) if live_dir else (Path(__file__).resolve().parents[1] / "ui_state" / "live")
    status_manager = StatusManager(live_dir_path / "status.json")
    alert_manager = AlertManager(live_dir_path / "alerts.jsonl")
    bar_tracker = PersistentBarTracker(live_dir_path / "last_bar.json")

    # Initialize state machine
    state_machine = StateMachine(
        initial_state=SystemState.INITIALIZING,
        event_logger=_log,
        run_id=run_id,
    )

    # Initialize reconnect controller (needs alert_manager)
    reconnect_config = ReconnectConfig(
        host=IbConfig.host,
        port=IbConfig.port,
        preferred_client_id=preferred_client_id,
    )
    reconnect_controller = ReconnectController(
        ib=ib,
        config=reconnect_config,
        event_logger=_log,
        run_id=run_id,
        alert_manager=alert_manager,
    )

    ts_print(
        f"[live] Connecting to TWS at {IbConfig.host}:{IbConfig.port} "
        f"(preferred clientId={preferred_client_id})",
    )

    detach_ib_error_logger: Callable[[], None] | None = None

    # Reconnect bookkeeping (wired later, after bars subscription exists).
    reconnect_stop = threading.Event()
    reconnect_thread: threading.Thread | None = None
    detach_disconnected_handler: Callable[[], None] | None = None
    
    # Connection health tracking
    connection_established_time: float | None = None
    
    # Shutdown flag (accessed by _heartbeat thread)
    shutdown_requested = False

    try:
        state_machine.transition_to(SystemState.CONNECTING, reason="initial_connection")
        connected_client_id = ConnectionManager.connect_with_unique_client_id(
            ib,
            host=IbConfig.host,
            port=IbConfig.port,
            preferred_client_id=preferred_client_id,
        )
        connection_established_time = time.time()
        state_machine.transition_to(SystemState.CONNECTED, reason="connection_established")
        ts_print(f"[live] Connected (clientId={connected_client_id}).")
        _log(
            {
                "type": "broker_connected",
                "run_id": run_id,
                "host": IbConfig.host,
                "port": int(IbConfig.port),
                "client_id": int(connected_client_id),
                "connection_time": connection_established_time,
            }
        )
        
        # Initialize status.json with CONNECTED state
        status_manager.update(
            state=SystemState.CONNECTED.value,
            connection_info={
                "uptime_minutes": 0.0,
                "last_reconnect": None,
            },
            last_bar_info={
                "time": None,
                "age_minutes": None,
                "expected_next": None,
            },
            position_info={"status": "UNKNOWN", "quantity": 0.0},
            alert_count=0,
            kill_switch_enabled=bool(is_kill_switch_enabled(live_dir)),
        )

        # Critical for diagnosing "connected but no data".
        detach_ib_error_logger = _attach_ib_error_logging(
            ib,
            log_fn=_log,
            run_id=run_id,
            symbol=cfg.symbol,
            frequency=cfg.frequency,
        )

        contract = Stock(cfg.symbol, "SMART", "USD")  # type: ignore[call-arg]
        ib.qualifyContracts(contract)

        predictor = LivePredictor.from_config(
            LivePredictorConfig(frequency=cfg.frequency, tsteps=cfg.tsteps),
        )

        # Pre-seed predictor buffer from historical CSV to enable immediate predictions.
        # This avoids waiting for 24+ bars on startup (critical for large timeframes like 240min).
        from src.config import get_hourly_data_csv_path
        historical_csv = get_hourly_data_csv_path(cfg.frequency)
        if os.path.exists(historical_csv):
            loaded = predictor.warmup_from_csv(historical_csv, cfg.frequency)
            ts_print(f"[live] Predictor warmed up with {loaded} bars from {historical_csv}")
        else:
            ts_print(f"[live] Historical CSV not found at {historical_csv}; starting cold.")

        # Reuse the same IB instance inside the broker (when backend=IBKR_TWS).
        broker = make_broker(cfg.backend, ib=ib)
        if hasattr(broker, "connect"):
            try:
                broker.connect()  # type: ignore[attr-defined]
            except Exception:
                # Not all brokers have connect(); ignore.
                pass

        # Cancel any stale orders from previous sessions to start fresh.
        # This prevents issues with orphaned brackets or orders placed with old configs.
        try:
            open_orders = broker.get_open_orders() if hasattr(broker, "get_open_orders") else []
            canceled_count = 0
            for order in open_orders:
                order_symbol = str(getattr(order, "symbol", ""))
                if order_symbol == cfg.symbol:
                    order_id = str(getattr(order, "order_id", ""))
                    try:
                        broker.cancel_order(order_id)
                        canceled_count += 1
                        _log({
                            "type": "stale_order_canceled",
                            "run_id": run_id,
                            "symbol": cfg.symbol,
                            "order_id": order_id,
                        })
                    except Exception as exc:
                        _log({
                            "type": "stale_order_cancel_failed",
                            "run_id": run_id,
                            "symbol": cfg.symbol,
                            "order_id": order_id,
                            "error": repr(exc),
                        })
            if canceled_count > 0:
                ts_print(f"[live] Canceled {canceled_count} stale order(s) for {cfg.symbol}")
                time.sleep(2.0)  # Give IBKR time to process cancellations
        except Exception as exc:
            ts_print(f"[live] Warning: failed to cleanup stale orders: {exc}")

        strat_cfg = make_strategy_config_from_defaults()
        exec_ctx = ExecutionContext(symbol=cfg.symbol)

        equity = float(cfg.initial_equity)
        has_open_position = False
        bar_count = 0

        # Track order lifecycle across snapshots (for `order_status` / `fill` events).
        order_state: dict[str, dict[str, object]] = {}

        def _get_position_status() -> dict[str, object]:
            """Get current position status for status reporting.
            
            Returns dict with 'status' (FLAT/OPEN/UNKNOWN) and 'quantity'.
            """
            # Return early if disconnected to avoid triggering unawaited coroutines
            if not ib.isConnected():
                return {"status": "UNKNOWN", "quantity": 0.0}
            
            try:
                positions = broker.get_positions() if hasattr(broker, "get_positions") else []
                if not positions:
                    return {"status": "FLAT", "quantity": 0.0}
                
                # Sum quantities across all positions
                total_qty = 0.0
                for pos in positions:
                    try:
                        qty = float(getattr(pos, "quantity", 0.0))
                        total_qty += qty
                    except Exception:
                        continue
                
                if abs(total_qty) < 0.01:
                    return {"status": "FLAT", "quantity": 0.0}
                return {"status": "OPEN", "quantity": total_qty}
            except Exception:
                return {"status": "UNKNOWN", "quantity": 0.0}

        def _log_broker_snapshot(*, where: str, bar_time: str | None = None) -> None:
            """Best-effort broker snapshot for observability.

            This uses the repo's Broker abstraction when available.
            """

            nonlocal order_state

            try:
                positions = broker.get_positions() if hasattr(broker, "get_positions") else []
            except Exception:
                positions = []

            try:
                open_orders = broker.get_open_orders() if hasattr(broker, "get_open_orders") else []
            except Exception:
                open_orders = []

            try:
                all_orders = broker.get_all_orders() if hasattr(broker, "get_all_orders") else list(open_orders)
            except Exception:
                all_orders = list(open_orders)

            try:
                acct = broker.get_account_summary() if hasattr(broker, "get_account_summary") else {}
            except Exception:
                acct = {}

                _log(
                    {
                        "type": "broker_snapshot",
                        "run_id": run_id,
                        "symbol": cfg.symbol,
                        "frequency": cfg.frequency,
                        "bar_time": bar_time,
                        "where": where,
                        "kill_switch_enabled": bool(is_kill_switch_enabled(live_dir)),
                        "positions": positions,
                        "open_orders": open_orders,
                        "account_summary": acct,
                    }
                )

            try:
                order_state, lifecycle_events = derive_order_lifecycle_events(
                    order_state,
                    list(all_orders) if all_orders is not None else [],
                    run_id=run_id,
                    symbol=cfg.symbol,
                    frequency=cfg.frequency,
                    where=where,
                    bar_time=bar_time,
                )
                for ev in lifecycle_events:
                    _log(ev)
            except Exception:
                # Never let observability crash the loop.
                pass


        bar_size_setting = _frequency_to_bar_size_setting(cfg.frequency)

        # Keep up-to-date bar stream. We request a small lookback window so IB has
        # enough context and then pushes incremental updates.
        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr="7 D",
            barSizeSetting=bar_size_setting,
            whatToShow="TRADES",
            useRTH=False,
            formatDate=1,
            keepUpToDate=True,
        )

        # Note: this initial batch is returned synchronously by ib_insync, but our
        # main handler only logs completed *new* bars. Log the initial snapshot so
        # "no bar events" doesn't hide the initial state.
        try:
            initial_n_bars = int(len(bars)) if bars is not None else 0
        except Exception:
            # Unit tests (and some ib_insync fakes) may return a bars object without __len__.
            initial_n_bars = 0
        initial_last_bar_time = None
        try:
            if bars and len(bars) > 0:
                initial_last_bar_time = str(getattr(bars[-1], "date", None))
        except Exception:
            initial_last_bar_time = None

        _log(
            {
                "type": "data_initial",
                "run_id": run_id,
                "symbol": cfg.symbol,
                "frequency": cfg.frequency,
                "n_bars": initial_n_bars,
                "last_bar_time": initial_last_bar_time,
            }
        )
        # Removed verbose logging: initial bars info (logged to JSONL instead)

        ts_print(f"[live] Subscribed to keepUpToDate bars ({bar_size_setting}) for {cfg.symbol}.")
        _log(
            {
                "type": "data_subscribed",
                "run_id": run_id,
                "bar_size_setting": bar_size_setting,
                "duration": "7 D",
                "what_to_show": "TRADES",
                "use_rth": False,
            }
        )

        # Initial snapshot (before any bars are processed).
        _log_broker_snapshot(where="after_subscribe")

        # Lightweight liveness tracking for "connected but silent" scenarios.
        liveness = {
            "last_update_monotonic": time.monotonic(),
            "last_bar_time": None,
            "last_has_new_bar": None,
        }

        # Best-effort dedupe for completed bars.
        last_completed_bar_time_iso: str | None = None

        # State transition: subscribed to data, ready to trade
        state_machine.transition_to(SystemState.SUBSCRIBED, reason="data_subscription_established")
        state_machine.transition_to(SystemState.TRADING, reason="ready_to_trade")

        def _on_bar_update(bar_list, has_new_bar: bool) -> None:  # noqa: ANN001
            nonlocal equity, has_open_position, bar_count, last_completed_bar_time_iso

            # Track liveness even for in-progress bar updates.
            try:
                liveness["last_update_monotonic"] = time.monotonic()
                liveness["last_has_new_bar"] = bool(has_new_bar)
            except Exception:
                pass

            # Only act on completed bars.
            if not has_new_bar:
                return
            if not bar_list:
                return
            
            # Clear stale data alert when bars resume
            try:
                alert_manager.clear_alert(f"stale_data_{cfg.symbol}")
            except Exception:
                pass

            try:
                b = bar_list[-1]

                # Best-effort timestamp normalization for logging.
                bar_time_raw = getattr(b, "date", None)
                bar_time_iso = None
                try:
                    if isinstance(bar_time_raw, datetime):
                        if bar_time_raw.tzinfo is None:
                            bar_time_iso = bar_time_raw.replace(tzinfo=timezone.utc).isoformat()
                        else:
                            bar_time_iso = bar_time_raw.astimezone(timezone.utc).isoformat()
                    elif bar_time_raw is not None:
                        bar_time_iso = str(bar_time_raw)
                except Exception:
                    bar_time_iso = None

                try:
                    liveness["last_bar_time"] = bar_time_iso
                except Exception:
                    pass

                # Dedupe: resubscribe/reconnect can replay the last completed bar.
                # Use persistent tracker for deduplication across restarts
                bar_hash = compute_bar_hash(b)
                if bar_time_iso is not None and bar_tracker.is_duplicate(bar_time_iso, bar_hash):
                    return
                if bar_time_iso is not None:
                    last_completed_bar_time_iso = bar_time_iso
                    bar_tracker.mark_processed(bar_time_iso, bar_hash)

                bar = {
                    "Time": bar_time_raw,
                    "Open": float(getattr(b, "open", 0.0)),
                    "High": float(getattr(b, "high", 0.0)),
                    "Low": float(getattr(b, "low", 0.0)),
                    "Close": float(getattr(b, "close", 0.0)),
                }

                _log(
                    {
                        "type": "bar",
                        "run_id": run_id,
                        "symbol": cfg.symbol,
                        "frequency": cfg.frequency,
                        "bar_time": bar_time_iso,
                        "open": float(bar["Open"]),
                        "high": float(bar["High"]),
                        "low": float(bar["Low"]),
                        "close": float(bar["Close"]),
                    }
                )

                bar_count += 1

                # During reconnect windows, keep logging bars but do not trade.
                if not state_machine.can_trade():
                    return
                every = max(1, int(getattr(cfg, "snapshot_every_n_bars", 1)))
                if bar_count % every == 0:
                    _log_broker_snapshot(where="on_bar", bar_time=bar_time_iso)

                predicted_price = predictor.update_and_predict(bar)
                current_price = float(bar["Close"])

                # Placeholder risk inputs; wire real ATR/sigma series later.
                model_error_sigma = max(1e-6, 0.5 * current_price * 0.01)
                atr = max(1e-6, current_price * 0.01)

                state = StrategyState(
                    current_price=current_price,
                    predicted_price=float(predicted_price),
                    model_error_sigma=float(model_error_sigma),
                    atr=float(atr),
                    account_equity=float(equity),
                    has_open_position=bool(has_open_position),
                )

                plan = compute_tp_sl_and_size(state, strat_cfg)

                decision_event: dict = {
                    "type": "decision",
                    "run_id": run_id,
                    "symbol": cfg.symbol,
                    "frequency": cfg.frequency,
                    "bar_time": bar_time_iso,
                    "close": float(current_price),
                    "predicted_price": float(predicted_price),
                    "model_error_sigma": float(model_error_sigma),
                    "atr": float(atr),
                    "has_open_position_before": bool(has_open_position),
                }

                if plan is None:
                    decision_event.update({"action": "NO_TRADE"})
                    _log(decision_event)
                    ts_print(
                        f"[live] {cfg.symbol} bar close={current_price:.2f} "
                        f"pred={predicted_price:.2f} -> no trade"
                    )
                    return

                decision_event.update(
                    {
                        "action": "TRADE",
                        "direction": int(plan.direction),
                        "size": float(plan.size),
                        "tp_price": float(plan.tp_price),
                        "sl_price": float(plan.sl_price),
                    }
                )

                # Kill switch: disable new order submissions when present.
                if is_kill_switch_enabled(live_dir):
                    decision_event["blocked_by_kill_switch"] = True
                    _log(decision_event)
                    ts_print("[live] KILL SWITCH enabled; blocking order submission.")
                    return

                _log(decision_event)

                try:
                    bracket_ids = submit_trade_plan_bracket(broker, plan, exec_ctx, current_price=current_price)
                except Exception as exc:  # noqa: BLE001
                    _log(
                        {
                            "type": "order_rejected",
                            "run_id": run_id,
                            "symbol": cfg.symbol,
                            "frequency": cfg.frequency,
                            "bar_time": bar_time_iso,
                            "direction": int(plan.direction),
                            "size": float(plan.size),
                            "tp_price": float(plan.tp_price),
                            "sl_price": float(plan.sl_price),
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    ts_print(f"[live] Bracket order submission failed: {exc}")
                    return

                has_open_position = True

                _log(
                    {
                        "type": "order_submitted",
                        "run_id": run_id,
                        "symbol": cfg.symbol,
                        "frequency": cfg.frequency,
                        "bar_time": bar_time_iso,
                        "entry_order_id": bracket_ids.entry_id if bracket_ids else None,
                        "tp_order_id": bracket_ids.tp_id if bracket_ids else None,
                        "sl_order_id": bracket_ids.sl_id if bracket_ids else None,
                        "direction": int(plan.direction),
                        "size": float(plan.size),
                        "tp_price": float(plan.tp_price),
                        "sl_price": float(plan.sl_price),
                    }
                )

                # Snapshot immediately after submitting an order.
                _log_broker_snapshot(where="after_order_submit", bar_time=bar_time_iso)

                # Show actual rounded quantity for clarity (plan.size is pre-rounding)
                actual_qty = int(round(plan.size))
                if actual_qty <= 0:
                    actual_qty = 1
                ts_print(
                    f"[live] TRADE {cfg.symbol} dir={plan.direction:+d} size={actual_qty} (plan={plan.size:.2f}) "
                    f"tp={plan.tp_price:.2f} sl={plan.sl_price:.2f} "
                    f"entry_id={bracket_ids.entry_id if bracket_ids else 'N/A'}"
                )

                if cfg.stop_after_first_trade:
                    ts_print("[live] stop_after_first_trade=True; stopping event loop.")
                    state_machine.transition_to(SystemState.STOPPED, reason="stop_after_first_trade")
                    ib.disconnect()

            except Exception as exc:  # noqa: BLE001
                _log(
                    {
                        "type": "error",
                        "run_id": run_id,
                        "where": "on_bar_update",
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                ts_print(f"[live] ERROR in bar handler: {exc}")

        bars.updateEvent += _on_bar_update

        # -----------------
        # Disconnect handling
        # -----------------
        # IB Gateway/TWS restarts will drop the socket. We keep the Python process
        # alive and attempt to reconnect + resubscribe.

        def _reconnect_loop() -> None:
            nonlocal bars, has_open_position, last_completed_bar_time_iso
            nonlocal connection_established_time

            while not reconnect_stop.is_set():
                # Use ReconnectController for all retry logic
                success, connected_client_id = reconnect_controller.attempt_reconnect(
                    is_market_hours_fn=lambda: _is_market_hours(premarket=True),
                    is_trading_day_fn=_is_trading_day,
                )

                if not success:
                    continue  # Backoff/sleep already handled by controller

                # Track connection uptime for health monitoring
                reconnect_time = time.time()
                connection_uptime_seconds = None
                if connection_established_time is not None:
                    connection_uptime_seconds = reconnect_time - connection_established_time
                connection_established_time = reconnect_time

                # Re-qualify contract and re-subscribe keepUpToDate stream.
                try:
                    contract2 = Stock(cfg.symbol, "SMART", "USD")  # type: ignore[call-arg]
                    ib.qualifyContracts(contract2)

                    # Detach handler from old bars object (best-effort) to avoid leaks.
                    try:
                        bars.updateEvent -= _on_bar_update
                    except Exception:
                        pass

                    bars = ib.reqHistoricalData(
                        contract2,
                        endDateTime="",
                        durationStr="7 D",
                        barSizeSetting=bar_size_setting,
                        whatToShow="TRADES",
                        useRTH=False,
                        formatDate=1,
                        keepUpToDate=True,
                    )
                    bars.updateEvent += _on_bar_update
                    last_completed_bar_time_iso = None

                    # Reconcile basic position state (best-effort).
                    try:
                        positions = broker.get_positions() if hasattr(broker, "get_positions") else []
                        has_open_position = bool(positions)
                    except Exception:
                        pass

                    _log(
                        {
                            "type": "data_resubscribed",
                            "run_id": run_id,
                            "symbol": cfg.symbol,
                            "frequency": cfg.frequency,
                            "bar_size_setting": bar_size_setting,
                        }
                    )

                    _log_broker_snapshot(where="after_reconnect")

                    # Force HMDS activation to ensure keepUpToDate works after reconnect
                    # Error 2107 (HMDS inactive) causes keepUpToDate to stop delivering bars
                    ts_print("[live] Activating HMDS data farm after reconnection...")
                    _activate_hmds(ib, contract2, bar_size_setting, _log, run_id, cfg.symbol)

                    # Transition back to trading state
                    state_machine.transition_to(SystemState.TRADING, reason="reconnect_successful")
                    reconnect_controller.reset_backoff()
                    return

                except Exception as exc:  # noqa: BLE001
                    _log(
                        {
                            "type": "resubscribe_failed",
                            "run_id": run_id,
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    ts_print(f"[live] Resubscription failed after reconnect: {exc}")
                    # Disconnect and retry
                    try:
                        ib.disconnect()
                    except Exception:
                        pass

        def _start_reconnector() -> None:
            nonlocal reconnect_thread
            if reconnect_stop.is_set():
                return
            if reconnect_thread is not None and reconnect_thread.is_alive():
                return
            reconnect_thread = threading.Thread(
                target=_reconnect_loop,
                name="ibkr_reconnector",
                daemon=True,
            )
            reconnect_thread.start()

        def _on_disconnected() -> None:
            nonlocal connection_established_time

            if not state_machine.is_running() or reconnect_stop.is_set():
                return

            # Track connection uptime for health monitoring
            disconnect_time = time.time()
            connection_uptime_seconds = None
            if connection_established_time is not None:
                connection_uptime_seconds = disconnect_time - connection_established_time

            # Transition to disconnected state, pause trading
            state_machine.transition_to(SystemState.DISCONNECTED, reason="connection_lost")
            state_machine.transition_to(SystemState.RECONNECTING, reason="attempting_reconnect")

            _log(
                {
                    "type": "ib_disconnected",
                    "run_id": run_id,
                    "host": IbConfig.host,
                    "port": int(IbConfig.port),
                    "disconnect_time": disconnect_time,
                    "connection_uptime_seconds": connection_uptime_seconds,
                }
            )
            uptime_msg = f" (uptime: {connection_uptime_seconds:.0f}s)" if connection_uptime_seconds else ""
            ts_print(f"[live] DISCONNECTED from TWS{uptime_msg}; pausing trading and starting reconnect loop...")
            _start_reconnector()

        disc_event = getattr(ib, "disconnectedEvent", None)
        if disc_event is not None:
            try:
                disc_event += _on_disconnected

                def _detach_disconnected() -> None:
                    try:
                        ib.disconnectedEvent -= _on_disconnected
                    except Exception:
                        pass

                detach_disconnected_handler = _detach_disconnected
            except Exception:
                detach_disconnected_handler = None

        # Heartbeat: log status periodically and warn if no updates arrive.
        hb_stop = threading.Event()

        def _heartbeat() -> None:
            nonlocal shutdown_requested
            last_poll_check_time = 0.0
            poll_check_interval = 300.0  # Check every 5 minutes
            
            while not hb_stop.wait(30.0):
                try:
                    age = float(time.monotonic() - float(liveness.get("last_update_monotonic", 0.0)))
                except Exception:
                    age = -1.0

                is_market_open = _is_market_hours(premarket=True)

                _log(
                    {
                        "type": "heartbeat",
                        "run_id": run_id,
                        "symbol": cfg.symbol,
                        "frequency": cfg.frequency,
                        "bar_count": int(bar_count),
                        "seconds_since_last_update": age,
                        "last_has_new_bar": liveness.get("last_has_new_bar"),
                        "last_bar_time": liveness.get("last_bar_time"),
                        "is_market_open": is_market_open,
                    }
                )
                
                # Update status.json with current metrics
                try:
                    # Calculate connection uptime
                    uptime_minutes = None
                    if connection_established_time is not None:
                        uptime_minutes = (time.time() - connection_established_time) / 60.0
                    
                    # Calculate last bar age
                    last_bar_time_str = liveness.get("last_bar_time")
                    last_bar_age_minutes = None
                    if last_bar_time_str:
                        try:
                            from dateutil import parser as date_parser
                            last_bar_dt = date_parser.parse(last_bar_time_str)
                            if last_bar_dt.tzinfo is None:
                                last_bar_dt = last_bar_dt.replace(tzinfo=timezone.utc)
                            last_bar_age_minutes = (datetime.now(timezone.utc) - last_bar_dt).total_seconds() / 60.0
                        except Exception:
                            pass
                    
                    # Get position status
                    position_info = _get_position_status()
                    
                    # Update status file
                    status_manager.update(
                        state=state_machine.current.value,
                        connection_info={
                            "uptime_minutes": uptime_minutes,
                            "last_reconnect": None,  # Could track this separately if needed
                        },
                        last_bar_info={
                            "time": last_bar_time_str,
                            "age_minutes": last_bar_age_minutes,
                            "expected_next": None,  # Could calculate based on frequency
                        },
                        position_info=position_info,
                        alert_count=alert_manager.count(),
                        kill_switch_enabled=bool(is_kill_switch_enabled(live_dir)),
                    )
                except Exception:
                    # Never crash heartbeat on status update failure
                    pass
                
                # Proactive polling: check if historical data has newer bars than subscription
                # This detects when IBKR's keepUpToDate stops working (common bug for larger bar sizes)
                current_time = time.time()
                if is_market_open and (current_time - last_poll_check_time >= poll_check_interval):
                    last_poll_check_time = current_time
                    try:
                        # Request latest historical bar to compare with subscription
                        # Skip if disconnected to avoid unawaited coroutines
                        if not ib.isConnected():
                            pass  # Skip check when disconnected
                        else:
                            # Get latest bar from historical data (non-keepUpToDate request)
                            check_bars = ib.reqHistoricalData(
                                contract,
                                endDateTime="",
                                durationStr="1 D",
                                barSizeSetting=bar_size_setting,
                                whatToShow="TRADES",
                                useRTH=False,
                                formatDate=1,
                                keepUpToDate=False,  # One-time request
                            )
                            
                            if check_bars and len(check_bars) > 0:
                                latest_historical_bar = check_bars[-1]
                                latest_historical_time = getattr(latest_historical_bar, "date", None)
                                
                                # Compare with last received bar from subscription
                                last_subscription_bar_time_str = liveness.get("last_bar_time")
                                
                                if latest_historical_time and last_subscription_bar_time_str:
                                    # Parse subscription bar time
                                    from dateutil import parser as date_parser
                                    try:
                                        last_subscription_time = date_parser.parse(last_subscription_bar_time_str)
                                        if last_subscription_time.tzinfo is None:
                                            last_subscription_time = last_subscription_time.replace(tzinfo=timezone.utc)
                                        
                                        # Ensure historical time has timezone
                                        if isinstance(latest_historical_time, datetime):
                                            if latest_historical_time.tzinfo is None:
                                                latest_historical_time = latest_historical_time.replace(tzinfo=timezone.utc)
                                            
                                            # If historical data has a newer bar, trigger reconnection
                                            if latest_historical_time > last_subscription_time:
                                                time_diff = (latest_historical_time - last_subscription_time).total_seconds()
                                                _log(
                                                    {
                                                        "type": "keepuptodate_stale_detected",
                                                        "run_id": run_id,
                                                        "last_subscription_bar": last_subscription_bar_time_str,
                                                        "latest_historical_bar": latest_historical_time.isoformat(),
                                                        "time_diff_seconds": time_diff,
                                                    }
                                                )
                                                # Add CRITICAL alert for keepUpToDate failure
                                                alert_manager.add_alert(
                                                    key=f"keepuptodate_stale_{cfg.symbol}",
                                                    severity=AlertSeverity.CRITICAL,
                                                    alert_type="keepuptodate_stale",
                                                    message=f"keepUpToDate subscription stale: historical bar {time_diff:.0f}s newer than subscription",
                                                )
                                                ts_print(
                                                    f"[live] keepUpToDate stale: latest historical bar is "
                                                    f"{time_diff:.0f}s newer than subscription. Forcing reconnect..."
                                                )
                                            # Pause trading and force disconnect
                                            state_machine.transition_to(
                                                SystemState.DISCONNECTED, reason="stale_keepuptodate_data"
                                            )
                                            try:
                                                ib.disconnect()
                                            except Exception:
                                                pass
                                    except Exception as parse_exc:
                                        # Failed to parse/compare times, skip this check
                                        pass
                    except Exception as poll_exc:
                        # Polling failed, log but don't crash heartbeat
                        _log(
                            {
                                "type": "keepuptodate_poll_error",
                                "run_id": run_id,
                                "error": repr(poll_exc),
                            }
                        )

                # If we haven't received *any* bar updates for a while, surface it.
                # Only warn during market hours to avoid spam when market is closed.
                warn_after = _no_update_warning_threshold_seconds(cfg.frequency)
                if age >= 0 and age > warn_after and is_market_open:
                    # Add WARNING alert for stale data
                    alert_manager.add_alert(
                        key=f"stale_data_{cfg.symbol}",
                        severity=AlertSeverity.WARNING,
                        alert_type="stale_data",
                        message=f"No bar updates for {age:.0f}s (threshold={warn_after:.0f}s)",
                    )
                    ts_print(
                        f"[live] WARNING: no bar updates received for {age:.0f}s "
                        f"(threshold={warn_after:.0f}s, frequency={cfg.frequency}). "
                        "Check TWS/Gateway logs for market data/permissions errors."
                    )

                    # During market hours, stale data likely means subscription died.
                    # Trigger reconnection to re-establish keepUpToDate stream.
                    if is_market_open and not shutdown_requested:
                        is_connected = getattr(ib, "isConnected", None)
                        if callable(is_connected) and is_connected():
                            _log(
                                {
                                    "type": "stale_data_reconnect_trigger",
                                    "run_id": run_id,
                                    "seconds_since_last_update": age,
                                    "threshold": warn_after,
                                    "is_market_open": is_market_open,
                                }
                            )
                            ts_print(
                                f"[live] Stale data during market hours ({age:.0f}s); "
                                "forcing disconnect to trigger re-subscription..."
                            )
                            # Pause trading before disconnect
                            state_machine.transition_to(
                                SystemState.DISCONNECTED, reason="stale_bar_data"
                            )
                            # Force disconnect to trigger reconnection logic
                            try:
                                ib.disconnect()
                            except Exception:
                                pass

        hb_thread = threading.Thread(target=_heartbeat, name="ibkr_live_heartbeat", daemon=True)
        hb_thread.start()

        ts_print("[live] Running event loop (Ctrl+C to stop)...")
        ib.run()

    except Exception as exc:  # noqa: BLE001
        _log(
            {
                "type": "error",
                "run_id": run_id,
                "where": "run_live_session",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        raise

    finally:
        # Stop heartbeat thread (if started).
        try:
            if "hb_stop" in locals():
                hb_stop.set()
        except Exception:
            pass
        try:
            if "hb_thread" in locals() and hb_thread is not None:
                hb_thread.join(timeout=1.0)
        except Exception:
            pass

        # Prevent reconnect loop from fighting shutdown.
        shutdown_requested = True
        try:
            reconnect_stop.set()
        except Exception:
            pass
        try:
            if detach_disconnected_handler is not None:
                detach_disconnected_handler()
        except Exception:
            pass
        try:
            if reconnect_thread is not None:
                reconnect_thread.join(timeout=2.0)
        except Exception:
            pass

        # Detach IB error logger (if attached).
        try:
            if detach_ib_error_logger is not None:
                detach_ib_error_logger()
        except Exception:
            pass

        # Disconnect broker first (it may own a separate IB connection).
        try:
            if broker is not None:
                disc = getattr(broker, "disconnect", None)
                if callable(disc):
                    disc()
        except Exception:
            pass

        # Disconnect the market-data IB connection.
        try:
            ib.disconnect()
        except Exception:
            pass

        _log(
            {
                "type": "run_end",
                "run_id": run_id,
                "symbol": cfg.symbol,
                "frequency": cfg.frequency,
                "kill_switch_enabled": bool(is_kill_switch_enabled(live_dir)),
            }
        )


def _resolve_symbol_frequency_from_active_config(
    *,
    cli_symbol: str | None,
    cli_frequency: str | None,
    active_config: dict | None,
) -> tuple[str, str]:
    """Resolve (symbol, frequency) for live run.

    Priority:
    1) Explicit CLI args (highest)
    2) configs/active.json["meta"]["symbol"/"frequency"]

    We intentionally do NOT silently fall back to hardcoded defaults if both are
    missing, because running the wrong frequency is a high-risk failure mode.
    """
    symbol = (cli_symbol or "").strip() or None
    frequency = (cli_frequency or "").strip() or None

    meta = None
    if isinstance(active_config, dict):
        m = active_config.get("meta")
        meta = m if isinstance(m, dict) else None

    if symbol is None and meta is not None:
        sym = meta.get("symbol")
        symbol = str(sym).strip() if sym is not None else None

    if frequency is None and meta is not None:
        freq = meta.get("frequency")
        frequency = str(freq).strip() if freq is not None else None

    if not symbol:
        raise SystemExit(
            "Missing --symbol and configs/active.json meta.symbol is not set. "
            "Pass --symbol explicitly or promote a config with symbol metadata."
        )
    if not frequency:
        raise SystemExit(
            "Missing --frequency and configs/active.json meta.frequency is not set. "
            "Pass --frequency explicitly or promote a config with frequency metadata."
        )

    return symbol.upper(), frequency


def main() -> None:
    """CLI entrypoint - parse args and delegate to src.live.engine."""
    p = argparse.ArgumentParser(description="IBKR/TWS live session runner (bar-by-bar model predictions).")
    p.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Trading symbol. If omitted, uses configs/active.json meta.symbol.",
    )
    p.add_argument(
        "--frequency",
        type=str,
        default=None,
        help="Bar frequency (e.g. 15min, 60min). If omitted, uses configs/active.json meta.frequency.",
    )
    p.add_argument("--tsteps", type=int, default=5)
    p.add_argument("--backend", type=str, default="IBKR_TWS")
    p.add_argument("--initial-equity", type=float, default=10_000.0)
    p.add_argument("--stop-after-first-trade", action="store_true")
    p.add_argument(
        "--client-id",
        type=int,
        default=None,
        help="Optional TWS clientId override; if omitted, uses TWS_CLIENT_ID env var / src.config.IB.",
    )
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id for the JSONL log (default: UTC timestamp).",
    )
    p.add_argument(
        "--no-disk-log",
        action="store_true",
        help="Disable writing ui_state/live JSONL logs.",
    )
    p.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Optional override for the live log directory (default: repo_root/ui_state/live).",
    )
    p.add_argument(
        "--snapshot-every-n-bars",
        type=int,
        default=1,
        help="Emit broker_snapshot event every N bars (default: 1).",
    )
    args = p.parse_args()

    # Import here to allow lazy loading
    from src.live.engine import LiveEngineConfig, run as run_engine

    # Resolve defaults from configs/active.json (promoted config) when CLI omits them.
    from src.core import config_library as _cfg_lib

    active_cfg = _cfg_lib.read_active_config() or {}
    symbol, frequency = _resolve_symbol_frequency_from_active_config(
        cli_symbol=args.symbol,
        cli_frequency=args.frequency,
        active_config=active_cfg,
    )

    cfg = LiveEngineConfig(
        symbol=symbol,
        frequency=frequency,
        tsteps=int(args.tsteps),
        backend=args.backend,
        initial_equity=float(args.initial_equity),
        stop_after_first_trade=bool(args.stop_after_first_trade),
        client_id=args.client_id,
        run_id=args.run_id,
        log_to_disk=not bool(args.no_disk_log),
        log_dir=args.log_dir,
        snapshot_every_n_bars=int(args.snapshot_every_n_bars),
    )
    run_engine(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
