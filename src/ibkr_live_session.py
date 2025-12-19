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
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from src.config import IB as IbConfig
from src.execution import ExecutionContext, submit_trade_plan
from src.live_log import (
    append_event,
    create_run_id,
    is_kill_switch_enabled,
    make_log_path,
)


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

    We use max(5 minutes, 2x bar period) as a conservative threshold.
    """
    base = 300.0
    mins = _frequency_to_minutes(freq)
    if mins is None or mins <= 0:
        return base
    return max(base, float(2 * mins * 60))


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
            sleeper = getattr(ib, "sleep", None)
            if callable(sleeper):
                sleeper(0.35)
            else:
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
        print(f"[live][ib_error] code={ev['error_code']} reqId={ev['req_id']} {ev['error']}")

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
    print(
        f"[live] Connecting to TWS at {IbConfig.host}:{IbConfig.port} "
        f"(preferred clientId={preferred_client_id})",
    )

    detach_ib_error_logger: Callable[[], None] | None = None

    # Reconnect bookkeeping (wired later, after bars subscription exists).
    shutdown_requested = False
    reconnect_stop = threading.Event()
    reconnect_thread: threading.Thread | None = None
    detach_disconnected_handler: Callable[[], None] | None = None

    try:
        connected_client_id = _connect_with_unique_client_id(
            ib,
            host=IbConfig.host,
            port=IbConfig.port,
            preferred_client_id=preferred_client_id,
        )
        print(f"[live] Connected (clientId={connected_client_id}).")
        _log(
            {
                "type": "broker_connected",
                "run_id": run_id,
                "host": IbConfig.host,
                "port": int(IbConfig.port),
                "client_id": int(connected_client_id),
            }
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

        # Reuse the same IB instance inside the broker (when backend=IBKR_TWS).
        broker = make_broker(cfg.backend, ib=ib)
        if hasattr(broker, "connect"):
            try:
                broker.connect()  # type: ignore[attr-defined]
            except Exception:
                # Not all brokers have connect(); ignore.
                pass

        strat_cfg = make_strategy_config_from_defaults()
        exec_ctx = ExecutionContext(symbol=cfg.symbol)

        equity = float(cfg.initial_equity)
        has_open_position = False
        bar_count = 0

        # Track order lifecycle across snapshots (for `order_status` / `fill` events).
        order_state: dict[str, dict[str, object]] = {}

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
        print(
            f"[live] Initial historical bars received: n={initial_n_bars} last={initial_last_bar_time}"
        )

        print(f"[live] Subscribed to keepUpToDate bars ({bar_size_setting}) for {cfg.symbol}.")
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

        # When false, we keep the event loop running but skip any trading actions.
        trading_enabled = True

        # Best-effort dedupe for completed bars.
        last_completed_bar_time_iso: str | None = None

        def _on_bar_update(bar_list, has_new_bar: bool) -> None:  # noqa: ANN001
            nonlocal equity, has_open_position, bar_count, trading_enabled, last_completed_bar_time_iso

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
                if bar_time_iso is not None and bar_time_iso == last_completed_bar_time_iso:
                    return
                if bar_time_iso is not None:
                    last_completed_bar_time_iso = bar_time_iso

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
                if not trading_enabled:
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
                    print(
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
                    print("[live] KILL SWITCH enabled; blocking order submission.")
                    return

                _log(decision_event)

                try:
                    order_id = submit_trade_plan(broker, plan, exec_ctx)
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
                    print(f"[live] Order submission failed: {exc}")
                    return

                has_open_position = True

                _log(
                    {
                        "type": "order_submitted",
                        "run_id": run_id,
                        "symbol": cfg.symbol,
                        "frequency": cfg.frequency,
                        "bar_time": bar_time_iso,
                        "order_id": order_id,
                        "direction": int(plan.direction),
                        "size": float(plan.size),
                        "tp_price": float(plan.tp_price),
                        "sl_price": float(plan.sl_price),
                    }
                )

                # Snapshot immediately after submitting an order.
                _log_broker_snapshot(where="after_order_submit", bar_time=bar_time_iso)

                print(
                    f"[live] TRADE {cfg.symbol} dir={plan.direction:+d} size={plan.size:.2f} "
                    f"tp={plan.tp_price:.2f} sl={plan.sl_price:.2f} order_id={order_id}"
                )

                if cfg.stop_after_first_trade:
                    print("[live] stop_after_first_trade=True; stopping event loop.")
                    shutdown_requested = True
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
                print(f"[live] ERROR in bar handler: {exc}")

        bars.updateEvent += _on_bar_update

        # -----------------
        # Disconnect handling
        # -----------------
        # IB Gateway/TWS restarts will drop the socket. We keep the Python process
        # alive and attempt to reconnect + resubscribe.

        def _reconnect_loop() -> None:
            nonlocal bars, trading_enabled, has_open_position, last_completed_bar_time_iso

            backoff = 1.0
            while not reconnect_stop.is_set():
                try:
                    # Wait a bit before hammering connect() while Gateway boots.
                    sleeper = getattr(ib, "sleep", None)
                    if callable(sleeper):
                        sleeper(backoff)
                    else:
                        time.sleep(backoff)

                    connected_client_id = _connect_with_unique_client_id(
                        ib,
                        host=IbConfig.host,
                        port=IbConfig.port,
                        preferred_client_id=preferred_client_id,
                    )

                    _log(
                        {
                            "type": "broker_reconnected",
                            "run_id": run_id,
                            "host": IbConfig.host,
                            "port": int(IbConfig.port),
                            "client_id": int(connected_client_id),
                        }
                    )
                    print(f"[live] Reconnected to TWS (clientId={connected_client_id}).")

                    # Re-qualify contract and re-subscribe keepUpToDate stream.
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

                    trading_enabled = True
                    return

                except Exception as exc:  # noqa: BLE001
                    _log(
                        {
                            "type": "reconnect_failed",
                            "run_id": run_id,
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    print(f"[live] Reconnect failed; will retry: {exc}")
                    backoff = min(30.0, max(1.0, backoff * 1.8))

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
            nonlocal trading_enabled, shutdown_requested

            if shutdown_requested or reconnect_stop.is_set():
                return

            # Pause trading immediately. We will resume after resubscribe.
            trading_enabled = False
            _log(
                {
                    "type": "ib_disconnected",
                    "run_id": run_id,
                    "host": IbConfig.host,
                    "port": int(IbConfig.port),
                }
            )
            print("[live] DISCONNECTED from TWS; pausing trading and starting reconnect loop...")
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
            while not hb_stop.wait(30.0):
                try:
                    age = float(time.monotonic() - float(liveness.get("last_update_monotonic", 0.0)))
                except Exception:
                    age = -1.0

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
                    }
                )

                # If we haven't received *any* bar updates for a while, surface it.
                warn_after = _no_update_warning_threshold_seconds(cfg.frequency)
                if age >= 0 and age > warn_after:
                    print(
                        f"[live] WARNING: no bar updates received for {age:.0f}s "
                        f"(threshold={warn_after:.0f}s, frequency={cfg.frequency}). "
                        "Check TWS/Gateway logs for market data/permissions errors."
                    )

        hb_thread = threading.Thread(target=_heartbeat, name="ibkr_live_heartbeat", daemon=True)
        hb_thread.start()

        print("[live] Running event loop (Ctrl+C to stop)...")
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
