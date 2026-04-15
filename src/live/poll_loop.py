"""Poll-based connect-on-demand live trading loop.

Replaces the persistent-connection / keepUpToDate architecture with a simpler
pattern for hourly+ bar frequencies:

    sleep → connect → fetch → process → disconnect → repeat

This eliminates all reconnection complexity (ReconnectController, HMDS
activation, stale-data polling, heartbeat threads) because each cycle
uses a fresh IB connection that is torn down immediately after use.

Survives PC sleep/wake: the wait loop polls wall-clock time every 30 s,
so oversleeping through a bar boundary simply triggers the cycle on wake.
"""
from __future__ import annotations

import os
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from src.broker import OrderRequest, OrderType, Side
from src.config import IB as IbConfig, get_hourly_data_csv_path
from src.execution import ExecutionContext, submit_trade_plan_bracket
from src.live.alerts import AlertManager, AlertSeverity
from src.live.persistence import PersistentBarTracker, compute_bar_hash
from src.live.reconnect import ConnectionManager
from src.live.state import StateMachine, SystemState
from src.live.status import StatusManager
from src.live_log import (
    append_event,
    create_run_id,
    is_kill_switch_enabled,
    make_log_path,
)
from src.live_predictor import LivePredictor, LivePredictorConfig
from src.strategy import StrategyState, compute_tp_sl_and_size
from src.trading_session import make_broker, make_strategy_config_from_defaults
from src.utils.timestamped_print import ts_print

try:
    from ib_insync import IB, Stock  # type: ignore[import]

    _HAVE_IB = True
except Exception:
    IB = object  # type: ignore[assignment]
    Stock = object  # type: ignore[assignment]
    _HAVE_IB = False


# ---------------------------------------------------------------------------
# Bar boundary calculation
# ---------------------------------------------------------------------------

def _get_eastern_tz():
    """Return the US/Eastern ZoneInfo, with fallback."""
    try:
        import zoneinfo
        return zoneinfo.ZoneInfo("America/New_York")
    except Exception:
        pass
    try:
        from backports.zoneinfo import ZoneInfo  # type: ignore[import]
        return ZoneInfo("America/New_York")
    except Exception:
        return timezone(timedelta(hours=-5))  # naive EST fallback


def compute_next_bar_boundary(
    frequency: str,
    now: datetime | None = None,
    *,
    buffer_minutes: int = 2,
) -> datetime:
    """Return the next wall-clock time when a bar completes + buffer.

    All arithmetic is in US/Eastern time so bars align to exchange hours.
    The returned datetime is timezone-aware (Eastern).

    Args:
        frequency: Bar frequency string, e.g. "60min", "240min".
        now: Current time (defaults to wall-clock UTC converted to Eastern).
        buffer_minutes: Extra minutes to wait after bar boundary for IBKR
            to finalise the bar data.

    Returns:
        Timezone-aware datetime (Eastern) of the next fetch time.
    """
    eastern = _get_eastern_tz()

    if now is None:
        now = datetime.now(timezone.utc).astimezone(eastern)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=eastern)
    else:
        now = now.astimezone(eastern)

    bar_minutes = _frequency_to_minutes(frequency)
    if bar_minutes is None:
        raise ValueError(f"Unsupported frequency for poll loop: {frequency!r}")

    # Align to the next bar boundary within the current day.
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    minutes_since_midnight = (now - midnight).total_seconds() / 60.0

    # Next boundary after *now*.
    bars_elapsed = int(minutes_since_midnight // bar_minutes)
    next_boundary_minutes = (bars_elapsed + 1) * bar_minutes

    next_boundary = midnight + timedelta(minutes=next_boundary_minutes)

    # If the boundary is still in the past (rounding edge), push forward.
    while next_boundary <= now:
        next_boundary += timedelta(minutes=bar_minutes)

    return next_boundary + timedelta(minutes=buffer_minutes)


def _frequency_to_minutes(freq: str) -> int | None:
    """Parse frequency string to minutes."""
    f = str(freq or "").lower().strip().replace(" ", "")
    if f.endswith("min") and f[:-3].isdigit():
        return int(f[:-3])
    mapping = {"1h": 60, "1hr": 60, "1hour": 60, "4h": 240, "4hr": 240, "4hour": 240}
    return mapping.get(f)


def _frequency_to_bar_size_setting(freq: str) -> str:
    """Convert frequency string to IB barSizeSetting."""
    f = freq.lower().strip().replace(" ", "")
    settings = {
        "1min": "1 min", "5min": "5 mins", "15min": "15 mins",
        "30min": "30 mins", "60min": "1 hour", "1h": "1 hour",
        "240min": "4 hours", "4h": "4 hours",
    }
    if f in settings:
        return settings[f]
    raise ValueError(f"Unsupported frequency: {freq!r}")


# ---------------------------------------------------------------------------
# Market hours helpers (simplified from ibkr_live_session.py)
# ---------------------------------------------------------------------------

def _is_market_hours(*, premarket: bool = True, at: datetime | None = None) -> bool:
    """Check if a given time is within US market hours.

    Args:
        premarket: Include pre-market hours (4 AM ET) in the window.
        at: Specific datetime to check.  Defaults to wall-clock now.
    """
    try:
        eastern = _get_eastern_tz()
        t = at.astimezone(eastern) if at is not None else datetime.now(timezone.utc).astimezone(eastern)
        if t.weekday() >= 5:
            return False
        minutes = t.hour * 60 + t.minute
        if _GW_RESTART_START_MINUTES <= minutes < _GW_RESTART_END_MINUTES:
            return False  # IB Gateway restart window — skip cycle
        start = 4 * 60 if premarket else 9 * 60 + 30
        return start <= minutes < 16 * 60
    except Exception:
        return True  # conservative fallback


def _is_trading_day(*, at: datetime | None = None) -> bool:
    """Check if a given date is a trading day (weekday + holiday calendar).

    Args:
        at: Specific datetime to check.  Defaults to wall-clock now.
    """
    try:
        eastern = _get_eastern_tz()
        t = at.astimezone(eastern) if at is not None else datetime.now(timezone.utc).astimezone(eastern)
        if t.weekday() >= 5:
            return False
        try:
            import pandas_market_calendars as mcal
            nyse = mcal.get_calendar("NASDAQ")
            today_str = t.strftime("%Y-%m-%d")
            schedule = nyse.schedule(start_date=today_str, end_date=today_str)
            return len(schedule) > 0
        except Exception:
            return True  # weekday, assume trading day
    except Exception:
        return True


# IB Gateway daily restart window (ET minutes).  reqHistoricalData hangs during
# this period, burning all retries.  _is_market_hours() blackouts cycles here.
_GW_RESTART_START_MINUTES: int = 3 * 60       # 03:00 ET
_GW_RESTART_END_MINUTES: int = 5 * 60 + 30   # 05:30 ET (30 min buffer past documented 05:00)

# Maximum seconds past the target bar boundary before trading is suppressed.
# The cycle still runs (model update, logging) but no orders are submitted.
_STALE_BAR_THRESHOLD_SECS: float = 15 * 60  # 15 minutes

# Hard timeout for a single poll cycle.  If a cycle (connect + fetch + process
# + disconnect) takes longer than this, a background timer force-disconnects
# the IB instance so that any hanging ib_insync call aborts immediately.
# This prevents the process from being stuck for hours when TWS connectivity
# flaps during PC sleep.
_CYCLE_TIMEOUT_SECS: float = 180  # 3 minutes


# ---------------------------------------------------------------------------
# Single poll cycle
# ---------------------------------------------------------------------------

def _run_single_cycle(
    *,
    cfg,  # LiveSessionConfig or LiveEngineConfig (duck-typed)
    bar_size_setting: str,
    preferred_client_id: int,
    predictor: LivePredictor,
    strat_cfg,
    exec_ctx: ExecutionContext,
    bar_tracker: PersistentBarTracker,
    status_manager: StatusManager,
    alert_manager: AlertManager,
    state_machine: StateMachine,
    log_fn: Callable[[dict], None],
    run_id: str,
    live_dir: Path | None,
    equity: float,
    has_open_position: bool,
    order_state: dict,
    suppress_trading: bool = False,
) -> tuple[float, bool, dict]:
    """Execute one connect-fetch-process-disconnect cycle.

    Args:
        suppress_trading: If True the cycle still fetches bars and updates
            the model / logs, but skips order submission (stale-bar guard).

    Returns updated (equity, has_open_position, order_state).
    """
    from src.live.order_utils import (
        check_positions_have_brackets,
        derive_order_lifecycle_events,
    )

    ib = IB()  # type: ignore[call-arg]
    broker = None

    # Hard timeout: if the cycle hasn't finished in _CYCLE_TIMEOUT_SECS,
    # force-disconnect so hanging ib_insync calls abort.
    def _force_disconnect():
        ts_print("[poll] Cycle timeout: forcing disconnect.")
        try:
            ib.disconnect()
        except Exception:
            pass

    cycle_timer = threading.Timer(_CYCLE_TIMEOUT_SECS, _force_disconnect)
    cycle_timer.daemon = True
    cycle_timer.start()

    try:
        # --- Connect ---
        state_machine.transition_to(SystemState.CONNECTING, reason="poll_cycle_start")
        connected_cid = ConnectionManager.connect_with_unique_client_id(
            ib,
            host=IbConfig.host,
            port=IbConfig.port,
            preferred_client_id=preferred_client_id,
            max_tries=25,
        )
        ts_print(f"[poll] Connected (clientId={connected_cid}).")

        log_fn({
            "type": "broker_connected",
            "run_id": run_id,
            "host": IbConfig.host,
            "port": int(IbConfig.port),
            "client_id": int(connected_cid),
        })

        # Broker (reuses this IB connection)
        broker = make_broker(cfg.backend, ib=ib)

        contract = Stock(cfg.symbol, "SMART", "USD")  # type: ignore[call-arg]
        ib.qualifyContracts(contract)

        # --- Fetch bars (one-shot) ---
        state_machine.transition_to(SystemState.PROCESSING, reason="fetching_bars")
        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr="7 D",
            barSizeSetting=bar_size_setting,
            whatToShow="TRADES",
            useRTH=False,
            formatDate=1,
            keepUpToDate=False,
        )

        if not bars:
            log_fn({"type": "poll_no_bars", "run_id": run_id, "symbol": cfg.symbol})
            ts_print("[poll] No bars returned from IBKR.")
            state_machine.transition_to(SystemState.SLEEPING, reason="no_bars")
            return equity, has_open_position, order_state

        b = bars[-1]
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
            pass

        # --- Deduplicate ---
        bar_hash = compute_bar_hash(b)
        if bar_time_iso is not None and bar_tracker.is_duplicate(bar_time_iso, bar_hash):
            ts_print(f"[poll] Bar {bar_time_iso} already processed, skipping.")
            state_machine.transition_to(SystemState.SLEEPING, reason="duplicate_bar")
            return equity, has_open_position, order_state

        bar = {
            "Time": bar_time_raw,
            "Open": float(getattr(b, "open", 0.0)),
            "High": float(getattr(b, "high", 0.0)),
            "Low": float(getattr(b, "low", 0.0)),
            "Close": float(getattr(b, "close", 0.0)),
        }

        log_fn({
            "type": "bar",
            "run_id": run_id,
            "symbol": cfg.symbol,
            "frequency": cfg.frequency,
            "bar_time": bar_time_iso,
            "open": bar["Open"],
            "high": bar["High"],
            "low": bar["Low"],
            "close": bar["Close"],
        })

        ts_print(
            f"[poll] Bar {bar_time_iso} O={bar['Open']:.2f} H={bar['High']:.2f} "
            f"L={bar['Low']:.2f} C={bar['Close']:.2f}"
        )

        # Mark as processed
        if bar_time_iso is not None:
            bar_tracker.mark_processed(bar_time_iso, bar_hash)

        # --- Broker snapshot ---
        try:
            positions = broker.get_positions() if hasattr(broker, "get_positions") else []
            open_orders = broker.get_open_orders() if hasattr(broker, "get_open_orders") else []
            all_orders = broker.get_all_orders() if hasattr(broker, "get_all_orders") else list(open_orders)
        except Exception:
            positions, open_orders, all_orders = [], [], []

        # Reconcile position state from broker — only count positions for the managed symbol
        has_open_position = any(
            getattr(p, "symbol", "") == cfg.symbol
            and abs(float(getattr(p, "quantity", 0.0))) >= 0.01
            for p in positions
        )

        # Order lifecycle events
        try:
            order_state, lifecycle_events = derive_order_lifecycle_events(
                order_state, list(all_orders),
                run_id=run_id, symbol=cfg.symbol, frequency=cfg.frequency,
                where="poll_cycle", bar_time=bar_time_iso,
            )
            for ev in lifecycle_events:
                log_fn(ev)
        except Exception:
            pass

        log_fn({
            "type": "broker_snapshot",
            "run_id": run_id,
            "symbol": cfg.symbol,
            "frequency": cfg.frequency,
            "bar_time": bar_time_iso,
            "where": "poll_cycle",
            "kill_switch_enabled": bool(is_kill_switch_enabled(live_dir)),
            "positions": [repr(p) for p in positions],
            "open_orders": [repr(o) for o in open_orders],
        })

        # --- Bracket check — only for the managed symbol ---
        bracket_warnings: list[dict] = []
        try:
            ib_trades = list(ib.trades()) if hasattr(ib, "trades") else []
            managed_positions = [p for p in positions if getattr(p, "symbol", "") == cfg.symbol]
            bracket_warnings = check_positions_have_brackets(
                managed_positions, open_orders,
                run_id=run_id, symbol=cfg.symbol, ib_trades=ib_trades,
            )
            for w in bracket_warnings:
                log_fn(w)
                alert_manager.add_alert(
                    key=f"missing_brackets_{cfg.symbol}",
                    severity=AlertSeverity.CRITICAL,
                    alert_type="missing_brackets",
                    message=f"Position {cfg.symbol} qty={w.get('quantity')} missing brackets",
                )
        except Exception:
            pass

        # --- Bracket auto-repair ---
        # If a position has been unprotected (no TP/SL) for two consecutive bar
        # cycles, close it at market immediately to cap the naked exposure.
        # One-bar grace period handles the race where the bracket fill lands
        # between our snapshot and the order going active.
        try:
            if bracket_warnings and bar_time_iso is not None:
                prev_unprotected = bar_tracker.get_unprotected_since(cfg.symbol)
                if prev_unprotected is not None:
                    # Second consecutive bar unprotected → emergency market close
                    closed_qty = 0.0
                    for p in managed_positions:
                        qty = float(getattr(p, "quantity", 0.0))
                        if abs(qty) < 0.01:
                            continue
                        close_side = Side.SELL if qty > 0 else Side.BUY
                        try:
                            broker.place_order(OrderRequest(
                                symbol=cfg.symbol,
                                side=close_side,
                                quantity=abs(qty),
                                order_type=OrderType.MARKET,
                            ))
                            closed_qty += abs(qty)
                        except Exception as close_exc:
                            log_fn({
                                "type": "bracket_autorepair_order_error",
                                "run_id": run_id,
                                "symbol": cfg.symbol,
                                "bar_time": bar_time_iso,
                                "error": repr(close_exc),
                            })
                    log_fn({
                        "type": "bracket_autorepair_closed",
                        "run_id": run_id,
                        "symbol": cfg.symbol,
                        "bar_time": bar_time_iso,
                        "first_unprotected_bar": prev_unprotected,
                        "closed_qty": closed_qty,
                    })
                    ts_print(
                        f"[poll] BRACKET AUTO-REPAIR: {cfg.symbol} naked position "
                        f"(unprotected since {prev_unprotected}), closed {closed_qty} shares at market."
                    )
                    alert_manager.add_alert(
                        key=f"autorepair_{cfg.symbol}",
                        severity=AlertSeverity.CRITICAL,
                        alert_type="bracket_autorepair",
                        message=(
                            f"{cfg.symbol} position closed at market by auto-repair "
                            f"(unprotected since {prev_unprotected})"
                        ),
                    )
                    bar_tracker.clear_unprotected(cfg.symbol)
                    # Keep has_open_position = True for this cycle so the strategy
                    # does NOT enter a new trade while the close order is still
                    # in-flight.  The next cycle's broker snapshot will confirm the
                    # position is flat and allow a fresh entry then.
                    has_open_position = True
                else:
                    # First bar unprotected — record it, give one more cycle
                    bar_tracker.mark_unprotected(cfg.symbol, bar_time_iso)
                    ts_print(
                        f"[poll] WARNING: {cfg.symbol} position missing brackets "
                        f"(bar {bar_time_iso}). Will auto-close next cycle if still unprotected."
                    )
            else:
                # Brackets healthy (or no position) — clear any stale flag
                bar_tracker.clear_unprotected(cfg.symbol)
        except Exception:
            pass

        # --- Predict + strategy ---
        predicted_price = predictor.update_and_predict(bar)
        current_price = float(bar["Close"])

        # Use the model's actual RMSE (in log-return space) as the uncertainty estimate.
        # Falls back to 0.5% of price if the metrics file was not available at load time.
        _rmse = predictor.model_rmse_logret if predictor.model_rmse_logret > 0 else 0.005
        model_error_sigma = max(1e-6, _rmse * current_price)
        atr = max(1e-6, current_price * 0.01)

        # Query real account equity and buying power from broker.
        real_equity = equity  # fallback to cfg.initial_equity
        buying_power: float | None = None
        try:
            acct = broker.get_account_summary() if hasattr(broker, "get_account_summary") else {}
            if acct:
                # Prefer NetLiquidation (total account value); fall back to EquityWithLoanValue.
                for key in ("NetLiquidation", "EquityWithLoanValue"):
                    if key in acct and acct[key] > 0:
                        real_equity = acct[key]
                        break
                # BuyingPower or AvailableFunds for margin constraint.
                for key in ("BuyingPower", "AvailableFunds"):
                    if key in acct and acct[key] > 0:
                        buying_power = acct[key]
                        break
                log_fn({
                    "type": "account_equity",
                    "run_id": run_id,
                    "symbol": cfg.symbol,
                    "real_equity": real_equity,
                    "buying_power": buying_power,
                    "cfg_initial_equity": float(equity),
                })
        except Exception:
            pass  # fall back to cfg.initial_equity

        state = StrategyState(
            current_price=current_price,
            predicted_price=float(predicted_price),
            model_error_sigma=float(model_error_sigma),
            atr=float(atr),
            account_equity=float(real_equity),
            has_open_position=bool(has_open_position),
            buying_power=buying_power,
        )

        plan = compute_tp_sl_and_size(state, strat_cfg)

        _pred_return = (float(predicted_price) / current_price) - 1.0 if current_price > 0 else 0.0
        _sigma_ret = float(model_error_sigma) / current_price if current_price > 0 else 0.0
        decision_event = {
            "type": "decision",
            "run_id": run_id,
            "symbol": cfg.symbol,
            "frequency": cfg.frequency,
            "bar_time": bar_time_iso,
            "close": current_price,
            "predicted_price": float(predicted_price),
            "predicted_return_pct": round(_pred_return * 100, 4),
            "sigma_return_pct": round(_sigma_ret * 100, 4),
            "long_threshold_pct": round((strat_cfg.k_sigma_long + strat_cfg.k_atr_long) * _sigma_ret * 100, 4),
            "has_open_position_before": bool(has_open_position),
            "no_trade_reason": (
                "open_position" if bool(has_open_position)
                else "signal_rejected" if plan is None
                else "trade_signal"
            ),
        }

        if plan is None:
            decision_event["action"] = "NO_TRADE"
            log_fn(decision_event)
            ts_print(
                f"[poll] {cfg.symbol} close={current_price:.2f} "
                f"pred={predicted_price:.2f} -> no trade"
            )
        elif suppress_trading:
            decision_event["action"] = "TRADE"
            decision_event["blocked_by_stale_bar"] = True
            log_fn(decision_event)
            ts_print("[poll] Stale bar (wake from sleep); skipping order submission.")
        elif is_kill_switch_enabled(live_dir):
            decision_event["action"] = "TRADE"
            decision_event["blocked_by_kill_switch"] = True
            log_fn(decision_event)
            ts_print("[poll] KILL SWITCH enabled; blocking order submission.")
        else:
            decision_event.update({
                "action": "TRADE",
                "direction": int(plan.direction),
                "size": float(plan.size),
                "tp_price": float(plan.tp_price),
                "sl_price": float(plan.sl_price),
            })
            log_fn(decision_event)

            try:
                bracket_ids = submit_trade_plan_bracket(
                    broker, plan, exec_ctx, current_price=current_price,
                )
                has_open_position = True

                log_fn({
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
                })

                actual_qty = max(1, int(round(plan.size)))
                ts_print(
                    f"[poll] TRADE {cfg.symbol} dir={plan.direction:+d} "
                    f"size={actual_qty} tp={plan.tp_price:.2f} sl={plan.sl_price:.2f}"
                )
            except Exception as exc:
                log_fn({
                    "type": "order_rejected",
                    "run_id": run_id,
                    "symbol": cfg.symbol,
                    "bar_time": bar_time_iso,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                })
                ts_print(f"[poll] Bracket order failed: {exc}")

        # --- Update status ---
        try:
            pos_qty = sum(float(getattr(p, "quantity", 0.0)) for p in positions)
            status_manager.update(
                state=SystemState.SLEEPING.value,
                connection_info={"uptime_minutes": None, "last_reconnect": None},
                last_bar_info={
                    "time": bar_time_iso,
                    "age_minutes": 0.0,
                    "expected_next": None,
                },
                position_info={
                    "status": "OPEN" if abs(pos_qty) >= 0.01 else "FLAT",
                    "quantity": pos_qty,
                },
                alert_count=alert_manager.count(),
                kill_switch_enabled=bool(is_kill_switch_enabled(live_dir)),
            )
        except Exception:
            pass

        state_machine.transition_to(SystemState.SLEEPING, reason="cycle_complete")
        return equity, has_open_position, order_state

    finally:
        cycle_timer.cancel()
        # Always disconnect
        try:
            if broker is not None:
                disc = getattr(broker, "disconnect", None)
                if callable(disc):
                    disc()
        except Exception:
            pass
        try:
            ib.disconnect()
        except Exception:
            pass
        ts_print("[poll] Disconnected.")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_poll_loop(cfg) -> None:
    """Run the poll-based live trading loop.

    Args:
        cfg: LiveSessionConfig or LiveEngineConfig (duck-typed; needs symbol,
             frequency, tsteps, backend, initial_equity, client_id, run_id,
             log_to_disk, log_dir).
    """
    if not _HAVE_IB:
        raise RuntimeError("ib_insync is required; pip install ib-insync")

    run_id = getattr(cfg, "run_id", None) or create_run_id()
    live_dir = Path(cfg.log_dir) if getattr(cfg, "log_dir", None) else None

    log_path: Path | None = None
    if getattr(cfg, "log_to_disk", True):
        log_path = make_log_path(
            symbol=cfg.symbol, frequency=cfg.frequency,
            run_id=run_id, live_dir=live_dir,
        )

    def _log(event: dict) -> None:
        if log_path is None:
            return
        try:
            append_event(log_path, event)
        except Exception:
            pass

    _log({
        "type": "run_start",
        "run_id": run_id,
        "symbol": cfg.symbol,
        "frequency": cfg.frequency,
        "mode": "poll",
        "backend": cfg.backend,
        "initial_equity": float(cfg.initial_equity),
        "kill_switch_enabled": bool(is_kill_switch_enabled(live_dir)),
    })

    bar_size_setting = _frequency_to_bar_size_setting(cfg.frequency)
    preferred_client_id = IbConfig.client_id if getattr(cfg, "client_id", None) is None else int(cfg.client_id)

    # Observability
    live_dir_path = Path(live_dir) if live_dir else (Path(__file__).resolve().parents[2] / "ui_state" / "live")
    status_manager = StatusManager(live_dir_path / "status.json")
    alert_manager = AlertManager(live_dir_path / "alerts.jsonl")
    bar_tracker = PersistentBarTracker(live_dir_path / "last_bar.json")

    state_machine = StateMachine(
        initial_state=SystemState.INITIALIZING,
        event_logger=_log,
        run_id=run_id,
    )

    # Predictor (load model once, reuse across cycles)
    predictor = LivePredictor.from_config(
        LivePredictorConfig(frequency=cfg.frequency, tsteps=cfg.tsteps),
    )
    historical_csv = get_hourly_data_csv_path(cfg.frequency)
    if os.path.exists(historical_csv):
        loaded = predictor.warmup_from_csv(historical_csv, cfg.frequency)
        ts_print(f"[poll] Predictor warmed up with {loaded} bars from {historical_csv}")

    strat_cfg = make_strategy_config_from_defaults()
    exec_ctx = ExecutionContext(symbol=cfg.symbol)

    equity = float(cfg.initial_equity)
    has_open_position = False
    order_state: dict = {}

    ts_print(
        f"[poll] Starting poll loop: {cfg.symbol} @ {cfg.frequency} "
        f"(bar_size={bar_size_setting})"
    )
    state_machine.transition_to(SystemState.SLEEPING, reason="initial")

    max_retries = 3
    _retry_base_delay = 30.0  # seconds; doubles each attempt (30 → 60)

    try:
        while True:
            # --- Wait for next bar boundary ---
            target = compute_next_bar_boundary(cfg.frequency)
            eastern = _get_eastern_tz()
            now = datetime.now(timezone.utc).astimezone(eastern)

            if target > now:
                wait_secs = (target - now).total_seconds()
                ts_print(
                    f"[poll] Next bar at {target.astimezone().strftime('%H:%M %Z')} "
                    f"(waiting {wait_secs / 60:.1f} min)"
                )

                # Tight poll loop — survives PC sleep/wake
                while datetime.now(timezone.utc).astimezone(eastern) < target:
                    time.sleep(30)

            # Detect wake-from-sleep: if we overshot target significantly,
            # the PC was likely sleeping.
            wake_now = datetime.now(timezone.utc).astimezone(eastern)
            overshoot_secs = (wake_now - target).total_seconds()
            if overshoot_secs > 60:
                ts_print(
                    f"[poll] Wake from sleep detected: overshot target by "
                    f"{overshoot_secs / 60:.1f} min"
                )

            # --- Skip if market was closed at the bar boundary ---
            # Use the *target* time (when the bar completed) so that bars
            # from market hours are still processed after PC wake.
            if not _is_trading_day(at=target) or not _is_market_hours(premarket=True, at=target):
                ts_print("[poll] Market closed at bar time, skipping cycle.")
                continue

            # --- Execute cycle with retries ---
            # Suppress trading when the bar is stale (PC slept past threshold).
            stale = overshoot_secs > _STALE_BAR_THRESHOLD_SECS

            success = False
            for attempt in range(1, max_retries + 1):
                try:
                    equity, has_open_position, order_state = _run_single_cycle(
                        cfg=cfg,
                        bar_size_setting=bar_size_setting,
                        preferred_client_id=preferred_client_id,
                        predictor=predictor,
                        strat_cfg=strat_cfg,
                        exec_ctx=exec_ctx,
                        bar_tracker=bar_tracker,
                        status_manager=status_manager,
                        alert_manager=alert_manager,
                        state_machine=state_machine,
                        log_fn=_log,
                        run_id=run_id,
                        live_dir=live_dir,
                        equity=equity,
                        has_open_position=has_open_position,
                        order_state=order_state,
                        suppress_trading=stale,
                    )
                    success = True
                    break
                except Exception as exc:
                    _log({
                        "type": "poll_cycle_error",
                        "run_id": run_id,
                        "attempt": attempt,
                        "max_retries": max_retries,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    })
                    ts_print(
                        f"[poll] Cycle failed (attempt {attempt}/{max_retries}): {exc}"
                    )
                    state_machine.transition_to(
                        SystemState.ERROR, reason=f"cycle_error_attempt_{attempt}",
                    )
                    if attempt < max_retries:
                        retry_delay = _retry_base_delay * (2 ** (attempt - 1))
                        time.sleep(retry_delay)

            if not success:
                ts_print("[poll] All retries exhausted, skipping to next bar.")
                state_machine.transition_to(
                    SystemState.SLEEPING, reason="retries_exhausted",
                )
                alert_manager.add_alert(
                    key="cycle_retries_exhausted",
                    severity=AlertSeverity.CRITICAL,
                    alert_type="cycle_retries_exhausted",
                    message=f"Bar cycle failed all {max_retries} retries — bar skipped",
                )

    except KeyboardInterrupt:
        ts_print("[poll] Interrupted by user.")
    finally:
        state_machine.transition_to(SystemState.STOPPED, reason="shutdown")
        _log({
            "type": "run_end",
            "run_id": run_id,
            "symbol": cfg.symbol,
            "frequency": cfg.frequency,
            "kill_switch_enabled": bool(is_kill_switch_enabled(live_dir)),
        })
        ts_print("[poll] Poll loop stopped.")
