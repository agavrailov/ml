"""Reconnection controller for IBKR live trading sessions.

Provides testable, standalone logic for connection management and reconnection
with exponential backoff, market hours awareness, and gateway restart detection.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from src.utils.timestamped_print import ts_print


@dataclass
class ReconnectConfig:
    """Configuration for reconnection logic."""

    host: str
    port: int
    preferred_client_id: int
    max_connect_tries: int = 25


class ConnectionManager:
    """Manages low-level IB connection with unique client ID discovery."""

    @staticmethod
    def connect_with_unique_client_id(
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


class ReconnectController:
    """Manages reconnection logic with market awareness and backoff."""

    def __init__(
        self,
        ib,  # noqa: ANN001
        config: ReconnectConfig,
        event_logger: Callable[[dict], None],
        run_id: str,
        alert_manager: object | None = None,
    ):
        """Initialize reconnect controller.

        Args:
            ib: ib_insync.IB instance
            config: Reconnection configuration
            event_logger: Function to log events (JSONL)
            run_id: Current run ID for logging
            alert_manager: Optional AlertManager for generating alerts
        """
        self._ib = ib
        self._config = config
        self._log = event_logger
        self._run_id = run_id
        self._alert_manager = alert_manager

        # Backoff state
        self._backoff = 1.0
        self._last_log_time = 0.0

        # Health monitoring
        self._failure_timestamps: list[float] = []
        self._restart_window_detected = False

    def reset_backoff(self) -> None:
        """Reset backoff to initial value (call after successful reconnect)."""
        self._backoff = 1.0

    def attempt_reconnect(
        self,
        *,
        is_market_hours_fn: Callable[[], bool],
        is_trading_day_fn: Callable[[], bool],
    ) -> tuple[bool, int | None]:
        """Attempt a single reconnection with appropriate backoff.

        Args:
            is_market_hours_fn: Function to check if market is currently open
            is_trading_day_fn: Function to check if today is a trading day

        Returns:
            Tuple of (success: bool, connected_client_id: int | None)
        """
        is_trading_day = is_trading_day_fn()
        is_market_time = is_market_hours_fn()

        # During completely closed market (weekends/holidays/night), use longer backoff
        if not is_trading_day or not is_market_time:
            # Market closed: exponential backoff 1-30 minutes
            off_hours_backoff = min(1800.0, max(60.0, self._backoff * 60.0))

            # Log at most once per 5 minutes during off hours to avoid spam
            now = time.time()
            if now - self._last_log_time >= 300.0:
                self._log(
                    {
                        "type": "reconnect_paused_off_hours",
                        "run_id": self._run_id,
                        "is_trading_day": is_trading_day,
                        "is_market_time": is_market_time,
                        "next_retry_seconds": off_hours_backoff,
                    }
                )
                ts_print(
                    f"[live] Market closed (trading_day={is_trading_day}, "
                    f"market_time={is_market_time}); will retry in {off_hours_backoff:.0f}s"
                )
                self._last_log_time = now

            time.sleep(off_hours_backoff)
            return (False, None)

        # Market hours or premarket: use normal backoff (1s -> 30s)
        time.sleep(self._backoff)

        try:
            connected_client_id = ConnectionManager.connect_with_unique_client_id(
                self._ib,
                host=self._config.host,
                port=self._config.port,
                preferred_client_id=self._config.preferred_client_id,
                max_tries=self._config.max_connect_tries,
            )

            self._log(
                {
                    "type": "broker_reconnected",
                    "run_id": self._run_id,
                    "host": self._config.host,
                    "port": int(self._config.port),
                    "client_id": int(connected_client_id),
                    "connection_time": time.time(),
                }
            )
            ts_print(f"[live] Reconnected to TWS (clientId={connected_client_id}).")

            return (True, connected_client_id)

        except Exception as exc:  # noqa: BLE001
            # Track failure for health monitoring
            now = time.time()
            self._failure_timestamps.append(now)
            # Keep only failures from last 2 minutes
            self._failure_timestamps[:] = [ts for ts in self._failure_timestamps if now - ts < 120.0]

            # Detect common non-transient errors that need longer backoff
            exc_str = str(exc).lower()
            error_repr = repr(exc).lower()

            # Check for TWS/Gateway not logged in or not ready
            is_not_logged_in = any(
                [
                    "not logged in" in exc_str,
                    "not logged in" in error_repr,
                    "not connected" in exc_str and "login" in exc_str,
                    "no security definition" in exc_str,  # Can occur when TWS not ready
                    "connection refused" in exc_str,
                    "connection reset" in exc_str,
                ]
            )

            # Check for Gateway restart indicators
            is_gateway_restart = any(
                [
                    "tws reset" in exc_str,
                    "server reset" in exc_str,
                    "reset by peer" in exc_str,
                    "connection aborted" in exc_str,
                ]
            ) or len(self._failure_timestamps) > 5  # >5 failures in 2 minutes

            # Detect typical Gateway restart window (11 PM - 12 AM EST on weekends)
            try:
                import zoneinfo

                eastern = zoneinfo.ZoneInfo("America/New_York")
                current_time = datetime.now(timezone.utc).astimezone(eastern)
                is_restart_window = (
                    current_time.weekday() in (5, 6) and 23 <= current_time.hour < 24  # Sat/Sun  # 11 PM - 12 AM
                )
            except Exception:
                is_restart_window = False

            # Handle Gateway restart scenario (detected or scheduled window)
            if is_gateway_restart or is_restart_window:
                # Gateway appears to be restarting: use very long backoff
                restart_backoff = 1800.0  # Fixed 30 min during restart

                if not self._restart_window_detected:
                    self._restart_window_detected = True
                    self._log(
                        {
                            "type": "gateway_restart_detected",
                            "run_id": self._run_id,
                            "error": repr(exc),
                            "failure_count_2min": len(self._failure_timestamps),
                            "is_restart_window": is_restart_window,
                            "is_restart_pattern": is_gateway_restart,
                            "next_retry_seconds": restart_backoff,
                        }
                    )
                    # Add INFO alert for gateway restart
                    if self._alert_manager is not None:
                        try:
                            from src.live.alerts import AlertSeverity
                            self._alert_manager.add_alert(
                                key="gateway_restart",
                                severity=AlertSeverity.INFO,
                                alert_type="gateway_restart",
                                message=f"IB Gateway restart detected (failures={len(self._failure_timestamps)}, window={is_restart_window})",
                            )
                        except Exception:
                            pass
                    ts_print(
                        f"[live] IB Gateway restart detected "
                        f"(failures={len(self._failure_timestamps)}, window={is_restart_window}). "
                        f"Will retry in {restart_backoff:.0f}s (30 min)."
                    )

                time.sleep(restart_backoff)
                self._backoff = 1.0  # Reset backoff after restart wait
                return (False, None)

            # If TWS not logged in (but not restarting), use longer backoff
            if is_not_logged_in:
                # Exponential backoff: 1 min -> 30 min
                login_backoff = min(1800.0, max(60.0, self._backoff * 10.0))

                # Log at most once per 5 minutes for login issues
                if now - self._last_log_time >= 300.0:
                    self._log(
                        {
                            "type": "reconnect_failed_not_logged_in",
                            "run_id": self._run_id,
                            "error": repr(exc),
                            "next_retry_seconds": login_backoff,
                        }
                    )
                    ts_print(
                        f"[live] TWS/Gateway not ready (not logged in or connection issue). "
                        f"Will retry in {login_backoff:.0f}s. Error: {exc}"
                    )
                    self._last_log_time = now

                time.sleep(login_backoff)
                # Increase backoff for next login attempt
                self._backoff = min(180.0, max(1.0, self._backoff * 1.8))
                return (False, None)

            # Normal reconnect failures: log during market hours only
            if is_trading_day and is_market_time:
                self._log(
                    {
                        "type": "reconnect_failed",
                        "run_id": self._run_id,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                ts_print(f"[live] Reconnect failed; will retry: {exc}")

            # Increase backoff for next attempt (capped at 30s during market hours)
            self._backoff = min(30.0, max(1.0, self._backoff * 1.8))
            return (False, None)
