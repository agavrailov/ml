"""IBKR/TWS live session runner.

This is a minimal wiring layer that:
- Connects to IBKR TWS / IB Gateway via ib_insync.
- Subscribes to a keepUpToDate historical bar stream (so we get OHLC bars).
- For each new completed bar, calls LivePredictor.update_and_predict(...).
- Feeds the result into the existing strategy/execution pipeline.

This is intentionally MVP-level:
- It uses placeholder ATR/sigma values (same pattern as trading_session.py).
- It only tracks "has_open_position" locally.
- It submits only the entry order (MARKET) via the configured broker.

Usage example:
    python -m src.ibkr_live_session --symbol NVDA --frequency 60min --backend IBKR_TWS

Note: requires ib_insync installed and a running TWS/IB Gateway.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import time
from typing import Optional

from src.config import IB as IbConfig
from src.execution import ExecutionContext, submit_trade_plan
from src.strategy import StrategyState, compute_tp_sl_and_size
from src.trading_session import make_strategy_config_from_defaults, make_broker
from src.live_predictor import LivePredictor, LivePredictorConfig

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


def _frequency_to_bar_size_setting(freq: str) -> str:
    f = freq.lower().strip()
    if f in {"15min", "15m"}:
        return "15 mins"
    if f in {"30min", "30m"}:
        return "30 mins"
    if f in {"60min", "60m", "1h", "1hr", "1hour"}:
        return "1 hour"
    if f in {"240min", "240m", "4h", "4hr", "4hour"}:
        return "4 hours"
    raise ValueError(f"Unsupported frequency for live bars: {freq!r}")


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


def run_live_session(cfg: LiveSessionConfig) -> None:
    if not _HAVE_IB:
        raise RuntimeError("ib_insync is required; install it with 'pip install ib-insync'")

    ib = IB()  # type: ignore[call-arg]

    preferred_client_id = IbConfig.client_id if cfg.client_id is None else int(cfg.client_id)
    print(
        f"[live] Connecting to TWS at {IbConfig.host}:{IbConfig.port} "
        f"(preferred clientId={preferred_client_id})",
    )

    connected_client_id = _connect_with_unique_client_id(
        ib,
        host=IbConfig.host,
        port=IbConfig.port,
        preferred_client_id=preferred_client_id,
    )
    print(f"[live] Connected (clientId={connected_client_id}).")

    try:
        contract = Stock(cfg.symbol, "SMART", "USD")  # type: ignore[call-arg]
        ib.qualifyContracts(contract)

        predictor = LivePredictor.from_config(
            LivePredictorConfig(frequency=cfg.frequency, tsteps=cfg.tsteps),
        )

        broker = make_broker(cfg.backend)
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

        print(f"[live] Subscribed to keepUpToDate bars ({bar_size_setting}) for {cfg.symbol}.")

        def _on_bar_update(bar_list, has_new_bar: bool) -> None:  # noqa: ANN001
            nonlocal equity, has_open_position

            # Only act on completed bars.
            if not has_new_bar:
                return
            if not bar_list:
                return

            b = bar_list[-1]
            bar = {
                "Time": getattr(b, "date", None),
                "Open": float(getattr(b, "open", 0.0)),
                "High": float(getattr(b, "high", 0.0)),
                "Low": float(getattr(b, "low", 0.0)),
                "Close": float(getattr(b, "close", 0.0)),
            }

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
            if plan is None:
                print(
                    f"[live] {cfg.symbol} bar close={current_price:.2f} pred={predicted_price:.2f} -> no trade"
                )
                return

            order_id = submit_trade_plan(broker, plan, exec_ctx)
            has_open_position = True
            print(
                f"[live] TRADE {cfg.symbol} dir={plan.direction:+d} size={plan.size:.2f} "
                f"tp={plan.tp_price:.2f} sl={plan.sl_price:.2f} order_id={order_id}"
            )

            if cfg.stop_after_first_trade:
                print("[live] stop_after_first_trade=True; stopping event loop.")
                ib.disconnect()

        bars.updateEvent += _on_bar_update

        print("[live] Running event loop (Ctrl+C to stop)...")
        ib.run()

    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


def main() -> None:
    p = argparse.ArgumentParser(description="IBKR/TWS live session runner (bar-by-bar model predictions).")
    p.add_argument("--symbol", type=str, default="NVDA")
    p.add_argument("--frequency", type=str, default="60min")
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
    args = p.parse_args()

    cfg = LiveSessionConfig(
        symbol=args.symbol,
        frequency=args.frequency,
        tsteps=int(args.tsteps),
        backend=args.backend,
        initial_equity=float(args.initial_equity),
        stop_after_first_trade=bool(args.stop_after_first_trade),
        client_id=args.client_id,
    )
    run_live_session(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
