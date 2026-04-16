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
from dataclasses import dataclass
from datetime import datetime, timezone


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


def _is_market_hours(*, premarket: bool = True) -> bool:
    """Check if current time is within US market trading hours (EST/EDT).

    Args:
        premarket: If True, includes pre-market hours (4:00 AM - 9:30 AM EST).
                  Regular market: 9:30 AM - 4:00 PM EST.

    Returns True during trading hours (Mon-Fri only), False otherwise.
    """
    try:
        import zoneinfo
    except ImportError:
        # Fallback if zoneinfo not available (Python < 3.9)
        try:
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
