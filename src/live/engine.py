"""Core live trading engine.

Provides the main trading loop logic with stable interfaces:
- Inputs: config dict or LiveEngineConfig dataclass
- Outputs: JSONL event log with typed events from src.live.contracts

The engine connects to IBKR/TWS, subscribes to live bars, runs predictions,
generates trading decisions, and submits orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.ibkr_live_session import LiveSessionConfig


@dataclass
class LiveEngineConfig:
    """Configuration for the live trading engine."""

    symbol: str = "NVDA"
    frequency: str = "60min"
    tsteps: int = 5
    backend: str = "IBKR_TWS"
    initial_equity: float = 10_000.0
    stop_after_first_trade: bool = False
    client_id: int | None = None
    run_id: str | None = None
    log_to_disk: bool = True
    log_dir: str | None = None
    snapshot_every_n_bars: int = 1

    @classmethod
    def from_dict(cls, config: dict) -> LiveEngineConfig:
        """Create config from dictionary."""
        return cls(
            symbol=config.get("symbol", "NVDA"),
            frequency=config.get("frequency", "60min"),
            tsteps=int(config.get("tsteps", 5)),
            backend=config.get("backend", "IBKR_TWS"),
            initial_equity=float(config.get("initial_equity", 10_000.0)),
            stop_after_first_trade=bool(config.get("stop_after_first_trade", False)),
            client_id=config.get("client_id"),
            run_id=config.get("run_id"),
            log_to_disk=bool(config.get("log_to_disk", True)),
            log_dir=config.get("log_dir"),
            snapshot_every_n_bars=int(config.get("snapshot_every_n_bars", 1)),
        )

    def to_legacy_config(self) -> LiveSessionConfig:
        """Convert to legacy LiveSessionConfig for backward compatibility."""
        return LiveSessionConfig(
            symbol=self.symbol,
            frequency=self.frequency,
            tsteps=self.tsteps,
            backend=self.backend,
            initial_equity=self.initial_equity,
            stop_after_first_trade=self.stop_after_first_trade,
            client_id=self.client_id,
            run_id=self.run_id,
            log_to_disk=self.log_to_disk,
            log_dir=self.log_dir,
            snapshot_every_n_bars=self.snapshot_every_n_bars,
        )


def run(config: LiveEngineConfig | dict, *, active_json_path: Path | None = None) -> None:
    """Run the live trading engine.

    Args:
        config: Engine configuration (dict or LiveEngineConfig)
        active_json_path: Optional path to configs/active.json for strategy overrides

    Uses the poll-based connect-on-demand architecture (src.live.poll_loop):
    for each bar cycle, connects to IBKR, fetches bars, runs prediction +
    strategy, executes trades, then disconnects.

    The engine respects the kill switch file (ui_state/live/KILL_SWITCH) to
    prevent new order submissions without stopping the daemon.
    """
    if isinstance(config, dict):
        cfg = LiveEngineConfig.from_dict(config)
    else:
        cfg = config

    from src.live.poll_loop import run_poll_loop

    legacy_cfg = cfg.to_legacy_config()
    run_poll_loop(legacy_cfg)
