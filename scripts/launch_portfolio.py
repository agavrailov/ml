#!/usr/bin/env python
"""Launch one ibkr_live_session daemon per symbol defined in configs/portfolio.json.

Usage:
    python -m scripts.launch_portfolio
    python -m scripts.launch_portfolio --dry-run   # print commands, don't launch

The launcher:
  - Reads configs/portfolio.json for symbol list, frequency, base_client_id
  - Assigns client IDs: base_client_id + index (e.g. NVDA=10, MSFT=11)
  - Launches each symbol as a subprocess
  - Monitors subprocesses; restarts any that exit with non-zero code (crash)
  - Shuts all down cleanly on Ctrl-C
"""
from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path

from src.portfolio.loader import load_portfolio_config
from src.config import BASE_DIR


def _build_command(symbol: str, frequency: str, client_id: int) -> list[str]:
    return [
        sys.executable, "-m", "src.ibkr_live_session",
        "--symbol", symbol,
        "--frequency", frequency,
        "--client-id", str(client_id),
    ]


def launch_portfolio(dry_run: bool = False) -> None:
    cfg = load_portfolio_config()
    symbols: list[str] = cfg["symbols"]
    frequency: str = cfg.get("frequency", "60min")
    base_cid: int = cfg.get("base_client_id", 10)

    procs: dict[str, subprocess.Popen] = {}
    commands: dict[str, list[str]] = {}

    for i, sym in enumerate(symbols):
        cid = base_cid + i
        cmd = _build_command(sym, frequency, cid)
        commands[sym] = cmd
        print(f"[launcher] {sym}  client_id={cid}  cmd: {' '.join(cmd)}")

    if dry_run:
        print("[launcher] dry-run: no processes started.")
        return

    def _shutdown(signum=None, frame=None):
        print("\n[launcher] Shutting down all daemons...")
        for sym, proc in procs.items():
            if proc.poll() is None:
                proc.terminate()
                print(f"[launcher]   {sym} terminated.")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Start all daemons
    for sym, cmd in commands.items():
        procs[sym] = subprocess.Popen(cmd, cwd=BASE_DIR)
        print(f"[launcher] Started {sym} (pid={procs[sym].pid})")

    # Monitor loop — restart crashed daemons with 30s delay
    while True:
        time.sleep(15)
        for sym, proc in list(procs.items()):
            ret = proc.poll()
            if ret is not None:
                print(f"[launcher] {sym} exited with code {ret}. Restarting in 30s...")
                time.sleep(30)
                procs[sym] = subprocess.Popen(commands[sym], cwd=BASE_DIR)
                print(f"[launcher] Restarted {sym} (pid={procs[sym].pid})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch all portfolio daemons from configs/portfolio.json")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without starting processes")
    args = parser.parse_args()
    launch_portfolio(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
