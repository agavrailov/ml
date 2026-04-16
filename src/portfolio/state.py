"""Shared portfolio state file.

Written by each symbol daemon after every cycle; read by sibling daemons
to compute available capital before sizing a new position.

State file schema (ui_state/portfolio/state.json):
    {
      "updated_utc": "2026-04-05T10:00:00Z",
      "total_equity": 50000.0,
      "positions": {
        "NVDA": {"quantity": 10, "market_value": 1500.0},
        "MSFT": {"quantity": 0, "market_value": 0.0}
      }
    }
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from src.portfolio.capital_allocator import CapitalAllocator


class PortfolioStateManager:
    """Thread-safe (file-atomic) read/write for portfolio state."""

    def __init__(self, state_path: str | Path) -> None:
        self._path = Path(state_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def read(self) -> dict:
        """Return current state dict, or empty template if file absent."""
        if not self._path.exists():
            return {"total_equity": 0.0, "positions": {}}
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return {"total_equity": 0.0, "positions": {}}

    def write_equity(self, total_equity: float) -> None:
        state = self.read()
        state["total_equity"] = total_equity
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        self._atomic_write(state)

    def write_position(self, symbol: str, quantity: float, market_value: float) -> None:
        state = self.read()
        state.setdefault("positions", {})[symbol.upper()] = {
            "quantity": quantity,
            "market_value": market_value,
        }
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        self._atomic_write(state)

    def available_capital_for(self, symbol: str, allocator: CapitalAllocator) -> float:
        """Ask the allocator how much capital `symbol` can use right now."""
        state = self.read()
        total_equity = float(state.get("total_equity", 0.0))
        open_market_values = {
            sym: float(pos.get("market_value", 0.0))
            for sym, pos in state.get("positions", {}).items()
        }
        return allocator.available_for(symbol, total_equity, open_market_values)

    def _atomic_write(self, state: dict) -> None:
        tmp = str(self._path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, str(self._path))
