"""Persistent state management for live trading sessions.

Provides state that survives restarts, such as last processed bar for deduplication.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class PersistentBarTracker:
    """Tracks last processed bar across restarts for deduplication.

    Also tracks unprotected positions (missing brackets) for auto-repair.
    State JSON schema:
        {
            "time": "<iso>",
            "hash": "<8-char-md5>",
            "unprotected_since": {"NVDA": "<iso-of-first-unprotected-bar>"}
        }
    """

    def __init__(self, state_file_path: Path):
        """Initialize bar tracker.

        Args:
            state_file_path: Path to state file (e.g., last_bar.json)
        """
        self._path = state_file_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._last_processed = self._load()

    def mark_processed(self, bar_time_iso: str, bar_hash: str) -> None:
        """Mark a bar as processed.

        Args:
            bar_time_iso: Bar timestamp in ISO format
            bar_hash: Hash of bar OHLC data
        """
        state = self._last_processed or {}
        state["time"] = bar_time_iso
        state["hash"] = bar_hash
        self._last_processed = state
        self._save()

    def is_duplicate(self, bar_time_iso: str, bar_hash: str) -> bool:
        """Check if bar has already been processed.

        Args:
            bar_time_iso: Bar timestamp in ISO format
            bar_hash: Hash of bar OHLC data

        Returns:
            True if this bar was already processed
        """
        if self._last_processed is None:
            return False
        return (
            self._last_processed.get("time") == bar_time_iso
            and self._last_processed.get("hash") == bar_hash
        )

    # ------------------------------------------------------------------
    # Unprotected-position tracking (bracket auto-repair)
    # ------------------------------------------------------------------

    def mark_unprotected(self, symbol: str, bar_time_iso: str) -> None:
        """Record that a position was found without bracket orders.

        Called when bracket check fails. If the symbol is already marked,
        the original timestamp is preserved (only the first detection matters).

        Args:
            symbol: Trading symbol (e.g., "NVDA")
            bar_time_iso: Bar timestamp when the missing bracket was first seen
        """
        state = self._last_processed or {}
        unprotected = dict(state.get("unprotected_since") or {})
        if symbol not in unprotected:
            unprotected[symbol] = bar_time_iso
        state["unprotected_since"] = unprotected
        self._last_processed = state
        self._save()

    def clear_unprotected(self, symbol: str) -> None:
        """Clear unprotected status for a symbol (brackets are healthy again).

        Args:
            symbol: Trading symbol to clear
        """
        state = self._last_processed or {}
        unprotected = dict(state.get("unprotected_since") or {})
        unprotected.pop(symbol, None)
        state["unprotected_since"] = unprotected
        self._last_processed = state
        self._save()

    def get_unprotected_since(self, symbol: str) -> str | None:
        """Return the bar timestamp when unprotection was first detected, or None.

        Args:
            symbol: Trading symbol to query

        Returns:
            ISO timestamp string if unprotected, None if brackets are healthy
        """
        if self._last_processed is None:
            return None
        return (self._last_processed.get("unprotected_since") or {}).get(symbol)

    def _save(self) -> None:
        """Save state to disk (best-effort)."""
        try:
            self._path.write_text(json.dumps(self._last_processed), encoding="utf-8")
        except Exception:
            # Never crash trading loop on persistence failure
            pass

    def _load(self) -> dict | None:
        """Load state from disk (best-effort).

        Returns:
            Loaded state dict or None if file doesn't exist/is invalid
        """
        if not self._path.exists():
            return None
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return None


def compute_bar_hash(bar) -> str:  # noqa: ANN001
    """Compute hash of bar OHLC data for deduplication.

    Args:
        bar: Bar dict with Open, High, Low, Close keys OR IB BarData object

    Returns:
        Short hash string (8 hex chars)
    """
    # Handle both dict and IB BarData object
    if isinstance(bar, dict):
        ohlc_str = f"{bar['Open']}{bar['High']}{bar['Low']}{bar['Close']}"
    else:
        # IB BarData object with attributes
        ohlc_str = f"{getattr(bar, 'open', 0)}{getattr(bar, 'high', 0)}{getattr(bar, 'low', 0)}{getattr(bar, 'close', 0)}"
    return hashlib.md5(ohlc_str.encode()).hexdigest()[:8]
