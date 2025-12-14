"""UI state management for Streamlit app.

Provides centralized access to session state and JSON-based history persistence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st


# Constants
UI_STATE_DIR = Path(__file__).resolve().parents[2] / "ui_state"
MAX_HISTORY_ROWS = 100


def get_ui_state() -> dict[str, Any]:
    """Return the centralized UI state dict, initializing if needed.

    This provides a single source of truth for long-lived state that may be
    persisted to JSON. The dict has well-known top-level keys for each tab.
    """
    if "ui_state" not in st.session_state:
        st.session_state["ui_state"] = {
            "data": {},
            "experiments": {},
            "training": {},
            "strategy": {},
            "backtests": {},
            "optimization": {},
            "walkforward": {},
        }
    return st.session_state["ui_state"]


def load_history(filename: str) -> list[dict]:
    """Load a history list from JSON in UI_STATE_DIR, returning [] if missing."""
    path = UI_STATE_DIR / filename
    if not path.exists():
        return []

    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def save_history(filename: str, history: list[dict]) -> None:
    """Save a history list to JSON in UI_STATE_DIR, creating the dir if needed."""
    UI_STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = UI_STATE_DIR / filename
    try:
        # Truncate to MAX_HISTORY_ROWS if needed (keep most recent).
        if len(history) > MAX_HISTORY_ROWS:
            history = history[-MAX_HISTORY_ROWS:]
        path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - best-effort persistence
        st.error(f"Failed to save history to {filename}: {exc}")


# Aliases for backward compatibility with tests
load_json_history = load_history
save_json_history = save_history
