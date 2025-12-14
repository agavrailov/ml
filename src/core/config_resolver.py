"""Config resolution: merge code defaults with user overrides.

This module provides a clean separation between:
- Code defaults in src/config.py (paths, model hyperparameters, search spaces)
- User-mutable strategy overrides in configs/active.json (risk params, k_sigma, k_atr)

The UI writes only to configs/active.json, ensuring config.py remains read-only
and eliminating regex-based edits that were a major source of regressions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _get_repo_root() -> Path:
    """Return the repository root directory (ml_lstm/)."""
    # src/core/config_resolver.py -> src/core -> src -> repo_root
    return Path(__file__).resolve().parents[2]


def _get_active_config_path() -> Path:
    """Return the path to configs/active.json."""
    return _get_repo_root() / "configs" / "active.json"


def _load_active_config() -> dict[str, Any]:
    """Load user overrides from configs/active.json.

    Returns an empty dict if the file does not exist or is invalid.
    """
    path = _get_active_config_path()
    if not path.exists():
        return {}

    try:
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            return {}
        data = json.loads(content)
        if not isinstance(data, dict):
            return {}
        return data
    except (OSError, json.JSONDecodeError):
        return {}


def _save_active_config(config: dict[str, Any]) -> None:
    """Save user overrides to configs/active.json, creating the directory if needed."""
    path = _get_active_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def get_strategy_defaults() -> dict[str, float]:
    """Return effective strategy defaults by merging config.py + active.json.

    Priority:
    1. User overrides from configs/active.json (highest)
    2. Code defaults from src/config.py

    Returns a dict with keys:
        risk_per_trade_pct, reward_risk_ratio,
        k_sigma_long, k_sigma_short,
        k_atr_long, k_atr_short
    """
    # Import inside function to avoid circular imports at module load time.
    from src.config import STRATEGY_DEFAULTS

    active = _load_active_config()
    strategy_overrides = active.get("strategy", {})

    return {
        "risk_per_trade_pct": float(
            strategy_overrides.get("risk_per_trade_pct", STRATEGY_DEFAULTS.risk_per_trade_pct)
        ),
        "reward_risk_ratio": float(
            strategy_overrides.get("reward_risk_ratio", STRATEGY_DEFAULTS.reward_risk_ratio)
        ),
        "k_sigma_long": float(
            strategy_overrides.get("k_sigma_long", STRATEGY_DEFAULTS.k_sigma_long)
        ),
        "k_sigma_short": float(
            strategy_overrides.get("k_sigma_short", STRATEGY_DEFAULTS.k_sigma_short)
        ),
        "k_atr_long": float(
            strategy_overrides.get("k_atr_long", STRATEGY_DEFAULTS.k_atr_long)
        ),
        "k_atr_short": float(
            strategy_overrides.get("k_atr_short", STRATEGY_DEFAULTS.k_atr_short)
        ),
    }


def save_strategy_defaults(
    *,
    risk_per_trade_pct: float,
    reward_risk_ratio: float,
    k_sigma_long: float,
    k_sigma_short: float,
    k_atr_long: float,
    k_atr_short: float,
) -> None:
    """Save strategy parameter overrides to configs/active.json.

    Does NOT edit src/config.py (which remains read-only for the UI).
    """
    active = _load_active_config()
    active.setdefault("strategy", {})

    active["strategy"]["risk_per_trade_pct"] = float(risk_per_trade_pct)
    active["strategy"]["reward_risk_ratio"] = float(reward_risk_ratio)
    active["strategy"]["k_sigma_long"] = float(k_sigma_long)
    active["strategy"]["k_sigma_short"] = float(k_sigma_short)
    active["strategy"]["k_atr_long"] = float(k_atr_long)
    active["strategy"]["k_atr_short"] = float(k_atr_short)

    _save_active_config(active)
