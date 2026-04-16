"""Config resolution: merge code defaults with user overrides.

This module provides a clean separation between:
- Code defaults in src/config.py (paths, model hyperparameters, search spaces)
- User-mutable strategy overrides in configs/active.json (risk params, k_sigma, k_atr)

The UI writes only to configs/active.json, ensuring config.py remains read-only
and eliminating regex-based edits that were a major source of regressions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _get_repo_root() -> Path:
    """Return the repository root directory (ml_lstm/)."""
    # src/core/config_resolver.py -> src/core -> src -> repo_root
    return Path(__file__).resolve().parents[2]


def _get_active_config_path() -> Path:
    """Return the path to configs/active.json."""
    return _get_repo_root() / "configs" / "active.json"


def _try_load_json(path: str) -> dict | None:
    """Load a JSON file, returning None on any error or if absent."""
    try:
        content = Path(path).read_text(encoding="utf-8").strip()
        data = json.loads(content) if content else None
        return data if isinstance(data, dict) else None
    except Exception:
        return None


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


def get_strategy_defaults(symbol: str | None = None) -> dict[str, float | bool]:
    """Return effective strategy defaults by merging config.py + active.json.

    Resolution order:
      1. configs/symbols/{SYMBOL}/active.json  (if symbol is specified and file exists)
      2. configs/active.json                   (global fallback)
      3. src/config.py STRATEGY_DEFAULTS       (code defaults)

    Returns a dict with keys:
        risk_per_trade_pct, reward_risk_ratio,
        k_sigma_long, k_sigma_short,
        k_atr_long, k_atr_short,
        enable_longs, allow_shorts
    """
    # Import inside function to avoid circular imports at module load time.
    from src.config import STRATEGY_DEFAULTS

    # 1. Per-symbol config
    strategy_overrides: dict = {}
    if symbol:
        symbol_path = str(_get_repo_root() / "configs" / "symbols" / symbol.upper() / "active.json")
        data = _try_load_json(symbol_path)
        if data and "strategy" in data:
            strategy_overrides = data["strategy"]

    # 2. Global active.json fallback
    if not strategy_overrides:
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
        "enable_longs": bool(
            strategy_overrides.get("enable_longs", STRATEGY_DEFAULTS.enable_longs)
        ),
        "allow_shorts": bool(
            strategy_overrides.get("allow_shorts", STRATEGY_DEFAULTS.allow_shorts)
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
    enable_longs: bool | None = None,
    allow_shorts: bool | None = None,
    symbol: str | None = None,
    frequency: str | None = None,
    source: str | None = None,
) -> None:
    """Save strategy parameter overrides to configs/active.json.

    Does NOT edit src/config.py (which remains read-only for the UI).

    Optional metadata (symbol/frequency/source) is stored under "meta" so that
    promotions into production carry enough context to avoid mixing timeframes.
    """
    active = _load_active_config()
    active.setdefault("strategy", {})

    active["strategy"]["risk_per_trade_pct"] = float(risk_per_trade_pct)
    active["strategy"]["reward_risk_ratio"] = float(reward_risk_ratio)
    active["strategy"]["k_sigma_long"] = float(k_sigma_long)
    active["strategy"]["k_sigma_short"] = float(k_sigma_short)
    active["strategy"]["k_atr_long"] = float(k_atr_long)
    active["strategy"]["k_atr_short"] = float(k_atr_short)
    
    # Save trading mode if provided
    if enable_longs is not None:
        active["strategy"]["enable_longs"] = bool(enable_longs)
    if allow_shorts is not None:
        active["strategy"]["allow_shorts"] = bool(allow_shorts)

    # Best-effort metadata for safety/auditability.
    if symbol is not None or frequency is not None or source is not None:
        from datetime import datetime, timezone

        active.setdefault("meta", {})
        if symbol is not None:
            active["meta"]["symbol"] = str(symbol).strip().upper()
        if frequency is not None:
            active["meta"]["frequency"] = str(frequency).strip()
        if source is not None:
            active["meta"]["source"] = str(source).strip() or None
        active["meta"]["updated_at_utc"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    _save_active_config(active)
