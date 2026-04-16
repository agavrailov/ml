"""Load and validate configs/portfolio.json."""
from __future__ import annotations

import json
import os
from pathlib import Path

from src.config import BASE_DIR

_DEFAULT_PORTFOLIO_CONFIG_PATH = os.path.join(BASE_DIR, "configs", "portfolio.json")


def load_portfolio_config(path: str = _DEFAULT_PORTFOLIO_CONFIG_PATH) -> dict:
    """Load portfolio config. Raises FileNotFoundError if missing."""
    content = Path(path).read_text(encoding="utf-8")
    cfg = json.loads(content)
    # Basic validation
    if "symbols" not in cfg or not cfg["symbols"]:
        raise ValueError(f"portfolio.json must have a non-empty 'symbols' list: {path}")
    return cfg


def make_allocator_from_config(path: str = _DEFAULT_PORTFOLIO_CONFIG_PATH):
    """Construct a CapitalAllocator from portfolio.json."""
    from src.portfolio.capital_allocator import AllocationConfig, CapitalAllocator
    cfg = load_portfolio_config(path)
    alloc_cfg = cfg.get("allocation", {})
    return CapitalAllocator(AllocationConfig(
        symbols=cfg["symbols"],
        max_gross_exposure_pct=alloc_cfg.get("max_gross_exposure_pct", 0.80),
        max_per_symbol_pct=alloc_cfg.get("max_per_symbol_pct", 0.30),
    ))
