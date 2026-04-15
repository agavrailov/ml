# Multi-Asset Portfolio Trading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the single-NVDA trading system to trade a configurable list of symbols simultaneously, with a shared portfolio risk layer that enforces per-symbol capital caps and gross exposure limits.

**Architecture:** One poll-loop daemon per symbol (each an independent process); a lightweight portfolio state file (`ui_state/portfolio/state.json`) is read by each daemon every cycle to get its capital allocation. A new `src/portfolio/` package owns capital allocation and exposure tracking. Model registry, data pipeline, and config are parameterized by symbol.

**Tech Stack:** Python 3.11, ib_insync, Keras/TensorFlow, pandas, existing broker/strategy abstractions.

---

## Phase overview

| Phase | What it delivers | Can ship independently? |
|-------|-----------------|------------------------|
| 1 | Symbol-parameterized infrastructure | Yes — no behaviour change, NVDA still works |
| 2 | Portfolio risk layer | Yes — pure library code, fully tested in isolation |
| 3 | Multi-daemon orchestration | Yes — adds MSFT/AAPL trading |
| 4 | Tooling: auto-retrain + UI | Yes — observability and automation |

---

## File map

### New files
| Path | Purpose |
|------|---------|
| `src/model_registry.py` | Symbol-keyed model lookup; wraps `models/symbol_registry.json` |
| `models/symbol_registry.json` | `{SYMBOL: {freq: {tsteps: {metrics}}}}` — replaces NVDA-only `best_hyperparameters.json` |
| `src/portfolio/__init__.py` | Package stub |
| `src/portfolio/capital_allocator.py` | Per-symbol capital limits; returns `available_for(symbol)` |
| `src/portfolio/exposure_tracker.py` | Reads broker positions; computes gross/net exposure |
| `src/portfolio/state.py` | Reads/writes `ui_state/portfolio/state.json` |
| `configs/portfolio.json` | Symbol list, frequency, base client ID |
| `configs/symbols/NVDA/active.json` | Per-symbol strategy config (migrated from `configs/active.json`) |
| `scripts/launch_portfolio.py` | Launches one daemon per symbol; manages subprocesses |
| `deploy/nssm_install_portfolio.bat` | Windows service for the portfolio launcher |
| `tests/test_model_registry.py` | Unit tests for model_registry.py |
| `tests/test_portfolio_capital_allocator.py` | Unit tests for capital_allocator.py |
| `tests/test_portfolio_exposure_tracker.py` | Unit tests for exposure_tracker.py |
| `tests/test_portfolio_state.py` | Unit tests for portfolio state I/O |

### Modified files
| Path | Change |
|------|--------|
| `src/config.py` | Add `symbol` param to `PathsConfig.hourly_data_csv`, `raw_data_csv`, `scaler_params_json`; add `CONTRACT_REGISTRY` dict; update module-level helpers |
| `src/daily_data_agent.py` | Accept `symbols: list[str]` param; loop over symbols |
| `src/train.py` | Accept `symbol` param; call `model_registry.update_best_model()` |
| `src/live/poll_loop.py` | Read `ui_state/portfolio/state.json` to get `available_capital` before sizing |
| `src/core/config_resolver.py` | Load per-symbol config from `configs/symbols/{SYMBOL}/active.json` |
| `scripts/auto_retrain.py` | Accept `--symbols` list; loop + skip symbols with no data |

---

## Phase 1 — Symbol-parameterized infrastructure

### Task 1: Symbol param on data path helpers in `config.py`

**Files:**
- Modify: `src/config.py`
- Modify: `src/data.py` (if `load_hourly_ohlc` calls `get_hourly_data_csv_path`)
- Test: `tests/test_config_paths.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_config_paths.py  (add to existing file)
from src.config import get_hourly_data_csv_path, get_scaler_params_json_path

def test_hourly_csv_default_symbol_is_nvda():
    path = get_hourly_data_csv_path("60min")
    assert "nvda_60min" in path.lower()

def test_hourly_csv_custom_symbol():
    path = get_hourly_data_csv_path("60min", symbol="MSFT")
    assert "msft_60min" in path.lower()
    assert "nvda" not in path.lower()

def test_scaler_params_custom_symbol():
    path = get_scaler_params_json_path("60min", symbol="MSFT")
    assert "msft" in path.lower()
```

- [ ] **Step 2: Run to confirm failures**
```
pytest tests/test_config_paths.py::test_hourly_csv_custom_symbol -v
```
Expected: FAILED — `get_hourly_data_csv_path` doesn't accept `symbol` kwarg.

- [ ] **Step 3: Update `PathsConfig` in `src/config.py`**

Replace these two methods in the `PathsConfig` dataclass:
```python
def raw_data_csv(self, symbol: str = "NVDA") -> str:
    return os.path.join(self.raw_data_dir, f"{symbol.lower()}_minute.csv")

def hourly_data_csv(self, frequency: str, symbol: str = "NVDA") -> str:
    return os.path.join(self.processed_data_dir, f"{symbol.lower()}_{frequency}.csv")

def scaler_params_json(self, frequency: str, symbol: str = "NVDA") -> str:
    return os.path.join(self.processed_data_dir, f"scaler_params_{symbol.lower()}_{frequency}.json")
```

Update the module-level helpers (find by name; they delegate to `PATHS`):
```python
def get_hourly_data_csv_path(frequency, symbol="NVDA", processed_data_dir=PROCESSED_DATA_DIR):
    return PATHS.hourly_data_csv(frequency, symbol)

def get_scaler_params_json_path(frequency, symbol="NVDA", processed_data_dir=PROCESSED_DATA_DIR):
    return PATHS.scaler_params_json(frequency, symbol)
```

Also add module-level constant (used by multi-symbol launchers):
```python
# After the existing PATHS = PathsConfig() line
RAW_DATA_CSV_NVDA = PATHS.raw_data_csv("NVDA")  # keeps legacy alias working
```

- [ ] **Step 4: Run tests**
```
pytest tests/test_config_paths.py -v
```
Expected: all pass.

- [ ] **Step 5: Verify no regressions**
```
pytest tests/ -q --ignore=tests/test_app_strategy_params.py -x
```
Expected: same pass count as before this task.

- [ ] **Step 6: Commit**
```bash
git add src/config.py tests/test_config_paths.py
git commit -m "feat(config): add symbol param to data path helpers (default=NVDA)"
```

---

### Task 2: Contract registry in `config.py`

**Files:**
- Modify: `src/config.py`
- Test: `tests/test_config_paths.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_config_paths.py (add)
from src.config import get_contract_details

def test_get_contract_details_nvda():
    cd = get_contract_details("NVDA")
    assert cd["symbol"] == "NVDA"
    assert cd["exchange"] == "SMART"

def test_get_contract_details_unknown_raises():
    import pytest
    with pytest.raises(KeyError):
        get_contract_details("ZZZZ")
```

- [ ] **Step 2: Run to confirm failures**
```
pytest tests/test_config_paths.py::test_get_contract_details_nvda -v
```
Expected: FAILED — `get_contract_details` not defined.

- [ ] **Step 3: Add registry to `src/config.py`**

Find `NVDA_CONTRACT_DETAILS` in config.py and replace/extend it:
```python
# Contract registry — add new symbols here
CONTRACT_REGISTRY: dict[str, dict] = {
    "NVDA": {
        "symbol": "NVDA",
        "exchange": "SMART",
        "currency": "USD",
        "sec_type": "STK",
    },
    "MSFT": {
        "symbol": "MSFT",
        "exchange": "SMART",
        "currency": "USD",
        "sec_type": "STK",
    },
    "AAPL": {
        "symbol": "AAPL",
        "exchange": "SMART",
        "currency": "USD",
        "sec_type": "STK",
    },
}

# Legacy alias — keeps existing code that uses NVDA_CONTRACT_DETAILS working
NVDA_CONTRACT_DETAILS = CONTRACT_REGISTRY["NVDA"]


def get_contract_details(symbol: str) -> dict:
    """Return contract details for a symbol. Raises KeyError if unknown."""
    return CONTRACT_REGISTRY[symbol.upper()]
```

- [ ] **Step 4: Run tests**
```
pytest tests/test_config_paths.py -v
```
Expected: all pass.

- [ ] **Step 5: Commit**
```bash
git add src/config.py tests/test_config_paths.py
git commit -m "feat(config): add CONTRACT_REGISTRY and get_contract_details()"
```

---

### Task 3: Multi-symbol data pipeline in `daily_data_agent.py`

**Files:**
- Modify: `src/daily_data_agent.py`
- Test: `tests/test_data_processing.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_data_processing.py (add)
import os
from unittest.mock import patch, MagicMock

def test_run_daily_pipeline_loops_over_symbols(tmp_path):
    """Pipeline should call resample for each requested symbol."""
    with patch("src.daily_data_agent.run_transform_minute_bars") as mock_transform, \
         patch("src.daily_data_agent.clean_raw_minute_data"), \
         patch("src.daily_data_agent.smart_fill_gaps"), \
         patch("src.daily_data_agent.run_gap_analysis", return_value=[]), \
         patch("os.path.exists", return_value=True):
        from src.daily_data_agent import run_daily_pipeline
        run_daily_pipeline(skip_ingestion=True, symbols=["NVDA", "MSFT"])
    calls = [c.args[0] for c in mock_transform.call_args_list]
    assert "NVDA" in calls
    assert "MSFT" in calls
```

- [ ] **Step 2: Run to confirm failure**
```
pytest tests/test_data_processing.py::test_run_daily_pipeline_loops_over_symbols -v
```
Expected: FAILED — `run_daily_pipeline` has no `symbols` param.

- [ ] **Step 3: Update `run_daily_pipeline` signature in `src/daily_data_agent.py`**

Change the function signature:
```python
def run_daily_pipeline(
    skip_ingestion: bool = False,
    symbols: list[str] | None = None,
) -> None:
```

At the top of the function body, add:
```python
    if symbols is None:
        symbols = ["NVDA"]
```

Find the hardcoded `run_transform_minute_bars("NVDA")` line and replace with:
```python
    for sym in symbols:
        print(f"[agent] Processing symbol: {sym}")
        run_transform_minute_bars(sym)
```

- [ ] **Step 4: Run tests**
```
pytest tests/test_data_processing.py -v
```
Expected: all pass including new test.

- [ ] **Step 5: Verify `run_transform_minute_bars` accepts symbol**

Check the implementation of `run_transform_minute_bars` — it likely already accepts a symbol arg since it was just hardcoded at the call site. If it uses the old `get_hourly_data_csv_path(frequency)` (without symbol), update it to call `get_hourly_data_csv_path(frequency, symbol=symbol)`.

- [ ] **Step 6: Commit**
```bash
git add src/daily_data_agent.py tests/test_data_processing.py
git commit -m "feat(data): multi-symbol data pipeline (symbols=['NVDA'] default)"
```

---

### Task 4: Symbol-keyed model registry (`src/model_registry.py`)

**Files:**
- Create: `src/model_registry.py`
- Create: `tests/test_model_registry.py`

This module replaces the scattered `best_hyperparameters.json` reads. It becomes the single source of truth for "which model is active for symbol X at frequency Y".

- [ ] **Step 1: Write failing tests**

```python
# tests/test_model_registry.py
import json
import os
import pytest
from pathlib import Path

@pytest.fixture
def registry_path(tmp_path):
    return tmp_path / "symbol_registry.json"

def test_get_best_model_empty_registry(registry_path):
    from src.model_registry import get_best_model_path
    result = get_best_model_path("NVDA", "60min", 5, registry_path=str(registry_path))
    assert result is None

def test_update_then_get(registry_path, tmp_path):
    from src.model_registry import update_best_model, get_best_model_path
    model_file = tmp_path / "my_model.keras"
    model_file.write_text("")
    update_best_model(
        symbol="NVDA",
        frequency="60min",
        tsteps=5,
        val_loss=0.0001,
        model_path=str(model_file),
        bias_path=None,
        hparams={"lstm_units": 64, "n_lstm_layers": 1},
        registry_path=str(registry_path),
    )
    result = get_best_model_path("NVDA", "60min", 5, registry_path=str(registry_path))
    assert result == str(model_file)

def test_update_only_replaces_if_better(registry_path, tmp_path):
    from src.model_registry import update_best_model, get_best_model_path
    model_a = tmp_path / "model_a.keras"
    model_b = tmp_path / "model_b.keras"
    model_a.write_text("")
    model_b.write_text("")
    update_best_model("NVDA", "60min", 5, 0.0001, str(model_a), None, {}, registry_path=str(registry_path))
    update_best_model("NVDA", "60min", 5, 0.0002, str(model_b), None, {}, registry_path=str(registry_path))  # worse
    result = get_best_model_path("NVDA", "60min", 5, registry_path=str(registry_path))
    assert result == str(model_a)  # still model_a

def test_migrate_from_legacy(registry_path, tmp_path):
    from src.model_registry import migrate_from_legacy_hps, get_best_model_path
    legacy = tmp_path / "best_hyperparameters.json"
    model_file = tmp_path / "my_model.keras"
    model_file.write_text("")
    legacy.write_text(json.dumps({
        "60min": {"5": {"validation_loss": 0.00005, "model_filename": model_file.name}}
    }))
    migrate_from_legacy_hps(
        legacy_hps_path=str(legacy),
        symbol="NVDA",
        model_registry_dir=str(tmp_path),
        registry_path=str(registry_path),
    )
    result = get_best_model_path("NVDA", "60min", 5, registry_path=str(registry_path))
    assert result == str(model_file)
```

- [ ] **Step 2: Run to confirm failures**
```
pytest tests/test_model_registry.py -v
```
Expected: 4 FAILED — module not found.

- [ ] **Step 3: Implement `src/model_registry.py`**

```python
"""Symbol-keyed model registry.

Single source of truth for which model is active per (symbol, frequency, tsteps).
Backed by ``models/symbol_registry.json``.

Schema::

    {
      "NVDA": {
        "60min": {
          "5": {
            "validation_loss": 6.02e-05,
            "model_path": "models/registry/my_lstm_model_60min_tsteps5_20260402.keras",
            "bias_path": null,
            "hparams": {"lstm_units": 64, ...},
            "promoted_at_utc": "2026-04-05T02:00:00Z"
          }
        }
      }
    }
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import BASE_DIR, MODEL_REGISTRY_DIR

_DEFAULT_REGISTRY_PATH = os.path.join(BASE_DIR, "models", "symbol_registry.json")


def _load(registry_path: str) -> dict:
    if not os.path.exists(registry_path):
        return {}
    try:
        content = Path(registry_path).read_text(encoding="utf-8").strip()
        return json.loads(content) if content else {}
    except Exception:
        return {}


def _save(data: dict, registry_path: str) -> None:
    tmp = registry_path + ".tmp"
    Path(tmp).parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, registry_path)


def get_best_model_path(
    symbol: str,
    frequency: str,
    tsteps: int,
    *,
    registry_path: str = _DEFAULT_REGISTRY_PATH,
) -> str | None:
    """Return the absolute model path for (symbol, frequency, tsteps), or None."""
    data = _load(registry_path)
    try:
        entry = data[symbol.upper()][frequency][str(tsteps)]
        path = entry.get("model_path")
        return path if path and os.path.exists(path) else None
    except KeyError:
        return None


def get_best_model_entry(
    symbol: str,
    frequency: str,
    tsteps: int,
    *,
    registry_path: str = _DEFAULT_REGISTRY_PATH,
) -> dict | None:
    """Return the full registry entry, or None."""
    data = _load(registry_path)
    try:
        return dict(data[symbol.upper()][frequency][str(tsteps)])
    except KeyError:
        return None


def update_best_model(
    symbol: str,
    frequency: str,
    tsteps: int,
    val_loss: float,
    model_path: str,
    bias_path: str | None,
    hparams: dict[str, Any],
    *,
    registry_path: str = _DEFAULT_REGISTRY_PATH,
    force: bool = False,
) -> bool:
    """Write a new model entry if it improves on the current best val_loss.

    Args:
        force: If True, always write regardless of val_loss comparison.

    Returns:
        True if the entry was written, False if the existing entry was better.
    """
    data = _load(registry_path)
    sym = symbol.upper()
    data.setdefault(sym, {}).setdefault(frequency, {})

    existing = data[sym][frequency].get(str(tsteps), {})
    existing_loss = existing.get("validation_loss", float("inf"))

    if not force and val_loss >= existing_loss:
        return False

    data[sym][frequency][str(tsteps)] = {
        "validation_loss": val_loss,
        "model_path": model_path,
        "bias_path": bias_path,
        "hparams": hparams,
        "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _save(data, registry_path)
    return True


def migrate_from_legacy_hps(
    *,
    legacy_hps_path: str,
    symbol: str = "NVDA",
    model_registry_dir: str = MODEL_REGISTRY_DIR,
    registry_path: str = _DEFAULT_REGISTRY_PATH,
) -> int:
    """One-time migration: copy best_hyperparameters.json into symbol_registry.json.

    Returns number of entries migrated.
    """
    if not os.path.exists(legacy_hps_path):
        return 0
    try:
        content = Path(legacy_hps_path).read_text(encoding="utf-8").strip()
        hps = json.loads(content) if content else {}
    except Exception:
        return 0

    count = 0
    for freq, tsteps_dict in (hps or {}).items():
        if not isinstance(tsteps_dict, dict):
            continue
        for tsteps_str, metrics in tsteps_dict.items():
            if not isinstance(metrics, dict):
                continue
            model_filename = metrics.get("model_filename")
            if not model_filename:
                continue
            model_path = os.path.join(model_registry_dir, model_filename)
            bias_filename = metrics.get("bias_correction_filename")
            bias_path = (
                os.path.join(model_registry_dir, bias_filename) if bias_filename else None
            )
            val_loss = float(metrics.get("validation_loss", float("inf")))
            hparams = {k: v for k, v in metrics.items()
                       if k not in ("validation_loss", "model_filename",
                                    "bias_correction_filename")}
            update_best_model(
                symbol=symbol,
                frequency=freq,
                tsteps=int(tsteps_str),
                val_loss=val_loss,
                model_path=model_path,
                bias_path=bias_path,
                hparams=hparams,
                registry_path=registry_path,
                force=True,
            )
            count += 1
    return count
```

- [ ] **Step 4: Run tests**
```
pytest tests/test_model_registry.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Run migration script once to seed `models/symbol_registry.json`**
```bash
python -c "
from src.model_registry import migrate_from_legacy_hps
n = migrate_from_legacy_hps(legacy_hps_path='best_hyperparameters.json')
print(f'Migrated {n} entries')
"
```
Expected: `Migrated 1 entries` (or however many frequency/tstep combos exist).

- [ ] **Step 6: Commit**
```bash
git add src/model_registry.py tests/test_model_registry.py models/symbol_registry.json
git commit -m "feat: symbol-keyed model registry; migrate NVDA from best_hyperparameters.json"
```

---

## Phase 2 — Portfolio risk layer

### Task 5: Capital allocator (`src/portfolio/capital_allocator.py`)

**Files:**
- Create: `src/portfolio/__init__.py`
- Create: `src/portfolio/capital_allocator.py`
- Create: `tests/test_portfolio_capital_allocator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_portfolio_capital_allocator.py
from src.portfolio.capital_allocator import AllocationConfig, CapitalAllocator

def test_available_when_no_positions():
    cfg = AllocationConfig(symbols=["NVDA", "MSFT"], max_gross_exposure_pct=0.8, max_per_symbol_pct=0.3)
    alloc = CapitalAllocator(cfg)
    avail = alloc.available_for("NVDA", total_equity=10_000.0, open_market_values={})
    assert avail == pytest.approx(3_000.0)  # 30% of 10k

def test_blocked_when_symbol_at_cap():
    cfg = AllocationConfig(symbols=["NVDA", "MSFT"], max_gross_exposure_pct=0.8, max_per_symbol_pct=0.3)
    alloc = CapitalAllocator(cfg)
    avail = alloc.available_for(
        "NVDA", total_equity=10_000.0,
        open_market_values={"NVDA": 3_100.0}  # already over per-symbol cap
    )
    assert avail == 0.0

def test_gross_exposure_limits_remaining():
    cfg = AllocationConfig(symbols=["NVDA", "MSFT"], max_gross_exposure_pct=0.8, max_per_symbol_pct=0.5)
    alloc = CapitalAllocator(cfg)
    # MSFT already uses 70% of equity — only 10% gross headroom remains
    avail = alloc.available_for(
        "NVDA", total_equity=10_000.0,
        open_market_values={"MSFT": 7_000.0}
    )
    assert avail == pytest.approx(1_000.0)  # min(5k per-sym cap, 1k gross headroom)

def test_unknown_symbol_returns_zero():
    cfg = AllocationConfig(symbols=["NVDA"], max_gross_exposure_pct=0.8, max_per_symbol_pct=0.3)
    alloc = CapitalAllocator(cfg)
    avail = alloc.available_for("ZZZZ", total_equity=10_000.0, open_market_values={})
    assert avail == 0.0

import pytest  # noqa: E402 (keep at bottom so fixture above works)
```

- [ ] **Step 2: Run to confirm failures**
```
pytest tests/test_portfolio_capital_allocator.py -v
```
Expected: 4 FAILED — module not found.

- [ ] **Step 3: Implement**

```python
# src/portfolio/__init__.py
```

```python
# src/portfolio/capital_allocator.py
"""Per-symbol capital allocation with gross-exposure guardrails."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AllocationConfig:
    symbols: list[str]
    max_gross_exposure_pct: float = 0.80  # max fraction of equity deployed across all positions
    max_per_symbol_pct: float = 0.30      # max fraction of equity in any single symbol


class CapitalAllocator:
    """Computes how much capital a symbol may use for a new position.

    Decision logic (returns the minimum of three constraints):
      1. Per-symbol cap: symbol_cap - current_symbol_exposure
      2. Gross cap:      gross_cap - total_open_exposure
      3. Zero if symbol is not in the configured symbol list
    """

    def __init__(self, config: AllocationConfig) -> None:
        self._cfg = config

    def available_for(
        self,
        symbol: str,
        total_equity: float,
        open_market_values: dict[str, float],
    ) -> float:
        """Return capital available for a new position in `symbol`.

        Args:
            symbol: Symbol to evaluate.
            total_equity: Current net liquidation value of the account.
            open_market_values: {symbol: current_market_value} for all open positions.

        Returns:
            Non-negative float — the maximum capital that can be deployed.
        """
        if symbol.upper() not in [s.upper() for s in self._cfg.symbols]:
            return 0.0

        total_open = sum(abs(v) for v in open_market_values.values())
        symbol_open = abs(open_market_values.get(symbol.upper(), 0.0))

        gross_cap = self._cfg.max_gross_exposure_pct * total_equity
        per_sym_cap = self._cfg.max_per_symbol_pct * total_equity

        gross_headroom = max(0.0, gross_cap - total_open)
        symbol_headroom = max(0.0, per_sym_cap - symbol_open)

        return min(gross_headroom, symbol_headroom)
```

- [ ] **Step 4: Run tests**
```
pytest tests/test_portfolio_capital_allocator.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**
```bash
git add src/portfolio/ tests/test_portfolio_capital_allocator.py
git commit -m "feat(portfolio): CapitalAllocator with per-symbol and gross-exposure caps"
```

---

### Task 6: Exposure tracker (`src/portfolio/exposure_tracker.py`)

**Files:**
- Create: `src/portfolio/exposure_tracker.py`
- Create: `tests/test_portfolio_exposure_tracker.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_portfolio_exposure_tracker.py
from src.portfolio.exposure_tracker import ExposureTracker, PositionSnapshot

def test_gross_exposure_two_longs():
    tracker = ExposureTracker()
    tracker.update([
        PositionSnapshot("NVDA", quantity=10, market_value=1_200.0),
        PositionSnapshot("MSFT", quantity=5, market_value=2_000.0),
    ], total_equity=10_000.0)
    assert tracker.gross_exposure_pct() == pytest.approx(0.32)

def test_net_exposure_long_and_short():
    tracker = ExposureTracker()
    tracker.update([
        PositionSnapshot("NVDA", quantity=10,  market_value=1_200.0),
        PositionSnapshot("MSFT", quantity=-5, market_value=-2_000.0),
    ], total_equity=10_000.0)
    assert tracker.net_exposure_pct() == pytest.approx(-0.08)   # (1200-2000)/10000

def test_symbol_exposure():
    tracker = ExposureTracker()
    tracker.update([
        PositionSnapshot("NVDA", quantity=10, market_value=1_500.0),
    ], total_equity=10_000.0)
    assert tracker.symbol_exposure_pct("NVDA") == pytest.approx(0.15)
    assert tracker.symbol_exposure_pct("MSFT") == 0.0

def test_open_market_values_dict():
    tracker = ExposureTracker()
    tracker.update([
        PositionSnapshot("NVDA", quantity=10, market_value=1_500.0),
        PositionSnapshot("MSFT", quantity=-5, market_value=-500.0),
    ], total_equity=10_000.0)
    mv = tracker.open_market_values()
    assert mv["NVDA"] == pytest.approx(1_500.0)
    assert mv["MSFT"] == pytest.approx(-500.0)

import pytest  # noqa: E402
```

- [ ] **Step 2: Run to confirm failures**
```
pytest tests/test_portfolio_exposure_tracker.py -v
```
Expected: 4 FAILED.

- [ ] **Step 3: Implement**

```python
# src/portfolio/exposure_tracker.py
"""Tracks open-position exposure across all symbols."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PositionSnapshot:
    symbol: str
    quantity: float    # positive = long, negative = short
    market_value: float  # signed (negative for short)


class ExposureTracker:
    """Computes gross/net exposure and per-symbol exposure from a position list.

    Updated each poll cycle from the broker snapshot.
    """

    def __init__(self) -> None:
        self._positions: list[PositionSnapshot] = []
        self._total_equity: float = 1.0

    def update(self, positions: list[PositionSnapshot], total_equity: float) -> None:
        """Replace current snapshot. Call once per poll cycle."""
        self._positions = list(positions)
        self._total_equity = max(total_equity, 1.0)

    def gross_exposure_pct(self) -> float:
        """Sum of |market_value| / equity."""
        total = sum(abs(p.market_value) for p in self._positions)
        return total / self._total_equity

    def net_exposure_pct(self) -> float:
        """Sum of signed market_value / equity (long minus short)."""
        total = sum(p.market_value for p in self._positions)
        return total / self._total_equity

    def symbol_exposure_pct(self, symbol: str) -> float:
        """Signed market_value of a single symbol / equity."""
        mv = next(
            (p.market_value for p in self._positions if p.symbol.upper() == symbol.upper()),
            0.0,
        )
        return mv / self._total_equity

    def open_market_values(self) -> dict[str, float]:
        """{symbol: signed_market_value} for all open positions."""
        return {p.symbol.upper(): p.market_value for p in self._positions}
```

- [ ] **Step 4: Run tests**
```
pytest tests/test_portfolio_exposure_tracker.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**
```bash
git add src/portfolio/exposure_tracker.py tests/test_portfolio_exposure_tracker.py
git commit -m "feat(portfolio): ExposureTracker for gross/net/per-symbol exposure"
```

---

### Task 7: Portfolio state file (`src/portfolio/state.py`)

Every poll-loop daemon reads this file once per cycle to get `available_capital` without needing to know about sibling daemons. Each daemon writes its own position after a trade.

**Files:**
- Create: `src/portfolio/state.py`
- Create: `tests/test_portfolio_state.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_portfolio_state.py
import json
from pathlib import Path
from src.portfolio.state import PortfolioStateManager

def test_write_and_read_roundtrip(tmp_path):
    mgr = PortfolioStateManager(tmp_path / "state.json")
    mgr.write_position("NVDA", quantity=10, market_value=1_500.0)
    mgr.write_position("MSFT", quantity=0, market_value=0.0)
    mgr.write_equity(50_000.0)
    state = mgr.read()
    assert state["total_equity"] == 50_000.0
    assert state["positions"]["NVDA"]["market_value"] == 1_500.0

def test_available_capital_respects_caps(tmp_path):
    from src.portfolio.capital_allocator import AllocationConfig, CapitalAllocator
    mgr = PortfolioStateManager(tmp_path / "state.json")
    mgr.write_equity(10_000.0)
    mgr.write_position("NVDA", quantity=10, market_value=2_800.0)
    mgr.write_position("MSFT", quantity=0, market_value=0.0)

    cfg = AllocationConfig(symbols=["NVDA", "MSFT"], max_per_symbol_pct=0.30)
    alloc = CapitalAllocator(cfg)
    avail = mgr.available_capital_for("MSFT", alloc)
    assert avail == pytest.approx(200.0)  # gross cap=8000; used=2800; headroom=5200 but per-sym cap=3000

import pytest  # noqa: E402
```

- [ ] **Step 2: Run to confirm failures**
```
pytest tests/test_portfolio_state.py -v
```

- [ ] **Step 3: Implement**

```python
# src/portfolio/state.py
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
```

- [ ] **Step 4: Run tests**
```
pytest tests/test_portfolio_state.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**
```bash
git add src/portfolio/state.py tests/test_portfolio_state.py
git commit -m "feat(portfolio): PortfolioStateManager — shared state for multi-daemon capital allocation"
```

---

## Phase 3 — Multi-daemon orchestration

### Task 8: Per-symbol config files

**Files:**
- Modify: `src/core/config_resolver.py`
- Create: `configs/symbols/NVDA/active.json` (migrated from `configs/active.json`)
- Test: `tests/test_config_resolver.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_config_resolver.py (add)
import json
from pathlib import Path

def test_get_strategy_defaults_per_symbol(tmp_path, monkeypatch):
    """config_resolver should read from configs/symbols/{SYMBOL}/active.json if present."""
    symbol_dir = tmp_path / "configs" / "symbols" / "MSFT"
    symbol_dir.mkdir(parents=True)
    (symbol_dir / "active.json").write_text(json.dumps({
        "strategy": {"k_sigma_long": 0.99},
        "meta": {"symbol": "MSFT"}
    }))
    monkeypatch.setenv("ML_LSTM_BASE_DIR", str(tmp_path))
    from importlib import reload
    import src.core.config_resolver as cr
    reload(cr)
    defaults = cr.get_strategy_defaults(symbol="MSFT")
    assert defaults["k_sigma_long"] == 0.99
```

- [ ] **Step 2: Run to confirm failure**
```
pytest tests/test_config_resolver.py::test_get_strategy_defaults_per_symbol -v
```
Expected: FAILED — `get_strategy_defaults` doesn't accept `symbol`.

- [ ] **Step 3: Update `src/core/config_resolver.py`**

Find `get_strategy_defaults()` and add the `symbol` parameter with fallback:

```python
def get_strategy_defaults(symbol: str | None = None) -> dict:
    """Load strategy defaults, preferring per-symbol config if available.

    Resolution order:
      1. configs/symbols/{SYMBOL}/active.json  (if symbol is specified and file exists)
      2. configs/active.json                   (global fallback)
      3. src/config.py STRATEGY_DEFAULTS       (code defaults)
    """
    from src.config import BASE_DIR, STRATEGY_DEFAULTS

    # 1. Per-symbol config
    if symbol:
        symbol_path = os.path.join(BASE_DIR, "configs", "symbols",
                                   symbol.upper(), "active.json")
        data = _try_load_json(symbol_path)
        if data and "strategy" in data:
            return {**_code_defaults(STRATEGY_DEFAULTS), **data["strategy"]}

    # 2. Global active.json
    global_path = os.path.join(BASE_DIR, "configs", "active.json")
    data = _try_load_json(global_path)
    if data and "strategy" in data:
        return {**_code_defaults(STRATEGY_DEFAULTS), **data["strategy"]}

    # 3. Code defaults
    return _code_defaults(STRATEGY_DEFAULTS)
```

Add helper `_try_load_json` if not already present:
```python
def _try_load_json(path: str) -> dict | None:
    try:
        content = Path(path).read_text(encoding="utf-8").strip()
        return json.loads(content) if content else None
    except Exception:
        return None
```

- [ ] **Step 4: Migrate NVDA config**
```bash
mkdir -p configs/symbols/NVDA
cp configs/active.json configs/symbols/NVDA/active.json
```

- [ ] **Step 5: Run tests**
```
pytest tests/test_config_resolver.py -v
```
Expected: all pass.

- [ ] **Step 6: Commit**
```bash
git add src/core/config_resolver.py configs/symbols/
git commit -m "feat(config): per-symbol strategy config in configs/symbols/{SYMBOL}/active.json"
```

---

### Task 9: Portfolio config file and allocator initialization

**Files:**
- Create: `configs/portfolio.json`
- Create: `src/portfolio/loader.py`
- Test: `tests/test_portfolio_state.py` (add to existing)

- [ ] **Step 1: Create `configs/portfolio.json`**

```json
{
  "_comment": "Edit this file to add/remove symbols. base_client_id: NVDA=10, MSFT=11, etc.",
  "symbols": ["NVDA"],
  "frequency": "60min",
  "tsteps": 5,
  "base_client_id": 10,
  "allocation": {
    "max_gross_exposure_pct": 0.80,
    "max_per_symbol_pct": 0.30
  }
}
```

- [ ] **Step 2: Write failing test**

```python
# tests/test_portfolio_state.py (add)
def test_load_portfolio_config(tmp_path):
    import json
    cfg_path = tmp_path / "portfolio.json"
    cfg_path.write_text(json.dumps({
        "symbols": ["NVDA", "MSFT"],
        "frequency": "60min",
        "tsteps": 5,
        "base_client_id": 10,
        "allocation": {"max_gross_exposure_pct": 0.8, "max_per_symbol_pct": 0.3}
    }))
    from src.portfolio.loader import load_portfolio_config
    cfg = load_portfolio_config(str(cfg_path))
    assert cfg["symbols"] == ["NVDA", "MSFT"]
    assert cfg["allocation"]["max_per_symbol_pct"] == 0.3
```

- [ ] **Step 3: Implement `src/portfolio/loader.py`**

```python
# src/portfolio/loader.py
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
```

- [ ] **Step 4: Run tests**
```
pytest tests/test_portfolio_state.py -v
```
Expected: all pass.

- [ ] **Step 5: Commit**
```bash
git add configs/portfolio.json src/portfolio/loader.py tests/test_portfolio_state.py
git commit -m "feat(portfolio): portfolio.json config + loader"
```

---

### Task 10: Poll loop portfolio integration

Each symbol daemon reads the portfolio state file once per cycle to get `available_capital`, then passes it as `buying_power` to `StrategyState`. This requires no changes to `strategy.py` — the existing `buying_power` field already does exactly this.

**Files:**
- Modify: `src/live/poll_loop.py`
- Test: `tests/test_poll_loop.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_poll_loop.py (add)
from unittest.mock import MagicMock, patch
from pathlib import Path

def test_poll_cycle_uses_portfolio_available_capital(tmp_path):
    """If portfolio state limits NVDA to $1000, buying_power passed to strategy is $1000."""
    state_file = tmp_path / "portfolio" / "state.json"
    state_file.parent.mkdir()
    import json
    state_file.write_text(json.dumps({
        "total_equity": 10_000.0,
        "positions": {"NVDA": {"quantity": 0, "market_value": 0.0}}
    }))

    # portfolio.json that caps NVDA at 10%
    portfolio_cfg = tmp_path / "portfolio.json"
    portfolio_cfg.write_text(json.dumps({
        "symbols": ["NVDA"],
        "frequency": "60min",
        "tsteps": 5,
        "base_client_id": 10,
        "allocation": {"max_gross_exposure_pct": 0.8, "max_per_symbol_pct": 0.10}
    }))

    captured_states = []
    original_compute = None

    def capture_state(state, cfg):
        captured_states.append(state)
        return None  # no trade

    with patch("src.live.poll_loop.compute_tp_sl_and_size", side_effect=capture_state):
        # ... run a single cycle with portfolio_state_path=state_file
        pass  # fill in with cycle invocation once poll_loop accepts portfolio params

    # After integration: buying_power <= 1000
    if captured_states:
        assert captured_states[0].buying_power <= 1_000.0
```

- [ ] **Step 2: Update `_run_single_cycle` in `src/live/poll_loop.py`**

Add two new parameters to `_run_single_cycle`:
```python
def _run_single_cycle(
    *,
    # ... existing params ...
    portfolio_state_path: Path | None = None,
    portfolio_config_path: str | None = None,
) -> tuple[float, bool, dict]:
```

Find the block that queries `buying_power` from the broker (around line 418-442 in the current file). After that block, add:

```python
        # --- Portfolio capital cap ---
        # If a shared portfolio state file is present, use the capital allocator
        # to further limit buying_power to this symbol's allocation.
        if portfolio_state_path is not None:
            try:
                from src.portfolio.state import PortfolioStateManager
                from src.portfolio.loader import make_allocator_from_config
                _ps = PortfolioStateManager(portfolio_state_path)
                _alloc = make_allocator_from_config(portfolio_config_path)
                _cap = _ps.available_capital_for(cfg.symbol, _alloc)
                if buying_power is None:
                    buying_power = _cap
                else:
                    buying_power = min(buying_power, _cap)
            except Exception:
                pass  # degrade gracefully — no portfolio cap applied
```

Also, after a successful order submission (find the `has_open_position = True` line), write back position:
```python
                # Update portfolio state so sibling daemons see this exposure
                if portfolio_state_path is not None:
                    try:
                        from src.portfolio.state import PortfolioStateManager
                        _ps = PortfolioStateManager(portfolio_state_path)
                        _size = max(1, int(round(plan.size)))
                        _mv = _size * current_price * plan.direction
                        _ps.write_position(cfg.symbol, quantity=_size * plan.direction, market_value=_mv)
                    except Exception:
                        pass
```

- [ ] **Step 3: Pass `portfolio_state_path` from `run_poll_loop`**

In `run_poll_loop`, derive the path and pass it to `_run_single_cycle`:
```python
    # Portfolio state path (optional — only if portfolio.json exists)
    from src.config import BASE_DIR
    _portfolio_cfg_path = os.path.join(BASE_DIR, "configs", "portfolio.json")
    _portfolio_state_path = (
        live_dir_path.parent / "portfolio" / "state.json"
        if os.path.exists(_portfolio_cfg_path) else None
    )
```

Pass these to every `_run_single_cycle` call:
```python
                    equity, has_open_position, order_state = _run_single_cycle(
                        # ... existing args ...
                        portfolio_state_path=_portfolio_state_path,
                        portfolio_config_path=_portfolio_cfg_path if _portfolio_state_path else None,
                    )
```

- [ ] **Step 4: Run existing poll_loop tests**
```
pytest tests/test_poll_loop.py -v
```
Expected: all existing tests still pass.

- [ ] **Step 5: Commit**
```bash
git add src/live/poll_loop.py
git commit -m "feat(live): poll loop reads portfolio state for per-symbol capital cap"
```

---

### Task 11: Multi-daemon launcher (`scripts/launch_portfolio.py`)

**Files:**
- Create: `scripts/launch_portfolio.py`
- Create: `deploy/nssm_install_portfolio.bat`

- [ ] **Step 1: Implement launcher**

```python
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
```

- [ ] **Step 2: Test the dry-run**
```bash
python -m scripts.launch_portfolio --dry-run
```
Expected output: prints one command line per symbol in `configs/portfolio.json`.

- [ ] **Step 3: Create NSSM installer for portfolio launcher**

```bat
@echo off
:: NSSM service for the multi-symbol portfolio launcher.
:: Run as Administrator.

set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%"
set "REPO_ROOT=%CD%"
popd

set "PYTHON_EXE=%REPO_ROOT%\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=%REPO_ROOT%\venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    for /f "delims=" %%i in ('where python 2^>nul') do set "PYTHON_EXE=%%i" & goto :py_found
    echo ERROR: Python not found.
    pause & exit /b 1
)
:py_found

set "LOG_DIR=%REPO_ROOT%\logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

set "WRAPPER=%REPO_ROOT%\deploy\run_portfolio.bat"
echo @echo off > "%WRAPPER%"
echo cd /d "%REPO_ROOT%" >> "%WRAPPER%"
echo "%PYTHON_EXE%" -m scripts.launch_portfolio >> "%LOG_DIR%\portfolio.log" 2^>^&1 >> "%WRAPPER%"

schtasks /delete /tn ml-lstm-portfolio /f > nul 2>&1
nssm stop  ml-lstm-portfolio > nul 2>&1
nssm remove ml-lstm-portfolio confirm > nul 2>&1

nssm install ml-lstm-portfolio "%WRAPPER%"
nssm set ml-lstm-portfolio AppDirectory "%REPO_ROOT%"
nssm set ml-lstm-portfolio AppRestartDelay 30000
nssm set ml-lstm-portfolio AppExit Default Restart
nssm set ml-lstm-portfolio Start SERVICE_AUTO_START
nssm set ml-lstm-portfolio AppStdout "%LOG_DIR%\portfolio_stdout.log"
nssm set ml-lstm-portfolio AppStderr "%LOG_DIR%\portfolio_stderr.log"
nssm set ml-lstm-portfolio AppRotateFiles 1
nssm set ml-lstm-portfolio AppRotateSeconds 86400

nssm start ml-lstm-portfolio
if errorlevel 1 (
    echo ERROR: service failed to start. Is IB Gateway running?
    pause & exit /b 1
)

echo Service ml-lstm-portfolio started.
pause
```
Save as `deploy/nssm_install_portfolio.bat`.

- [ ] **Step 4: Commit**
```bash
git add scripts/launch_portfolio.py deploy/nssm_install_portfolio.bat
git commit -m "feat: multi-daemon portfolio launcher + NSSM installer"
```

---

## Phase 4 — Tooling

### Task 12: Update `auto_retrain.py` for multiple symbols

**Files:**
- Modify: `scripts/auto_retrain.py`

- [ ] **Step 1: Add `--symbols` argument to CLI**

In `main()`, change:
```python
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="Symbols to retrain (default: reads from configs/portfolio.json)",
    )
```

At the start of `main()`, resolve symbols:
```python
    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        try:
            from src.portfolio.loader import load_portfolio_config
            symbols = load_portfolio_config()["symbols"]
        except Exception:
            symbols = [args.symbol.upper()]  # legacy single-symbol fallback
```

- [ ] **Step 2: Loop `run_auto_retrain` over symbols**

```python
    reports = []
    overall_exit = 0
    for sym in symbols:
        print(f"\n[auto_retrain] === Symbol: {sym} ===")
        report = run_auto_retrain(
            frequency=args.frequency,
            tsteps=args.tsteps,
            symbol=sym.lower(),
            dry_run=args.dry_run,
        )
        reports.append(report)
        if report.get("error"):
            overall_exit = 1

    sys.exit(overall_exit)
```

- [ ] **Step 3: Update `_stage_train` to write to `model_registry.py`**

At the end of a successful train stage, add a call to update the symbol registry:
```python
        # Register in symbol_registry.json (only if better than existing)
        from src.model_registry import update_best_model
        from src.config import get_run_hyperparameters
        hps = get_run_hyperparameters(frequency, tsteps)
        update_best_model(
            symbol=symbol.upper(),
            frequency=frequency,
            tsteps=tsteps,
            val_loss=val_loss,
            model_path=model_path,
            bias_path=bias_path,
            hparams=hps,
        )
```

Note: `symbol` needs to be threaded through `_stage_train` — add it as a parameter.

- [ ] **Step 4: Test multi-symbol dry run**
```bash
python -m scripts.auto_retrain --dry-run
```
Expected: runs data pipeline for each symbol in `configs/portfolio.json`.

- [ ] **Step 5: Commit**
```bash
git add scripts/auto_retrain.py
git commit -m "feat(retrain): multi-symbol retrain loop reads from portfolio.json"
```

---

### Task 13: Add a second symbol end-to-end (MSFT smoke test)

This task validates the entire stack with a real second symbol.

- [ ] **Step 1: Add MSFT data files**
```bash
# Fetch MSFT data (requires live IBKR connection):
python src/data_ingestion.py --symbol MSFT --start 2024-01-01

# Or if offline, copy and rename NVDA data as a placeholder:
cp data/processed/nvda_60min.csv data/processed/msft_60min.csv
```

- [ ] **Step 2: Add MSFT to portfolio config**

Edit `configs/portfolio.json`:
```json
{
  "symbols": ["NVDA", "MSFT"],
  "frequency": "60min",
  "tsteps": 5,
  "base_client_id": 10,
  "allocation": {
    "max_gross_exposure_pct": 0.80,
    "max_per_symbol_pct": 0.30
  }
}
```

- [ ] **Step 3: Train MSFT model**
```bash
python -m src.train --frequency 60min --tsteps 5 --symbol MSFT
```
(requires `src/train.py` to accept `--symbol` arg, which was added in Task 3 of Phase 1 implicitly — verify the train.py `--symbol` arg exists; add it if missing following the same pattern as `--frequency`.)

- [ ] **Step 4: Dry-run launcher**
```bash
python -m scripts.launch_portfolio --dry-run
```
Expected: prints two command lines (NVDA client_id=10, MSFT client_id=11).

- [ ] **Step 5: Verify capital allocator with two symbols**
```python
python -c "
from src.portfolio.loader import make_allocator_from_config
from src.portfolio.state import PortfolioStateManager
from pathlib import Path

alloc = make_allocator_from_config()
state = PortfolioStateManager('ui_state/portfolio/state.json')
state.write_equity(50000.0)
state.write_position('NVDA', 10, 1500.0)
state.write_position('MSFT', 0, 0.0)
print('NVDA available:', state.available_capital_for('NVDA', alloc))
print('MSFT available:', state.available_capital_for('MSFT', alloc))
"
```
Expected: NVDA ~$13,500 (30% cap - existing $1500), MSFT ~$15,000 (full 30% cap).

- [ ] **Step 6: Run full test suite**
```
pytest tests/ -q --ignore=tests/test_app_strategy_params.py
```
Expected: same pass count as before Phase 3, plus new portfolio tests.

- [ ] **Step 7: Commit**
```bash
git add configs/portfolio.json configs/symbols/MSFT/
git commit -m "feat: add MSFT to portfolio config — multi-asset trading enabled"
```

---

## Self-review checklist

**Spec coverage:**
| Requirement | Task |
|-------------|------|
| Parameterize data paths by symbol | Task 1 |
| Contract registry for multi-symbol IB connections | Task 2 |
| Symbol-keyed model registry | Task 4 |
| Per-symbol capital allocation cap | Task 5 |
| Gross portfolio exposure cap | Task 5 |
| Position correlation tracking (via ExposureTracker) | Task 6 |
| Shared state between daemons | Task 7 |
| Per-symbol strategy config | Task 8 |
| One daemon per symbol | Task 11 |
| Portfolio-aware position sizing | Task 10 |
| Multi-symbol autonomous retraining | Task 12 |
| Windows service for multi-daemon launcher | Task 11 |

**Known gaps deferred to future work:**
- Correlation matrix between symbols (requires market data library; deferred — the ExposureTracker exposes the primitives needed)
- Portfolio-level Value at Risk (deferred — single-symbol VaR is implicit in per-symbol caps)
- UI multi-symbol view (deferred — the event log files per symbol are already symbol-namespaced)

---

## Execution order

Phases must be executed in order (each phase depends on the previous). Within a phase, tasks are independent and can run in parallel if using subagent-driven development.

Estimated sessions:
- Phase 1: 1 session (4 tasks, mostly config plumbing)
- Phase 2: 1 session (3 tasks, pure library code)
- Phase 3: 2 sessions (4 tasks, integration work)
- Phase 4: 1 session (2 tasks, wiring)
