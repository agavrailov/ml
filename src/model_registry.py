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

import contextlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import BASE_DIR, MODEL_REGISTRY_DIR

_DEFAULT_REGISTRY_PATH = os.path.join(BASE_DIR, "models", "symbol_registry.json")


@contextlib.contextmanager
def _file_lock(lock_path: str, timeout: float = 15.0, poll: float = 0.05):
    """Cross-platform advisory file lock via exclusive file creation.

    Uses ``O_CREAT | O_EXCL`` which is atomic on both Windows (NTFS) and Unix.
    Retries every *poll* seconds until *timeout* expires, then raises TimeoutError.
    Always removes the lock file on exit, even if the caller raises.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Could not acquire registry lock {lock_path!r} within {timeout}s"
                )
            time.sleep(poll)
    try:
        yield
    finally:
        try:
            os.unlink(lock_path)
        except OSError:
            pass


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
    lock_path = registry_path + ".lock"
    with _file_lock(lock_path):
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
