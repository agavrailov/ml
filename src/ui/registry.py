"""Model registry helpers for UI."""

from __future__ import annotations

import json
import re
from pathlib import Path


# Regex for parsing model filenames
_REGISTRY_MODEL_RE = re.compile(
    r"^my_lstm_model_(?P<frequency>.+?)_tsteps(?P<tsteps>\d+)_(?P<date>\d{8})_(?P<time>\d{6})\.keras$"
)


def parse_registry_model_filename(filename: str) -> dict | None:
    """Parse registry model filename into structured fields.

    Expected format:
        my_lstm_model_{frequency}_tsteps{tsteps}_{YYYYMMDD}_{HHMMSS}.keras
    """
    m = _REGISTRY_MODEL_RE.match(str(filename))
    if not m:
        return None

    freq = m.group("frequency")
    tsteps = int(m.group("tsteps"))
    d = m.group("date")
    tm = m.group("time")

    # Format into second-resolution ISO timestamp.
    ts_iso = f"{d[0:4]}-{d[4:6]}-{d[6:8]}T{tm[0:2]}:{tm[2:4]}:{tm[4:6]}"

    return {
        "model_filename": str(filename),
        "frequency": freq,
        "tsteps": tsteps,
        "timestamp": ts_iso,
        "stamp": f"{d}_{tm}",
    }


def list_registry_models(registry_dir: Path) -> list[dict]:
    """List all model artifacts in the model registry.

    Returns rows suitable for rendering in the UI.
    """
    if not registry_dir.exists() or not registry_dir.is_dir():
        return []

    rows: list[dict] = []
    for p in registry_dir.iterdir():
        if not p.is_file():
            continue
        info = parse_registry_model_filename(p.name)
        if info is None:
            continue

        # Best-effort bias-correction inference (same stamp).
        bias_name = f"bias_correction_{info['frequency']}_tsteps{int(info['tsteps'])}_{info['stamp']}.json"
        bias_path = registry_dir / bias_name
        info["bias_correction_filename"] = bias_name if bias_path.exists() else None

        # Per-model metrics (validation loss, etc.)
        metrics_name = str(info["model_filename"]).replace(".keras", ".metrics.json")
        metrics_path = registry_dir / metrics_name
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8") or "{}")
                if isinstance(metrics, dict) and metrics.get("validation_loss") is not None:
                    info["validation_loss"] = float(metrics.get("validation_loss"))
            except Exception:
                pass

        rows.append(info)

    # Sort most-recent first (timestamp is ISO so lexicographic sort works).
    return sorted(rows, key=lambda r: r.get("timestamp", ""), reverse=True)


def promote_training_row(
    *,
    row: dict,
    best_hps_path: Path,
    frequency: str,
    tsteps: int,
) -> None:
    """Write best_hyperparameters.json entry for (frequency, tsteps) from a history row."""

    best_hps_overall: dict = {}
    if best_hps_path.exists():
        try:
            content = best_hps_path.read_text(encoding="utf-8").strip()
            best_hps_overall = json.loads(content) if content else {}
        except json.JSONDecodeError:
            best_hps_overall = {}

    freq_key = frequency
    tsteps_key = str(int(tsteps))
    best_hps_overall.setdefault(freq_key, {})

    # Minimal required fields + useful metadata for later resolution.
    best_hps_overall[freq_key][tsteps_key] = {
        "validation_loss": float(row.get("validation_loss")),
        "model_filename": row.get("model_filename"),
        "bias_correction_filename": row.get("bias_correction_filename"),
        "lstm_units": row.get("lstm_units"),
        "learning_rate": row.get("learning_rate"),
        "epochs": row.get("epochs"),
        "batch_size": row.get("batch_size"),
        "n_lstm_layers": row.get("n_lstm_layers"),
        "stateful": row.get("stateful"),
        "optimizer_name": row.get("optimizer_name"),
        "loss_function": row.get("loss_function"),
        "features_to_use": row.get("features_to_use"),
    }

    best_hps_path.write_text(json.dumps(best_hps_overall, indent=4), encoding="utf-8")
