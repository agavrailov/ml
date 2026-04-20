"""Training-window metadata for the active model.

Records the ``[train_start_date, train_end_date)`` interval that was used to
fit the currently-active scaler and model so that downstream consumers
(backtests, predictions CSV generation, UI defaults) can refuse to run on
bars the model never saw.

The metadata lives alongside the scaler parameters JSON — they share the same
lifetime (both are rewritten on every train), which keeps the window in lock-
step with the scaler/model without introducing a second sidecar file.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import get_scaler_params_json_path


# Historical default: ``src.train.train_model`` hard-coded ``Time >= 2023-01-01``
# before the window started being persisted. Models trained before this change
# carry no sidecar, so fall back to this value.
LEGACY_DEFAULT_TRAIN_START = "2023-01-01"


@dataclass(frozen=True)
class TrainingWindow:
    """Interval a model was trained on.

    ``start`` is inclusive, ``end`` is exclusive. ``end`` may be ``None`` when
    training extends to the most recent available data.
    """

    start: pd.Timestamp
    end: Optional[pd.Timestamp]
    source: str  # "sidecar" | "legacy_default"

    def contains(self, ts: pd.Timestamp) -> bool:
        if ts < self.start:
            return False
        if self.end is not None and ts >= self.end:
            return False
        return True


def _load_scaler_json(symbol: str, frequency: str) -> dict | None:
    try:
        path = Path(get_scaler_params_json_path(frequency, symbol=symbol))
    except Exception:
        return None
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def get_training_window(symbol: str, frequency: str) -> TrainingWindow:
    """Return the training window for ``(symbol, frequency)``.

    Reads the ``train_window`` block from the scaler-params JSON when present
    (written by :func:`src.train.train_model` at train time). Falls back to
    the legacy default of ``2023-01-01`` with no end cap when the sidecar
    predates window persistence.
    """

    data = _load_scaler_json(symbol, frequency)
    if data is not None:
        tw = data.get("train_window")
        if isinstance(tw, dict):
            start_raw = tw.get("start_date")
            end_raw = tw.get("end_date")
            if start_raw:
                start_ts = pd.to_datetime(start_raw)
                end_ts = pd.to_datetime(end_raw) if end_raw else None
                return TrainingWindow(start=start_ts, end=end_ts, source="sidecar")

    return TrainingWindow(
        start=pd.to_datetime(LEGACY_DEFAULT_TRAIN_START),
        end=None,
        source="legacy_default",
    )


def write_training_window(
    symbol: str,
    frequency: str,
    *,
    train_start_date: pd.Timestamp | str,
    train_end_date: pd.Timestamp | str | None,
) -> None:
    """Persist the training window into the scaler-params JSON sidecar.

    Idempotent: overwrites any existing ``train_window`` block. Called from
    :func:`src.train.train_model` right after the scaler JSON is written so
    the two always stay in sync.
    """

    path = Path(get_scaler_params_json_path(frequency, symbol=symbol))
    if not path.exists():
        return

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(data, dict):
        return

    start_ts = pd.to_datetime(train_start_date)
    end_ts = pd.to_datetime(train_end_date) if train_end_date is not None else None

    data["train_window"] = {
        "start_date": start_ts.strftime("%Y-%m-%d"),
        "end_date": end_ts.strftime("%Y-%m-%d") if end_ts is not None else None,
    }
    path.write_text(json.dumps(data, indent=4), encoding="utf-8")


def clamp_backtest_range(
    symbol: str,
    frequency: str,
    requested_start: pd.Timestamp | str | None,
    requested_end: pd.Timestamp | str | None,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None, list[str]]:
    """Clamp a requested backtest range to the model's training window.

    Returns the effective ``(start, end)`` plus a list of human-readable
    warnings explaining any clamping that took place. Callers are expected to
    surface these warnings to the user (print/logger/Streamlit ``st.warning``).
    """

    tw = get_training_window(symbol, frequency)
    warnings: list[str] = []

    start_ts = pd.to_datetime(requested_start) if requested_start else None
    end_ts = pd.to_datetime(requested_end) if requested_end else None

    effective_start = start_ts
    if effective_start is None or effective_start < tw.start:
        if effective_start is not None and effective_start < tw.start:
            warnings.append(
                f"[{symbol}] Backtest start {effective_start.date()} is before the model's "
                f"training start {tw.start.date()} (source={tw.source}). Clamping to "
                f"{tw.start.date()} to avoid evaluating on bars the model never saw."
            )
        effective_start = tw.start

    effective_end = end_ts
    if tw.end is not None and effective_end is not None and effective_end > tw.end:
        warnings.append(
            f"[{symbol}] Backtest end {effective_end.date()} is after the model's "
            f"training end {tw.end.date()}. Keeping the requested end but note that bars "
            "past training end are unseen by the scaler."
        )

    return effective_start, effective_end, warnings
