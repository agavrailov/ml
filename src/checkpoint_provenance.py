"""Sidecar-JSON provenance for prediction checkpoints.

The LSTM prediction checkpoint (``backtests/<sym>_<freq>_model_predictions_checkpoint.csv``)
can silently drift from the OHLC data it was generated against if any upstream
code (feature engineering, resampling, model version) changes between runs.
Historically the only alignment check was "first and last Time match", which
allowed a subtle ``add_features`` mutation bug to produce a checkpoint whose
endpoints matched but whose interior rows were off by ~20 warmup bars.

This module writes a ``*.meta.json`` alongside every checkpoint capturing the
fingerprint of the OHLC frame the checkpoint was generated from.  Readers
validate the fingerprint and refuse to silently reuse a stale checkpoint.

See ``docs/debugging-heuristics.md`` for background.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CheckpointMeta:
    """Provenance fingerprint for a prediction checkpoint."""

    schema_version: int
    symbol: str
    frequency: str
    ohlc_sha256: str
    ohlc_n_rows: int
    ohlc_first_time: str
    ohlc_last_time: str
    model_path: Optional[str]
    scaler_sha256: Optional[str]
    generated_at_utc: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> "CheckpointMeta":
        return cls(
            schema_version=int(d.get("schema_version", 0)),
            symbol=str(d["symbol"]),
            frequency=str(d["frequency"]),
            ohlc_sha256=str(d["ohlc_sha256"]),
            ohlc_n_rows=int(d["ohlc_n_rows"]),
            ohlc_first_time=str(d["ohlc_first_time"]),
            ohlc_last_time=str(d["ohlc_last_time"]),
            model_path=d.get("model_path"),
            scaler_sha256=d.get("scaler_sha256"),
            generated_at_utc=str(d["generated_at_utc"]),
        )


def compute_ohlc_sha256(data: pd.DataFrame) -> str:
    """Deterministic hash of an OHLC frame's Time + OHLC columns.

    Uses only the columns that define "the data we generated predictions from":
    Time, Open, High, Low, Close.  Feature columns (SMA, RSI, ...) are derived
    and intentionally excluded — they are reproducible from OHLC alone.
    """
    cols = ["Time"] + [c for c in ("Open", "High", "Low", "Close") if c in data.columns]
    frame = data[cols].copy()
    frame["Time"] = pd.to_datetime(frame["Time"]).astype("int64")
    for c in cols[1:]:
        frame[c] = frame[c].astype(np.float64)
    buf = frame.to_numpy().tobytes()
    return hashlib.sha256(buf).hexdigest()


def compute_file_sha256(path: str) -> Optional[str]:
    """SHA256 of a file's bytes, or None if it cannot be read."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            h = hashlib.sha256()
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
            return h.hexdigest()
    except OSError:
        return None


def sidecar_path_for(checkpoint_csv_path: str) -> str:
    """Return the sidecar-JSON path for a given checkpoint CSV path."""
    base, _ = os.path.splitext(checkpoint_csv_path)
    return base + ".meta.json"


def build_meta(
    *,
    data: pd.DataFrame,
    symbol: str,
    frequency: str,
    model_path: Optional[str] = None,
    scaler_path: Optional[str] = None,
) -> CheckpointMeta:
    times = pd.to_datetime(data["Time"])
    return CheckpointMeta(
        schema_version=SCHEMA_VERSION,
        symbol=symbol.upper(),
        frequency=frequency,
        ohlc_sha256=compute_ohlc_sha256(data),
        ohlc_n_rows=int(len(data)),
        ohlc_first_time=str(times.iloc[0]),
        ohlc_last_time=str(times.iloc[-1]),
        model_path=model_path,
        scaler_sha256=compute_file_sha256(scaler_path) if scaler_path else None,
        generated_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def write_sidecar(checkpoint_csv_path: str, meta: CheckpointMeta) -> str:
    path = sidecar_path_for(checkpoint_csv_path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(meta.to_json())
    logger.debug("Wrote checkpoint sidecar meta to %s", path)
    return path


def read_sidecar(checkpoint_csv_path: str) -> Optional[CheckpointMeta]:
    path = sidecar_path_for(checkpoint_csv_path)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return CheckpointMeta.from_dict(d)
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("Failed to parse checkpoint sidecar at %s: %s", path, exc)
        return None


def validate_against(
    meta: CheckpointMeta,
    *,
    data: pd.DataFrame,
    symbol: str,
    frequency: str,
) -> tuple[bool, str]:
    """Return (is_valid, reason).

    Compares the sidecar fingerprint against the OHLC frame the caller intends
    to run a backtest on.  If any of (symbol, frequency, hash, row count,
    endpoints) mismatch, the checkpoint is treated as stale.
    """
    if meta.schema_version != SCHEMA_VERSION:
        return False, (
            f"schema_version mismatch: sidecar={meta.schema_version} "
            f"code={SCHEMA_VERSION}"
        )
    if meta.symbol.upper() != symbol.upper():
        return False, f"symbol mismatch: sidecar={meta.symbol} data={symbol}"
    if meta.frequency != frequency:
        return False, (
            f"frequency mismatch: sidecar={meta.frequency} data={frequency}"
        )
    if meta.ohlc_n_rows != len(data):
        return False, (
            f"row count mismatch: sidecar={meta.ohlc_n_rows} data={len(data)}"
        )
    times = pd.to_datetime(data["Time"])
    if str(times.iloc[0]) != meta.ohlc_first_time:
        return False, (
            f"first Time mismatch: sidecar={meta.ohlc_first_time} "
            f"data={times.iloc[0]}"
        )
    if str(times.iloc[-1]) != meta.ohlc_last_time:
        return False, (
            f"last Time mismatch: sidecar={meta.ohlc_last_time} "
            f"data={times.iloc[-1]}"
        )
    actual_hash = compute_ohlc_sha256(data)
    if actual_hash != meta.ohlc_sha256:
        return False, (
            f"ohlc_sha256 mismatch: sidecar={meta.ohlc_sha256[:12]}... "
            f"data={actual_hash[:12]}..."
        )
    return True, "ok"
