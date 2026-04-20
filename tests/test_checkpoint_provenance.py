"""Tests for the checkpoint sidecar-JSON provenance helper.

See Pattern 4 in docs/debugging-heuristics.md.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd

from src.checkpoint_provenance import (
    SCHEMA_VERSION,
    build_meta,
    compute_ohlc_sha256,
    read_sidecar,
    sidecar_path_for,
    validate_against,
    write_sidecar,
)


def _make_ohlc(n: int = 50, *, start: str = "2024-01-01 09:00") -> pd.DataFrame:
    rng = np.random.default_rng(seed=1)
    return pd.DataFrame({
        "Time": pd.date_range(start, periods=n, freq="1h").astype(str),
        "Open":  100 + rng.standard_normal(n).cumsum(),
        "High":  101 + rng.standard_normal(n).cumsum(),
        "Low":    99 + rng.standard_normal(n).cumsum(),
        "Close": 100 + rng.standard_normal(n).cumsum(),
    })


def test_ohlc_sha256_is_deterministic_on_identical_frames():
    df_a = _make_ohlc()
    df_b = _make_ohlc()
    assert compute_ohlc_sha256(df_a) == compute_ohlc_sha256(df_b)


def test_ohlc_sha256_changes_when_a_single_value_changes():
    df = _make_ohlc()
    h_before = compute_ohlc_sha256(df)
    df.loc[10, "Close"] = float(df.loc[10, "Close"]) + 0.01
    h_after = compute_ohlc_sha256(df)
    assert h_before != h_after


def test_roundtrip_write_read(tmp_path):
    df = _make_ohlc()
    ckpt = tmp_path / "nvda_60min_model_predictions_checkpoint.csv"
    ckpt.write_text("Time,predicted_price\n")  # dummy
    meta = build_meta(data=df, symbol="NVDA", frequency="60min")
    write_sidecar(str(ckpt), meta)
    sidecar = sidecar_path_for(str(ckpt))
    assert os.path.exists(sidecar)
    # Re-read and verify fields roundtrip.
    loaded = read_sidecar(str(ckpt))
    assert loaded is not None
    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.symbol == "NVDA"
    assert loaded.frequency == "60min"
    assert loaded.ohlc_n_rows == len(df)
    assert loaded.ohlc_sha256 == meta.ohlc_sha256


def test_stale_checkpoint_rejected_when_data_changes(tmp_path):
    """The headline guardrail: a sidecar written against one OHLC frame
    must NOT validate against a differently-shaped or differently-valued
    frame.  This prevents silent reuse of stale checkpoints.
    """
    df_v1 = _make_ohlc(n=50)
    ckpt = tmp_path / "nvda_60min_model_predictions_checkpoint.csv"
    ckpt.write_text("Time,predicted_price\n")
    meta = build_meta(data=df_v1, symbol="NVDA", frequency="60min")
    write_sidecar(str(ckpt), meta)

    # Case A: different row count.
    df_v2 = _make_ohlc(n=100)
    ok, reason = validate_against(meta, data=df_v2, symbol="NVDA", frequency="60min")
    assert not ok
    assert "row count" in reason.lower()

    # Case B: same row count but a single value changed.
    df_v3 = df_v1.copy()
    df_v3.loc[20, "Open"] = float(df_v3.loc[20, "Open"]) + 0.5
    ok, reason = validate_against(meta, data=df_v3, symbol="NVDA", frequency="60min")
    assert not ok
    assert "sha256" in reason.lower()

    # Case C: wrong symbol.
    ok, reason = validate_against(meta, data=df_v1, symbol="MSFT", frequency="60min")
    assert not ok
    assert "symbol" in reason.lower()

    # Case D: wrong frequency.
    ok, reason = validate_against(meta, data=df_v1, symbol="NVDA", frequency="15min")
    assert not ok
    assert "frequency" in reason.lower()

    # Case E: everything matches.
    ok, reason = validate_against(meta, data=df_v1, symbol="NVDA", frequency="60min")
    assert ok, f"expected valid, got: {reason}"


def test_read_sidecar_returns_none_when_absent(tmp_path):
    ckpt = tmp_path / "does_not_exist_model_predictions_checkpoint.csv"
    assert read_sidecar(str(ckpt)) is None


def test_read_sidecar_returns_none_on_garbage(tmp_path):
    ckpt = tmp_path / "nvda_60min_model_predictions_checkpoint.csv"
    ckpt.write_text("")
    sidecar = sidecar_path_for(str(ckpt))
    with open(sidecar, "w") as f:
        f.write("{this is not valid json")
    assert read_sidecar(str(ckpt)) is None
