"""Regression tests for ``src.core.training_window``.

Verifies sidecar read/write, legacy-default fallback, and the backtest-range
clamp behaviour (clamping an earlier ``start_date`` with a warning, leaving
in-range windows untouched).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def isolated_scaler(tmp_path, monkeypatch):
    """Redirect ``get_scaler_params_json_path`` to a temp file per test."""

    scaler_path = tmp_path / "scaler_params_test_60min.json"

    def _fake_path(frequency, symbol: str = "NVDA", processed_data_dir=None):
        return str(scaler_path)

    # Patch at both import sites — one used inside training_window, one used by config.
    monkeypatch.setattr("src.core.training_window.get_scaler_params_json_path", _fake_path)

    return scaler_path


def _write_scaler(path: Path, extra: dict | None = None) -> None:
    payload = {
        "mean": {"Open": 100.0},
        "std": {"Open": 10.0},
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_legacy_default_when_no_sidecar(isolated_scaler):
    """Absent sidecar → fall back to the 2023-01-01 legacy default."""
    from src.core.training_window import get_training_window, LEGACY_DEFAULT_TRAIN_START

    tw = get_training_window(symbol="TEST", frequency="60min")
    assert tw.source == "legacy_default"
    assert tw.start == pd.to_datetime(LEGACY_DEFAULT_TRAIN_START)
    assert tw.end is None


def test_reads_sidecar_when_present(isolated_scaler):
    _write_scaler(
        isolated_scaler,
        extra={"train_window": {"start_date": "2024-03-15", "end_date": "2025-10-01"}},
    )
    from src.core.training_window import get_training_window

    tw = get_training_window(symbol="TEST", frequency="60min")
    assert tw.source == "sidecar"
    assert tw.start == pd.Timestamp("2024-03-15")
    assert tw.end == pd.Timestamp("2025-10-01")


def test_write_training_window_is_idempotent(isolated_scaler):
    _write_scaler(isolated_scaler)
    from src.core.training_window import write_training_window, get_training_window

    write_training_window(
        symbol="TEST",
        frequency="60min",
        train_start_date="2024-01-01",
        train_end_date="2025-01-01",
    )
    tw1 = get_training_window(symbol="TEST", frequency="60min")

    # Overwrite with a new range — should replace cleanly, not append.
    write_training_window(
        symbol="TEST",
        frequency="60min",
        train_start_date="2024-06-01",
        train_end_date=None,
    )
    tw2 = get_training_window(symbol="TEST", frequency="60min")

    assert tw1.start == pd.Timestamp("2024-01-01")
    assert tw2.start == pd.Timestamp("2024-06-01")
    assert tw2.end is None

    # The scaler mean/std keys must survive the rewrite.
    data = json.loads(isolated_scaler.read_text(encoding="utf-8"))
    assert data["mean"] == {"Open": 100.0}
    assert data["std"] == {"Open": 10.0}


def test_clamp_pulls_start_forward_with_warning(isolated_scaler):
    _write_scaler(
        isolated_scaler,
        extra={"train_window": {"start_date": "2023-01-01", "end_date": None}},
    )
    from src.core.training_window import clamp_backtest_range

    start, end, warnings = clamp_backtest_range(
        symbol="TEST",
        frequency="60min",
        requested_start="2022-01-03",
        requested_end="2026-04-16",
    )

    assert start == pd.Timestamp("2023-01-01")
    assert end == pd.Timestamp("2026-04-16")
    assert len(warnings) == 1
    assert "before the model's training start" in warnings[0]


def test_clamp_leaves_in_range_request_untouched(isolated_scaler):
    _write_scaler(
        isolated_scaler,
        extra={"train_window": {"start_date": "2023-01-01", "end_date": None}},
    )
    from src.core.training_window import clamp_backtest_range

    start, end, warnings = clamp_backtest_range(
        symbol="TEST",
        frequency="60min",
        requested_start="2024-06-01",
        requested_end="2025-06-01",
    )

    assert start == pd.Timestamp("2024-06-01")
    assert end == pd.Timestamp("2025-06-01")
    assert warnings == []


def test_clamp_warns_when_end_past_training_end(isolated_scaler):
    _write_scaler(
        isolated_scaler,
        extra={"train_window": {"start_date": "2023-01-01", "end_date": "2025-01-01"}},
    )
    from src.core.training_window import clamp_backtest_range

    start, end, warnings = clamp_backtest_range(
        symbol="TEST",
        frequency="60min",
        requested_start="2024-01-01",
        requested_end="2026-01-01",
    )

    # End is kept (not hard-clamped) but a warning is raised so the user knows
    # they're evaluating on bars past the training window.
    assert start == pd.Timestamp("2024-01-01")
    assert end == pd.Timestamp("2026-01-01")
    assert len(warnings) == 1
    assert "after the model's training end" in warnings[0]


def test_clamp_defaults_missing_start_to_training_start(isolated_scaler):
    _write_scaler(
        isolated_scaler,
        extra={"train_window": {"start_date": "2023-01-01", "end_date": None}},
    )
    from src.core.training_window import clamp_backtest_range

    start, end, warnings = clamp_backtest_range(
        symbol="TEST",
        frequency="60min",
        requested_start=None,
        requested_end="2025-06-01",
    )

    # No explicit start → silently use training start (no warning: user
    # didn't ask for anything risky).
    assert start == pd.Timestamp("2023-01-01")
    assert end == pd.Timestamp("2025-06-01")
    assert warnings == []
