"""Tests for scripts.generate_predictions_csv.

We verify that generate_predictions_for_csv:
- Loads a small OHLC CSV.
- Calls the batched prediction helpers with a PredictionContext.
- Writes a CSV with Time and predicted_price of the expected length.

We mock out the heavy ML bits (build_prediction_context, predict_sequence_batch)
so the test is fast and deterministic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from scripts.generate_predictions_csv import generate_predictions_for_csv


def _write_dummy_ohlc_csv(tmp_path: Path, freq: str) -> str:
    """Create a small dummy OHLC CSV and return its path."""
    times = pd.date_range("2023-01-01", periods=10, freq="H")
    df = pd.DataFrame(
        {
            "Time": times,
            "Open": np.linspace(100.0, 101.0, len(times)),
            "High": np.linspace(100.5, 101.5, len(times)),
            "Low": np.linspace(99.5, 100.5, len(times)),
            "Close": np.linspace(100.2, 101.2, len(times)),
        }
    )
    csv_path = tmp_path / f"nvda_{freq}.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def test_generate_predictions_for_csv_uses_batched_context(tmp_path: Path, monkeypatch: Any) -> None:
    freq = "60min"
    source_csv = _write_dummy_ohlc_csv(tmp_path, freq)

    # Point get_hourly_data_csv_path to our dummy CSV.
    monkeypatch.setenv("ML_LSTM_BASE_DIR", str(tmp_path))

    from src import config as config_mod

    # Ensure PATHS points to tmp_path/data/processed for this test.
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    def _fake_hourly_path(frequency: str, processed_data_dir: str = str(processed_dir)) -> str:  # noqa: ARG001
        return source_csv

    monkeypatch.setattr(config_mod, "get_hourly_data_csv_path", _fake_hourly_path)

    # Mock PredictionContext and batched predictor.
    dummy_ctx = MagicMock()
    dummy_ctx.features_to_use = ["Open", "High", "Low", "Close"]
    dummy_ctx.tsteps = 5
    dummy_ctx.std_vals = {"Open": 1.0}
    dummy_ctx.mean_vals = {"Open": 0.0}
    dummy_ctx.scaler_params = {"mean": {"Open": 0.0}, "std": {"Open": 1.0}}

    with (
        patch("scripts.generate_predictions_csv.build_prediction_context", return_value=dummy_ctx),
        patch("scripts.generate_predictions_csv.add_features", side_effect=lambda df, feats: df),
        patch("scripts.generate_predictions_csv.apply_standard_scaler", side_effect=lambda df, cols, sp: df),
        patch(
            "scripts.generate_predictions_csv.predict_sequence_batch",
            return_value=np.linspace(0.0, 1.0, 10 - dummy_ctx.tsteps + 1),
        ),
    ):
        output_path = tmp_path / "preds.csv"
        generate_predictions_for_csv(frequency=freq, output_path=str(output_path), max_rows=None)

    assert output_path.exists()
    out_df = pd.read_csv(output_path)

    # We expect one prediction per raw bar (aligned by Time).
    assert list(out_df.columns) == ["Time", "predicted_price"]
    assert len(out_df) == 10

    # The warmup region (first tsteps-1) may be NaN; later rows must be finite.
    warmup = dummy_ctx.tsteps - 1
    assert out_df["predicted_price"].iloc[warmup:].notna().all()
