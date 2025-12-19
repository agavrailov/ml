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

    # Stub out the shared model prediction provider so that we avoid heavy ML
    # and focus purely on the CSV generation and alignment logic.
    #
    # The generator now reuses the checkpoint written by `_make_model_prediction_provider`,
    # so our fake provider must also write a compatible checkpoint CSV.
    def _fake_model_provider(data_arg: pd.DataFrame, frequency: str):  # noqa: ARG001
        series = np.linspace(0.0, 1.0, len(data_arg))
        sigma_series = np.full(len(data_arg), 0.5, dtype=np.float32)

        from pathlib import Path

        checkpoint_path = Path("backtests") / f"nvda_{frequency}_model_predictions_checkpoint.csv"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "Time": pd.to_datetime(data_arg["Time"]).reset_index(drop=True),
                "predicted_price": series,
                "model_error_sigma": sigma_series,
            }
        ).to_csv(checkpoint_path, index=False)

        def provider(i: int, row: pd.Series) -> float:  # type: ignore[name-defined]
            return float(series[i])

        return provider, sigma_series

    output_path = tmp_path / "preds.csv"
    checkpoint_path = None
    try:
        with patch("scripts.generate_predictions_csv._make_model_prediction_provider", side_effect=_fake_model_provider):
            generate_predictions_for_csv(frequency=freq, output_path=str(output_path), max_rows=None)

        assert output_path.exists()
    finally:
        # Cleanup any checkpoint we wrote in the repo-local backtests/ dir.
        from pathlib import Path

        checkpoint_path = Path("backtests") / f"nvda_{freq}_model_predictions_checkpoint.csv"
        if checkpoint_path.exists():
            checkpoint_path.unlink()

    assert output_path.exists()
    out_df = pd.read_csv(output_path)

    # We expect one prediction per raw bar (aligned by Time).
    # The generator now also includes model_error_sigma and an
    # already_corrected flag so that CSV-mode can avoid double-correction.
    assert list(out_df.columns) == [
        "Time",
        "predicted_price",
        "model_error_sigma",
        "already_corrected",
    ]
    assert len(out_df) == 10

    # All predictions must be finite after the provider is applied.
    assert out_df["predicted_price"].notna().all()
    assert np.isfinite(out_df["predicted_price"]).all()
