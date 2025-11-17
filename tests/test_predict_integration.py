"""Integration tests for PredictionContext and batched predictions.

These tests hit the real model and scaler artifacts for a given frequency
(60min by default). They are slower and depend on trained models and
`best_hyperparameters.json`, so they are suitable as optional / slow tests.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from src.config import get_hourly_data_csv_path, TSTEPS
from src.predict import build_prediction_context, predict_sequence_batch
from src.data_processing import add_features
from src.data_utils import apply_standard_scaler


@pytest.mark.slow  # type: ignore[attr-defined]
def test_batched_predictions_real_model_no_nans() -> None:
    freq = "60min"

    # Try to build a real PredictionContext; if artifacts are missing, skip.
    try:
        ctx = build_prediction_context(frequency=freq, tsteps=TSTEPS)
    except FileNotFoundError:
        pytest.skip("No trained model/scaler found for 60min; skipping integration test.")

    csv_path = get_hourly_data_csv_path(freq)
    if not os.path.exists(csv_path):
        pytest.skip(f"Hourly CSV for {freq} not found at {csv_path}; skipping integration test.")

    # Use a small tail of the real data to keep the test reasonably fast.
    df = pd.read_csv(csv_path)
    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"])
    df = df.tail(500)

    # Feature engineering + scaling as in backtest.
    df_featured = add_features(df.copy(), ctx.features_to_use)
    feature_cols = [c for c in df_featured.columns if c != "Time"]
    df_normalized = apply_standard_scaler(df_featured, feature_cols, ctx.scaler_params)

    preds_normalized = predict_sequence_batch(ctx, df_normalized)

    # Basic sanity checks on shape and finiteness of non-warmup region.
    assert preds_normalized.ndim == 1
    assert len(preds_normalized) <= len(df_featured)

    # Warmup region of length TSTEPS-1 is expected; beyond that, predictions
    # should be finite.
    if len(preds_normalized) > 0:
        assert np.isfinite(preds_normalized).all()
