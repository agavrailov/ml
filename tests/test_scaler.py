"""Tests for the StandardScaler helpers in ``src.data_utils``.

See Pattern 3 (silent NaN propagation) in docs/debugging-heuristics.md.
"""
from __future__ import annotations

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import pytest

from src.data_utils import apply_standard_scaler, fit_standard_scaler


def _sample_frame(n: int = 40) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed=7)
    df = pd.DataFrame({
        "Time": pd.date_range("2024-01-01 09:00", periods=n, freq="1h"),
        "Open":  100 + rng.standard_normal(n).cumsum(),
        "Close": 100 + rng.standard_normal(n).cumsum(),
        "Feat":  rng.standard_normal(n),
    })
    return df, ["Open", "Close", "Feat"]


def test_apply_standard_scaler_passes_on_clean_input():
    df, cols = _sample_frame()
    _, _, params = fit_standard_scaler(df, cols)
    norm = apply_standard_scaler(df, cols, params)
    # Output has the same shape and no NaNs introduced.
    assert norm.shape == df.shape
    assert not norm[cols].isna().any().any()
    # Standardised series have near-zero mean / unit std (modulo finite-sample).
    for c in cols:
        assert abs(norm[c].mean()) < 1e-6
        assert abs(norm[c].std() - 1.0) < 1e-6


def test_apply_scaler_rejects_nan_input():
    """NaN in the feature frame must be rejected up front — NaN / std = NaN
    silently propagates through the scaler and poisons every downstream
    stage (model input, prediction, bias correction).
    """
    df, cols = _sample_frame()
    _, _, params = fit_standard_scaler(df, cols)
    # Inject a single NaN into the middle of Close.
    df.loc[10, "Close"] = np.nan

    with pytest.raises(ValueError) as exc:
        apply_standard_scaler(df, cols, params)

    msg = str(exc.value)
    # Error must identify which column(s) are bad and how many NaNs.
    assert "Close" in msg
    assert "NaN" in msg or "nan" in msg.lower()
