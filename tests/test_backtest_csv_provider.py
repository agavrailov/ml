from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import _make_csv_prediction_provider


def test_csv_provider_aligns_on_time_and_fills_nans() -> None:
    """When both data and predictions have Time, align on Time and ffill/bfill NaNs."""
    times = pd.date_range("2023-01-01", periods=5, freq="H")
    data = pd.DataFrame(
        {
            "Time": times,
            "Open":  [1, 2, 3, 4, 5],
            "High":  [1, 2, 3, 4, 5],
            "Low":   [1, 2, 3, 4, 5],
            "Close": [1, 2, 3, 4, 5],
        }
    )

    # Predictions cover only the middle 3 timestamps and contain an internal NaN.
    preds = pd.DataFrame(
        {
            "Time": times[1:4],
            "predicted_price": [10.0, np.nan, 12.0],
        }
    )

    provider = _make_csv_prediction_provider(preds, data)
    values = [provider(i, data.iloc[i]) for i in range(len(data))]

    # After merge + ffill + bfill, all values should be finite.
    assert all(np.isfinite(values))

    # Edge timestamps should be filled from nearest available predictions.
    assert values[0] == pytest.approx(10.0)
    assert values[-1] == pytest.approx(12.0)


def test_csv_provider_positional_padding_and_truncation() -> None:
    """When Time is absent, ensure padding/truncation behaviour is correct."""
    data = pd.DataFrame(
        {
            "Open":  [1, 2, 3],
            "High":  [1, 2, 3],
            "Low":   [1, 2, 3],
            "Close": [1, 2, 3],
        }
    )

    # Case 1: shorter predictions -> pad by repeating last value.
    preds_short = pd.DataFrame({"predicted_price": [10.0, 20.0]})
    provider_short = _make_csv_prediction_provider(preds_short, data)
    vals_short = [provider_short(i, data.iloc[i]) for i in range(len(data))]
    assert vals_short == [10.0, 20.0, 20.0]

    # Case 2: longer predictions -> truncate to data length.
    preds_long = pd.DataFrame({"predicted_price": [1.0, 2.0, 3.0, 4.0]})
    provider_long = _make_csv_prediction_provider(preds_long, data)
    vals_long = [provider_long(i, data.iloc[i]) for i in range(len(data))]
    assert vals_long == [1.0, 2.0, 3.0]
