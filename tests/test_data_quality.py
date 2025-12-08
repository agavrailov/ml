import os
from datetime import datetime, timedelta

import pandas as pd

from src.data_quality import (
    analyze_raw_minute_data,
    compute_quality_kpi,
    format_quality_report,
    validate_hourly_ohlc,
    validate_feature_frame,
)


def _make_raw_minute_csv(tmp_path) -> str:
    """Create a tiny but valid raw-minute CSV and return its path."""

    csv_path = tmp_path / "nvda_minute.csv"
    start = datetime(2023, 1, 2, 9, 30)
    rows = []
    for i in range(10):
        ts = start + timedelta(minutes=i)
        rows.append(
            {
                "DateTime": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "Open": 100.0 + i,
                "High": 101.0 + i,
                "Low": 99.0 + i,
                "Close": 100.5 + i,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def test_analyze_raw_minute_data_and_kpi(tmp_path):
    csv_path = _make_raw_minute_csv(tmp_path)

    checks = analyze_raw_minute_data(csv_path)
    assert checks, "expected at least one check result"

    kpi = compute_quality_kpi(checks)
    assert 0.0 <= kpi["score_0_100"] <= 100.0
    assert kpi["n_total"] == len(checks)

    report = format_quality_report(checks, kpi, dataset_name="Test raw data")
    assert "Test raw data quality report" in report


def test_validate_hourly_ohlc_happy_path():
    df = pd.DataFrame(
        {
            "Time": pd.date_range("2023-01-01", periods=3, freq="H"),
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [100.5, 101.5, 102.5],
        }
    )

    # Should not raise
    validate_hourly_ohlc(df, context="test_hourly")


def test_validate_hourly_ohlc_raises_on_missing_columns():
    df = pd.DataFrame({"Time": pd.date_range("2023-01-01", periods=1, freq="H")})
    try:
        validate_hourly_ohlc(df, context="test_bad_hourly")
    except ValueError as exc:
        assert "missing required columns" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing OHLC columns")


def test_validate_feature_frame_happy_path():
    times = pd.date_range("2023-01-01", periods=5, freq="H")
    df = pd.DataFrame(
        {
            "Time": times,
            "f1": [1, 2, 3, 4, 5],
            "f2": [10, 11, 12, 13, 14],
        }
    )

    validate_feature_frame(df, ["f1", "f2"], context="test_features")


def test_validate_feature_frame_raises_on_nans():
    times = pd.date_range("2023-01-01", periods=3, freq="H")
    df = pd.DataFrame(
        {
            "Time": times,
            "f1": [1.0, float("nan"), 3.0],
        }
    )

    try:
        validate_feature_frame(df, ["f1"], context="test_nan_features")
    except ValueError as exc:
        assert "NaNs detected" in str(exc)
    else:
        raise AssertionError("expected ValueError for NaNs in feature columns")