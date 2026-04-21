import json
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


# ---------------------------------------------------------------------------
# Intraday gaps: overnight pre-market gaps must not count
# ---------------------------------------------------------------------------

def _make_csv_with_overnight_gap(tmp_path) -> str:
    """CSV with a same-day gap from 03:00 → 12:30 UTC (pure pre-market)."""
    rows = []
    for minute in range(5):
        ts = datetime(2022, 4, 7, 2, 55) + timedelta(minutes=minute)
        rows.append({"DateTime": ts.strftime("%Y-%m-%d %H:%M:%S"),
                     "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0})
    # resume at 12:30 UTC (still pre-market; market opens ~13:30-14:30 UTC)
    for minute in range(5):
        ts = datetime(2022, 4, 7, 12, 30) + timedelta(minutes=minute)
        rows.append({"DateTime": ts.strftime("%Y-%m-%d %H:%M:%S"),
                     "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0})
    path = str(tmp_path / "overnight_gap.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_intraday_gaps_check_excludes_overnight_premarket(tmp_path):
    csv_path = _make_csv_with_overnight_gap(tmp_path)
    checks = analyze_raw_minute_data(csv_path)
    gap_check = next(c for c in checks if c["id"] == "intraday_gaps")
    # Overnight pre-market gap should not be counted → status pass or 0 gaps
    assert "0 intra-day" in gap_check["details"], (
        f"Expected 0 intraday trading-hours gaps but got: {gap_check['details']}"
    )
    assert gap_check["status"] == "pass"


# ---------------------------------------------------------------------------
# compute_quality_kpi: weight_override is honoured
# ---------------------------------------------------------------------------

def test_compute_quality_kpi_honours_weight_override():
    checks_no_override = [
        {"id": "a", "status": "pass"},
        {"id": "b", "status": "warn"},  # default weight 0.6
    ]
    kpi_default = compute_quality_kpi(checks_no_override)

    checks_with_override = [
        {"id": "a", "status": "pass"},
        {"id": "b", "status": "warn", "weight_override": 0.9},  # near-pass severity
    ]
    kpi_override = compute_quality_kpi(checks_with_override)

    assert kpi_override["score_0_100"] > kpi_default["score_0_100"], (
        "Higher weight_override should yield higher score"
    )


def test_compute_quality_kpi_more_real_gaps_lower_score():
    """Data with more trading-hours gaps should score lower via weight_override."""
    checks_few_gaps = [
        {"id": "a", "status": "pass"},
        {"id": "b", "status": "warn", "weight_override": max(0.5, 1.0 - 2 * 0.02)},
    ]
    checks_many_gaps = [
        {"id": "a", "status": "pass"},
        {"id": "b", "status": "warn", "weight_override": max(0.5, 1.0 - 20 * 0.02)},
    ]
    score_few = compute_quality_kpi(checks_few_gaps)["score_0_100"]
    score_many = compute_quality_kpi(checks_many_gaps)["score_0_100"]
    assert score_few > score_many, (
        f"2 real gaps ({score_few:.1f}) should score higher than 20 real gaps ({score_many:.1f})"
    )


# ---------------------------------------------------------------------------
# gap_analysis_json check: uses overlaps_trading_hours field
# ---------------------------------------------------------------------------

def test_gap_analysis_json_check_uses_overlap_field(tmp_path, monkeypatch):
    """gap_analysis_json check should give 'pass' when all gaps are pre-market."""
    gaps_data = [
        {"start": "2022-04-07T02:59:00", "end": "2022-04-07T12:30:00",
         "duration": "0 days 09:31:00", "overlaps_trading_hours": False},
        {"start": "2022-04-08T02:59:00", "end": "2022-04-08T12:00:00",
         "duration": "0 days 09:01:00", "overlaps_trading_hours": False},
    ]
    gap_json = str(tmp_path / "gaps.json")
    with open(gap_json, "w") as f:
        json.dump(gaps_data, f)

    # Monkeypatch GAP_ANALYSIS_OUTPUT_JSON so data_quality reads our file
    import src.data_quality as dq_mod
    monkeypatch.setattr(dq_mod, "GAP_ANALYSIS_OUTPUT_JSON", gap_json, raising=False)
    # Also patch the imported name used inside the function
    import src.data_processing as dp_mod
    monkeypatch.setattr(dp_mod, "GAP_ANALYSIS_OUTPUT_JSON", gap_json, raising=False)

    csv_path = _make_raw_minute_csv(tmp_path)
    checks = analyze_raw_minute_data(csv_path)
    gap_check = next((c for c in checks if c["id"] == "gap_analysis_json"), None)
    if gap_check is not None:
        assert gap_check["status"] == "pass", (
            f"All pre-market gaps → expected pass, got: {gap_check}"
        )