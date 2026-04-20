import io
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import pytest

from src.backtest import _align_predictions_to_data, _load_predictions_csv, run_backtest_on_dataframe
from src.checkpoint_provenance import (
    build_meta as _build_ckpt_meta,
    read_sidecar as _read_ckpt_sidecar,
    sidecar_path_for as _sidecar_path_for,
    validate_against as _validate_ckpt_sidecar,
    write_sidecar as _write_ckpt_sidecar,
)


def _toy_ohlc(*, n: int = 6) -> pd.DataFrame:
    # Small deterministic OHLC series.
    # Use non-zero ranges so ATR-like calculations don't collapse to 0.
    t0 = pd.Timestamp("2025-01-01T00:00:00Z")
    times = [t0 + pd.Timedelta(hours=i) for i in range(n)]

    close = np.linspace(100.0, 100.0 + (n - 1), n)
    df = pd.DataFrame(
        {
            "Time": times,
            "Open": close,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
        }
    )
    return df


def test_align_predictions_to_data_prefers_time_join() -> None:
    data = _toy_ohlc(n=4)

    # Predictions intentionally shuffled and missing one timestamp.
    preds = pd.DataFrame(
        {
            "Time": [
                data.loc[1, "Time"],
                data.loc[0, "Time"],
                data.loc[3, "Time"],
            ],
            "predicted_price": [111.0, 110.0, 113.0],
        }
    )

    aligned = _align_predictions_to_data(preds, data)

    assert len(aligned) == len(data)
    assert aligned[0] == 110.0
    assert aligned[1] == 111.0
    # Missing time -> NaN.
    assert np.isnan(aligned[2])
    assert aligned[3] == 113.0


def test_backtest_csv_mode_tolerates_nan_predictions_via_close_fallback(tmp_path) -> None:
    data = _toy_ohlc(n=10)

    # Provide a predictions CSV that is all-NaN; provider should fall back to Close.
    preds = pd.DataFrame({"Time": data["Time"], "predicted_price": [np.nan] * len(data)})
    p = tmp_path / "preds.csv"
    preds.to_csv(p, index=False)

    res = run_backtest_on_dataframe(
        data,
        prediction_mode="csv",
        predictions_csv=str(p),
        # Ensure we don't accidentally allow shorts by default.
        allow_shorts=False,
        enable_longs=True,
    )

    # Contract/invariant: equity curve must have one point per bar.
    assert len(res.equity_curve) == len(data)

    # With all NaN predictions -> close fallback -> zero signal; expect no trades.
    assert len(res.trades) == 0
    assert res.final_equity == res.equity_curve[0]


def test_load_predictions_csv_deduplicates_on_duplicate_time(tmp_path, capsys) -> None:
    """Deduplication guardrail: _load_predictions_csv must warn and remove
    duplicate Time rows, keeping the first occurrence.

    The NVDA production CSV had 380 duplicate timestamps (two prediction runs
    appended to the same file). Without dedup the Time-based merge in
    _make_model_prediction_provider produces extra rows and can misalign the
    equity curve.  This test pins that the dedup fires and produces a clean
    one-row-per-timestamp frame.
    """
    t0 = pd.Timestamp("2025-01-01T09:00:00")
    times = [t0 + pd.Timedelta(hours=i) for i in range(5)]

    # Row 2 is duplicated with a DIFFERENT predicted_price.  The first
    # occurrence (120.0) must win; the later one (999.0) must be dropped.
    rows = [
        {"Time": times[0], "predicted_price": 110.0},
        {"Time": times[1], "predicted_price": 111.0},
        {"Time": times[2], "predicted_price": 120.0},  # first occurrence
        {"Time": times[2], "predicted_price": 999.0},  # duplicate — must be dropped
        {"Time": times[3], "predicted_price": 113.0},
        {"Time": times[4], "predicted_price": 114.0},
    ]
    p = tmp_path / "preds.csv"
    pd.DataFrame(rows).to_csv(p, index=False)

    result = _load_predictions_csv(str(p))

    # Must warn to stdout.
    captured = capsys.readouterr()
    assert "WARNING" in captured.out, "Expected WARNING for duplicate timestamps"
    assert "duplicate" in captured.out.lower()

    # One row per timestamp.
    assert len(result) == 5

    # First occurrence wins for the duplicated timestamp.
    dup_row = result[result["Time"] == times[2]]
    assert len(dup_row) == 1
    assert float(dup_row["predicted_price"].iloc[0]) == 120.0, (
        "Expected first occurrence (120.0) kept, not the duplicate (999.0)"
    )


def test_load_predictions_csv_no_warning_on_clean_input(tmp_path, capsys) -> None:
    """No WARNING should be emitted when the predictions CSV is already clean."""
    t0 = pd.Timestamp("2025-06-01T09:00:00")
    rows = [{"Time": t0 + pd.Timedelta(hours=i), "predicted_price": 100.0 + i} for i in range(6)]
    p = tmp_path / "clean_preds.csv"
    pd.DataFrame(rows).to_csv(p, index=False)

    _load_predictions_csv(str(p))

    captured = capsys.readouterr()
    assert "WARNING" not in captured.out, (
        "No duplicate-time WARNING should be emitted for clean input"
    )


def test_checkpoint_interior_time_alignment_detects_off_by_n_drift(tmp_path) -> None:
    """Pattern 2 regression: endpoint-only checkpoint validation is not
    enough — a sidecar written against one OHLC frame must be rejected
    against a frame whose endpoints match but whose interior rows have
    drifted (e.g. by add_features-style off-by-N misalignment).
    """
    n = 50
    t0 = pd.Timestamp("2025-01-01T00:00:00")
    times_a = [t0 + pd.Timedelta(hours=i) for i in range(n)]
    data_a = pd.DataFrame({
        "Time": times_a,
        "Open": np.linspace(100, 149, n),
        "High": np.linspace(101, 150, n),
        "Low":  np.linspace(99, 148, n),
        "Close": np.linspace(100.5, 149.5, n),
    })

    # data_b has the same endpoints but interior rows shifted by a handful
    # of seconds — exactly the class of corruption the add_features
    # mutation bug produced (Times drifted by the warmup-row offset while
    # first/last stayed recognisable).
    times_b = list(times_a)
    for i in range(1, n - 1):
        times_b[i] = times_a[i] + pd.Timedelta(seconds=30)
    data_b = data_a.copy()
    data_b["Time"] = times_b

    ckpt = tmp_path / "nvda_60min_model_predictions_checkpoint.csv"
    ckpt.write_text("Time,predicted_price\n")
    meta = _build_ckpt_meta(data=data_a, symbol="NVDA", frequency="60min")
    _write_ckpt_sidecar(str(ckpt), meta)

    # Endpoints match, but the interior content differs → sidecar must
    # reject the mismatched frame because the content hash changed.
    ok, reason = _validate_ckpt_sidecar(
        meta, data=data_b, symbol="NVDA", frequency="60min",
    )
    assert not ok
    assert "sha256" in reason.lower() or "time" in reason.lower()
