import pandas as pd
import numpy as np

from src.backtest import _align_predictions_to_data, run_backtest_on_dataframe


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
