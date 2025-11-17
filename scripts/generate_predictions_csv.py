"""Generate per-bar LSTM predictions for NVDA hourly data.

This script loads `data/processed/nvda_<frequency>.csv`, runs the
existing `predict_future_prices` function in a sliding-window fashion,
and writes a predictions CSV with columns:

- Time
- predicted_price

Usage (from repo root):

    python -m scripts.generate_predictions_csv --frequency 60min \
        --output backtests/nvda_60min_predictions.csv

This CSV can then be consumed by `src.backtest.py` with
`--prediction-mode=csv --predictions-csv <path>`.
"""
from __future__ import annotations

import argparse
import os
from typing import Optional

import pandas as pd
from tqdm import tqdm

from src.config import get_hourly_data_csv_path, TSTEPS
from src.predict import predict_future_prices


def generate_predictions_for_csv(
    frequency: str,
    output_path: str,
    max_rows: Optional[int] = None,
) -> None:
    """Generate per-bar predictions for the given frequency and write to CSV."""

    source_path = get_hourly_data_csv_path(frequency)
    if not os.path.exists(source_path):
        raise FileNotFoundError(
            f"Source OHLC CSV not found at {source_path}. "
            "Run your data pipeline / resampling first."
        )

    df = pd.read_csv(source_path)
    if "Time" not in df.columns:
        raise ValueError(
            f"Expected 'Time' column in {source_path} for alignment; found columns: {list(df.columns)}"
        )

    if max_rows is not None and max_rows > 0:
        df = df.tail(max_rows)

    preds = []
    times = []

    # Minimum rows needed so that feature engineering + TSTEPS window works.
    # This mirrors the logic in src.predict.__main__ (21-day SMA → 20 extra rows).
    min_rows_for_features = 20 + TSTEPS

    print(f"Generating predictions for {len(df)} bars from {source_path}...")

    for i in tqdm(range(len(df)), desc="predict", unit="bar"):
        window = df.iloc[: i + 1].copy()
        # Ensure Time is present and properly typed
        if "Time" in window.columns:
            window["Time"] = pd.to_datetime(window["Time"])

        # Warmup: before we have enough rows for full feature engineering,
        # fall back to Close as a neutral prediction.
        if len(window) < min_rows_for_features:
            p = float(window["Close"].iloc[-1])
        else:
            try:
                p = float(predict_future_prices(window, frequency=frequency))
            except ValueError as e:
                msg = str(e)
                # Allow only the explicit "not enough data after feature
                # engineering" error to degrade to Close; surface everything
                # else so issues with the model/scaler aren't hidden.
                if "Not enough data after feature engineering" in msg:
                    p = float(window["Close"].iloc[-1])
                else:
                    raise
        preds.append(p)
        times.append(window["Time"].iloc[-1])

    out_df = pd.DataFrame({"Time": times, "predicted_price": preds})
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"Wrote predictions for {len(out_df)} bars to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate per-bar predictions CSV for NVDA.")
    parser.add_argument(
        "--frequency",
        type=str,
        default="60min",
        help="Resample frequency to use (e.g. '15min', '60min').",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output predictions CSV.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit on the number of most recent rows to process.",
    )

    args = parser.parse_args()
    generate_predictions_for_csv(
        frequency=args.frequency,
        output_path=args.output,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
