"""Generate per-bar LSTM predictions for NVDA data.

This script loads `data/processed/nvda_<frequency>.csv`, runs the existing
LSTM prediction pipeline in a sliding-window fashion, and writes a
predictions CSV with columns:

- Time
- predicted_price

Usage (from repo root):

    python -m scripts.generate_predictions_csv --frequency 60min \
        --output backtests/nvda_60min_predictions.csv

This CSV can then be consumed by `src.backtest.py` with
`--prediction-mode=csv --predictions-csv <path>` or by `src.paper_trade.py`.
"""
from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src import config as config_mod
from src.backtest import _make_model_prediction_provider


def generate_predictions_for_csv(
    frequency: str,
    output_path: str,
    max_rows: Optional[int] = None,
) -> None:
    """Generate per-bar predictions for the given frequency and write to CSV.

    This implementation reuses the same model prediction pipeline as
    :func:`src.backtest._make_model_prediction_provider` so that CSV-based
    backtests see the *exact* same predictions and sigma series as model-mode.
    """

    source_path = config_mod.get_hourly_data_csv_path(frequency)
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
        df = df.tail(max_rows).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    print(f"Generating predictions for {len(df)} bars from {source_path}...")

    # Ensure Time is typed as datetime for alignment.
    df["Time"] = pd.to_datetime(df["Time"])

    # Reuse the shared model prediction provider, which will either load an
    # existing aligned checkpoint or run the full batched prediction pipeline
    # and write a new checkpoint. This guarantees consistency with
    # model-mode backtests.
    provider, sigma_series = _make_model_prediction_provider(df.copy(), frequency=frequency)

    # Materialize per-bar predictions by calling the provider over the DataFrame.
    preds = np.empty(len(df), dtype=float)
    for i in range(len(df)):
        preds[i] = float(provider(i, df.iloc[i]))

    # Build output frame. We include model_error_sigma and a boolean flag to
    # indicate that these predictions have already passed through the
    # bias-correction layer, so CSV-mode backtests can avoid double-correction.
    out_df = pd.DataFrame(
        {
            "Time": df["Time"].astype("datetime64[ns]"),
            "predicted_price": preds,
            "model_error_sigma": np.asarray(sigma_series, dtype=float),
            "already_corrected": True,
        }
    )

    # tqdm for user feedback, iterating only to show progress, not to compute.
    for _ in tqdm(range(len(out_df)), desc="predict", unit="bar"):
        pass

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
