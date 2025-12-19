"""Generate per-bar LSTM predictions for NVDA data.

This script loads `data/processed/nvda_<frequency>.csv`, runs the existing
LSTM prediction pipeline in a sliding-window fashion, and writes a
predictions CSV.

Important performance note
- We reuse the model prediction checkpoint written by
  `src.backtest._make_model_prediction_provider`.
- We avoid per-row `df.iloc[i]` loops for large 1min datasets.

The output CSV contains columns:
- Time
- predicted_price
- model_error_sigma
- already_corrected

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

import numpy as np
import pandas as pd

from src import config as config_mod
from src.backtest import _make_model_prediction_provider


def generate_predictions_for_csv(
    frequency: str,
    output_path: str,
    max_rows: Optional[int] = None,
) -> None:
    """Generate per-bar predictions for the given frequency and write to CSV.

    We reuse the same model prediction pipeline as
    :func:`src.backtest._make_model_prediction_provider` and then reuse its
    on-disk checkpoint to avoid slow per-row pandas access on large datasets.
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

    # Ensure the model checkpoint exists and is aligned to this exact dataset.
    # `_make_model_prediction_provider` will either load and reuse an aligned
    # checkpoint, or compute predictions and write a new checkpoint.
    _make_model_prediction_provider(df.copy(), frequency=frequency)

    checkpoint_path = os.path.join(
        "backtests",
        f"nvda_{frequency}_model_predictions_checkpoint.csv",
    )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Expected model prediction checkpoint at {checkpoint_path} but it was not created."
        )

    # Load checkpoint and align to our requested OHLC slice by Time.
    ckpt = pd.read_csv(checkpoint_path, low_memory=False)
    if "Time" not in ckpt.columns or "predicted_price" not in ckpt.columns:
        raise ValueError(
            f"Checkpoint {checkpoint_path} missing required columns. Found columns: {list(ckpt.columns)}"
        )

    ckpt["Time"] = pd.to_datetime(ckpt["Time"], errors="coerce")
    merged = pd.merge(
        df[["Time"]].copy(),
        ckpt[[c for c in ("Time", "predicted_price", "model_error_sigma") if c in ckpt.columns]].copy(),
        on="Time",
        how="left",
    )

    # Backwards compatible: older checkpoints may not have sigma.
    if "model_error_sigma" not in merged.columns:
        merged["model_error_sigma"] = 0.0

    out_df = pd.DataFrame(
        {
            "Time": merged["Time"].astype("datetime64[ns]"),
            "predicted_price": merged["predicted_price"].astype(float),
            "model_error_sigma": merged["model_error_sigma"].astype(float),
            "already_corrected": True,
        }
    )

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
