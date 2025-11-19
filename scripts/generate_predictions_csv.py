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
from src.data_processing import add_features
from src.data_utils import apply_standard_scaler
from src.predict import build_prediction_context, predict_sequence_batch


def generate_predictions_for_csv(
    frequency: str,
    output_path: str,
    max_rows: Optional[int] = None,
) -> None:
    """Generate per-bar predictions for the given frequency and write to CSV.

    This implementation reuses the shared :class:`PredictionContext` and
    :func:`predict_sequence_batch` helpers so that the LSTM model and scaler
    are built once and predictions are computed in a single batched call.
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
        df = df.tail(max_rows)

    print(f"Generating predictions for {len(df)} bars from {source_path}...")

    # Ensure Time is typed as datetime.
    df["Time"] = pd.to_datetime(df["Time"])

    # Build a reusable prediction context for this (frequency, TSTEPS).
    ctx = build_prediction_context(frequency=frequency, tsteps=config_mod.TSTEPS)

    # Feature engineering over the entire dataset.
    df_featured = add_features(df.copy(), ctx.features_to_use)

    # Normalize using the stored scaler.
    feature_cols = [c for c in df_featured.columns if c != "Time"]
    df_normalized = apply_standard_scaler(df_featured, feature_cols, ctx.scaler_params)

    # Batched predictions over all sliding windows.
    preds_normalized = predict_sequence_batch(ctx, df_normalized)

    # Align predictions to the feature-engineered DataFrame.
    m = len(df_featured)
    preds_feat_full = np.full(shape=(m,), fill_value=np.nan, dtype=np.float32)
    if len(preds_normalized) > 0:
        preds_feat_full[ctx.tsteps - 1 : ctx.tsteps - 1 + len(preds_normalized)] = preds_normalized

    # Denormalize using the stored scaler (Open as target).
    denorm_feat = preds_feat_full * ctx.std_vals["Open"] + ctx.mean_vals["Open"]

    # Align predictions back to the original raw data via Time. This accounts
    # for any rows dropped during feature engineering (rolling windows).
    preds_df = pd.DataFrame(
        {
            "Time": df_featured["Time"].reset_index(drop=True),
            "predicted_price": denorm_feat,
        }
    )
    merged = pd.merge(
        df[["Time"]].reset_index(drop=True),
        preds_df,
        on="Time",
        how="left",
    )

    out_df = merged[["Time", "predicted_price"]].copy()

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
