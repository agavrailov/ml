"""Generate per-bar LSTM predictions for one or all symbols.

This script loads `data/processed/<symbol>_<frequency>.csv`, runs the existing
LSTM prediction pipeline in a sliding-window fashion, and writes predictions CSVs.

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

    # Single symbol
    python -m scripts.generate_predictions_csv --frequency 60min --symbol NVDA \
        --output backtests/nvda_60min_predictions.csv

    # All available symbols (auto-discovers from data/processed/)
    python -m scripts.generate_predictions_csv --frequency 60min

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
from src.backtest import _make_model_prediction_provider, PREDICTIONS_DIR


def generate_predictions_for_csv(
    frequency: str,
    output_path: str,
    symbol: str = "NVDA",
    max_rows: Optional[int] = None,
) -> None:
    """Generate per-bar predictions for the given frequency and write to CSV.

    We reuse the same model prediction pipeline as
    :func:`src.backtest._make_model_prediction_provider` and then reuse its
    on-disk checkpoint to avoid slow per-row pandas access on large datasets.
    """

    source_path = config_mod.get_hourly_data_csv_path(frequency, symbol=symbol)
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

    print(f"[{symbol}] Generating predictions for {len(df)} bars from {source_path}...")

    # Ensure Time is typed as datetime for alignment.
    df["Time"] = pd.to_datetime(df["Time"])

    # Ensure the model checkpoint exists and is aligned to this exact dataset.
    # `_make_model_prediction_provider` will either load and reuse an aligned
    # checkpoint, or compute predictions and write a new checkpoint.
    _make_model_prediction_provider(df.copy(), frequency=frequency, symbol=symbol)

    checkpoint_path = os.path.join(
        PREDICTIONS_DIR,
        f"{symbol.lower()}_{frequency}_model_predictions_checkpoint.csv",
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

    # Guard: ensure the checkpoint covers the full requested OHLC period.
    # A partial checkpoint (e.g. from a date-filtered backtest run) would
    # silently produce NaN predictions for the uncovered bars.
    if len(ckpt) < len(df):
        raise RuntimeError(
            f"Checkpoint at {checkpoint_path} has only {len(ckpt)} rows but OHLC data "
            f"has {len(df)} rows.  The checkpoint was likely written by a date-filtered "
            "backtest run and does not cover the full history.  Delete the checkpoint "
            "file and re-run this script to regenerate it from scratch."
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


def get_available_symbols(frequency: str) -> list[str]:
    """Discover all available symbols by scanning data/processed/ for {symbol}_{frequency}.csv files."""
    processed_dir = config_mod.PathsConfig().processed_data_dir
    symbols = []

    if not os.path.exists(processed_dir):
        return symbols

    for filename in os.listdir(processed_dir):
        if filename.endswith(f"_{frequency}.csv"):
            # Extract symbol from filename like "nvda_60min.csv" -> "NVDA"
            symbol = filename.replace(f"_{frequency}.csv", "").upper()
            symbols.append(symbol)

    return sorted(symbols)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate per-bar predictions CSV for one or all symbols.")
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Symbol to generate predictions for (e.g. NVDA, JPM, GS). If omitted, generates for all available symbols.",
    )
    parser.add_argument(
        "--frequency",
        type=str,
        default="60min",
        help="Resample frequency to use (e.g. '15min', '60min').",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the output predictions CSV. If omitted with no --symbol, generates backtests/{symbol}_{frequency}_predictions.csv for each symbol.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit on the number of most recent rows to process.",
    )

    args = parser.parse_args()

    # Determine which symbols to process
    if args.symbol:
        symbols = [args.symbol]
        if not args.output:
            raise ValueError("--output is required when --symbol is specified")
    else:
        symbols = get_available_symbols(args.frequency)
        if not symbols:
            raise ValueError(f"No data files found for frequency '{args.frequency}' in data/processed/")
        print(f"Auto-discovered symbols: {', '.join(symbols)}")

    # Process each symbol
    for symbol in symbols:
        if args.symbol:
            # Single symbol mode: use provided output path
            output_path = args.output
        else:
            # Multi-symbol mode: generate output path
            output_path = f"backtests/{symbol.lower()}_{args.frequency}_predictions.csv"

        try:
            generate_predictions_for_csv(
                frequency=args.frequency,
                output_path=output_path,
                symbol=symbol,
                max_rows=args.max_rows,
            )
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            if args.symbol:
                # If single symbol was explicitly requested, re-raise
                raise
            # If multi-symbol, continue to next symbol
            continue


if __name__ == "__main__":  # pragma: no cover
    main()
