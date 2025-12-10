from __future__ import annotations

import argparse
from datetime import datetime

import pandas as pd

from src import config as cfg
from src.data import load_hourly_ohlc
from src.train import train_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain LSTM model as of a given date using a rolling 18–24 month window.",
    )
    parser.add_argument("--frequency", type=str, default=cfg.FREQUENCY, help="Resample frequency, e.g. 15min, 60min.")
    parser.add_argument("--tsteps", type=int, default=cfg.TSTEPS, help="TSTEPS to use when training.")
    parser.add_argument("--as-of-date", type=str, required=True, help="Retrain boundary date (YYYY-MM-DD).")
    parser.add_argument(
        "--train-lookback-months",
        type=int,
        default=24,
        help="Training lookback in months before as-of-date (default 24).",
    )

    args = parser.parse_args()

    as_of_ts = pd.to_datetime(args.as_of_date).normalize()

    # Infer earliest available timestamp from hourly data.
    df_full = load_hourly_ohlc(args.frequency)
    if "Time" not in df_full.columns or df_full.empty:
        raise SystemExit("Hourly OHLC data must be non-empty and contain a 'Time' column.")

    times = pd.to_datetime(df_full["Time"])
    data_start = times.min().normalize()

    train_end = as_of_ts
    train_start = train_end - pd.DateOffset(months=args.train_lookback_months)
    if train_start < data_start:
        train_start = data_start

    train_start_str = train_start.strftime("%Y-%m-%d")
    train_end_str = train_end.strftime("%Y-%m-%d")

    print(
        f"Retraining model for frequency={args.frequency}, TSTEPS={args.tsteps} "
        f"on window [{train_start_str}, {train_end_str})",
    )

    result = train_model(
        frequency=args.frequency,
        tsteps=args.tsteps,
        train_start_date=train_start_str,
        train_end_date=train_end_str,
    )

    if result is None:
        print("Training failed or not enough data in the specified window.")
    else:
        final_val_loss, model_path, bias_correction_path = result
        print(
            "Training complete. "
            f"val_loss={final_val_loss:.4e}, model={model_path}, bias_correction={bias_correction_path}",
        )


if __name__ == "__main__":  # pragma: no cover
    main()
