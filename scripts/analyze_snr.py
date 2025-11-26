from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd

from src.config import (
    get_hourly_data_csv_path,
    ROWS_AHEAD,
    K_SIGMA_ERR,
    K_ATR_MIN_TP,
)


def load_data(freq: str, preds_path: str | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load OHLC data and predictions for a frequency.

    If ``preds_path`` is None, we try a small set of common filenames under
    ``backtests/``:
    - nvda_<freq>_model_predictions_checkpoint.csv
    - nvda_<freq>_predictions.csv
    - nvda_<freq>_predictions.csv (legacy)
    """

    ohlc_path = get_hourly_data_csv_path(freq)
    if not os.path.exists(ohlc_path):
        raise FileNotFoundError(f"OHLC CSV not found at {ohlc_path}")

    if preds_path is None:
        candidates = [
            os.path.join("backtests", f"nvda_{freq}_model_predictions_checkpoint.csv"),
            os.path.join("backtests", f"nvda_{freq}_predictions.csv"),
            os.path.join("backtests", f"nvda_{freq}_predictions.csv"),
            os.path.join("backtests", f"nvda_{freq}.csv"),
        ]
        preds_path = next((p for p in candidates if os.path.exists(p)), None)
        if preds_path is None:
            raise FileNotFoundError(
                "Could not find predictions file. Tried: "
                + ", ".join(candidates)
            )
    else:
        if not os.path.exists(preds_path):
            raise FileNotFoundError(f"Predictions CSV not found at {preds_path}")

    df = pd.read_csv(ohlc_path)
    preds = pd.read_csv(preds_path)

    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"])
    if "Time" in preds.columns:
        preds["Time"] = pd.to_datetime(preds["Time"])

    return df, preds


def compute_snr_series(
    df: pd.DataFrame,
    preds: pd.DataFrame,
    window: int = 100,
    k_sigma_err: float = K_SIGMA_ERR,
) -> pd.DataFrame:
    """Compute per-bar predicted_return, actual_return, sigma_return, usable_return and SNR.

    All returns are forward log returns on Open over ROWS_AHEAD bars.
    ``k_sigma_err`` controls how much residual sigma is subtracted from the
    predicted return when computing ``usable_return``.
    """

    if "Time" in df.columns and "Time" in preds.columns:
        merged = pd.merge(
            df,
            preds[["Time", "predicted_price"]],
            on="Time",
            how="inner",
        )
    else:
        # Fallback to positional alignment
        n = min(len(df), len(preds))
        merged = df.iloc[:n].copy()
        merged["predicted_price"] = preds["predicted_price"].iloc[:n].to_numpy()

    # Require Open and predicted_price columns
    if "Open" not in merged.columns:
        raise ValueError("Open column not found after merge")

    price_t = merged["Open"].astype(float)
    price_t_fut = price_t.shift(-ROWS_AHEAD)
    pred_price = merged["predicted_price"].astype(float)

    # Forward log returns
    predicted_return = np.log(pred_price / price_t)
    actual_return = np.log(price_t_fut / price_t)

    residual = actual_return - predicted_return

    # Rolling sigma of residuals (return space)
    sigma_return = (
        pd.Series(residual)
        .rolling(window=window, min_periods=window)
        .std()
        .to_numpy()
    )

    # Usable return like in strategy: r_pred - k_sigma_err * sigma
    usable_return = predicted_return - k_sigma_err * sigma_return

    # SNR: usable / sigma
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(sigma_return > 0, usable_return / sigma_return, np.nan)

    out = merged.copy()
    out["predicted_return"] = predicted_return
    out["actual_return"] = actual_return
    out["residual"] = residual
    out["sigma_return"] = sigma_return
    out["usable_return"] = usable_return
    out["snr"] = snr

    return out


def summarize_snr(
    df_snr: pd.DataFrame,
    k_sigma_err: float,
    snr_threshold: float,
) -> None:
    """Print summary stats for early vs late halves of the dataset."""

    n = len(df_snr)
    if n == 0:
        print("No data after SNR computation.")
        return

    mid = n // 2
    parts = {
        "early": df_snr.iloc[:mid],
        "late": df_snr.iloc[mid:],
    }

    print("\n=== Global thresholds ===")
    print(f"k_sigma_err   = {k_sigma_err}")
    print(f"SNR threshold = {snr_threshold}")

    for name, part in parts.items():
        mask = np.isfinite(part["predicted_return"]) & np.isfinite(part["sigma_return"]) & (part["sigma_return"] > 0)
        part_valid = part.loc[mask]
        if part_valid.empty:
            print(f"\n[{name}] no valid points for SNR stats")
            continue

        pr = part_valid["predicted_return"]
        ar = part_valid["actual_return"]
        sig = part_valid["sigma_return"]
        snr = part_valid["snr"]

        print(f"\n=== {name.upper()} half ===")
        print(f"bars:                  {len(part_valid)}")
        print(f"mean(predicted_return): {pr.mean(): .6f}")
        print(f"std(predicted_return):  {pr.std(): .6f}")
        print(f"mean(actual_return):    {ar.mean(): .6f}")
        print(f"mean(sigma_return):     {sig.mean(): .6f}")
        print(f"median(sigma_return):   {sig.median(): .6f}")
        print(f"mean(SNR):              {snr.mean(): .3f}")
        print(f"median(SNR):            {snr.median(): .3f}")
        frac_high_snr = (snr >= snr_threshold).mean()
        print(f"frac(SNR >= {snr_threshold}): {frac_high_snr: .3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SNR of model predictions vs actual returns.")
    parser.add_argument("--frequency", type=str, default="15min", help="Frequency, e.g. 15min, 60min")
    parser.add_argument("--window", type=int, default=100, help="Rolling window for residual sigma (bars)")
    parser.add_argument("--predictions-csv", type=str, default=None, help="Optional explicit path to predictions CSV.")
    args = parser.parse_args()

    df, preds = load_data(args.frequency, preds_path=args.predictions_csv)

    # Small grid of (k_sigma_err, snr_threshold) pairs to inspect.
    grid = [
        (0.0, 0.3),
        (0.0, 0.5),
        (0.25, 0.5),
        (0.25, 0.7),
        (0.5, 0.7),
    ]

    for k_sigma_err, snr_threshold in grid:
        df_snr = compute_snr_series(df, preds, window=args.window, k_sigma_err=k_sigma_err)
        summarize_snr(df_snr, k_sigma_err=k_sigma_err, snr_threshold=snr_threshold)

if __name__ == "__main__":  # pragma: no cover
    main()
