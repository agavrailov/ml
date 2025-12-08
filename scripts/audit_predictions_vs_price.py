"""Audit standalone LSTM predictions vs processed OHLC data.

For a given frequency (e.g. 15min, 60min), this script:

- Loads the processed OHLC CSV via src.data.load_hourly_ohlc.
- Loads the per-bar predictions CSV (Time, predicted_price).
- Joins on Time and computes error metrics vs the chosen target price
  horizon (by default, Open shifted by ROWS_AHEAD bars).
- Prints summary alignment stats and MAE/RMSE per calendar year.

Usage (from repo root):

    python -m scripts.audit_predictions_vs_price --frequency 60min \
        --predictions-csv backtests/nvda_60min_predictions.csv

If --predictions-csv is omitted, the script uses
src.config.get_predictions_csv_path("nvda", frequency).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    ROWS_AHEAD,
    FREQUENCY as CFG_FREQUENCY,
    get_predictions_csv_path,
)
from src.data import load_hourly_ohlc


@dataclass
class AuditConfig:
    frequency: str
    predictions_csv: Optional[str]
    target_horizon: int = ROWS_AHEAD


def _load_hourly(frequency: str) -> pd.DataFrame:
    df = load_hourly_ohlc(frequency)
    if "Time" not in df.columns:
        raise ValueError(f"Hourly data for {frequency} is missing 'Time' column.")
    df = df.copy()
    df["Time"] = pd.to_datetime(df["Time"])
    return df


def _load_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Time" not in df.columns or "predicted_price" not in df.columns:
        raise ValueError(
            f"Predictions CSV {path} must contain 'Time' and 'predicted_price' columns; "
            f"found {list(df.columns)}",
        )
    df = df.copy()
    df["Time"] = pd.to_datetime(df["Time"])
    # Guard against duplicate timestamps in predictions.
    if df["Time"].duplicated().any():
        dup_count = int(df["Time"].duplicated().sum())
        print(f"[audit] WARNING: {dup_count} duplicate Time rows in predictions; keeping first occurrence.")
        df = df.sort_values("Time").drop_duplicates(subset=["Time"], keep="first")
    return df


def _compute_target_prices(df_hourly: pd.DataFrame, rows_ahead: int) -> pd.Series:
    """Compute target prices aligned with the model's training horizon.

    For now we assume the model predicts a forward price based on Open[t+rows_ahead].
    This mirrors the training target definition used for log-return labels.
    """

    if "Open" not in df_hourly.columns:
        raise ValueError("Hourly data is missing 'Open' column for target computation.")

    return df_hourly["Open"].shift(-rows_ahead)


def _year_from_time(ts: pd.Timestamp) -> int:
    return int(ts.year)


def _mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return mae, rmse


def run_audit(cfg: AuditConfig) -> None:
    hourly = _load_hourly(cfg.frequency)

    preds_path = (
        cfg.predictions_csv
        if cfg.predictions_csv is not None
        else get_predictions_csv_path("nvda", cfg.frequency)
    )
    print(f"[audit] Using predictions CSV: {preds_path}")
    preds = _load_predictions(preds_path)

    merged = pd.merge(
        hourly,
        preds,
        on="Time",
        how="left",
        indicator=True,
    )

    n_hourly = len(hourly)
    n_preds = len(preds)
    n_joined = len(merged)
    n_missing_pred = int(merged["predicted_price"].isna().sum())

    print("[audit] Alignment summary:")
    print(f"  hourly bars:      {n_hourly}")
    print(f"  prediction rows:  {n_preds}")
    print(f"  joined rows:      {n_joined}")
    print(f"  missing preds:    {n_missing_pred}")

    # Compute target prices and restrict to rows with both target and prediction.
    merged["target_price"] = _compute_target_prices(merged, cfg.target_horizon)

    mask = merged[["predicted_price", "target_price"]].notna().all(axis=1)
    df_eval = merged.loc[mask].copy()
    if df_eval.empty:
        print("[audit] No rows with both predicted_price and target_price; nothing to evaluate.")
        return

    print(f"[audit] Rows with both prediction and target: {len(df_eval)}")

    y_true = df_eval["target_price"].to_numpy(dtype=float)
    y_pred = df_eval["predicted_price"].to_numpy(dtype=float)

    overall_mae, overall_rmse = _mae_rmse(y_true, y_pred)
    print("[audit] Overall error metrics (all years):")
    print(f"  MAE:  {overall_mae:.6f}")
    print(f"  RMSE: {overall_rmse:.6f}")

    # Per-year metrics.
    df_eval["year"] = df_eval["Time"].apply(_year_from_time)
    years = sorted(df_eval["year"].unique())
    print("[audit] Per-year error metrics:")
    for year in years:
        sub = df_eval[df_eval["year"] == year]
        y_t = sub["target_price"].to_numpy(dtype=float)
        y_p = sub["predicted_price"].to_numpy(dtype=float)
        mae, rmse = _mae_rmse(y_t, y_p)
        print(f"  {year}: n={len(sub):5d}  MAE={mae:.6f}  RMSE={rmse:.6f}")

    # Simple time-shift sanity: check correlation across small lags.
    print("[audit] Time-shift correlation check (target shifted by k bars):")
    for lag in [-2, -1, 0, 1, 2]:
        if lag == 0:
            label = "k=0  (no shift)"
        elif lag < 0:
            label = f"k={lag:2d} (target advanced)"
        else:
            label = f"k=+{lag:1d} (target delayed)"

        shifted = sub_target = df_eval["target_price"].shift(lag)
        mask_corr = shifted.notna() & df_eval["predicted_price"].notna()
        if not mask_corr.any():
            print(f"  {label}: insufficient overlap for correlation")
            continue

        a = df_eval.loc[mask_corr, "predicted_price"].to_numpy(dtype=float)
        b = shifted.loc[mask_corr].to_numpy(dtype=float)
        if a.size < 2:
            print(f"  {label}: insufficient data for correlation")
            continue

        # Guard against zero-variance edge cases.
        if np.std(a) == 0.0 or np.std(b) == 0.0:
            print(f"  {label}: zero variance; correlation undefined")
            continue

        corr = float(np.corrcoef(a, b)[0, 1])
        print(f"  {label}: corr={corr:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit standalone LSTM predictions vs processed OHLC data.",
    )
    parser.add_argument(
        "--frequency",
        type=str,
        default=CFG_FREQUENCY,
        help="Resample frequency (e.g. '15min', '60min'). Defaults to config FREQUENCY.",
    )
    parser.add_argument(
        "--predictions-csv",
        type=str,
        default=None,
        help=(
            "Optional override for predictions CSV. When omitted, uses "
            "src.config.get_predictions_csv_path('nvda', frequency)."
        ),
    )

    args = parser.parse_args()
    cfg = AuditConfig(frequency=args.frequency, predictions_csv=args.predictions_csv)
    run_audit(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
