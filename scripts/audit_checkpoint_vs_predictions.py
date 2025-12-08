"""Audit checkpoint predictions vs standalone predictions and prices.

For a given frequency (e.g. 15min, 60min), this script:

- Loads processed OHLC data via src.data.load_hourly_ohlc.
- Loads the model prediction checkpoint CSV produced by the backtest pipeline.
- Loads the standalone predictions CSV produced by scripts.generate_predictions_csv.
- Joins all three on Time and computes basic error metrics:

  - MAE/RMSE of checkpoint vs target price (Open shifted by ROWS_AHEAD).
  - MAE/RMSE of standalone vs the same target.
  - MAE of checkpoint vs standalone predictions.

Usage (from repo root):

    python -m scripts.audit_checkpoint_vs_predictions --frequency 60min
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
    get_hourly_data_csv_path,
    get_predictions_csv_path,
)
from src.data import load_hourly_ohlc


@dataclass
class CheckpointAuditConfig:
    frequency: str
    checkpoint_csv: Optional[str]
    predictions_csv: Optional[str]
    target_horizon: int = ROWS_AHEAD


def _load_hourly(frequency: str) -> pd.DataFrame:
    df = load_hourly_ohlc(frequency)
    if "Time" not in df.columns:
        raise ValueError(f"Hourly data for {frequency} is missing 'Time' column.")
    df = df.copy()
    df["Time"] = pd.to_datetime(df["Time"])
    return df


def _load_generic_predictions(path: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Time" not in df.columns or "predicted_price" not in df.columns:
        raise ValueError(
            f"{label} CSV {path} must contain 'Time' and 'predicted_price' columns; "
            f"found {list(df.columns)}",
        )
    df = df.copy()
    df["Time"] = pd.to_datetime(df["Time"])
    if df["Time"].duplicated().any():
        dup_count = int(df["Time"].duplicated().sum())
        print(f"[audit] WARNING: {dup_count} duplicate Time rows in {label}; keeping first occurrence.")
        df = df.sort_values("Time").drop_duplicates(subset=["Time"], keep="first")
    return df


def _compute_target_prices(df_hourly: pd.DataFrame, rows_ahead: int) -> pd.Series:
    if "Open" not in df_hourly.columns:
        raise ValueError("Hourly data is missing 'Open' column for target computation.")
    return df_hourly["Open"].shift(-rows_ahead)


def _mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return mae, rmse


def run_audit(cfg: CheckpointAuditConfig) -> None:
    hourly = _load_hourly(cfg.frequency)

    # Resolve file paths with sensible defaults.
    ckpt_path = (
        cfg.checkpoint_csv
        if cfg.checkpoint_csv is not None
        else f"backtests/nvda_{cfg.frequency}_model_predictions_checkpoint.csv"
    )
    preds_path = (
        cfg.predictions_csv
        if cfg.predictions_csv is not None
        else get_predictions_csv_path("nvda", cfg.frequency)
    )

    print(f"[audit] Using checkpoint CSV:  {ckpt_path}")
    print(f"[audit] Using standalone CSV: {preds_path}")

    ckpt = _load_generic_predictions(ckpt_path, label="checkpoint")
    preds = _load_generic_predictions(preds_path, label="standalone")

    # Join hourly with both prediction sources.
    merged = pd.merge(hourly, ckpt, on="Time", how="left", suffixes=("", "_ckpt"))
    merged = merged.rename(columns={"predicted_price": "ckpt_price"})

    merged = pd.merge(merged, preds, on="Time", how="left", suffixes=("", "_preds"))
    # After this, columns are: ... , ckpt_price, predicted_price (standalone).
    merged = merged.rename(columns={"predicted_price": "standalone_price"})

    n_hourly = len(hourly)
    n_ckpt = len(ckpt)
    n_preds = len(preds)

    print("[audit] Alignment summary:")
    print(f"  hourly bars:          {n_hourly}")
    print(f"  checkpoint rows:      {n_ckpt}")
    print(f"  standalone rows:      {n_preds}")

    n_missing_ckpt = int(merged["ckpt_price"].isna().sum())
    n_missing_preds = int(merged["standalone_price"].isna().sum())
    print(f"  missing checkpoint:   {n_missing_ckpt}")
    print(f"  missing standalone:   {n_missing_preds}")

    # Compute target prices and restrict to rows with complete data.
    merged["target_price"] = _compute_target_prices(merged, cfg.target_horizon)

    mask_all = merged[["target_price", "ckpt_price", "standalone_price"]].notna().all(axis=1)
    df_eval = merged.loc[mask_all].copy()
    if df_eval.empty:
        print("[audit] No rows with full target + both prediction sources; nothing to evaluate.")
        return

    print(f"[audit] Rows with full data: {len(df_eval)}")

    y_true = df_eval["target_price"].to_numpy(dtype=float)
    y_ckpt = df_eval["ckpt_price"].to_numpy(dtype=float)
    y_preds = df_eval["standalone_price"].to_numpy(dtype=float)

    ckpt_mae, ckpt_rmse = _mae_rmse(y_true, y_ckpt)
    preds_mae, preds_rmse = _mae_rmse(y_true, y_preds)
    diff_mae, diff_rmse = _mae_rmse(y_preds, y_ckpt)

    print("[audit] Overall error metrics vs target:")
    print(f"  checkpoint: MAE={ckpt_mae:.6f}  RMSE={ckpt_rmse:.6f}")
    print(f"  standalone: MAE={preds_mae:.6f}  RMSE={preds_rmse:.6f}")
    print("[audit] Disagreement between checkpoint and standalone:")
    print(f"  MAE(ckpt - standalone)={diff_mae:.6f}  RMSE={diff_rmse:.6f}")

    # Simple per-year breakdown.
    df_eval["year"] = pd.to_datetime(df_eval["Time"]).dt.year
    years = sorted(df_eval["year"].unique())
    print("[audit] Per-year metrics (vs target):")
    for year in years:
        sub = df_eval[df_eval["year"] == year]
        y_t = sub["target_price"].to_numpy(dtype=float)
        y_c = sub["ckpt_price"].to_numpy(dtype=float)
        y_p = sub["standalone_price"].to_numpy(dtype=float)
        c_mae, c_rmse = _mae_rmse(y_t, y_c)
        p_mae, p_rmse = _mae_rmse(y_t, y_p)
        d_mae, d_rmse = _mae_rmse(y_p, y_c)
        print(
            f"  {year}: n={len(sub):5d}  "
            f"ckpt(MAE={c_mae:.6f}, RMSE={c_rmse:.6f})  "
            f"stand(MAE={p_mae:.6f}, RMSE={p_rmse:.6f})  "
            f"diff(MAE={d_mae:.6f}, RMSE={d_rmse:.6f})",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit checkpoint vs standalone predictions and prices.",
    )
    parser.add_argument(
        "--frequency",
        type=str,
        default=CFG_FREQUENCY,
        help="Resample frequency (e.g. '15min', '60min'). Defaults to config FREQUENCY.",
    )
    parser.add_argument(
        "--checkpoint-csv",
        type=str,
        default=None,
        help=(
            "Optional override for checkpoint CSV. When omitted, uses "
            "backtests/nvda_<frequency>_model_predictions_checkpoint.csv."
        ),
    )
    parser.add_argument(
        "--predictions-csv",
        type=str,
        default=None,
        help=(
            "Optional override for standalone predictions CSV. When omitted, uses "
            "src.config.get_predictions_csv_path('nvda', frequency)."
        ),
    )

    args = parser.parse_args()
    cfg = CheckpointAuditConfig(
        frequency=args.frequency,
        checkpoint_csv=args.checkpoint_csv,
        predictions_csv=args.predictions_csv,
    )
    run_audit(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
