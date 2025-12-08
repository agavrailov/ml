"""Compare 15min prediction sources vs actual prices at sample points in 2025.

This script inspects:
- backtests/nvda_15min_model_predictions_checkpoint.csv  (backtest pipeline)
- backtests/nvda_15min_predictions.csv                    (standalone predictions)
- data/processed/nvda_15min.csv                           (resampled OHLC)
- data/raw/nvda_minute.csv                                (raw minute data)

and prints 10 random timestamps in 2025 with all four values side by side.
"""
from __future__ import annotations

import random

import pandas as pd


def _parse_times(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        cl = col.lower()
        if cl.startswith("time") or cl.startswith("datetime"):
            df[col] = pd.to_datetime(df[col])
    return df


def main() -> None:
    ckpt = _parse_times(pd.read_csv("backtests/nvda_15min_model_predictions_checkpoint.csv"))
    preds = _parse_times(pd.read_csv("backtests/nvda_15min_predictions.csv"))
    data15 = _parse_times(pd.read_csv("data/processed/nvda_15min.csv"))
    raw = _parse_times(pd.read_csv("data/raw/nvda_minute.csv"))

    start = pd.Timestamp("2025-01-01")
    end = pd.Timestamp("2025-12-31")
    mask_2025 = (data15["Time"] >= start) & (data15["Time"] <= end)
    times_2025 = data15.loc[mask_2025, "Time"].unique()
    if len(times_2025) == 0:
        print("NO_DATA_2025_IN_15MIN")
        return

    random.seed(0)
    k = min(10, len(times_2025))
    choices = sorted(random.sample(list(times_2025), k=k))

    raw_col = "DateTime" if "DateTime" in raw.columns else "Time"

    print("SAMPLE_POINTS_2025")
    for ts in choices:
        ts = pd.Timestamp(ts)
        r15 = data15.loc[data15["Time"] == ts].iloc[0]

        row_ckpt = ckpt.loc[ckpt["Time"] == ts]
        row_preds = preds.loc[preds["Time"] == ts]
        price_ckpt = float(row_ckpt["predicted_price"].iloc[0]) if not row_ckpt.empty else float("nan")
        price_preds = float(row_preds["predicted_price"].iloc[0]) if not row_preds.empty else float("nan")

        raw_row = raw.loc[raw[raw_col] == ts]
        raw_close = float(raw_row["Close"].iloc[0]) if not raw_row.empty else float("nan")

        print(
            f"TS={ts} "
            f"CLOSE15={r15['Close']:.4f} "
            f"RAW_CLOSE={raw_close:.4f} "
            f"CKPT={price_ckpt:.4f} "
            f"PREDS={price_preds:.4f}"
        )


if __name__ == "__main__":
    main()
