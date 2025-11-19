"""Plot equity curve and trade density over time for a backtest.

This script expects:

- An equity CSV from `src.backtest.py` with columns: `step`, `equity`.
- A trades CSV from `src.backtest.py` with columns including: `entry_index`, `exit_index`.
- The underlying OHLC CSV with a `Time` column (for mapping steps to timestamps).

Usage (from repo root):

    python -m scripts.plot_backtest_diagnostics \
        --equity backtests/nvda_60min_equity.csv \
        --trades backtests/nvda_60min_trades.csv \
        --price-csv data/processed/nvda_60min.csv

The script will generate a matplotlib figure with:
- Top: equity curve over time.
- Bottom: trade count per month (entry density).

The figure is saved to ``backtests/backtest_diagnostics.png`` so that it can be
opened later without blocking the CLI or tests.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_and_trade_density(
    equity_path: str,
    trades_path: str,
    price_csv_path: str,
) -> None:
    equity = pd.read_csv(equity_path)
    trades = pd.read_csv(trades_path)
    prices = pd.read_csv(price_csv_path)

    if "Time" not in prices.columns:
        raise ValueError(f"Expected 'Time' column in {price_csv_path}.")

    # Align equity with timestamps by position (step index -> row index).
    n = min(len(equity), len(prices))
    equity = equity.iloc[:n].copy()
    equity["Time"] = pd.to_datetime(prices["Time"].iloc[:n])

    # Build a trade timestamp series based on entry_index -> Time.
    if "entry_index" not in trades.columns:
        raise ValueError("Trades CSV must contain 'entry_index' column.")

    entry_times = []
    for idx in trades["entry_index"]:
        if 0 <= idx < len(prices):
            entry_times.append(prices["Time"].iloc[idx])
    trades_series = pd.to_datetime(pd.Series(entry_times))

    # Aggregate trade entries per month.
    trade_counts = trades_series.dt.to_period("M").value_counts().sort_index()

    fig, (ax_eq, ax_tr) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Equity curve
    ax_eq.plot(equity["Time"], equity["equity"], label="Equity")
    ax_eq.set_ylabel("Equity")
    ax_eq.set_title("Equity curve")
    ax_eq.grid(True, alpha=0.3)

    # Trade density
    ax_tr.bar(trade_counts.index.to_timestamp(), trade_counts.values, width=20, align="center")
    ax_tr.set_ylabel("Trades / month")
    ax_tr.set_title("Trade entry density")
    ax_tr.grid(True, alpha=0.3)

    fig.autofmt_xdate()
    plt.tight_layout()

    # Save diagnostics figure under backtests/ (non-blocking for scripts/tests).\r
    out_dir = Path("backtests")\r
    out_dir.mkdir(parents=True, exist_ok=True)\r
    out_path = out_dir / "backtest_diagnostics.png"\r
    fig.savefig(out_path, dpi=150)\r
    plt.close(fig)\r
    print(f"Saved diagnostics plot to {out_path}")\r
\r
\r
def main() -> None:\r
    parser = argparse.ArgumentParser(description="Plot equity and trade density for a backtest.")
    parser.add_argument("--equity", required=True, help="Path to equity CSV (from src.backtest).")
    parser.add_argument("--trades", required=True, help="Path to trades CSV (from src.backtest).")
    parser.add_argument("--price-csv", required=True, help="Path to underlying OHLC CSV (with Time column).")

    args = parser.parse_args()
    plot_equity_and_trade_density(args.equity, args.trades, args.price_csv)


if __name__ == "__main__":  # pragma: no cover
    main()
