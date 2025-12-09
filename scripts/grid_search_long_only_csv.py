from __future__ import annotations

import argparse
import itertools
from typing import Iterable

import numpy as np
import pandas as pd

from src.backtest import run_backtest_for_ui
from src.config import (
    FREQUENCY,
    RISK_PER_TRADE_PCT,
    REWARD_RISK_RATIO,
    K_SIGMA_LONG,
    K_ATR_LONG,
)


def _frange(start: float, stop: float, step: float) -> Iterable[float]:
    """Simple inclusive float range with a safety cap on steps."""
    if step <= 0:
        yield start
        return
    current = start
    for _ in range(1000):  # safety cap
        if current > stop + 1e-9:
            break
        yield float(current)
        current += step


def run_grid_search(
    frequency: str,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    """Run a small grid search for long-only CSV-mode backtests.

    We sweep k_sigma_long, k_atr_long, risk_per_trade_pct, reward_risk_ratio
    around the current config defaults and return a DataFrame of results
    sorted by total_return descending.
    """

    # Center grid around current defaults with modest spreads.
    base_k_sigma = float(K_SIGMA_LONG)
    base_k_atr = float(K_ATR_LONG)
    base_risk = float(RISK_PER_TRADE_PCT)
    base_rr = float(REWARD_RISK_RATIO)

    k_sigma_vals = sorted(set([
        round(v, 2)
        for v in [
            max(0.1, base_k_sigma * 0.5),
            max(0.1, base_k_sigma * 0.75),
            base_k_sigma,
            base_k_sigma * 1.25,
            base_k_sigma * 1.5,
        ]
    ]))

    k_atr_vals = sorted(set([
        round(v, 2)
        for v in [
            max(0.1, base_k_atr * 0.5),
            max(0.1, base_k_atr * 0.75),
            base_k_atr,
            base_k_atr * 1.25,
            base_k_atr * 1.5,
        ]
    ]))

    risk_vals = sorted(set([
        round(v, 4)
        for v in [
            max(0.005, base_risk * 0.5),
            base_risk,
            min(0.05, base_risk * 1.5),
        ]
    ]))

    rr_vals = sorted(set([
        round(v, 2)
        for v in [
            max(1.5, base_rr * 0.5),
            max(1.5, base_rr * 0.75),
            base_rr,
            base_rr * 1.25,
        ]
    ]))

    combos = list(itertools.product(k_sigma_vals, k_atr_vals, risk_vals, rr_vals))

    print(f"Running long-only CSV grid search on {frequency} with {len(combos)} combos...")

    rows: list[dict] = []
    for idx, (k_sigma_long, k_atr_long, risk_pct, rr) in enumerate(combos, start=1):
        print(
            f"[{idx}/{len(combos)}] k_sigma_long={k_sigma_long:.2f}, "
            f"k_atr_long={k_atr_long:.2f}, risk={risk_pct:.3f}, rr={rr:.2f}",
            flush=True,
        )

        equity_df, trades_df, metrics = run_backtest_for_ui(
            frequency=frequency,
            prediction_mode="csv",
            start_date=start_date,
            end_date=end_date,
            predictions_csv=None,  # default NVDA path
            risk_per_trade_pct=risk_pct,
            reward_risk_ratio=rr,
            k_sigma_err=None,
            k_atr_min_tp=None,
            k_sigma_long=k_sigma_long,
            k_sigma_short=k_sigma_long,  # symmetric for this search
            k_atr_long=k_atr_long,
            k_atr_short=k_atr_long,
            enable_longs=True,
            allow_shorts=False,
        )

        rows.append(
            {
                "k_sigma_long": k_sigma_long,
                "k_atr_long": k_atr_long,
                "risk_per_trade_pct": risk_pct,
                "reward_risk_ratio": rr,
                "total_return": metrics.get("total_return", 0.0),
                "cagr": metrics.get("cagr", 0.0),
                "max_drawdown": metrics.get("max_drawdown", 0.0),
                "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
                "win_rate": metrics.get("win_rate", 0.0),
                "n_trades": metrics.get("n_trades", 0),
                "final_equity": metrics.get("final_equity", 0.0),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df.sort_values(by="total_return", ascending=False, inplace=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search long-only CSV backtest params.")
    parser.add_argument("--frequency", type=str, default=FREQUENCY, help="e.g. 15min, 60min")
    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD or None")
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD or None")
    parser.add_argument("--top", type=int, default=10, help="Number of top rows to print")

    args = parser.parse_args()

    df = run_grid_search(
        frequency=args.frequency,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    if df.empty:
        print("NO_RESULTS")
        return

    top_n = max(1, min(int(args.top), len(df)))
    print("\nTOP_RESULTS")
    print(df.head(top_n).to_string(index=False))


if __name__ == "__main__":  # pragma: no cover
    main()
