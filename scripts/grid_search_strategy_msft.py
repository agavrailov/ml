"""Strategy parameter grid search for MSFT (60min, bidirectional).

Sweeps k_sigma_long/short and k_atr_long/short over a coarse grid, ranks
results by Sharpe ratio, and saves the best combo to
``configs/symbols/MSFT/active.json`` via ``save_strategy_defaults``.

Uses the pre-computed model predictions checkpoint so inference is not
re-run on every combo (fast: ~2-3 min for 1,296 combos).

Usage (from repo root):
    python scripts/grid_search_strategy_msft.py
    python scripts/grid_search_strategy_msft.py --top 30 --no-save
"""
from __future__ import annotations

import argparse
import itertools
import os
from typing import Iterable

import pandas as pd

SYMBOL = "MSFT"
FREQUENCY = "60min"
PREDICTIONS_CSV = os.path.join("backtests", "msft_60min_model_predictions_checkpoint.csv")
RISK_PER_TRADE_PCT = 0.015   # keep fixed (from placeholder config)
REWARD_RISK_RATIO = 2.5      # keep fixed


def _frange(start: float, stop: float, step: float) -> Iterable[float]:
    """Inclusive float range with a safety cap on iterations."""
    if step <= 0:
        yield start
        return
    current = start
    for _ in range(1000):
        if current > stop + 1e-9:
            break
        yield round(float(current), 4)
        current += step


def run_grid_search(
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    from src.backtest import run_backtest_for_ui

    if not os.path.exists(PREDICTIONS_CSV):
        raise FileNotFoundError(
            f"Predictions checkpoint not found: {PREDICTIONS_CSV}\n"
            "Run a full backtest first:\n"
            "  python -m src.backtest --symbol MSFT --frequency 60min "
            "--prediction-mode model --report"
        )

    k_sigma_long_vals  = list(_frange(0.20, 0.70, 0.10))  # 6 values
    k_sigma_short_vals = list(_frange(0.30, 0.80, 0.10))  # 6 values
    k_atr_long_vals    = list(_frange(0.25, 0.75, 0.10))  # 6 values
    k_atr_short_vals   = list(_frange(0.30, 0.80, 0.10))  # 6 values

    combos = list(itertools.product(
        k_sigma_long_vals,
        k_sigma_short_vals,
        k_atr_long_vals,
        k_atr_short_vals,
    ))
    total = len(combos)
    print(f"Grid search: {SYMBOL} {FREQUENCY} | {total} combos | predictions from checkpoint")
    print(f"Params: k_sigma_long x k_sigma_short x k_atr_long x k_atr_short")
    print(f"        {k_sigma_long_vals} x {k_sigma_short_vals}")
    print(f"        {k_atr_long_vals} x {k_atr_short_vals}")
    print()

    rows: list[dict] = []
    for idx, (ksl, kss, kal, kas) in enumerate(combos, start=1):
        if idx % 100 == 0 or idx == 1:
            print(f"  [{idx:4d}/{total}] ksl={ksl:.2f} kss={kss:.2f} kal={kal:.2f} kas={kas:.2f}", flush=True)

        try:
            _eq, _trades, metrics = run_backtest_for_ui(
                symbol=SYMBOL,
                frequency=FREQUENCY,
                prediction_mode="csv",
                predictions_csv=PREDICTIONS_CSV,
                start_date=start_date,
                end_date=end_date,
                risk_per_trade_pct=RISK_PER_TRADE_PCT,
                reward_risk_ratio=REWARD_RISK_RATIO,
                k_sigma_long=ksl,
                k_sigma_short=kss,
                k_atr_long=kal,
                k_atr_short=kas,
                enable_longs=True,
                allow_shorts=True,
            )
        except Exception as exc:
            print(f"    ERROR: {exc}", flush=True)
            continue

        rows.append({
            "k_sigma_long":     ksl,
            "k_sigma_short":    kss,
            "k_atr_long":       kal,
            "k_atr_short":      kas,
            "sharpe_ratio":     metrics.get("sharpe_ratio", 0.0),
            "profit_factor":    metrics.get("profit_factor", 0.0),
            "total_return":     metrics.get("total_return", 0.0),
            "cagr":             metrics.get("cagr", 0.0),
            "max_drawdown":     metrics.get("max_drawdown", 0.0),
            "win_rate":         metrics.get("win_rate", 0.0),
            "n_trades":         metrics.get("n_trades", 0),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df.sort_values(by="sharpe_ratio", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Strategy grid search for MSFT 60min.")
    parser.add_argument("--start-date",    type=str,  default=None,  help="YYYY-MM-DD")
    parser.add_argument("--end-date",      type=str,  default=None,  help="YYYY-MM-DD")
    parser.add_argument("--top",           type=int,  default=20,    help="Rows to print (default 20)")
    parser.add_argument("--no-save",       action="store_true",      help="Skip saving best params to config")
    parser.add_argument("--save-results",  type=str,  default=None,  help="Path to write full results CSV")
    args = parser.parse_args()

    df = run_grid_search(start_date=args.start_date, end_date=args.end_date)

    if df.empty:
        print("No results collected.")
        return

    top_n = min(args.top, len(df))
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(f"\n=== TOP {top_n} by Sharpe Ratio ===")
    print(df.head(top_n).to_string(index=True))

    best = df.iloc[0]
    print(f"\n=== WINNER ===")
    print(f"  k_sigma_long  = {best.k_sigma_long:.2f}")
    print(f"  k_sigma_short = {best.k_sigma_short:.2f}")
    print(f"  k_atr_long    = {best.k_atr_long:.2f}")
    print(f"  k_atr_short   = {best.k_atr_short:.2f}")
    print(f"  Sharpe        = {best.sharpe_ratio:.3f}")
    print(f"  PF            = {best.profit_factor:.3f}")
    print(f"  Return        = {best.total_return * 100:.1f}%")
    print(f"  Max DD        = {best.max_drawdown * 100:.1f}%")
    print(f"  Trades        = {int(best.n_trades)}")

    if args.save_results:
        df.to_csv(args.save_results, index=False)
        print(f"\nFull results saved to {args.save_results}")

    if not args.no_save:
        from src.core.config_resolver import save_strategy_defaults
        save_strategy_defaults(
            symbol=SYMBOL,
            frequency=FREQUENCY,
            risk_per_trade_pct=RISK_PER_TRADE_PCT,
            reward_risk_ratio=REWARD_RISK_RATIO,
            k_sigma_long=float(best.k_sigma_long),
            k_sigma_short=float(best.k_sigma_short),
            k_atr_long=float(best.k_atr_long),
            k_atr_short=float(best.k_atr_short),
            enable_longs=True,
            allow_shorts=True,
            source="grid_search_strategy_msft.py",
        )
        print(f"\nSaved to configs/symbols/MSFT/active.json")
    else:
        print("\n(--no-save: skipped writing config)")


if __name__ == "__main__":
    main()
