"""Walk-forward evaluation of the top-N strategy parameter sets for any symbol.

Loads the top-N rows from the grid search results CSV, runs walk-forward
evaluation for each param set, and prints a ranked summary table sorted by
mean out-of-sample Sharpe across folds.

Usage (from repo root):
    python -m scripts.walkforward_param_sweep --symbol JPM
    python -m scripts.walkforward_param_sweep --symbol XOM --top 20 --test-span-months 3
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

FREQUENCY = "60min"
RISK_PER_TRADE_PCT = 0.015
REWARD_RISK_RATIO = 2.5


def _paths(symbol: str) -> dict[str, str]:
    s = symbol.lower()
    return {
        "predictions": os.path.join("backtests", f"{s}_{FREQUENCY}_model_predictions_checkpoint.csv"),
        "grid_csv":    os.path.join("backtests", f"{s}_grid_search_top20.csv"),
    }


def run_sweep(
    symbol: str,
    top_n: int,
    test_span_months: int,
    first_test_start: str,
) -> pd.DataFrame:
    from scripts.run_walkforward_backtest import run_walkforward

    paths = _paths(symbol)

    if not os.path.exists(paths["grid_csv"]):
        raise FileNotFoundError(
            f"Grid search results not found at {paths['grid_csv']}.\n"
            f"Run the grid search first:\n"
            f"  python -m scripts.grid_search_strategy --symbol {symbol} "
            f"--no-save --save-results {paths['grid_csv']}"
        )

    grid_df = pd.read_csv(paths["grid_csv"])
    param_sets = grid_df.head(top_n)
    total = len(param_sets)

    print(f"Walk-forward param sweep: {symbol} {FREQUENCY}")
    print(f"Evaluating top {total} param sets | test_span={test_span_months}m | first_test={first_test_start}")
    print(f"Predictions: {paths['predictions']}")
    print()

    summary_rows: list[dict] = []

    for rank, (_, row) in enumerate(param_sets.iterrows(), start=1):
        ksl  = float(row["k_sigma_long"])
        kss  = float(row["k_sigma_short"])
        kal  = float(row["k_atr_long"])
        kas  = float(row["k_atr_short"])

        print(
            f"[{rank:2d}/{total}] ksl={ksl:.2f} kss={kss:.2f} "
            f"kal={kal:.2f} kas={kas:.2f} "
            f"(in-sample Sharpe={row['sharpe_ratio']:.2f})",
            flush=True,
        )

        try:
            fold_df = run_walkforward(
                frequency=FREQUENCY,
                tsteps=5,
                symbol=symbol,
                test_span_months=test_span_months,
                train_lookback_months=24,
                min_lookback_months=6,
                t_start=None,
                t_end=None,
                predictions_csv=paths["predictions"],
                first_test_start=first_test_start,
                k_sigma_long=ksl,
                k_sigma_short=kss,
                k_atr_long=kal,
                k_atr_short=kas,
                risk_per_trade_pct=RISK_PER_TRADE_PCT,
                reward_risk_ratio=REWARD_RISK_RATIO,
            )
        except Exception as exc:
            print(f"    ERROR: {exc}", flush=True)
            continue

        n_folds = len(fold_df)
        if n_folds == 0:
            continue

        mean_sharpe  = fold_df["sharpe_ratio"].mean()
        mean_pf      = fold_df["profit_factor"].mean()
        mean_dd      = fold_df["max_drawdown"].mean()
        mean_ret     = fold_df["total_return"].mean()
        mean_trades  = fold_df["n_trades"].mean()
        pct_positive = (fold_df["total_return"] > 0).mean()
        min_sharpe   = fold_df["sharpe_ratio"].min()

        print(
            f"         -> folds={n_folds} mean_sharpe={mean_sharpe:.2f} "
            f"min_sharpe={min_sharpe:.2f} mean_pf={mean_pf:.2f} "
            f"mean_dd={mean_dd*100:.1f}% pct_pos={pct_positive*100:.0f}%",
            flush=True,
        )

        summary_rows.append({
            "rank_insample":    rank,
            "k_sigma_long":     ksl,
            "k_sigma_short":    kss,
            "k_atr_long":       kal,
            "k_atr_short":      kas,
            "insample_sharpe":  row["sharpe_ratio"],
            "n_folds":          n_folds,
            "oos_sharpe_mean":  mean_sharpe,
            "oos_sharpe_min":   min_sharpe,
            "oos_pf_mean":      mean_pf,
            "oos_dd_mean":      mean_dd,
            "oos_ret_mean":     mean_ret,
            "oos_trades_mean":  mean_trades,
            "pct_pos_folds":    pct_positive,
        })

    df = pd.DataFrame(summary_rows)
    if df.empty:
        return df

    df.sort_values(by="oos_sharpe_mean", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward sweep of top strategy param sets.")
    parser.add_argument("--symbol",             type=str, required=True,        help="Symbol to evaluate (e.g. JPM, XOM)")
    parser.add_argument("--top",                type=int, default=20,           help="Number of param sets to evaluate (default 20)")
    parser.add_argument("--test-span-months",   type=int, default=3,            help="Walk-forward test window in months (default 3)")
    parser.add_argument("--first-test-start",   type=str, default="2024-06-01", help="First test window start date")
    parser.add_argument("--output",             type=str, default=None,         help="Save ranked summary to this CSV path")
    args = parser.parse_args()
    symbol = args.symbol.upper()

    df = run_sweep(
        symbol=symbol,
        top_n=args.top,
        test_span_months=args.test_span_months,
        first_test_start=args.first_test_start,
    )

    if df.empty:
        print("No results collected.")
        return

    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)

    print(f"\n=== WALK-FORWARD RANKING (sorted by OOS mean Sharpe) ===")
    cols = [
        "rank_insample", "k_sigma_long", "k_sigma_short", "k_atr_long", "k_atr_short",
        "oos_sharpe_mean", "oos_sharpe_min", "oos_pf_mean", "oos_dd_mean",
        "oos_trades_mean", "pct_pos_folds",
    ]
    print(df[cols].to_string(index=True))

    best = df.iloc[0]
    print(f"\n=== OOS WINNER ===")
    print(f"  k_sigma_long  = {best.k_sigma_long:.2f}")
    print(f"  k_sigma_short = {best.k_sigma_short:.2f}")
    print(f"  k_atr_long    = {best.k_atr_long:.2f}")
    print(f"  k_atr_short   = {best.k_atr_short:.2f}")
    print(f"  OOS Sharpe (mean/min) = {best.oos_sharpe_mean:.3f} / {best.oos_sharpe_min:.3f}")
    print(f"  OOS PF (mean)         = {best.oos_pf_mean:.3f}")
    print(f"  OOS DD (mean)         = {best.oos_dd_mean*100:.1f}%")
    print(f"  Folds positive        = {best.pct_pos_folds*100:.0f}%")

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
