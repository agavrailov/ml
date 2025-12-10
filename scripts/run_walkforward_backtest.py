from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from src import config as cfg
from src.data import load_hourly_ohlc
from src.walkforward import TimeWindow, generate_walkforward_windows, slice_df_by_window, infer_data_horizon
from src.backtest import run_backtest_on_dataframe, _compute_backtest_metrics  # type: ignore[attr-defined]


def run_walkforward(
    frequency: str,
    tsteps: int,
    symbol: str,
    test_span_months: int,
    train_lookback_months: int,
    min_lookback_months: int,
    t_start: str | None,
    t_end: str | None,
    predictions_csv: str | None,
    first_test_start: str | None,
) -> pd.DataFrame:
    """Run 3-month walk-forward backtests over the available data.

    This script does **not** retrain the model per fold; it assumes that
    per-bar predictions are already available in ``predictions_csv`` and
    evaluates strategy performance across successive 3-month test windows.
    """

    df_full = load_hourly_ohlc(frequency)
    if "Time" not in df_full.columns or df_full.empty:
        raise SystemExit("Hourly OHLC data must be non-empty and contain a 'Time' column.")

    # Determine overall horizon (optionally overridden by CLI args).
    data_t_start, data_t_end = infer_data_horizon(df_full)
    eff_t_start = t_start or data_t_start
    eff_t_end = t_end or data_t_end

    windows: List[Tuple[TimeWindow, TimeWindow]] = generate_walkforward_windows(
        eff_t_start,
        eff_t_end,
        test_span_months=test_span_months,
        train_lookback_months=train_lookback_months,
        min_lookback_months=min_lookback_months,
        first_test_start=first_test_start,
    )
    if not windows:
        raise SystemExit("No walk-forward windows generated for the supplied horizon.")

    if predictions_csv is None:
        predictions_csv = cfg.get_predictions_csv_path(symbol, frequency)

    if not os.path.exists(predictions_csv):
        raise SystemExit(
            f"Predictions CSV not found at {predictions_csv}. "
            "Generate it first (e.g. via scripts.generate_predictions_csv or the Streamlit UI)."
        )

    rows: list[dict] = []
    for fold_idx, (train_w, test_w) in enumerate(windows, start=1):
        test_df = slice_df_by_window(df_full, test_w)
        if test_df.empty:
            continue

        result = run_backtest_on_dataframe(
            test_df,
            initial_equity=cfg.INITIAL_EQUITY,
            frequency=frequency,
            prediction_mode="csv",
            predictions_csv=predictions_csv,
        )
        metrics = _compute_backtest_metrics(result, initial_equity=cfg.INITIAL_EQUITY, data=test_df)

        row = {
            "fold_idx": fold_idx,
            "train_start": train_w.start,
            "train_end": train_w.end,
            "test_start": test_w.start,
            "test_end": test_w.end,
            "total_return": metrics.get("total_return", 0.0),
            "cagr": metrics.get("cagr", 0.0),
            "max_drawdown": metrics.get("max_drawdown", 0.0),
            "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
            "win_rate": metrics.get("win_rate", 0.0),
            "profit_factor": metrics.get("profit_factor", 0.0),
            "n_trades": metrics.get("n_trades", 0),
            "final_equity": result.final_equity,
        }
        rows.append(row)

    if not rows:
        raise SystemExit("Walk-forward backtests produced no results (all windows empty?).")

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 3-month walk-forward backtests and write a summary CSV.")
    parser.add_argument("--frequency", type=str, default=cfg.FREQUENCY, help="Resample frequency, e.g. 15min, 60min.")
    parser.add_argument("--tsteps", type=int, default=cfg.TSTEPS, help="TSTEPS used by the model (only used in output filename).")
    parser.add_argument("--symbol", type=str, default="nvda", help="Symbol label for predictions CSV and summary filename.")
    parser.add_argument("--test-span-months", type=int, default=3, help="Length of each test window in months.")
    parser.add_argument("--train-lookback-months", type=int, default=24, help="Target training lookback in months (for reporting only).")
    parser.add_argument("--min-lookback-months", type=int, default=18, help="Minimum desired lookback for the first test window.")
    parser.add_argument("--first-test-start", type=str, default="2023-07-01", help="First test window start date (YYYY-MM-DD).")
    parser.add_argument("--t-start", type=str, default=None, help="Optional overall start date (YYYY-MM-DD) for walk-forward horizon.")
    parser.add_argument("--t-end", type=str, default=None, help="Optional overall end date (YYYY-MM-DD) for walk-forward horizon.")
    parser.add_argument("--predictions-csv", type=str, default=None, help="Path to per-bar predictions CSV to use.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output CSV path for the walk-forward summary. "
            "Defaults to backtests/walkforward_{symbol}_{frequency}_tsteps{tsteps}.csv."
        ),
    )

    args = parser.parse_args()

    df_summary = run_walkforward(
        frequency=args.frequency,
        tsteps=args.tsteps,
        symbol=args.symbol,
        test_span_months=args.test_span_months,
        train_lookback_months=args.train_lookback_months,
        min_lookback_months=args.min_lookback_months,
        t_start=args.t_start,
        t_end=args.t_end,
        predictions_csv=args.predictions_csv,
        first_test_start=args.first_test_start,
    )

    if args.output is None:
        out_dir = Path(cfg.BASE_DIR) / "backtests"  # type: ignore[attr-defined]
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / f"walkforward_{args.symbol.lower()}_{args.frequency}_tsteps{args.tsteps}.csv"
    else:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)

    df_summary.to_csv(output, index=False)
    print(f"Wrote walk-forward summary for {len(df_summary)} folds to {output}")


if __name__ == "__main__":  # pragma: no cover
    main()
