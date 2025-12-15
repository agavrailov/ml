"""Walkforward job handler - runs robustness analysis across parameter sets and folds."""

from __future__ import annotations

import pandas as pd
import numpy as np

from src import config as cfg
from src.backtest import run_backtest_on_dataframe, _compute_backtest_metrics
from src.core.contracts import WalkForwardRequest, WalkForwardResult
from src.data import load_hourly_ohlc
from src.jobs import store
from src.walkforward import (
    generate_walkforward_windows,
    infer_data_horizon,
    slice_df_by_window,
)


def run(job_id: str, request: WalkForwardRequest) -> None:
    """Execute walk-forward robustness analysis and write results to artifacts/.

    Args:
        job_id: Unique job identifier
        request: Walk-forward parameters including window config and parameter sets

    Writes:
        - artifacts/results.csv: Per-fold results for all parameter sets
        - artifacts/summary.csv: Aggregate Sharpe statistics per parameter set
        - result.json: Summary with best/worst parameter labels
    """
    frequency = request.frequency
    symbol = request.symbol
    t_start = request.t_start
    t_end = request.t_end
    test_span_months = request.test_span_months
    train_lookback_months = request.train_lookback_months
    min_lookback_months = request.min_lookback_months
    first_test_start = request.first_test_start
    predictions_csv = request.predictions_csv
    parameter_sets = request.parameter_sets or []

    # Load OHLC data
    df_full = load_hourly_ohlc(frequency)
    if df_full.empty:
        raise ValueError(f"No OHLC data available for frequency {frequency}")

    # Infer data horizon if not provided
    if not t_start or not t_end:
        data_t_start, data_t_end = infer_data_horizon(df_full)
        t_start = t_start or data_t_start
        t_end = t_end or data_t_end

    # Resolve predictions CSV path
    if not predictions_csv:
        predictions_csv = cfg.get_predictions_csv_path(symbol, frequency)

    # Generate walk-forward windows
    windows = generate_walkforward_windows(
        t_start,
        t_end,
        test_span_months=test_span_months,
        train_lookback_months=train_lookback_months,
        min_lookback_months=min_lookback_months,
        first_test_start=first_test_start,
    )

    if not windows:
        raise ValueError(f"No walk-forward windows generated for horizon [{t_start}, {t_end})")

    if not parameter_sets:
        raise ValueError("No parameter sets provided for robustness evaluation")

    # Run backtests across all parameter sets and folds
    rows: list[dict] = []
    
    total_runs = len(parameter_sets) * len(windows)
    run_count = 0

    for param_idx, p_row in enumerate(parameter_sets, start=1):
        label = str(p_row.get("label", "unnamed"))
        k_sigma_long = float(p_row["k_sigma_long"])
        k_sigma_short = float(p_row["k_sigma_short"])
        k_atr_long = float(p_row["k_atr_long"])
        k_atr_short = float(p_row["k_atr_short"])
        risk_pct = float(p_row["risk_per_trade_pct"])
        rr = float(p_row["reward_risk_ratio"])

        for fold_idx, (train_w, test_w) in enumerate(windows, start=1):
            run_count += 1
            # Update progress periodically
            if run_count % max(1, total_runs // 50) == 0 or run_count == 1:
                progress = run_count / total_runs
                store.update_progress(
                    job_id,
                    progress,
                    f"Param {param_idx}/{len(parameter_sets)}, Fold {fold_idx}/{len(windows)}",
                )
            test_df = slice_df_by_window(df_full, test_w)

            if test_df.empty:
                rows.append(
                    {
                        "param_label": label,
                        "fold_idx": fold_idx,
                        "train_start": train_w.start,
                        "train_end": train_w.end,
                        "test_start": test_w.start,
                        "test_end": test_w.end,
                        "sharpe_ratio": np.nan,
                        "total_return": np.nan,
                        "cagr": np.nan,
                        "max_drawdown": np.nan,
                        "win_rate": np.nan,
                        "profit_factor": np.nan,
                        "n_trades": 0,
                        "final_equity": np.nan,
                    }
                )
            else:
                bt_result = run_backtest_on_dataframe(
                    data=test_df,
                    initial_equity=cfg.INITIAL_EQUITY,
                    frequency=frequency,
                    prediction_mode="csv",
                    predictions_csv=predictions_csv,
                    risk_per_trade_pct=risk_pct,
                    reward_risk_ratio=rr,
                    k_sigma_long=k_sigma_long,
                    k_sigma_short=k_sigma_short,
                    k_atr_long=k_atr_long,
                    k_atr_short=k_atr_short,
                )
                metrics = _compute_backtest_metrics(
                    bt_result,
                    initial_equity=cfg.INITIAL_EQUITY,
                    data=test_df,
                )

                rows.append(
                    {
                        "param_label": label,
                        "fold_idx": fold_idx,
                        "train_start": train_w.start,
                        "train_end": train_w.end,
                        "test_start": test_w.start,
                        "test_end": test_w.end,
                        "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
                        "total_return": metrics.get("total_return", 0.0),
                        "cagr": metrics.get("cagr", 0.0),
                        "max_drawdown": metrics.get("max_drawdown", 0.0),
                        "win_rate": metrics.get("win_rate", 0.0),
                        "profit_factor": metrics.get("profit_factor", 0.0),
                        "n_trades": metrics.get("n_trades", 0),
                        "final_equity": bt_result.final_equity,
                    }
                )

    # Write detailed results CSV
    results_df = pd.DataFrame(rows)
    results_csv_path = store.write_df_csv_artifact(job_id, "results.csv", results_df)

    # Compute aggregate Sharpe statistics per parameter label
    sharpe_stats = (
        results_df.groupby("param_label")["sharpe_ratio"]
        .agg(
            mean_sharpe="mean",
            std_sharpe="std",
            min_sharpe="min",
            max_sharpe="max",
            n_folds="count",
        )
        .reset_index()
    )

    # Add fraction of positive Sharpe ratios
    frac_positive = (
        results_df.assign(is_pos=results_df["sharpe_ratio"] > 0)
        .groupby("param_label")["is_pos"]
        .mean()
        .reset_index(name="p_sharpe_gt_0")
    )
    sharpe_stats = sharpe_stats.merge(frac_positive, on="param_label", how="left")

    # Compute robustness score (mean/std Sharpe)
    sharpe_stats["robustness_score"] = sharpe_stats["mean_sharpe"] / sharpe_stats[
        "std_sharpe"
    ].replace(0, np.nan)

    # Write summary CSV
    summary_csv_path = store.write_df_csv_artifact(job_id, "summary.csv", sharpe_stats)

    # Prepare result summary
    best_row = sharpe_stats.sort_values("mean_sharpe", ascending=False).iloc[0]
    worst_row = sharpe_stats.sort_values("mean_sharpe", ascending=True).iloc[0]

    result = WalkForwardResult(
        results_csv=str(results_csv_path),
        summary_csv=str(summary_csv_path),
    )

    store.write_result(
        job_id,
        {
            **result.to_dict(),
            "summary": {
                "n_param_sets": len(sharpe_stats),
                "n_folds": len(windows),
                "best_label": str(best_row["param_label"]),
                "best_mean_sharpe": float(best_row["mean_sharpe"]),
                "worst_label": str(worst_row["param_label"]),
                "worst_mean_sharpe": float(worst_row["mean_sharpe"]),
            },
        },
    )
