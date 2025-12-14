"""Optimize job handler - runs parameter grid search."""

from __future__ import annotations

from itertools import product

import pandas as pd

from src.backtest import run_backtest_for_ui
from src.config import get_predictions_csv_path
from src.core.contracts import OptimizeRequest, OptimizeResult
from src.jobs import store


def run(job_id: str, request: OptimizeRequest) -> None:
    """Execute optimization grid search and write results to artifacts/.

    Args:
        job_id: Unique job identifier
        request: Optimization parameters including grid definition

    Writes:
        - artifacts/results.csv: Full grid results sorted by Sharpe ratio
        - result.json: Summary with best Sharpe and total return
    """
    frequency = request.frequency
    start_date = request.start_date
    end_date = request.end_date
    trade_side = request.trade_side
    param_grid = request.param_grid or {}

    # Resolve predictions CSV path
    predictions_csv = request.predictions_csv
    if not predictions_csv:
        predictions_csv = get_predictions_csv_path("nvda", frequency)

    # Build parameter ranges from grid definition
    param_ranges: dict[str, list[float]] = {}
    for param_name, grid_spec in param_grid.items():
        if isinstance(grid_spec, dict):
            # Grid search: {start, stop, step}
            start = float(grid_spec.get("start", 0.0))
            stop = float(grid_spec.get("stop", start))
            step = float(grid_spec.get("step", 1.0))

            values: list[float] = []
            if step > 0 and stop >= start:
                current = start
                for _ in range(1000):  # Safety cap
                    if current > stop + 1e-9:
                        break
                    values.append(float(current))
                    current += step
            param_ranges[param_name] = values if values else [start]
        else:
            # Fixed value
            param_ranges[param_name] = [float(grid_spec)]

    # Determine enable_longs and allow_shorts from trade_side
    if trade_side == "Long only":
        enable_longs = True
        allow_shorts = False
    elif trade_side == "Short only":
        enable_longs = False
        allow_shorts = True
    else:  # "Long & short"
        enable_longs = True
        allow_shorts = True

    # Run grid search
    results_rows: list[dict] = []
    names = list(param_ranges.keys())
    all_values = [param_ranges[n] for n in names]

    for combo in product(*all_values):
        combo_params = dict(zip(names, combo))

        # Extract strategy parameters with defaults
        k_sigma_long = combo_params.get("k_sigma_long")
        k_sigma_short = combo_params.get("k_sigma_short")
        k_atr_long = combo_params.get("k_atr_long")
        k_atr_short = combo_params.get("k_atr_short")
        risk_per_trade_pct = combo_params.get("risk_per_trade_pct")
        reward_risk_ratio = combo_params.get("reward_risk_ratio")

        equity_df, trades_df, metrics = run_backtest_for_ui(
            frequency=frequency,
            prediction_mode="csv",
            start_date=start_date,
            end_date=end_date,
            predictions_csv=predictions_csv,
            risk_per_trade_pct=risk_per_trade_pct,
            reward_risk_ratio=reward_risk_ratio,
            k_sigma_err=None,
            k_atr_min_tp=None,
            k_sigma_long=k_sigma_long,
            k_sigma_short=k_sigma_short,
            k_atr_long=k_atr_long,
            k_atr_short=k_atr_short,
            enable_longs=enable_longs,
            allow_shorts=allow_shorts,
        )

        results_rows.append(
            {
                "k_sigma_long": k_sigma_long,
                "k_sigma_short": k_sigma_short,
                "k_atr_long": k_atr_long,
                "k_atr_short": k_atr_short,
                "risk_per_trade_pct": risk_per_trade_pct,
                "reward_risk_ratio": reward_risk_ratio,
                "total_return": metrics.get("total_return", 0.0),
                "cagr": metrics.get("cagr", 0.0),
                "max_drawdown": metrics.get("max_drawdown", 0.0),
                "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
                "profit_factor": metrics.get("profit_factor", 0.0),
                "win_rate": metrics.get("win_rate", 0.0),
                "n_trades": metrics.get("n_trades", 0),
                "final_equity": metrics.get("final_equity", 0.0),
            }
        )

    # Sort by Sharpe ratio descending
    results_df = pd.DataFrame(results_rows)
    results_df = results_df.sort_values(by="sharpe_ratio", ascending=False, na_position="last")

    # Write results CSV
    results_csv_path = store.write_df_csv_artifact(job_id, "results.csv", results_df)

    # Compute summary
    best_sharpe = float(results_df["sharpe_ratio"].max()) if not results_df.empty else 0.0
    best_total_return = (
        float(results_df["total_return"].max()) if not results_df.empty else 0.0
    )
    n_runs = len(results_df)

    result = OptimizeResult(
        results_csv=results_csv_path,
        summary={
            "best_sharpe": best_sharpe,
            "best_total_return": best_total_return,
            "n_runs": n_runs,
        },
    )

    store.write_result(job_id, result.to_dict())
