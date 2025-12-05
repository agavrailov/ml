from __future__ import annotations

from pathlib import Path

import nbformat


NB_PATH = Path("notebooks/backtest_param_grid_local.ipynb")


def main() -> None:
    nb = nbformat.read(NB_PATH, as_version=4)

    # Canonical multi-line sources for key cells. These will be split into
    # lists of lines later so that no individual string contains a "\\n".
    intro_markdown_src = """# Backtest Parameter Grid (Local Jupyter)

This notebook runs parameter grid explorations over the trading strategy using the existing backtest engine, **for local Jupyter**.

Steps:
1. Start Jupyter from the project root (where `src/` lives).
2. Open this notebook from the `notebooks/` directory.
3. Run the cells from top to bottom.
"""

    new_root_src = """from pathlib import Path
import sys

# Assume this notebook lives in '<project_root>/notebooks'.
# Derive the project root as the parent directory and ensure it is on sys.path
# so that 'src' is importable without changing the process working directory.
project_root = Path().resolve().parent

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

print("Project root:", project_root)
print("Has src?:", (project_root / "src").exists())
"""

    imports_src = """import itertools
import os

import numpy as np
import pandas as pd

from src.backtest import (
    _compute_atr_series,
    _compute_backtest_metrics,
    _make_naive_prediction_provider,
    _make_model_prediction_provider,
    _load_predictions_csv,
    _make_csv_prediction_provider,
)
from src.backtest_engine import BacktestConfig, run_backtest
from src.strategy import StrategyConfig
from src.config import FREQUENCY, get_hourly_data_csv_path, get_predictions_csv_path
"""

    new_config_src = """# Configuration for this notebook run
symbol = "nvda"
frequency = FREQUENCY  # Default from global config; override here if desired, e.g. "60min"
initial_equity = 10_000.0
prediction_mode = "csv"  # "naive", "model", or "csv"

csv_path = get_hourly_data_csv_path(frequency)
print("Using OHLC data from:", csv_path)
data = pd.read_csv(csv_path)

# Basic sanity check
required_cols = {"Open", "High", "Low", "Close"}
missing = required_cols - set(data.columns)
if missing:
    raise ValueError(f"Data file {csv_path} is missing required columns: {missing}")

# Compute ATR(14) and a scalar ATR proxy
atr_series = _compute_atr_series(data, window=14)
atr_like = float(atr_series.dropna().mean()) if not atr_series.dropna().empty else 1.0
print(f"Mean ATR proxy: {atr_like:.4f}")

# Build prediction provider
if prediction_mode == "naive":
    provider = _make_naive_prediction_provider(offset_multiple=2.0, atr_like=atr_like)
elif prediction_mode == "model":
    provider = _make_model_prediction_provider(data, frequency=frequency)
elif prediction_mode == "csv":
    predictions_csv = get_predictions_csv_path(symbol, frequency)
    print("Using predictions CSV:", predictions_csv)
    preds_df = _load_predictions_csv(predictions_csv)
    provider = _make_csv_prediction_provider(preds_df, data)
else:
    raise ValueError(f"Unknown prediction_mode: {prediction_mode}")
"""

    run_one_src = """def run_one(
    strat_cfg: StrategyConfig,
    commission_per_unit_per_leg: float = 0.005,
    min_commission_per_order: float = 1.0,
):
    '''Run a single backtest with the given strategy and commission settings.'''
    bt_cfg = BacktestConfig(
        initial_equity=initial_equity,
        strategy_config=strat_cfg,
        model_error_sigma=atr_like,
        fixed_atr=atr_like,
        commission_per_unit_per_leg=commission_per_unit_per_leg,
        min_commission_per_order=min_commission_per_order,
    )

    result = run_backtest(
        data=data,
        prediction_provider=provider,
        cfg=bt_cfg,
        atr_series=atr_series,
        model_error_sigma_series=atr_series,
    )

    metrics = _compute_backtest_metrics(
        result,
        initial_equity=initial_equity,
        data=data,
    )

    row = {
        "final_equity": result.final_equity,
        "n_trades": len(result.trades),
        **metrics,
    }
    return result, row
"""

    grid_a_src = """# Grid A: risk_per_trade_pct × reward_risk_ratio
risk_grid = [0.0025, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]  # 0.0025, 0.005, 0.01, 0.03
rr_grid = [1.0, 1.5, 2.0, 3.0, 3.5]  # [1.5, 2.0, 3.0]

rows = []
for risk_pct, rr in itertools.product(risk_grid, rr_grid):
    strat = StrategyConfig(
        risk_per_trade_pct=risk_pct,
        reward_risk_ratio=rr,
        k_sigma_err=0.5,
        k_atr_min_tp=3,
    )
    _, res_row = run_one(strat)
    res_row.update(
        {
            "grid": "risk_rr",
            "risk_per_trade_pct": risk_pct,
            "reward_risk_ratio": rr,
            "k_sigma_err": strat.k_sigma_err,
            "k_atr_min_tp": strat.k_atr_min_tp,
        }
    )
    rows.append(res_row)

df_risk_rr = pd.DataFrame(rows).sort_values("sharpe_ratio", ascending=False)
df_risk_rr.head(10)
"""

    grid_b_src = """# Grid B: model trust (k_sigma_err) × noise filter (k_atr_min_tp)
k_sigma_grid = [0.25, 0.5, 0.75, 1.0]
k_atr_grid = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

rows = []
for k_sigma, k_atr_min_tp in itertools.product(k_sigma_grid, k_atr_grid):
    strat = StrategyConfig(
        risk_per_trade_pct=0.01,
        reward_risk_ratio=2.0,
        k_sigma_err=k_sigma,
        k_atr_min_tp=k_atr_min_tp,
    )
    _, res_row = run_one(strat)
    res_row.update(
        {
            "grid": "noise_filters",
            "risk_per_trade_pct": strat.risk_per_trade_pct,
            "reward_risk_ratio": strat.reward_risk_ratio,
            "k_sigma_err": k_sigma,
            "k_atr_min_tp": k_atr_min_tp,
        }
    )
    rows.append(res_row)

df_noise = pd.DataFrame(rows).sort_values("sharpe_ratio", ascending=False)
df_noise.head(10)
"""

    for cell in nb.cells:
        src_obj = cell.get("source", "")
        if isinstance(src_obj, list):
            src = "".join(src_obj)
        else:
            src = src_obj

        meta_id = cell.get("metadata", {}).get("id")

        # Override specific cells by metadata id to ensure clean, multi-line sources.
        if meta_id == "intro-local" and cell.cell_type == "markdown":
            cell["source"] = intro_markdown_src
        elif meta_id == "set-project-root" and cell.cell_type == "code":
            cell["source"] = new_root_src
        elif meta_id == "imports" and cell.cell_type == "code":
            cell["source"] = imports_src
        elif meta_id == "load-data" and cell.cell_type == "code":
            cell["source"] = new_config_src
        elif meta_id == "helper-run-one" and cell.cell_type == "code":
            cell["source"] = run_one_src
        elif meta_id == "grid-risk-rr" and cell.cell_type == "code":
            cell["source"] = grid_a_src
        elif meta_id == "grid-noise-filters" and cell.cell_type == "code":
            cell["source"] = grid_b_src
        else:
            # Fallback: apply targeted fixes on existing source if needed.
            if cell.cell_type == "code":
                if "from src.config import FREQUENCY, get_hourly_data_csv_path" in src and "get_predictions_csv_path" not in src:
                    src = src.replace(
                        "from src.config import FREQUENCY, get_hourly_data_csv_path",
                        "from src.config import FREQUENCY, get_hourly_data_csv_path, get_predictions_csv_path",
                    )

                if "res_row.updatea" in src:
                    src = src.replace("res_row.updatea", "res_row.update")

                cell["source"] = src

        # Clear outputs/execution counts for code cells
        if cell.cell_type == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

        # Clear outputs/execution counts for code cells
        if cell.cell_type == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    # Normalize sources: convert to list of lines without embedded "\n" characters.
    # However, for markdown cells, we need to keep empty lines as separate elements.
    for cell in nb.cells:
        src_obj = cell.get("source", "")
        if isinstance(src_obj, list):
            joined = "".join(src_obj)
        else:
            joined = src_obj
        if joined is None:
            joined = ""
        
        # Split into lines, preserving structure.
        # For markdown, each paragraph / line break is a separate element in the list.
        # For code, each line is a separate element.
        lines = joined.splitlines(keepends=False)
        cell["source"] = lines

    nbformat.write(nb, NB_PATH)


if __name__ == "__main__":
    main()
