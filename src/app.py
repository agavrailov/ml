import sys
import os
import re
import json
from pathlib import Path
import importlib

# Ensure repository root is on sys.path so `src` can be imported when running
# `streamlit run src/app.py` from the project root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.backtest import run_backtest_for_ui
import src.config as cfg_mod


CONFIG_PATH = Path(__file__).with_name("config.py")
PARAMS_STATE_PATH = Path(__file__).with_name("ui_strategy_params.json")


def _load_strategy_defaults() -> dict:
    """Reload src.config and return current strategy defaults.

    This ensures that when config.py is edited on disk (e.g. via the Save
    button below), subsequent reruns of the app see the updated values
    without needing to restart the Streamlit server.
    """

    global cfg_mod
    cfg_mod = importlib.reload(cfg_mod)
    return {
        "k_sigma_err": float(cfg_mod.K_SIGMA_ERR),
        "k_atr_min_tp": float(cfg_mod.K_ATR_MIN_TP),
        "risk_per_trade_pct": float(cfg_mod.RISK_PER_TRADE_PCT),
        "reward_risk_ratio": float(cfg_mod.REWARD_RISK_RATIO),
    }


def _build_default_params_df(defaults: dict) -> pd.DataFrame:
    """Return the default parameter grid based on current config defaults."""

    return pd.DataFrame(
        [
            {
                "Parameter": "k_sigma_err",
                "Value": defaults["k_sigma_err"],
                "Start": defaults["k_sigma_err"],
                "Step": 0.1,
                "Stop": defaults["k_sigma_err"],
                "Optimize": True,
            },
            {
                "Parameter": "k_atr_min_tp",
                "Value": defaults["k_atr_min_tp"],
                "Start": defaults["k_atr_min_tp"],
                "Step": 0.1,
                "Stop": defaults["k_atr_min_tp"],
                "Optimize": False,
            },
            {
                "Parameter": "risk_per_trade_pct",
                "Value": defaults["risk_per_trade_pct"],
                "Start": defaults["risk_per_trade_pct"],
                "Step": 0.001,
                "Stop": defaults["risk_per_trade_pct"],
                "Optimize": False,
            },
            {
                "Parameter": "reward_risk_ratio",
                "Value": defaults["reward_risk_ratio"],
                "Start": defaults["reward_risk_ratio"],
                "Step": 0.1,
                "Stop": defaults["reward_risk_ratio"],
                "Optimize": False,
            },
        ]
    )


def _load_params_grid(defaults: dict) -> pd.DataFrame:
    """Load the parameter grid from disk, falling back to defaults.

    This persists Value/Start/Step/Stop/Optimize across UI reloads, while
    still using config.py defaults when the state file does not exist.
    """

    if PARAMS_STATE_PATH.exists():
        try:
            raw = PARAMS_STATE_PATH.read_text(encoding="utf-8")
            records = json.loads(raw)
            if isinstance(records, list) and records:
                df = pd.DataFrame(records)
                # Ensure all required columns exist; fill missing with defaults.
                required_cols = {"Parameter", "Value", "Start", "Step", "Stop", "Optimize"}
                missing = required_cols - set(df.columns)
                if missing:
                    # Merge with a fresh default df as a template.
                    base = _build_default_params_df(defaults)
                    df = pd.merge(
                        base,
                        df,
                        on="Parameter",
                        how="left",
                        suffixes=("_base", ""),
                    )
                    # Prefer loaded values when present; otherwise use base.
                    rows = []
                    for _, row in df.iterrows():
                        name = row["Parameter"]
                        base_row = base[base["Parameter"] == name].iloc[0]
                        rows.append(
                            {
                                "Parameter": name,
                                "Value": row.get("Value", base_row["Value"]),
                                "Start": row.get("Start", base_row["Start"]),
                                "Step": row.get("Step", base_row["Step"]),
                                "Stop": row.get("Stop", base_row["Stop"]),
                                "Optimize": bool(row.get("Optimize", base_row["Optimize"])),
                            }
                        )
                    df = pd.DataFrame(rows)
                return df
        except Exception:
            # Fall back to defaults on any parse error.
            pass

    return _build_default_params_df(defaults)


def _save_params_grid(df: pd.DataFrame) -> None:
    """Persist the full parameter grid (Value/Start/Step/Stop/Optimize)."""

    try:
        records = df.to_dict(orient="records")
        PARAMS_STATE_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - best-effort persistence
        st.error(f"Failed to save parameter grid: {exc}")


def _save_strategy_defaults_to_config(
    *,
    risk_per_trade_pct: float,
    reward_risk_ratio: float,
    k_sigma_err: float,
    k_atr_min_tp: float,
) -> None:
    """Persist current strategy parameters into src/config.py.

    This updates the default values inside ``StrategyDefaultsConfig`` so that
    future CLI runs and fresh UI sessions see the new defaults.
    """

    try:
        text = CONFIG_PATH.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - best-effort persistence
        st.error(f"Failed to read config.py: {exc}")
        return

    patterns = {
        "risk_per_trade_pct": r"(risk_per_trade_pct: float = )([0-9.eE+\-]+)",
        "reward_risk_ratio": r"(reward_risk_ratio: float = )([0-9.eE+\-]+)",
        "k_sigma_err": r"(k_sigma_err: float = )([0-9.eE+\-]+)",
        "k_atr_min_tp": r"(k_atr_min_tp: float = )([0-9.eE+\-]+)",
    }
    values = {
        "risk_per_trade_pct": risk_per_trade_pct,
        "reward_risk_ratio": reward_risk_ratio,
        "k_sigma_err": k_sigma_err,
        "k_atr_min_tp": k_atr_min_tp,
    }

    new_text = text
    for name, pattern in patterns.items():
        value = values[name]

        def _repl(match, value=value):
            # Keep the left-hand side (type annotation and "=") and replace only the literal.
            return f"{match.group(1)}{value}"

        new_text, n = re.subn(pattern, _repl, new_text, count=1)
        if n == 0:
            st.warning(f"Pattern not found in config.py for {name}: {pattern}")

    if new_text != text:
        try:
            CONFIG_PATH.write_text(new_text, encoding="utf-8")
        except OSError as exc:  # pragma: no cover - best-effort persistence
            st.error(f"Failed to write config.py: {exc}")


def _run_backtest(
    frequency: str,
    start_date: str | None,
    end_date: str | None,
    risk_per_trade_pct: float | None,
    reward_risk_ratio: float | None,
    k_sigma_err: float | None,
    k_atr_min_tp: float | None,
):
    """Non-cached wrapper around run_backtest_for_ui (model mode only)."""
    return run_backtest_for_ui(
        frequency=frequency,
        prediction_mode="model",
        start_date=start_date,
        end_date=end_date,
        predictions_csv=None,
        risk_per_trade_pct=risk_per_trade_pct,
        reward_risk_ratio=reward_risk_ratio,
        k_sigma_err=k_sigma_err,
        k_atr_min_tp=k_atr_min_tp,
    )


st.title("Backtest UI (MVP)")

# For now we focus on a single trained model frequency (15min) and model-based
# predictions only. Other modes (naive/csv) are removed to keep the UI simple.
freq = st.selectbox("Frequency", ["15min"], index=0)

# Reload current defaults from config.py so changes on disk are visible
# without restarting the Streamlit server.
_defaults = _load_strategy_defaults()

# Parameter grid (similar to MT5 Strategy Tester): each row is a parameter
# with Value / Start / Step / Stop / Optimize. The grid is persisted to a
# JSON sidecar so manual edits survive app reloads.
st.subheader("Strategy parameters")
params_df_initial = _load_params_grid(_defaults)
params_df = st.data_editor(
    params_df_initial,
    num_rows="fixed",
    key="strategy_params",
    use_container_width=True,
    hide_index=True,
)

# Extract current "Value" settings from the parameter grid.
_param_values = {row["Parameter"]: row["Value"] for _, row in params_df.iterrows()}
k_sigma_val = float(_param_values.get("k_sigma_err", _defaults["k_sigma_err"]))
k_atr_val = float(_param_values.get("k_atr_min_tp", _defaults["k_atr_min_tp"]))
risk_pct_val = float(_param_values.get("risk_per_trade_pct", _defaults["risk_per_trade_pct"]))
rr_val = float(_param_values.get("reward_risk_ratio", _defaults["reward_risk_ratio"]))

if st.button("Save parameters to config.py"):
    # Persist single-run defaults to config.py (used by CLIs and as base
    # defaults) and the full grid (Value/Start/Step/Stop/Optimize) to a
    # JSON sidecar used only by this UI.
    _save_strategy_defaults_to_config(
        risk_per_trade_pct=risk_pct_val,
        reward_risk_ratio=rr_val,
        k_sigma_err=k_sigma_val,
        k_atr_min_tp=k_atr_val,
    )
    _save_params_grid(params_df)
    st.success("Saved strategy defaults and parameter grid (Value/Start/Step/Stop).")

col1, col2 = st.columns(2)
with col1:
    start_date = st.text_input("Start date (YYYY-MM-DD, optional)", "")
with col2:
    end_date = st.text_input("End date (YYYY-MM-DD, optional)", "")

# Prediction source is always the trained LSTM model; no CSV/naive modes.
mode = st.radio("Mode", ["Run backtest", "Optimize"], horizontal=True)

if mode == "Run backtest" and st.button("Run backtest"):
    with st.spinner("Running backtest..."):
        equity_df, trades_df, metrics = _run_backtest(
            frequency=freq,
            start_date=start_date or None,
            end_date=end_date or None,
            risk_per_trade_pct=risk_pct_val,
            reward_risk_ratio=rr_val,
            k_sigma_err=k_sigma_val,
            k_atr_min_tp=k_atr_val,
        )

    time_col = "Time" if "Time" in equity_df.columns else "step"

    st.subheader("Equity curve")
    if not equity_df.empty:
        # Plot a downsampled view of the ENTIRE equity curve to keep the UI responsive.
        max_points = 2000
        if len(equity_df) > max_points:
            step = max(1, len(equity_df) // max_points)
            equity_to_plot = equity_df.iloc[::step]
        else:
            equity_to_plot = equity_df
        st.line_chart(equity_to_plot.set_index(time_col)["equity"])
    else:
        st.write("No equity data to display.")

    st.subheader("Metrics")
    st.json(metrics)

elif mode == "Optimize" and st.button("Run optimization"):
    st.write("Running grid search over parameter ranges...")

    # Build value ranges for each parameter based on Start/Stop/Step.
    param_ranges: dict[str, list[float]] = {}
    for _, row in params_df.iterrows():
        name = row["Parameter"]
        optimize = bool(row.get("Optimize", True))

        # If not optimizing this parameter, keep it fixed at the current Value.
        if not optimize:
            param_ranges[name] = [float(row["Value"])]
            continue

        start = float(row["Start"])
        stop = float(row["Stop"])
        step = float(row["Step"])

        values: list[float] = []
        if step > 0 and stop >= start:
            current = start
            # Cap iterations defensively to avoid infinite loops.
            for _ in range(1000):
                if current > stop + 1e-9:
                    break
                values.append(float(current))
                current += step
        # Fallback: if range is invalid, just use the current Value.
        if not values:
            values = [float(row["Value"])]

        param_ranges[name] = values

    # Compute total combinations.
    from itertools import product
    total_runs = 1
    for vals in param_ranges.values():
        total_runs *= max(len(vals), 1)

    max_runs = 200
    if total_runs > max_runs:
        st.error(f"Grid is too large ({total_runs} runs). Reduce ranges or steps (limit = {max_runs}).")
    else:
        progress = st.progress(0.0)
        results_rows: list[dict] = []
        names = list(param_ranges.keys())
        all_values = [param_ranges[n] for n in names]

        run_idx = 0
        for combo in product(*all_values):
            combo_params = dict(zip(names, combo))

            k_sigma = float(combo_params.get("k_sigma_err", k_sigma_val))
            k_atr = float(combo_params.get("k_atr_min_tp", k_atr_val))
            risk_pct = float(combo_params.get("risk_per_trade_pct", risk_pct_val))
            rr = float(combo_params.get("reward_risk_ratio", rr_val))

            equity_df, trades_df, metrics = _run_backtest(
                frequency=freq,
                start_date=start_date or None,
                end_date=end_date or None,
                risk_per_trade_pct=risk_pct,
                reward_risk_ratio=rr,
                k_sigma_err=k_sigma,
                k_atr_min_tp=k_atr,
            )

            results_rows.append(
                {
                    "k_sigma_err": k_sigma,
                    "k_atr_min_tp": k_atr,
                    "risk_per_trade_pct": risk_pct,
                    "reward_risk_ratio": rr,
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

            run_idx += 1
            progress.progress(run_idx / total_runs)

        if results_rows:
            results_df = pd.DataFrame(results_rows)
            # Sort by total_return descending by default.
            results_df = results_df.sort_values(by="total_return", ascending=False)

            st.subheader("Optimization results")
            st.dataframe(
                results_df.reset_index(drop=True),
                use_container_width=True,
            )

            # ------------------------------------------------------------------
            # Heatmaps: k_sigma_err (x) vs k_atr_min_tp (y) for key metrics.
            # We use the most common risk_per_trade_pct and reward_risk_ratio to
            # select a 2D slice through the grid.
            # ------------------------------------------------------------------
            st.subheader("Heatmaps (k_sigma_err vs k_atr_min_tp)")
            if not results_df.empty:
                mode_risk = results_df["risk_per_trade_pct"].mode().iloc[0]
                mode_rr = results_df["reward_risk_ratio"].mode().iloc[0]

                slice_df = results_df[
                    (results_df["risk_per_trade_pct"] == mode_risk)
                    & (results_df["reward_risk_ratio"] == mode_rr)
                ]

                if not slice_df.empty:
                    sharpe_grid = slice_df.pivot(
                        index="k_atr_min_tp",
                        columns="k_sigma_err",
                        values="sharpe_ratio",
                    ).sort_index().sort_index(axis=1)

                    ret_grid = slice_df.pivot(
                        index="k_atr_min_tp",
                        columns="k_sigma_err",
                        values="total_return",
                    ).sort_index().sort_index(axis=1)

                    mdd_grid = slice_df.pivot(
                        index="k_atr_min_tp",
                        columns="k_sigma_err",
                        values="max_drawdown",
                    ).sort_index().sort_index(axis=1)

                    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

                    grids = [
                        (sharpe_grid, axes[0], "Sharpe Ratio", "viridis"),
                        (ret_grid, axes[1], "Total Return", "YlOrRd"),
                        (mdd_grid, axes[2], "Max Drawdown", "RdYlGn_r"),
                    ]

                    for grid, ax, title, cmap in grids:
                        im = ax.imshow(grid.values, aspect="auto", cmap=cmap)
                        ax.set_title(title)
                        ax.set_xlabel("k_sigma_err")
                        # Only show y-label on the first plot to reduce clutter.
                        ax.set_ylabel("k_atr_min_tp" if ax is axes[0] else "")

                        # Tick labels from the DataFrame indices/columns.
                        ax.set_xticks(range(len(grid.columns)))
                        ax.set_xticklabels([f"{v:g}" for v in grid.columns], rotation=45)
                        ax.set_yticks(range(len(grid.index)))
                        ax.set_yticklabels([f"{v:g}" for v in grid.index])

                        # Annotate cells with numeric values.
                        for i, y_val in enumerate(grid.index):
                            for j, x_val in enumerate(grid.columns):
                                val = grid.loc[y_val, x_val]
                                ax.text(
                                    j,
                                    i,
                                    f"{val:.2f}",
                                    ha="center",
                                    va="center",
                                    fontsize=8,
                                    color="black",
                                )

                        # Add a small colorbar per subplot.
                        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info(
                        "No 2D slice available for heatmaps with the current parameter ranges.",
                    )
