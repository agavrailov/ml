"""Streamlit UI entrypoint.

This is the main entry point for the Streamlit UI. It sets up the application
and delegates to page modules for rendering each tab.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path for imports.
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.backtest import run_backtest_for_ui
from src.data_processing import (
    clean_raw_minute_data,
    convert_minute_to_timeframe,
    prepare_keras_input_data,
)
from src.train import train_model
from src.config import (
    FREQUENCY,
    TSTEPS,
    RESAMPLE_FREQUENCIES,
    TSTEPS_OPTIONS,
    LSTM_UNITS_OPTIONS,
    BATCH_SIZE_OPTIONS,
    N_LSTM_LAYERS_OPTIONS,
    STATEFUL_OPTIONS,
    FEATURES_TO_USE_OPTIONS,
    MODEL_REGISTRY_DIR,
    get_run_hyperparameters,
    get_predictions_csv_path,
)
from scripts.generate_predictions_csv import generate_predictions_for_csv

# Import UI helpers
from src.ui.state import (
    get_ui_state,
    load_history,
    save_history,
    MAX_HISTORY_ROWS,
)
from src.ui.formatting import (
    format_timestamp,
    filter_training_history,
    filter_backtest_history,
    filter_optimization_history,
    get_best_training_row,
)
from src.ui.registry import (
    list_registry_models,
    promote_training_row,
)
from src.core.config_resolver import get_strategy_defaults, save_strategy_defaults

# Configure Streamlit page
st.set_page_config(layout="wide")

# Initialize global frequency in session state
if "global_frequency" not in st.session_state:
    if "60min" in RESAMPLE_FREQUENCIES:
        st.session_state["global_frequency"] = "60min"
    else:
        st.session_state["global_frequency"] = FREQUENCY

# Path constants
CONFIG_PATH = Path(__file__).parent.parent / "config.py"
PARAMS_STATE_PATH = Path(__file__).parent.parent / "ui_strategy_params.json"


# Helper functions for strategy parameter grid management
def _build_default_params_df(defaults: dict) -> pd.DataFrame:
    """Return the default parameter grid based on current config defaults."""
    return pd.DataFrame(
        [
            {
                "Parameter": "k_sigma_long",
                "Value": defaults["k_sigma_long"],
                "Start": defaults["k_sigma_long"],
                "Step": 0.1,
                "Stop": defaults["k_sigma_long"],
                "Optimize": True,
            },
            {
                "Parameter": "k_sigma_short",
                "Value": defaults["k_sigma_short"],
                "Start": defaults["k_sigma_short"],
                "Step": 0.1,
                "Stop": defaults["k_sigma_short"],
                "Optimize": True,
            },
            {
                "Parameter": "k_atr_long",
                "Value": defaults["k_atr_long"],
                "Start": defaults["k_atr_long"],
                "Step": 0.1,
                "Stop": defaults["k_atr_long"],
                "Optimize": False,
            },
            {
                "Parameter": "k_atr_short",
                "Value": defaults["k_atr_short"],
                "Start": defaults["k_atr_short"],
                "Step": 0.1,
                "Stop": defaults["k_atr_short"],
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
    """Load the parameter grid from disk, falling back to defaults."""
    if PARAMS_STATE_PATH.exists():
        try:
            raw = PARAMS_STATE_PATH.read_text(encoding="utf-8")
            records = json.loads(raw)
            if isinstance(records, list) and records:
                df = pd.DataFrame(records)
                required_cols = {"Parameter", "Value", "Start", "Step", "Stop", "Optimize"}
                missing = required_cols - set(df.columns)
                if not missing:
                    return df
        except Exception:
            pass
    return _build_default_params_df(defaults)


def _save_params_grid(df: pd.DataFrame | object) -> None:
    """Persist the full parameter grid."""
    try:
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        records = df.to_dict(orient="records")
        PARAMS_STATE_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")
    except Exception as exc:
        st.error(f"Failed to save parameter grid: {exc}")


def _save_strategy_defaults_to_config(
    *,
    risk_per_trade_pct: float,
    reward_risk_ratio: float,
    k_sigma_long: float,
    k_sigma_short: float,
    k_atr_long: float,
    k_atr_short: float,
) -> None:
    """Persist strategy parameters to configs/active.json."""
    try:
        save_strategy_defaults(
            risk_per_trade_pct=risk_per_trade_pct,
            reward_risk_ratio=reward_risk_ratio,
            k_sigma_long=k_sigma_long,
            k_sigma_short=k_sigma_short,
            k_atr_long=k_atr_long,
            k_atr_short=k_atr_short,
        )
    except Exception as exc:
        st.error(f"Failed to save strategy defaults: {exc}")


def _load_strategy_defaults() -> dict:
    """Return effective strategy defaults from config_resolver."""
    defaults = get_strategy_defaults()
    # Add backwards-compatible aliases for legacy code.
    defaults["k_sigma_err"] = defaults["k_sigma_long"]
    defaults["k_atr_min_tp"] = defaults["k_atr_long"]
    return defaults


def _run_backtest(
    frequency: str,
    start_date: str | None,
    end_date: str | None,
    risk_per_trade_pct: float | None,
    reward_risk_ratio: float | None,
    k_sigma_long: float | None,
    k_sigma_short: float | None,
    k_atr_long: float | None,
    k_atr_short: float | None,
    enable_longs: bool | None,
    allow_shorts: bool | None,
):
    """Run backtest using CSV predictions mode."""
    predictions_csv = get_predictions_csv_path("nvda", frequency)
    if predictions_csv and not os.path.exists(predictions_csv):
        raise FileNotFoundError(
            f"Predictions CSV not found: '{predictions_csv}'. "
            "Use the 'Generate predictions CSV for NVDA' button first."
        )

    return run_backtest_for_ui(
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


def _start_optimization() -> None:
    st.session_state["optimization_running"] = True


def _stop_optimization() -> None:
    st.session_state["optimization_running"] = False


# Main UI
st.title("LSTM Backtesting & Training UI")

tab_live, tab_data, tab_experiments, tab_train, tab_backtest, tab_walkforward = st.tabs(
    [
        "Live",
        "Data",
        "Hyperparameter Experiments",
        "Train & Promote",
        "Backtest / Strategy",
        "Walk-Forward Analysis",
    ]
)

with tab_live:
    from src.ui.pages import live_page

    live_page.render_live_tab(st=st, pd=pd, plt=plt)

with tab_data:
    from src.ui.pages import data_page

    data_page.render_data_tab(
        st=st,
        pd=pd,
        os=os,
        Path=Path,
        project_root=repo_root,
        RESAMPLE_FREQUENCIES=RESAMPLE_FREQUENCIES,
        FREQUENCY=FREQUENCY,
        clean_raw_minute_data=clean_raw_minute_data,
        convert_minute_to_timeframe=convert_minute_to_timeframe,
        prepare_keras_input_data=prepare_keras_input_data,
    )

with tab_experiments:
    from src.ui.pages import experiments_page

    experiments_page.render_experiments_tab(
        st=st,
        pd=pd,
        os=os,
        train_model=train_model,
        FREQUENCY=FREQUENCY,
        TSTEPS=TSTEPS,
        RESAMPLE_FREQUENCIES=RESAMPLE_FREQUENCIES,
        TSTEPS_OPTIONS=TSTEPS_OPTIONS,
        LSTM_UNITS_OPTIONS=LSTM_UNITS_OPTIONS,
        N_LSTM_LAYERS_OPTIONS=N_LSTM_LAYERS_OPTIONS,
        BATCH_SIZE_OPTIONS=BATCH_SIZE_OPTIONS,
        STATEFUL_OPTIONS=STATEFUL_OPTIONS,
        FEATURES_TO_USE_OPTIONS=FEATURES_TO_USE_OPTIONS,
        MAX_HISTORY_ROWS=MAX_HISTORY_ROWS,
        get_run_hyperparameters=get_run_hyperparameters,
        get_ui_state=get_ui_state,
        load_json_history=load_history,
        save_json_history=save_history,
    )

with tab_train:
    from src.ui.pages import train_page

    train_page.render_train_tab(
        st=st,
        pd=pd,
        json=json,
        Path=Path,
        os=os,
        FREQUENCY=FREQUENCY,
        TSTEPS=TSTEPS,
        RESAMPLE_FREQUENCIES=RESAMPLE_FREQUENCIES,
        TSTEPS_OPTIONS=TSTEPS_OPTIONS,
        LSTM_UNITS_OPTIONS=LSTM_UNITS_OPTIONS,
        N_LSTM_LAYERS_OPTIONS=N_LSTM_LAYERS_OPTIONS,
        BATCH_SIZE_OPTIONS=BATCH_SIZE_OPTIONS,
        STATEFUL_OPTIONS=STATEFUL_OPTIONS,
        FEATURES_TO_USE_OPTIONS=FEATURES_TO_USE_OPTIONS,
        MODEL_REGISTRY_DIR=MODEL_REGISTRY_DIR,
        MAX_HISTORY_ROWS=MAX_HISTORY_ROWS,
        get_run_hyperparameters=get_run_hyperparameters,
        get_ui_state=get_ui_state,
        load_json_history=load_history,
        save_json_history=save_history,
        format_timestamp=format_timestamp,
        list_registry_models=list_registry_models,
        promote_training_row=promote_training_row,
    )

with tab_backtest:
    from src.ui.pages import backtest_page
    import src.config as cfg_mod

    backtest_page.render_backtest_tab(
        st=st,
        pd=pd,
        plt=plt,
        os=os,
        cfg_mod=cfg_mod,
        MAX_HISTORY_ROWS=MAX_HISTORY_ROWS,
        get_predictions_csv_path=get_predictions_csv_path,
        generate_predictions_for_csv=generate_predictions_for_csv,
        load_strategy_defaults=_load_strategy_defaults,
        load_params_grid=_load_params_grid,
        save_params_grid=_save_params_grid,
        save_strategy_defaults_to_config=_save_strategy_defaults_to_config,
        run_backtest=_run_backtest,
        filter_backtest_history=filter_backtest_history,
        filter_optimization_history=filter_optimization_history,
        format_timestamp=format_timestamp,
        get_ui_state=get_ui_state,
        load_json_history=load_history,
        save_json_history=save_history,
        start_optimization=_start_optimization,
        stop_optimization=_stop_optimization,
    )

with tab_walkforward:
    from src.ui.pages import walkforward_page

    walkforward_page.render_walkforward_tab(
        st=st,
        pd=pd,
        plt=plt,
        FREQUENCY=FREQUENCY,
        RESAMPLE_FREQUENCIES=RESAMPLE_FREQUENCIES,
        MAX_HISTORY_ROWS=MAX_HISTORY_ROWS,
        get_ui_state=get_ui_state,
        load_strategy_defaults=_load_strategy_defaults,
        save_json_history=save_history,
    )
