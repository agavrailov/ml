"""Streamlit UI entrypoint.

This is the main entry point for the Streamlit UI. It sets up the application
and delegates to page modules for rendering each tab.

Navigation approach:
- Uses horizontal tabs (st.tabs) for single-page navigation
- Page modules are in src/ui/page_modules/ (NOT src/ui/pages/) to prevent
  Streamlit's auto-discovery which would create unwanted sidebar navigation
- All state persists naturally across tabs without page reloads
- Trade-off: No URL routing (tabs not bookmarkable), but faster UX
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

# Heavy imports (TensorFlow/Keras) are lazy-loaded inside tabs to avoid 5s startup delay
# from src.backtest import run_backtest_for_ui
# from src.data_processing import (...)
# from src.train import train_model

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

# Import UI components for modern styling
from src.ui import components

# Configure Streamlit page
# Note: sidebar is hidden since we use tabs instead of multipage
st.set_page_config(
    page_title="LSTM Trading System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Apply custom CSS for professional styling
components.inject_custom_css(st)

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
    """Load the parameter grid from disk, merging saved values with defaults.
    
    This ensures that:
    - If a saved file exists, Value column comes from saved data
    - Missing columns are filled from defaults
    - Missing parameters are added from defaults
    """
    default_df = _build_default_params_df(defaults)
    
    if not PARAMS_STATE_PATH.exists():
        return default_df
    
    try:
        raw = PARAMS_STATE_PATH.read_text(encoding="utf-8")
        records = json.loads(raw)
        if not isinstance(records, list) or not records:
            return default_df
            
        saved_df = pd.DataFrame(records)
        
        # Merge: start with defaults, then update with saved values
        result_df = default_df.copy()
        
        for _, saved_row in saved_df.iterrows():
            param_name = saved_row.get("Parameter")
            if param_name and param_name in result_df["Parameter"].values:
                # Update the matching row with saved values (preserving defaults for missing cols)
                idx = result_df[result_df["Parameter"] == param_name].index[0]
                for col in saved_row.index:
                    if col in result_df.columns and pd.notna(saved_row[col]):
                        result_df.loc[idx, col] = saved_row[col]
        
        return result_df
    except Exception:
        return default_df


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
    enable_longs: bool | None = None,
    allow_shorts: bool | None = None,
    symbol: str | None = None,
    frequency: str | None = None,
    source: str | None = "ui_manual_deploy",
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
            enable_longs=enable_longs,
            allow_shorts=allow_shorts,
            symbol=symbol,
            frequency=frequency,
            source=source,
        )
    except Exception as exc:
        st.error(f"Failed to save strategy defaults: {exc}")


def _load_strategy_defaults(symbol: str | None = None) -> dict:
    """Return effective strategy defaults from config_resolver.

    When ``symbol`` is provided, resolves per-symbol overrides first
    (configs/symbols/{SYMBOL}/active.json) before falling back to the global
    config and code defaults.
    """
    defaults = get_strategy_defaults(symbol)
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
    symbol: str | None = None,
):
    """Run backtest using CSV predictions mode."""
    from src.backtest import run_backtest_for_ui

    sym = (symbol or st.session_state.get("global_symbol", "NVDA")).upper()
    predictions_csv = get_predictions_csv_path(sym.lower(), frequency)
    if predictions_csv and not os.path.exists(predictions_csv):
        raise FileNotFoundError(
            f"Predictions CSV not found: '{predictions_csv}'. "
            f"Use the 'Generate predictions CSV' button for {sym} first."
        )

    return run_backtest_for_ui(
        symbol=sym,
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
_title_col, _sym_col = st.columns([5, 1])
with _title_col:
    st.title("LSTM Backtesting & Training UI")
with _sym_col:
    from src.core.config_resolver import get_configured_symbols
    _configured_symbols = get_configured_symbols()
    _global_symbol = st.selectbox(
        "Symbol",
        _configured_symbols,
        index=0,
        key="global_symbol",
    )

# ── Persistent context strip (P1) ───────────────────────────────────
# A single line showing the active symbol/freq + readiness signals so the
# user always knows what they're operating on, regardless of which tab.
def _render_context_strip() -> None:
    from src.core.config_resolver import get_strategy_defaults
    from src.config import get_hourly_data_csv_path
    from src.model_registry import get_best_model_path

    sym = st.session_state.get("global_symbol", "NVDA")
    freq = st.session_state.get("global_frequency", FREQUENCY)
    tsteps = TSTEPS

    # Readiness checks
    ohlc_path = get_hourly_data_csv_path(freq, symbol=sym)
    has_data = os.path.exists(ohlc_path)

    has_model = bool(get_best_model_path(sym, freq, tsteps))

    pred_path = get_predictions_csv_path(sym.lower(), freq)
    has_preds = os.path.exists(pred_path)

    try:
        defaults = get_strategy_defaults(sym)
        has_params = bool(defaults) and "k_sigma_long" in defaults
    except Exception:
        has_params = False

    # Live daemon status — peek at last status.json for this symbol
    live_dir = Path(__file__).resolve().parents[2] / "ui_state" / "live"
    is_live_running = False
    if live_dir.exists():
        for status_file in live_dir.glob("**/status.json"):
            try:
                status = json.loads(status_file.read_text(encoding="utf-8"))
                if str(status.get("symbol", "")).upper() == sym.upper():
                    state = str(status.get("state", "")).upper()
                    if state in ("RUNNING", "TRADING", "SLEEPING", "PROCESSING"):
                        is_live_running = True
                        break
            except Exception:
                continue

    # Theme-aware: green for OK, red for missing — colors are visible on both
    # light and dark backgrounds.  Container uses Streamlit's CSS vars so it
    # adapts to the active theme automatically.
    def _badge(ok: bool, label: str) -> str:
        if ok:
            return f"<span style='color:#22c55e;font-weight:700'>✓</span> {label}"
        return f"<span style='color:#ef4444;font-weight:700'>✗</span> {label}"

    live_badge = (
        "<span style='color:#22c55e;font-weight:700'>● live</span>"
        if is_live_running
        else "<span style='color:#9ca3af;font-weight:700'>○ offline</span>"
    )

    parts = [
        f"<strong>{sym}</strong>",
        f"@ <strong>{freq}</strong>",
        _badge(has_data, "data"),
        _badge(has_model, "model"),
        _badge(has_preds, "predictions"),
        _badge(has_params, "params"),
        live_badge,
    ]
    st.markdown(
        "<div style='padding:8px 14px;"
        "background:var(--secondary-background-color, rgba(128,128,128,0.12));"
        "color:var(--text-color, inherit);"
        "border-radius:6px;font-size:0.92em;margin-bottom:10px'>"
        + "  ·  ".join(parts)
        + "</div>",
        unsafe_allow_html=True,
    )

_render_context_strip()

# ── Tab order (P2): daily-use first ─────────────────────────────────
tab_live, tab_backtest, tab_portfolio, tab_walkforward, tab_train, tab_data, tab_experiments = st.tabs(
    [
        "Live",
        "Backtest / Strategy",
        "Portfolio",
        "Walk-Forward Analysis",
        "Train & Promote",
        "Data",
        "Hyperparameter Experiments",
    ]
)

with tab_live:
    from src.ui.page_modules import live_page

    live_page.render_live_tab(st=st, pd=pd, plt=plt)

with tab_data:
    from src.ui.page_modules import data_page
    from src.data_processing import (
        clean_raw_minute_data,
        convert_minute_to_timeframe,
        prepare_keras_input_data,
    )

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
    from src.ui.page_modules import experiments_page

    experiments_page.render_experiments_tab(
        st=st,
        pd=pd,
        os=os,
        train_model=None,  # Lazy-loaded inside the page module
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
    from src.ui.page_modules import train_page

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
    from src.ui.page_modules import backtest_page
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
    from src.ui.page_modules import walkforward_page

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

with tab_portfolio:
    from src.ui.page_modules import portfolio_page

    portfolio_page.render_portfolio_tab(st=st, pd=pd, plt=plt)
