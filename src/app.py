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
# NOTE: TWS/ib_insync ingestion is triggered via a subprocess CLI call from the
# Data tab to avoid event-loop issues in the Streamlit runtime.
from src.data_processing import (
    clean_raw_minute_data,
    convert_minute_to_timeframe,
    prepare_keras_input_data,
)
from src.data import load_hourly_ohlc

# Use wide layout so plots can take full screen width.
st.set_page_config(layout="wide")
import src.config as cfg_mod
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

# Shared frequency across tabs, defaulting to 60min when available.
if "global_frequency" not in st.session_state:
    if "60min" in RESAMPLE_FREQUENCIES:
        st.session_state["global_frequency"] = "60min"
    else:
        st.session_state["global_frequency"] = FREQUENCY

CONFIG_PATH = Path(__file__).with_name("config.py")
PARAMS_STATE_PATH = Path(__file__).with_name("ui_strategy_params.json")

# UI state directory and persistence paths.
UI_STATE_DIR = Path(__file__).resolve().parent.parent / "ui_state"
MAX_HISTORY_ROWS = 100  # Maximum rows to keep in any history table.


def _get_ui_state() -> dict:
    """Return the centralized UI state dict, initializing if needed.

    This provides a single source of truth for long-lived state that may be
    persisted to JSON. The dict has well-known top-level keys for each tab.
    """
    if "ui_state" not in st.session_state:
        st.session_state["ui_state"] = {
            "data": {},
            "experiments": {},
            "training": {},
            "strategy": {},
            "backtests": {},
            "optimization": {},
            "walkforward": {},
        }
    return st.session_state["ui_state"]


def _load_json_history(filename: str) -> list[dict]:
    """Load a history list from JSON in UI_STATE_DIR, returning [] if missing."""
    path = UI_STATE_DIR / filename
    if not path.exists():
        return []
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def _save_json_history(filename: str, history: list[dict]) -> None:
    """Save a history list to JSON in UI_STATE_DIR, creating the dir if needed."""
    UI_STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = UI_STATE_DIR / filename
    try:
        # Truncate to MAX_HISTORY_ROWS if needed (keep most recent).
        if len(history) > MAX_HISTORY_ROWS:
            history = history[-MAX_HISTORY_ROWS:]
        path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - best-effort persistence
        st.error(f"Failed to save history to {filename}: {exc}")


def _filter_training_history(history: list[dict], *, frequency: str, tsteps: int) -> list[dict]:
    """Return training history rows matching a specific (frequency, tsteps)."""
    rows = [
        r
        for r in (history or [])
        if isinstance(r, dict)
        and r.get("frequency") == frequency
        and int(r.get("tsteps", -1)) == int(tsteps)
    ]

    # Most-recent first.
    return sorted(rows, key=lambda r: r.get("timestamp", ""), reverse=True)


def _get_best_training_row(rows: list[dict]) -> dict | None:
    """Return the row with minimal validation_loss, or None."""
    best: dict | None = None
    best_loss = float("inf")

    for r in rows or []:
        try:
            loss = float(r.get("validation_loss"))
        except Exception:
            continue

        if loss < best_loss:
            best_loss = loss
            best = r

    return best


_REGISTRY_MODEL_RE = re.compile(
    r"^my_lstm_model_(?P<frequency>.+?)_tsteps(?P<tsteps>\d+)_(?P<date>\d{8})_(?P<time>\d{6})\.keras$"
)


def _format_timestamp_iso_seconds(ts: str | None) -> str | None:
    """Return an ISO timestamp truncated to seconds (YYYY-MM-DDTHH:MM:SS)."""
    if not ts:
        return None
    s = str(ts)

    # Fast-path: keep only the first 19 chars, which correspond to seconds.
    # Examples:
    # - 2025-12-12T10:58:23.123456+00:00 -> 2025-12-12T10:58:23
    # - 2025-12-12T10:58:23 -> 2025-12-12T10:58:23
    if len(s) >= 19 and s[4] == "-" and s[10] == "T":
        return s[:19]

    try:
        t = pd.to_datetime(s, utc=True, errors="coerce")
        if pd.isna(t):
            return None
        return t.strftime("%Y-%m-%dT%H:%M:%S")
    except Exception:
        return None


def _parse_registry_model_filename(filename: str) -> dict | None:
    """Parse registry model filename into structured fields.

    Expected format:
        my_lstm_model_{frequency}_tsteps{tsteps}_{YYYYMMDD}_{HHMMSS}.keras
    """
    m = _REGISTRY_MODEL_RE.match(str(filename))
    if not m:
        return None

    freq = m.group("frequency")
    tsteps = int(m.group("tsteps"))
    d = m.group("date")
    tm = m.group("time")

    # Format into second-resolution ISO timestamp.
    ts_iso = f"{d[0:4]}-{d[4:6]}-{d[6:8]}T{tm[0:2]}:{tm[2:4]}:{tm[4:6]}"

    return {
        "model_filename": str(filename),
        "frequency": freq,
        "tsteps": tsteps,
        "timestamp": ts_iso,
        "stamp": f"{d}_{tm}",
    }


def _list_registry_models(registry_dir: Path) -> list[dict]:
    """List all model artifacts in the model registry.

    Returns rows suitable for rendering in the UI.
    """
    if not registry_dir.exists() or not registry_dir.is_dir():
        return []

    rows: list[dict] = []
    for p in registry_dir.iterdir():
        if not p.is_file():
            continue
        info = _parse_registry_model_filename(p.name)
        if info is None:
            continue

        # Best-effort bias-correction inference (same stamp).
        bias_name = f"bias_correction_{info['frequency']}_tsteps{int(info['tsteps'])}_{info['stamp']}.json"
        bias_path = registry_dir / bias_name
        info["bias_correction_filename"] = bias_name if bias_path.exists() else None

        # Per-model metrics (validation loss, etc.)
        metrics_name = str(info["model_filename"]).replace(".keras", ".metrics.json")
        metrics_path = registry_dir / metrics_name
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8") or "{}")
                if isinstance(metrics, dict) and metrics.get("validation_loss") is not None:
                    info["validation_loss"] = float(metrics.get("validation_loss"))
            except Exception:
                pass

        rows.append(info)

    # Sort most-recent first (timestamp is ISO so lexicographic sort works).
    return sorted(rows, key=lambda r: r.get("timestamp", ""), reverse=True)


def _promote_training_row(
    *,
    row: dict,
    best_hps_path: Path,
    frequency: str,
    tsteps: int,
) -> None:
    """Write best_hyperparameters.json entry for (frequency, tsteps) from a history row."""

    best_hps_overall: dict = {}
    if best_hps_path.exists():
        try:
            content = best_hps_path.read_text(encoding="utf-8").strip()
            best_hps_overall = json.loads(content) if content else {}
        except json.JSONDecodeError:
            best_hps_overall = {}

    freq_key = frequency
    tsteps_key = str(int(tsteps))
    best_hps_overall.setdefault(freq_key, {})

    # Minimal required fields + useful metadata for later resolution.
    best_hps_overall[freq_key][tsteps_key] = {
        "validation_loss": float(row.get("validation_loss")),
        "model_filename": row.get("model_filename"),
        "bias_correction_filename": row.get("bias_correction_filename"),
        "lstm_units": row.get("lstm_units"),
        "learning_rate": row.get("learning_rate"),
        "epochs": row.get("epochs"),
        "batch_size": row.get("batch_size"),
        "n_lstm_layers": row.get("n_lstm_layers"),
        "stateful": row.get("stateful"),
        "optimizer_name": row.get("optimizer_name"),
        "loss_function": row.get("loss_function"),
        "features_to_use": row.get("features_to_use"),
    }

    best_hps_path.write_text(json.dumps(best_hps_overall, indent=4), encoding="utf-8")


def _load_strategy_defaults() -> dict:
    """Reload src.config and return current strategy defaults.

    This ensures that when config.py is edited on disk (e.g. via the Save
    button below), subsequent reruns of the app see the updated values
    without needing to restart the Streamlit server.
    """

    # Reload via sys.modules so we do not depend on a particular alias being
    # present in ``sys.modules`` (e.g. after interactive usage or tests).
    module_name = "src.config"
    module = sys.modules.get(module_name)
    if module is not None:
        module = importlib.reload(module)
    else:
        module = importlib.import_module(module_name)

    # Keep a global alias for callers that still reference ``cfg_mod``.
    global cfg_mod
    cfg_mod = module

    # Prefer dedicated long/short constants when available; fall back to shared.
    base_k_sigma_long = float(getattr(module, "K_SIGMA_LONG", module.K_SIGMA_ERR))
    base_k_sigma_short = float(getattr(module, "K_SIGMA_SHORT", module.K_SIGMA_ERR))
    base_k_atr_long = float(getattr(module, "K_ATR_LONG", module.K_ATR_MIN_TP))
    base_k_atr_short = float(getattr(module, "K_ATR_SHORT", module.K_ATR_MIN_TP))

    return {
        # Shared aliases (kept for backwards compatibility and UI expectations)
        "k_sigma_err": float(module.K_SIGMA_ERR),
        "k_atr_min_tp": float(module.K_ATR_MIN_TP),
        # Side-specific defaults for UI/backtests
        "k_sigma_long": base_k_sigma_long,
        "k_sigma_short": base_k_sigma_short,
        "k_atr_long": base_k_atr_long,
        "k_atr_short": base_k_atr_short,
        "risk_per_trade_pct": float(module.RISK_PER_TRADE_PCT),
        "reward_risk_ratio": float(module.REWARD_RISK_RATIO),
    }


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

                        value = row.get("Value", base_row["Value"])
                        if pd.isna(value):
                            value = base_row["Value"]

                        start = row.get("Start", base_row["Start"])
                        if pd.isna(start):
                            start = base_row["Start"]

                        step = row.get("Step", base_row["Step"])
                        if pd.isna(step):
                            step = base_row["Step"]

                        stop = row.get("Stop", base_row["Stop"])
                        if pd.isna(stop):
                            stop = base_row["Stop"]

                        optimize_val = row.get("Optimize", base_row["Optimize"])
                        if pd.isna(optimize_val):
                            optimize_val = base_row["Optimize"]

                        rows.append(
                            {
                                "Parameter": name,
                                "Value": value,
                                "Start": start,
                                "Step": step,
                                "Stop": stop,
                                "Optimize": bool(optimize_val),
                            }
                        )
                    df = pd.DataFrame(rows)
                return df
        except Exception:
            # Fall back to defaults on any parse error.
            pass

    return _build_default_params_df(defaults)


def _save_params_grid(df: pd.DataFrame | object) -> None:
    """Persist the full parameter grid (Value/Start/Step/Stop/Optimize).

    Be defensive about the input type: older session_state entries or callers
    might pass a list/dict instead of a DataFrame.
    """

    try:
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        records = df.to_dict(orient="records")
        PARAMS_STATE_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - best-effort persistence
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
        "k_sigma_long": r"(k_sigma_long: float = )([0-9.eE+\-]+)",
        "k_sigma_short": r"(k_sigma_short: float = )([0-9.eE+\-]+)",
        "k_atr_long": r"(k_atr_long: float = )([0-9.eE+\-]+)",
        "k_atr_short": r"(k_atr_short: float = )([0-9.eE+\-]+)",
    }
    values = {
        "risk_per_trade_pct": risk_per_trade_pct,
        "reward_risk_ratio": reward_risk_ratio,
        "k_sigma_long": k_sigma_long,
        "k_sigma_short": k_sigma_short,
        "k_atr_long": k_atr_long,
        "k_atr_short": k_atr_short,
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


def _filter_backtest_history(history: list[dict], *, frequency: str) -> list[dict]:
    """Return backtest history rows matching a specific frequency."""
    rows = [
        r
        for r in (history or [])
        if isinstance(r, dict) and r.get("frequency") == frequency
    ]

    # Most-recent first.
    return sorted(rows, key=lambda r: r.get("timestamp", ""), reverse=True)


def _filter_optimization_history(history: list[dict], *, frequency: str) -> list[dict]:
    """Return optimization history rows matching a specific frequency."""
    rows = [
        r
        for r in (history or [])
        if isinstance(r, dict) and r.get("frequency") == frequency
    ]

    # Most-recent first.
    return sorted(rows, key=lambda r: r.get("timestamp", ""), reverse=True)


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
    """Non-cached wrapper around run_backtest_for_ui (CSV mode by default).

    The UI uses precomputed per-bar predictions (CSV mode) for backtests by
    default, which is simpler and more robust for experiments.
    """

    from src.config import get_predictions_csv_path as _get_predictions_csv_path

    predictions_csv = _get_predictions_csv_path("nvda", frequency)
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


def _set_optimization_running(is_running: bool) -> None:
    # Used to prevent Live-tab auto-refresh from interrupting long-running
    # optimization runs.
    st.session_state["optimization_running"] = bool(is_running)


def _start_optimization() -> None:
    _set_optimization_running(True)


def _stop_optimization() -> None:
    _set_optimization_running(False)


st.title("LSTM Backtesting & Training UI")

# Six main tabs matching the end-to-end workflow.
# "Live" is intended to be the primary operational dashboard during market hours.
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

# --------------------------------------------------------------------------------------
# Tab 0: Live - operational dashboard (Step 1: log-derived KPIs + recent events)
# --------------------------------------------------------------------------------------
with tab_live:
    from src.ui.pages import live_page

    live_page.render_live_tab(
        st=st,
        pd=pd,
        plt=plt,
    )

# --------------------------------------------------------------------------------------
# Tab 1: Data - ingestion, cleaning, resampling, and feature preview
# --------------------------------------------------------------------------------------
with tab_data:
    from src.ui.pages import data_page

    data_page.render_data_tab(
        st=st,
        pd=pd,
        os=os,
        Path=Path,
        project_root=Path(__file__).resolve().parents[1],
        RESAMPLE_FREQUENCIES=RESAMPLE_FREQUENCIES,
        FREQUENCY=FREQUENCY,
        clean_raw_minute_data=clean_raw_minute_data,
        convert_minute_to_timeframe=convert_minute_to_timeframe,
        prepare_keras_input_data=prepare_keras_input_data,
    )
# --------------------------------------------------------------------------------------
# Tab 2: Hyperparameter Experiments - short runs, no promotion
# --------------------------------------------------------------------------------------
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
        get_ui_state=_get_ui_state,
        load_json_history=_load_json_history,
        save_json_history=_save_json_history,
    )
# --------------------------------------------------------------------------------------
# Tab 3: Train & Promote - iterative runs + explicit promotion
# --------------------------------------------------------------------------------------
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
        get_ui_state=_get_ui_state,
        load_json_history=_load_json_history,
        save_json_history=_save_json_history,
        format_timestamp=_format_timestamp_iso_seconds,
        list_registry_models=_list_registry_models,
        promote_training_row=_promote_training_row,
    )

# --------------------------------------------------------------------------------------
# Tab 4: Backtest / Strategy - existing UI
# --------------------------------------------------------------------------------------
with tab_backtest:
    from src.ui.pages import backtest_page

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
        filter_backtest_history=_filter_backtest_history,
        filter_optimization_history=_filter_optimization_history,
        format_timestamp=_format_timestamp_iso_seconds,
        get_ui_state=_get_ui_state,
        load_json_history=_load_json_history,
        save_json_history=_save_json_history,
        start_optimization=_start_optimization,
        stop_optimization=_stop_optimization,
    )


# --------------------------------------------------------------------------------------
# Tab 5: Walk-Forward Analysis - train per fold OR robustness by parameter set
# --------------------------------------------------------------------------------------
with tab_walkforward:
    from src.ui.pages import walkforward_page

    walkforward_page.render_walkforward_tab(
        st=st,
        pd=pd,
        plt=plt,
        FREQUENCY=FREQUENCY,
        RESAMPLE_FREQUENCIES=RESAMPLE_FREQUENCIES,
        MAX_HISTORY_ROWS=MAX_HISTORY_ROWS,
        get_ui_state=_get_ui_state,
        load_strategy_defaults=_load_strategy_defaults,
        save_json_history=_save_json_history,
    )
