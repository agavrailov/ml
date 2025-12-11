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
from src.evaluate_model import evaluate_model_performance
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
    get_run_hyperparameters,
    get_latest_best_model_path,
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
    default, which is simpler and more robust for experiments. Model mode can
    still be enabled by calling :func:`run_backtest_for_ui` directly with
    ``prediction_mode="model"`` when needed.
    """

    from src.config import get_predictions_csv_path as _get_predictions_csv_path

    predictions_csv = _get_predictions_csv_path("nvda", frequency)

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
st.title("LSTM Backtesting & Training UI")

# Five main tabs matching the end-to-end workflow.
tab_data, tab_experiments, tab_train, tab_backtest, tab_walkforward = st.tabs(
    ["Data", "Hyperparameter Experiments", "Train & Promote", "Backtest / Strategy", "Walk-Forward Analysis"]
)

# --------------------------------------------------------------------------------------
# Tab 1: Data - ingestion, cleaning, resampling, and feature preview
# --------------------------------------------------------------------------------------
with tab_data:
    st.subheader("1. Data ingestion & preparation")

    from src.config import RAW_DATA_CSV, PROCESSED_DATA_DIR, FREQUENCY as CFG_FREQ
    from src.data_quality import (
        analyze_raw_minute_data,
        compute_quality_kpi,
        format_quality_report,
        get_missing_trading_days,
    )

    st.markdown("### Raw minute data")
    st.write(f"Raw data CSV: `{RAW_DATA_CSV}`")

    st.markdown("#### Data quality checks")
    if st.button("Run data quality checks on raw minute data"):
        try:
            checks = analyze_raw_minute_data(RAW_DATA_CSV)
            if not checks:
                st.info("No checks were run (file may be missing or empty).")
            else:
                kpi = compute_quality_kpi(checks)

                # Overall KPI summary
                c_kpi1, c_kpi2, c_kpi3 = st.columns(3)
                with c_kpi1:
                    st.metric("Data quality score", f"{kpi['score_0_100']:.1f} / 100")
                with c_kpi2:
                    st.metric("Checks passed", f"{kpi['n_pass']} / {kpi['n_total']}")
                with c_kpi3:
                    st.metric("Warnings / Failures", f"{kpi['n_warn']} warn, {kpi['n_fail']} fail")

                # Detailed table of all checks
                checks_df = pd.DataFrame(checks).sort_values(["category", "id"])
                st.dataframe(checks_df, width="stretch")

                # Prominent status banner
                if kpi["n_fail"]:
                    st.error(
                        f"{kpi['n_fail']} checks FAILED. Review the table above before trusting the data.",
                    )
                elif kpi["n_warn"]:
                    st.warning(
                        f"All critical checks passed, but {kpi['n_warn']} checks have WARNINGS. "
                        "Review them for potential issues.",
                    )
                else:
                    st.success("All data quality checks passed.")

                # Downloadable plain-text/Markdown report
                report_text = format_quality_report(checks, kpi, dataset_name="Raw minute data")
                st.download_button(
                    label="Download data quality report",
                    data=report_text,
                    file_name="raw_data_quality_report.txt",
                    mime="text/plain",
                )
        except Exception as exc:  # pragma: no cover - UI convenience only
            st.error(f"Data quality checks failed: {exc}")

    st.markdown("#### Backfill missing trading days via TWS")
    if st.button("Backfill missing trading days from IB/TWS"):
        try:
            # Determine holiday-aware missing trading days based on NASDAQ calendar.
            missing_days = get_missing_trading_days(RAW_DATA_CSV, calendar_name="NASDAQ")
            if not missing_days:
                st.info("No missing trading days detected (holiday-aware NASDAQ calendar).")
            else:
                # Use the earliest and latest missing day as the backfill range.
                start_missing = min(missing_days)
                end_missing = max(missing_days)

                # IB's historical API uses an end timestamp that is effectively
                # exclusive; add one day so we cover the full last missing date.
                start_str = start_missing.strftime("%Y-%m-%d")
                end_str = (end_missing + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

                import subprocess as _subprocess
                import sys as _sys

                project_root = Path(__file__).resolve().parents[1]
                cmd = [
                    _sys.executable,
                    "-m",
                    "src.data_ingestion",
                    "--start",
                    start_str,
                    "--end",
                    end_str,
                    "--strict-range",
                ]

                st.write(
                    "Triggering TWS backfill for missing trading days via CLI:",
                    " ".join(cmd),
                )
                with st.spinner(
                    f"Backfilling missing trading days from {start_str} to {end_str} via TWS...",
                ):
                    result = _subprocess.run(
                        cmd,
                        cwd=project_root,
                        check=False,
                        capture_output=True,
                        text=True,
                    )

                if result.returncode == 0:
                    # Clean the raw minute data to sort and de-duplicate after backfill.
                    try:
                        clean_raw_minute_data(RAW_DATA_CSV)
                        st.success(
                            "Backfill completed and raw minute data re-cleaned. "
                            "Re-run data quality checks to verify the gaps are closed.",
                        )
                    except Exception as exc:  # pragma: no cover - best-effort cleanup
                        st.warning(
                            "Backfill completed, but cleaning the raw data failed: "
                            f"{exc}",
                        )
                else:
                    st.error(
                        f"Backfill CLI exited with code {result.returncode}. "
                        "See logs below.",
                    )

                if result.stdout:
                    with st.expander("Backfill log (stdout)"):
                        st.code(result.stdout)
                if result.stderr:
                    with st.expander("Backfill log (stderr)"):
                        st.code(result.stderr)
        except Exception as exc:  # pragma: no cover - UI convenience only
            st.error(f"Backfill of missing trading days failed: {exc}")
    col_ingest1, col_ingest2 = st.columns(2)
    with col_ingest1:
        if st.button("Fetch/update raw NVDA data from TWS"):
            try:
                # Run the ingestion CLI in a subprocess to avoid importing ib_insync
                # (and its event-loop setup) into the Streamlit runtime.
                import subprocess, sys as _sys
                project_root = Path(__file__).resolve().parents[1]
                cmd = [_sys.executable, "-m", "src.data_ingestion"]

                status = st.empty()
                progress = st.progress(0.0)
                status.write(f"Running: {' '.join(cmd)} in {project_root}")

                with st.spinner("Fetching historical data via TWS... this may take several minutes."):
                    result = subprocess.run(
                        cmd,
                        cwd=project_root,
                        check=False,
                        capture_output=True,
                        text=True,
                    )

                progress.progress(1.0)

                if result.returncode == 0:
                    st.success("TWS historical ingestion completed successfully.")
                else:
                    st.error(f"Ingestion CLI exited with code {result.returncode}.")

                if result.stdout:
                    with st.expander("Ingestion log (stdout)"):
                        st.code(result.stdout)
                if result.stderr:
                    with st.expander("Ingestion log (stderr)"):
                        st.code(result.stderr)
            except Exception as exc:  # pragma: no cover - UI side-effect only
                st.error(f"Failed to trigger ingestion: {exc}")

    with col_ingest2:
        if st.button("Clean raw minute data (sort, dedupe)"):
            try:
                clean_raw_minute_data(RAW_DATA_CSV)
                st.success("Raw minute data cleaned.")
            except Exception as exc:  # pragma: no cover - UI side-effect only
                st.error(f"Failed to clean raw minute data: {exc}")

    st.markdown("### Hourly data & features")
    _global_freq = st.session_state.get("global_frequency", CFG_FREQ)
    if _global_freq not in RESAMPLE_FREQUENCIES:
        _global_freq = RESAMPLE_FREQUENCIES[0]
    data_freq = st.selectbox(
        "Frequency for resampling & features",
        RESAMPLE_FREQUENCIES,
        index=RESAMPLE_FREQUENCIES.index(_global_freq),
        key="data_freq_select",
    )
    st.session_state["global_frequency"] = data_freq

    col_resample, col_features = st.columns(2)
    with col_resample:
        if st.button("Resample minute → hourly CSV"):
            try:
                os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
                convert_minute_to_timeframe(RAW_DATA_CSV, data_freq, PROCESSED_DATA_DIR)
                st.success(f"Resampled data saved for {data_freq}.")
            except Exception as exc:  # pragma: no cover - UI side-effect only
                st.error(f"Failed to resample data: {exc}")

    with col_features:
        if st.button("Prepare features for Keras input"):
            try:
                from src.config import FEATURES_TO_USE_OPTIONS as _FEAT_OPTS, get_hourly_data_csv_path

                default_features = _FEAT_OPTS[0]
                hourly_path = get_hourly_data_csv_path(data_freq)
                df_feat, feat_cols = prepare_keras_input_data(hourly_path, default_features)
                st.session_state["data_feature_preview"] = (df_feat.head(200), feat_cols)
                st.success(f"Prepared features for {data_freq}: {feat_cols}")
            except Exception as exc:  # pragma: no cover - UI side-effect only
                st.error(f"Failed to prepare features: {exc}")

    if "data_feature_preview" in st.session_state:
        df_prev, feat_cols_prev = st.session_state["data_feature_preview"]
        st.markdown("#### Feature preview (first 200 rows)")
        st.write(f"Columns: {feat_cols_prev}")
        st.dataframe(df_prev, width="stretch")

# --------------------------------------------------------------------------------------
# Tab 2: Hyperparameter Experiments - short runs, no promotion
# --------------------------------------------------------------------------------------
with tab_experiments:
    st.subheader("2. Hyperparameter experiments (no promotion)")

    ui_state = _get_ui_state()
    exp_state = ui_state.setdefault("experiments", {})
    if "runs" not in exp_state:
        exp_state["runs"] = _load_json_history("experiments_runs.json")

    # Frequency / TSTEPS for experiments.
    _global_freq = st.session_state.get("global_frequency", FREQUENCY)
    if _global_freq not in RESAMPLE_FREQUENCIES:
        _global_freq = RESAMPLE_FREQUENCIES[0]
    exp_freq = st.selectbox(
        "Frequency",
        RESAMPLE_FREQUENCIES,
        index=RESAMPLE_FREQUENCIES.index(_global_freq),
        key="exp_freq_select",
    )
    st.session_state["global_frequency"] = exp_freq
    try:
        exp_tsteps_idx = TSTEPS_OPTIONS.index(TSTEPS)
    except ValueError:
        exp_tsteps_idx = 0
    exp_tsteps = st.selectbox(
        "Sequence length (TSTEPS)",
        TSTEPS_OPTIONS,
        index=exp_tsteps_idx,
        key="exp_tsteps_select",
    )

    # Use current best hyperparameters as defaults for experiments.
    exp_hps = get_run_hyperparameters(frequency=exp_freq, tsteps=exp_tsteps)

    c1, c2, c3 = st.columns(3)
    with c1:
        try:
            idx_units = LSTM_UNITS_OPTIONS.index(exp_hps["lstm_units"])
        except ValueError:
            idx_units = 0
        exp_lstm_units = st.selectbox(
            "LSTM units",
            LSTM_UNITS_OPTIONS,
            index=idx_units,
            key="exp_lstm_units_select",
        )

        try:
            idx_layers = N_LSTM_LAYERS_OPTIONS.index(exp_hps["n_lstm_layers"])
        except ValueError:
            idx_layers = 0
        exp_n_layers = st.selectbox(
            "LSTM layers",
            N_LSTM_LAYERS_OPTIONS,
            index=idx_layers,
            key="exp_lstm_layers_select",
        )

    with c2:
        try:
            idx_bs = BATCH_SIZE_OPTIONS.index(exp_hps["batch_size"])
        except ValueError:
            idx_bs = 0
        exp_batch_size = st.selectbox(
            "Batch size",
            BATCH_SIZE_OPTIONS,
            index=idx_bs,
            key="exp_batch_size_select",
        )

        exp_epochs = st.slider(
            "Epochs (short experiments)",
            min_value=1,
            max_value=50,
            value=min(10, int(exp_hps["epochs"])),
            key="exp_epochs_slider",
        )

    with c3:
        lr_choices = [0.0005, 0.001, 0.003, 0.01]
        default_lr = float(exp_hps["learning_rate"])
        if default_lr not in lr_choices:
            lr_choices = sorted(set(lr_choices + [default_lr]))
        try:
            idx_lr = lr_choices.index(default_lr)
        except ValueError:
            idx_lr = 0
        exp_lr = st.selectbox(
            "Learning rate",
            lr_choices,
            index=idx_lr,
            key="exp_lr_select",
        )

        try:
            idx_stateful = STATEFUL_OPTIONS.index(exp_hps["stateful"])
        except ValueError:
            idx_stateful = 0
        exp_stateful = st.selectbox(
            "Stateful LSTM",
            STATEFUL_OPTIONS,
            index=idx_stateful,
            key="exp_stateful_select",
        )

    exp_feature_set_idx = st.selectbox(
        "Feature set",
        options=list(range(len(FEATURES_TO_USE_OPTIONS))),
        format_func=lambda i: f"Set {i + 1}: " + ", ".join(FEATURES_TO_USE_OPTIONS[i]),
        key="exp_feature_set_select",
    )
    exp_features_to_use = FEATURES_TO_USE_OPTIONS[exp_feature_set_idx]

    st.caption("Experiments run shorter trainings and DO NOT update best_hyperparameters.json.")

    if st.button("Run single experiment"):
        progress = st.progress(0.0)
        status = st.empty()
        try:
            status.write("Running short training experiment...")
            progress.progress(0.1)

            result = train_model(
                frequency=exp_freq,
                tsteps=exp_tsteps,
                lstm_units=int(exp_lstm_units),
                learning_rate=float(exp_lr),
                epochs=int(exp_epochs),
                current_batch_size=int(exp_batch_size),
                n_lstm_layers=int(exp_n_layers),
                stateful=bool(exp_stateful),
                features_to_use=exp_features_to_use,
            )

            if result is None:
                status.write("")
                progress.progress(0.0)
                st.error("Experiment training failed or not enough data.")
            else:
                final_val_loss, model_path, bias_path = result
                progress.progress(1.0)
                status.write("Experiment finished.")

                # Present validation loss in scientific notation and show only
                # the model filename (not the full path) in the status message.
                st.success(
                    f"Experiment val_loss={final_val_loss:.3e}, "
                    f"model={os.path.basename(model_path)}"
                )

                # Append to in-memory experiments table in session_state.
                # Store validation loss as a scientific-notation string and only
                # the filenames for the model and bias-correction artifacts.
                row = {
                    "timestamp": pd.Timestamp.utcnow().isoformat(),
                    "frequency": exp_freq,
                    "tsteps": exp_tsteps,
                    "lstm_units": int(exp_lstm_units),
                    "batch_size": int(exp_batch_size),
                    "learning_rate": float(exp_lr),
                    "n_lstm_layers": int(exp_n_layers),
                    "stateful": bool(exp_stateful),
                    "features_to_use": ",".join(exp_features_to_use),
                    "epochs": int(exp_epochs),
                    "validation_loss": f"{float(final_val_loss):.3e}",
                    "model": os.path.basename(model_path),
                    "correction_path": os.path.basename(bias_path),
                }
                # Backwards-compatible: keep legacy session_state list, but
                # treat ui_state.experiments.runs as the primary store.
                st.session_state.setdefault("lstm_experiments", []).append(row)

                runs: list[dict] = exp_state.get("runs", [])
                runs.append(row)
                if len(runs) > MAX_HISTORY_ROWS:
                    runs = runs[-MAX_HISTORY_ROWS:]
                exp_state["runs"] = runs
                _save_json_history("experiments_runs.json", runs)
        except Exception as exc:  # pragma: no cover - UI convenience
            progress.progress(0.0)
            status.write("")
            st.error(f"Experiment failed: {exc}")

    # Experiments table with an action to load a row into the Train tab.
    exp_rows = exp_state.get("runs") or st.session_state.get("lstm_experiments", [])
    if exp_rows:
        st.markdown("### Recorded experiments")
        exp_df = pd.DataFrame(exp_rows)
        st.dataframe(exp_df, width="stretch")

        # Simple selector to copy a row's hyperparameters into the Train tab.
        idx_to_use = st.number_input(
            "Select experiment row index to load into 'Train & Promote' tab",
            min_value=0,
            max_value=len(exp_rows) - 1,
            step=1,
            value=len(exp_rows) - 1,
            key="exp_row_select",
        )
        if st.button("Load selected experiment into Train & Promote"):
            chosen = exp_rows[int(idx_to_use)]
            st.session_state["train_prefill"] = chosen
            # Mirror into ui_state.training so the Train tab can prefer it.
            train_state = _get_ui_state().setdefault("training", {})
            train_state["train_prefill"] = chosen
            _ = train_state  # silence linters about unused variable in UI code
            st.success("Loaded experiment into Train & Promote tab. Switch to that tab to train fully.")

# --------------------------------------------------------------------------------------
# Tab 3: Train & Promote - full training with auto-promotion when better
# --------------------------------------------------------------------------------------
with tab_train:
    st.subheader("3. Train full model & promote when better")

    ui_state = _get_ui_state()
    train_state = ui_state.setdefault("training", {})
    if "history" not in train_state:
        train_state["history"] = _load_json_history("training_history.json")

    # Prefill from last experiment if available, otherwise from current best.
    prefill = train_state.get("train_prefill") or st.session_state.get("train_prefill")
    if prefill:
        train_freq = prefill["frequency"]
        train_tsteps = int(prefill["tsteps"])
    else:
        train_freq = FREQUENCY
        train_tsteps = TSTEPS

    # Use shared global frequency as the primary default.
    _global_freq = st.session_state.get("global_frequency", train_freq)
    if _global_freq not in RESAMPLE_FREQUENCIES:
        _global_freq = train_freq if train_freq in RESAMPLE_FREQUENCIES else RESAMPLE_FREQUENCIES[0]
    train_freq = st.selectbox(
        "Frequency",
        RESAMPLE_FREQUENCIES,
        index=RESAMPLE_FREQUENCIES.index(_global_freq),
        key="train_freq_select",
    )
    st.session_state["global_frequency"] = train_freq
    try:
        train_tsteps_idx = TSTEPS_OPTIONS.index(train_tsteps)
    except ValueError:
        train_tsteps_idx = 0
    train_tsteps = st.selectbox(
        "Sequence length (TSTEPS)",
        TSTEPS_OPTIONS,
        index=train_tsteps_idx,
        key="train_tsteps_select",
    )

    # Resolve effective hyperparameters for this run.
    base_hps = get_run_hyperparameters(frequency=train_freq, tsteps=train_tsteps)
    if prefill:
        # Override with prefill when present.
        base_hps.update(
            {
                "lstm_units": int(prefill["lstm_units"]),
                "batch_size": int(prefill["batch_size"]),
                "learning_rate": float(prefill["learning_rate"]),
                "epochs": int(prefill["epochs"]),
                "n_lstm_layers": int(prefill["n_lstm_layers"]),
                "stateful": bool(prefill["stateful"]),
                # features_to_use is handled via index below.
            }
        )

    c1t, c2t, c3t = st.columns(3)
    with c1t:
        try:
            idx_units_t = LSTM_UNITS_OPTIONS.index(base_hps["lstm_units"])
        except ValueError:
            idx_units_t = 0
        train_lstm_units = st.selectbox(
            "LSTM units",
            LSTM_UNITS_OPTIONS,
            index=idx_units_t,
            key="train_lstm_units_select",
        )

        try:
            idx_layers_t = N_LSTM_LAYERS_OPTIONS.index(base_hps["n_lstm_layers"])
        except ValueError:
            idx_layers_t = 0
        train_n_layers = st.selectbox(
            "LSTM layers",
            N_LSTM_LAYERS_OPTIONS,
            index=idx_layers_t,
            key="train_lstm_layers_select",
        )

    with c2t:
        try:
            idx_bs_t = BATCH_SIZE_OPTIONS.index(base_hps["batch_size"])
        except ValueError:
            idx_bs_t = 0
        train_batch_size = st.selectbox(
            "Batch size",
            BATCH_SIZE_OPTIONS,
            index=idx_bs_t,
            key="train_batch_size_select",
        )

        train_epochs = st.slider(
            "Epochs",
            min_value=1,
            max_value=200,
            value=int(base_hps["epochs"]),
            key="train_epochs_slider",
        )

    with c3t:
        lr_choices_t = [0.0005, 0.001, 0.003, 0.01]
        default_lr_t = float(base_hps["learning_rate"])
        if default_lr_t not in lr_choices_t:
            lr_choices_t = sorted(set(lr_choices_t + [default_lr_t]))
        try:
            idx_lr_t = lr_choices_t.index(default_lr_t)
        except ValueError:
            idx_lr_t = 0
        train_lr = st.selectbox(
            "Learning rate",
            lr_choices_t,
            index=idx_lr_t,
            key="train_lr_select",
        )

        try:
            idx_stateful_t = STATEFUL_OPTIONS.index(base_hps["stateful"])
        except ValueError:
            idx_stateful_t = 0
        train_stateful = st.selectbox(
            "Stateful LSTM",
            STATEFUL_OPTIONS,
            index=idx_stateful_t,
            key="train_stateful_select",
        )

    # Feature set selection (respecting prefill when present).
    if prefill and "features_to_use" in prefill:
        feats_prefill = prefill["features_to_use"].split(",")
        try:
            train_feat_idx = next(
                i for i, fs in enumerate(FEATURES_TO_USE_OPTIONS) if fs == feats_prefill
            )
        except StopIteration:
            train_feat_idx = 0
    else:
        train_feat_idx = 0

    train_feature_set_idx = st.selectbox(
        "Feature set",
        options=list(range(len(FEATURES_TO_USE_OPTIONS))),
        index=train_feat_idx,
        format_func=lambda i: f"Set {i + 1}: " + ", ".join(FEATURES_TO_USE_OPTIONS[i]),
        key="train_feature_set_select",
    )
    train_features_to_use = FEATURES_TO_USE_OPTIONS[train_feature_set_idx]

    # Show current best for this (frequency, tsteps).
    st.markdown("### Current best for this Frequency / TSTEPS")
    import os as _os

    best_hps_path_train = Path(__file__).resolve().parents[1] / "best_hyperparameters.json"
    best_block = None
    if best_hps_path_train.exists():
        try:
            content = best_hps_path_train.read_text(encoding="utf-8").strip()
            if content:
                all_best = json.loads(content)
                freq_block = all_best.get(train_freq, {})
                best_block = freq_block.get(str(train_tsteps))
        except json.JSONDecodeError:
            best_block = None

    if best_block:
        st.write(
            f"Best val_loss={best_block.get('validation_loss', float('inf')):.6f}, "
            f"model={best_block.get('model_filename', '<unknown>')}"
        )
    else:
        st.write("No best entry yet for this Frequency/TSTEPS.")

    # Full training with auto-promotion when better, plus predictions CSV.
    if st.button("Train full model & auto-promote if better"):
        progress = st.progress(0.0)
        status = st.empty()
        try:
            status.write("Step 1/3: Training full LSTM model...")
            progress.progress(0.1)

            result = train_model(
                frequency=train_freq,
                tsteps=train_tsteps,
                lstm_units=int(train_lstm_units),
                learning_rate=float(train_lr),
                epochs=int(train_epochs),
                current_batch_size=int(train_batch_size),
                n_lstm_layers=int(train_n_layers),
                stateful=bool(train_stateful),
                features_to_use=train_features_to_use,
            )

            if result is None:
                status.write("")
                progress.progress(0.0)
                st.error("Training failed or not enough data.")
            else:
                final_val_loss, model_path, bias_correction_path = result
                progress.progress(0.6)
                status.write("Step 2/3: Updating best_hyperparameters.json (auto-promotion if better)...")

                promoted = False

                # Auto-promotion logic (same pattern as before).
                if best_hps_path_train.exists():
                    try:
                        content = best_hps_path_train.read_text(encoding="utf-8").strip()
                        best_hps_overall = json.loads(content) if content else {}
                    except json.JSONDecodeError:
                        best_hps_overall = {}
                else:
                    best_hps_overall = {}

                freq_key = train_freq
                tsteps_key = str(train_tsteps)
                if freq_key not in best_hps_overall:
                    best_hps_overall[freq_key] = {}

                prev_best = best_hps_overall[freq_key].get(tsteps_key, {})
                prev_loss = prev_best.get("validation_loss", float("inf"))

                if final_val_loss < prev_loss:
                    best_hps_overall[freq_key][tsteps_key] = {
                        "validation_loss": float(final_val_loss),
                        "model_filename": _os.path.basename(model_path),
                        "lstm_units": int(train_lstm_units),
                        "learning_rate": float(train_lr),
                        "epochs": int(train_epochs),
                        "batch_size": int(train_batch_size),
                        "n_lstm_layers": int(train_n_layers),
                        "stateful": bool(train_stateful),
                        "optimizer_name": base_hps["optimizer_name"],
                        "loss_function": base_hps["loss_function"],
                        "features_to_use": train_features_to_use,
                        "bias_correction_filename": _os.path.basename(bias_correction_path),
                    }
                    best_hps_path_train.write_text(
                        json.dumps(best_hps_overall, indent=4), encoding="utf-8"
                    )
                    st.info("Auto-promoted this model as new best for this Frequency/TSTEPS.")
                    promoted = True
                else:
                    st.info(
                        f"Existing best validation loss for (freq={freq_key}, tsteps={tsteps_key}) "
                        f"is {prev_loss:.6f}; not promoting this run.",
                    )

                progress.progress(0.8)
                status.write("Step 3/3: Generating per-bar predictions CSV for NVDA...")

                predictions_csv_path = get_predictions_csv_path("nvda", train_freq)
                generate_predictions_for_csv(frequency=train_freq, output_path=predictions_csv_path)

                progress.progress(1.0)
                status.write("Training and prediction generation complete.")

                st.success(
                    f"Training finished. Validation loss: {final_val_loss:.6f}. "
                    f"Predictions CSV written to: {predictions_csv_path}",
                )
                st.code(str(predictions_csv_path))

                # Record last training run and append a compact history row.
                train_state["last_train_run"] = {
                    "timestamp": pd.Timestamp.utcnow().isoformat(),
                    "frequency": train_freq,
                    "tsteps": int(train_tsteps),
                    "lstm_units": int(train_lstm_units),
                    "batch_size": int(train_batch_size),
                    "learning_rate": float(train_lr),
                    "epochs": int(train_epochs),
                    "n_lstm_layers": int(train_n_layers),
                    "stateful": bool(train_stateful),
                    "features_to_use": train_features_to_use,
                    "validation_loss": float(final_val_loss),
                    "promoted": bool(promoted),
                    "model_filename": os.path.basename(model_path),
                    "bias_correction_filename": os.path.basename(bias_correction_path),
                    "predictions_csv_path": str(predictions_csv_path),
                }
                history: list[dict] = train_state.get("history", [])
                history.append(
                    {
                        "timestamp": train_state["last_train_run"]["timestamp"],
                        "frequency": train_freq,
                        "tsteps": int(train_tsteps),
                        "validation_loss": float(final_val_loss),
                        "promoted": bool(promoted),
                        "model_filename": os.path.basename(model_path),
                    }
                )
                if len(history) > MAX_HISTORY_ROWS:
                    history = history[-MAX_HISTORY_ROWS:]
                train_state["history"] = history
                _save_json_history("training_history.json", history)
        except Exception as exc:  # pragma: no cover - UI convenience
            progress.progress(0.0)
            status.write("")
            st.error(f"Error during full training pipeline: {exc}")

    # Recent training runs table.
    train_history = train_state.get("history", [])
    if train_history:
        st.markdown("### Recent training runs (most recent first)")
        train_hist_df = pd.DataFrame(train_history)
        if "timestamp" in train_hist_df.columns:
            train_hist_df = train_hist_df.sort_values("timestamp", ascending=False)
        st.dataframe(train_hist_df, width="stretch")

    st.markdown("---")
    st.subheader("Evaluate best model")

    if st.button("Evaluate best model for this frequency / TSTEPS"):
        with st.spinner("Resolving best model and running evaluation..."):
            (
                best_model_path,
                bias_correction_path,
                features_trained,
                lstm_units_trained,
                n_lstm_layers_trained,
            ) = get_latest_best_model_path(target_frequency=train_freq, tsteps=train_tsteps)

            if best_model_path is None:
                st.error(
                    f"No best model found for frequency={train_freq}, TSTEPS={train_tsteps}. "
                    "Train at least one model first.",
                )
            else:
                if features_trained is None:
                    features_trained = train_features_to_use
                n_features_trained = len(features_trained)
                effective_lstm_units = lstm_units_trained or int(train_lstm_units)
                effective_n_layers = n_lstm_layers_trained or int(train_n_layers)

                mae, corr = evaluate_model_performance(
                    model_path=best_model_path,
                    frequency=train_freq,
                    tsteps=train_tsteps,
                    n_features=n_features_trained,
                    lstm_units=effective_lstm_units,
                    n_lstm_layers=effective_n_layers,
                    stateful=bool(train_stateful),
                    features_to_use=features_trained,
                    bias_correction_path=bias_correction_path,
                )

        if "best_model_path" in locals() and best_model_path is not None:
            # Cache last evaluation in ui_state (session-only, not persisted).
            train_state["last_evaluation"] = {
                "frequency": train_freq,
                "tsteps": int(train_tsteps),
                "mae": float(mae),
                "corr": float(corr),
            }

            st.success("Evaluation complete.")
            st.write(
                f"**MAE (log returns)**: `{mae:.6f}`  |  "
                f"**Correlation (actual vs predicted log returns)**: `{corr:.4f}`",
            )

            models_dir = Path(__file__).resolve().parent.parent / "models"
            eval_plot = models_dir / "evaluation_plot.png"
            residuals_plot = models_dir / "residuals_histogram.png"

            if eval_plot.exists():
                st.image(
                    str(eval_plot),
                    caption="Evaluation: actual vs predicted log returns",
                    use_column_width=True,
                )
            else:
                st.info("Evaluation plot not found on disk.")

            if residuals_plot.exists():
                st.image(
                    str(residuals_plot),
                    caption="Residuals distribution (log returns)",
                    use_column_width=True,
                )
            else:
                st.info("Residuals histogram not found on disk.")

# --------------------------------------------------------------------------------------
# Tab 4: Backtest / Strategy - existing UI
# --------------------------------------------------------------------------------------
with tab_backtest:
    ui_state = _get_ui_state()
    bt_state = ui_state.setdefault("backtests", {})

    # Lazily load backtest history from disk once per session.
    if "history" not in bt_state:
        bt_state["history"] = _load_json_history("backtests_history.json")

    # Allow selecting any configured resample frequency; default to the main config FREQUENCY.
    _available_freqs = getattr(cfg_mod, "RESAMPLE_FREQUENCIES", ["15min"])
    _default_freq = st.session_state.get("global_frequency", getattr(cfg_mod, "FREQUENCY", _available_freqs[0]))
    try:
        _default_index = _available_freqs.index(_default_freq)
    except ValueError:
        _default_index = 0
    freq = st.selectbox("Frequency", _available_freqs, index=_default_index)
    if freq in _available_freqs:
        st.session_state["global_frequency"] = freq

    # Optional convenience: regenerate the predictions CSV for the selected
    # frequency without leaving the UI. This uses the same unified prediction
    # pipeline as the CLI so that CSV-mode backtests see the same signals as
    # model-mode.
    if st.button("Generate predictions CSV for NVDA (current frequency)"):
        predictions_csv_path = get_predictions_csv_path("nvda", freq)
        try:
            with st.spinner(f"Generating predictions CSV for NVDA at {freq}..."):
                generate_predictions_for_csv(frequency=freq, output_path=predictions_csv_path)
            st.success(f"Predictions CSV written to: {predictions_csv_path}")
            st.code(str(predictions_csv_path))
        except Exception as exc:  # pragma: no cover - UI convenience only
            st.error(f"Failed to generate predictions CSV: {exc}")

    # Reload current defaults from config.py so changes on disk are visible
    # without restarting the Streamlit server.
    _defaults = _load_strategy_defaults()

    # Parameter grid (similar to MT5 Strategy Tester): each row is a parameter
    # with Value / Start / Step / Stop / Optimize. The grid is persisted to a
    # JSON sidecar so manual edits survive app reloads.
    st.subheader("Strategy parameters")

    # Initialize from any existing widget value (key "strategy_params"). On the
    # very first run, fall back to loading from disk via _load_params_grid.
    stored_params = st.session_state.get("strategy_params", None)
    if isinstance(stored_params, pd.DataFrame):
        params_df_initial = stored_params
    elif stored_params is not None:
        try:
            params_df_initial = pd.DataFrame(stored_params)
        except Exception:
            params_df_initial = _load_params_grid(_defaults)
    else:
        params_df_initial = _load_params_grid(_defaults)

    params_df = st.data_editor(
        params_df_initial,
        num_rows="fixed",
        key="strategy_params",
        width="stretch",
        hide_index=True,
        column_config={
            "Parameter": st.column_config.TextColumn("Parameter", disabled=True),
            "Value": st.column_config.NumberColumn("Value"),
            "Start": st.column_config.NumberColumn("Start"),
            "Step": st.column_config.NumberColumn("Step"),
            "Stop": st.column_config.NumberColumn("Stop"),
            "Optimize": st.column_config.CheckboxColumn("Optimize"),
        },
    )

    # Extract current "Value" settings from the parameter grid.
    _param_values = {row["Parameter"]: row["Value"] for _, row in params_df.iterrows()}
    k_sigma_long_val = float(_param_values.get("k_sigma_long", _defaults["k_sigma_long"]))
    k_sigma_short_val = float(_param_values.get("k_sigma_short", _defaults["k_sigma_short"]))
    k_atr_long_val = float(_param_values.get("k_atr_long", _defaults["k_atr_long"]))
    k_atr_short_val = float(_param_values.get("k_atr_short", _defaults["k_atr_short"]))
    risk_pct_val = float(_param_values.get("risk_per_trade_pct", _defaults["risk_per_trade_pct"]))
    rr_val = float(_param_values.get("reward_risk_ratio", _defaults["reward_risk_ratio"]))


    if st.button("Save parameters to config.py"):
        # Persist single-run defaults to config.py (used by CLIs and as base
        # defaults) and the full grid (Value/Start/Step/Stop/Optimize) to a
        # JSON sidecar used only by this UI.
        _save_strategy_defaults_to_config(
            risk_per_trade_pct=risk_pct_val,
            reward_risk_ratio=rr_val,
            # Persist both long and short filters into config.py.
            k_sigma_long=k_sigma_long_val,
            k_sigma_short=k_sigma_short_val,
            k_atr_long=k_atr_long_val,
            k_atr_short=k_atr_short_val,
        )
        # Use the latest in-memory grid from the widget (including any unsaved edits).
        current_params_df = st.session_state.get("strategy_params", params_df)
        _save_params_grid(current_params_df)
        st.success("Saved strategy defaults and parameter grid (Value/Start/Step/Stop).")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.text_input("Start date (YYYY-MM-DD, optional)", "")
    with col2:
        end_date = st.text_input("End date (YYYY-MM-DD, optional)", "")

    # Prediction source is always the trained LSTM model; no CSV/naive modes.
    mode = st.radio("Mode", ["Run backtest", "Optimize"], horizontal=True)

    trade_side = st.radio(
        "Trades to include",
        ["Long only", "Short only", "Long & short"],
        horizontal=True,
    )
    if trade_side == "Long only":
        enable_longs_flag = True
        allow_shorts_flag = False
    elif trade_side == "Short only":
        enable_longs_flag = False
        allow_shorts_flag = True
    else:  # "Long & short"
        enable_longs_flag = True
        allow_shorts_flag = True

    if mode == "Run backtest" and st.button("Run backtest"):
        with st.spinner("Running backtest..."):
            equity_df, trades_df, metrics = _run_backtest(
                frequency=freq,
                start_date=start_date or None,
                end_date=end_date or None,
                risk_per_trade_pct=risk_pct_val,
                reward_risk_ratio=rr_val,
                k_sigma_long=k_sigma_long_val,
                k_sigma_short=k_sigma_short_val,
                k_atr_long=k_atr_long_val,
                k_atr_short=k_atr_short_val,
                enable_longs=enable_longs_flag,
                allow_shorts=allow_shorts_flag,
            )

        # Store last run (in-session) and append a compact summary row to history.
        bt_state["last_run"] = {
            "inputs": {
                "frequency": freq,
                "start_date": start_date or None,
                "end_date": end_date or None,
                "trade_side": trade_side,
                "enable_longs": enable_longs_flag,
                "allow_shorts": allow_shorts_flag,
                "risk_per_trade_pct": risk_pct_val,
                "reward_risk_ratio": rr_val,
                "k_sigma_long": k_sigma_long_val,
                "k_sigma_short": k_sigma_short_val,
                "k_atr_long": k_atr_long_val,
                "k_atr_short": k_atr_short_val,
            },
            "metrics": metrics,
            "equity_df": equity_df,
        }

        history: list[dict] = bt_state.get("history", [])
        summary_row = {
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "frequency": freq,
            "trade_side": trade_side,
            "total_return": float(metrics.get("total_return", 0.0)),
            "cagr": float(metrics.get("cagr", 0.0)),
            "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
            "sharpe_ratio": float(metrics.get("sharpe_ratio", 0.0)),
            "win_rate": float(metrics.get("win_rate", 0.0)),
            "profit_factor": float(metrics.get("profit_factor", 0.0)),
            "n_trades": int(metrics.get("n_trades", 0)),
            "final_equity": float(metrics.get("final_equity", 0.0)),
        }
        history.append(summary_row)
        # Enforce history cap and persist summaries.
        if len(history) > MAX_HISTORY_ROWS:
            history = history[-MAX_HISTORY_ROWS:]
        bt_state["history"] = history
        _save_json_history("backtests_history.json", history)

    # Always render the last backtest result (if any), regardless of button state.
    last_run = bt_state.get("last_run")
    if last_run is not None:
        equity_df = last_run.get("equity_df", pd.DataFrame())
        metrics = last_run.get("metrics", {})

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

            # Use matplotlib so we can show equity and NVDA price on separate axes.
            x = (
                pd.to_datetime(equity_to_plot[time_col])
                if time_col == "Time" and "Time" in equity_to_plot.columns
                else equity_to_plot.index
            )

            fig, ax_equity = plt.subplots(figsize=(12, 6))
            ax_price = ax_equity.twinx()

            # Transparent background for embedding in dark/light themes.
            fig.patch.set_alpha(0.0)
            ax_equity.set_facecolor("none")
            ax_price.set_facecolor("none")

            # Equity on primary y-axis.
            ax_equity.plot(x, equity_to_plot["equity"], color="tab:orange", label="Equity")
            ax_equity.set_ylabel("Equity", color="tab:orange")
            ax_equity.tick_params(axis="y", labelcolor="tab:orange")

            # NVDA price on secondary y-axis (if available).
            if "price" in equity_to_plot.columns:
                ax_price.plot(x, equity_to_plot["price"], color="tab:blue", alpha=0.7, label="NVDA price")
                ax_price.set_ylabel("NVDA price", color="tab:blue")
                ax_price.tick_params(axis="y", labelcolor="tab:blue")

            ax_equity.set_xlabel("Time" if time_col == "Time" else "Bar index")
            ax_equity.set_title("Equity vs NVDA price")

            # Combined legend from both axes.
            lines_e, labels_e = ax_equity.get_legend_handles_labels()
            lines_p, labels_p = ax_price.get_legend_handles_labels()
            if lines_e or lines_p:
                ax_equity.legend(lines_e + lines_p, labels_e + labels_p, loc="upper left")

            fig.tight_layout()
            st.pyplot(fig)
        else:
            st.write("No equity data to display.")

        st.subheader("Metrics")
        # Present key metrics in a human-friendly table (percentages, rounding, separators).
        formatted_metrics = [
            ("Total return", f"{metrics.get('total_return', 0.0) * 100:.0f}%"),
            ("CAGR", f"{metrics.get('cagr', 0.0) * 100:.0f}%"),
            ("Max drawdown", f"{metrics.get('max_drawdown', 0.0) * 100:.0f}%"),
            ("Sharpe ratio", f"{metrics.get('sharpe_ratio', 0.0):.2f}"),
            ("Win rate", f"{metrics.get('win_rate', 0.0) * 100:.0f}%"),
            ("Profit factor", f"{metrics.get('profit_factor', 0.0):.2f}"),
            ("Period", metrics.get('period', '')),
            ("Number of trades", f"{int(metrics.get('n_trades', 0))}"),
            ("Final equity", f"{metrics.get('final_equity', 0.0):,.0f}"),
        ]
        metrics_df = pd.DataFrame(formatted_metrics, columns=["Metric", "Value"])
        st.table(metrics_df)
    else:
        st.info("Run a backtest to see the equity curve and metrics.")

    # Backtest summary history table.
    history_for_view = bt_state.get("history", [])
    if history_for_view:
        st.markdown("### Backtest history (most recent first)")
        hist_df = pd.DataFrame(history_for_view)
        # Sort by timestamp descending when present.
        if "timestamp" in hist_df.columns:
            hist_df = hist_df.sort_values("timestamp", ascending=False)
        st.dataframe(hist_df, width="stretch")

    # ----------------------------------------------------------------------------------
    # Optimization mode UI
    # ----------------------------------------------------------------------------------
    if mode == "Optimize":
        ui_state = _get_ui_state()
        opt_state = ui_state.setdefault("optimization", {})
        if "history" not in opt_state:
            opt_state["history"] = _load_json_history("optimization_history.json")

        # Separate the action of running optimization from displaying results so
        # that results/plots persist even if the parameter table is edited.
        run_optimization = st.button("Run optimization")
        results_df = None

        if run_optimization:
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
                st.error(
                    f"Grid is too large ({total_runs} runs). Reduce ranges or steps (limit = {max_runs}).",
                )
            else:
                progress = st.progress(0.0)
                results_rows: list[dict] = []
                names = list(param_ranges.keys())
                all_values = [param_ranges[n] for n in names]

                run_idx = 0
                for combo in product(*all_values):
                    combo_params = dict(zip(names, combo))

                    # Extract per-combination parameter values, falling back to
                    # the current grid "Value" when a given knob is not
                    # optimized.
                    k_sigma_long_combo = float(combo_params.get("k_sigma_long", k_sigma_long_val))
                    k_sigma_short_combo = float(combo_params.get("k_sigma_short", k_sigma_short_val))
                    k_atr_long_combo = float(combo_params.get("k_atr_long", k_atr_long_val))
                    k_atr_short_combo = float(combo_params.get("k_atr_short", k_atr_short_val))
                    risk_pct = float(combo_params.get("risk_per_trade_pct", risk_pct_val))
                    rr = float(combo_params.get("reward_risk_ratio", rr_val))


                    equity_df, trades_df, metrics = _run_backtest(
                        frequency=freq,
                        start_date=start_date or None,
                        end_date=end_date or None,
                        risk_per_trade_pct=risk_pct,
                        reward_risk_ratio=rr,
                        k_sigma_long=k_sigma_long_combo,
                        k_sigma_short=k_sigma_short_combo,
                        k_atr_long=k_atr_long_combo,
                        k_atr_short=k_atr_short_combo,
                        enable_longs=enable_longs_flag,
                        allow_shorts=allow_shorts_flag,
                    )

                    results_rows.append(
                        {
                            # Side-specific parameters so we can distinguish long vs short.
                            "k_sigma_long": k_sigma_long_combo,
                            "k_sigma_short": k_sigma_short_combo,
                            "k_atr_long": k_atr_long_combo,
                            "k_atr_short": k_atr_short_combo,
                            # Risk/return configuration.
                            "risk_per_trade_pct": risk_pct,
                            "reward_risk_ratio": rr,
                            # Performance metrics.
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
                    # Persist results in-session and record a compact summary to history.
                    opt_state["last_run"] = {
                        "frequency": freq,
                        "start_date": start_date or None,
                        "end_date": end_date or None,
                        "trade_side": trade_side,
                        "results_df": results_df,
                    }
                    st.session_state["optimization_results"] = results_df

                    hist: list[dict] = opt_state.get("history", [])
                    best_sharpe = float(results_df["sharpe_ratio"].max()) if "sharpe_ratio" in results_df.columns else 0.0
                    best_total_return = float(results_df["total_return"].max()) if "total_return" in results_df.columns else 0.0
                    hist.append(
                        {
                            "timestamp": pd.Timestamp.utcnow().isoformat(),
                            "frequency": freq,
                            "trade_side": trade_side,
                            "n_runs": int(len(results_df)),
                            "best_sharpe": best_sharpe,
                            "best_total_return": best_total_return,
                        }
                    )
                    if len(hist) > MAX_HISTORY_ROWS:
                        hist = hist[-MAX_HISTORY_ROWS:]
                    opt_state["history"] = hist
                    _save_json_history("optimization_history.json", hist)
        else:
            # Reuse the most recent optimization results, if any.
            last = opt_state.get("last_run")
            if last is not None and "results_df" in last:
                results_df = last["results_df"]
            else:
                results_df = st.session_state.get("optimization_results")

        # If we have results (either from a new run or from session_state),
        # display the table, allow loading a row into the parameter grid, and
        # render the heatmaps.
        if results_df is not None and not results_df.empty:
            st.subheader("Optimization results")
            display_df = results_df.reset_index(drop=True)
            st.dataframe(display_df, width="stretch")

            # Simple selector to copy a result row's parameters into the
            # strategy parameter grid (Value column).
            if len(display_df) > 1:
                idx_to_use = st.number_input(
                    "Select result row index to load into strategy parameters",
                    min_value=0,
                    max_value=len(display_df) - 1,
                    step=1,
                    value=0,
                    key="opt_result_row_select",
                )
            else:
                idx_to_use = 0

            if st.button("Load selected result into strategy parameters"):
                chosen = display_df.iloc[int(idx_to_use)]

                # Map optimization result columns back into the strategy grid's
                # parameter names (which use explicit long/short knobs).
                mapping = {
                    "k_sigma_long": float(chosen["k_sigma_long"]),
                    "k_sigma_short": float(chosen["k_sigma_short"]),
                    "k_atr_long": float(chosen["k_atr_long"]),
                    "k_atr_short": float(chosen["k_atr_short"]),
                    "risk_per_trade_pct": float(chosen["risk_per_trade_pct"]),
                    "reward_risk_ratio": float(chosen["reward_risk_ratio"]),
                }

                # Update the Value column of the strategy parameter grid.
                params_df_updated = params_df.copy()
                for name, value in mapping.items():
                    mask = params_df_updated["Parameter"] == name
                    if mask.any():
                        params_df_updated.loc[mask, "Value"] = value

                _save_params_grid(params_df_updated)
                st.success(
                    "Loaded optimization parameters into strategy grid. "
                    "You can now run an individual backtest with these values.",
                )

            # Export top-N Sharpe parameter sets to Walk-Forward robustness tab.
            st.markdown("#### Export parameter sets to Walk-Forward robustness")
            max_top_n = max(1, min(10, len(display_df)))
            top_n_to_export = st.number_input(
                "Top N by Sharpe to export",
                min_value=1,
                max_value=max_top_n,
                value=min(3, max_top_n),
                step=1,
                key="opt_to_wf_top_n",
            )
            if st.button("Send top N by Sharpe to Walk-Forward tab"):
                # Sort by Sharpe descending and take the top N rows.
                best_for_export = display_df.sort_values(
                    by="sharpe_ratio", ascending=False
                ).head(int(top_n_to_export))

                wf_rows: list[dict] = []
                for rank, (_, row) in enumerate(best_for_export.iterrows(), start=1):
                    wf_rows.append(
                        {
                            "label": f"opt_{rank}",
                            "k_sigma_long": float(row["k_sigma_long"]),
                            "k_sigma_short": float(row["k_sigma_short"]),
                            "k_atr_long": float(row["k_atr_long"]),
                            "k_atr_short": float(row["k_atr_short"]),
                            "risk_per_trade_pct": float(row["risk_per_trade_pct"]),
                            "reward_risk_ratio": float(row["reward_risk_ratio"]),
                            "enabled": True,
                        }
                    )

                if wf_rows:
                    wf_param_df = pd.DataFrame(wf_rows)
                    # Seed for the Walk-Forward tab. We use a separate key
                    # (wf_param_grid_seed) so we never write directly to the
                    # widget's own key (wf_param_grid_editor), which Streamlit
                    # manages.
                    st.session_state["wf_param_grid_seed"] = wf_param_df
                    st.success(
                        f"Exported {len(wf_rows)} parameter sets to Walk-Forward robustness tab. "
                        "Switch to 'Walk-Forward Analysis' → 'Robustness by parameter set' to run Sharpe-based tests.",
                    )

            # ------------------------------------------------------------------
            # Heatmaps: k_sigma (x) vs k_atr (y) for key metrics, using the
            # active side (long or short) based on the trade_side selection.
            # ------------------------------------------------------------------
            st.subheader("Heatmaps (k_sigma vs k_atr for active side)")

            if enable_longs_flag and not allow_shorts_flag:
                sigma_col = "k_sigma_long"
                atr_col = "k_atr_long"
            elif allow_shorts_flag and not enable_longs_flag:
                sigma_col = "k_sigma_short"
                atr_col = "k_atr_short"
            else:
                # Long & short  visualize long-side filters by convention.
                sigma_col = "k_sigma_long"
                atr_col = "k_atr_long"

            agg_cols = [
                sigma_col,
                atr_col,
                "total_return",
                "max_drawdown",
                "sharpe_ratio",
            ]
            slice_df = (
                results_df[agg_cols]
                .groupby([sigma_col, atr_col], as_index=False)
                .max()
            )

            if not slice_df.empty:
                sharpe_grid = slice_df.pivot(
                    index=atr_col,
                    columns=sigma_col,
                    values="sharpe_ratio",
                ).sort_index().sort_index(axis=1)

                ret_grid = slice_df.pivot(
                    index=atr_col,
                    columns=sigma_col,
                    values="total_return",
                ).sort_index().sort_index(axis=1)

                mdd_grid = slice_df.pivot(
                    index=atr_col,
                    columns=sigma_col,
                    values="max_drawdown",
                ).sort_index().sort_index(axis=1)

                # Larger figure so the three heatmaps use most of the screen.
                fig, axes = plt.subplots(1, 3, figsize=(20, 6))

                grids = [
                    (sharpe_grid, axes[0], "Sharpe Ratio", "viridis"),
                    (ret_grid, axes[1], "Total Return", "YlOrRd"),
                    (mdd_grid, axes[2], "Max Drawdown", "RdYlGn_r"),
                ]

                for grid, ax, title, cmap in grids:
                    im = ax.imshow(grid.values, aspect="auto", cmap=cmap)
                    ax.set_title(title)
                    ax.set_xlabel(sigma_col)
                    # Only show y-label on the first plot to reduce clutter.
                    ax.set_ylabel(atr_col if ax is axes[0] else "")

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

        # Optimization summary history table.
        opt_history = opt_state.get("history", [])
        if opt_history:
            st.markdown("### Optimization history (most recent first)")
            opt_hist_df = pd.DataFrame(opt_history)
            if "timestamp" in opt_hist_df.columns:
                opt_hist_df = opt_hist_df.sort_values("timestamp", ascending=False)
            st.dataframe(opt_hist_df, width="stretch")



# --------------------------------------------------------------------------------------
# Tab 5: Walk-Forward Analysis - train per fold OR robustness by parameter set
# --------------------------------------------------------------------------------------
with tab_walkforward:
    ui_state = _get_ui_state()
    wf_state = ui_state.setdefault("walkforward", {})

    from src import config as _cfg
    from src.walkforward import (
        generate_walkforward_windows as _generate_walkforward_windows,
        slice_df_by_window as _slice_df_by_window,
        infer_data_horizon as _infer_data_horizon,
    )
    from src.data import load_hourly_ohlc as _load_hourly_ohlc
    from src.backtest import run_backtest_on_dataframe as _run_bt_df, _compute_backtest_metrics as _bt_metrics  # type: ignore[attr-defined]
    import numpy as _np
    import os as _os

    st.subheader("5. Walk-forward analysis (robustness by parameter set)")

    _global_freq = st.session_state.get("global_frequency", FREQUENCY)
    if _global_freq not in RESAMPLE_FREQUENCIES:
        _global_freq = RESAMPLE_FREQUENCIES[0]
    wf_freq = st.selectbox(
        "Frequency",
        RESAMPLE_FREQUENCIES,
        index=RESAMPLE_FREQUENCIES.index(_global_freq),
        key="wf_freq_select",
    )
    st.session_state["global_frequency"] = wf_freq

    # Load data once to infer available horizon.
    try:
        df_full = _load_hourly_ohlc(wf_freq)
    except Exception as exc:  # pragma: no cover - UI convenience only
        st.error(f"Failed to load hourly OHLC for {wf_freq}: {exc}")
        df_full = pd.DataFrame()

    if not df_full.empty and "Time" in df_full.columns:
        try:
            data_t_start, data_t_end = _infer_data_horizon(df_full)
        except Exception:
            data_t_start = data_t_end = ""
    else:
        data_t_start = data_t_end = ""

    col_wf1, col_wf2 = st.columns(2)
    with col_wf1:
        wf_t_start = st.text_input(
            "Overall start date (YYYY-MM-DD, optional)",
            value=data_t_start or "",
            key="wf_t_start",
        )
    with col_wf2:
        wf_t_end = st.text_input(
            "Overall end date (YYYY-MM-DD, optional)",
            value=data_t_end or "",
            key="wf_t_end",
        )

    st.markdown("### Window configuration")
    c_w1, c_w2, c_w3 = st.columns(3)
    with c_w1:
        wf_test_span_months = st.number_input(
            "Test span (months)",
            min_value=1,
            max_value=24,
            value=3,
            step=1,
            key="wf_test_span_months",
        )
    with c_w2:
        wf_train_lookback_months = st.number_input(
            "Train lookback (months)",
            min_value=1,
            max_value=60,
            value=24,
            step=1,
            key="wf_train_lookback_months",
        )
    with c_w3:
        wf_min_lookback_months = st.number_input(
            "Min first-lookback (months)",
            min_value=1,
            max_value=60,
            value=18,
            step=1,
            key="wf_min_lookback_months",
        )

    wf_first_test_start = st.text_input(
        "First test start date (YYYY-MM-DD, optional)",
        value="",
        key="wf_first_test_start",
    )

    # Preview generated windows (shared by both modes).
    if st.button("Preview walk-forward windows"):
        if df_full.empty or not data_t_start or not data_t_end:
            st.error("Cannot generate windows: OHLC data is missing or invalid.")
        else:
            eff_start = wf_t_start or data_t_start
            eff_end = wf_t_end or data_t_end

            try:
                windows = _generate_walkforward_windows(
                    eff_start,
                    eff_end,
                    test_span_months=int(wf_test_span_months),
                    train_lookback_months=int(wf_train_lookback_months),
                    min_lookback_months=int(wf_min_lookback_months),
                    first_test_start=wf_first_test_start or None,
                )
            except Exception as exc:  # pragma: no cover - UI convenience only
                windows = []
                st.error(f"Failed to generate windows: {exc}")

            if not windows:
                st.info("No walk-forward windows generated for the supplied horizon.")
            else:
                rows = []
                for i, (tr_w, te_w) in enumerate(windows, start=1):
                    rows.append(
                        {
                            "fold_idx": i,
                            "train_start": tr_w.start,
                            "train_end": tr_w.end,
                            "test_start": te_w.start,
                            "test_end": te_w.end,
                        }
                    )
                st.dataframe(pd.DataFrame(rows), width="stretch")

    # ----------------------------------------------------------------------------------
    # Robustness by parameter set (Sharpe across folds)
    # ----------------------------------------------------------------------------------
    st.markdown("### Strategy parameter sets (k_sigma, k_atr, risk, RR)")

    defaults = _load_strategy_defaults()
    base_rows = [
        {
            "label": "config_defaults",
            "k_sigma_long": defaults["k_sigma_long"],
            "k_sigma_short": defaults["k_sigma_short"],
            "k_atr_long": defaults["k_atr_long"],
            "k_atr_short": defaults["k_atr_short"],
            "risk_per_trade_pct": defaults["risk_per_trade_pct"],
            "reward_risk_ratio": defaults["reward_risk_ratio"],
            "enabled": True,
        },
    ]

    # If the Backtest tab has just exported a new set of parameter rows, prefer
    # that seed exactly once (we pop it so it doesn't keep overwriting manual
    # edits). When applying a fresh seed, we must also clear the existing widget
    # state for "wf_param_grid_editor" so that Streamlit reinitializes the
    # data_editor with the new DataFrame instead of reusing the old value.
    seed = st.session_state.pop("wf_param_grid_seed", None)
    if seed is not None:
        try:
            param_df_initial = seed if isinstance(seed, pd.DataFrame) else pd.DataFrame(seed)
        except Exception:
            param_df_initial = pd.DataFrame(base_rows)

        # Reset the widget state so that the new seed takes effect on this run.
        if "wf_param_grid_editor" in st.session_state:
            del st.session_state["wf_param_grid_editor"]
    else:
        stored = st.session_state.get("wf_param_grid_editor", None)
        if isinstance(stored, pd.DataFrame):
            param_df_initial = stored
        elif stored is not None:
            try:
                param_df_initial = pd.DataFrame(stored)
            except Exception:
                param_df_initial = pd.DataFrame(base_rows)
        else:
            param_df_initial = pd.DataFrame(base_rows)

    # Use a dedicated widget key ("wf_param_grid_editor") and do not write to
    # this key elsewhere in the script after the widget is created. Streamlit
    # manages the widget's value in session_state for us and avoids revert loops.
    param_df = st.data_editor(
        param_df_initial,
        num_rows="dynamic",
        key="wf_param_grid_editor",
        width="stretch",
        hide_index=True,
    )

    wf_symbol = st.text_input(
        "Symbol label for predictions CSV & summary filename",
        value="nvda",
        key="wf_symbol",
    ).strip()

    st.caption(
        "Robustness mode reuses fixed model predictions (CSV) and varies only the strategy parameters per fold."
    )

    run_robust = st.button("Run robustness evaluation (Sharpe across folds)")
    results_df: pd.DataFrame | None = None

    if run_robust:
        if df_full.empty or not data_t_start or not data_t_end:
            st.error("Cannot run walk-forward: OHLC data is missing or invalid for this frequency.")
        else:
            eff_start = wf_t_start or data_t_start
            eff_end = wf_t_end or data_t_end

            try:
                windows = _generate_walkforward_windows(
                    eff_start,
                    eff_end,
                    test_span_months=int(wf_test_span_months),
                    train_lookback_months=int(wf_train_lookback_months),
                    min_lookback_months=int(wf_min_lookback_months),
                    first_test_start=wf_first_test_start or None,
                )
            except Exception as exc:  # pragma: no cover - UI convenience only
                windows = []
                st.error(f"Failed to generate windows: {exc}")

            if not windows:
                st.error("No walk-forward windows generated for the supplied horizon.")
            else:
                # Determine which parameter rows are enabled.
                if "enabled" in param_df.columns:
                    enabled_df = param_df[param_df["enabled"].astype(bool)].copy()
                else:
                    enabled_df = param_df.copy()

                if enabled_df.empty:
                    st.error("No parameter sets enabled. Enable at least one row in the table above.")
                else:
                    predictions_csv = _cfg.get_predictions_csv_path(wf_symbol.lower(), wf_freq)
                    if not _os.path.exists(predictions_csv):
                        st.error(
                            f"Predictions CSV not found at {predictions_csv}. "
                            "Generate it first (e.g. via 'Generate predictions CSV' button in the Backtest tab).",
                        )
                    else:
                        progress = st.progress(0.0)
                        status = st.empty()
                        rows: list[dict] = []

                        total_runs = len(enabled_df) * len(windows)
                        run_idx = 0

                        for _, p_row in enabled_df.iterrows():
                            label = str(p_row.get("label", "unnamed"))
                            k_sigma_long = float(p_row["k_sigma_long"])
                            k_sigma_short = float(p_row["k_sigma_short"])
                            k_atr_long = float(p_row["k_atr_long"])
                            k_atr_short = float(p_row["k_atr_short"])
                            risk_pct = float(p_row["risk_per_trade_pct"])
                            rr = float(p_row["reward_risk_ratio"])

                            for fold_idx, (train_w, test_w) in enumerate(windows, start=1):
                                status.write(
                                    f"Param set '{label}': fold {fold_idx}/{len(windows)} "
                                    f"test [{test_w.start}, {test_w.end})"
                                )

                                test_df = _slice_df_by_window(df_full, test_w)
                                if test_df.empty:
                                    rows.append(
                                        {
                                            "param_label": label,
                                            "fold_idx": fold_idx,
                                            "train_start": train_w.start,
                                            "train_end": train_w.end,
                                            "test_start": test_w.start,
                                            "test_end": test_w.end,
                                            "sharpe_ratio": _np.nan,
                                            "total_return": _np.nan,
                                            "cagr": _np.nan,
                                            "max_drawdown": _np.nan,
                                            "win_rate": _np.nan,
                                            "profit_factor": _np.nan,
                                            "n_trades": 0,
                                            "final_equity": _np.nan,
                                        }
                                    )
                                else:
                                    bt_result = _run_bt_df(
                                        data=test_df,
                                        initial_equity=_cfg.INITIAL_EQUITY,
                                        frequency=wf_freq,
                                        prediction_mode="csv",
                                        predictions_csv=predictions_csv,
                                        risk_per_trade_pct=risk_pct,
                                        reward_risk_ratio=rr,
                                        k_sigma_long=k_sigma_long,
                                        k_sigma_short=k_sigma_short,
                                        k_atr_long=k_atr_long,
                                        k_atr_short=k_atr_short,
                                    )
                                    metrics = _bt_metrics(
                                        bt_result,
                                        initial_equity=_cfg.INITIAL_EQUITY,
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

                                run_idx += 1
                                progress.progress(run_idx / max(total_runs, 1))

                        progress.progress(1.0)
                        status.write("Robustness evaluation complete.")

                        if rows:
                            results_df = pd.DataFrame(rows)
                            wf_state["robust_results_df"] = results_df
                            st.session_state["wf_robust_results"] = results_df

    # Reuse latest robustness results if available and we didn't just run.
    if results_df is None:
        # Prefer ui_state-backed result, fall back to legacy session_state key.
        if "robust_results_df" in wf_state:
            results_df = wf_state["robust_results_df"]
        else:
            results_df = st.session_state.get("wf_robust_results")

    if results_df is not None and not results_df.empty:
        st.subheader("Sharpe-based robustness summary")

        # Aggregate Sharpe statistics per parameter label.
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
        frac_positive = (
            results_df.assign(is_pos=results_df["sharpe_ratio"] > 0)
            .groupby("param_label")["is_pos"]
            .mean()
            .reset_index(name="p_sharpe_gt_0")
        )
        sharpe_stats = sharpe_stats.merge(frac_positive, on="param_label", how="left")
        sharpe_stats["robustness_score"] = sharpe_stats["mean_sharpe"] / sharpe_stats["std_sharpe"].replace(0, _np.nan)

        # Store latest summary in-session and append a compact history row.
        wf_state["summary_df"] = sharpe_stats
        robust_hist: list[dict] = wf_state.get("robust_history", [])
        if not sharpe_stats.empty:
            best_row = sharpe_stats.sort_values("mean_sharpe", ascending=False).iloc[0]
            worst_row = sharpe_stats.sort_values("mean_sharpe", ascending=True).iloc[0]
            robust_hist.append(
                {
                    "timestamp": pd.Timestamp.utcnow().isoformat(),
                    "frequency": wf_freq,
                    "n_param_sets": int(len(sharpe_stats)),
                    "best_label": str(best_row["param_label"]),
                    "best_mean_sharpe": float(best_row["mean_sharpe"]),
                    "worst_label": str(worst_row["param_label"]),
                    "worst_mean_sharpe": float(worst_row["mean_sharpe"]),
                }
            )
            if len(robust_hist) > MAX_HISTORY_ROWS:
                robust_hist = robust_hist[-MAX_HISTORY_ROWS:]
            wf_state["robust_history"] = robust_hist
            _save_json_history("wf_robust_history.json", robust_hist)

        st.dataframe(sharpe_stats, width="stretch")

        # Line chart: Sharpe per fold per parameter set.
        st.markdown("#### Sharpe ratio per fold")
        available_labels = sorted(results_df["param_label"].dropna().unique().tolist())
        selected_labels = st.multiselect(
            "Parameter sets to plot",
            available_labels,
            default=available_labels,
            key="wf_sharpe_line_labels",
        )

        plot_df = results_df[results_df["param_label"].isin(selected_labels)].copy()
        if not plot_df.empty:
            pivot = (
                plot_df.pivot(index="fold_idx", columns="param_label", values="sharpe_ratio")
                .sort_index()
            )
            fig, ax = plt.subplots(figsize=(10, 4))
            for label in pivot.columns:
                ax.plot(pivot.index, pivot[label], marker="o", label=label)
            ax.set_xlabel("Fold index")
            ax.set_ylabel("Sharpe ratio")
            ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
            ax.set_title("Sharpe per fold by parameter set")
            ax.legend(loc="best")
            fig.tight_layout()
            st.pyplot(fig)

        # Scatter: mean vs std of Sharpe.
        st.markdown("#### Mean vs. std of Sharpe per parameter set")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.scatter(sharpe_stats["std_sharpe"], sharpe_stats["mean_sharpe"], alpha=0.8)
        for _, row in sharpe_stats.iterrows():
            ax2.text(row["std_sharpe"], row["mean_sharpe"], row["param_label"], fontsize=8)
        ax2.set_xlabel("Sharpe std across folds (lower is better)")
        ax2.set_ylabel("Sharpe mean across folds (higher is better)")
        ax2.set_title("Sharpe robustness: mean vs std")
        fig2.tight_layout()
        st.pyplot(fig2)
