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


CONFIG_PATH = Path(__file__).with_name("config.py")
PARAMS_STATE_PATH = Path(__file__).with_name("ui_strategy_params.json")


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
        # Shared defaults (still used in some places)
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
    """Non-cached wrapper around run_backtest_for_ui (model mode only)."""
    return run_backtest_for_ui(
        frequency=frequency,
        prediction_mode="model",
        start_date=start_date,
        end_date=end_date,
        predictions_csv=None,
        risk_per_trade_pct=risk_per_trade_pct,
        reward_risk_ratio=reward_risk_ratio,
        k_sigma_long=k_sigma_long,
        k_sigma_short=k_sigma_short,
        k_atr_long=k_atr_long,
        k_atr_short=k_atr_short,
        enable_longs=enable_longs,
        allow_shorts=allow_shorts,
    )


st.title("LSTM Backtesting & Training UI")

# Four main tabs matching the end-to-end workflow.
tab_data, tab_experiments, tab_train, tab_backtest = st.tabs(
    ["Data", "Hyperparameter Experiments", "Train & Promote", "Backtest / Strategy"]
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
    data_freq = st.selectbox(
        "Frequency for resampling & features",
        RESAMPLE_FREQUENCIES,
        index=RESAMPLE_FREQUENCIES.index(CFG_FREQ) if CFG_FREQ in RESAMPLE_FREQUENCIES else 0,
        key="data_freq_select",
    )

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

    # Frequency / TSTEPS for experiments.
    exp_freq = st.selectbox(
        "Frequency",
        RESAMPLE_FREQUENCIES,
        index=RESAMPLE_FREQUENCIES.index(FREQUENCY) if FREQUENCY in RESAMPLE_FREQUENCIES else 0,
        key="exp_freq_select",
    )
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
                st.session_state.setdefault("lstm_experiments", []).append(row)
        except Exception as exc:  # pragma: no cover - UI convenience
            progress.progress(0.0)
            status.write("")
            st.error(f"Experiment failed: {exc}")

    # Experiments table with an action to load a row into the Train tab.
    exp_rows = st.session_state.get("lstm_experiments", [])
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
            st.success("Loaded experiment into Train & Promote tab. Switch to that tab to train fully.")

# --------------------------------------------------------------------------------------
# Tab 3: Train & Promote - full training with auto-promotion when better
# --------------------------------------------------------------------------------------
with tab_train:
    st.subheader("3. Train full model & promote when better")

    # Prefill from last experiment if available, otherwise from current best.
    prefill = st.session_state.get("train_prefill")
    if prefill:
        train_freq = prefill["frequency"]
        train_tsteps = int(prefill["tsteps"])
    else:
        train_freq = FREQUENCY
        train_tsteps = TSTEPS

    train_freq = st.selectbox(
        "Frequency",
        RESAMPLE_FREQUENCIES,
        index=RESAMPLE_FREQUENCIES.index(train_freq) if train_freq in RESAMPLE_FREQUENCIES else 0,
        key="train_freq_select",
    )
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
        except Exception as exc:  # pragma: no cover - UI convenience
            progress.progress(0.0)
            status.write("")
            st.error(f"Error during full training pipeline: {exc}")

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
    # Allow selecting any configured resample frequency; default to the main config FREQUENCY.
    _available_freqs = getattr(cfg_mod, "RESAMPLE_FREQUENCIES", ["15min"])
    _default_freq = getattr(cfg_mod, "FREQUENCY", _available_freqs[0])
    try:
        _default_index = _available_freqs.index(_default_freq)
    except ValueError:
        _default_index = 0
    freq = st.selectbox("Frequency", _available_freqs, index=_default_index)

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
        _save_params_grid(params_df)
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

    elif mode == "Optimize":
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
                        k_sigma_long=k_sigma_long_val,
                        k_sigma_short=k_sigma_short_val,
                        k_atr_long=k_atr_long_val,
                        k_atr_short=k_atr_short_val,
                        enable_longs=enable_longs_flag,
                        allow_shorts=allow_shorts_flag,
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
                    # Persist results so they survive subsequent UI interactions.
                    st.session_state["optimization_results"] = results_df
        else:
            # Reuse the most recent optimization results, if any.
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
                mapping = {
                    "k_sigma_err": float(chosen["k_sigma_err"]),
                    "k_atr_min_tp": float(chosen["k_atr_min_tp"]),
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

            # ------------------------------------------------------------------
            # Heatmaps: k_sigma_err (x) vs k_atr_min_tp (y) for key metrics.
            # We use the most common risk_per_trade_pct and reward_risk_ratio to
            # select a 2D slice through the grid.
            # ------------------------------------------------------------------
            st.subheader("Heatmaps (k_sigma_err vs k_atr_min_tp)")
            # Aggregate over any varying risk_per_trade_pct / reward_risk_ratio so that
            # each (k_sigma_err, k_atr_min_tp) pair maps to a single row that matches
            # the values in the Optimization results table.
            agg_cols = [
                "k_sigma_err",
                "k_atr_min_tp",
                "total_return",
                "max_drawdown",
                "sharpe_ratio",
            ]
            slice_df = (
                results_df[agg_cols]
                .groupby(["k_sigma_err", "k_atr_min_tp"], as_index=False)
                .max()
            )

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
