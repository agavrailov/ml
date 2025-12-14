from __future__ import annotations


def render_walkforward_tab(
    *,
    st,
    pd,
    plt,
    FREQUENCY: str,
    RESAMPLE_FREQUENCIES: list[str],
    MAX_HISTORY_ROWS: int,
    get_ui_state,
    load_strategy_defaults,
    save_json_history,
) -> None:
    ui_state = get_ui_state()
    wf_state = ui_state.setdefault("walkforward", {})

    from src import config as _cfg
    from src.walkforward import (
        generate_walkforward_windows as _generate_walkforward_windows,
        infer_data_horizon as _infer_data_horizon,
        slice_df_by_window as _slice_df_by_window,
    )
    from src.data import load_hourly_ohlc as _load_hourly_ohlc

    # NOTE: ignore[attr-defined] to keep parity with the original UI code.
    from src.backtest import (
        run_backtest_on_dataframe as _run_bt_df,
        _compute_backtest_metrics as _bt_metrics,  # type: ignore[attr-defined]
    )

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

    defaults = load_strategy_defaults()
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

    # Base grid: manual config/defaults that always exists.
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

    param_df = st.data_editor(
        param_df_initial,
        num_rows="dynamic",
        key="wf_param_grid_editor",
        width="stretch",
        hide_index=True,
    )

    # Optional second grid: parameter sets exported from Optimization tab.
    wf_seed_raw = st.session_state.get("wf_param_grid_seed")
    wf_seed_df: pd.DataFrame | None = None
    if wf_seed_raw is not None:
        try:
            wf_seed_df_initial = (
                wf_seed_raw if isinstance(wf_seed_raw, pd.DataFrame) else pd.DataFrame(wf_seed_raw)
            )
        except Exception:
            wf_seed_df_initial = None

        if wf_seed_df_initial is not None and not wf_seed_df_initial.empty:
            # Ensure an "enabled" column exists, defaulting to True.
            if "enabled" not in wf_seed_df_initial.columns:
                wf_seed_df_initial["enabled"] = True

            st.markdown("#### Exported parameter sets from Optimization tab")
            wf_seed_df = st.data_editor(
                wf_seed_df_initial,
                num_rows="dynamic",
                key="wf_param_grid_seed_editor",
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

    # Use out-of-process job system (prevents Streamlit reruns from interrupting).
    from src.core.contracts import WalkForwardResult as _WalkForwardResult
    from src.jobs import store as _job_store
    from src.jobs.types import JobType as _JobType
    import subprocess as _subprocess
    import sys as _sys
    import uuid as _uuid

    active_job_id = wf_state.get("active_job_id")

    # Poll active job status.
    if active_job_id:
        st.markdown(f"### Active walk-forward job: `{active_job_id}`")
        st.caption(f"Run dir: `{_job_store.run_dir(active_job_id)}`")
        st.caption(f"Log: `{_job_store.artifacts_dir(active_job_id) / 'run.log'}`")

        st.button("Refresh walk-forward job status", key="wf_job_refresh")

        job_status = _job_store.read_status(active_job_id)
        if job_status is None:
            st.info("Job status not written yet.")
        else:
            st.write(
                {
                    "state": job_status.state,
                    "created_at_utc": job_status.created_at_utc,
                    "started_at_utc": job_status.started_at_utc,
                    "finished_at_utc": job_status.finished_at_utc,
                    "error": job_status.error,
                }
            )

            if job_status.state == "FAILED":
                if job_status.traceback:
                    with st.expander("Show traceback", expanded=False):
                        st.code(job_status.traceback)

            if job_status.state == "SUCCEEDED":
                res_path = _job_store.result_path(active_job_id)
                if res_path.exists():
                    try:
                        res_obj = _job_store.read_json(res_path)
                        summary = res_obj.get("summary", {})
                        st.success(
                            f"Walk-forward complete: {summary.get('n_param_sets', 0)} param sets, "
                            f"{summary.get('n_folds', 0)} folds"
                        )

                        # Load results from CSVs.
                        results_csv_path = _job_store.artifacts_dir(active_job_id) / "results.csv"
                        summary_csv_path = _job_store.artifacts_dir(active_job_id) / "summary.csv"

                        if results_csv_path.exists():
                            results_df = pd.read_csv(results_csv_path)
                            wf_state["robust_results_df"] = results_df

                        if summary_csv_path.exists():
                            sharpe_stats = pd.read_csv(summary_csv_path)
                            wf_state["summary_df"] = sharpe_stats

                            # Record to history (once per job).
                            recorded = wf_state.setdefault("recorded_job_ids", [])
                            if active_job_id not in recorded:
                                robust_hist: list[dict] = wf_state.get("robust_history", [])
                                robust_hist.append(
                                    {
                                        "timestamp": pd.Timestamp.utcnow().isoformat(),
                                        "frequency": wf_freq,
                                        "n_param_sets": summary.get("n_param_sets", 0),
                                        "best_label": summary.get("best_label", ""),
                                        "best_mean_sharpe": summary.get("best_mean_sharpe", 0.0),
                                        "worst_label": summary.get("worst_label", ""),
                                        "worst_mean_sharpe": summary.get("worst_mean_sharpe", 0.0),
                                    }
                                )
                                if len(robust_hist) > MAX_HISTORY_ROWS:
                                    robust_hist = robust_hist[-MAX_HISTORY_ROWS:]
                                wf_state["robust_history"] = robust_hist
                                save_json_history("wf_robust_history.json", robust_hist)
                                recorded.append(active_job_id)
                    except Exception as exc:
                        st.error(f"Failed to load walk-forward results: {exc}")

    # Launch new walk-forward job.
    run_robust = st.button("Run robustness evaluation (Sharpe across folds)")
    if run_robust:
        if df_full.empty or not data_t_start or not data_t_end:
            st.error("Cannot run walk-forward: OHLC data is missing or invalid for this frequency.")
        else:
            eff_start = wf_t_start or data_t_start
            eff_end = wf_t_end or data_t_end

            # Determine which parameter rows are enabled across both grids.
            if "enabled" in param_df.columns:
                enabled_df = param_df[param_df["enabled"].astype(bool)].copy()
            else:
                enabled_df = param_df.copy()

            # Append any enabled rows from the exported-seed grid.
            if wf_seed_df is not None and not wf_seed_df.empty:
                if "enabled" in wf_seed_df.columns:
                    extra = wf_seed_df[wf_seed_df["enabled"].astype(bool)].copy()
                else:
                    extra = wf_seed_df.copy()
                if not extra.empty:
                    enabled_df = pd.concat([enabled_df, extra], ignore_index=True)

            if enabled_df.empty:
                st.error(
                    "No parameter sets enabled. Enable at least one row in the tables above."
                )
            else:
                predictions_csv = _cfg.get_predictions_csv_path(wf_symbol.lower(), wf_freq)
                if not _os.path.exists(predictions_csv):
                    st.error(
                        f"Predictions CSV not found at {predictions_csv}. "
                        "Generate it first (e.g. via 'Generate predictions CSV' button in the Backtest tab).",
                    )
                else:
                    try:
                        job_id = _uuid.uuid4().hex

                        # Convert enabled_df to list of parameter set dicts.
                        parameter_sets = enabled_df.to_dict(orient="records")

                        request_obj = {
                            "frequency": wf_freq,
                            "symbol": wf_symbol.lower(),
                            "t_start": eff_start or None,
                            "t_end": eff_end or None,
                            "test_span_months": int(wf_test_span_months),
                            "train_lookback_months": int(wf_train_lookback_months),
                            "min_lookback_months": int(wf_min_lookback_months),
                            "first_test_start": wf_first_test_start or None,
                            "predictions_csv": str(predictions_csv),
                            "parameter_sets": parameter_sets,
                        }

                        _job_store.write_request(job_id, request_obj)

                        log_path = _job_store.artifacts_dir(job_id) / "run.log"
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(log_path, "a", encoding="utf-8") as log_f:
                            _subprocess.Popen(
                                [
                                    _sys.executable,
                                    "-m",
                                    "src.jobs.run",
                                    "--job-id",
                                    job_id,
                                    "--job-type",
                                    _JobType.WALKFORWARD.value,
                                    "--request",
                                    str(_job_store.request_path(job_id)),
                                ],
                                stdout=log_f,
                                stderr=_subprocess.STDOUT,
                            )

                        wf_state["active_job_id"] = job_id
                        st.success(
                            f"Started walk-forward job `{job_id}`. "
                            "Use 'Refresh walk-forward job status' to monitor progress."
                        )
                    except Exception as exc:  # pragma: no cover
                        st.error(f"Error starting walk-forward job: {exc}")

    # Render results (persists across reruns).
    results_df = wf_state.get("robust_results_df")
    sharpe_stats = wf_state.get("summary_df")

    if results_df is not None and not results_df.empty and sharpe_stats is not None and not sharpe_stats.empty:
        st.markdown("---")
        st.subheader("Walk-forward results")
        st.markdown("### Sharpe-based robustness summary")
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
