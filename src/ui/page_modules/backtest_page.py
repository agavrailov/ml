from __future__ import annotations

# Import UI components for modern styling
from src.ui import components


def render_backtest_tab(
    *,
    st,
    pd,
    plt,
    os,
    cfg_mod,
    MAX_HISTORY_ROWS: int,
    get_predictions_csv_path,
    generate_predictions_for_csv,
    load_strategy_defaults,
    load_params_grid,
    save_params_grid,
    save_strategy_defaults_to_config,
    run_backtest,
    filter_backtest_history,
    filter_optimization_history,
    format_timestamp,
    get_ui_state,
    load_json_history,
    save_json_history,
    start_optimization,
    stop_optimization,
) -> None:
    ui_state = get_ui_state()
    bt_state = ui_state.setdefault("backtests", {})

    # Lazily load backtest history from disk once per session.
    if "history" not in bt_state:
        bt_state["history"] = load_json_history("backtests_history.json")

    # Allow selecting any configured resample frequency; default to the main config FREQUENCY.
    _available_freqs = getattr(cfg_mod, "RESAMPLE_FREQUENCIES", ["15min"])
    _default_freq = st.session_state.get(
        "global_frequency", getattr(cfg_mod, "FREQUENCY", _available_freqs[0])
    )
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
    _defaults = load_strategy_defaults()

    # Parameter grid (similar to MT5 Strategy Tester): each row is a parameter
    # with Value / Start / Step / Stop / Optimize. The grid is persisted to a
    # JSON sidecar so manual edits survive app reloads.
    st.subheader("Strategy parameters")

    # Use the persisted (JSON) parameter grid as the base value for the editor.
    #
    # Note: st.data_editor stores its internal edit state in st.session_state under
    # the widget key, which is not the same thing as the edited DataFrame.
    # Rely on the DataFrame returned by st.data_editor instead.
    params_df_initial = load_params_grid(_defaults)

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
    k_sigma_short_val = float(
        _param_values.get("k_sigma_short", _defaults["k_sigma_short"])
    )
    k_atr_long_val = float(_param_values.get("k_atr_long", _defaults["k_atr_long"]))
    k_atr_short_val = float(_param_values.get("k_atr_short", _defaults["k_atr_short"]))
    risk_pct_val = float(
        _param_values.get("risk_per_trade_pct", _defaults["risk_per_trade_pct"])
    )
    rr_val = float(_param_values.get("reward_risk_ratio", _defaults["reward_risk_ratio"]))

    if st.button("Save parameters to config.py"):
        # Basic validation: reward:risk and risk% must be positive. If these are
        # invalid, the strategy will reject all trades.
        if rr_val <= 0.0:
            st.error("Cannot save: reward_risk_ratio must be > 0.")
        elif risk_pct_val <= 0.0:
            st.error("Cannot save: risk_per_trade_pct must be > 0.")
        else:
            # Persist single-run defaults to config.py (used by CLIs and as base
            # defaults) and the full grid (Value/Start/Step/Stop/Optimize) to a
            # JSON sidecar used only by this UI.
            save_strategy_defaults_to_config(
                risk_per_trade_pct=risk_pct_val,
                reward_risk_ratio=rr_val,
                # Persist both long and short filters into config.py.
                k_sigma_long=k_sigma_long_val,
                k_sigma_short=k_sigma_short_val,
                k_atr_long=k_atr_long_val,
                k_atr_short=k_atr_short_val,
            )
            # Persist the *edited DataFrame* returned by st.data_editor.
            # (st.session_state["strategy_params"] may contain internal edit state.)
            save_params_grid(params_df)
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
        # Basic validation: reward:risk and risk% must be positive.
        if rr_val <= 0.0:
            st.error(
                "reward_risk_ratio must be > 0 (otherwise the strategy rejects all trades)."
            )
        elif risk_pct_val <= 0.0:
            st.error("risk_per_trade_pct must be > 0.")
        else:
            try:
                with st.spinner("Running backtest..."):
                    equity_df, trades_df, metrics = run_backtest(
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
                save_json_history("backtests_history.json", history)
            except Exception as exc:  # pragma: no cover - UI path only
                st.error(f"Backtest failed: {exc}")

    # Always render the last backtest result (if any), regardless of button state.
    last_run = bt_state.get("last_run")
    if last_run is not None:
        equity_df = last_run.get("equity_df", pd.DataFrame())
        metrics = last_run.get("metrics", {})

        # Professional metrics display with gradient cards and conditional coloring
        components.render_backtest_metrics(st, pd, metrics)
        
        # Professional equity chart with dual-axis visualization
        components.render_equity_chart(st, plt, pd, equity_df, title="Equity vs NVDA Price")
    else:
        st.info("Run a backtest to see the equity curve and metrics.")

    # Backtest summary history table (filtered by selected frequency).
    history_all = bt_state.get("history", [])
    history_for_view = filter_backtest_history(history_all, frequency=freq)

    # Professional history table with sorting and selection
    components.render_history_table(
        st,
        pd,
        history=history_for_view,
        title=f"Backtest History for {freq}",
        columns=["timestamp", "trade_side", "total_return", "sharpe_ratio", "max_drawdown", "n_trades"],
        on_row_select=None  # Optional: add callback to reload parameters from history
    )

    with st.expander("Show all backtest history (all frequencies)", expanded=False):
        if history_all:
            df_all = pd.DataFrame(history_all)
            if "timestamp" in df_all.columns:
                df_all["timestamp"] = df_all["timestamp"].apply(format_timestamp)
                df_all = df_all.sort_values("timestamp", ascending=False)
            st.dataframe(df_all, width="stretch")
        else:
            st.write("(none)")

    # ----------------------------------------------------------------------------------
    # Optimization mode UI
    # ----------------------------------------------------------------------------------
    if mode == "Optimize":
        ui_state = get_ui_state()
        opt_state = ui_state.setdefault("optimization", {})
        if "history" not in opt_state:
            opt_state["history"] = load_json_history("optimization_history.json")

        # Use out-of-process job system (prevents Streamlit reruns from interrupting).
        from src.core.contracts import OptimizeResult as _OptimizeResult
        from src.jobs import store as _job_store
        from src.jobs.types import JobType as _JobType
        import subprocess as _subprocess
        import sys as _sys
        import uuid as _uuid

        active_job_id = opt_state.get("active_job_id")

        # Poll active job status.
        if active_job_id:
            st.markdown(f"### Active optimization job: `{active_job_id}`")
            st.caption(f"Run dir: `{_job_store.run_dir(active_job_id)}`")
            st.caption(f"Log: `{_job_store.artifacts_dir(active_job_id) / 'run.log'}`")

            st.button("Refresh optimization job status", key="opt_job_refresh")

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
                    stop_optimization()
                    if job_status.traceback:
                        with st.expander("Show traceback", expanded=False):
                            st.code(job_status.traceback)

                if job_status.state == "SUCCEEDED":
                    stop_optimization()
                    res_path = _job_store.result_path(active_job_id)
                    if res_path.exists():
                        try:
                            res_obj = _job_store.read_json(res_path)
                            res = _OptimizeResult(**res_obj)
                        except Exception:
                            res = None

                        if res is not None:
                            st.success(f"Optimization complete: {res.summary.get('n_runs', 0)} runs")

                            # Load results from CSV.
                            results_csv_path = _job_store.artifacts_dir(active_job_id) / "results.csv"
                            if results_csv_path.exists():
                                results_df = pd.read_csv(results_csv_path)

                                # Record to history (once per job).
                                recorded = opt_state.setdefault("recorded_job_ids", [])
                                if active_job_id not in recorded:
                                    hist: list[dict] = opt_state.get("history", [])
                                    hist.append(
                                        {
                                            "timestamp": pd.Timestamp.utcnow().isoformat(),
                                            "frequency": freq,
                                            "trade_side": trade_side,
                                            "n_runs": res.summary.get("n_runs", 0),
                                            "best_sharpe": res.summary.get("best_sharpe", 0.0),
                                            "best_total_return": res.summary.get("best_total_return", 0.0),
                                        }
                                    )
                                    if len(hist) > MAX_HISTORY_ROWS:
                                        hist = hist[-MAX_HISTORY_ROWS:]
                                    opt_state["history"] = hist
                                    save_json_history("optimization_history.json", hist)
                                    recorded.append(active_job_id)

                                # Store results for rendering below.
                                opt_state["last_run"] = {
                                    "frequency": freq,
                                    "start_date": start_date or None,
                                    "end_date": end_date or None,
                                    "trade_side": trade_side,
                                    "results_df": results_df,
                                }

        # Launch new optimization job.
        run_optimization = st.button("Run optimization", on_click=start_optimization)
        if run_optimization:
            try:
                # Build param_grid from params_df.
                param_grid: dict[str, dict[str, float] | float] = {}
                for _, row in params_df.iterrows():
                    name = row["Parameter"]
                    optimize = bool(row.get("Optimize", True))

                    if not optimize:
                        # Fixed value.
                        param_grid[name] = float(row["Value"])
                    else:
                        # Grid search.
                        param_grid[name] = {
                            "start": float(row["Start"]),
                            "stop": float(row["Stop"]),
                            "step": float(row["Step"]),
                        }

                job_id = _uuid.uuid4().hex

                request_obj = {
                    "frequency": freq,
                    "start_date": start_date or None,
                    "end_date": end_date or None,
                    "trade_side": trade_side,
                    "param_grid": param_grid,
                    "prediction_mode": "csv",
                    "predictions_csv": None,  # Will use default.
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
                            _JobType.OPTIMIZE.value,
                            "--request",
                            str(_job_store.request_path(job_id)),
                        ],
                        stdout=log_f,
                        stderr=_subprocess.STDOUT,
                    )

                opt_state["active_job_id"] = job_id
                st.success(f"Started optimization job `{job_id}`. Use 'Refresh optimization job status' to monitor progress.")
            except Exception as exc:  # pragma: no cover
                st.error(f"Error starting optimization job: {exc}")

        # Render results (persists across reruns).
        results_df = None
        last = opt_state.get("last_run")
        if last is not None and "results_df" in last:
            results_df = last["results_df"]

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

                save_params_grid(params_df_updated)
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
            slice_df = results_df[agg_cols].groupby([sigma_col, atr_col], as_index=False).max()

            if not slice_df.empty:
                sharpe_grid = (
                    slice_df.pivot(
                        index=atr_col,
                        columns=sigma_col,
                        values="sharpe_ratio",
                    )
                    .sort_index()
                    .sort_index(axis=1)
                )

                ret_grid = (
                    slice_df.pivot(
                        index=atr_col,
                        columns=sigma_col,
                        values="total_return",
                    )
                    .sort_index()
                    .sort_index(axis=1)
                )

                mdd_grid = (
                    slice_df.pivot(
                        index=atr_col,
                        columns=sigma_col,
                        values="max_drawdown",
                    )
                    .sort_index()
                    .sort_index(axis=1)
                )

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

        # Optimization summary history table (filtered by selected frequency).
        opt_history_all = opt_state.get("history", [])
        opt_history_for_view = filter_optimization_history(opt_history_all, frequency=freq)

        st.markdown(f"### Optimization history for `{freq}` (most recent first)")
        if opt_history_for_view:
            opt_hist_df = pd.DataFrame(opt_history_for_view)
            if "timestamp" in opt_hist_df.columns:
                opt_hist_df["timestamp"] = opt_hist_df["timestamp"].apply(format_timestamp)
                opt_hist_df = opt_hist_df.sort_values("timestamp", ascending=False)
            st.dataframe(opt_hist_df, width="stretch")
        else:
            st.caption("No optimizations recorded for this frequency yet.")

        with st.expander("Show all optimization history (all frequencies)", expanded=False):
            if opt_history_all:
                df_all = pd.DataFrame(opt_history_all)
                if "timestamp" in df_all.columns:
                    df_all["timestamp"] = df_all["timestamp"].apply(format_timestamp)
                    df_all = df_all.sort_values("timestamp", ascending=False)
                st.dataframe(df_all, width="stretch")
            else:
                st.write("(none)")
