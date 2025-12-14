from __future__ import annotations


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
                ax_price.plot(
                    x,
                    equity_to_plot["price"],
                    color="tab:blue",
                    alpha=0.7,
                    label="NVDA price",
                )
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
            ("Period", metrics.get("period", "")),
            ("Number of trades", f"{int(metrics.get('n_trades', 0))}"),
            ("Final equity", f"{metrics.get('final_equity', 0.0):,.0f}"),
        ]
        metrics_df = pd.DataFrame(formatted_metrics, columns=["Metric", "Value"])
        st.table(metrics_df)
    else:
        st.info("Run a backtest to see the equity curve and metrics.")

    # Backtest summary history table (filtered by selected frequency).
    history_all = bt_state.get("history", [])
    history_for_view = filter_backtest_history(history_all, frequency=freq)

    st.markdown(f"### Backtest history for `{freq}` (most recent first)")
    if history_for_view:
        hist_df = pd.DataFrame(history_for_view)
        if "timestamp" in hist_df.columns:
            hist_df["timestamp"] = hist_df["timestamp"].apply(format_timestamp)
            hist_df = hist_df.sort_values("timestamp", ascending=False)
        st.dataframe(hist_df, width="stretch")
    else:
        st.caption("No backtests recorded for this frequency yet.")

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

        # Separate the action of running optimization from displaying results so
        # that results/plots persist even if the parameter table is edited.
        #
        # IMPORTANT: We set a session_state flag via on_click so the Live tab's
        # auto-refresh can be disabled before this run starts executing.
        run_optimization = st.button("Run optimization", on_click=start_optimization)
        results_df = None

        if run_optimization:
            try:
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
                        k_sigma_long_combo = float(
                            combo_params.get("k_sigma_long", k_sigma_long_val)
                        )
                        k_sigma_short_combo = float(
                            combo_params.get("k_sigma_short", k_sigma_short_val)
                        )
                        k_atr_long_combo = float(
                            combo_params.get("k_atr_long", k_atr_long_val)
                        )
                        k_atr_short_combo = float(
                            combo_params.get("k_atr_short", k_atr_short_val)
                        )
                        risk_pct = float(combo_params.get("risk_per_trade_pct", risk_pct_val))
                        rr = float(combo_params.get("reward_risk_ratio", rr_val))

                        equity_df, trades_df, metrics = run_backtest(
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
                        best_sharpe = (
                            float(results_df["sharpe_ratio"].max())
                            if "sharpe_ratio" in results_df.columns
                            else 0.0
                        )
                        best_total_return = (
                            float(results_df["total_return"].max())
                            if "total_return" in results_df.columns
                            else 0.0
                        )
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
                        save_json_history("optimization_history.json", hist)
            finally:
                # Defensive: if anything throws during the grid search, ensure we
                # re-enable Live auto-refresh.
                stop_optimization()
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
