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
    """Backtest page: Run backtests, optimize parameters, and deploy to live.
    
    User stories:
    0) Select frequency, dates, trade direction
    1) Run backtest → see results (equity curve + KPIs)
    2) Optimize parameters → see active job and results
    3) Load values from production config
    4) Deploy parameters to live trading
    """
    # Inject CSS once
    components.inject_custom_css(st)
    
    ui_state = get_ui_state()
    bt_state = ui_state.setdefault("backtests", {})

    # Lazily load backtest history from disk once per session
    if "history" not in bt_state:
        bt_state["history"] = load_json_history("backtests_history.json")

    # ==========================================================================
    # SECTION 1: CONFIGURATION (Compact)
    # ==========================================================================
    st.markdown("### ⚙ Configuration")

    # Borrow ideas from Streamlit demo-stockpeers: use bordered containers and URL query params for state.
    with st.container(border=True):
        # Row 1: Frequency, dates, trade direction (all in one row)
        col_freq, col_start, col_end, col_trade = st.columns([1, 2, 2, 2])

        # Read query params (if any)
        _qp = st.query_params if hasattr(st, "query_params") else {}

        with col_freq:
            _available_freqs = getattr(cfg_mod, "RESAMPLE_FREQUENCIES", ["15min"])
            _default_freq = st.session_state.get(
                "global_frequency", getattr(cfg_mod, "FREQUENCY", _available_freqs[0])
            )
            # If ?freq= is present and valid, honor it as default
            _qp_freq = _qp.get("freq") if _qp else None
            if _qp_freq in _available_freqs:
                _default_freq = _qp_freq
            try:
                _default_index = _available_freqs.index(_default_freq)
            except ValueError:
                _default_index = 0
            freq = st.selectbox("Frequency", _available_freqs, index=_default_index, key="bt_freq")
            if freq in _available_freqs:
                st.session_state["global_frequency"] = freq
                if hasattr(st, "query_params"):
                    st.query_params["freq"] = freq

        with col_start:
            _qp_start = _qp.get("start", "") if _qp else ""
            start_date = st.text_input("Start (YYYY-MM-DD)", _qp_start, key="bt_start", placeholder="Optional")
            if hasattr(st, "query_params"):
                if start_date:
                    st.query_params["start"] = start_date
                else:
                    st.query_params.pop("start", None)

        with col_end:
            _qp_end = _qp.get("end", "") if _qp else ""
            end_date = st.text_input("End (YYYY-MM-DD)", _qp_end, key="bt_end", placeholder="Optional")
            if hasattr(st, "query_params"):
                if end_date:
                    st.query_params["end"] = end_date
                else:
                    st.query_params.pop("end", None)

        with col_trade:
            trade_options = ["Long & short", "Long only", "Short only"]
            _qp_trade = _qp.get("trade") if _qp else None
            _default_trade = _qp_trade if _qp_trade in trade_options else trade_options[0]
            if hasattr(st, "pills"):
                trade_side = st.pills(
                    "Trade Direction",
                    options=trade_options,
                    default=_default_trade,
                    key="bt_trade_side",
                )
            else:
                trade_side = st.selectbox(
                    "Trade Direction",
                    trade_options,
                    index=trade_options.index(_default_trade),
                    key="bt_trade_side",
                )
            if hasattr(st, "query_params"):
                st.query_params["trade"] = trade_side

    # Parse trade direction
    if trade_side == "Long only":
        enable_longs_flag = True
        allow_shorts_flag = False
    elif trade_side == "Short only":
        enable_longs_flag = False
        allow_shorts_flag = True
    else:  # "Long & short"
        enable_longs_flag = True
        allow_shorts_flag = True

    st.divider()
    
    # ==========================================================================
    # SECTION 2: STRATEGY PARAMETERS
    # ==========================================================================
    st.markdown("### ▶ Strategy Parameters")
    
    # Load current defaults and parameter grid
    _defaults = load_strategy_defaults()
    params_df_initial = load_params_grid(_defaults)

    # Action buttons row: Load from production vs Save to production
    col_load, col_save, col_gen = st.columns([1, 1, 1])
    
    with col_load:
        if st.button("↓ Load from Production Config", width='stretch'):
            # Reload defaults and rebuild grid
            _defaults = load_strategy_defaults()
            params_df_initial = load_params_grid(_defaults)
            # Force update by clearing cached state
            if "strategy_params" in st.session_state:
                del st.session_state["strategy_params"]
            st.success("✓ Loaded parameters from production config")
            st.rerun()
    
    with col_gen:
        predictions_csv_path = get_predictions_csv_path("nvda", freq)
        if st.button("⚡ Generate Predictions CSV", width='stretch'):
            try:
                with st.spinner(f"Generating predictions for {freq}..."):
                    generate_predictions_for_csv(frequency=freq, output_path=predictions_csv_path)
                st.success(f"✓ Generated: {predictions_csv_path.name}")
            except Exception as exc:
                st.error(f"✕ Failed: {exc}")
    
    # Parameter grid editor (reusing component)
    params_df = st.data_editor(
        params_df_initial,
        num_rows="fixed",
        key="strategy_params",
        width='stretch',
        hide_index=True,
        column_config={
            "Parameter": st.column_config.TextColumn("Parameter", disabled=True),
            "Value": st.column_config.NumberColumn("Value", format="%.4f"),
            "Start": st.column_config.NumberColumn("Start", format="%.4f"),
            "Step": st.column_config.NumberColumn("Step", format="%.4f"),
            "Stop": st.column_config.NumberColumn("Stop", format="%.4f"),
            "Optimize": st.column_config.CheckboxColumn("Optimize"),
        },
    )

    # Extract current values
    _param_values = {row["Parameter"]: row["Value"] for _, row in params_df.iterrows()}
    k_sigma_long_val = float(_param_values.get("k_sigma_long", _defaults["k_sigma_long"]))
    k_sigma_short_val = float(_param_values.get("k_sigma_short", _defaults["k_sigma_short"]))
    k_atr_long_val = float(_param_values.get("k_atr_long", _defaults["k_atr_long"]))
    k_atr_short_val = float(_param_values.get("k_atr_short", _defaults["k_atr_short"]))
    risk_pct_val = float(_param_values.get("risk_per_trade_pct", _defaults["risk_per_trade_pct"]))
    rr_val = float(_param_values.get("reward_risk_ratio", _defaults["reward_risk_ratio"]))
    
    with col_save:
        if st.button("↑ Deploy to Production Config", type="primary", width='stretch'):
            # Validation
            if rr_val <= 0.0:
                st.error("✕ reward_risk_ratio must be > 0")
            elif risk_pct_val <= 0.0:
                st.error("✕ risk_per_trade_pct must be > 0")
            else:
                # Save to production config with trading mode
                save_strategy_defaults_to_config(
                    risk_per_trade_pct=risk_pct_val,
                    reward_risk_ratio=rr_val,
                    k_sigma_long=k_sigma_long_val,
                    k_sigma_short=k_sigma_short_val,
                    k_atr_long=k_atr_long_val,
                    k_atr_short=k_atr_short_val,
                    enable_longs=enable_longs_flag,
                    allow_shorts=allow_shorts_flag,
                    symbol="NVDA",
                    frequency=freq,
                    source="ui_manual_deploy",
                )
                save_params_grid(params_df)
                st.success(f"✓ Deployed parameters to production config ({trade_side})")
    
    st.divider()

    # ==========================================================================
    # SECTION 2B: CONFIG LIBRARY (CANDIDATES)
    # ==========================================================================
    from src.core import config_library as _cfg_lib

    st.markdown("### 📦 Config Library (Candidates)")
    symbol = "NVDA"  # TODO: make this selectable when multi-symbol trading is enabled.

    # Display current active config metadata (best-effort).
    active_cfg = _cfg_lib.read_active_config() or {}
    active_meta = active_cfg.get("meta") if isinstance(active_cfg.get("meta"), dict) else {}
    if active_meta:
        st.caption(
            "Active config: "
            f"symbol={active_meta.get('symbol', '—')}, "
            f"frequency={active_meta.get('frequency', '—')}, "
            f"updated_at_utc={active_meta.get('updated_at_utc', active_meta.get('promoted_at_utc', '—'))}"
        )

    # Save current parameter grid values as a candidate.
    with st.container(border=True):
        col_label, col_save_cand, col_refresh = st.columns([2, 1, 1])

        with col_label:
            cand_label = st.text_input(
                "Candidate label (optional)",
                key=f"cfg_cand_label_{freq}",
                placeholder="e.g. 'post-opt backtest v2'",
            )

        # NOTE: Streamlit columns don't support vertical alignment. We add a small spacer
        # so the buttons align with the bottom of the (taller) text_input.
        _btn_spacer = "<div style='height: 1.6rem'></div>"

        with col_save_cand:
            st.markdown(_btn_spacer, unsafe_allow_html=True)
            if st.button("💾 Save as Candidate", width="stretch", key=f"cfg_save_cand_{freq}"):
                # Attach last backtest metrics if they exist for this frequency.
                metrics_for_save = None
                last = bt_state.get("last_run")
                if isinstance(last, dict):
                    inputs = last.get("inputs")
                    if isinstance(inputs, dict) and inputs.get("frequency") == freq:
                        m = last.get("metrics")
                        if isinstance(m, dict):
                            metrics_for_save = m

                row = _cfg_lib.save_candidate(
                    symbol=symbol,
                    frequency=freq,
                    strategy={
                        "risk_per_trade_pct": risk_pct_val,
                        "reward_risk_ratio": rr_val,
                        "k_sigma_long": k_sigma_long_val,
                        "k_sigma_short": k_sigma_short_val,
                        "k_atr_long": k_atr_long_val,
                        "k_atr_short": k_atr_short_val,
                    },
                    label=cand_label,
                    source="ui_backtest",
                    metrics=metrics_for_save,
                )
                st.success(f"✓ Saved candidate: {row.get('id')}")
                st.rerun()

        with col_refresh:
            st.markdown(_btn_spacer, unsafe_allow_html=True)
            if st.button("↻ Refresh", width="stretch", key=f"cfg_refresh_{freq}"):
                st.rerun()

    # List candidates for current (symbol, frequency).
    rows = _cfg_lib.list_candidates(symbol=symbol, frequency=freq, limit=200)
    if not rows:
        st.info("No saved candidates yet for this symbol/frequency.")
    else:
        df_cands = pd.DataFrame(rows)

        # Friendly timestamp formatting.
        if "created_at_utc" in df_cands.columns:
            try:
                df_cands["created_at_utc"] = pd.to_datetime(df_cands["created_at_utc"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass

        df_cands.insert(0, "Select", False)

        df_cands_edited = st.data_editor(
            df_cands,
            key=f"cfg_candidates_table_{freq}",
            hide_index=True,
            width="stretch",
            column_config={
                "Select": st.column_config.CheckboxColumn("Select"),
            },
            disabled=[c for c in df_cands.columns if c != "Select"],
        )

        selected = df_cands_edited[df_cands_edited["Select"] == True]  # noqa: E712

        col_load_cand, col_promote_cand = st.columns([1, 1])

        def _selected_candidate_id() -> str | None:
            if selected.empty:
                return None
            try:
                return str(selected.iloc[0]["id"])
            except Exception:
                return None

        with col_load_cand:
            if st.button("↓ Load Selected Candidate", width="stretch", key=f"cfg_load_cand_{freq}"):
                cid = _selected_candidate_id()
                if not cid:
                    st.error("Select a candidate row first.")
                else:
                    payload = _cfg_lib.load_candidate(cid) or {}
                    strat = payload.get("strategy") if isinstance(payload.get("strategy"), dict) else {}

                    # Update the parameter grid and persist it for the editor.
                    for k, v in strat.items():
                        if k in set(params_df["Parameter"]):
                            params_df.loc[params_df["Parameter"] == k, "Value"] = float(v)
                    save_params_grid(params_df)

                    # Force the grid editor to refresh.
                    if "strategy_params" in st.session_state:
                        del st.session_state["strategy_params"]

                    st.success(f"✓ Loaded candidate into editor: {cid}")
                    st.rerun()

        with col_promote_cand:
            if st.button(
                "↑ Promote Selected to Production",
                type="primary",
                width="stretch",
                key=f"cfg_promote_cand_{freq}",
            ):
                cid = _selected_candidate_id()
                if not cid:
                    st.error("Select a candidate row first.")
                else:
                    _cfg_lib.promote_candidate(cid)
                    st.success(f"✓ Promoted candidate to production: {cid}")
                    st.rerun()

    st.divider()

    # ==========================================================================
    # SECTION 3: BACKTEST & OPTIMIZATION TABS
    # ==========================================================================
    tab_backtest, tab_optimize = st.tabs(["▸ Backtest", "◆ Optimization"])
    
    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1: BACKTEST
    # ─────────────────────────────────────────────────────────────────────────
    with tab_backtest:
        st.markdown("#### Run Backtest")
        
        if st.button("▸ Run Backtest", type="primary", use_container_width=True, key="run_bt_btn"):
            # Validation
            if rr_val <= 0.0:
                st.error("✕ reward_risk_ratio must be > 0")
            elif risk_pct_val <= 0.0:
                st.error("✕ risk_per_trade_pct must be > 0")
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

                    # Store last run
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

                    # Add to history
                    history: list[dict] = bt_state.get("history", [])
                    summary_row = {
                        "timestamp": pd.Timestamp.utcnow().isoformat(),
                        "frequency": freq,
                        "trade_side": trade_side,
                        "k_sigma_long": k_sigma_long_val,
                        "k_sigma_short": k_sigma_short_val,
                        "k_atr_long": k_atr_long_val,
                        "k_atr_short": k_atr_short_val,
                        "risk_per_trade_pct": risk_pct_val,
                        "reward_risk_ratio": rr_val,
                        "sharpe_ratio": float(metrics.get("sharpe_ratio", 0.0)),
                        "cagr": float(metrics.get("cagr", 0.0)),
                        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
                        "n_trades": int(metrics.get("n_trades", 0)),
                        "final_equity": float(metrics.get("final_equity", 0.0)),
                        "total_return": float(metrics.get("total_return", 0.0)),
                        "win_rate": float(metrics.get("win_rate", 0.0)),
                        "profit_factor": float(metrics.get("profit_factor", 0.0)),
                    }
                    history.append(summary_row)
                    if len(history) > MAX_HISTORY_ROWS:
                        history = history[-MAX_HISTORY_ROWS:]
                    bt_state["history"] = history
                    save_json_history("backtests_history.json", history)
                    
                    st.success("✓ Backtest complete!")
                    st.rerun()
                except Exception as exc:
                    st.error(f"✕ Backtest failed: {exc}")
        
        st.divider()
        
        # Backtest results visualization
        st.markdown("#### Latest Results")
        last_run = bt_state.get("last_run")
        if last_run is not None:
            equity_df = last_run.get("equity_df", pd.DataFrame())
            metrics = last_run.get("metrics", {})

            # Two-column layout: chart on left (larger), metrics on right (compact)
            col_chart, col_metrics = st.columns([3, 1])
            
            with col_chart:
                # Use component for equity chart
                components.render_equity_chart(st, plt, pd, equity_df, title="Equity vs NVDA Price")
            
            with col_metrics:
                # Use component for metrics display
                components.render_backtest_metrics(st, pd, metrics)
        else:
            st.info("→ Run a backtest to see results")

        st.divider()
        
        # Backtest history table
        st.markdown("#### Results History")
        history_all = bt_state.get("history", [])
        history_for_view = filter_backtest_history(history_all, frequency=freq)

        def on_backtest_retest(rows: list[dict]) -> None:
            """Load parameters from first selected row and prepare for retest."""
            if not rows:
                return
            
            # Use first selected row
            row = rows[0]
            
            # Update parameter grid with selected values
            for _, param_row in params_df.iterrows():
                param_name = param_row["Parameter"]
                if param_name in row:
                    params_df.loc[params_df["Parameter"] == param_name, "Value"] = row[param_name]
            save_params_grid(params_df)
            st.success(f"✓ Loaded parameters from backtest result. Click 'Run Backtest' to retest.")
            st.rerun()
        
        def on_backtest_add_to_wf(rows: list[dict]) -> None:
            """Add selected backtest parameters to walk-forward table."""
            if not rows:
                return
            
            wf_entries = []
            for idx, row in enumerate(rows):
                wf_entry = {
                    "label": f"bt_{pd.Timestamp.now().strftime('%H%M%S')}_{idx+1}",
                    "k_sigma_long": float(row.get("k_sigma_long", 0)),
                    "k_sigma_short": float(row.get("k_sigma_short", 0)),
                    "k_atr_long": float(row.get("k_atr_long", 0)),
                    "k_atr_short": float(row.get("k_atr_short", 0)),
                    "risk_per_trade_pct": float(row.get("risk_per_trade_pct", 0)),
                    "reward_risk_ratio": float(row.get("reward_risk_ratio", 0)),
                    "enabled": True,
                }
                wf_entries.append(wf_entry)
            
            # Add to walk-forward session state
            if "wf_param_grid_seed" not in st.session_state:
                st.session_state["wf_param_grid_seed"] = pd.DataFrame(wf_entries)
            else:
                existing = st.session_state["wf_param_grid_seed"]
                st.session_state["wf_param_grid_seed"] = pd.concat([existing, pd.DataFrame(wf_entries)], ignore_index=True)
            st.success(f"✓ Added {len(wf_entries)} parameter set(s) to Walk-Forward")

        with st.container(border=True):
            components.render_backtest_results_table(
                st, pd, history_for_view, 
                title=f"Backtest results for {freq}",
                enable_actions=True,
                on_retest=on_backtest_retest,
                on_add_to_wf=on_backtest_add_to_wf,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2: OPTIMIZATION
    # ─────────────────────────────────────────────────────────────────────────
    with tab_optimize:
        st.markdown("#### Run Optimization")
        
        opt_state = ui_state.setdefault("optimization", {})

        # Use out-of-process job system
        from src.core.contracts import OptimizeResult as _OptimizeResult
        from src.jobs import store as _job_store
        from src.jobs.types import JobType as _JobType
        import subprocess as _subprocess
        import sys as _sys
        import uuid as _uuid

        active_job_id = opt_state.get("active_job_id")
        
        # Start optimization button
        if st.button("◆ Run Optimization", type="primary", use_container_width=True, key="run_opt_btn"):
            try:
                # Build param_grid from params_df
                param_grid: dict[str, dict[str, float] | float] = {}
                for _, row in params_df.iterrows():
                    name = row["Parameter"]
                    optimize = bool(row.get("Optimize", True))

                    if not optimize:
                        param_grid[name] = float(row["Value"])
                    else:
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
                    "predictions_csv": None,
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
                start_optimization()
                st.success(f"✓ Started optimization job `{job_id}`")
                st.rerun()
            except Exception as exc:
                st.error(f"✕ Error starting optimization: {exc}")
        
        st.divider()
        
        # Poll active job status
        if active_job_id:
            # Check status immediately when page loads
            job_status = _job_store.read_status(active_job_id)
            
            # Auto-refresh while job is running
            if job_status and job_status.state == "RUNNING":
                try:
                    from streamlit_autorefresh import st_autorefresh
                    # Refresh every 2 seconds while running
                    st_autorefresh(interval=2000, key="opt_autorefresh")
                except ImportError:
                    pass
            
            st.markdown(f"#### ↻ Active Job: `{active_job_id}`")
            st.caption(f"Log: `{_job_store.artifacts_dir(active_job_id) / 'run.log'}`")
            if job_status and job_status.state == "RUNNING":
                st.caption("🔄 Auto-refreshing every 2 seconds...")
            elif job_status and job_status.state == "SUCCEEDED":
                st.caption("✓ Job completed successfully")
            
            if job_status is None:
                st.info("⋯ Job starting...")
            else:
                # Use component for job status display
                if job_status.state == "RUNNING":
                    components.render_status_badge(st, "RUNNING", "Running")
                    st.caption(f"Started: {job_status.started_at_utc}")
                    
                    # Show progress bar if available
                    if job_status.progress is not None:
                        components.render_progress_bar(
                            st,
                            progress=job_status.progress,
                            message=job_status.progress_message,
                        )
                elif job_status.state == "SUCCEEDED":
                    components.render_status_badge(st, "SUCCEEDED", "Complete")
                elif job_status.state == "FAILED":
                    components.render_status_badge(st, "FAILED", "Failed")
                    st.error(f"✕ {job_status.error or 'Unknown error'}")
                    if job_status.traceback:
                        with st.expander("▼ Show Traceback"):
                            st.code(job_status.traceback)
                else:
                    components.render_status_badge(st, "QUEUED", job_status.state)

                if job_status.state in ["SUCCEEDED", "FAILED"]:
                    stop_optimization()

                if job_status.state == "SUCCEEDED":
                    res_path = _job_store.result_path(active_job_id)
                    if res_path.exists():
                        try:
                            res_obj = _job_store.read_json(res_path)
                            res = _OptimizeResult(**res_obj)
                        except Exception:
                            res = None

                        if res is not None:
                            # Use component for optimization summary
                            metrics = [
                                {"label": "Total Runs", "value": str(res.summary.get('n_runs', 0)), "icon": "#", "color": "info"},
                                {"label": "Best Sharpe", "value": f"{res.summary.get('best_sharpe', 0):.2f}", "icon": "↗", "color": "success"},
                                {"label": "Best Return", "value": f"{res.summary.get('best_total_return', 0)*100:.1f}%", "icon": "$", "color": "success"},
                            ]
                            components.render_kpi_row(st, metrics)

                            # Load results from CSV
                            results_csv_path = _job_store.artifacts_dir(active_job_id) / "results.csv"
                            if results_csv_path.exists():
                                results_df = pd.read_csv(results_csv_path)

                                # Store results for rendering below
                                opt_state["last_run"] = {
                                    "frequency": freq,
                                    "start_date": start_date or None,
                                    "end_date": end_date or None,
                                    "trade_side": trade_side,
                                    "results_df": results_df,
                                }

        st.divider()
        
        # Optimization results
        st.markdown("#### Optimization Results")
        results_df = None
        last = opt_state.get("last_run")
        if last is not None and "results_df" in last:
            results_df = last["results_df"]

        if results_df is not None and not results_df.empty:
            display_df = results_df.reset_index(drop=True).sort_values("sharpe_ratio", ascending=False)
            
            def on_opt_retest(rows: list[dict]) -> None:
                """Load parameters from first selected optimization run."""
                if not rows:
                    return
                
                # Use first selected row
                row = rows[0]
                
                # Update parameter grid with selected values
                for _, param_row in params_df.iterrows():
                    param_name = param_row["Parameter"]
                    if param_name in row:
                        params_df.loc[params_df["Parameter"] == param_name, "Value"] = row[param_name]
                save_params_grid(params_df)
                st.success(f"✓ Loaded parameters from optimization result. Switch to Backtest tab to run.")
                st.rerun()
            
            def on_opt_add_to_wf(rows: list[dict]) -> None:
                """Add selected optimization parameters to walk-forward table."""
                if not rows:
                    return
                
                wf_entries = []
                for idx, row in enumerate(rows):
                    wf_entry = {
                        "label": f"opt_{pd.Timestamp.now().strftime('%H%M%S')}_{idx+1}",
                        "k_sigma_long": float(row.get("k_sigma_long", 0)),
                        "k_sigma_short": float(row.get("k_sigma_short", 0)),
                        "k_atr_long": float(row.get("k_atr_long", 0)),
                        "k_atr_short": float(row.get("k_atr_short", 0)),
                        "risk_per_trade_pct": float(row.get("risk_per_trade_pct", 0)),
                        "reward_risk_ratio": float(row.get("reward_risk_ratio", 0)),
                        "enabled": True,
                    }
                    wf_entries.append(wf_entry)
                
                # Add to walk-forward session state
                if "wf_param_grid_seed" not in st.session_state:
                    st.session_state["wf_param_grid_seed"] = pd.DataFrame(wf_entries)
                else:
                    existing = st.session_state["wf_param_grid_seed"]
                    st.session_state["wf_param_grid_seed"] = pd.concat([existing, pd.DataFrame(wf_entries)], ignore_index=True)
                st.success(f"✓ Added {len(wf_entries)} parameter set(s) to Walk-Forward")
            
            # Show top results using reusable component
            components.render_backtest_results_table(
                st, pd, display_df, 
                title="Optimization runs", 
                max_rows=20,
                enable_actions=True,
                on_retest=on_opt_retest,
                on_add_to_wf=on_opt_add_to_wf,
            )

            # Heatmaps
            with st.expander("▼ Heatmaps (k_sigma vs k_atr)", expanded=False):
                if enable_longs_flag and not allow_shorts_flag:
                    sigma_col = "k_sigma_long"
                    atr_col = "k_atr_long"
                elif allow_shorts_flag and not enable_longs_flag:
                    sigma_col = "k_sigma_short"
                    atr_col = "k_atr_short"
                else:
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
                    st.info("No 2D slice available for heatmaps")
        else:
            st.info("→ Run an optimization to see results")
        
