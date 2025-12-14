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
        sharpe_stats["robustness_score"] = sharpe_stats["mean_sharpe"] / sharpe_stats[
            "std_sharpe"
        ].replace(0, _np.nan)

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
            save_json_history("wf_robust_history.json", robust_hist)

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
