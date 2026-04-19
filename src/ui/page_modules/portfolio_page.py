"""Portfolio-level backtest UI tab.

Runs multi-symbol backtests with shared capital allocation constraints,
displaying portfolio-level metrics (Sharpe, drawdown, pairwise correlation).
"""
from __future__ import annotations

import os


def render_portfolio_tab(*, st, pd, plt) -> None:
    st.subheader("Portfolio Backtest")

    from src.config import get_hourly_data_csv_path, get_predictions_csv_path
    from src.core.config_resolver import get_configured_symbols, get_strategy_defaults
    from src.portfolio.capital_allocator import AllocationConfig
    from src.portfolio.portfolio_backtest import (
        PortfolioBacktestConfig,
        SymbolBarData,
        align_symbol_data,
        run_portfolio_backtest,
    )
    from src.strategy import StrategyConfig

    configured_symbols = get_configured_symbols()

    # ── Configuration panel ─────────────────────────────────────────
    with st.container(border=True):
        col_eq, col_sizing, col_start, col_end = st.columns([1, 1, 1, 1])

        with col_eq:
            initial_equity = st.number_input(
                "Initial equity ($)", value=50000, min_value=1000, step=5000, key="pf_equity",
            )

        with col_sizing:
            sizing_mode = st.selectbox(
                "Sizing mode", ["equal", "erc"], key="pf_sizing",
            )

        with col_start:
            pf_start = st.text_input("Start (YYYY-MM-DD)", "", key="pf_start", placeholder="Optional")

        with col_end:
            pf_end = st.text_input("End (YYYY-MM-DD)", "", key="pf_end", placeholder="Optional")

    # Symbol checkboxes
    st.markdown("**Symbols to include:**")
    sym_cols = st.columns(min(len(configured_symbols), 6))
    selected_symbols = []
    for i, sym in enumerate(configured_symbols):
        with sym_cols[i % len(sym_cols)]:
            if st.checkbox(sym, value=True, key=f"pf_sym_{sym}"):
                selected_symbols.append(sym)

    if len(selected_symbols) < 2:
        st.info("Select at least 2 symbols to run a portfolio backtest.")
        return

    # Check data availability
    frequency = st.session_state.get("global_frequency", "60min")
    missing_data = []
    missing_preds = []
    for sym in selected_symbols:
        ohlc_path = get_hourly_data_csv_path(frequency, symbol=sym)
        pred_path = get_predictions_csv_path(sym.lower(), frequency)
        if not os.path.exists(ohlc_path):
            missing_data.append(sym)
        if not os.path.exists(pred_path):
            missing_preds.append(sym)

    if missing_data:
        st.warning(f"Missing OHLC data for: {', '.join(missing_data)}. Ingest data first.")
    if missing_preds:
        st.warning(f"Missing predictions CSV for: {', '.join(missing_preds)}. Run backtest in model mode first.")

    can_run = not missing_data and not missing_preds

    # ── Run button ──────────────────────────────────────────────────
    if st.button("Run Portfolio Backtest", disabled=not can_run, type="primary"):
        st.session_state.pop("pf_last_result", None)
        with st.spinner("Running portfolio backtest..."):
            try:
                # Load and align symbol data
                raw_dfs = {}
                for sym in selected_symbols:
                    ohlc_path = get_hourly_data_csv_path(frequency, symbol=sym)
                    raw_dfs[sym] = pd.read_csv(ohlc_path)

                aligned = align_symbol_data(raw_dfs)
                if not aligned:
                    st.error("No overlapping timestamps found across symbols.")
                    return

                # Build per-symbol prediction providers and strategy configs
                symbol_data = {}
                per_symbol_strategy = {}
                for sym in selected_symbols:
                    ohlc_df = aligned[sym]
                    pred_path = get_predictions_csv_path(sym.lower(), frequency)
                    pred_df = pd.read_csv(pred_path)

                    # Align predictions to OHLC by Time
                    if "Time" in pred_df.columns:
                        pred_df["Time"] = pd.to_datetime(pred_df["Time"])
                        pred_df = pred_df.set_index("Time")

                    # Build prediction provider
                    pred_col = "predicted_price" if "predicted_price" in pred_df.columns else "prediction"
                    sigma_col = next(
                        (c for c in pred_df.columns if "sigma" in c.lower() or "residual" in c.lower()),
                        None,
                    )

                    pred_values = pred_df[pred_col].values if pred_col in pred_df.columns else None
                    sigma_values = pred_df[sigma_col].values if sigma_col and sigma_col in pred_df.columns else None

                    def _make_pred_fn(pv=pred_values, ohlc=ohlc_df):
                        def fn(i, row):
                            if pv is not None and i < len(pv):
                                return float(pv[i])
                            return float(row["Close"]) * 1.001  # fallback: near-flat
                        return fn

                    atr_series = None
                    if "ATR_14" in ohlc_df.columns:
                        atr_series = ohlc_df["ATR_14"].reset_index(drop=True)

                    sigma_series = None
                    if sigma_values is not None:
                        sigma_series = pd.Series(sigma_values[:len(ohlc_df)]).reset_index(drop=True)

                    symbol_data[sym] = SymbolBarData(
                        ohlc=ohlc_df.reset_index(drop=True),
                        predictions=_make_pred_fn(),
                        atr_series=atr_series,
                        model_error_sigma_series=sigma_series,
                        fixed_atr=1.0,
                        fixed_model_error_sigma=0.005,
                    )

                    # Load per-symbol strategy
                    defaults = get_strategy_defaults(sym)
                    per_symbol_strategy[sym] = StrategyConfig(
                        risk_per_trade_pct=defaults.get("risk_per_trade_pct", 0.02),
                        reward_risk_ratio=defaults.get("reward_risk_ratio", 2.5),
                        k_sigma_long=defaults.get("k_sigma_long", 0.35),
                        k_sigma_short=defaults.get("k_sigma_short", 0.5),
                        k_atr_long=defaults.get("k_atr_long", 0.45),
                        k_atr_short=defaults.get("k_atr_short", 0.5),
                        enable_longs=defaults.get("enable_longs", True),
                        allow_shorts=defaults.get("allow_shorts", True),
                    )

                # Build config
                alloc_cfg = AllocationConfig(
                    symbols=selected_symbols,
                    max_gross_exposure_pct=0.80,
                    max_per_symbol_pct=0.25,
                    sizing_mode=sizing_mode,
                )

                config = PortfolioBacktestConfig(
                    symbols=selected_symbols,
                    initial_equity=float(initial_equity),
                    allocation_config=alloc_cfg,
                    per_symbol_strategy=per_symbol_strategy,
                )

                result = run_portfolio_backtest(symbol_data, config)
                summary = result.summary()
                corr = result.pairwise_pnl_correlation()

                st.session_state["pf_last_result"] = {
                    "summary": summary,
                    "equity_curve": result.portfolio_equity_curve,
                    "per_symbol_equity": {k: list(v) for k, v in result.per_symbol_equity.items()},
                    "correlation": corr.to_dict() if not corr.empty else None,
                    "selected_symbols": selected_symbols,
                }

            except Exception as exc:
                st.error(f"Portfolio backtest failed: {exc}")
                import traceback
                with st.expander("Error details"):
                    st.code(traceback.format_exc())

    # ── Results (persisted across reruns via session_state) ─────────
    stored = st.session_state.get("pf_last_result")
    if stored is None:
        return

    summary = stored["summary"]
    equity_curve = stored["equity_curve"]
    per_symbol_equity = stored["per_symbol_equity"]
    corr_dict = stored["correlation"]
    result_symbols = stored["selected_symbols"]

    st.markdown("---")
    st.markdown("### Results")

    # Metric cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Portfolio Sharpe", f"{summary['portfolio_sharpe']:.2f}")
    with m2:
        st.metric("Max Drawdown", f"{summary['max_drawdown']*100:.1f}%")
    with m3:
        st.metric("Total Return", f"{summary['total_return']*100:.1f}%")
    with m4:
        total_trades = sum(summary["per_symbol_trades"].values())
        st.metric("Total Trades", str(total_trades))

    # Per-symbol breakdown
    st.markdown("#### Per-Symbol Performance")
    sym_rows = []
    for sym in result_symbols:
        sym_rows.append({
            "Symbol": sym,
            "Trades": summary["per_symbol_trades"].get(sym, 0),
            "Sharpe": f"{summary['per_symbol_sharpe'].get(sym, 0):.2f}",
        })
    st.dataframe(pd.DataFrame(sym_rows), use_container_width=True)

    # Portfolio equity curve
    st.markdown("#### Equity Curve")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(equity_curve, linewidth=2, label="Portfolio")
    for sym in result_symbols:
        eq = per_symbol_equity.get(sym, [])
        if eq:
            ax.plot(eq, alpha=0.5, linewidth=1, label=sym)
    ax.set_xlabel("Bar")
    ax.set_ylabel("Equity ($)")
    ax.set_title("Portfolio Equity Curve")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    # Correlation matrix
    if corr_dict is not None:
        corr = pd.DataFrame(corr_dict)
        st.markdown("#### Pairwise PnL Correlation")
        st.dataframe(
            corr.style.format("{:.3f}").background_gradient(cmap="RdYlGn_r", vmin=-1, vmax=1),
            use_container_width=True,
        )
        avg_corr = corr.values[~(corr.values == 1.0)].mean() if len(corr) > 1 else 0
        st.write(f"Average pairwise correlation: **{avg_corr:.3f}**")
