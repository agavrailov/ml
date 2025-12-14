"""Reusable UI components for LSTM Trading System.

This module provides a comprehensive library of polished, reusable components
that understand the business logic of the trading system:
- Models, experiments, and training runs
- Backtests, optimizations, and walk-forward analysis
- Live trading sessions
- Jobs and async operations

Design principles:
- Components are stateless (take data, return rendered UI)
- Consistent visual language across all tabs
- Responsive layouts with proper spacing
- Clear action buttons with confirmation patterns
- Professional metric displays with trends
"""

from __future__ import annotations
from typing import Any, Callable, Literal
import pandas as pd


# =============================================================================
# THEME & STYLING
# =============================================================================

def inject_custom_css(st) -> None:
    """Inject custom CSS for consistent styling across the app."""
    st.markdown("""
    <style>
        /* Main theme variables */
        :root {
            --primary-color: #667eea;
            --primary-dark: #5a67d8;
            --success-color: #48bb78;
            --warning-color: #ed8936;
            --danger-color: #f56565;
            --info-color: #4299e1;
            --neutral-color: #718096;
            
            --card-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            --card-shadow-hover: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }
        
        /* Card containers */
        .metric-card {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
            padding: 1.5rem;
            border-radius: 0.75rem;
            color: white;
            box-shadow: var(--card-shadow);
            transition: transform 0.2s, box-shadow 0.2s;
            margin-bottom: 1rem;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: var(--card-shadow-hover);
        }
        
        .metric-card-success { background: linear-gradient(135deg, #48bb78 0%, #38a169 100%); }
        .metric-card-warning { background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%); }
        .metric-card-danger { background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%); }
        .metric-card-info { background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%); }
        .metric-card-neutral { background: linear-gradient(135deg, #718096 0%, #4a5568 100%); }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            line-height: 1;
            margin-top: 0.5rem;
        }
        .metric-label {
            font-size: 0.875rem;
            opacity: 0.9;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .metric-trend {
            font-size: 0.875rem;
            margin-top: 0.5rem;
        }
        
        /* Status badges */
        .status-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .status-running { background-color: #4299e1; color: white; }
        .status-succeeded { background-color: #48bb78; color: white; }
        .status-failed { background-color: #f56565; color: white; }
        .status-queued { background-color: #718096; color: white; }
        
        /* Action buttons */
        .action-button-group {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin-top: 1rem;
        }
        
        /* Section headers */
        .section-header {
            font-size: 1.5rem;
            font-weight: 600;
            margin-top: 2rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--primary-color);
        }
        
        /* Data tables */
        .data-table-container {
            background: white;
            border-radius: 0.5rem;
            padding: 1rem;
            box-shadow: var(--card-shadow);
            margin-bottom: 1.5rem;
        }
        
        /* Progress indicators */
        .progress-container {
            background: #e2e8f0;
            border-radius: 9999px;
            height: 0.5rem;
            overflow: hidden;
            margin-top: 0.5rem;
        }
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, var(--primary-color), var(--primary-dark));
            transition: width 0.3s ease;
        }
        
        /* Info panels */
        .info-panel {
            background: #f7fafc;
            border-left: 4px solid var(--info-color);
            padding: 1rem;
            border-radius: 0.25rem;
            margin-bottom: 1rem;
        }
        .warning-panel {
            background: #fffaf0;
            border-left: 4px solid var(--warning-color);
            padding: 1rem;
            border-radius: 0.25rem;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# METRIC CARDS
# =============================================================================

def render_metric_card(
    st,
    label: str,
    value: str | float | int,
    trend: str | None = None,
    color: Literal["primary", "success", "warning", "danger", "info", "neutral"] = "primary",
    icon: str | None = None,
) -> None:
    """Render a professional metric card with optional trend indicator.
    
    Args:
        st: Streamlit module
        label: Metric label (e.g., "Sharpe Ratio")
        value: Metric value (will be formatted)
        trend: Optional trend text (e.g., "+12.5% vs last")
        color: Card color scheme
        icon: Optional emoji icon
    """
    color_class = f"metric-card-{color}"
    icon_html = f'<span style="font-size: 2rem; margin-right: 0.5rem;">{icon}</span>' if icon else ""
    trend_html = f'<div class="metric-trend">{trend}</div>' if trend else ""
    
    st.markdown(f"""
    <div class="metric-card {color_class}">
        {icon_html}
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {trend_html}
    </div>
    """, unsafe_allow_html=True)


def render_kpi_row(
    st,
    metrics: list[dict[str, Any]],
) -> None:
    """Render a row of KPI metric cards.
    
    Args:
        st: Streamlit module
        metrics: List of metric dicts with keys: label, value, trend?, color?, icon?
    """
    cols = st.columns(len(metrics))
    for col, metric in zip(cols, metrics):
        with col:
            render_metric_card(
                st,
                label=metric["label"],
                value=metric["value"],
                trend=metric.get("trend"),
                color=metric.get("color", "primary"),
                icon=metric.get("icon"),
            )


# =============================================================================
# STATUS BADGES
# =============================================================================

def render_status_badge(
    st,
    status: Literal["RUNNING", "SUCCEEDED", "FAILED", "QUEUED"],
    text: str | None = None,
) -> None:
    """Render a status badge with color coding.
    
    Args:
        st: Streamlit module
        status: Job/process status
        text: Optional custom text (defaults to status)
    """
    status_lower = status.lower()
    display_text = text or status
    st.markdown(f'<span class="status-badge status-{status_lower}">{display_text}</span>', unsafe_allow_html=True)


# =============================================================================
# MODEL & EXPERIMENT COMPONENTS
# =============================================================================

def render_model_card(
    st,
    model_name: str,
    val_loss: float | None,
    timestamp: str | None,
    frequency: str,
    tsteps: int,
    hyperparams: dict[str, Any],
    actions: list[dict[str, Any]] | None = None,
) -> None:
    """Render a model card with metadata and action buttons.
    
    Args:
        st: Streamlit module
        model_name: Model filename
        val_loss: Validation loss (can be None)
        timestamp: Training timestamp
        frequency: Data frequency
        tsteps: Sequence length
        hyperparams: Dict of hyperparameters
        actions: List of action dicts with keys: label, callback, type (primary/secondary/danger)
    """
    with st.expander(f"🤖 {model_name}", expanded=False):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**Validation Loss:** {val_loss:.6f}" if val_loss is not None else "**Validation Loss:** Unknown")
            st.markdown(f"**Frequency:** {frequency} | **TSteps:** {tsteps}")
            st.caption(f"Trained: {timestamp or 'Unknown'}")
            
            with st.expander("Hyperparameters", expanded=False):
                for key, val in hyperparams.items():
                    st.text(f"{key}: {val}")
        
        with col2:
            if actions:
                for action in actions:
                    button_type = action.get("type", "secondary")
                    if button_type == "primary":
                        btn = st.button(action["label"], key=f"action_{model_name}_{action['label']}", type="primary")
                    else:
                        btn = st.button(action["label"], key=f"action_{model_name}_{action['label']}")
                    
                    if btn and "callback" in action:
                        action["callback"]()


def render_experiment_table(
    st,
    pd,
    experiments: list[dict[str, Any]],
    on_select: Callable[[dict], None] | None = None,
) -> None:
    """Render an interactive experiments table with selection.
    
    Args:
        st: Streamlit module
        pd: Pandas module
        experiments: List of experiment run dicts
        on_select: Callback when a row is selected for use
    """
    if not experiments:
        st.info("No experiments recorded yet. Run your first experiment to see results here.")
        return
    
    df = pd.DataFrame(experiments)
    
    # Format validation loss column
    if "validation_loss" in df.columns:
        df["validation_loss"] = df["validation_loss"].apply(lambda x: f"{float(x):.3e}" if isinstance(x, (int, float, str)) else x)
    
    st.markdown("### 🧪 Experiment History")
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    if on_select and len(experiments) > 0:
        selected_idx = st.number_input(
            "Select experiment to load into training",
            min_value=0,
            max_value=len(experiments) - 1,
            value=len(experiments) - 1,
            key="exp_select_idx"
        )
        
        if st.button("✓ Load Selected Experiment", type="primary"):
            on_select(experiments[int(selected_idx)])


# =============================================================================
# TRAINING & JOB MONITORING
# =============================================================================

def render_job_status(
    st,
    job_id: str,
    job_type: str,
    status: dict[str, Any],
    result: dict[str, Any] | None = None,
    on_refresh: Callable[[], None] | None = None,
) -> None:
    """Render a comprehensive job status panel with progress and results.
    
    Args:
        st: Streamlit module
        job_id: Job identifier
        job_type: Type of job (train, backtest, optimize, etc.)
        status: Job status dict with keys: state, created_at_utc, started_at_utc, finished_at_utc, error
        result: Optional result dict when job succeeds
        on_refresh: Optional refresh callback
    """
    state = status.get("state", "UNKNOWN")
    
    st.markdown(f"### 🔄 Active {job_type.title()} Job")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.code(job_id, language=None)
        render_status_badge(st, state)
    
    with col2:
        if on_refresh:
            st.button("🔄 Refresh", on_click=on_refresh, key=f"refresh_{job_id}")
    
    # Timeline
    col_created, col_started, col_finished = st.columns(3)
    with col_created:
        st.caption("Created")
        st.text(status.get("created_at_utc", "—"))
    with col_started:
        st.caption("Started")
        st.text(status.get("started_at_utc", "—"))
    with col_finished:
        st.caption("Finished")
        st.text(status.get("finished_at_utc", "—"))
    
    # Error handling
    if state == "FAILED":
        st.error(f"❌ Job failed: {status.get('error', 'Unknown error')}")
        if status.get("traceback"):
            with st.expander("Show traceback"):
                st.code(status["traceback"])
    
    # Success results
    elif state == "SUCCEEDED" and result:
        st.success("✅ Job completed successfully")
        
        if job_type == "train":
            render_training_result(st, result)
        elif job_type == "optimize":
            render_optimization_summary(st, result)
        elif job_type == "backtest":
            st.json(result)


def render_training_result(st, result: dict[str, Any]) -> None:
    """Render training job results."""
    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric_card(st, "Validation Loss", f"{result.get('validation_loss', 0):.6f}", color="success")
    with col2:
        st.markdown(f"**Model:** `{result.get('model_filename', 'N/A')}`")
    with col3:
        st.markdown(f"**Bias Correction:** `{result.get('bias_correction_filename', 'N/A')}`")


def render_optimization_summary(st, result: dict[str, Any]) -> None:
    """Render optimization job summary."""
    summary = result.get("summary", {})
    
    metrics = [
        {"label": "Total Runs", "value": summary.get("n_runs", 0), "icon": "🔢", "color": "info"},
        {"label": "Best Sharpe", "value": f"{summary.get('best_sharpe', 0):.2f}", "icon": "📈", "color": "success"},
        {"label": "Best Return", "value": f"{summary.get('best_total_return', 0)*100:.1f}%", "icon": "💰", "color": "success"},
    ]
    
    render_kpi_row(st, metrics)


# =============================================================================
# BACKTEST COMPONENTS
# =============================================================================

def render_backtest_metrics(
    st,
    pd,
    metrics: dict[str, Any],
) -> None:
    """Render backtest metrics in a professional grid layout.
    
    Args:
        st: Streamlit module
        pd: Pandas module
        metrics: Dict of backtest metrics
    """
    st.markdown("### 📊 Performance Metrics")
    
    # Top row: Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_return = metrics.get("total_return", 0.0) * 100
        color = "success" if total_return > 0 else "danger"
        render_metric_card(st, "Total Return", f"{total_return:.1f}%", color=color, icon="💰")
    
    with col2:
        sharpe = metrics.get("sharpe_ratio", 0.0)
        color = "success" if sharpe > 1.0 else "warning" if sharpe > 0 else "danger"
        render_metric_card(st, "Sharpe Ratio", f"{sharpe:.2f}", color=color, icon="📈")
    
    with col3:
        max_dd = metrics.get("max_drawdown", 0.0) * 100
        color = "success" if max_dd > -10 else "warning" if max_dd > -20 else "danger"
        render_metric_card(st, "Max Drawdown", f"{max_dd:.1f}%", color=color, icon="📉")
    
    with col4:
        win_rate = metrics.get("win_rate", 0.0) * 100
        color = "success" if win_rate > 50 else "warning" if win_rate > 40 else "neutral"
        render_metric_card(st, "Win Rate", f"{win_rate:.1f}%", color=color, icon="🎯")
    
    # Second row: Additional metrics
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        cagr = metrics.get("cagr", 0.0) * 100
        render_metric_card(st, "CAGR", f"{cagr:.1f}%", color="info", icon="📊")
    
    with col6:
        profit_factor = metrics.get("profit_factor", 0.0)
        render_metric_card(st, "Profit Factor", f"{profit_factor:.2f}", color="info", icon="⚖️")
    
    with col7:
        n_trades = metrics.get("n_trades", 0)
        render_metric_card(st, "Total Trades", str(n_trades), color="neutral", icon="🔄")
    
    with col8:
        final_equity = metrics.get("final_equity", 0.0)
        render_metric_card(st, "Final Equity", f"${final_equity:,.0f}", color="primary", icon="💵")
    
    # Detailed metrics table
    with st.expander("📋 Detailed Metrics"):
        metrics_display = [
            ("Period", metrics.get("period", "N/A")),
            ("Total Return", f"{metrics.get('total_return', 0)*100:.2f}%"),
            ("CAGR", f"{metrics.get('cagr', 0)*100:.2f}%"),
            ("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.3f}"),
            ("Max Drawdown", f"{metrics.get('max_drawdown', 0)*100:.2f}%"),
            ("Win Rate", f"{metrics.get('win_rate', 0)*100:.2f}%"),
            ("Profit Factor", f"{metrics.get('profit_factor', 0):.3f}"),
            ("Number of Trades", str(metrics.get("n_trades", 0))),
            ("Final Equity", f"${metrics.get('final_equity', 0):,.2f}"),
        ]
        
        df = pd.DataFrame(metrics_display, columns=["Metric", "Value"])
        st.table(df)


def render_equity_chart(
    st,
    plt,
    pd,
    equity_df: pd.DataFrame,
    title: str = "Equity Curve",
) -> None:
    """Render an equity curve chart with price comparison.
    
    Args:
        st: Streamlit module
        plt: Matplotlib.pyplot module
        pd: Pandas module
        equity_df: DataFrame with equity, Time, and optionally price columns
        title: Chart title
    """
    if equity_df.empty:
        st.warning("No equity data to display")
        return
    
    st.markdown(f"### 📈 {title}")
    
    time_col = "Time" if "Time" in equity_df.columns else "step"
    
    # Downsample if too many points
    max_points = 2000
    if len(equity_df) > max_points:
        step = max(1, len(equity_df) // max_points)
        equity_to_plot = equity_df.iloc[::step]
    else:
        equity_to_plot = equity_df
    
    x = (
        pd.to_datetime(equity_to_plot[time_col])
        if time_col == "Time" and "Time" in equity_to_plot.columns
        else equity_to_plot.index
    )
    
    fig, ax_equity = plt.subplots(figsize=(14, 7))
    ax_price = ax_equity.twinx()
    
    # Styling
    fig.patch.set_alpha(0.0)
    ax_equity.set_facecolor("none")
    ax_price.set_facecolor("none")
    
    # Plot equity
    ax_equity.plot(x, equity_to_plot["equity"], color="#667eea", linewidth=2.5, label="Portfolio Equity")
    ax_equity.set_ylabel("Equity ($)", color="#667eea", fontsize=12, fontweight="bold")
    ax_equity.tick_params(axis="y", labelcolor="#667eea")
    ax_equity.grid(True, alpha=0.2, linestyle="--")
    
    # Plot price if available
    if "price" in equity_to_plot.columns:
        ax_price.plot(
            x,
            equity_to_plot["price"],
            color="#48bb78",
            alpha=0.6,
            linewidth=2,
            label="NVDA Price",
        )
        ax_price.set_ylabel("NVDA Price ($)", color="#48bb78", fontsize=12, fontweight="bold")
        ax_price.tick_params(axis="y", labelcolor="#48bb78")
    
    ax_equity.set_xlabel("Time" if time_col == "Time" else "Bar Index", fontsize=12, fontweight="bold")
    ax_equity.set_title(title, fontsize=16, fontweight="bold", pad=20)
    
    # Legend
    lines_e, labels_e = ax_equity.get_legend_handles_labels()
    lines_p, labels_p = ax_price.get_legend_handles_labels()
    if lines_e or lines_p:
        ax_equity.legend(lines_e + lines_p, labels_e + labels_p, loc="upper left", framealpha=0.9)
    
    fig.tight_layout()
    st.pyplot(fig)


# =============================================================================
# PARAMETER GRID COMPONENTS
# =============================================================================

def render_parameter_grid(
    st,
    pd,
    params_df: pd.DataFrame,
    on_save: Callable[[pd.DataFrame], None] | None = None,
) -> pd.DataFrame:
    """Render an editable parameter grid (MT5 Strategy Tester style).
    
    Args:
        st: Streamlit module
        pd: Pandas module
        params_df: DataFrame with columns: Parameter, Value, Start, Step, Stop, Optimize
        on_save: Optional callback when user saves
    
    Returns:
        Updated DataFrame after editing
    """
    st.markdown("### ⚙️ Strategy Parameters")
    st.caption("MT5-style parameter grid. Edit 'Value' for single backtests, or configure Start/Step/Stop for optimization.")
    
    edited_df = st.data_editor(
        params_df,
        num_rows="fixed",
        key="strategy_params_grid",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Parameter": st.column_config.TextColumn("Parameter", disabled=True, width="medium"),
            "Value": st.column_config.NumberColumn("Value", format="%.4f", width="small"),
            "Start": st.column_config.NumberColumn("Start", format="%.4f", width="small"),
            "Step": st.column_config.NumberColumn("Step", format="%.4f", width="small"),
            "Stop": st.column_config.NumberColumn("Stop", format="%.4f", width="small"),
            "Optimize": st.column_config.CheckboxColumn("Optimize", width="small"),
        },
    )
    
    if on_save:
        if st.button("💾 Save Parameters", type="primary"):
            on_save(edited_df)
    
    return edited_df


# =============================================================================
# HISTORY TABLES
# =============================================================================

def render_history_table(
    st,
    pd,
    history: list[dict[str, Any]],
    title: str,
    columns: list[str] | None = None,
    on_row_select: Callable[[dict], None] | None = None,
) -> None:
    """Render a sortable history table with optional row selection.
    
    Args:
        st: Streamlit module
        pd: Pandas module
        history: List of history records
        title: Table title
        columns: Optional list of columns to display (None = all)
        on_row_select: Optional callback when row is selected
    """
    if not history:
        st.info(f"No {title.lower()} recorded yet.")
        return
    
    st.markdown(f"### 📜 {title}")
    
    df = pd.DataFrame(history)
    
    # Format timestamp if present
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
        df = df.sort_values("timestamp", ascending=False)
    
    # Select columns
    if columns:
        df = df[[col for col in columns if col in df.columns]]
    
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    if on_row_select and len(history) > 0:
        selected_idx = st.number_input(
            "Select row to use",
            min_value=0,
            max_value=len(history) - 1,
            value=0,
            key=f"history_select_{title}"
        )
        
        if st.button(f"✓ Use Selected {title}", key=f"use_history_{title}"):
            on_row_select(history[int(selected_idx)])


# =============================================================================
# DATA QUALITY COMPONENTS
# =============================================================================

def render_data_quality_summary(
    st,
    checks: list[dict[str, Any]],
    kpi: dict[str, Any],
) -> None:
    """Render data quality check results with KPI summary.
    
    Args:
        st: Streamlit module
        checks: List of check result dicts
        kpi: KPI summary dict with keys: score_0_100, n_pass, n_total, n_warn, n_fail
    """
    st.markdown("### 🔍 Data Quality Report")
    
    # KPI metrics
    metrics = [
        {"label": "Quality Score", "value": f"{kpi['score_0_100']:.1f}/100", "color": "success" if kpi['score_0_100'] > 90 else "warning", "icon": "⭐"},
        {"label": "Checks Passed", "value": f"{kpi['n_pass']}/{kpi['n_total']}", "color": "success", "icon": "✅"},
        {"label": "Warnings", "value": str(kpi['n_warn']), "color": "warning", "icon": "⚠️"},
        {"label": "Failures", "value": str(kpi['n_fail']), "color": "danger", "icon": "❌"},
    ]
    
    render_kpi_row(st, metrics)
    
    # Status banner
    if kpi["n_fail"]:
        st.error(f"❌ {kpi['n_fail']} checks FAILED. Review the table below.")
    elif kpi["n_warn"]:
        st.warning(f"⚠️ {kpi['n_warn']} checks have WARNINGS.")
    else:
        st.success("✅ All data quality checks passed!")
    
    # Detailed checks table
    with st.expander("📋 Detailed Check Results", expanded=kpi["n_fail"] > 0):
        import pandas as pd
        df = pd.DataFrame(checks).sort_values(["category", "id"])
        st.dataframe(df, use_container_width=True, hide_index=True)


# =============================================================================
# WALK-FORWARD COMPONENTS
# =============================================================================

def render_walkforward_results(
    st,
    pd,
    plt,
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame | None = None,
) -> None:
    """Render walk-forward analysis results with fold breakdown.
    
    Args:
        st: Streamlit module
        pd: Pandas module
        plt: Matplotlib.pyplot module
        results_df: Per-fold results DataFrame
        summary_df: Optional summary statistics per parameter set
    """
    st.markdown("### 🔄 Walk-Forward Results")
    
    if results_df.empty:
        st.info("No walk-forward results yet.")
        return
    
    # Summary metrics
    avg_sharpe = results_df["sharpe_ratio"].mean()
    avg_return = results_df["total_return"].mean() * 100
    n_folds = len(results_df)
    
    metrics = [
        {"label": "Avg Sharpe", "value": f"{avg_sharpe:.2f}", "color": "success" if avg_sharpe > 1 else "warning", "icon": "📊"},
        {"label": "Avg Return", "value": f"{avg_return:.1f}%", "color": "success" if avg_return > 0 else "danger", "icon": "💰"},
        {"label": "Folds", "value": str(n_folds), "color": "info", "icon": "🔢"},
    ]
    
    render_kpi_row(st, metrics)
    
    # Per-fold results
    st.markdown("#### 📊 Per-Fold Performance")
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    
    # Heatmap if parameter sets present
    if "label" in results_df.columns and len(results_df["label"].unique()) > 1:
        st.markdown("#### 🌡️ Robustness Heatmap (Sharpe by Fold)")
        
        pivot = results_df.pivot_table(
            index="label",
            columns="fold_idx",
            values="sharpe_ratio",
            aggfunc="mean"
        )
        
        fig, ax = plt.subplots(figsize=(12, max(4, len(pivot) * 0.5)))
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=3)
        
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"F{i}" for i in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        
        ax.set_xlabel("Fold", fontweight="bold")
        ax.set_ylabel("Parameter Set", fontweight="bold")
        ax.set_title("Sharpe Ratio Heatmap", fontweight="bold", pad=15)
        
        # Annotate cells
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.iloc[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color="black")
        
        plt.colorbar(im, ax=ax, label="Sharpe Ratio")
        fig.tight_layout()
        st.pyplot(fig)
    
    # Summary table
    if summary_df is not None and not summary_df.empty:
        st.markdown("#### 📈 Parameter Set Summary")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)


# =============================================================================
# ACTION PANELS
# =============================================================================

def render_action_panel(
    st,
    title: str,
    description: str,
    actions: list[dict[str, Any]],
) -> None:
    """Render an action panel with multiple action buttons.
    
    Args:
        st: Streamlit module
        title: Panel title
        description: Panel description
        actions: List of action dicts with keys: label, callback, type (primary/secondary/danger), disabled?
    """
    st.markdown(f"### {title}")
    st.caption(description)
    
    cols = st.columns(len(actions))
    
    for col, action in zip(cols, actions):
        with col:
            disabled = action.get("disabled", False)
            button_type = action.get("type", "secondary")
            
            if button_type == "primary":
                btn = st.button(action["label"], key=f"action_{title}_{action['label']}", type="primary", disabled=disabled)
            elif button_type == "danger":
                btn = st.button(action["label"], key=f"action_{title}_{action['label']}", disabled=disabled)
            else:
                btn = st.button(action["label"], key=f"action_{title}_{action['label']}", disabled=disabled)
            
            if btn and "callback" in action and not disabled:
                action["callback"]()


# =============================================================================
# LIVE TRADING COMPONENTS
# =============================================================================

def render_live_session_status(
    st,
    session_active: bool,
    last_event: dict[str, Any] | None,
    stats: dict[str, Any] | None,
) -> None:
    """Render live trading session status panel.
    
    Args:
        st: Streamlit module
        session_active: Whether live session is active
        last_event: Most recent event dict
        stats: Session statistics dict
    """
    st.markdown("### 🔴 Live Trading Status")
    
    status_color = "success" if session_active else "neutral"
    status_text = "ACTIVE" if session_active else "INACTIVE"
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        render_metric_card(st, "Session Status", status_text, color=status_color, icon="🔴" if session_active else "⚪")
    
    with col2:
        if last_event:
            st.markdown("**Last Event:**")
            event_type = last_event.get("event_type", "unknown")
            timestamp = last_event.get("timestamp", "unknown")
            st.caption(f"{event_type} at {timestamp}")
            
            with st.expander("Event Details"):
                st.json(last_event)
    
    if stats:
        st.markdown("#### 📊 Session Statistics")
        
        metrics = [
            {"label": "Positions Opened", "value": str(stats.get("positions_opened", 0)), "icon": "📈", "color": "info"},
            {"label": "Positions Closed", "value": str(stats.get("positions_closed", 0)), "icon": "📉", "color": "info"},
            {"label": "Current P&L", "value": f"${stats.get('current_pnl', 0):,.2f}", "icon": "💰", "color": "success" if stats.get("current_pnl", 0) > 0 else "danger"},
        ]
        
        render_kpi_row(st, metrics)
