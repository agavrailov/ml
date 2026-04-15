"""Quick test to verify UI components render correctly."""

import streamlit as st
from src.ui import components

# Apply CSS
components.inject_custom_css(st)

st.title("🎨 Component Library Test")

# Test metric cards
st.header("1. Metric Cards Test")
metrics = [
    {"label": "Sharpe Ratio", "value": "2.14", "color": "success", "icon": "📈"},
    {"label": "Total Return", "value": "43.2%", "color": "success", "icon": "💰"},
    {"label": "Max Drawdown", "value": "-12.3%", "color": "warning", "icon": "📉"},
    {"label": "Win Rate", "value": "58%", "color": "success", "icon": "🎯"},
]
components.render_kpi_row(st, metrics)

# Test status badge
st.header("2. Status Badge Test")
col1, col2, col3 = st.columns(3)
with col1:
    components.render_status_badge(st, "RUNNING", "Training Job")
with col2:
    components.render_status_badge(st, "SUCCEEDED", "Completed")
with col3:
    components.render_status_badge(st, "FAILED", "Error")

# Test backtest metrics
st.header("3. Backtest Metrics Test")
import pandas as pd
test_metrics = {
    "total_return": 0.432,
    "sharpe_ratio": 2.14,
    "max_drawdown": -0.123,
    "win_rate": 0.58,
    "cagr": 0.381,
    "profit_factor": 1.85,
    "n_trades": 42,
    "final_equity": 14320.0,
    "period": "2023-01-01 to 2024-12-01"
}
components.render_backtest_metrics(st, pd, test_metrics)

st.success("✅ If you see colorful gradient cards above, components are working!")
st.info("💡 If you see plain text instead, there may be a CSS loading issue.")
