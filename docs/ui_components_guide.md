# UI Components Library Guide

## Overview

The `src/ui/components.py` module provides a comprehensive, production-ready library of reusable UI components specifically designed for the LSTM trading system. These components understand your business logic and provide a consistent, professional user experience across all tabs.

## Design Philosophy

### 1. **Business Logic Awareness**
Components are not generic UI widgets—they understand:
- Models, experiments, and training runs
- Backtests, optimizations, and walk-forward analysis
- Live trading sessions and market data
- Async job execution patterns

### 2. **Stateless Components**
All components are pure functions: they take data as input and render UI. No hidden state management—this makes them:
- Easy to test
- Easy to reason about
- Easy to compose

### 3. **Consistent Visual Language**
- Gradient metric cards with color-coded meanings
- Status badges with semantic colors
- Professional typography and spacing
- Responsive layouts that adapt to screen size

### 4. **Action-Oriented Design**
Components include action buttons and callbacks, making it easy to:
- Promote models
- Load experiment results into training
- Select optimization results
- Export parameter sets

---

## Component Catalog

### Theme & Styling

#### `inject_custom_css(st)`
**Purpose:** Apply consistent styling across the entire app.

**Usage:**
```python
from src.ui import components

# In your main app.py, call once at startup
components.inject_custom_css(st)
```

**What it does:**
- Defines color variables (primary, success, warning, danger, info, neutral)
- Styles metric cards with gradients and shadows
- Styles status badges
- Styles data tables and panels

---

### Metric Cards

#### `render_metric_card(st, label, value, trend?, color?, icon?)`
**Purpose:** Display a single KPI in a beautiful gradient card.

**Business Use Cases:**
- Training: validation loss, epoch count
- Backtest: Sharpe ratio, total return, max drawdown
- Optimization: best Sharpe, number of runs
- Live: current P&L, positions count

**Example:**
```python
# Show Sharpe ratio with conditional coloring
sharpe = metrics["sharpe_ratio"]
color = "success" if sharpe > 1.0 else "warning" if sharpe > 0 else "danger"
components.render_metric_card(
    st, 
    label="Sharpe Ratio", 
    value=f"{sharpe:.2f}",
    trend="+0.3 vs last run",  # Optional
    color=color,
    icon="📈"
)
```

#### `render_kpi_row(st, metrics)`
**Purpose:** Display multiple KPIs in a row for dashboard-style views.

**Example:**
```python
metrics = [
    {"label": "Total Return", "value": "43.2%", "color": "success", "icon": "💰"},
    {"label": "Sharpe", "value": "2.1", "color": "success", "icon": "📈"},
    {"label": "Max DD", "value": "-12.3%", "color": "warning", "icon": "📉"},
    {"label": "Win Rate", "value": "58%", "color": "success", "icon": "🎯"},
]
components.render_kpi_row(st, metrics)
```

---

### Status Badges

#### `render_status_badge(st, status, text?)`
**Purpose:** Show job/process status with semantic colors.

**Business Use Cases:**
- Training jobs: QUEUED → RUNNING → SUCCEEDED/FAILED
- Optimization jobs: same lifecycle
- Live trading: ACTIVE/INACTIVE

**Example:**
```python
job_status = job_store.read_status(job_id)
components.render_status_badge(st, job_status.state)
```

---

### Model & Experiment Components

#### `render_model_card(st, model_name, val_loss, timestamp, frequency, tsteps, hyperparams, actions?)`
**Purpose:** Display a model with metadata and action buttons.

**Business Use Cases:**
- Model registry browser
- Training results display
- Model comparison

**Example:**
```python
components.render_model_card(
    st,
    model_name="lstm_60min_ts5_20241214_143022.keras",
    val_loss=0.000234,
    timestamp="2024-12-14 14:30:22",
    frequency="60min",
    tsteps=5,
    hyperparams={
        "lstm_units": 128,
        "batch_size": 64,
        "learning_rate": 0.003,
        "epochs": 50,
    },
    actions=[
        {
            "label": "Promote to Best",
            "callback": lambda: promote_model(model_name),
            "type": "primary"
        },
        {
            "label": "Delete",
            "callback": lambda: delete_model(model_name),
            "type": "danger"
        }
    ]
)
```

#### `render_experiment_table(st, pd, experiments, on_select?)`
**Purpose:** Display experiment history with selection for training.

**Business Use Cases:**
- Experiment history in Hyperparameter Experiments tab
- Select best experiment to train fully

**Example:**
```python
def load_experiment_to_training(experiment):
    st.session_state["train_prefill"] = experiment
    st.success("Loaded into training!")

components.render_experiment_table(
    st,
    pd,
    experiments=experiment_runs_list,
    on_select=load_experiment_to_training
)
```

---

### Training & Job Monitoring

#### `render_job_status(st, job_id, job_type, status, result?, on_refresh?)`
**Purpose:** Comprehensive job monitoring panel with timeline and results.

**Business Use Cases:**
- Training job monitoring (Train & Promote tab)
- Optimization job monitoring (Backtest / Strategy tab)
- Walk-forward job monitoring

**Example:**
```python
from src.jobs import store as job_store

job_id = train_state.get("active_job_id")
if job_id:
    job_status = job_store.read_status(job_id)
    result = job_store.read_result(job_id) if job_status.state == "SUCCEEDED" else None
    
    components.render_job_status(
        st,
        job_id=job_id,
        job_type="train",
        status=job_status.to_dict(),
        result=result,
        on_refresh=lambda: st.rerun()
    )
```

---

### Backtest Components

#### `render_backtest_metrics(st, pd, metrics)`
**Purpose:** Professional backtest metrics display with 8 key KPIs.

**Business Use Cases:**
- Single backtest results display
- Walk-forward per-fold results
- Optimization result preview

**Example:**
```python
# After running backtest
equity_df, trades_df, metrics = run_backtest(...)

components.render_backtest_metrics(st, pd, metrics)
```

**What you get:**
- Top row: Total Return, Sharpe, Max Drawdown, Win Rate (all with conditional coloring)
- Bottom row: CAGR, Profit Factor, Total Trades, Final Equity
- Expandable detailed metrics table

#### `render_equity_chart(st, plt, pd, equity_df, title?)`
**Purpose:** Professional equity curve chart with dual-axis price comparison.

**Business Use Cases:**
- Backtest equity visualization
- Walk-forward per-fold equity curves
- Live session P&L tracking

**Example:**
```python
# equity_df must have columns: equity, Time (optional), price (optional)
components.render_equity_chart(
    st,
    plt,
    pd,
    equity_df=equity_df,
    title="Backtest Results: 2023-01-01 to 2024-12-01"
)
```

**Features:**
- Automatic downsampling for large datasets (keeps UI responsive)
- Dual y-axis (equity + NVDA price)
- Color-coded lines
- Transparent background (works in dark/light mode)

---

### Parameter Grid Components

#### `render_parameter_grid(st, pd, params_df, on_save?)`
**Purpose:** MT5 Strategy Tester-style parameter grid editor.

**Business Use Cases:**
- Strategy parameter editing in Backtest tab
- Optimization parameter configuration
- Walk-forward parameter sets

**Example:**
```python
params_df = load_params_grid(defaults)

# Render editable grid
edited_df = components.render_parameter_grid(
    st,
    pd,
    params_df=params_df,
    on_save=lambda df: save_params_grid(df)
)

# Use edited values for backtest
k_sigma_long = edited_df[edited_df["Parameter"] == "k_sigma_long"]["Value"].iloc[0]
```

---

### History Tables

#### `render_history_table(st, pd, history, title, columns?, on_row_select?)`
**Purpose:** Sortable, filterable history table with row selection.

**Business Use Cases:**
- Backtest history
- Optimization history
- Training history
- Any time-series event log

**Example:**
```python
backtest_history = bt_state.get("history", [])

components.render_history_table(
    st,
    pd,
    history=backtest_history,
    title="Backtest History",
    columns=["timestamp", "frequency", "sharpe_ratio", "total_return", "n_trades"],
    on_row_select=lambda row: load_backtest_params(row)
)
```

---

### Data Quality Components

#### `render_data_quality_summary(st, checks, kpi)`
**Purpose:** Beautiful data quality report with KPIs and detailed checks.

**Business Use Cases:**
- Data tab quality checks visualization
- Pre-training data validation

**Example:**
```python
from src.data_quality import analyze_raw_minute_data, compute_quality_kpi

checks = analyze_raw_minute_data(RAW_DATA_CSV)
kpi = compute_quality_kpi(checks)

components.render_data_quality_summary(st, checks, kpi)
```

**What you get:**
- 4 KPI cards: Quality Score, Checks Passed, Warnings, Failures
- Color-coded status banner
- Expandable detailed checks table
- Auto-expands if failures present

---

### Walk-Forward Components

#### `render_walkforward_results(st, pd, plt, results_df, summary_df?)`
**Purpose:** Comprehensive walk-forward analysis visualization.

**Business Use Cases:**
- Walk-forward robustness results
- Parameter set stability analysis
- Per-fold performance breakdown

**Example:**
```python
# results_df columns: fold_idx, label, train_start, train_end, test_start, test_end,
#                      sharpe_ratio, total_return, max_drawdown, n_trades
# summary_df columns: label, avg_sharpe, min_sharpe, max_sharpe, std_sharpe

components.render_walkforward_results(
    st,
    pd,
    plt,
    results_df=wf_results,
    summary_df=wf_summary
)
```

**What you get:**
- 3 summary KPIs: Avg Sharpe, Avg Return, Fold Count
- Per-fold results table
- Robustness heatmap (if multiple parameter sets)
- Parameter set summary table

---

### Action Panels

#### `render_action_panel(st, title, description, actions)`
**Purpose:** Organized action buttons with descriptions.

**Business Use Cases:**
- Data ingestion controls
- Model management actions
- Export/import operations

**Example:**
```python
components.render_action_panel(
    st,
    title="Data Pipeline",
    description="Fetch, clean, and process data for model training",
    actions=[
        {
            "label": "Fetch from TWS",
            "callback": lambda: trigger_data_fetch(),
            "type": "primary"
        },
        {
            "label": "Clean Data",
            "callback": lambda: clean_raw_data(),
            "type": "secondary"
        },
        {
            "label": "Process Features",
            "callback": lambda: prepare_features(),
            "type": "secondary"
        }
    ]
)
```

---

### Live Trading Components

#### `render_live_session_status(st, session_active, last_event, stats)`
**Purpose:** Live trading session monitoring panel.

**Business Use Cases:**
- Live tab session status
- Real-time P&L tracking
- Position monitoring

**Example:**
```python
# Read from live trading log
last_event = read_last_event_from_log()
session_active = check_if_live_session_running()
stats = compute_live_session_stats()

components.render_live_session_status(
    st,
    session_active=session_active,
    last_event=last_event,
    stats=stats
)
```

---

## Integration Examples

### Example 1: Modernize Backtest Tab

**Before:**
```python
# Old code in backtest_page.py
st.subheader("Backtest Results")
st.write(f"Total Return: {metrics['total_return']*100:.2f}%")
st.write(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
st.write(f"Max DD: {metrics['max_drawdown']*100:.2f}%")
```

**After:**
```python
# New code using components
from src.ui import components

components.inject_custom_css(st)  # Once per tab
components.render_backtest_metrics(st, pd, metrics)
components.render_equity_chart(st, plt, pd, equity_df)
```

---

### Example 2: Modernize Training Tab

**Before:**
```python
st.write(f"Training job {job_id}")
st.write(f"Status: {job_status.state}")
if job_status.state == "SUCCEEDED":
    st.write(f"Validation loss: {result.validation_loss}")
```

**After:**
```python
from src.ui import components

components.render_job_status(
    st,
    job_id=job_id,
    job_type="train",
    status=job_status.to_dict(),
    result=result,
    on_refresh=lambda: st.rerun()
)
```

---

### Example 3: Modernize Experiments Tab

**Before:**
```python
if experiments:
    df = pd.DataFrame(experiments)
    st.dataframe(df)
    idx = st.number_input("Select row", 0, len(experiments)-1)
    if st.button("Load"):
        load_to_training(experiments[idx])
```

**After:**
```python
from src.ui import components

components.render_experiment_table(
    st,
    pd,
    experiments=experiments,
    on_select=load_to_training
)
```

---

## Migration Strategy

### Phase 1: Infrastructure (Immediate)
1. ✅ Create `src/ui/components.py` (DONE)
2. Add `inject_custom_css()` call to `src/ui/app.py`
3. Verify no visual regressions

### Phase 2: High-Impact Pages (Week 1)
1. **Backtest tab** (biggest visual win)
   - Replace metrics display with `render_backtest_metrics()`
   - Replace equity plot with `render_equity_chart()`
   - Add backtest history table with `render_history_table()`

2. **Train tab**
   - Replace job status with `render_job_status()`
   - Add model registry with `render_model_card()`

### Phase 3: Medium-Impact Pages (Week 2)
3. **Experiments tab**
   - Replace table with `render_experiment_table()`

4. **Optimization (within Backtest tab)**
   - Use `render_job_status()` for optimization jobs
   - Use `render_optimization_summary()` for results

### Phase 4: Polish Pages (Week 3)
5. **Data tab**
   - Replace quality checks with `render_data_quality_summary()`
   - Add action panels for data operations

6. **Walk-forward tab**
   - Replace results with `render_walkforward_results()`

7. **Live tab**
   - Add `render_live_session_status()`

---

## Best Practices

### 1. **Always Use Type Hints**
```python
from typing import Callable

def my_render_function(
    st,
    pd,
    data: list[dict[str, Any]],
    on_select: Callable[[dict], None] | None = None
) -> None:
    ...
```

### 2. **Keep Components Stateless**
```python
# ❌ Bad: Component manages state
def render_backtest(st):
    if "last_backtest" not in st.session_state:
        st.session_state["last_backtest"] = None
    ...

# ✅ Good: Component receives state as parameter
def render_backtest(st, last_backtest: dict | None):
    if last_backtest is None:
        st.info("No backtest run yet")
        return
    ...
```

### 3. **Use Callbacks for Actions**
```python
# Component doesn't know HOW to promote, just WHEN
actions = [
    {
        "label": "Promote",
        "callback": lambda: promote_training_row(row, best_hps_path, freq, tsteps),
        "type": "primary"
    }
]
```

### 4. **Follow Naming Conventions**
- `render_*` for components that display data
- `*_card` for single-item displays
- `*_table` for list/collection displays
- `*_chart` for visualizations

---

## Testing Components

### Manual Testing Checklist
- [ ] Metric cards display correct colors based on thresholds
- [ ] Status badges show correct colors for all states
- [ ] Job status shows timeline correctly
- [ ] Equity charts downsample large datasets
- [ ] Parameter grid saves correctly
- [ ] History tables sort by timestamp descending
- [ ] Action buttons trigger callbacks

### Visual Regression Testing
1. Take screenshots of old UI
2. Apply components
3. Take screenshots of new UI
4. Compare side-by-side
5. Verify improvements in:
   - Color consistency
   - Spacing/layout
   - Typography
   - Visual hierarchy

---

## Future Enhancements

### Additional Components to Consider
1. **Model Comparison Component**
   - Side-by-side model metrics
   - Training history charts
   - Feature importance comparison

2. **Trade List Component**
   - Sortable/filterable trade table
   - Per-trade P&L visualization
   - Export to CSV

3. **Real-Time Progress Component**
   - Live progress bar for long-running jobs
   - ETA calculation
   - Cancellation button

4. **Parameter Sensitivity Heatmap**
   - 2D heatmap for any two parameters
   - Interactive hover tooltips
   - Export to image

5. **Notification Toast Component**
   - Success/error toasts
   - Auto-dismiss
   - Action buttons in toasts

---

## Conclusion

This component library transforms your Streamlit UI from functional to professional. Key benefits:

✅ **Consistency:** All tabs use the same visual language  
✅ **Maintainability:** Update styling in one place  
✅ **Discoverability:** New developers see examples of best practices  
✅ **Extensibility:** Easy to add new components following existing patterns  
✅ **Business Logic Integration:** Components understand trading system concepts  

Start with high-impact pages (Backtest, Train) and gradually migrate all tabs. Each component has been designed with your actual business logic in mind, not generic UI needs.
