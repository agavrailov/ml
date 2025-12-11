# `src/app.py` UI & state requirements

This document captures the intended behaviour and UX/state requirements for the Streamlit app in `src/app.py`.

## Global goals

- Provide a **predictable, task-oriented workflow** across tabs:
  1. Data ingestion & preparation
  2. Hyperparameter experiments
  3. Train & promote best models
  4. Backtest & optimize strategy parameters
  5. Walk-forward robustness analysis
- Avoid surprising loss of user input or results:
  - When the user **enters parameters**, they should be stored.
  - When the user **runs backtests/optimizations/walk-forward**, results should remain visible after reruns and be discoverable later.
- Keep implementation **simple and explicit**:
  - Minimal abstractions.
  - No hidden global state beyond `st.session_state` and small JSON sidecars.

## State management requirements

1. **Single UI state entry point**
   - Use `st.session_state["ui_state"]` as the main container for long-lived UI state.
   - `ui_state` structure (conceptual):
     - `data`: data tab state (global frequency, recent data-quality runs, feature preview).
     - `experiments`: hyperparameter experiments (list of runs).
     - `training`: full training & evaluation summaries.
     - `strategy`: strategy parameter grid for backtests/optimization.
     - `backtests`: last backtest run + summary history.
     - `optimization`: last optimization run + summary history.
     - `walkforward`: parameter sets and robustness results.
   - Provide a small helper to initialize and return `ui_state`.

2. **Persistence across Streamlit restarts**
   - Key histories must survive app restarts via JSON files under a small `ui_state/` directory in the repo:
     - Backtest summary history.
     - Optimization summary history.
     - Walk-forward robustness summary history.
     - Strategy parameter grid (already via `ui_strategy_params.json`).
     - Walk-forward parameter sets grid.
   - On app startup (first use of each section), load history/grids from disk into `ui_state` if files exist.
   - On each new run that mutates a history/grid, write the updated data back to JSON.
   - History tables are capped at a fixed maximum length (e.g. last 100 rows) by dropping the oldest entries.

3. **Session-level state vs. persisted state**
   - In-memory (`ui_state` only, not persisted):
     - "Last run" details that are large or primarily visual:
       - Full equity curve DataFrame for the last backtest.
       - Full optimization result grid for the last optimization.
       - Per-fold walk-forward robustness DataFrame.
       - Latest model evaluation metrics & locations of plots.
   - Persisted (JSON):
     - Compact summaries: timestamps, frequencies, trade side, and key metrics needed to reconstruct **tables**, not plots.

4. **History size limits**
   - Use a constant `MAX_HISTORY_ROWS` (default 100) per history type:
     - If appending a new row pushes the length above the limit, drop the oldest rows.
   - This keeps JSON files small and table rendering fast while still providing recent context.

## Per-tab behavioural requirements

### Data tab (`tab_data`)

- Maintain `global_frequency` in `ui_state["data"]` (and optionally mirrored to the existing `"global_frequency"` key for backward compatibility).
- When running **data quality checks**:
  - Compute quality metrics as today.
  - Append a summary entry to `ui_state["data"]["data_quality_runs"]` including:
    - Timestamp.
    - Dataset path.
    - Overall score.
    - Number of checks, passes, warnings, failures.
  - Optionally persist `data_quality_runs` to JSON.
  - Display a small table of recent runs below the data-quality controls.
- When preparing features for Keras input:
  - Store a small preview (e.g. first 200 rows + column list) in `ui_state["data"]["feature_preview"]`.
  - Render the preview from this state on rerun.

### Hyperparameter Experiments tab (`tab_experiments`)

- Back experiments with `ui_state["experiments"]["runs"]` instead of an ad-hoc `"lstm_experiments"` list.
- On each "Run single experiment":
  - Run training as today.
  - Append a run record with:
    - Timestamp.
    - Frequency and TSTEPS.
    - All hyperparameters (units, layers, batch_size, epochs, learning_rate, stateful, feature set).
    - Validation loss.
    - Model filename and bias-correction filename.
  - Persist `experiments["runs"]` to JSON (e.g. `ui_state/experiments_runs.json`).
- Display all experiment runs as a dataframe (sortable/filterable).
- Allow selecting a row (by index or label) and store its dict as `ui_state["training"]["train_prefill"]` for the Train & Promote tab.

### Train & Promote tab (`tab_train`)

- Use `ui_state["training"]["train_prefill"]` to prefill frequency, TSTEPS, and hyperparams when available.
- On "Train full model & auto-promote if better":
  - Behaviour of training, promotion, and prediction CSV generation remains unchanged.
  - After a successful run, write a `last_train_run` record into `ui_state["training"]` with:
    - Timestamp.
    - Frequency and TSTEPS.
    - Hyperparams and feature set used.
    - Validation loss.
    - Whether promotion happened.
    - Model filename and bias-correction filename.
    - Predictions CSV path.
  - Append a compact summary row to `ui_state["training"]["history"]` and persist to JSON (e.g. `ui_state/training_history.json`).
- Render a "Recent training runs" table from `training["history"]`.
- On "Evaluate best model":
  - After evaluation, store results (MAE, correlation, paths to plots) in `ui_state["training"]["last_evaluation"]` so they can be re-rendered on refresh.

### Backtest / Strategy tab (`tab_backtest`)

1. **Strategy parameters**
   - Continue to use the data editor for the parameter grid.
   - Back it with `ui_state["strategy"]["param_grid_df"]` and `ui_strategy_params.json`:
     - On first run, load from `ui_strategy_params.json` via `_load_params_grid` (existing behaviour).
     - On each rerender, write the current DataFrame into `ui_state["strategy"]["param_grid_df"]`.
     - Auto-save the grid to `ui_strategy_params.json` using `_save_params_grid`.
   - The "Save parameters to config.py" button remains responsible only for updating numeric defaults in `config.py`.

2. **Backtest runs**
   - Split logic into:
     - A **runner** that calls `_run_backtest` once and stores results, and
     - A **renderer** that always reads from stored state.
   - When the user clicks "Run backtest":
     - Call `_run_backtest` with the current inputs.
     - Store the result in `ui_state["backtests"]["last_run"]`, including:
       - Inputs: frequency, date range, trade side, key parameter values.
       - Metrics dict.
       - Full equity curve DataFrame (session only, not persisted to JSON).
     - Append a summary row to `ui_state["backtests"]["history"]`, containing:
       - Timestamp.
       - Frequency.
       - Trade side.
       - Total return, CAGR, max drawdown, Sharpe, win rate, profit factor.
       - Number of trades, final equity.
     - Truncate `history` to `MAX_HISTORY_ROWS` and persist this summary list to JSON.
   - On every rerun, regardless of button presses:
     - If `last_run` exists, re-render the equity curve and metrics table from it.
     - If `history` is non-empty, show a "Backtest history" table using the summary entries (most recent first).
   - The user should **not** lose their last backtest plot/metrics when any other widget interaction causes a rerun.

3. **Optimization**
   - Maintain current behaviour for running grid search and computing metrics.
   - When the user clicks "Run optimization":
     - Run the grid as today.
     - Store into `ui_state["optimization"]["last_run"]`:
       - Timestamp.
       - Frequency, date range, trade side.
       - Parameter ranges used.
       - Optimization result DataFrame (`results_df`, in-session only).
     - Append a compact summary row into `ui_state["optimization"]["history"]` and persist summaries to JSON.
   - On rerun without pressing the button:
     - If `last_run` exists, re-display the optimization results dataframe and associated controls.
   - Keep existing actions:
     - Load a row into the strategy parameter grid.
     - Export top-N Sharpe rows to the Walk-forward tab.

### Walk-Forward Analysis tab (`tab_walkforward`)

- Parameter grid for robustness:
  - Backed by `ui_state["walkforward"]["param_sets_df"]`.
  - Persist to JSON (e.g. `ui_state/walkforward_param_sets.json`).
  - If the Backtest tab exports a seed, use that data to initialize the grid for the current session **once**, then continue editing.
- Robustness evaluation (Sharpe across folds):
  - When run, compute folds as today.
  - Store the per-fold results DataFrame as `ui_state["walkforward"]["robust_results_df"]`.
  - Compute an aggregated summary (mean/std/min/max Sharpe, fraction of Sharpe>0, n_folds) per parameter label and store as `ui_state["walkforward"]["summary_df"]`.
  - Append a compact summary of this robustness run to a persisted history list (JSON) including:
    - Timestamp.
    - Frequency.
    - Window parameters.
    - Basic robustness statistics (e.g. best mean Sharpe, worst mean Sharpe).
- On rerun without pressing the button:
  - If `robust_results_df` and `summary_df` exist, re-display summary table and plots.

## Non-goals / constraints

- Do **not** introduce a heavy state-management framework; keep everything as:
  - `st.session_state["ui_state"]` (nested dicts, DataFrames), plus
  - A few small JSON files under a `ui_state/` directory.
- Avoid changing core backtest / model / data logic; the refactor should primarily change how the UI stores and presents results.
- Maintain backwards compatibility for existing tests and behaviours where possible (e.g. continue exposing private helpers like `_load_strategy_defaults`, `_build_default_params_df`, `_load_params_grid`, `_save_params_grid`).
