# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is an LSTM-based price prediction and automated trading system for NVDA stock. The system operates in phases:
1. **Data ingestion**: Minute-level OHLC data from Interactive Brokers (IB/TWS)
2. **Processing pipeline**: Cleaning, gap-filling, resampling to configurable frequencies (15min, 60min, 240min)
3. **Model training**: Stateful LSTM models with hyperparameter tuning and model registry
4. **Trading**: Backtesting, walk-forward validation, and live trading with TP/SL-based risk management

The system is moving from long-only to bidirectional trading (long/short), with future support for multiple tickers planned (see `docs/ROADMAP.md`).

## Common commands

### Dependencies & environment

Python dependencies are listed in `requirements.txt` (TensorFlow, pandas, numpy, scikit-learn, FastAPI, uvicorn, pydantic, streamlit, altair, pytest, ib_insync).

Install them into your active environment:

```bash
pip install -r requirements.txt
```

On Windows, `run.bat` assumes a virtual environment at `.\venv` and activates `.\venv\Scripts\activate` before starting services. Ensure that venv exists and has the requirements installed if you rely on `run.bat`.

### Running the API and UI

From the repository root:

- Start FastAPI API (development):

  ```bash
  uvicorn api.main:app --reload --port 8000
  ```

  Then open Swagger docs at `http://127.0.0.1:8000/docs`.

- Start Streamlit UI (workflow UI):

  ```bash
  streamlit run src/ui/app.py
  ```

  Note: `src/ui/app.py` currently delegates to the legacy monolithic app at `src/app.py`.
  As pages are migrated into `src/ui/pages/*`, we will retire `src/app.py`.

- Windows helper script to start both API and UI and open the browser:

  ```bat
  run.bat
  ```

The UI expects the API at `http://localhost:8000/predict` and, for auto-fetch mode, a processed hourly CSV at `data/processed/nvda_hourly.csv`.

### Tests

Pytest is used for the test suite under `tests/`, with `tests/conftest.py` adding the project root to `sys.path`.

Run all tests from the repo root:

```bash
pytest
```

Run a single test file:

```bash
pytest tests/test_model.py
```

Run a single test case:

```bash
pytest tests/test_model.py::test_build_lstm_model_architecture_single_layer_stateful
```

### Data ingestion and daily pipeline

Several scripts manage ingestion of NVDA minute-level data from Interactive Brokers (IB/TWS) and preprocessing:

- One-off IB ingestion of NVDA minute data (uses IB connection settings from `src/config.py`):

  ```bash
  python -m src.data_ingestion --symbol NVDA --start 2024-01-01 --end 2024-02-01
  ```

- Daily data pipeline agent that chains ingestion, cleaning, gap analysis/filling, and resampling/feature engineering:

  ```bash
  python src/daily_data_agent.py
  ```

  Add `--skip-ingestion` for dry runs that operate only on existing raw data.

- Windows helper script that runs the daily pipeline:

  ```bat
  update_data.bat
  ```

  This calls `python src/daily_data_agent.py` using the active virtual environment.

### Job runner (async task execution)

For long-running jobs (backtests, training, walk-forward, optimization), the UI uses an async job runner:

```bash
python -m src.jobs.run --job-id <uuid> --job-type <BACKTEST|TRAIN|OPTIMIZE|WALKFORWARD> --request <path-to-request.json>
```

Job state and artifacts are stored under `runs/<job_id>/` with:
- `request.json`: Input parameters
- `status.json`: Job state (QUEUED, RUNNING, SUCCEEDED, FAILED)
- `result.json`: Output artifacts (trades CSV, equity CSV, metrics)

The UI polls `status.json` for progress updates.

### Training, experiments, and evaluation

Core training and experimentation scripts are under `src/` and operate on processed hourly data produced by the pipelines above.

- Train a single LSTM model using current defaults / tuned hyperparameters:

  ```bash
  python src/train.py
  ```

  This writes models under `models/registry/` and maintains `best_hyperparameters.json` in the project root.

- Run a hyperparameter experiment grid across frequencies, TSTEPS, and model settings (can be expensive):

  ```bash
  python src/experiment_runner.py
  ```

  Results are saved to `experiment_results.json` and `best_hyperparameters.json`.

- Evaluate a saved model (MAE, RMSE, correlation, plots) using the recent validation window:

  ```bash
  python src/evaluate_model.py
  ```

  `evaluate_model_performance(...)` is also imported by `experiment_runner.py` for programmatic evaluation.

- Hyperparameter tuning with KerasTuner (alternative to the manual grid runner):

  ```bash
  python src/tune_hyperparameters.py
  ```

  This expects `TRAINING_DATA_CSV` and `SCALER_PARAMS_JSON` (see config section) and writes `best_hyperparameters.json`.

- Rolling-window retraining of the model (starting from the latest registry model):

  ```bash
  python src/retrain.py --train_window_size 2000 --validation_window_size 500
  ```

  CLI arguments let you override LSTM units, learning rate, batch size, and window sizes.

- Promote the most recently saved model to be the active one (writes `models/active_model.txt`):

  ```bash
  python src/promote_model.py
  ```

  Note: `promote_model.py` currently depends on `get_latest_model_path()` from `src.config`, while `config.py` exposes `get_latest_best_model_path(...)`. Adjustments may be required before using this script in production flows.

### Backtesting

Run backtests using the unified backtest engine (shared by both offline and paper-trading paths):

```bash
python -m src.backtest --initial-equity 10000 --start-date 2024-01-01 --end-date 2024-12-31
```

Key parameters:
- `--frequency 60min`: Trading timeframe
- `--prediction-mode model|csv`: Use live model or pre-generated predictions
- `--predictions-csv`: Path to predictions CSV (if using csv mode)
- `--risk-per-trade-pct 0.01`: Risk percentage per trade
- `--k-sigma-err 0.5`, `--k-atr-min-tp 1.5`: Strategy filter thresholds

Outputs (saved under `backtests/`):
- Equity plot PNG with parameters in filename
- Trades CSV
- Equity curve CSV
- Backtest metrics (Sharpe, max drawdown, win rate, profit factor)

Walk-forward backtesting:

```bash
python scripts/run_walkforward_backtest.py --frequency 60min --test-span-months 3 --train-lookback-months 24
```

### Live trading

Live trading connects to IBKR/TWS via `ib_insync` and runs the same strategy logic as backtests:

```bash
python -m src.ibkr_live_session --symbol NVDA --frequency 60min --tsteps 5 --initial-equity 10000
```

Safety mechanisms:
- Kill switch file: `ui_state/live/KILL_SWITCH` prevents order submission without stopping the daemon
- Reconnect controller handles IB disconnections with backoff
- JSONL event log for all trading decisions and orders

Live session state:
- `ui_state/live/configs/active.json`: Current strategy parameters
- `ui_state/live/logs/<run_id>.jsonl`: Event log (bars, predictions, trades)
- Events: `BAR_RECEIVED`, `PREDICTION_GENERATED`, `DECISION_MADE`, `ORDER_SUBMITTED`, `ORDER_FILLED`

### Prediction / inference utilities

The main prediction entry point is the FastAPI `/predict` endpoint, which internally calls `src.predict.predict_future_prices`.

For ad-hoc local experimentation (without FastAPI), you can run `src/predict.py` directly:

```bash
python src/predict.py
```

The script’s `__main__` section demonstrates how to construct dummy OHLC input data of length `20 + TSTEPS` and call `predict_future_prices(...)`.

### Gap analysis utilities

Utilities for identifying gaps in raw minute data and checking data files:

- Analyze gaps (long weekday gaps in minute data) and write a JSON summary:

  ```bash
  python analyze_gaps.py data/raw/nvda_minute.csv data/processed/missing_trading_days.json
  ```

- Simple existence/emptiness check for the primary raw CSV:

  ```bash
  python check_file.py
  ```

Both are wired into `daily_data_agent.py` via the gap analysis and gap-filling pipeline.


## High-level architecture

### Overall system

This repository implements an end-to-end LSTM-based price prediction system for NVDA stock, including:

- Data ingestion from Interactive Brokers (IB/TWS) into minute-level OHLC CSVs.
- Cleaning, gap analysis/filling, resampling to hourly bars, and feature engineering.
- Training and evaluating stateful LSTM models, with a registry of models and hyperparameters.
- Hyperparameter tuning and retraining workflows.
- A prediction API (FastAPI) and interactive UI (Streamlit) that expose the latest model.

The core package is under `src/`, with tests under `tests/`, an API under `api/`, and a UI under `ui/`.

### Key architecture patterns

**Config-driven design**: Almost all paths, hyperparameters, and runtime settings flow through `src/config.py` via dataclasses (`PathsConfig`, `TrainingConfig`, `IbConfig`, `MarketConfig`). When adding new parameters, extend the relevant config dataclass rather than hardcoding values.

**Stateful vs non-stateful models**: Training uses stateful LSTMs with fixed `batch_shape=(batch_size, tsteps, n_features)` for sequential learning. Inference requires non-stateful clones (dynamic batch size) created via `build_lstm_model(...)` with `stateful=False` and `load_stateful_weights_into_non_stateful_model(...)`.

**Model registry and selection**: Models are saved to `models/registry/` with timestamped filenames. `best_hyperparameters.json` tracks the best model per `(frequency, TSTEPS)` by validation loss. `get_latest_best_model_path(frequency, tsteps)` selects the active model; `promote_model.py` writes `models/active_model.txt` for the API.

**Shared backtest engine**: Both offline backtesting (`src/backtest.py`) and paper trading (`src/paper_trade.py`) delegate to `src.backtest_engine.run_backtest()`, ensuring identical entry/exit logic, commission handling, and ATR-based filters. Any strategy changes must be made in the shared engine.

**ATR-based volatility**: Average True Range (ATR) is computed per-bar via `_compute_atr_series(data, window=14)` and used for SNR filters and position sizing. Both backtest and live trading use the same per-bar ATR series (not scalar approximations).

**Bias correction**: After training, `train.py` computes a rolling mean residual on validation data and saves it to `bias_correction_<frequency>_tsteps<...>.json`. This is applied at prediction time to compensate for systematic over/under-prediction.

### Data flow and critical dependencies

**Raw → curated → processed pipeline**:
1. `data/raw/nvda_minute.csv`: Raw minute bars from IB/TWS (via `src/ingestion/tws_historical.py`)
2. `data/processed/nvda_minute_curated.csv`: Cleaned, gap-filled snapshot (via `src/ingestion/curated_minute.py`)
3. `data/processed/nvda_<frequency>.csv`: Resampled OHLC (via `convert_minute_to_timeframe`)
4. `data/processed/training_data_<frequency>.csv`: Feature-enriched data (via `prepare_keras_input_data`)

**Scaler parameters**: Saved as `data/processed/scaler_params_<frequency>.json` during training, loaded at prediction time to ensure consistent normalization.

**Predictions CSV**: For backtest modes using pre-computed predictions, generate via `scripts/generate_predictions_csv.py` and save to `backtests/<symbol>_<frequency>_predictions.csv`.

**Job artifacts**: Async jobs write to `runs/<job_id>/` with structured JSON outputs (request, status, result).

**UI state**: The Streamlit UI persists state under `ui_state/` for live trading configs, kill switch, and session logs.

**Windows environment**: This repo is primarily developed on Windows. Batch scripts (`.bat`) assume a virtual environment at `.\venv`. When contributing cross-platform features, test both PowerShell and bash paths.

---

### Configuration and paths (`src/config.py`)

`src/config.py` is the central configuration module and is referenced across virtually all other modules. It provides:

- **Base directory and paths** via dataclasses:
  - `BASE_DIR` defaults to the project root, but can be overridden with `ML_LSTM_BASE_DIR`.
  - `PathsConfig` computes:
    - `raw_data_dir` → default `BASE_DIR/data/raw`
    - `processed_data_dir` → default `BASE_DIR/data/processed`
    - `model_save_path` → default `BASE_DIR/models/my_lstm_model.keras`
    - `model_registry_dir` → default `BASE_DIR/models/registry`
    - `active_model_path_file` → default `BASE_DIR/models/active_model.txt`
  - Helper methods for derived paths:
    - `raw_data_csv()` → `nvda_minute.csv`
    - `hourly_data_csv(frequency)` → `nvda_<frequency>.csv`
    - `training_data_csv(frequency)` / `scaler_params_json(frequency)` for training artifacts.

- **Training hyperparameters** wrapped in `TrainingConfig`:
  - Defaults for a single run: `FREQUENCY`, `TSTEPS`, `ROWS_AHEAD`, `TR_SPLIT`, `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`, `LSTM_UNITS`, dropout rates, `N_LSTM_LAYERS`, `STATEFUL`, `N_FEATURES`, `OPTIMIZER_NAME`, `LOSS_FUNCTION`.
  - Search spaces: `RESAMPLE_FREQUENCIES`, `TSTEPS_OPTIONS`, `LSTM_UNITS_OPTIONS`, `BATCH_SIZE_OPTIONS`, `DROPOUT_RATE_OPTIONS`, `N_LSTM_LAYERS_OPTIONS`, `STATEFUL_OPTIONS`, and `FEATURES_TO_USE_OPTIONS` (feature subsets).

- **IB / TWS configuration** via `IbConfig`:
  - Host/port/client ID (`TWS_HOST`, `TWS_PORT`, `TWS_CLIENT_ID`), max concurrent requests (`TWS_MAX_CONCURRENT_REQUESTS`), and NVDA contract details (`NVDA_CONTRACT_DETAILS`).

- **Market configuration** via `MarketConfig`:
  - Exchange calendar name (`EXCHANGE_CALENDAR_NAME`), timezone, and market open/close times for trading-day-aware operations.

- **Hyperparameter persistence and model selection helpers**:
  - `load_tuned_hyperparameters(...)` and `get_run_hyperparameters(...)` read `best_hyperparameters.json` and overlay tuned values onto defaults.
  - `get_hourly_data_csv_path(...)`, `get_training_data_csv_path(...)`, `get_scaler_params_json_path(...)` centralize path construction.
  - `get_active_model_path()` reads the active model pointer file.
  - `get_latest_best_model_path(target_frequency, tsteps)` selects the best model (lowest validation loss) from `best_hyperparameters.json`, returning its path plus bias-correction and metadata.

When changing directory layout, hyperparameters, or IB settings, update `src/config.py` first. Most downstream scripts consume its constants and helper functions rather than hardcoding paths.

---

### Data ingestion and raw data management

#### `src/ingestion/` and `src/data_ingestion.py`

The TWS/IB ingestion core lives under the `src/ingestion/` package:

- `src/ingestion/tws_historical.py` implements the asynchronous ingestion of historical data from IB/TWS into `RAW_DATA_CSV` (default `data/raw/nvda_minute.csv`). Key pieces:
  - Uses `ib_insync` and IB’s `reqHistoricalDataAsync` to fetch OHLC bars.
  - `_get_latest_timestamp_from_csv(...)` reads the last `DateTime` from the existing CSV to resume from where the previous run left off.
  - `fetch_historical_data(...)`:
    - Connects to TWS using `TWS_HOST`, `TWS_PORT`, `TWS_CLIENT_ID` and builds a `Stock`/`Forex`/`CFD`/`Future` contract based on `NVDA_CONTRACT_DETAILS`.
    - Computes an effective concurrency level (`effective_concurrency`) based on IB pacing rules and `TWS_MAX_CONCURRENT_REQUESTS`.
    - Walks backwards in daily chunks from `end_date` to `start_date` (or to the last CSV timestamp if `strict_range=False`), requesting 1-day windows and aggregating them in batches before writing to CSV.
    - Writes a canonical `DateTime,Open,High,Low,Close` CSV, ensuring chronological order and limited concurrency.
  - `trigger_historical_ingestion(...)` is a small convenience wrapper around `fetch_historical_data` that is suitable for CLI usage and orchestration.

- `src/ingestion/curated_minute.py` defines the curated-minute layer:
  - `CURATED_MINUTE_PATH` (currently `data/processed/nvda_minute_curated.csv`) is a snapshot of cleaned, gap-handled minute data.
  - `run_transform_minute_bars("NVDA")` cleans `RAW_DATA_CSV` (via `clean_raw_minute_data(...)`) and writes canonical `DateTime,Open,High,Low,Close` columns to `CURATED_MINUTE_PATH`.
  - `get_raw_bars(...)` and `get_curated_bars(...)` are small helpers that load and optionally time-filter these CSVs.

- `src/data_ingestion.py` is now a thin wrapper around this core, re-exporting `_get_latest_timestamp_from_csv`, `fetch_historical_data`, and `trigger_historical_ingestion` so existing imports and the `python src/data_ingestion.py` CLI keep working.

`tests/test_data_ingestion.py` targets the ingestion core by patching `IB` and `util.df` in `src.ingestion.tws_historical` while still importing `fetch_historical_data` from `src.data_ingestion`.

#### Daily pipeline vs. one-off ingestion

The preferred operational entrypoint for keeping data up to date is `src/daily_data_agent.py`, which wraps the ingestion core, gap analysis/filling, curated-minute generation, and resampling/feature engineering into one sequence.

`update_data.bat` is a Windows helper script that activates the virtual environment and runs `python src/daily_data_agent.py`.

#### `src/daily_data_agent.py`

This module is the high-level “daily pipeline agent” that ties multiple tools together and now orchestrates the entire raw → curated-minute → hourly flow:

- `run_gap_analysis()` shell-executes `analyze_gaps.py` on `RAW_DATA_CSV`, writing gap metadata to `GAP_ANALYSIS_OUTPUT_JSON` (under `PROCESSED_DATA_DIR`).
- `smart_fill_gaps()` reloads the raw CSV, applies `fill_gaps(...)` from `src.data_processing` using gap metadata, and writes a gap-filled CSV back.
- `ingest_new_data()` wraps the ingestion core (`fetch_historical_data(...)` from `src.ingestion`) to pull new minute data since a default start date into `RAW_DATA_CSV`.
- `run_transform_minute_bars("NVDA")` (imported from `src.ingestion`) creates a curated-minute snapshot at `CURATED_MINUTE_PATH` from the cleaned, gap-handled raw data.
- `resample_and_add_features()`:
  - Ensures `PROCESSED_DATA_DIR` exists.
  - Calls `convert_minute_to_timeframe(CURATED_MINUTE_PATH, FREQUENCY, PROCESSED_DATA_DIR)` to resample curated minute NVDA data to the configured `FREQUENCY` (e.g. `15min`, `60min`).
  - Uses `FEATURES_TO_USE_OPTIONS[0]` and `prepare_keras_input_data(...)` to engineer features for the configured frequency.
- `run_daily_pipeline(skip_ingestion=False)` wires everything into one sequence:
  1. Optionally ingest new raw data from IB.
  2. Clean and deduplicate raw minute data.
  3. Run gap analysis and fill small weekday gaps.
  4. Create a curated-minute snapshot via `run_transform_minute_bars`.
  5. Resample curated minutes to hourly and engineer features for training.

This is the best entry point for a once-per-day maintenance job: it leaves you with both up-to-date curated-minute data and hourly feature-ready CSVs under `data/processed/`.

---

### Data processing and feature engineering (`src/data_processing.py`)

This module converts raw minute-level data into hourly (or other frequency) feature-enriched datasets for model training and evaluation:

- `convert_minute_to_timeframe(input_csv_path, frequency, processed_data_dir=PROCESSED_DATA_DIR)`:
  - Reads minute-level `DateTime`-indexed OHLC data.
  - Uses `DataFrame.resample(frequency)` with `first/max/min/last` aggregation for OHLC.
  - Drops empty intervals, resets the index, renames `DateTime` → `Time`, and formats timestamps as `YYYY-MM-DD HH:MM:SS`.
  - Writes `nvda_<frequency>.csv` under `processed_data_dir`.

- `add_features(df, features_to_generate)`:
  - Ensures `Time` is a datetime column.
  - Optionally adds:
    - `SMA_7`, `SMA_21` (rolling averages of `Close`).
    - `RSI` (14-period RSI on `Close`).
    - `Hour` and `DayOfWeek` (time-based features).
  - Drops rows with NaNs introduced by rolling windows.

- `prepare_keras_input_data(input_hourly_csv_path, features_to_use)`:
  - Reads an hourly CSV.
  - Calls `add_features(...)` with a superset of all possible features.
  - Filters down to the requested `features_to_use`, keeping `Time` plus those feature columns.
  - Returns `(df_filtered, feature_cols)` where `feature_cols` is the exact list used in downstream pipelines.

- `clean_raw_minute_data(input_csv_path)`:
  - Handles legacy `index` column names, renaming to `DateTime`.
  - Parses `DateTime`, sorts chronologically, drops duplicate timestamps, and rewrites the cleaned CSV.

- `fill_gaps(df, identified_gaps, max_fill_hours=48)`:
  - Filled gaps are derived from JSON output of `analyze_gaps.py`.
  - For gaps shorter than `max_fill_hours`, synthesizes intermediate one-minute bars by forward-filling OHLC values.
  - Leaves very long gaps (likely holidays/outages) untouched but logs them.

`tests/test_data_processing.py` verifies resampling behavior, feature generation, and Keras input preparation.

---

### Data utilities and sequence generation (`src/data_utils.py`)

`src/data_utils.py` contains model-agnostic utilities used in both training and evaluation:

- `fit_standard_scaler(df, feature_cols)` and `apply_standard_scaler(df, feature_cols, scaler_params)` implement a simple mean/std scaler, with zero std devs clamped to 1.0. Parameters are JSON-serializable and persisted by training code.
- `get_effective_data_length(data, sequence_length, rows_ahead)` computes how many `(sequence_length, rows_ahead)` pairs can be formed, taking into account label shifting and exclusion of the `Time` column.
- `create_sequences_for_stateless_lstm(...)` builds `(X, y)` for non-stateful models, with shape `(n_samples, sequence_length, n_features)` and `(n_samples, 1)`.
- `create_sequences_for_stateful_lstm(...)` enforces that the number of samples is a multiple of `batch_size`, truncating as needed to maintain contiguous sequences.

These functions are heavily exercised in `tests/test_data_utils.py` and are central to both `train.py` and `evaluate_model.py`.

---

### Model definition (`src/model.py`)

`src/model.py` encapsulates the LSTM architecture and the mapping between stateful and stateless models:

- `build_lstm_model(input_shape, lstm_units, batch_size, learning_rate, n_lstm_layers, stateful, optimizer_name, loss_function)`:
  - For `stateful=True`, creates an `Input` with fixed `batch_shape=(batch_size, tsteps, n_features)`; for `stateful=False`, uses dynamic batch size.
  - Stacks `n_lstm_layers` LSTM layers, using `return_sequences=True` on all but the last.
  - Adds a single `Dropout` layer after each LSTM layer (using `DROPOUT_RATE_1` from config).
  - Final layer is a `Dense(1)` predicting a single-step price.
  - Compiles the model with either RMSprop or Adam and a specified loss.

- `load_stateful_weights_into_non_stateful_model(stateful_model, non_stateful_model)`:
  - Transfers weights between architecture-compatible models (typically stateful → non-stateful) so inference can be done without stateful constraints.

`tests/test_model.py` covers architecture, compilation, forward passes, and weight-transfer correctness for both stateful and non-stateful variants.

---

### Training, experiments, tuning, and retraining

#### `src/train.py`

This is the main training orchestrator for a single (frequency, TSTEPS) configuration:

- Reloads `src.config` at runtime to pick up updated environment-based overrides.
- Pulls tuned or default hyperparameters via `get_run_hyperparameters(frequency, tsteps)` and allows explicit overrides through arguments.
- Prepares data:
  - Locates the hourly CSV with `get_hourly_data_csv_path(frequency)`.
  - Uses `prepare_keras_input_data(...)` to add features and filter to the selected feature subset.
  - Splits into train/validation sets using `TR_SPLIT`.
  - Fits the standard scaler on training data and applies it to both train/val.
  - Uses `create_sequences_for_stateful_lstm(...)` to build windowed sequences.
- Builds and trains a stateful LSTM via `build_lstm_model(...)`, with `EarlyStopping` on validation loss.
- Saves the trained model into `MODEL_REGISTRY_DIR` with a timestamped filename.
- Computes a mean residual (bias) on the validation set by:
  - Building a non-stateful clone, transferring weights, predicting on validation sequences, and denormalizing.
  - Saving bias statistics to `bias_correction_<frequency>_tsteps<...>.json`.
- Updates `best_hyperparameters.json` with validation loss, model filename, and key hyperparameters for the given `(frequency, TSTEPS)`.

The `__main__` section trains a model using config defaults and updates `best_hyperparameters.json` accordingly.

#### `src/experiment_runner.py`

This module implements two related flows:

- `run_single_experiment(params)` re-runs a single experiment using a parameters dict (often loaded from `experiment_results.json`), reusing `get_run_hyperparameters(...)` as a baseline and delegating training to `train_model(...)` and evaluation to `evaluate_model_performance(...)`.

- `run_experiments()` performs a Cartesian grid over
  `RESAMPLE_FREQUENCIES`, `TSTEPS_OPTIONS`, `LSTM_UNITS_OPTIONS`, `BATCH_SIZE_OPTIONS`, `N_LSTM_LAYERS_OPTIONS`, `STATEFUL_OPTIONS`, `FEATURES_TO_USE_OPTIONS`, `OPTIMIZER_OPTIONS`, and `LOSS_FUNCTION_OPTIONS` (often patched to smaller sets in tests).

For each combination, it:

1. Calls `convert_minute_to_timeframe(...)` on `RAW_DATA_CSV` to ensure hourly data for the frequency.
2. Calls `train_model(...)` with the exact hyperparameters.
3. Evaluates the model via `evaluate_model_performance(...)` (which loads hourly data and scalers via `src.config`).
4. Appends results (including validation loss, MAE, correlation, and model path) to an in-memory list.
5. Maintains and updates `best_hyperparameters.json` with the best loss per `(frequency, TSTEPS)`.

At the end, it writes `experiment_results.json` and the updated `best_hyperparameters.json`.

#### `src/tune_hyperparameters.py`

An alternative tuning path using KerasTuner:

- Defines `build_tuned_model(hp)` with search spaces for `lstm_units`, `learning_rate`, and `batch_size`.
- `MyTuner(kt.Hyperband)` customizes `run_trial` to:
  - Load pre-built training data from `TRAINING_DATA_CSV`.
  - Truncate data to respect `batch_size` and the effective sequence length computed by `get_effective_data_length(...)`.
  - Build stateful sequences via `create_sequences_for_stateful_lstm(...)`.
- Runs `tuner.search(...)`, then writes best hyperparameters to `best_hyperparameters.json` (a flat dictionary, distinct from the per-frequency structure used by `train.py`/`experiment_runner.py`).

Keep in mind the two formats of `best_hyperparameters.json` when combining tuner output with training/experiments.

#### `src/retrain.py`

Implements a rolling-window retraining strategy based on the latest registry model:

- Loads global best hyperparameters if `best_hyperparameters.json` exists (flat keys like `lstm_units`, `learning_rate`, `batch_size`).
- Prepares feature data from processed hourly CSVs (using `prepare_keras_input_data(...)`).
- Slides a window of `train_window_size + validation_window_size` across the time series and, for each window:
  - Fits a scaler on the train subset and applies it to train/val subsets.
  - Builds sequences via `create_sequences_for_stateful_lstm(...)`, truncating to multiples of `batch_size`.
  - Retrains from the base model’s weights with early stopping.
  - Tracks the best validation loss and saves the corresponding retrained model under `MODEL_REGISTRY_DIR`.

The CLI (`__main__`) exposes window sizes and key hyperparameters as arguments.

---

### Evaluation (`src/evaluate_model.py`)

`evaluate_model_performance(...)` is the central evaluation function and is used both as a standalone script and by `experiment_runner.py`:

- Accepts a model path plus evaluation hyperparameters, including `validation_window_size`, `correction_window_size`, `frequency`, `tsteps`, and `features_to_use`.
- Reloads `src.config` and falls back to `FEATURES_TO_USE_OPTIONS[0]` if `features_to_use` is not provided.
- Loads the stateful model from `model_path` and constructs a non-stateful clone using `build_lstm_model(...)`, transferring weights with `load_stateful_weights_into_non_stateful_model(...)`.
- Loads `nvda_<frequency>.csv` and the associated scaler parameters JSON.
- Prepares a feature-enriched DataFrame via `prepare_keras_input_data(...)`, slices the most recent `validation_window_size` rows, and creates stateless sequences via `create_sequences_for_stateless_lstm(...)`.
- Runs predictions, denormalizes them and the ground truth using stored `Open` mean/std.
- Applies a rolling bias and amplitude correction over a sliding window (`correction_window_size`) to downweight systematic residuals and volatility mis-scaling.
- Computes MAE, MSE, RMSE, and Pearson correlation, and produces a two-panel plot (actual vs corrected predicted prices, and residuals over time).

This is the best place to plug in new metrics or change evaluation windows, as many higher-level flows defer to it.

---

### Prediction and online serving

#### `src/predict.py`

`predict_future_prices(...)` is the core prediction function used by the API and examples:

- Accepts a raw OHLC `DataFrame` with enough history to:
  - Compute the longest rolling technical indicator (e.g. `SMA_21`), and
  - Still retain `tsteps` rows for model input.
- Uses `get_latest_best_model_path(target_frequency, tsteps)` to locate the best model according to `best_hyperparameters.json` and loads it as a stateful model.
- Reads the tuned hyperparameters for the given `(frequency, tsteps)` from `best_hyperparameters.json` (e.g. `lstm_units`, `n_lstm_layers`, `optimizer_name`, `loss_function`).
- Builds a non-stateful clone using `build_lstm_model(...)` and transfers weights.
- Loads the scaler parameters for the frequency via `get_scaler_params_json_path(frequency)`.
- Calls `add_features(...)` and `apply_standard_scaler(...)` to engineer and normalize features, then selects the last `tsteps` rows for prediction.
- Reshapes to `(1, tsteps, n_features)` and runs a forward pass to obtain a single normalized prediction, which is then denormalized using stored `Open` mean/std.

The `__main__` guard demonstrates usage with dummy data and is useful as a template when writing new batch prediction scripts.

#### `api/main.py` (FastAPI)

Defines the public HTTP interface to the model:

- Pydantic models:
  - `OHLCDataPoint` with `Open`, `High`, `Low`, `Close` fields.
  - `PredictionRequest` containing a list of `OHLCDataPoint` objects.

- Endpoints:
  - `GET /health` → `{ "status": "ok" }` (simple liveness check).
  - `POST /predict`:
    - Validates that at least `20 + TSTEPS` OHLC points are provided (20 for feature engineering, TSTEPS for the model input sequence).
    - Builds a `DataFrame` in a fixed column order (`Open`, `High`, `Low`, `Close`).
    - Delegates to `predict_future_prices(...)` from `src.predict`.
    - Reads the active model pointer via `get_active_model_path()`, and extracts a `model_version` from the model filename.

Note that model selection in the API is currently based on `get_active_model_path()` (text file pointer), while `predict.py` itself uses `get_latest_best_model_path(...)`. Keeping these two in sync (e.g. by having `promote_model.py` write the same model chosen by the tuner/experiments) is important for consistent behavior.

`tests/test_api.py` provides good examples of how to call the API with mocked `predict_future_prices`, and how the validation logic behaves with different payloads.

#### `ui/app.py` (Streamlit)

`ui/app.py` is a streamlined Streamlit frontend for interactive use:

- Configures `API_URL = "http://localhost:8000/predict"` and expects the FastAPI app to be running.
- Supports three input modes:
  1. Manual entry of exactly `TSTEPS` rows via `st.data_editor`.
  2. Upload of a CSV with exactly `TSTEPS` rows and `Open,High,Low,Close` columns.
  3. Auto-fetch: reads the last `TSTEPS` rows from `data/processed/nvda_hourly.csv` (with a `Time` column), optionally creating a `DateTime` column for plotting.
- On `"Predict Future Price"` click, posts `{"data": [ ... OHLC dicts ... ]}` to the `/predict` endpoint and displays:
  - Predicted price.
  - Model version from the API response.
  - Optional data source info (time range for auto-fetched data).
- Builds a combined candlestick chart of historical data plus a single predicted point using Altair, showing the last 200 data points plus prediction.

The UI code is a useful reference for how to structure requests and interpret responses from the API.

#### UI architecture (`src/ui/`)

The Streamlit UI uses **horizontal tabs** (not multipage sidebar) for single-page app simplicity:

- `src/ui/app.py`: Main entry point with `st.tabs()`
- `src/ui/page_modules/`: Tab content modules (NOT named `pages/` to prevent Streamlit auto-discovery)
  - `live_page.py`: Monitor live trading sessions
  - `data_page.py`: Ingest and prepare OHLC data
  - `experiments_page.py`: Hyperparameter search
  - `train_page.py`: Full model training
  - `backtest_page.py`: Test and optimize strategies
  - `walkforward_page.py`: Robustness validation
- `src/ui/state.py`: Centralized session state management via `get_ui_state()`
- `src/ui/formatting.py`: Display helpers
- `src/ui/registry.py`: Model registry operations

State is persisted to JSON for history tables and parameter grids. All tabs share the same session state.

---

### Tests and their implications for usage

The `tests/` directory offers concrete examples of expected behavior and usage patterns:

- `tests/conftest.py` ensures `src` and `api` imports work by adding the project root to `sys.path`.
- `tests/test_data_ingestion.py` exercises the IB ingestion flow, verifying IB calls, argument patterns, and CSV writes (using mocks).
- `tests/test_data_processing.py` encodes the expected resampling semantics and feature-engineering behavior, including how `FREQUENCY` is used and how features are selected.
- `tests/test_data_utils.py` defines the expected shapes and properties of sequence generators and scalers.
- `tests/test_experiment_runner.py` shows how experiments are assembled and how `best_hyperparameters.json` and `experiment_results.json` are written.
- `tests/test_model.py` defines the expected LSTM architecture (number of layers, statefulness, activations, optimizer configuration), and verifies that weights can be transferred between stateful/non-stateful variants.
- `tests/test_api.py` documents the API contract: required payload shape, error messages for insufficient length or malformed inputs, and handling of internal errors in `predict_future_prices`.

When changing public APIs, data shapes, or training/evaluation semantics, consult and update the corresponding tests to keep the behavior aligned with these expectations.

---

## Development standards

### Coding principles

- **Simplicity first**: Prefer the simplest implementation that works. Avoid unnecessary abstractions, layers, or configuration unless clearly justified.
- **Minimal diffs**: Default to minimal diffs instead of large rewrites when fixing bugs or adding features.
- **Explicit over implicit**: Avoid designs that rely on subtle global state, cross-module side effects, or tight coupling that isn't obvious from function signatures.
- **Technical debt tracking**: When introducing complexity (deeper nesting, more states, non-obvious invariants), document it in `docs/technical_debt.md` with suggested follow-up refactors.

### Before making changes

- **Think before acting**: Consider if a fix requires a larger, more general refactoring. Measure benefits vs cost (labor and complexity).
- **Suggest improvements**: When possible, propose enhancements and wait for confirmation before executing.
- **Test coverage**: For every non-trivial change, generate corresponding tests covering edge cases, negative paths, and regression scenarios.

### Testing

All tests live under `tests/` and can be run via:

```bash
pytest                          # All tests
pytest tests/test_model.py      # Single test file
pytest tests/test_model.py::test_build_lstm_model_architecture_single_layer_stateful  # Single test
```

Test files mirror `src/` structure and use fixtures from `tests/conftest.py` for path setup and mocking.

### IBKR configuration

Interactive Brokers connection settings in `src/config.py` via `IbConfig`:

- **IB Gateway paper trading** (default): `host=127.0.0.1`, `port=4002`
- **TWS paper trading**: `host=127.0.0.1`, `port=7497`
- **TWS live trading**: `host=127.0.0.1`, `port=7496`

Override via environment variables:
```bash
export TWS_HOST=127.0.0.1
export TWS_PORT=4002
export TWS_CLIENT_ID=1
export IBKR_ACCOUNT=U16442949  # Production account ID
```

Available production accounts (from `src/config.py`):
- `U16442949`: Robots (default)
- `U16452783`: Kids
- `U16485076`: AI
- `U16835894`: M7
- `U22210084`: ChinaTech

Ensure IB Gateway/TWS is running before executing ingestion or live trading commands.

### Common pitfalls

- **Model selection inconsistency**: The API uses `get_active_model_path()` (reading `models/active_model.txt`), while `predict.py` uses `get_latest_best_model_path(...)` from `best_hyperparameters.json`. Keep these in sync via `promote_model.py`.
- **Stateful vs non-stateful**: Training requires stateful models; inference requires non-stateful clones. Always use `load_stateful_weights_into_non_stateful_model(...)` when loading for prediction.
- **Scaler mismatch**: Predictions fail silently if scaler parameters don't match the frequency. Ensure `scaler_params_<frequency>.json` exists and was generated from the same data used for training.
- **Insufficient history**: The API requires `20 + TSTEPS` data points (20 for feature engineering, TSTEPS for model input). Always validate input length.
- **ATR calculation**: Use per-bar ATR series (`_compute_atr_series(data, window=14)`) for strategy filters, not scalar approximations. The backtest engine and live trading both expect per-bar series.

---

## Trading system & backtesting (roadmap)

A trading system for this repository is planned as a **separate layer** on top of the existing prediction API and data pipeline. The goal is to turn price predictions into executable trading strategies that can be validated thoroughly in backtests before any live or semi-live trading.

High-level phases:

1. **Phase 0 – Offline backtesting (local, no AWS)**
   - Build a simple backtesting engine that consumes:
     - Historical OHLCV from `data/processed/nvda_*.csv`.
     - Model predictions via `src.predict.predict_future_prices` or saved predictions.
   - Implement one or two basic strategies (e.g., long/flat based on predicted next-bar return, with fixed stop loss / take profit).
   - CLI (to be implemented) should allow running many parameterized backtests and writing PnL/metrics to a results file.

2. **Phase 1 – Local paper trading (no AWS, broker-simulated)**
   - Use live or near-live data from IB/TWS, but keep all order execution simulated locally.
   - Reuse the same strategy and risk rules as in Phase 0; reuse the backtester components for intraday evaluation.

3. **Phase 2 – Optional AWS-based components**
   - Once strategies pass backtesting and paper-trading criteria, introduce AWS services for:
     - Hosting the prediction API (e.g., containerized FastAPI on ECS/Fargate or similar).
     - Running scheduled or event-driven strategy evaluation jobs.
     - Persisting trades, signals, and performance metrics in a managed database.
   - Live trading integration with a broker should remain behind clear feature flags and be guarded by risk limits.

Design documentation for the trading system lives under:

- `docs/trading system/trading_system_requirements.md`
- `docs/trading system/trading_system_hld.md`
- `docs/trading system/trading_system_test_strategy.md`
- `docs/trading system/trading_system_runbook.md`
- `docs/trading system/trading_system_migration_plan.md`

These documents mirror the structure of the data ingestion docs and should be kept in sync with any future trading-related code you add (backtesting engine, strategy modules, broker adapters, AWS deployment scripts).

---

### ATR, backtest vs. paper-trade, and strategy consistency

The trading system and backtesting components make heavy use of Average True Range (ATR) and share a common engine:

- **Per-bar ATR(14) series is the canonical volatility measure.**
  - `src.backtest._compute_atr_series(data, window=14)` computes an ATR time series on the active timeframe (e.g. 14 hourly bars).
  - Both the backtest CLI (`src/backtest.py`) and paper-trading scaffold (`src/paper_trade.py`) compute this per-bar ATR series and pass it into the core engine.
  - The ATR series is used both as:
    - `atr_series` (for ATR-based entry filters and position sizing), and
    - `model_error_sigma_series` (currently a proxy until true residual-based sigmas are wired).

- **Scalar ATR-like values are secondary and derived.**
  - In a few places we compute a scalar `atr_like` as the mean of the non-NaN ATR series.
  - This scalar is passed into `BacktestConfig` for backwards compatibility, but the actual trading logic now prefers the full per-bar ATR via `atr_series`.
  - When making design decisions, treat the per-bar ATR series as the source of truth for volatility-aware filters and sizing.

- **Backtest and paper-trade now share the same engine.**
  - The core trading loop lives in `src.backtest_engine.run_backtest`, which takes:
    - `data` (OHLC with `Open, High, Low, Close` and optionally `Time`),
    - a `prediction_provider` callable,
    - a `BacktestConfig` (containing `StrategyConfig` and commissions),
    - optional `atr_series` and `model_error_sigma_series`.
  - The backtest CLI (`python -m src.backtest`) constructs `BacktestConfig`, computes ATR(14), and calls `run_backtest` with both scalar and per-bar ATR inputs.
  - The paper-trade helper `run_paper_trading_over_dataframe(...)` in `src/paper_trade.py` now:
    - Computes the same per-bar ATR(14) series,
    - Builds a `BacktestConfig` via `PaperTradingConfig` (mirroring `BacktestConfig`),
    - Constructs a CSV-based `prediction_provider`, and
    - Delegates to `run_backtest(...)` with `atr_series` and `model_error_sigma_series` set to the ATR series.
  - As a result, backtests and paper-trade runs use identical entry/exit rules, commission logic, and volatility handling; differences in equity should come from inputs (data, prediction mode, date range), not from duplicated logic.

- **Date ranges are explicit and shared.**
  - `BacktestWindowConfig` in `src/config.py` exposes `default_start_date` / `default_end_date` (`BACKTEST_DEFAULT_START_DATE` / `BACKTEST_DEFAULT_END_DATE`), which default to `None` (use full data range).
  - `_apply_date_range(data, start_date, end_date)` in `src/backtest.py` slices a `Time` column to a `[start_date, end_date]` window and returns `(sliced_data, date_from_str, date_to_str)`.
  - The backtest CLI accepts `--start-date` / `--end-date` (YYYY-MM-DD) and applies `_apply_date_range` before running the engine.
  - `src/paper_trade.py` uses the same helper and CLI options, so both tools can be aligned for walk-forward or specific test windows.

- **Equity plot filenames encode strategy + window.**
  - `_plot_price_and_equity_with_trades(...)` in `src/backtest.py` saves figures under `backtests/` with filenames:
    - `[SYMBOL]-[FREQ]-[k_sigma_err]-[k_atr_min_tp]-[risk_per_trade_pct]-[reward_risk_ratio]-[DateFrom]-[DateTo].png`.
  - Both the backtest and paper-trade CLIs call this helper with their effective frequency, strategy parameters, and the sliced data, so plots are directly comparable across runs and walk-forward slices.

When evolving the trading system (e.g., introducing residual-based sigma series instead of ATR proxies, or adding new filters), prefer to:

1. Extend `StrategyConfig` / `StrategyDefaultsConfig` in `src/config.py`.
2. Wire new per-bar series into `run_backtest` via additional optional parameters.
3. Keep `src/backtest.py` and `src/paper_trade.py` as *thin* orchestration layers that share this common engine, rather than duplicating trading logic.
