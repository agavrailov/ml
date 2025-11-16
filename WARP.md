# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

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

- Start Streamlit UI (talks to the API on port 8000):

  ```bash
  streamlit run ui/app.py
  ```

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
  python src/data_ingestion.py
  ```

- Daily data pipeline agent that chains ingestion, cleaning, gap analysis/filling, and resampling/feature engineering:

  ```bash
  python src/daily_data_agent.py
  ```

  Add `--skip-ingestion` for dry runs that operate only on existing raw data.

- Alternative historical data updater (more focused on incremental update & cleanup):

  ```bash
  python src/data_updater.py
  ```

- Windows helper script that runs the data updater and then hourly processing:

  ```bat
  update_data.bat
  ```

  This calls `python src/data_updater.py` and `python src/data_processing.py` in sequence.

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

#### `src/data_ingestion.py`

This module handles asynchronous ingestion of historical data from IB/TWS into `RAW_DATA_CSV` (default `data/raw/nvda_minute.csv`). Key pieces:

- Uses `ib_insync` and IB’s `reqHistoricalDataAsync` to fetch OHLC bars.
- `_get_latest_timestamp_from_csv(...)` reads the last `DateTime` from the existing CSV to resume from where the previous run left off.
- `fetch_historical_data(...)`:
  - Connects to TWS using `TWS_HOST`, `TWS_PORT`, `TWS_CLIENT_ID` and builds a `Stock`/`Forex`/`CFD`/`Future` contract based on `NVDA_CONTRACT_DETAILS`.
  - Computes an effective concurrency level (`effective_concurrency`) based on IB pacing rules and `TWS_MAX_CONCURRENT_REQUESTS`.
  - Walks backwards in daily chunks from `end_date` to `start_date` (or to the last CSV timestamp if `strict_range=False`), requesting 1-day windows and aggregating them in batches before writing to CSV.
  - Writes a canonical `DateTime,Open,High,Low,Close` CSV, ensuring chronological order and limited concurrency.

This module is heavily exercised by `tests/test_data_ingestion.py`, which mocks IB and `pandas` utilities to assert call patterns and CSV writes.

#### `src/data_updater.py`

`update_historical_data()` orchestrates an incremental historical update:

- Loads existing `RAW_DATA_CSV` into a time-indexed DataFrame, deduplicating by `DateTime`.
- Uses `_get_latest_timestamp_from_csv(...)` to determine the new fetch start time.
- Calls `fetch_historical_data(...)` to append new minute bars up to `now`.
- After fetching, reloads the full dataset and performs a final sort/deduplication via `clean_raw_minute_data(...)` from `src.data_processing`.
- Contains (currently commented-out) logic for detailed exchange-calendar-based gap filling using `exchange_calendars` and market hours from `src.config`.

`update_data.bat` is built around this script, followed by a processing step.

#### `src/daily_data_agent.py`

This module is the high-level “daily pipeline agent” that ties multiple tools together:

- `run_gap_analysis()` shell-executes `analyze_gaps.py` on `RAW_DATA_CSV`, writing gap metadata to `GAP_ANALYSIS_OUTPUT_JSON` (under `PROCESSED_DATA_DIR`).
- `smart_fill_gaps()` reloads the raw CSV, applies `fill_gaps(...)` from `src.data_processing` using gap metadata, and writes a gap-filled CSV back.
- `ingest_new_data()` wraps `fetch_historical_data(...)` to pull new minute data since a default start date.
- `resample_and_add_features()`:
  - Ensures `PROCESSED_DATA_DIR` exists.
  - Calls `convert_minute_to_timeframe(...)` to resample minute NVDA data to the configured `FREQUENCY` (e.g. `15min`, `60min`).
  - Uses `FEATURES_TO_USE_OPTIONS[0]` and `prepare_keras_input_data(...)` to engineer features for the configured frequency.
- `run_daily_pipeline(skip_ingestion=False)` wires everything into one sequence:
  1. Optionally ingest new raw data from IB.
  2. Clean and deduplicate raw minute data.
  3. Run gap analysis and fill small weekday gaps.
  4. Resample and engineer features for training.

This is the best entry point for a once-per-day maintenance job.

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
