# Model / Pipeline Refactor Plan

Scope: **unify core training–prediction–backtest pipeline** while keeping notebooks and existing CLI flows working.

- **Do not change**: notebooks, existing CLI entry points or user-facing scripts.
- **Change only**: internal modules in `src/` and their dependencies.
- **Guiding principles**:
  - Prefer minimal diffs over big rewrites.
  - Introduce structure only when it removes real pain (duplication, ambiguity).
  - Add tests for every non‑trivial behavior change.

Core production pipeline today:

1. `nvda_minute.csv` → resample → `nvda_15min.csv`
2. features + log-returns → LSTM → `model.keras`
3. model → log-return predictions → price predictions
4. predictions + strategy params → equity curve

---

## Milestone 1 – Unified Model API + Minimal Config Core

**Goal:** There is exactly **one** way to build, train, save, and load the LSTM model. Notebooks/CLI keep using existing scripts, but those scripts delegate into this unified layer.

**Status:** Implemented.
- `ModelConfig` + `get_model_config` added in `src.config`.
- `build_model`, `load_model`, and `train_and_save_model` implemented in `src.model`.
- All real code paths load models via `src.model.load_model`.
- `train.py` delegates fit+save to `train_and_save_model` while keeping data prep and bias-correction logic local.

### 1.1 Create `src/model.py` (new module)

**Responsibilities:**
- Own model construction, training, saving, loading, and prediction.

**Target public API (first pass):**
- `build_model(model_cfg) -> tf.keras.Model`
- `train_model(train_data, model_cfg, *, model_path=None) -> Path`
- `load_model(model_cfg, model_path: Path | None = None) -> tf.keras.Model`
- `predict_log_returns(model, features) -> np.ndarray`

**Notes:**
- Keep inputs simple: `model_cfg` only needs what the model truly requires (e.g. input size, TSTEPS, layers, etc.).
- Any required paths can come from a very small config helper (see 1.2).

### 1.2 Introduce Minimal `ModelConfig` in `src/config.py`

**Goal:** Avoid full config overhaul. Add just enough structure to support `model.py`.

Steps:
1. Define a `ModelConfig` dataclass with only the fields needed by `model.py` (e.g. `tsteps`, input dimension, model save path, etc.).
2. Implement a helper function, e.g. `get_model_config()` that returns a `ModelConfig` instance based on existing module-level constants.
3. Keep all existing config globals and aliases **unchanged** for now (backwards compatible).

### 1.3 Refactor Training Entry Points to Use `model.py`

**Goal:** Every training pathway goes through `model.train_model`.

Steps:
1. Identify all training scripts/modules (e.g. `train.py`, `retrain.py`, `promote_model.py`, etc.).
2. For each:
   - Replace inline model-building / training logic with calls to `build_model` / `train_model`.
   - Keep command-line behavior and arguments the same.
3. Keep data preparation as-is for now; only the model interaction moves into `model.py`.

Success criteria:
- There is exactly **one** place where `tf.keras.models.load_model` is called: in `src/model.py`.
- Training scripts no longer call Keras APIs directly; they use `model.py`.

### 1.4 Refactor Prediction & Backtest to Use `model.load_model`

**Goal:** Predictions and backtests stop loading the model in ad-hoc ways.

Steps:
1. Identify modules that load/instantiate the model for prediction (e.g. `predict.py`, `backtest.py`, any helpers).
2. Replace direct `load_model` / model-building code with calls to `model.load_model` and `model.predict_log_returns`.
3. Preserve all public behavior (CLI flags, notebook-facing functions, output formats).

Success criteria:
- There is exactly **one** place that knows how to construct & load the model.
- Running the existing prediction and backtest scripts yields equivalent outputs (within numerical tolerance).

### 1.5 Tests for Milestone 1

Add or update tests to cover:

- **Model round-trip:**
  - Build → train on a tiny toy dataset → save → load → predict.
  - Predictions from saved+loaded model match predictions from the in-memory model (within small tolerance).
- **Config consistency:**
  - `ModelConfig` fields match current expectations (e.g. `TSTEPS` stays consistent with model input shape).
- **End-to-end smoke:**
  - Run a short training + prediction + backtest on a small fixed subset of data and assert that key metrics (e.g. number of trades, P&L sign) are unchanged vs. pre-refactor baselines.

Only after Milestone 1 is stable and tested do we proceed.

---

## Milestone 2 – Data Pipeline Consolidation (`src/data.py`)

**Goal:** Make the data path into the model explicit and centralized, without changing notebooks/CLI.

**Status:** Implemented (within current scope).
- `src.data.load_hourly_ohlc` and `src.data.load_hourly_features` wrap the existing
  CSV + feature-engineering pipeline.
- `train.py`, `evaluate_model.py`, and the `backtest` CLI now consume hourly OHLC
  and feature frames exclusively via `src.data` for the default NVDA path.
- Resampling from minute → hourly remains in `src.data_processing.convert_minute_to_timeframe`,
  invoked by the existing data-processing script; this is already a single entrypoint
  and does not need further abstraction for now.

### 2.1 Create `src/data.py`

**Responsibilities:**
- Loading raw price data.
- Resampling (e.g. minute → 15min).
- Feature engineering and log-return computation.

**Initial public API (keep minimal):**
- `load_resampled_prices(symbol, frequency) -> DataFrame`
- `make_features(prices_df, model_cfg) -> (X, y | None)`

Notes:
- Internally, `data.py` can use existing helpers and config; do not over-engineer.
- Do **not** attempt a large config refactor here; it should read from existing config wherever possible.

### 2.2 Migrate Core Training Path to Use `data.py`

Steps:
1. In training scripts, replace scattered loading/resampling/feature logic with calls to `data.load_resampled_prices` and `data.make_features`.
2. Keep behavior (symbol, frequency, date ranges) identical.

Success criteria:
- There is a single, obvious code path from raw CSV → features used by the model.
- All training scripts pull features from `data.py` APIs.

### 2.3 Migrate Prediction Path to Use `data.py`

Steps:
1. Update prediction scripts to call `data.load_resampled_prices` and `data.make_features` instead of duplicating logic.
2. Ensure feature alignment is unchanged (no off-by-one errors between features & targets).

### 2.4 Tests for Milestone 2

Add or update tests:

- **Resampling test:**
  - Given a small synthetic minute-level price series, `load_resampled_prices` produces the expected 15min bars.
- **Feature shape + alignment:**
  - `make_features` returns arrays of expected shapes; last `X` aligns with last `y` where applicable.
- **Regression test:**
  - For a fixed dataset snapshot, feature arrays match pre-refactor versions (within numerical tolerance).

---

## Milestone 3 – Strategy & Backtest Cleanup (`src/strategy.py`)

**Goal:** Isolate strategy logic and parameters from backtest plumbing, while leaving entry points intact.

### 3.1 Create `src/strategy.py`

**Responsibilities:**
- Strategy parameters and position-sizing rules.
- Converting predictions + prices into trades.
- Computing equity curves.

**Initial public API:**
- `StrategyParams` dataclass with:
  - `k_sigma_err`
  - `k_atr_min_tp`
  - `reward_risk_ratio`
  - `risk_per_trade_pct`
- Core functions:
  - `generate_trades(price_series, predictions, params: StrategyParams) -> trades_df`
  - `compute_equity_curve(trades_df, initial_capital, ...) -> equity_df`

### 3.2 Refactor `backtest.py` to Use `strategy.py`

Steps:
1. Move signal-generation and TP/SL logic from `backtest.py` into `strategy.py`.
2. Keep `backtest.py` as a thin orchestrator that:
   - Gets prices + predictions.
   - Constructs `StrategyParams` (possibly from existing config/global values).
   - Calls `generate_trades` and `compute_equity_curve`.
3. Preserve CLI behavior and outputs.

Success criteria:
- Strategy rules and parameters are clearly visible in `strategy.py`.
- `backtest.py` is primarily wiring/orchestration code.

### 3.3 Tests for Milestone 3

Add or update tests:

- **Deterministic strategy test:**
  - For a small, synthetic price and prediction series, `generate_trades` produces a known set of trades.
- **Equity curve invariants:**
  - Equity curve respects position sizing and reward/risk constraints.
- **Regression test:**
  - For a fixed historical slice and seed, key backtest metrics (e.g. trade count, P&L, max drawdown) stay within acceptable bounds compared to pre-refactor.

---

## Milestone 4 – Config Cleanup (Optional, After Stability)

**Goal:** Reduce config duplication once core modules use structured configs internally. This step is optional and should only be started after the system is stable under the new architecture.

### 4.1 Inventory of Config Usage

Steps:
1. Search for uses of legacy config globals and aliases across the codebase.
2. Mark which ones are:
   - Used only in internal modules (`data.py`, `model.py`, `strategy.py`).
   - Exposed via notebooks/CLI.

### 4.2 Gradual Migration to Dataclasses

Steps:
1. For internal-only settings, switch call sites to use dataclass-backed config objects (`ModelConfig`, `StrategyParams`, etc.).
2. Keep legacy globals as shims for any notebooks/CLI still importing them.

### 4.3 Optional Alias Removal

Only when **all** known call sites (including notebooks) have been updated:
- Remove obsolete aliases from `config.py`.
- Keep a short migration note in this file (or in README) explaining removed names and their replacements.

---

## Out of Scope for This Plan

To keep this refactor bounded and safe, **we explicitly do not**:

- Change any notebook logic or structure.
- Change user-facing CLI interfaces or script arguments.
- Introduce new configuration systems (e.g. YAML/JSON configs) beyond minimal dataclasses.
- Perform large rewrites of modules unrelated to the core pipeline.

---

## How to Use This Plan

1. Work milestone by milestone; do not start Milestone 2 until Milestone 1 is stable and tested.
2. Within a milestone, keep PRs / change sets small and focused (e.g. "Introduce model.py and update predict/backtest only").
3. After each milestone, run the full existing workflow (ingestion → train → predict → backtest) and compare key outputs to ensure no regressions.
4. Update this document if real-world constraints require deviations from the plan.