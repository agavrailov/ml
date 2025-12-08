# Data Pipeline & Predictions Research

_Last updated: 2025-12-08_

## 1. Goal

Have a **clean, aligned, reliable, and robust** data and prediction pipeline across all stages:

- Raw minute → cleaned + gap-handled
- Resampled hourly (15min/60min)
- Feature frames used by the LSTM
- Model predictions (standalone CSVs and backtest checkpoints)
- Backtests / paper trading

## 2. Key findings about the data pipeline

### 2.1. Raw and hourly data

- `src.data_processing.convert_minute_to_timeframe` correctly resamples minute → hourly OHLC.
  - Verified by unit test `tests/test_data_processing.py::test_convert_minute_to_timeframe` using synthetic data.
- New invariants are enforced by `src.data_quality` + `src.data`:
  - **Hourly OHLC** (`load_hourly_ohlc`):
    - Columns: `Time, Open, High, Low, Close`.
    - `Time` is datetime, strictly increasing, unique.
    - No NaNs in OHLC.
  - **Feature frames** (`load_hourly_features`):
    - `Time` present and datetime-like.
    - Requested feature columns exist.
    - No NaNs in feature columns.
- These invariants are applied everywhere that goes through `src.data`:
  - Training (`src.train`), evaluation (`src.evaluate_model`), backtests (`src.backtest`), UI (`src/app`), experiment runner, and now `src.paper_trade`.

### 2.2. Raw data quality

- `src.data_quality.analyze_raw_minute_data` performs structural/time-coverage/validity checks on `RAW_DATA_CSV`.
- `compute_quality_kpi` aggregates pass/warn/fail into a 0–100 score.
- `daily_data_agent.run_daily_pipeline` now:
  - Cleans raw data.
  - Analyzes and fills gaps (`smart_fill_gaps`).
  - Logs a **data quality snapshot** (score + counts) for raw minute data.

### 2.3. Standalone prediction pipeline

Path:

> hourly OHLC → features → scaler → LSTM → predictions → CSV

Implemented by `scripts/generate_predictions_csv.py`:

- Uses `build_prediction_context` and `predict_sequence_batch` (same core as evaluation).
- Produces **per-bar predictions CSVs**:
  - `backtests/nvda_60min_predictions.csv`
  - `backtests/nvda_15min_predictions.csv`
- Alignment and error audits via `scripts/audit_predictions_vs_price.py`:
  - For both 60min and 15min:
    - One-to-one alignment on `Time` between hourly OHLC and predictions.
    - No structural mismatches besides the expected tail where targets are undefined.
    - MAE/RMSE vs `Open[t+ROWS_AHEAD]` are reasonable and change smoothly over years.
  - Example (order of magnitude):
    - 60min:
      - 2023: MAE ~0.14
      - 2024: MAE ~0.51
      - 2025: MAE ~0.64
    - 15min:
      - 2023: MAE ~0.17
      - 2024: MAE ~0.35
      - 2025: MAE ~0.36

Conclusion: **The standalone prediction pipeline is clean, aligned, and numerically sane.**

### 2.4. Checkpoint predictions (model-mode backtest path)

Historically, `_make_model_prediction_provider` in `src.backtest`:

- Builds features + scaling on the fly.
- Runs the LSTM model.
- Applies bias-only correction in price space.
- Writes per-bar checkpoint CSVs used by model-mode backtests:
  - `backtests/nvda_<freq>_model_predictions_checkpoint.csv`.

Audit via `scripts/audit_checkpoint_vs_predictions.py` (before checkpoint files were removed) showed:

- **Large duplicate Time rows** in checkpoint CSVs (222 for 60min, 7,322 for 15min).
- After de-duplication, alignment by `Time` was OK, but:
  - Checkpoint predictions deviated massively from both targets and standalone predictions, especially in 2024–2025.
  - For 60min (examples):
    - 2023: ckpt MAE ~0.53 vs target (vs standalone ~0.14).
    - 2024: ckpt MAE ~4.8 (vs standalone ~0.5).
    - 2025: ckpt MAE ~9.2 (vs standalone ~0.64).
  - For 15min:
    - 2022: ckpt ≈ standalone ≈ target (healthy).
    - 2023+: ckpt MAE grows into **tens of dollars**, while standalone stays <1.

These numbers indicate a **systemic distortion** in the checkpoint path (likely in correction / residual handling) rather than in the core model or data.

Later, checkpoint files were removed, and attempts to re-audit now fail with `FileNotFoundError` (no ckpt CSVs present anymore).

Conclusion: the **checkpoint/model-mode path was not reliable** and is currently disabled in practice.


## 3. Tests and tools added

- `tests/test_data_quality.py`:
  - Covers raw-minute analysis + KPI.
  - Tests `validate_hourly_ohlc` and `validate_feature_frame` for happy-path and error conditions.
- `tests/test_data_module.py`:
  - Ensures `src.data.load_hourly_ohlc` and `load_hourly_features` delegate correctly and enforce invariants.
- `scripts/audit_predictions_vs_price.py`:
  - Joins processed OHLC and standalone prediction CSV by `Time`.
  - Computes MAE/RMSE overall and per year.
  - Runs a simple time-shift correlation check across small lags.
- `scripts/audit_checkpoint_vs_predictions.py`:
  - Joins processed OHLC, checkpoint predictions, and standalone predictions.
  - Computes MAE/RMSE vs targets and MAE/RMSE of checkpoint vs standalone, per year.
- `src/paper_trade.py`:
  - Now uses `load_hourly_ohlc` when no custom CSV path is provided, so hourly invariants are enforced here as well.

These tests/tools give a consistent view of **data quality and prediction alignment** across the pipeline.


## 4. CSV mode vs model mode

### 4.1. CSV mode (precomputed predictions)

**What it is**

- Run `scripts.generate_predictions_csv.py` to produce `Time, predicted_price` for each bar.
- Backtests and paper-trade read these CSVs as the prediction source.

**Pros for experiments/backtests**

- **Simple & robust**:
  - Single well-tested path from hourly OHLC through features/scaler to predictions.
- **Fast & reproducible**:
  - Predictions are computed once; backtests can be re-run many times cheaply.
  - Same predictions CSV → same backtest outcomes.
- **Good for strategy experiments**:
  - Change strategy parameters, risk, filters, etc. without recomputing model output.

**Cons**

- Slightly less “live-like”:
  - Backtests don’t exercise feature/scaler/model code *inside* the backtest loop.
- Requires discipline:
  - Must regenerate CSV after retraining/promoting models or changing features.
  - Need file/version bookkeeping.

### 4.2. Model mode (on-the-fly predictions)

**What it is**

- Backtest pipeline computes features + scaling + `model.predict` per bar in memory.
- Optionally applies bias-correction and computes residual-sigma.

**Pros**

- **Closer to real-time behaviour**:
  - Simulates the pipeline a live system would run bar-by-bar.
- Useful for **pipeline validation experiments**:
  - Does online feature/scaler logic match training?
  - Does bias correction / residual sigma help?

**Cons (as observed)**

- Complex and fragile:
  - Harder to reason about and debug when things drift.
  - Historical checkpoint path produced extremely large errors in 2024–2025.
- Slower for experiments:
  - Recomputes predictions each run.
- Additional moving parts (correction, sigma, checkpoint writing) introduce more failure modes.


## 5. Recommendation: CSV as default mode

Given the actual behaviour observed and priorities:

- **Simplicity**
- **Reliable, aligned data**
- **Robust pipeline**
- **Still wanting realistic model-mode backtests (eventually)**

The recommended architecture is:

1. **CSV predictions as the primary mode for backtests and experiments.**
   - Main backtest and paper-trading entrypoints should default to using `backtests/nvda_<freq>_predictions.csv`.
   - Strategy experiments (parameter sweeps, risk tweaks) should run on top of precomputed predictions.

2. **Model mode as an experimental / validation path.**
   - Keep model-mode implementation, but:
     - Treat it as *experimental* until its outputs are proven numerically close to CSV predictions.
     - Avoid depending on checkpoint CSVs for production decisions.

3. **Use the audit scripts as trust gates.**
   - `audit_predictions_vs_price.py` remains the main check for prediction/data alignment.
   - `audit_checkpoint_vs_predictions.py` is used when re-enabling model-mode to:
     - Verify that checkpoint MAE vs target ≈ standalone MAE.
     - Verify that MAE(checkpoint – standalone) is small across years.
     - Verify no pathological growth in error in later years.


## 6. How to use CSV mode in practice

1. **Generate predictions CSVs** (once per model/version):

   ```bash
   # 60min
   python -m scripts.generate_predictions_csv \
       --frequency 60min \
       --output backtests/nvda_60min_predictions.csv

   # 15min
   python -m scripts.generate_predictions_csv \
       --frequency 15min \
       --output backtests/nvda_15min_predictions.csv
   ```

2. **Run audits to confirm sanity**:

   ```bash
   python -m scripts.audit_predictions_vs_price --frequency 60min
   python -m scripts.audit_predictions_vs_price --frequency 15min
   ```

   - Check that MAE/RMSE are in expected ranges and change smoothly over years.

3. **Use CSV in backtests / paper trade**:

   - Backtest:
     - (Desired future default) a mode where `src.backtest` loads `nvda_<freq>_predictions.csv` and uses those predictions instead of model-mode.
   - Paper trade:
     - `src.paper_trade` already accepts `--predictions-csv` and uses CSV-based predictions.


## 7. How to safely (re)switch to model mode

When/if you want to use on-the-fly model-mode backtests again:

1. **Refactor prediction logic to share one core function**

   - Extract a helper such as `compute_predictions_on_dataframe(df, frequency, tsteps)` that:
     - Builds `PredictionContext`.
     - Runs `add_features` + scaler.
     - Returns `Time, predicted_price` series.
   - Use this in **both**:
     - `scripts.generate_predictions_csv`.
     - `_make_model_prediction_provider` in `src.backtest`.

   This guarantees that CSV and model-mode share the same prediction engine.

2. **Disable or simplify bias correction initially**

   - Start with **no correction** (or a very conservative bias-only correction) in model-mode.
   - Only introduce more complex correction/residual logic after it is shown to improve MAE in controlled experiments.

3. **Re-enable checkpoint CSVs and compare to CSV predictions**

   - Run a model-mode backtest that writes a fresh checkpoint CSV, e.g.:
     - `backtests/nvda_<freq>_model_predictions_checkpoint.csv`.
   - Run:

     ```bash
     python -m scripts.audit_checkpoint_vs_predictions --frequency 60min
     python -m scripts.audit_checkpoint_vs_predictions --frequency 15min
     ```

   - Only accept model-mode as “trusted” if:
     - Checkpoint MAE vs target is close to CSV MAE vs target for all important years.
     - MAE(checkpoint – standalone) is small.
     - Errors do not blow up in 2024–2025 or other future windows.

4. **Keep CSV mode as a baseline**

   - Even after model-mode is trusted, keep CSV mode and audits as a baseline reference.
   - When changing model architecture, features, or correction logic:
     - Regenerate CSVs.
     - Re-run both audits.
     - Confirm model-mode remains within tolerance of CSV behaviour.


## 8. TODO / Follow-ups

1. **Backtest defaults**
   - [ ] Update `src.backtest` CLI to support a clear CSV-based prediction mode (if not already) and make it the default for routine experiments.
   - [ ] Document the CSV-mode workflow in the main project README / runbooks.

2. **Model-mode refactor**
   - [ ] Extract a shared `compute_predictions_on_dataframe` used by both `generate_predictions_csv` and `_make_model_prediction_provider`.
   - [ ] Add an optional debug mode to `_make_model_prediction_provider` that dumps pre/post-correction predictions and residuals for a chosen window.

3. **Checkpoint hygiene**
   - [ ] When reintroducing checkpoint CSVs, ensure they are written fresh per-run or de-duplicated on `Time` before saving.

4. **Continuous validation**
   - [ ] Integrate `audit_predictions_vs_price` (and conditionally `audit_checkpoint_vs_predictions`) into a small CI or pre-release check for model changes.
   - [ ] Periodically run audits focused on recent months/years to catch regime-related drift early.
