# Walk-forward backtesting and live training policy

This document defines how we split time into training/validation/test windows for **walk-forward backtests** and how we mirror that logic in **live retraining**. The goal is:

- No information leak from the test period into model training or hyperparameter selection.
- A simple, well-defined schedule that can be implemented in notebooks/CLIs.
- 3‑month test windows, each with its own preceding training period.

Throughout, "Time" refers to the `Time` column in the resampled OHLC CSVs (e.g. `nvda_15min.csv`, `nvda_60min.csv`). The policy applies to any resample frequency.

## 1. Data horizon and notation

Let:

- `T_start` = earliest timestamp available in the resampled OHLC data we are willing to use for training.
  - Currently we **hard-cut** training data to `Time >= 2023‑01‑01` in `src/train.py`. This can be adjusted later, but the policy stays the same.
- `T_end_hist` = last timestamp available in historical data.

We define a sequence of **non‑overlapping 3‑month test windows**:

- Test window k: `[Test_k_start, Test_k_end)` with length ≈ 3 months.
- Each test window has its own **training window**:
  - `Train_k_start = max(T_start, Test_k_start − Train_lookback)`
  - `Train_k_end = Test_k_start` (strictly before the test period)

Where:

- `Train_lookback` = **18–24 months** (rolling 1.5–2‑year window) by default.
  - If we do not have a full 24 months available before `Test_k_start`, we use whatever is available starting from `T_start`.

Within each training window we further split into **train** and **validation** segments.

## 2. Walk-forward backtest policy (offline)

We construct a sequence of folds `{k = 1..K}`, each with:

- A **training+validation window** (used for fitting and hyperparameter selection).
- A **test window** (used only for trading backtest evaluation).

### 2.1. Window shapes per fold

For each fold k:

1. **Choose test window**
   - Let `Test_1_start` be the first test start date we want to evaluate, e.g. `2024‑01‑01`.
   - Define `Test_k_start = Test_1_start + (k − 1) * 3 months`.
   - Define `Test_k_end = Test_k_start + 3 months` (or clipped to `T_end_hist`).

2. **Define training window**
   - `Train_k_end = Test_k_start`.
   - `Train_k_start = max(T_start, Train_k_end − 24 months)`.

3. **Split training window into train / validation**
   - Training+validation pool: `[Train_k_start, Train_k_end)`.
   - Use a simple chronological split inside that range, e.g.:
     - **Train**: first 80% of timestamps in `[Train_k_start, Train_k_end)`.
     - **Validation**: last 20% of timestamps in `[Train_k_start, Train_k_end)`.
   - This is equivalent to the current `TR_SPLIT` logic, but applied only to rows that fall inside the window.

### 2.2. Procedure per fold

For each fold k:

1. **Extract data**
   - From the resampled CSV (15min / 60min), filter rows by `Time` to get:
     - Training+validation frame: `TrainVal_k = {rows: Train_k_start <= Time < Train_k_end}`.
     - Test frame: `Test_k = {rows: Test_k_start <= Time < Test_k_end}`.

2. **Train models on `TrainVal_k` only**
   - Run `train_model` restricted to `TrainVal_k` (not the whole 2023+ data):
     - Apply internal train/validation split via `TR_SPLIT` **within this window**.
     - Use validation metrics (e.g. validation loss, MAE, correlation) to select hyperparameters or decide promotion.
   - Save the trained model and its `bias_correction_*.json` into `MODEL_REGISTRY_DIR`.

3. **Generate predictions only for `Test_k`**
   - Using the promoted model for this fold, run the unified prediction pipeline on `Test_k`.
   - Write per‑bar predictions CSV (or an in‑memory DataFrame) aligned to `Test_k`.

4. **Run backtest on `Test_k` only**
   - Call `run_backtest_on_dataframe(Test_k, prediction_mode="csv" or "model")`.
   - Collect metrics (CAGR, Sharpe, max drawdown, win rate, etc.).

5. **Aggregate across folds**
   - After all `K` folds are run, aggregate performance over all test windows:
     - Report distribution of per‑fold metrics.
     - Optionally concatenate equity curves to view “stitched” walk‑forward performance.

This gives a realistic picture of how the pipeline would have performed over rolling 3‑month periods, with **only past data** available at each decision point.

## 3. Live training and retraining policy

We mirror the walk‑forward structure for live trading, replacing “test window” with “live trading window”.

### 3.1. Initial go‑live

1. **Choose go‑live date** `GoLive_start` (e.g. end of the last historical walk‑forward test window).
2. **Training window for initial live model**
   - `Train_live_start = max(T_start, GoLive_start − 24 months)`.
   - `Train_live_end = GoLive_start`.
3. **Train & select model**
   - Restrict data to `[Train_live_start, Train_live_end)` and run training the same way as in walk‑forward folds:
     - Chronological split into train/validation inside this range.
     - Select best hyperparameters and model using validation metrics.
   - Promote the chosen model by updating `best_hyperparameters.json` / active model path.
4. **Start live trading**
   - Use this promoted model for all trading decisions from `GoLive_start` onwards.

### 3.2. Scheduled retraining cadence

We adopt a **3‑month retraining cadence** aligned with the 3‑month evaluation windows:

- Every 3 months, at dates:
  - `R_1 = GoLive_start + 3 months`,
  - `R_2 = GoLive_start + 6 months`,
  - `R_3 = GoLive_start + 9 months`, ...

At each retrain point `R_j`:

1. **Freeze data up to `R_j`**
   - Only use data where `Time < R_j` for training/validation.

2. **Define rolling training window**
   - `Train_j_end = R_j`.
   - `Train_j_start = max(T_start, Train_j_end − 24 months)`.

3. **Train candidate models on `[Train_j_start, Train_j_end)`**
   - Same logic as walk‑forward folds:
     - Chronological train/validation split inside this window.
     - Possible hyperparameter search (via short "experiment" runs) restricted to this window.

4. **Evaluate and decide promotion**
   - Evaluate candidate(s) using validation metrics and optionally a short **paper‑trading backtest** on the most recent few months *prior to* `R_j` (but still `< R_j`).
   - Only promote the new model to “active” if it beats the current active model on chosen metrics (e.g. validation loss, Sharpe, drawdown).

5. **Use new model from `R_j` onward**
   - The model active during `(R_j, R_j + 3 months]` is the one trained on `[Train_j_start, Train_j_end)`.
   - Between retrain points we keep the model fixed (no continuous online updating) for simplicity and reproducibility.

### 3.3. Emergency retraining (optional)

In addition to the 6‑month schedule, we can define **trigger‑based retraining** when the live system’s metrics degrade severely, for example:

- Live rolling 3‑month Sharpe drops below a threshold.
- Max drawdown exceeds a risk limit relative to backtested expectations.

When a trigger fires:

- Run an **unscheduled retrain** using the same rolling 24‑month window up to “now”, then evaluate and potentially promote a new model.
- Document the trigger and the model change for auditability.

## 4. Summary

- Offline, we perform **walk‑forward backtests** with **3‑month test windows**, each preceded by up to **18–24 months** of training+validation data, split chronologically.
- Online, we use the **same 18–24‑month rolling window** and **3‑month cadence** for scheduled retraining:
  - Train on the past ~18–24 months.
  - Hold the model fixed for the next 3 months of live trading.
  - Repeat.
- This keeps the evaluation and live workflows aligned, enforces strict time ordering, and avoids data leakage while remaining operationally simple to implement in the existing CLIs and notebooks.
