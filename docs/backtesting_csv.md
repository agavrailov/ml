# CSV-based backtesting workflows

This project now has a single, unified prediction pipeline used by both model-mode
backtests and CSV-mode backtests. This document describes the two supported
workflows and the expected CSV formats.

## 1. Model checkpoint replay (highest parity)

In this workflow you run a model-mode backtest once to generate a checkpoint CSV
with per-bar predictions and model error sigma, and then optionally replay that
checkpoint via CSV-mode.

### Steps

1. Run a model-mode backtest:

   ```bash
   python -m src.backtest \
     --frequency 60min \
     --prediction-mode model \
     --initial-equity 10000 \
     --report
   ```

   This writes a checkpoint under `backtests/`:

   - `backtests/nvda_<freq>_model_predictions_checkpoint.csv`

   with columns:

   - `Time`
   - `predicted_price` (bias-corrected price predictions)
   - `model_error_sigma` (per-bar residual sigma in price space)

2. Replay the checkpoint via CSV-mode on the same data/frequency:

   ```bash
   python -m src.backtest \
     --frequency 60min \
     --prediction-mode csv \
     --predictions-csv backtests/nvda_60min_model_predictions_checkpoint.csv \
     --initial-equity 10000 \
     --report
   ```

   When the `--predictions-csv` path points at the default checkpoint, the
   backtester reuses the same provider and sigma series that model-mode would
   use. An invariant test (`tests/test_backtest_checkpoint_replay.py`) enforces
   that the equity curve and trade list match model-mode on realistic slices.

### When to use

Use this workflow when you want:

- Exact parity between model-mode and CSV-mode on the same OHLC slice.
- To inspect or archive the per-bar predictions and sigma used in a run.

## 2. Script-generated predictions CSV

In this workflow you generate a predictions CSV via the shared prediction
pipeline and then run CSV-mode backtests against that file.

### Steps

1. Generate predictions CSV for a frequency:

   ```bash
   python -m scripts.generate_predictions_csv \
     --frequency 60min \
     --output backtests/nvda_60min_predictions.csv
   ```

   This script now delegates to `src.backtest._make_model_prediction_provider`,
   so it uses the same feature engineering, scaler, model, and
   bias-correction/sigma logic as model-mode. The generated CSV has columns:

   - `Time`
   - `predicted_price`
   - `model_error_sigma`
   - `already_corrected` (boolean, always `True` for this script)

2. Run a CSV-mode backtest using this file:

   ```bash
   python -m src.backtest \
     --frequency 60min \
     --prediction-mode csv \
     --predictions-csv backtests/nvda_60min_predictions.csv \
     --initial-equity 10000 \
     --report
   ```

   Because `already_corrected=True` and `model_error_sigma` are present, the
   CSV-mode path:

   - Aligns `predicted_price` and `model_error_sigma` to the OHLC data by
     `Time`.
   - Avoids applying the bias-correction layer a second time.
   - Uses the stored sigma series directly instead of recomputing it.

### When to use

Use this workflow when you want:

- A reusable predictions file for grid searches, UI experiments, or diagnostics.
- CSV-based backtests that are consistent with the model pipeline without
  needing to run model-mode every time.

## Required CSV schema for CSV-mode

CSV-mode backtests expect at minimum:

- `Time` (parseable as datetimes)
- `predicted_price` (float)

Optionally, CSVs may include:

- `model_error_sigma` (float): per-bar sigma series. When present, CSV-mode
  will align and use it directly.
- `already_corrected` (bool): if `True`, CSV-mode assumes predictions already
  passed through the bias-correction layer and will avoid re-applying it.

If neither `model_error_sigma` nor `already_corrected` are present, CSV-mode
falls back to:

- Applying a single bias-correction pass in price space.
- Estimating residual sigma via a rolling window and, if that collapses,
  substituting ATR as a proxy.

## Notes

- All of the above workflows are covered by tests to prevent regressions
  (`tests/test_backtest_checkpoint_replay.py`,
  `tests/test_generate_predictions_csv.py`,
  `tests/test_backtest_csv_mode_sigma.py`,
  `tests/test_csv_vs_model_consistency.py`).
- For high-stakes analysis, prefer either checkpoint replay or the
  script-generated CSV, as these now share the same underlying model
  prediction pipeline as the backtester.
