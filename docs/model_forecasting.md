# LSTM Forecast Horizon and Bars Ahead

## What exactly does the model predict?
The LSTM is trained to predict forward log returns on the Open price over a fixed horizon `ROWS_AHEAD` bars.

Given a sliding window of length `TSTEPS` ending at bar index `t`:
- Input window: bars `[t - TSTEPS + 1, …, t]`
- Target label:
  
  `r_t = log(Open_{t + ROWS_AHEAD}) - log(Open_t)`

By default:
- `ROWS_AHEAD = 1` (see `TrainingConfig.rows_ahead` in `src/config.py`)
- The model predicts the move from the current bar’s Open (`Open_t`) to the next bar’s Open (`Open_{t+1}`).

At inference time, we turn the model’s log-return prediction back into a price via:

`P_hat_{t + ROWS_AHEAD} = Open_t * exp(r_hat_t)`

So, with the defaults:

> The current implementation is a 1-bar-ahead forecast: from bar `t` to bar `t+1`.

This applies both to:
- Batch prediction (`predict_future_prices` in `src/predict.py`)
- Live / bar-by-bar prediction (`LivePredictor.update_and_predict` in `src/live_predictor.py`)

## Changing how many bars ahead we predict
The horizon (how many bars ahead we predict) is controlled by `ROWS_AHEAD`:
- Defined in `TrainingConfig` as `rows_ahead`
- Exposed as the module-level constant `ROWS_AHEAD` in `src/config.py`
- Used to build training targets in `train_model` (see `_make_log_return_targets` in `src/train.py`)

To change the horizon from 1 bar to `K` bars ahead:
1. Set `TrainingConfig.rows_ahead = K` (and thus `ROWS_AHEAD = K`).
2. Retrain the model for the desired `frequency` and `TSTEPS`.

After retraining, each prediction window ending at bar `t` will estimate the move from `Open_t` to `Open_{t+K}` and `predict_future_prices` / `LivePredictor` will return the corresponding price `K` bars ahead.

## Multi-step forecasts (sequence of future bars)
Currently, the model architecture is single-output:
- The final layer in `src/model.py` is `Dense(units=1)`.
- The code uses that single scalar log-return prediction to produce one future price per window.

This means:
- Today, we support one-step-ahead forecasts at a configurable horizon (`ROWS_AHEAD`).
- We do not yet output a full path `[t+1, t+2, …, t+N]` directly from the model.

To forecast multiple future bars, there are two main extension options.

### 1. Recursive 1-step forecasting (no retraining required)
- Keep `ROWS_AHEAD = 1`.
- At inference, repeatedly:
  - Predict the next bar’s price.
  - Append that predicted bar to the input history.
  - Slide the window and predict again.
- Pros: reuses the existing trained model.
- Cons: prediction errors accumulate over multiple steps.

### 2. Direct multi-step output (requires retraining)
- Change training targets so each window’s label is a vector of future log returns (for example, the next `K` bars).
- Change the final Dense layer to output `K` values instead of `1`.
- Update training and prediction code to handle vector outputs.
- Pros: more stable multi-step forecasts and better control over multi-horizon behavior.
- Cons: more complex code changes and full retraining required.

Currently, the production path uses a single predicted bar at horizon `ROWS_AHEAD`, with `ROWS_AHEAD = 1` by default.