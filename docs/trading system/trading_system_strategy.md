# Trading Strategy Specification (MVP)

## 1. Purpose

This document defines the **initial trading strategy** for the MVP backtesting and paper-trading system. It is intentionally simple but strongly aligned with the existing LSTM prediction model and volatility regime.

The goals are to:

- Exploit the model’s **price forecast** as the main edge.
- Avoid trades where the predicted move is **small relative to volatility and model error**.
- Enforce **capital-based risk** and a configurable **reward:risk** ratio.

---

## 2. Instruments, Timeframe, Horizon

- **Instrument (MVP):** NVDA stock (can be generalized later).
- **Timeframe:** Hourly bars from `data/processed/nvda_hourly.csv`.
- **Prediction horizon:**
  - MVP: Single-step horizon (`H = 1` bar ahead) using the current LSTM.
  - Future: Multi-step horizons (e.g. 2–3 bars ahead) via `ROWS_AHEAD` models or iterative forecasting.

All rules below are defined for a **long-only** strategy on hourly bars but are structured to allow extension.

---

## 3. Inputs

For each decision time `t` (end of bar):

- `P0` – current close or execution reference price at `t`.
- `P_pred` – model-predicted price at horizon `H` (e.g. one bar ahead).
- `predicted_move = P_pred - P0`.
- `sigma_err` – typical model error at horizon `H` (e.g. RMSE or residual std from `evaluate_model_performance`).
- `ATR_H` – Average True Range on the trading timeframe (e.g. 14-period ATR on hourly bars).
- `account_equity` – current equity in the backtest/paper-trading account.

Configuration parameters:

- `risk_per_trade_pct` – fraction of equity to risk per trade (e.g. 1%).
- `reward_risk_ratio` – target nominal reward:risk ratio (e.g. 2.0).
- `k_sigma_err` – how many residual std devs to subtract from predicted move (e.g. 1.0).
- `atr_period` – ATR lookback (e.g. 14 bars).
- `k_atr_min_tp` – minimum TP distance in ATR units (e.g. 0.5–1.0).

These will be supplied through the backtester configuration.

---

## 4. Signal Generation Logic

### 4.1 Long Entry Conditions

At each bar `t`, consider a new long entry only if there is no open position.

1. **Directional edge:**
   - `predicted_move > 0` (model expects price to rise).

2. **Prediction-confidence filter (error-adjusted move):**
   - Define the **usable move**:
     - `usable_move = predicted_move - k_sigma_err * sigma_err`.
   - Require `usable_move > 0`.

3. **Volatility-aware filter (ATR-based minimum TP):**
   - Compute `min_tp_dist_atr = k_atr_min_tp * ATR_H`.
   - Require `usable_move >= min_tp_dist_atr`.

4. **Combined threshold:**
   - Optionally enforce a single threshold:
     - `usable_move >= max(k_sigma_err * sigma_err, k_atr_min_tp * ATR_H)`.

If all conditions pass, generate a **long** entry candidate.

### 4.2 No Trade / Flat Condition

- If any of the above conditions fail, **no new trade** is opened.
- If a position is already open, the exit rules (Section 5) still apply.

---

## 5. Exit, Stop Loss, and Take Profit

For a new long trade opened at price `P0`:

### 5.1 Take Profit (TP)

- Use the error- and volatility-filtered `usable_move` as the maximum realistic reward:
  - `tp_dist = usable_move`.
  - `TP_price = P0 + tp_dist`.

This ties TP directly to the model’s forecast, discounted by typical error and ATR.

### 5.2 Stop Loss (SL) from Reward:Risk Ratio

Given `reward_risk_ratio = RR` and `tp_dist`:

- `stop_dist = tp_dist / RR`.
- `SL_price = P0 - stop_dist`.

This preserves the nominal R:R (e.g. 2:1) but makes **reward** the dependent variable based on model + volatility instead of forcing a fixed 2R move.

### 5.3 Position Sizing (Capital-Based Risk)

- Max risk in currency per trade:
  - `max_risk_notional = risk_per_trade_pct * account_equity`.
- Approximate risk per share (ignoring gaps and slippage):
  - `risk_per_share ≈ stop_dist`.
- Position size:
  - `size = max_risk_notional / risk_per_share` (rounded to integer shares/contracts).

If `size` falls below a minimum tradable unit (e.g. 1 share), the backtester may skip the trade.

### 5.4 Exit Rules

For an open long position:

- **Hard exits:**
  - Exit at `TP_price` (take profit hit).
  - Exit at `SL_price` (stop loss hit).

- **Optional time-based or prediction-based exit (future extension):**
  - If after `N` bars the model prediction no longer supports the position (e.g., `predicted_move <= 0` or `usable_move` drops below thresholds), exit at market.

MVP implementation can start with only hard TP/SL exits.

---

## 6. Handling Small Movements and Noise

The combination of **ATR filter** and **error-adjusted prediction** is explicitly designed to avoid trading on noise:

- We require `usable_move` to be sufficiently large compared to both:
  - Model error (`sigma_err`), and
  - Typical hourly volatility (`ATR_H`).

This means the strategy **does not attempt scalping** small price fluctuations within the hourly bar’s normal range.

If later you want to support small-move strategies, they should be modeled as a **separate regime/strategy** with explicit assumptions (lower costs, different timeframes), not shoehorned into this one.

---

## 7. Backtesting Parameters to Explore

Example parameter ranges for experiments:

- `risk_per_trade_pct` – 0.25% to 2%.
- `reward_risk_ratio` – 1.5 to 3.0.
- `k_sigma_err` – 0.5 to 2.0.
- `atr_period` – 10 to 20 bars.
- `k_atr_min_tp` – 0.5 to 1.5.

These can be swept by the backtester to understand robustness.

---

## 8. Future Extensions

- **Multi-horizon predictions:**
  - Use multiple horizons (e.g. 1, 2, 3 bars ahead) to shape TP and exit rules based on the predicted path.
- **Regime filters:**
  - Only trade in certain volatility or trend regimes determined by the model or additional indicators.
- **Shorts and multi-asset:**
  - Mirror the logic for short entries and extend to more symbols once NVDA performance is validated.

This spec should remain the single source of truth for the MVP trading logic. Changes to strategy behavior should be reflected here and accompanied by updated backtests and regression tests.