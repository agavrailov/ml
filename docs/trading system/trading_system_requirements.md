# Trading System & Backtesting Requirements (Repo-Scoped)

## 1. Purpose & Scope

This document defines **repo-specific** requirements for building a trading system on top of the existing LSTM prediction stack. The focus is on:

- Turning model predictions into **trading signals** and **simulated trades**.
- Providing a **backtesting framework** to validate strategies before going live.
- Keeping the initial MVP **local-only and zero-cost** (no AWS, no real-money orders).
- Preparing a path for **optional AWS-based automation** once strategies are robust.

The design assumes NVDA-focused LSTM models and local CSV data under `data/`, but should be easy to generalize.

### In-Scope (MVP & near term)

- The concrete entry/exit and risk rules for the MVP strategy are defined in `trading_system_strategy.md` and referenced here.

1. **Offline backtesting engine (Phase 0)**
   - Consume historical OHLCV from `data/processed/nvda_*.csv`.
   - Call `src.predict.predict_future_prices` (or cached predictions) to derive signals.
   - Simulate orders, PnL, and basic risk metrics entirely offline.

2. **Strategy & risk definition layer**
   - Minimal, explicit representation of:
     - Entry/exit rules (based on predicted returns, thresholds, and optional indicators).
     - Position sizing (e.g., fixed fraction of capital, max position per trade).
     - Risk limits (e.g., per-trade stop, daily max loss, max concurrent positions = 1 for MVP).

3. **Backtest configuration & CLI**
   - Simple CLI (Python script) to run backtests over a given period with:
     - Strategy parameters.
     - Train/validation/test splits.
     - Output of metrics (CAGR, Sharpe, max drawdown, hit rate, etc.) and trade logs to a file.

4. **Paper-trading mode (Phase 1, still local)**
   - Optional mode that consumes near-real-time data (e.g., from IB/TWS) but does **not** send real orders.
   - Reuses the same strategy abstraction and order simulation as offline backtests.

5. **Design for future AWS-based live trading (Phase 2+)**
   - Clearly document how the same components (signals, risk checks, backtester) can be hosted or orchestrated on AWS later.
   - No AWS resources required for MVP; AWS is a deployment target, not a prerequisite.

### Out of Scope (for MVP)

- Real-money order routing and execution.
- Complex portfolio management across many symbols (MVP is single-symbol NVDA).
- Latency-sensitive HFT or market-making.
- Generic feature store or distributed backtesting framework.

These can be considered in later phases after robust single-symbol strategies are validated.

---

## 2. Functional Requirements

### 2.1 Backtesting Engine (Offline)

**FR-1: Input data sources**  
The backtester MUST be able to read:

- Historical OHLCV from `data/processed/nvda_hourly.csv` (default) and optionally other frequencies.
- Model predictions, either by:
  - Calling `predict_future_prices` on sliding windows of OHLC data, or
  - Reading precomputed predictions from a CSV/JSON file (for speed when running many backtests).

**FR-2: Trading calendar & session handling**  
The backtester MUST:

- Respect the same trading calendar used in `src.config.MarketConfig`.
- Avoid trading outside regular trading hours (RTH) in the default configuration.

**FR-3: Strategy API**  
Provide a minimal strategy interface, e.g.:

- `generate_signal(state) -> {"action": "LONG"|"FLAT"|"SHORT", "size": float, "meta": ...}`
- Where `state` includes:
  - Current bar OHLCV.
  - Recent history window.
  - Model prediction(s) for future price/return.
  - Optional indicators (SMA, RSI, etc.).

MVP can implement a single built-in strategy (e.g., long-only NVDA when predicted return > threshold) but the code should not preclude adding more.

**FR-4: Order simulation & PnL**  
The backtester MUST:

- Simulate trades at a configurable execution price (e.g., next-bar open or bar VWAP approximation).
- Support long-only positions for MVP (no leverage, no shorting).
- Track per-trade and cumulative PnL, including:
  - Gross PnL.
  - Net PnL (after fixed per-trade cost/slippage model).
  - Max drawdown and equity curve.

**FR-5: Risk controls in backtest**  
The backtester MUST enforce basic risk controls:

- Per-trade stop loss (e.g., fixed % from entry price).
- Per-trade take profit.
- Max number of open positions (1 for MVP).
- Optional daily loss limit that stops trading for the day.

**FR-6: Metrics & reporting**  
For any backtest run, the system MUST output:

- Summary metrics (CAGR, Sharpe ratio, max drawdown, win rate, average win/loss, trades per year).
- A trade log (entry/exit times, prices, size, PnL per trade).
- A simple equity curve time series.

Outputs MAY be written to CSV/JSON files under `backtests/` or similar.

### 2.2 Paper Trading (Local, Optional)

**FR-7: Real-time data feed abstraction**  
The system SHOULD define an interface for consuming live bars from IB/TWS (or another source) that:

- Matches the structure used by the backtester (timestamp, OHLCV, symbol).
- Can be mocked for tests.

**FR-8: Real-time signal evaluation**  
Given a new bar, the paper-trading loop MUST:

- Update state/history.
- Compute/lookup the model prediction.
- Call the same `generate_signal` logic as in offline backtests.
- Simulate trades in real time using the same PnL logic as offline.

No real orders should be sent to a broker in MVP; this is a dry-run mode.

### 2.3 Future AWS-Based Live Trading (Non-MVP)

**FR-9: AWS deployment hooks**  
The design MUST:

- Identify which components could later be hosted on AWS (e.g., FastAPI prediction service, strategy runner, trade logger).
- Avoid tight coupling to local file paths that would block containerization.

**FR-10: Broker adapter abstraction**  
For future live trading, define an adapter interface (even if the implementation is deferred):

- `submit_order(order) -> OrderId`
- `get_order_status(order_id)`
- `cancel_order(order_id)`

MVP can have a **no-op implementation** that just logs the intended actions.

---

## 3. Non-Functional Requirements

### 3.1 Performance & Scale (Backtesting)

**NFR-1: Speed for parameter sweeps**  
The backtester should:

- Run a single-parameterized backtest over several years of hourly data within **seconds to a few minutes** on a typical dev machine.
- Support running many backtests in a loop (e.g., grid over thresholds and stops) without manual intervention.

**NFR-2: Determinism**  
Backtests MUST be deterministic given the same:

- Input data.
- Model snapshot or prediction file.
- Strategy parameters.

### 3.2 Interoperability & Backwards Compatibility

**NFR-3: Reuse existing config and data layout**  
The trading/backtesting code MUST:

- Reuse `src.config` for paths and market calendar where possible.
- Consume the same processed data that training/evaluation scripts use.

**NFR-4: Non-intrusive integration**  
MVP code should:

- Not change the behavior of existing training, evaluation, or prediction scripts.
- Be add-on functionality that can be developed and tested independently.

### 3.3 Safety

**NFR-5: No real-money risk in MVP**  

- The default configuration MUST NOT send real orders to any broker.
- Any future live trading functionality MUST be explicitly opt-in and clearly separated (e.g., different entry script, environment flag).

---

## 4. Testing Requirements (Repo-Specific)

**TR-1: Unit tests for strategy & PnL logic**  
Add unit tests to cover:

- Signal generation for synthetic price/prediction patterns.
- Order lifecycle and PnL calculation on small toy data.
- Risk rule enforcement (stops, max positions).

**TR-2: Integration tests for backtesting loop**  
Add tests that:

- Run a short backtest (e.g., a few days of synthetic or sampled NVDA data).
- Verify the number of trades, aggregate PnL, and key metrics against expected values.

**TR-3: Regression tests using golden datasets**  

- Maintain one or more small “golden” backtest scenarios where outputs are checked into the repo.
- Ensure future code changes do not change results unless intentionally.

---

## 5. Open Questions

1. **Execution price model**  
   - Use next-bar open, mid-price, or a simple slippage model for executions?
2. **Prediction source for backtests**  
   - On-the-fly model calls (slower, but closer to real-time) vs. precomputed prediction files (faster for large sweeps)?
3. **Strategy complexity**  
   - How quickly to move from a single-threshold long-only strategy to more complex ones (multi-timeframe, regime switches)?
4. **External vs. internal backtesting libraries**  
   - Integrate an existing library (e.g., `backtrader`) vs. maintain a lightweight custom backtester aligned with this repo’s data structures.

This document should be updated as implementation decisions are made and early experiments with backtesting/trading are run.