# Trading System & Backtesting High-Level Design (HLD)

## 1. Overview

This HLD describes how to turn the existing LSTM-based price prediction stack into a **trading system**, with a strong emphasis on:

- **Offline backtesting first** (no broker, no AWS, no cost).
- A clean separation between **signals**, **backtesting**, and **execution**.
- A path to **optional AWS-based automation** after sufficient validation.

The design assumes NVDA hourly data and models as implemented in `src/`, but keeps concepts general.

---

## 2. Phased Architecture

We evolve the system in phases, each building on the previous one.

### Phase 0 – Offline Backtesting (Local, Zero AWS)

- **Goal:** Evaluate strategies using historical data and model predictions with no external dependencies.
- **Components:**
  1. **Data Loader**
     - Reads historical OHLCV from `data/processed/nvda_hourly.csv` (and other frequencies as needed).
  2. **Prediction Provider**
     - Either:
       - Calls `predict_future_prices` on sliding windows; or
       - Loads precomputed predictions from disk.
  3. **Strategy Engine**
     - Converts predictions and features into discrete trade signals (`LONG`, `FLAT`, optional `SHORT`).
  4. **Backtest Engine**
     - Simulates order fills, PnL, and risk rules.
     - Produces metrics and trade logs.

All computations run on the local machine using existing CSV and model artifacts.

### Phase 1 – Local Paper Trading (No AWS, Simulated Execution)

- **Goal:** Run the same strategies on near-real-time data, but with **simulated** orders.
- **Components:**
  1. **Real-Time Data Adapter**
     - Thin wrapper around IB/TWS or another feed to produce bar events identical in shape to historical bars.
  2. **Real-Time Strategy Loop**
     - For each new bar:
       - Update state.
       - Get prediction.
       - Generate signals and simulated trades.
  3. **Live Metrics Logger**
     - Logs simulated trades and PnL progression to disk.

The same strategy and risk code is reused from Phase 0; only the data source changes.

### Phase 2 – Optional AWS-Based Trading Services

- **Goal:** Host parts of the system on AWS once strategies prove robust.
- **Candidate components for AWS:**
  1. **Prediction Service**
     - Containerized FastAPI (from `api/main.py`) deployed to ECS/Fargate or similar.
  2. **Strategy Runner**
     - Scheduled or event-driven jobs (e.g., using AWS EventBridge + Lambda/ECS task) that:
       - Fetch latest bars.
       - Call the prediction service.
       - Produce signals.
  3. **Execution & Risk Service**
     - A service that receives approved signals and interacts with a broker via a pluggable adapter.
  4. **Storage**
     - Managed storage (RDS, DynamoDB, S3) for trades, signals, and performance metrics.

The existing local backtester remains the reference and safety net.

---

## 3. Logical Components

> The detailed trading rules (entries, exits, risk, and filters) are specified in `trading_system_strategy.md`. This HLD focuses on components and data flow.

### 3.1 Data & Prediction Layer

**Responsibilities:** Provide historical and live market data plus model predictions in a unified shape.

- **Historical Data Loader**
  - Reads hourly NVDA data from `data/processed/`.
  - Applies the same feature engineering as in training (via `src/data_processing.py`).

- **Prediction Provider**
  - Abstracts `predict_future_prices` so the backtester can request predictions for a given timestamp or bar sequence.
  - Optional support for cached predictions:
    - A simple file-based format like `backtests/predictions_nvda_hourly.csv` with columns: `Time`, `predicted_price`, `predicted_return`, etc.

### 3.2 Strategy Layer

**Responsibilities:** Map features and predictions to trade actions.

- **Strategy Interface** (conceptual)
  - Input: `StrategyState` containing:
    - Current and recent OHLCV bars.
    - Latest prediction(s).
    - Optional indicators (SMA, RSI, etc.).
  - Output: `Signal` with:
    - `action`: `LONG`, `FLAT` (MVP) or `SHORT` (future).
    - `size`: fraction of capital or fixed number of shares/contracts.

- **Example MVP Strategy**
  - Long-only NVDA when predicted next-bar return exceeds a threshold.
  - Close the position when:
    - Stop loss or take profit is hit; or
    - Prediction drops below a lower threshold.

### 3.3 Backtesting Engine

**Responsibilities:** Simulate trading over historical data.

- **Core Concepts:**
  - `BacktestConfig` – defines:
    - Date range.
    - Initial capital.
    - Strategy parameters (thresholds, stop distance, etc.).
    - Execution model (next-bar open, slippage assumptions).
  - `Position` and `Order` models.
  - `BacktestResult` – metrics + trade log + equity curve.

- **Execution Model (MVP)**
  - All trades executed at next-bar open price.
  - Per-trade fixed cost and/or slippage applied as a simple deduction.
  - No partial fills or order-book modeling.

- **Risk Management**
  - Per-trade stop and take profit.
  - Max open positions = 1.
  - Optional daily loss cap.

### 3.4 Paper Trading Loop

**Responsibilities:** Apply the same logic as backtesting in a streaming context.

- **Data Flow:**
  1. Receive new bar.
  2. Update state/history window.
  3. Request prediction.
  4. Generate signal and update simulated positions.
  5. Append to a local log and update PnL.

Implementation can reuse most of the backtester core but with a different data source.

### 3.5 Future Execution & AWS Integration

**Responsibilities:** (Phase 2+) Bridge signals to actual broker orders and host components on AWS.

- **Broker Adapter**
  - Encapsulates any broker-specific API (IB, etc.).
  - Enables switching between a `SimulatedBroker` (for tests) and `InteractiveBrokersBroker` for real trading.

- **AWS Hosting**
  - Prediction and strategy services can run as containers.
  - State (positions, signals) stored in a managed database.
  - Backtesting remains local or can be containerized for parameter sweeps on cloud compute as desired.

#### 3.5.1 IBKR implementation (paper-first)

When the broker adapter is implemented against IBKR, the concrete design SHOULD:

- Use an `IBKRBroker` implementation that connects to TWS or IB Gateway (via a library such as `ib_insync`).
- Keep IBKR specifics isolated behind the generic broker interface used by the strategy and backtester.
- Wrap `IBKRBroker` with a `RiskManagedBroker` responsible for enforcing max daily loss, per-symbol limits, and kill-switch behavior.
- Reuse the same strategy interface and signal-to-order mapping as in the offline backtester and simulated paper-trading loop.
- Treat IBKR **paper accounts** as the default target; any live-money mode must be explicitly configured and opt-in.

---

## 4. Data Flow (Phase 0 Backtest)

1. **Load historical data** from `data/processed/nvda_hourly.csv`.
2. **Prepare features** using the same logic as training (`prepare_keras_input_data`).
3. For each bar in the backtest window:
   - Construct the required window for the model (long-enough history for indicators + `tsteps`).
   - **Get prediction** for the next bar.
   - **Call strategy** to generate a signal.
   - **Simulate execution** and update positions and cash.
4. At the end of the run, compute **metrics** and write results to disk.

This loop must be efficient enough to be run many times with different parameters.

---

## 5. Versioning & Reproducibility

- Tie each backtest to:
  - A specific model file (path under `models/registry/`).
  - A copy or hash of `best_hyperparameters.json`.
  - Strategy parameters and backtest configuration.
- Store this metadata in an accompanying JSON next to backtest results so any run can be reproduced later.

---

## 6. Non-Functional Considerations

- **Simplicity first:** keep the initial backtester small and readable rather than generic and framework-like.
- **Safety:** no live trading or AWS resources are required until Phase 2; early phases are entirely local and simulated.
- **Extensibility:** strategy, data feed, and broker interfaces should make it straightforward to plug in new strategies or deployment targets later.