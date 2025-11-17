# Trading System Operational Runbook

## 1. Purpose

This runbook describes how to operate the trading system across phases, with an emphasis on **offline backtesting** and **safe, simulated runs** before any real-money trading.

---

## 2. Phase 0 – Offline Backtesting

### 2.1 Typical Workflow

1. **Prepare data**
   - Ensure `data/processed/nvda_hourly.csv` is up to date (via `src/daily_data_agent.py`).

2. **Select model & configuration**
   - Pick a model from `models/registry/` and note the path.
   - Decide on strategy parameters (thresholds, stops, etc.).

3. **Run backtest**
   - Invoke the backtest script (to be implemented), e.g.:
     - `python src/backtest.py --config backtests/example_config.yaml`
   - Monitor runtime and output logs.

4. **Review results**
   - Inspect summary metrics (CAGR, Sharpe, max drawdown).
   - Review trade logs for obvious issues (over-trading, odd entries/exits).
   - Compare against previous runs if doing a regression check.

### 2.2 Daily Checklist (When Actively Iterating)

- Confirm data freshness (last trading day present in processed CSVs).
- For any new model version:
  - Run at least one reference backtest with a standard config.
  - Compare metrics vs. previous baseline.

---

## 3. Phase 1 – Local Paper Trading (Simulated)

### 3.1 Starting Paper Trading

1. **Check data feed**
   - Ensure IB/TWS or other feed is running and reachable.

2. **Start paper-trading loop**
   - Run the paper-trading script (to be implemented), e.g.:
     - `python src/paper_trade.py --config configs/paper_trade_nvda.yaml`
   - Confirm that simulated trades and PnL are being logged locally.

3. **Monitor**
   - Periodically check logs for errors and unusual behavior.
   - Compare live-simulated PnL against what you would expect from backtests over the same period.

### 3.2 Handling Common Issues

- **Data feed interruptions**
  - Pause the paper-trading loop if bars stop arriving.
  - Restart once the feed is stable; document any gaps.

- **Strategy misbehavior (too many trades, odd entries)**
  - Stop the loop.
  - Reproduce the behavior in an offline backtest using recorded bars.
  - Fix strategy logic or parameters, then re-run tests.

---

## 4. Phase 2 – Optional AWS & Live Trading (Future)

> Note: This phase is **not part of the MVP**. It is included here for future reference.

### 4.1 AWS-Based Components

- **Prediction service**
  - Deployed as a containerized FastAPI app.
- **Strategy runner**
  - Scheduled jobs (e.g., EventBridge + Lambda/ECS) that call the prediction service and generate signals.
- **Execution service**
  - Interfaces with a broker adapter to send real orders.
- **Storage**
  - Managed database for trades, signals, and performance.

### 4.2 Safety Practices

- Start with **paper accounts** or a full simulation mode even on AWS.
- Introduce real-money trading only with:
  - Strict position and risk limits.
  - Clear monitoring and alerting.
  - Manual “kill switch” to stop trading.

---

## 5. Incident Management (Across Phases)

### 5.1 Unexpected Strategy Behavior

**Symptoms:**
- Sudden surge in trades.
- Large unexpected loss.
- Positions held outside intended hours.

**Steps:**
1. Stop the backtest or paper-trading run.
2. Export logs (trades, predictions, prices) for the affected period.
3. Reproduce in a controlled offline backtest.
4. Fix logic or configuration.
5. Add regression tests to cover the scenario.

### 5.2 Data Anomalies

**Symptoms:**
- Spikes or gaps in price series affecting decisions.

**Steps:**
1. Inspect the underlying processed data in `data/processed/`.
2. Verify ingestion and gap-filling behavior.
3. If data is wrong, fix ingestion/processing and re-run backtests.

---

## 6. Change Management

- For any change to strategy logic, backtester, or prediction provider:
  - Update design docs (`trading_system_requirements`, `trading_system_hld`) as needed.
  - Add or update unit/integration/regression tests.
  - Run a standard set of reference backtests before and after the change.

Keep this runbook updated as you gain operational experience with backtests and simulations.