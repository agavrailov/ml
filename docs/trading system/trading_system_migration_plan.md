# Trading System Migration & Rollout Plan

## 1. Purpose

This document outlines a **phased plan** for introducing trading capabilities on top of the existing prediction system, with a strong emphasis on:

- Backtesting and validation before any live execution.
- Avoiding AWS cost and real-money risk in early phases.
- Maintaining a clear rollback path at each step.

---

## 2. Current vs. Target State

### 2.1 Current State

- LSTM-based model training, tuning, and evaluation are implemented.
- A FastAPI prediction endpoint and Streamlit UI exist for manual exploration.
- No structured trading system (strategies, backtester, or execution layer) exists yet.

### 2.2 Target State

- A modular trading system with:
  - Offline backtesting.
  - Paper-trading loop.
  - Optional AWS-hosted components and broker integration.
- Strong test coverage and reproducible backtest workflows.

---

## 3. Phased Plan

### Phase 0 – Design & Backtesting MVP (Local, No AWS)

**Goals:**
- Define trading system requirements and architecture (this doc, requirements, HLD).
- Implement a minimal but useful backtester and example strategy.

**Tasks:**
- Implement:
  - Data loader for historical OHLCV.
  - Prediction provider (live calls or cached predictions).
  - Simple long-only NVDA strategy.
  - Backtest engine with PnL and basic risk rules.
- Create initial reference backtests and regression tests.

**Exit Criteria:**
- At least one backtest over multi-month NVDA data runs successfully.
- Metrics and trade logs are inspected and considered reasonable.
- Regression tests exist for at least one golden scenario.

**Rollback:**
- Backtester code is isolated; removing it does not affect training or prediction.

---

### Phase 1 – Local Paper Trading (Simulated Execution)

**Goals:**
- Apply the same strategies to near-real-time data.
- Keep execution fully simulated; no broker orders.

**Tasks:**
- Implement a real-time data adapter (e.g., IB/TWS wrapper) that produces bars.
- Implement a paper-trading loop reusing the backtester’s core logic.
- Log simulated trades and PnL during live market hours.

**Exit Criteria:**
- Paper-trading loop runs for several sessions without crashes.
- Behavior is consistent with backtests over the same periods.

**Rollback:**
- Stop paper-trading script; no persistent state changes in production systems.

---

### Phase 2 – Optional AWS Deployment (No Real Money Yet)

**Goals:**
- Host prediction and strategy evaluation on AWS for reliability and flexibility.
- Still operate in simulation or paper-trading mode.

**Tasks:**
- Containerize the prediction API and deploy to AWS (ECS/Fargate or similar).
- Implement a strategy runner service/job that calls the prediction API and computes signals.
- Store signals and simulated trades in AWS-managed storage.

**Exit Criteria:**
- End-to-end simulated runs work via AWS-hosted services.
- Metrics and logs remain accessible and consistent.

**Rollback:**
- Shut down AWS services; continue using local backtesting and paper trading.

---

### Phase 3 – Controlled Live Trading (Optional, Post-Validation)

> This phase is intentionally **beyond the MVP** and should only be attempted after extensive testing and paper trading.

**Goals:**
- Enable small-scale real-money trading with strict risk controls.

**Tasks:**
- Implement a broker adapter and execution service.
- Introduce configuration flags to switch between simulated and real execution.
- Establish monitoring, alerts, and manual controls (kill switch).

**Exit Criteria:**
- Small, controlled live trading runs with limited capital behave as expected.
- No critical incidents over a predefined observation period.

**Rollback:**
- Switch execution back to simulated mode.
- Disable or decommission live-trading entrypoints.

---

## 4. Validation & Acceptance Criteria

For each phase, validate:

- **Data & Metrics:**
  - Backtests produce stable results across runs.
  - Paper-trading PnL aligns with what backtests predict.

- **Safety:**
  - No accidental broker orders in phases 0–2.
  - Clear separation between simulated and live execution.

- **Observability:**
  - Logs and metrics are sufficient to debug strategy behavior.

---

## 5. Communication & Documentation

- Update `WARP.md` and the trading system docs (requirements, HLD, test strategy, runbook) as phases are implemented.
- Maintain a simple changelog for:
  - Strategy changes.
  - Backtester changes.
  - Any deployment to AWS or brokers.

This plan should be refined as practical experience is gained from backtests and paper trading.