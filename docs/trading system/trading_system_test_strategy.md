# Trading System & Backtesting Test Strategy

## 1. Purpose

This document defines the testing strategy for the trading system components (strategy logic, backtester, and paper-trading loop), with a focus on validating behavior before any live or AWS-based deployment.

---

## 2. Testing Objectives

- Verify correctness of signal generation, order simulation, and PnL calculations.
- Ensure that backtests are deterministic and reproducible.
- Detect regressions when strategies, models, or data handling change.
- Build confidence before moving from offline backtests to paper trading and (later) any real execution.

---

## 3. Test Types

### 3.1 Unit Tests

**Scope:** Individual functions and small components with external dependencies mocked.

**Targets:**
- Strategy decision functions (`generate_signal`).
- Order and position update logic.
- PnL and risk-rule calculations (stops, take-profit, daily loss limits).
- Helper utilities (e.g., equity curve construction, metric calculations).

**Examples:**
- `test_generate_signal_long_when_predicted_return_above_threshold`.
- `test_stop_loss_triggers_exit_when_price_moves_against_position`.
- `test_daily_loss_cap_stops_trading_for_rest_of_day`.

### 3.2 Integration Tests

**Scope:** Multiple components working together on small, realistic datasets.

**Targets:**
- End-to-end backtest over a small historical window (e.g., a few days of NVDA hourly data).
- Backtest integration with prediction provider (mocked vs. real model).
- Paper-trading loop using a mocked real-time data feed.

**Examples:**
- `test_backtest_runs_and_produces_expected_number_of_trades`.
- `test_backtest_metrics_match_expected_values_on_synthetic_series`.
- `test_paper_trading_loop_uses_same_logic_as_backtester`.

### 3.3 Regression / Golden Backtest Tests

**Scope:** Guard against unintended changes in strategy behavior over time.

**Approach:**
- Maintain small **golden backtest scenarios**:
  - Fixed input data (synthetic or sampled).
  - Fixed strategy parameters.
  - Fixed model or prediction file.
- Store expected:
  - Summary metrics.
  - Aggregate PnL.
  - Key trade statistics.

**Examples:**
- `test_golden_backtest_nvda_week_equity_curve_stable`.
- `test_golden_backtest_trade_count_and_max_drawdown_unchanged`.

### 3.4 Performance Tests

**Scope:** Assess whether the backtester can handle long histories and parameter sweeps.

**Approach:**
- Run backtests over multi-year NVDA history on a typical dev machine.
- Run simple parameter sweeps (e.g., threshold grid) and measure runtime.

**Metrics:**
- Runtime per backtest.
- Throughput (backtests per minute) for typical grids.

### 3.5 Contract Tests (APIs & Interfaces)

**Scope:** Ensure that key interfaces remain stable for future AWS deployment and external integration.

**Targets:**
- Strategy interface (inputs, outputs).
- Prediction provider interface (how predictions are requested and returned).
- Broker adapter interface (even if no real broker is used yet).

**Examples:**
- `test_strategy_interface_accepts_expected_state_fields`.
- `test_prediction_provider_can_be_swapped_between_live_and_cached`.

---

## 4. Test Data Strategy

### 4.1 Synthetic Data

Use synthetic price series with known patterns for unit tests:
- Simple uptrends, downtrends, and ranges.
- Known “signals” where an ideal strategy would enter/exit.

This makes it easy to reason about expected trades and PnL.

### 4.2 Sampled Real Data

For integration and regression tests:
- Use small slices of real NVDA data from `data/processed/`.
- Keep them checked into the repo or generated deterministically from seed data.

Ensure no sensitive information is included.

---

## 5. Tooling & Execution

- Use `pytest` for running tests (aligned with the rest of the repo).
- Mirror the `src/` layout under `tests/` (e.g., `tests/test_trading_backtest.py`).
- Use fixtures for:
  - Synthetic series and predictions.
  - Temporary directories for backtest outputs.
  - Mock prediction providers and data feeds.

CI should run:
- Unit and integration tests on each commit touching trading/backtesting code.
- Regression (golden) tests when relevant files change (strategies, backtester, prediction provider).

---

## 6. Coverage Goals

- **Unit tests:**
  - >80% function coverage for core strategy and backtester modules.
- **Integration tests:**
  - At least one end-to-end backtest test per core strategy.
- **Regression tests:**
  - At least one golden scenario for NVDA over a multi-day window.

These goals can be revisited as the trading system grows in complexity.