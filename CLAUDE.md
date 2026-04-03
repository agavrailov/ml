# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LSTM-based automated trading system for NVDA stock. Predicts price movements using a stateful LSTM model and executes trades through Interactive Brokers (IBKR). Full stack: data ingestion → model training → backtesting/walk-forward validation → live execution.

## Commands

### Testing
```bash
pytest                                        # Run all tests
pytest tests/test_model.py                    # Single test file
pytest tests/test_model.py::test_name         # Single test
pytest -k "backtest"                          # Tests matching keyword
pytest -v                                     # Verbose output
```

### Running the Application
```bash
# Windows launchers
runIBKR.bat   # Live trading daemon + Streamlit UI
runUI.bat     # UI only

# Individual services
streamlit run src/ui/app.py
uvicorn api.main:app --reload --port 8000
python -m src.ibkr_live_session --symbol NVDA --frequency 60min --client-id 2
```

### Data & Training
```bash
python src/daily_data_agent.py               # Full data pipeline (ingestion → features)
python src/daily_data_agent.py --skip-ingestion  # Dry run without IBKR
python src/train.py                          # Train with current config
python src/experiment_runner.py              # Hyperparameter grid search
python src/evaluate_model.py                 # Evaluate active model
python src/backtest.py --csv <predictions.csv>
python scripts/run_walkforward_backtest.py   # Walk-forward validation
```

### Job Runner (Streamlit decoupling)
```bash
python -m src.jobs.run --job-id <uuid> --job-type TRAIN --request '{"key": "val"}'
```

## Architecture

### High-Level Data Flow
```
IBKR (TWS/Gateway)
  → Raw minute OHLC (data/raw/)
  → Daily Data Agent: clean, resample, add features (SMA, RSI, ATR)
  → Processed CSVs (data/processed/nvda_{frequency}.csv)
  → LSTM Training → Model Registry (models/registry/*.keras)
  → Backtesting (historical) OR Live Daemon (real-time)
  → Job artifacts (runs/<job_id>/) OR JSONL event logs (ui_state/live/)
  → Streamlit UI polls job status and reads event logs
```

### Key Subsystems

**`src/config.py`** — Central configuration. `PathsConfig`, `TrainingConfig`, `StrategyConfig`, `IB` (broker settings). Environment variable overrides: `ML_LSTM_RAW_DATA_DIR`, `ML_LSTM_MODEL_SAVE_PATH`, `ML_LSTM_MODEL_REGISTRY_DIR`.

**`src/core/`** — Config contracts and resolution. `config_resolver.py` merges code defaults + `best_hyperparameters.json` + per-job `request.json` overrides. `config_library.py` has named config templates.

**`src/strategy.py`** — Trading logic. `StrategyConfig` holds thresholds (`k_sigma_long/short`, `k_atr_long/short`, risk %). `TradePlan` computes entry, TP, SL prices. Long-only by default; bidirectional mode is implemented but not default.

**`src/backtest_engine.py`** — Bar-by-bar simulation with commission/slippage. Outputs trades CSV, equity curve, Sharpe/drawdown metrics.

**`src/live/`** — Live trading subsystem (runs as separate daemon process):
- `poll_loop.py`: subscribes to IBKR live bars, runs feature engineering and prediction each completed bar
- `engine.py`: main trading loop
- `reconnect.py`: `ReconnectController` handles TWS/Gateway disconnects and position reconciliation
- `state.py`: explicit `StateMachine` (IDLE → CONNECTED → TRADING, with RECONNECTING)
- `alerts.py`: `AlertManager` monitors bracket order health and stale data
- `persistence.py`: MD5 deduplication to prevent duplicate bar processing

**`src/jobs/`** — Async job system decoupling Streamlit from long-running tasks (train, backtest, optimize). Each job: `runs/<job_id>/` with `request.json`, `status.json` (QUEUED|RUNNING|SUCCEEDED|FAILED), `result.json`, and output artifacts.

**`src/ui/`** — Streamlit UI. Polls job status files, reads `ui_state/live/*.jsonl` event logs for live trading view. `page_modules/` has per-workflow pages.

### Model Registry
- Training saves timestamped `.keras` files to `models/registry/`
- `models/active_model.txt` tracks current active model path
- `best_hyperparameters.json` stores tuned hyperparameters from grid search

### Live Trading Operational State
- Events logged as append-only JSONL in `ui_state/live/`
- Kill switch: creating `ui_state/live/KILL_SWITCH` file disables order submission cleanly
- Live daemon is intentionally separate from Streamlit (resilient to UI crashes)

### Process Model
- **Streamlit UI**: in-process, single-user
- **Live Trading**: separate subprocess daemon (client-id 2 by default)
- **Jobs**: spawned subprocesses (non-blocking for UI)
- **API**: optional FastAPI wrapper (`api/main.py`, port 8000)

## Important Design Decisions

- The live daemon (`src/ibkr_live_session.py`) is the entry point; it owns the IBKR connection and spawns the `live/` subsystem. Do not conflate it with the Streamlit process.
- `src/data_ingestion.py` and `src/daily_data_agent.py` require a live IBKR TWS/Gateway connection. Use `--skip-ingestion` for offline development.
- The job system (`src/jobs/`) is filesystem-backed with no database — job state lives in `runs/`. This is intentional for simplicity.
- Stateful LSTM requires batch size consistency between training and inference; see `src/model.py` and `src/predict.py` for how state is reset per sequence.
- Walk-forward validation (`scripts/run_walkforward_backtest.py`) is the primary validation strategy — not a simple train/test split.
