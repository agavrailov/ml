# Multi-Symbol Trading Architecture

**Audience:** Developer onboarding, debugging, or extending the trading system  
**Last updated:** 2026-04-16

---

## 1. Overview

The system runs one independent trading daemon per symbol. Each daemon owns its own IBKR connection, runs its own LSTM model, and manages its own bracket orders. Daemons coordinate capital allocation through a single shared JSON state file written atomically after each trade.

```
scripts/launch_portfolio.py
  │
  ├── subprocess: src.ibkr_live_session --symbol NVDA --client-id 10
  │     └── src/live/poll_loop.py   (LSTM → signal → bracket order)
  │
  ├── subprocess: src.ibkr_live_session --symbol MSFT --client-id 11
  │     └── src/live/poll_loop.py
  │
  └── shared state: ui_state/portfolio/state.json
        (each daemon reads before sizing; writes after trading)
```

**Key design decisions:**
- **Process isolation** — a crash or IBKR disconnect in one symbol does not affect others
- **No shared memory** — coordination is file-based, not in-process
- **Graceful degradation** — if the shared state file is missing or corrupt, each daemon falls back to broker-reported buying power and continues trading

---

## 2. Configuration

### `configs/portfolio.json`

The single source of truth for which symbols are active:

```json
{
  "_comment": "Edit to add/remove symbols. base_client_id: NVDA=10, MSFT=11, etc.",
  "symbols": ["NVDA", "MSFT"],
  "frequency": "60min",
  "tsteps": 5,
  "base_client_id": 10,
  "allocation": {
    "max_gross_exposure_pct": 0.80,
    "max_per_symbol_pct": 0.30
  }
}
```

| Field | Description |
|---|---|
| `symbols` | Ordered list — index determines client ID offset |
| `frequency` | Bar size for all symbols (all must share one frequency) |
| `tsteps` | LSTM lookback window |
| `base_client_id` | IBKR client ID for symbol[0]; symbol[i] gets `base + i` |
| `max_gross_exposure_pct` | Max fraction of account equity across all open positions |
| `max_per_symbol_pct` | Max fraction of account equity in any single symbol |

### Per-symbol strategy overrides

Each symbol can override any `StrategyConfig` field via:

```
configs/symbols/{SYMBOL}/active.json
```

Resolution order (highest priority first):
1. `configs/symbols/NVDA/active.json` — per-symbol override
2. `configs/active.json` — global active config
3. `src/config.py` `StrategyDefaultsConfig` — code defaults

Typical per-symbol file:
```json
{
  "strategy": {
    "k_sigma_long": 0.35,
    "k_atr_long": 0.45,
    "k_sigma_short": 0.50,
    "k_atr_short": 0.50
  }
}
```

### Contract registry

All tradable symbols must be registered in `src/config.py`:

```python
CONTRACT_REGISTRY: dict[str, dict] = {
    "NVDA": {"symbol": "NVDA", "exchange": "SMART", "currency": "USD", "secType": "STK"},
    "MSFT": {"symbol": "MSFT", "exchange": "SMART", "currency": "USD", "secType": "STK"},
    # Add new symbols here before adding to portfolio.json
}
```

---

## 3. Launcher — `scripts/launch_portfolio.py`

Reads `portfolio.json`, spawns one `src.ibkr_live_session` subprocess per symbol, and monitors them.

```
python -m scripts.launch_portfolio            # start all daemons
python -m scripts.launch_portfolio --dry-run  # print commands only
```

**Client ID assignment:**
```
NVDA → base_client_id + 0 = 10
MSFT → base_client_id + 1 = 11
JPM  → base_client_id + 2 = 12   (future)
XOM  → base_client_id + 3 = 13   (future)
```

**Crash recovery:** the monitor loop (`time.sleep(15)`) checks each process every 15 seconds. If a daemon exits with a non-zero code, the launcher waits 30 seconds then restarts it automatically.

**Shutdown:** `Ctrl-C` or `SIGTERM` sends `terminate()` to all child processes and exits cleanly.

---

## 4. Per-Symbol Daemon — `src/ibkr_live_session.py`

Each daemon runs a single-symbol poll loop indefinitely. It:

1. Loads the LSTM model for the configured frequency
2. Waits for the next 60-min bar boundary
3. Connects to IBKR TWS/Gateway with its unique `client_id`
4. Fetches 7 days of historical bars for its symbol
5. Runs the LSTM predictor, evaluates strategy, sizes and submits a bracket order (if signalled)
6. Updates the shared portfolio state file
7. Disconnects and sleeps until the next bar

The daemon is **stateless between bars** — IBKR connection is fresh each cycle. This means a PC sleep or network dropout only loses at most one bar.

---

## 5. Capital Allocation — How Daemons Share Equity

### Shared state file

Path: `ui_state/portfolio/state.json`

```json
{
  "updated_utc": "2026-04-16T10:00:00Z",
  "total_equity": 50000.0,
  "positions": {
    "NVDA": {"quantity": 10, "market_value": 1500.0},
    "MSFT": {"quantity": 0,  "market_value": 0.0}
  }
}
```

**Writes** happen after every successful bracket order submission.  
**Reads** happen before every position sizing calculation.  
**Atomicity** is guaranteed via temp-file swap (`os.replace`) — no partial reads.

### `PortfolioStateManager` — `src/portfolio/state.py`

| Method | Description |
|---|---|
| `read()` | Returns current state dict; returns empty template if file absent or corrupt |
| `write_equity(equity)` | Updates `total_equity` field |
| `write_position(symbol, qty, mv)` | Updates one symbol's position in `positions` |
| `available_capital_for(symbol, allocator)` | Reads state, asks `CapitalAllocator`, returns float |

### `CapitalAllocator` — `src/portfolio/capital_allocator.py`

Enforces two constraints and returns the binding (smaller) limit:

```python
gross_headroom  = max_gross_exposure_pct × equity  −  Σ|all open market values|
symbol_headroom = max_per_symbol_pct × equity      −  |this symbol's market value|

available = min(gross_headroom, symbol_headroom)
```

**Example** (equity=$50k, NVDA open $12k, MSFT open $8k, limits 80%/30%):
```
gross_cap    = 0.80 × 50000 = 40000
total_open   = 12000 + 8000 = 20000
gross_hdroom = 40000 − 20000 = 20000

per_sym_cap  = 0.30 × 50000 = 15000
nvda_hdroom  = 15000 − 12000 = 3000

→ NVDA can deploy up to $3000 more (per-symbol cap is binding)
```

### Integration in `poll_loop.py`

```python
# 1. Query real equity + broker buying power
acct = broker.get_account_summary()
real_equity = acct["NetLiquidation"]
buying_power = acct["BuyingPower"]

# 2. Further cap to portfolio allocation
if portfolio_state_path is not None:
    _ps    = PortfolioStateManager(portfolio_state_path)
    _alloc = make_allocator_from_config(portfolio_config_path)
    _cap   = _ps.available_capital_for(cfg.symbol, _alloc)
    buying_power = min(buying_power, _cap)   # binding constraint

# 3. Pass to strategy (position sizing uses buying_power as hard limit)
state = StrategyState(..., buying_power=buying_power)
```

**Graceful degradation:** if any step above raises an exception, it is silently caught and trading continues using only the broker's reported buying power. No daemon crashes due to portfolio coordination failures.

---

## 6. Full Bar Cycle — Data Flow

```
Bar closes (top of hour)
│
├─ NVDA daemon wakes
│   ├─ Connect to IBKR (client_id=10)
│   ├─ Fetch 7d NVDA 60-min bars
│   ├─ Run LSTM predictor → predicted_price
│   ├─ Query account: NetLiquidation, BuyingPower
│   ├─ Read ui_state/portfolio/state.json
│   ├─ CapitalAllocator.available_for("NVDA", equity, positions)
│   ├─ compute_tp_sl_and_size(StrategyState) → TradePlan or None
│   ├─ [if plan] submit_trade_plan_bracket() → bracket order IDs
│   ├─ [if plan] PortfolioStateManager.write_position("NVDA", qty, mv)
│   ├─ Log decision + order events to ui_state/live/nvda_*.jsonl
│   └─ Disconnect from IBKR
│
└─ MSFT daemon wakes (same bar boundary, ~seconds later)
    ├─ Connect to IBKR (client_id=11)
    ├─ Fetch 7d MSFT 60-min bars
    ├─ Read ui_state/portfolio/state.json  ← sees NVDA's updated position
    ├─ CapitalAllocator.available_for("MSFT", equity, positions)
    └─ ... (same flow)
```

> ⚠️ Both daemons wake at the same bar boundary. There is a small race window (seconds) where MSFT reads the state file before NVDA has written its new position. This is acceptable — the position it misses was just submitted and will appear on the next bar.

---

## 7. How to Add a New Symbol

### Step 1 — Register the contract (`src/config.py`)

```python
CONTRACT_REGISTRY: dict[str, dict] = {
    ...
    "JPM": {"symbol": "JPM", "exchange": "SMART", "currency": "USD", "secType": "STK"},
}
```

### Step 2 — Add to portfolio (`configs/portfolio.json`)

```json
{
  "symbols": ["NVDA", "MSFT", "JPM"],
  "base_client_id": 10
}
```

JPM will automatically receive `client_id = 12`.

### Step 3 — Ingest historical data (requires IBKR connection)

```bash
python src/daily_data_agent.py --symbol JPM
# or without touching existing data:
python src/daily_data_agent.py --symbol JPM --skip-existing
```

### Step 4 — Train LSTM model

```bash
python src/train.py --symbol JPM --frequency 60min
```

The trained model is saved to `models/registry/` and registered in `models/active_model.txt` (per-symbol).

### Step 5 — Walk-forward backtest

```bash
python scripts/run_walkforward_backtest.py --symbol JPM --frequency 60min
```

Review Sharpe, max drawdown, and trade count. Target: Sharpe ≥ 1.5 before live.

### Step 6 — Create per-symbol strategy config

```bash
mkdir configs/symbols/JPM
```

`configs/symbols/JPM/active.json`:
```json
{
  "strategy": {
    "k_sigma_long": 0.40,
    "k_atr_long": 0.50,
    "k_sigma_short": 0.55,
    "k_atr_short": 0.55,
    "risk_per_trade_pct": 0.01
  }
}
```

Tune `k_sigma` and `k_atr` to the symbol's typical signal-to-noise; JPM and XOM are lower-vol and may need tighter thresholds than NVDA.

### Step 7 — Lower per-symbol cap

With 4+ symbols, reduce `max_per_symbol_pct` so the total can actually fill:

```json
"allocation": {
  "max_gross_exposure_pct": 0.80,
  "max_per_symbol_pct": 0.20
}
```

### Step 8 — Paper trade first

Run for ≥ 4 weeks on IBKR paper account before deploying live capital.

### Step 9 — Restart the launcher

```bash
python -m scripts.launch_portfolio
```

---

## 8. Monitoring & Debugging

### Event logs (per-symbol JSONL)

Each daemon appends structured events to:
```
ui_state/live/{symbol}_events.jsonl
```

Key event types:

| `type` | When emitted |
|---|---|
| `account_equity` | Every bar — real equity + buying power queried |
| `decision` | Every bar — action, predicted return, thresholds, no_trade_reason |
| `order_submitted` | When a bracket order is placed |
| `bracket_check` | AlertManager health check result |

### Portfolio state file

```bash
cat ui_state/portfolio/state.json
```

Shows all open positions and total equity as seen by the last daemon that wrote.

### Common issues

| Symptom | Likely cause | Fix |
|---|---|---|
| Daemon crashes immediately | Symbol not in `CONTRACT_REGISTRY` | Add entry to `src/config.py` |
| Orders rejected by IBKR | Duplicate client_id | Ensure no other IBKR connection uses same `client_id` |
| `available_capital = 0` always | Gross cap already hit, or symbol not in `portfolio.json` symbols list | Check state file; reduce existing positions |
| One symbol trades, others don't | Model file missing for symbol | Run `src/train.py --symbol X` |
| State file not updating | Write exception silently caught | Check `bracket_check` / `order_submitted` events in JSONL |

### Kill switch

Create this file to block all order submissions across all daemons without stopping them:

```bash
touch ui_state/live/KILL_SWITCH
```

Remove it to resume trading:

```bash
rm ui_state/live/KILL_SWITCH
```

---

## 9. Key File Index

| File | Purpose |
|---|---|
| `configs/portfolio.json` | Symbol list, frequency, client IDs, allocation caps |
| `configs/symbols/{SYM}/active.json` | Per-symbol strategy overrides |
| `src/config.py` — `CONTRACT_REGISTRY` | IBKR contract details per symbol |
| `scripts/launch_portfolio.py` | Spawns and monitors one daemon per symbol |
| `src/ibkr_live_session.py` | Single-symbol daemon entry point |
| `src/live/poll_loop.py` | Core trading loop — LSTM → signal → order |
| `src/portfolio/capital_allocator.py` | Gross + per-symbol exposure caps |
| `src/portfolio/state.py` | Atomic read/write of shared position state |
| `src/strategy.py` | Signal threshold logic, TradePlan sizing |
| `ui_state/portfolio/state.json` | Live shared state (written by daemons) |
| `ui_state/live/KILL_SWITCH` | Create this file to halt all order submission |
