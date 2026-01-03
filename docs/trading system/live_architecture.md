# Live Trading System - Architecture

## Overview

The live trading system is an LSTM-based automated trading engine that connects to Interactive Brokers via IB Gateway, processes real-time market data, makes trading decisions, and executes orders with risk management.

**Key Design Principles:**
- Single IB connection shared across all components (market data + execution)
- Explicit state machine (no boolean flag soup)
- Persistent deduplication (survives restarts)
- Observable at all times (status files + UI + logs)
- Graceful reconnection with market hours awareness

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Operator Interface                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Streamlit   │  │  CLI Status  │  │ status.json  │     │
│  │  Dashboard   │  │   (scripts)  │  │  (polling)   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  Observability Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │StatusManager │  │AlertManager  │  │   EventLog   │     │
│  │(status.json) │  │(alerts.jsonl)│  │  (JSONL)     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Core Trading Loop                         │
│                  (src/ibkr_live_session.py)                  │
│                                                              │
│  ┌──────────────────────────────────────────┐              │
│  │          State Machine                    │              │
│  │  INITIALIZING → CONNECTING → CONNECTED    │              │
│  │       → SUBSCRIBED → TRADING              │              │
│  └──────────────────────────────────────────┘              │
│                           │                                  │
│  ┌────────────┬───────────┴────────┬────────────┐          │
│  │            │                    │            │          │
│  ▼            ▼                    ▼            ▼          │
│ ┌────┐  ┌─────────┐  ┌──────────┐  ┌──────────┐          │
│ │IB  │  │Reconnect│  │Persistent│  │ Live     │          │
│ │Loop│  │Controller│  │BarTracker│  │Predictor │          │
│ └────┘  └─────────┘  └──────────┘  └──────────┘          │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  IB Gateway / TWS                            │
│              (Interactive Brokers API)                       │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. State Machine (`src/live/state.py`)

**Purpose**: Replace scattered boolean flags with explicit state tracking

**States**:
- `INITIALIZING` - System startup
- `CONNECTING` - Establishing IB connection
- `CONNECTED` - IB connection established
- `SUBSCRIBED` - Market data subscription active
- `TRADING` - Actively making trading decisions
- `DISCONNECTED` - Lost connection (transient)
- `RECONNECTING` - Attempting reconnection
- `WAITING_FOR_GATEWAY` - Detected gateway restart
- `FATAL_ERROR` - Unrecoverable error (human intervention needed)
- `STOPPED` - Clean shutdown

**Key Methods**:
- `transition_to(new_state, reason)` - Change state with logging
- `can_trade()` - Returns True only in TRADING state
- `is_running()` - True unless STOPPED or FATAL_ERROR

**Design Notes**:
- All transitions logged to JSONL with timestamps
- Tracks time spent in each state
- Thread-safe for concurrent status checks

### 2. Reconnection Controller (`src/live/reconnect.py`)

**Purpose**: Testable reconnection logic with exponential backoff and market awareness

**Components**:

**ConnectionManager** (static methods):
- `connect_with_unique_client_id()` - Try clientIds until one succeeds
- Handles IB error 326 (clientId already in use)
- Probes connection validity with `reqCurrentTime()`

**ReconnectController**:
- Exponential backoff: 1s → 2s → 4s → ... → 30min (capped)
- Market hours awareness: longer backoff when market closed
- Gateway restart detection: rapid failures trigger wait state
- Health monitoring: tracks failure timestamps

**Backoff Strategy**:
```python
# During market hours
backoff = min(30, 2 ** attempt_count)  # 1, 2, 4, 8, 16, 30s

# Outside market hours
backoff = min(1800, backoff * 60)  # 60, 120, ... 1800s (30min)
```

### 3. Observability System

**4-Tier Architecture** (from most to least important):

#### Tier 0: Streamlit Dashboard (Primary Interface)
- **Location**: `src/ui/page_modules/live_page.py`
- **Purpose**: Real-time operator visibility
- **Features**:
  - Status banner (health at a glance)
  - Active alerts display (critical/warning)
  - KPI cards (engine, data feed, broker, position)
  - Broker snapshot (positions, orders, account)
  - Charts (price, predictions, trade markers)
  - Audit log (searchable, exportable)
- **Auto-refresh**: 30 seconds (configurable)

#### Tier 1: Status File (`ui_state/live/status.json`)
- **Purpose**: Headless monitoring (scripts, cron jobs)
- **Updated by**: StatusManager (every heartbeat)
- **Schema**:
```json
{
  "state": "TRADING",
  "connection": {
    "status": "CONNECTED",
    "uptime_minutes": 123.4
  },
  "data_feed": {
    "last_bar_time": "2025-01-03T18:00:00Z",
    "age_minutes": 2.3
  },
  "position": {
    "status": "OPEN",
    "quantity": 10
  },
  "alerts": {
    "count": 0
  },
  "kill_switch": false,
  "updated_at": "2025-01-03T18:05:00Z"
}
```

#### Tier 2: Alerts (`ui_state/live/alerts.jsonl`)
- **Purpose**: Actionable issues only (no noise)
- **Managed by**: AlertManager
- **Deduplication**: By alert key (avoid spam)
- **Severities**: INFO, WARNING, CRITICAL
- **Format**: One JSON object per line
```json
{"ts": "2025-01-03T18:00:00Z", "severity": "CRITICAL", "type": "missing_bracket_orders", "msg": "Position without brackets", "key": "brackets_123"}
```

#### Tier 3: Event Log (`ui_state/live/live_{symbol}_{freq}_{run_id}.jsonl`)
- **Purpose**: Complete audit trail
- **Event types**: bar, decision, order_submitted, order_status, fill, error, state_transition, etc.
- **Retention**: Keep all session logs indefinitely (compressed)

#### Tier 4: Console Output
- **Purpose**: Real-time debugging (minimal)
- **Content**: State transitions and errors only

### 4. Persistent Bar Tracker (`src/live/persistence.py`)

**Purpose**: Bar deduplication that survives restarts

**Problem Solved**: IB Gateway may re-deliver bars after reconnection

**Implementation**:
- Compute MD5 hash of bar OHLC+timestamp
- Store in `ui_state/live/last_bar.json`
- Check hash before processing new bars

**Schema**:
```json
{
  "bar_time": "2025-01-03T18:00:00Z",
  "hash": "abc123def456...",
  "symbol": "NVDA",
  "frequency": "240min"
}
```

### 5. Live Predictor (`src.lstm_live_predictor`)

**Purpose**: Real-time LSTM inference

**Features**:
- Buffer management (24 bars for LSTM context)
- CSV warm-up (load historical bars on startup)
- Prediction caching (avoid recomputing same bar)

**Integration**:
- Called once per bar close
- Returns predicted next-bar price
- Decision logic uses prediction + current price

## Data Flow

### Startup Sequence
1. `INITIALIZING` - Load config, create components
2. `CONNECTING` - ConnectionManager finds free clientId
3. `CONNECTED` - Attach error handlers, qualify contracts
4. `SUBSCRIBED` - Subscribe to market data (reqHistoricalData)
5. Warm up predictor from CSV (24 bars)
6. `TRADING` - Start heartbeat loop, process bars

### Bar Processing
1. IB delivers bar via callback (`onBarUpdate`)
2. PersistentBarTracker checks hash (dedupe)
3. Log bar event to JSONL
4. LivePredictor makes prediction
5. Decision logic (strategy) determines action
6. Execute order via Broker abstraction
7. StatusManager updates status.json
8. AlertManager checks for issues (missing brackets)

### Reconnection Flow
1. Detect disconnection (ib.disconnectedEvent)
2. StateMachine → DISCONNECTED
3. ReconnectController starts loop:
   - Check market hours
   - Apply backoff
   - Attempt connection
   - Probe with reqCurrentTime()
4. On success:
   - StateMachine → CONNECTED
   - Resubscribe to data
   - StateMachine → SUBSCRIBED → TRADING
5. On repeated failure:
   - Increase backoff
   - Check for gateway restart pattern
   - StateMachine → WAITING_FOR_GATEWAY if detected

## Configuration

### IB Connection (`src/config.py` → `IbConfig`)
- **Host**: `127.0.0.1` (localhost)
- **Port**: `4002` (IB Gateway paper trading default)
- **ClientId**: Auto-incremented until free slot found

### Trading Parameters (`src/ibkr_live_session.py`)
- **Symbol**: NVDA (default, configurable via CLI)
- **Frequency**: 240min (4-hour bars)
- **LSTM tsteps**: 24 bars lookback
- **Initial equity**: $100,000 (paper trading)

### Observability
- **Status update frequency**: Every heartbeat (~5s)
- **Alert deduplication**: By unique key
- **Log retention**: Indefinite (manual cleanup)

## Threading Model

**Single-threaded event loop** with background reconnection thread:

- **Main thread**: IB event loop (`ib.run()`)
  - Bar callbacks
  - Error handlers
  - Disconnection events

- **Reconnection thread**: Spawned only when disconnected
  - Runs ReconnectController loop
  - Terminates on successful reconnection
  - Uses `threading.Event` for coordination

**Synchronization**:
- All state transitions thread-safe (simple assignments)
- EventLog writes atomic (line-buffered JSONL)
- StatusManager uses temp-file-rename for atomicity

## Error Handling

### Levels
1. **Transient** (auto-recovery):
   - Connection drops → RECONNECTING
   - Gateway restart → WAITING_FOR_GATEWAY
   - Timeouts → Retry with backoff

2. **Degraded** (continue with alert):
   - Missing bracket orders → Alert + continue
   - Stale data → Alert + wait

3. **Fatal** (human intervention):
   - Authentication failure → FATAL_ERROR
   - Account locked → FATAL_ERROR
   - Unhandled exception in bar handler → FATAL_ERROR

### Kill Switch
**Purpose**: Emergency trading halt

**Trigger**:
- Manual (Streamlit button)
- Automated (future: position limit exceeded)

**Effect**:
- Decision logic checks `is_kill_switch_enabled()`
- If enabled: Log decision as "blocked_by_kill_switch"
- No orders submitted

**State**: Stored in `ui_state/live/kill_switch.flag`

## Testing Strategy

### Unit Tests
- `tests/test_reconnect_controller.py` - Reconnection logic
- `tests/test_state_machine.py` - State transitions
- `tests/test_bar_tracker.py` - Deduplication

### Integration Tests
- `tests/test_ibkr_live_session.py` - End-to-end with mock IB

### Manual Testing
- Gateway restart simulation (kill gateway mid-session)
- Network interruption (disable wifi)
- Duplicate bar delivery (reconnect during bar)

## Deployment

### Prerequisites
- Python 3.10+
- IB Gateway or TWS running
- Valid IB account with data subscriptions

### Files to Persist
- `ui_state/live/` directory (status, alerts, bar tracker)
- Session JSONL logs (archive periodically)
- Model weights (loaded on startup)

### Monitoring
- Cron job: `python -m scripts.status` every 5 minutes
- Alert on: state=FATAL_ERROR, alert_count>0, last_bar_age>30min

## Future Enhancements

### Planned
1. **More Alert Triggers**:
   - Stale data detection (already in ibkr_live_session.py, needs AlertManager integration)
   - Gateway restart detection (already tracked, needs alert)
   - Position reconciliation failures

2. **Tests** (Phase 1.3):
   - Unit tests for ReconnectController
   - Mock IB connection for integration tests

3. **Performance Metrics**:
   - Order latency tracking
   - Bar processing time histogram
   - Reconnection frequency dashboard

### Possible
- Multi-symbol support (current: single symbol per session)
- Dynamic position sizing based on volatility
- WebSocket API for external monitoring
- Telegram/SMS alerts for critical issues

## File Structure

```
src/
├── ibkr_live_session.py       # Main entry point
├── live/
│   ├── state.py               # StateMachine, SystemState enum
│   ├── reconnect.py           # ReconnectController, ConnectionManager
│   ├── status.py              # StatusManager (status.json)
│   ├── alerts.py              # AlertManager (alerts.jsonl)
│   ├── persistence.py         # PersistentBarTracker
│   └── contracts.py           # Event schemas (StateTransitionEvent, etc)
├── lstm_live_predictor.py     # Real-time LSTM inference
└── ui/page_modules/live_page.py  # Streamlit dashboard

scripts/
└── status.py                  # CLI status checker

ui_state/live/
├── status.json                # Current system status
├── alerts.jsonl               # Alert history
├── last_bar.json              # Persistent bar deduplication
└── live_NVDA_240min_*.jsonl   # Session event logs

docs/trading_system/
├── live_architecture.md       # This document
└── live_ops_runbook.md        # Operator procedures
```

## References

- [IB API Documentation](https://interactivebrokers.github.io/tws-api/)
- [ib_insync Library](https://ib-insync.readthedocs.io/)
- Original refactoring plan: `[internal plan document]`
