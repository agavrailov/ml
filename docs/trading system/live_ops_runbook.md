# Live Trading System - Operations Runbook

## Quick Start

### System Health Check (5-second glance)

**Option 1: Streamlit Dashboard (Primary)**
1. Open browser to `http://localhost:8501`
2. Navigate to "Live Ops Dashboard" tab
3. Check status banner at top:
   - ✓ Green "System Healthy" = all good
   - ⚠️ Yellow with alert count = needs attention
   - ❌ Red "Connection Issue" = critical problem
   - 🛑 Orange "Kill Switch ENABLED" = trading paused

**Option 2: CLI Status Check (No Browser)**
```powershell
python -m scripts.status
```

Output shows:
- System state (TRADING/RECONNECTING/etc)
- Connection status and uptime
- Last bar age (should be < 5min for 4h bars)
- Active alerts count
- Position status

**Option 3: Status File (Scripting/Monitoring)**
```powershell
cat ui_state/live/status.json
```

## System States

The system operates with explicit states (no more boolean soup):

| State | Meaning | Action Required |
|-------|---------|----------------|
| **INITIALIZING** | System starting up | Wait 5-10 seconds |
| **CONNECTING** | Establishing IB connection | Wait, should transition quickly |
| **CONNECTED** | Connected to IB Gateway | Normal (subscribing to data) |
| **SUBSCRIBED** | Receiving market data | Normal (waiting for bars) |
| **TRADING** | Actively trading | ✓ Healthy state |
| **DISCONNECTED** | Lost connection | Check if gateway is running |
| **RECONNECTING** | Attempting to reconnect | Wait (auto-recovery) |
| **WAITING_FOR_GATEWAY** | Gateway restart detected | Wait ~30s for gateway to come back |
| **FATAL_ERROR** | Unrecoverable error | ❌ Manual intervention required |
| **STOPPED** | Clean shutdown | Restart if needed |

## Common Issues & Fixes

### Issue: "Connection Issue" banner / DISCONNECTED state

**Symptom**: Status shows DISCONNECTED or RECONNECTING

**Diagnosis**:
1. Check if IB Gateway is running:
   ```powershell
   Get-Process | Where-Object {$_.ProcessName -like "*ibgateway*"}
   ```

2. Check console output for error messages

**Fix**:
- If gateway crashed: Restart IB Gateway, system will auto-reconnect
- If network issue: System will auto-retry with exponential backoff
- If persists >5min during market hours: Check IB account status

### Issue: Stale Data (last bar >10 min ago for 4h bars)

**Symptom**: Last bar age exceeds expected frequency

**Diagnosis**:
1. Check if market is open (system trades NVDA on 4h bars)
2. Check IB Gateway data subscriptions
3. Review alerts for "stale_data" type

**Fix**:
- During market hours: Check IB data permissions
- Off market hours: Normal, ignore
- If persists: Restart session

### Issue: Missing Bracket Orders Alert

**Symptom**: Alert shows "missing_bracket_orders"

**Diagnosis**: System opened position but bracket orders not detected

**Fix**:
1. Check broker snapshot in Streamlit:
   - "Broker snapshot" section shows open orders
   - If position exists but no orders: CRITICAL
2. Manual action: Place stop-loss/take-profit orders manually via TWS
3. Investigate: Check logs for order submission errors

### Issue: Gateway Restart (weekends ~11 PM EST)

**Symptom**: State shows WAITING_FOR_GATEWAY

**Expected Behavior**: 
- IB Gateway restarts automatically (especially weekends)
- System detects restart and waits ~30s
- Auto-reconnects when gateway is back

**Action**: None required unless stuck >5 minutes

### Issue: No Bars After Wake-from-Sleep

**Symptom**: System shows TRADING state but no bars received for >5 minutes during market hours

**Diagnosis**: HMDS data farm went inactive (error 2107) after computer wake

**Expected Behavior**:
- System automatically activates HMDS after reconnection
- Bars should resume within 2-3 minutes
- Stale data detection triggers reconnection if HMDS stays inactive >10 min

**Manual Fix** (if stale data persists >15 min):
1. Check IB Gateway is logged in
2. Restart live session (system will force HMDS activation)
3. If recurring: Check IB data permissions

### Issue: Kill Switch Enabled

**Symptom**: Banner shows "Kill Switch ENABLED"

**Cause**: Operator manually enabled kill switch OR automated safety trigger

**Fix**:
1. Verify you want to resume trading
2. In Streamlit, click "Disable kill switch" button
3. System will resume trading on next bar

## Monitoring Checklist

### Daily (Market Hours)
- [ ] Check status banner (green = healthy)
- [ ] Verify last bar age < 10 minutes
- [ ] Check position matches expectations
- [ ] Review any new alerts

### Weekly
- [ ] Review alerts.jsonl for patterns
- [ ] Check total bars received vs expected (4h × trading hours)
- [ ] Verify bracket orders exist for all positions
- [ ] Review session notes

### Monthly
- [ ] Archive old session logs
- [ ] Review state transition times in logs
- [ ] Check reconnection frequency (should be low)

## Alert Types

| Alert Type | Severity | Meaning | Action |
|------------|----------|---------|--------|
| `missing_bracket_orders` | CRITICAL | Position without stop/limit | Manually place orders |
| `stale_data` | WARNING | No bars for >2× frequency | Check IB connection |
| `gateway_restart` | INFO | Gateway restarting | Wait for auto-reconnect |
| `position_mismatch` | CRITICAL | Position reconciliation failed | Review broker snapshot |

## Files & Locations

| File | Purpose | Location |
|------|---------|----------|
| status.json | Current system status | `ui_state/live/status.json` |
| alerts.jsonl | Alert history | `ui_state/live/alerts.jsonl` |
| last_bar.json | Persistent bar dedup | `ui_state/live/last_bar.json` |
| live_*.jsonl | Event log (per session) | `ui_state/live/live_NVDA_240min_*.jsonl` |
| kill_switch.flag | Kill switch state | `ui_state/live/kill_switch.flag` |

## Starting/Stopping Sessions

### Start Live Session
```powershell
# NVDA on 4-hour bars (default)
python -m src.ibkr_live_session --symbol NVDA --frequency 240min

# With custom equity
python -m src.ibkr_live_session --symbol NVDA --frequency 240min --initial-equity 100000
```

### Stop Live Session
1. **Graceful**: Press Ctrl+C in console (system will finish current bar processing)
2. **Emergency**: Enable kill switch in Streamlit (blocks new orders immediately)
3. **Force**: Kill process (not recommended - may leave orphaned orders)

## Emergency Procedures

### Critical Position Issue
1. Enable kill switch immediately (Streamlit or create `ui_state/live/kill_switch.flag`)
2. Open TWS and manually manage position
3. Review logs and alerts to understand cause
4. Create note in Streamlit explaining actions taken

### System Unresponsive
1. Check if process is running: `Get-Process python`
2. Check IB Gateway process
3. Review last 20 lines of console output
4. If frozen >5min: Kill and restart both gateway and session

### Gateway Login Issues
- IB Gateway may require 2FA re-login
- Check gateway window for prompts
- Session will auto-reconnect once gateway is authenticated

## Notes & Annotations

Use the "Session notes" section in Streamlit to document:
- Why you paused/resumed trading
- Parameter changes
- Incident context
- Gateway restarts
- Manual interventions

Notes are appended to the session JSONL log for audit trail.

## Troubleshooting Logs

### Console Output
- Shows state transitions only (minimal noise)
- Errors printed with traceback
- Location: Console where session was started

### Event Logs (JSONL)
- Complete audit trail
- Search by order_id, bar_time, or event type
- Export filtered view from Streamlit "Audit log" section

### Status Updates
- Real-time: `python -m scripts.status`
- Historical: Read past events from session JSONL

## Performance Metrics

**Expected Behavior** (NVDA 4h bars):
- Bars received: 2-3 per trading day
- Reconnections: <1 per week (except scheduled gateway restarts)
- Alerts: 0 during normal operation
- Order submission latency: <2 seconds from bar close

**Anomaly Triggers**:
- >5 reconnections per day = investigate IB connection quality
- Stale data >30 min during market hours = critical issue
- Missing brackets alert = immediate action required

## Contact & Escalation

If you encounter issues not covered in this runbook:
1. Check alerts.jsonl for additional context
2. Review architecture docs: `docs/trading_system/live_architecture.md`
3. Search event logs for similar patterns
4. Document issue in session notes for future reference
