# Trading System Dashboard UX Vision (Reference)
Purpose
This document captures the long-term UX vision for a “Live Ops Dashboard” in the Streamlit UI, plus incremental steps to reach it without over-building.

## North star
The Live tab is the main operational surface during market hours. It should answer, in order:
1. Is the system safe and healthy right now?
2. What did it just decide and why?
3. What did we send to the broker and what actually happened?
4. Can we reconstruct the entire session later for debugging?

## Information hierarchy
Tier 0 (always visible)
Safety / state
- Trading mode (SIM vs IBKR)
- Kill switch status (armed/disarmed)
- Connectivity health (data feed + broker), with last heartbeat time
Exposure
- Current positions (qty, avg price) per symbol
- Open orders count (and any rejected)
- Last action summary: last bar time, last decision, last order id

Tier 1 (visible but secondary)
PnL
- Today realized / unrealized (or clearly “not available yet” until fills/PnL are implemented)
- Equity/session curve (even coarse)
Model + data sanity
- Latest prediction value and confidence proxy (sigma / residual)
- Data freshness (time since last bar)
- Consecutive “no trade” count (helps spot stalled logic)

Tier 2 (drill-down)
Auditability
- Event log timeline: bar → prediction → decision → order → fill
- Raw broker snapshots
- Errors with full stack traces
- Export / download buttons (CSV/JSONL)

## Proposed Live tab layout
Header row (session control)
- Session selector (current + prior sessions)
- Refresh now
- Kill switch (big, obvious): stop sending new orders
- Optional: “Annotate session” (append a note event to the log)

Row 1: KPI cards (Tier 0)
- Engine status: RUNNING / STOPPED
- Data feed: OK / STALE (last bar timestamp)
- Broker: CONNECTED / DISCONNECTED
- Exposure: position summary + open orders
- Last decision: NO TRADE vs LONG/SHORT size=…
- Last order: order_id + status if known

Row 2: “What just happened?” (Tier 1)
Two panes:
- Recent decisions (last N bars): time, predicted vs close, decision, reason
- Orders (last N): submitted/cancelled/rejected (later: filled/partial)
Design intent: skimmable default; full detail via row expansion/expander.

Row 3: Charts (Tier 1, collapsible)
- Price with decision/order markers
- Prediction vs price (or predicted return)
- PnL/equity curve (when available)

Row 4: Audit log (Tier 2)
- Filter by event type (decision/order/error)
- Search by order_id
- Export filtered view

## User journey
Start of day
- Open Live tab → see SAFE / CONNECTED / NO POSITION / NO OPEN ORDERS
- Start session → record run metadata (model version, strategy params)

During session
- Mostly live in KPI cards + recent decisions
- Drill into Orders or Audit log when anything looks odd

Incident
- Hit kill switch
- Immediately confirm: order submission disabled
- (Future) cancel open orders / flatten

After session / debugging
- Select a prior run
- Replay timeline + charts
- Export for offline analysis

## Incremental delivery plan (small steps → vision)
Step 1 (minimal, highest ROI)
- Persist live-session events to disk (append-only, crash-safe)
- Live tab is read-only: status + recent events + error panel
- All KPI cards derived from the log (no broker polling required yet)

Step 2 (next value)
- Periodically log broker snapshots: open orders, positions, account summary
- UI shows “broker truth” next to “strategy thinks”

Step 3 (bigger)
- Log fills and compute realized/unrealized PnL
- Add PnL KPI cards and equity chart

## Logging format recommendation
Use JSONL (newline-delimited JSON) per run:
- Safe for concurrent append/read between live runner and Streamlit UI
- Tolerates partial writes (ignore last incomplete line)
Suggested location: ui_state/live/
Suggested event types: run_start, bar, decision, order_submitted, broker_snapshot, error, run_end
