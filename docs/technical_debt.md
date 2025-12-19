# Technical debt
This file tracks known complexity/tech-debt items that were introduced intentionally to ship functionality.

## IBKR live reconnect controller (2025-12-19)
We added an in-process reconnect loop in `src/ibkr_live_session.py` by wiring `ib.disconnectedEvent` to:
- pause trading (`trading_enabled=False`)
- reconnect with backoff
- re-qualify contract
- resubscribe `reqHistoricalData(..., keepUpToDate=True)`

Why this is debt
- The reconnect logic is embedded inside `run_live_session()` with several closure variables, which makes it harder to unit test and reason about.
- Bar deduplication is best-effort (based on `bar_time_iso` equality) and may not prevent all duplicate-trade edge cases after reconnect.

Suggested follow-ups
- Extract a small `ReconnectController` helper (pure Python, dependency-injected) to remove closure complexity.
- Introduce explicit idempotency keys for trade decisions (persist last processed completed bar timestamp + decision hash to disk).
- On reconnect, reconcile state from broker more fully (open orders/executions) and rebuild `has_open_position` from that source of truth.
