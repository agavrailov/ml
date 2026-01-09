# Technical debt
This file tracks known complexity/tech-debt items that were introduced intentionally to ship functionality.

## IBKR live reconnect controller (2025-12-19) - ✅ RESOLVED
**Original Issue**: Reconnect logic embedded in closures, hard to test.

**Resolution (January 2026)**: 
- ✅ Extracted `ReconnectController` to `src/live/reconnect.py` (testable class)
- ✅ Extracted `ConnectionManager` for client ID discovery
- ✅ Implemented persistent bar deduplication via MD5 hash (`src/live/persistence.py`)
- ✅ Added explicit state machine (`src/live/state.py`) replacing boolean flags
- ✅ Implemented position reconciliation on reconnect

**Remaining**: Unit tests for ReconnectController (deferred but low risk)

---

## Live Trading Observability Integration (January 2026) - ✅ COMPLETED

**Completed Work**:
- ✅ StatusManager fully wired up in heartbeat loop (updates every 30s)
- ✅ All 4 alert types integrated:
  - missing_brackets (CRITICAL)
  - stale_data (WARNING)
  - keepuptodate_stale (CRITICAL)
  - gateway_restart (INFO)
- ✅ Console output reduced to essential messages only
- ✅ Streamlit UI displays status banner and alerts
- ✅ CLI status script (`python -m scripts.status`) functional
- ✅ Comprehensive documentation (runbook + architecture)

**Status**: Production-ready with complete monitoring

---

## HMDS Inactivity Fix (January 2026) - ✅ COMPLETED

**Problem**: After computer wake-from-sleep or reconnection, IBKR's HMDS (historical data) farm goes inactive (error 2107). The `keepUpToDate` subscription remains alive but stops delivering bars indefinitely.

**Root Cause**: HMDS farm enters "inactive but available upon demand" state. System was not requesting data to trigger activation.

**Solution Implemented**:
- ✅ Created `_activate_hmds()` helper function (ibkr_live_session.py:549-606)
- ✅ Call HMDS activation after every reconnection (line 1321)
- ✅ Reduced stale data threshold from `max(5min, 1x bar period)` to `min(10min, max(5min, 0.5x bar period))`
- ✅ Updated documentation (live_architecture.md, live_ops_runbook.md)

**Impact**:
- Before: 4-hour bars could wait up to 4 hours for stale data detection
- After: Bars resume within 2-3 minutes via HMDS activation; max 10-minute detection if activation fails

**Status**: Deployed and ready for testing

---

## Unit Test Coverage - DEFERRED

**Priority**: Medium

**Missing Tests**:
- `tests/test_reconnect_controller.py` - ReconnectController class
- `tests/test_state_machine.py` - StateMachine transitions
- `tests/test_bar_tracker.py` - PersistentBarTracker
- `tests/test_status_manager.py` - StatusManager
- `tests/test_alert_manager.py` - AlertManager

**Rationale**: Working code prioritized over test-first development to complete observability integration.

**Estimated Effort**: 2-3 days for complete coverage

**Risk**: Low - Components are simple and manually tested via integration

---

## Position Reconciliation Enhancements

**Priority**: Low

**Current State**: Basic reconciliation exists (line 1217-1221 in ibkr_live_session.py)

**Potential Improvements**:
1. Orphaned position detection (positions without strategy state)
2. Automated recovery (close orphaned positions, place missing brackets)
3. Quantity mismatch handling

**Complexity**: High - introduces decision-making logic that could cause unexpected behavior

**Recommendation**: Only implement if orphaned positions become a recurring issue

---

## Code Complexity in run_live_session()

**Priority**: Low

**Current State**: ~1200 lines with nested closures

**Recommendation**: Extract smaller functions for better testability

**Risk**: Low - code works, refactoring might introduce bugs

**Estimated Effort**: 3-4 days for careful extraction

**Value**: Moderate - mainly improves maintainability

---

## Multi-Symbol Support

**Priority**: Deferred

**Current State**: One symbol per session

**Blocker**: Architecture assumes single symbol throughout (predictor, strategy, logs)

**Effort**: Large refactoring (~2-3 weeks)

**Recommendation**: Run multiple parallel sessions for multi-symbol trading
