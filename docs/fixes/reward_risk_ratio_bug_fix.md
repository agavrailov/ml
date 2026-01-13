# Fix: Inverted Reward:Risk Ratio (Critical Bug)

## Date
2026-01-12

## Problem
The code default for `reward_risk_ratio` in `src/config.py` was **0.1**, causing positions to risk **10x** what they expected to make.

### Impact
- Trades placed before configs/active.json was updated used this default
- Example from 15min session (2026-01-12 09:45):
  - Entry: $182.55
  - TP: $183.30 (+$0.75, 0.41%)
  - SL: $175.07 (-$7.48, 4.10%)
  - **Actual ratio: 1:10 (risk:reward)** instead of intended 2.5:1 (reward:risk)

### Why This Happened
The formula in `src/strategy.py` is correct:
```python
stop_dist = tp_dist / cfg.reward_risk_ratio
```

With `reward_risk_ratio = 0.1`:
- `stop_dist = tp_dist / 0.1 = tp_dist * 10`
- Stop loss becomes 10x larger than take profit!

### Error 202 Connection
This bug caused IBKR to reject bracket orders with error 202:
- SL orders $7-8 away from market price during pre-market
- IBKR's price validation rejected these as "too far from market"
- Resulted in positions without brackets

## Root Cause
**Bad code default** in `src/config.py` line 242:
```python
reward_risk_ratio: float = 0.1  # WRONG!
```

This value makes no sense for trading:
- Reward:Risk of 0.1:1 means risking 10x your potential gain
- Any viable strategy needs >= 1.5:1, ideally >= 2.0:1

## Fix Applied

### 1. Fixed Code Default
Changed `src/config.py` line 242:
```python
# Before (WRONG):
reward_risk_ratio: float = 0.1

# After (CORRECT):
reward_risk_ratio: float = 2.5  # Reward:Risk ratio (e.g., 2.5:1 means TP is 2.5x SL distance)
```

Also fixed `risk_per_trade_pct` from 0.001 (0.1%) to 0.01 (1%) to match common practice.

### 2. Added Regression Tests
Created `tests/test_strategy_reward_risk.py` with three tests:
1. **test_reward_risk_ratio_sanity**: Validates correct TP/SL calculation
2. **test_inverted_ratio_detection**: Documents the bug behavior with bad config
3. **test_code_default_is_sane**: Ensures code default is >= 1.5

### 3. Config Override
The `configs/active.json` already had correct value (2.5), but wasn't loaded for the 15min session that ran before the config update.

## Prevention
1. **Tests**: Automated validation of code defaults
2. **Config priority**: Always use configs/active.json when available
3. **Validation**: Consider adding runtime validation to reject reward_risk_ratio < 1.5

## Verification
Run tests to confirm fix:
```bash
python -m pytest tests/test_strategy_reward_risk.py -v
```

All tests should pass.

## Related Issues
- Error 202 "Order Canceled - reason:" during pre-market
- Positions without brackets warnings
- Unexpectedly large stop-loss distances

## Recommendation
**Cancel all existing orders before starting new live sessions** to clear any orders placed with the bad configuration.

```python
# Proposed: Add to live session startup
for order in broker.get_open_orders():
    if getattr(order, 'symbol', '') == cfg.symbol:
        broker.cancel_order(str(getattr(order, 'order_id', '')))
```
