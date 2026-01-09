# Fix: Pre-Market Orders Not Executing

## Problem
Orders placed before market open (pre-market, before 09:30 EST) were being queued but not executing properly. Log showed:
```
Warning: Your order will not be placed at the exchange until 2026-01-07 09:30:00 US/Eastern.
```

Orders were converted to LIMIT orders (correct) but remained queued, and bracket orders were canceled hours later.

## Root Cause
The IBKR API requires the `outsideRth=True` flag on orders to explicitly opt-in to extended hours trading. Without this flag:
- Pre-market orders (04:00-09:30 EST) are queued until market open
- After-market orders (16:00-20:00 EST) may be rejected
- Bracket order parent-child timing can break if entry is delayed

Even though we correctly converted MARKET→LIMIT during extended hours, we never set the `outsideRth` flag.

## Solution
Added `outsideRth=True` flag to all orders placed outside regular market hours (09:30-16:00 EST):

### Changes in `src/ibkr_broker.py`:

1. **Single orders** (`_make_ib_order`):
   ```python
   # Enable extended hours trading when outside regular market hours
   if not _is_regular_market_hours():
       setattr(ib_order, "outsideRth", True)
   ```

2. **Bracket orders** (`place_bracket_order`):
   - Entry order: `outsideRth=True` when outside regular hours
   - TP order: `outsideRth=True` when outside regular hours
   - SL order: `outsideRth=True` when outside regular hours

All three child orders need the flag to ensure they execute immediately when conditions are met.

## Testing
Added comprehensive tests in `tests/test_ibkr_extended_hours.py`:
- `test_outside_rth_flag_set_during_extended_hours()` - verifies flag is set
- `test_outside_rth_flag_not_set_during_regular_hours()` - verifies flag not set during regular hours
- `test_bracket_orders_have_outside_rth_during_extended_hours()` - verifies all 3 bracket orders have flag

All 12 extended hours tests pass.

## Expected Behavior After Fix

### Pre-Market (04:00-09:30 EST)
✅ Orders execute immediately at limit price (no queuing)
✅ Bracket orders link correctly with immediate activation
✅ No "will not be placed until 09:30" warnings

### Regular Hours (09:30-16:00 EST)
✅ Orders execute normally
✅ `outsideRth` flag NOT set (not needed)

### After-Market (16:00-20:00 EST)
✅ Orders execute immediately at limit price
✅ Bracket orders work correctly

## Related Documentation
- `docs/extended_hours_order_handling.md` - updated to document `outsideRth` flag
- IBKR API Reference: `Order.outsideRth` attribute

## Date
2026-01-07
