# Extended Hours Order Handling

## Problem
During pre-market (4:00 AM - 9:30 AM EST) and after-market (4:00 PM - 8:00 PM EST) sessions, IBKR only accepts **limit orders**. Market orders will be rejected with an error.

## Solution
The `IBKRBrokerTws` broker automatically detects trading session and converts market orders to aggressive limit orders during extended hours.

## Implementation

### Detection
- `_is_regular_market_hours()` checks if current time is within regular trading hours (9:30 AM - 4:00 PM EST, Mon-Fri)
- Uses timezone-aware detection via `zoneinfo` (Python 3.9+) or `backports.zoneinfo`
- Falls back to conservative assumption (regular hours) if timezone detection unavailable

### Conversion Logic
When a market order is placed during extended hours:

1. **Use provided price** if available (from bar close or caller)
   - Avoids market data subscription issues during extended hours
   
2. **Fallback: Fetch current price** via IBKR market data API if not provided
   - Tries: last price → close price → bid/ask midpoint
   
3. **Apply slippage buffer** (1%)
   - BUY orders: `limit_price = current_price × 1.01`
   - SELL orders: `limit_price = current_price × 0.99`
   
4. **Convert to limit order** with calculated price

5. **Set `outsideRth=True`** flag on the order
   - Tells IBKR the order is intentionally placed outside regular hours
   - Without this flag, orders are queued until market open (09:30 EST)

### What Gets Converted
- ✅ Single market orders via `place_order()`
- ✅ Bracket market entry orders via `place_bracket_order()`
- ⏭️ Limit orders pass through unchanged
- ⏭️ Orders during regular hours pass through unchanged

## Usage

No code changes required. The broker handles conversions transparently:

```python
# During extended hours, this market order is automatically
# converted to an aggressive limit order
broker.place_order(OrderRequest(
    symbol="NVDA",
    side=Side.BUY,
    quantity=10,
    order_type=OrderType.MARKET,  # Auto-converted to LIMIT
))

# Bracket orders: provide current_price to avoid market data subscription issues
submit_trade_plan_bracket(
    broker, plan, ctx, 
    current_price=150.0  # From bar close or other source
)

# Or use broker directly (current_price calculated from entry_limit_price if provided)
broker.place_bracket_order(BracketOrderRequest(
    symbol="NVDA",
    side=Side.BUY,
    quantity=10,
    entry_order_type=OrderType.MARKET,  # Auto-converted to LIMIT
    entry_limit_price=151.50,  # Pre-calculated aggressive limit (optional)
    tp_price=160.0,
    sl_price=140.0,
))
```

## Error Handling

If current price cannot be fetched during extended hours and no `entry_limit_price` is provided:
```
ValueError: Cannot place market order for NVDA during extended hours: 
current price unavailable. Please use a limit order instead.
```

**Solutions**:
1. **Recommended**: Pass `current_price` to `submit_trade_plan_bracket()` (e.g., from bar close)
2. **Alternative**: Specify an explicit limit price in the bracket order:
```python
broker.place_bracket_order(BracketOrderRequest(
    symbol="NVDA",
    side=Side.BUY,
    quantity=10,
    entry_order_type=OrderType.MARKET,
    entry_limit_price=150.0,  # Manual limit price
    tp_price=160.0,
    sl_price=140.0,
))
```
3. **Alternative**: Subscribe to extended hours market data in IBKR

### Bracket Order Cancellations (Error 202)

You may see:
```
code=202 reqId=XX Order Canceled - reason:
```

**Common cause**: Entry order filled too quickly (esp. at market open/close with high volatility)

Bracket orders use parent-child linking where TP/SL orders only become active after the entry fills. At volatile times (market open/close), the entry may fill before IBKR can properly activate the child orders, causing them to be canceled.

**This is normal at market open** - the entry order (26) succeeded, but child orders (27, 28) were canceled due to fast fill timing.

**Position remains unprotected** - you'll need to:
- Manually add TP/SL orders after entry fills
- Or monitor the position and manage risk manually
- Consider using slightly wider entry prices during volatile periods to slow fills

## Testing
See `tests/test_ibkr_extended_hours.py` for comprehensive test coverage:
- Market hours detection (regular, pre-market, after-market)
- Market-to-limit conversion with correct slippage buffer
- Limit order pass-through
- Bracket order handling
- Error cases (price unavailable)

## Technical Details

### Time Detection
- Uses `America/New_York` timezone for EST/EDT handling
- Regular hours: 9:30 AM - 4:00 PM EST
- Weekends automatically excluded
- Market holidays NOT currently handled (could add `pandas_market_calendars`)

### Slippage Buffer
- 1% chosen as aggressive but reasonable buffer
- Ensures high fill probability during extended hours liquidity
- Can be adjusted in `_adjust_order_for_extended_hours()` if needed

### Price Fetching
- Uses `reqMktData()` with 0.5s wait for data
- Cleans up subscription via `cancelMktData()`
- Best-effort: returns `None` if unavailable

## Future Enhancements
1. Configurable slippage buffer per symbol or account
2. Market holiday awareness via `pandas_market_calendars`
3. Logging/events when auto-conversion occurs
