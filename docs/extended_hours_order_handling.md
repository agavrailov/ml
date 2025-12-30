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

1. **Fetch current price** via IBKR market data API
   - Tries: last price → close price → bid/ask midpoint
   
2. **Apply slippage buffer** (1%)
   - BUY orders: `limit_price = current_price × 1.01`
   - SELL orders: `limit_price = current_price × 0.99`
   
3. **Convert to limit order** with calculated price

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

# Bracket orders also handled
broker.place_bracket_order(BracketOrderRequest(
    symbol="NVDA",
    side=Side.BUY,
    quantity=10,
    entry_order_type=OrderType.MARKET,  # Auto-converted to LIMIT
    tp_price=160.0,
    sl_price=140.0,
))
```

## Error Handling

If current price cannot be fetched during extended hours:
```
ValueError: Cannot place market order for NVDA during extended hours: 
current price unavailable. Please use a limit order instead.
```

**Workaround**: Specify an explicit limit price:
```python
broker.place_order(OrderRequest(
    symbol="NVDA",
    side=Side.BUY,
    quantity=10,
    order_type=OrderType.LIMIT,
    limit_price=150.0,  # Manual limit price
))
```

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
2. Support for `outsideRth=True` IBKR flag for explicit extended hours opt-in
3. Market holiday awareness via `pandas_market_calendars`
4. Logging/events when auto-conversion occurs
