# Trading Platform Roadmap

## Current Platform
- Single-ticker (NVDA) LSTM prediction platform
- Long-only strategy with TP/SL-based risk management
- Structured modules: config, data pipeline, model, strategy, backtest engine
- Rolling bias correction and model registry in place

## Option A: Multiple Tickers
**Goal:** Add support for multiple, preferably low-correlated tickers to increase robustness and Sharpe ratio.

**Pros**
- Portfolio diversification across uncorrelated assets
- Robustness to regime changes in a single stock
- Potential to leverage transfer learning or shared feature engineering

**Challenges / Work Items**
- Parameterize symbol throughout config and paths (partially done via `PathsConfig` but currently NVDA-specific in some places)
- Decide on per-ticker vs multi-ticker model architecture
- Implement portfolio-level capital allocation and risk limits (max exposure per ticker, sector, correlation)
- Extend backtest engine and reporting to portfolio metrics (per-ticker PnL, correlation, portfolio Sharpe)

## Option B: Long + Short (Bidirectional Trading)
**Goal:** Allow the strategy to take both BUY (long) and SELL (short) positions.

**Pros**
- Doubles the opportunity set on the same data
- Can profit in downtrends and volatility spikes
- Better risk-adjusted returns if the model predicts downside as well as upside

**Challenges / Work Items**
- Extend `TradePlan` and backtest engine to include trade direction (LONG/SHORT)
- Generalize `compute_tp_sl_and_size` to symmetric long/short logic
- Update PnL, commission, and risk calculations for short positions
- Optionally include borrow fees, margin constraints, and shorting restrictions

**Why this first?**
- Lower conceptual and implementation complexity than full multi-ticker support
- Uses existing NVDA data pipeline and model as-is
- Provides immediate feedback on whether the model has useful information about downside moves
- Builds a more general strategy engine that multi-ticker support can reuse

## Other Future Enhancements

### 1. Dynamic Position Sizing
- Volatility-scaling or Kelly-style sizing based on recent ATR or residual sigma
- Aim: smoother equity curve and improved Sharpe

### 2. Multi-Timeframe Signals / Ensembles
- Combine models or features from 15min and 60min (and possibly 240min)
- Use ensemble logic (vote, weighted average, or regime filters)

### 3. Walk-Forward Optimization
- Optimize strategy and model hyperparameters on rolling windows
- Evaluate on out-of-sample slices to reduce overfitting

### 4. Exit Logic Enhancements
- Trailing stops instead of fixed TP
- Time-based exits (max holding period)

### 5. Options Overlay (Long-Term)
- Use options (covered calls/puts, spreads) alongside or instead of stock
- More complex but can improve risk/reward profile
