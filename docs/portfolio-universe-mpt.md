# Trading Universe & Portfolio Construction
## Modern Portfolio Theory Framework

**Status:** Approved design — Phase 1 implementation pending  
**Last updated:** 2026-04-16  
**Audience:** Quant/developer — explains *why* these assets were chosen and how sizing works

---

## 1. Context

The LSTM trading system predicts price movements on 60-min bars and executes bracketed orders through Interactive Brokers. Starting from a single-asset NVDA system, the goal is to expand to a multi-asset portfolio that maximises **portfolio-level Sharpe ratio** while reducing maximum drawdown through sector diversification.

The core MPT insight: adding more correlated tech assets (e.g. AMD, INTC) provides almost no diversification benefit. The biggest vol reduction comes from **sector independence**, not more names.

---

## 2. Investment Style — Long-Term Portfolio Analysis

The long-term equity portfolio reveals the investment style that informed universe selection:

| Theme | Examples | ~Weight |
|---|---|---|
| AI / Mega-cap tech | CRWV, GOOGL, SNOW, MSFT, BBAI, SOUN | 36% |
| Quantum / Deep-tech | IONQ, RGTI | 7% |
| New energy | OKLO, SMR, PLUG | 6% |
| Future mobility | SERV, JOBY, AEVA, TSLA | 6% |
| Large-cap anchor | AAPL, AMZN, INTC | 9% |
| Commodity / Value | CF, MGY | 10% |
| SaaS / Cloud | DOCN, OKTA, LMND | 8% |
| International | JD.com, Meituan, Baidu (SEHK) | 8% |

**Style characteristics:**
- High-conviction, thematic, concentrated top positions (top 3 = ~30%)
- Comfortable holding illiquid and speculative names long-term (IONQ, LUNR, QS)
- Includes commodity tail-hedges (CF fertilizers, MGY oil) — macro awareness
- Not purely momentum: holds turnaround thesis (INTC)

**Key implication for the trading system:** the LSTM portfolio must be *more conservative* than the long-term book. LSTM edge is technical signal quality on 60-min bars, not thematic conviction. Speculative micro-caps produce noisy, unlearnable signals and suffer wide bid-ask spreads that destroy edge.

---

## 3. Universe Construction Filters

### Hard Filters (all must pass)

| Filter | Threshold | Rationale |
|---|---|---|
| Exchange | NYSE / NASDAQ only | IBKR access, US market hours |
| Average daily volume | ≥ $300M/day | Tight spreads on 60-min bars, no slippage |
| Market cap | ≥ $20B | Institutional participation, smooth price action |
| Price history | ≥ 3 years | Sufficient data to train and walk-forward validate LSTM |

### Soft Ranking Filters

- **Pairwise correlation** with existing assets < 0.50 (target ≈ 0.30)
- **Standalone Sharpe** in LSTM walk-forward backtest ≥ 1.5 before portfolio inclusion
- **Volatility regime** compatible with 60-min strategy — not too slow (wide bars, few signals) nor too chaotic (gap risk)

---

## 4. Approved Universe — 6 Symbols

| # | Symbol | Sector | Role in Portfolio |
|---|---|---|---|
| 1 | **NVDA** | Semiconductors | Core alpha source; highest standalone Sharpe (2.88) |
| 2 | **MSFT** | Mega-cap Tech | Second tech leg; strong LSTM signal |
| 3 | **JPM** | Financials | Rate-sensitive; low tech correlation |
| 4 | **XOM** | Energy | Real-asset; inflation hedge; near-zero tech correlation |
| 5 | **UNH** | Healthcare | Defensive; GDP-uncorrelated; near-zero tech correlation |
| 6 | **AMZN** | Consumer / Tech | Held in long-term book; distinct vol regime from NVDA |

### Pairwise Correlation Matrix (approximate, 1-year rolling)

```
         NVDA   MSFT   JPM    UNH    XOM    AMZN
NVDA     1.00   0.70   0.35   0.20   0.15   0.55
MSFT     0.70   1.00   0.40   0.25   0.18   0.60
JPM      0.35   0.40   1.00   0.30   0.42   0.38
UNH      0.20   0.25   0.30   1.00   0.10   0.22
XOM      0.15   0.18   0.42   0.10   1.00   0.20
AMZN     0.55   0.60   0.38   0.22   0.20   1.00
```

**Average pairwise ρ ≈ 0.32** — vs ρ = 0.70 for the original NVDA+MSFT-only system.  
This represents a dramatic reduction in portfolio variance for the same expected return.

> ⚠️ Verify correlations with actual daily return data before going live. These are indicative estimates based on historical regimes.

---

## 5. Assets Excluded from Trading System

The following names are held in the long-term equity book but explicitly excluded from the LSTM trading universe:

| Symbol(s) | Reason for Exclusion |
|---|---|
| IONQ, RGTI, QS | ADV < $100M; wide spreads; price driven by news events, not technicals |
| LUNR, JOBY, AEVA, SERV | Micro-cap; event-driven gaps; LSTM signal is noise |
| OKLO, SMR | Thin float; regulatory/sentiment-driven moves |
| CRWV | Insufficient price history (< 3 years) for LSTM training |
| PLUG | ADV borderline; highly sentiment-driven; poor technical Sharpe |
| JD, Meituan, Baidu (SEHK) | Trade on Hong Kong exchange; out of US market hours |

**Rule of thumb:** if the stock regularly moves ±5% on a single news item unrelated to macro, it cannot be reliably predicted by a bar-level LSTM.

---

## 6. Position Sizing — Equal Risk Contribution (ERC)

### Problem with flat percentage caps

`configs/portfolio.json` currently uses `max_per_symbol_pct = 0.30` (flat 30% per asset). With 6 assets of very different volatility — NVDA (σ ≈ 2.5%/day) vs XOM (σ ≈ 1.0%/day) — equal-dollar allocation means NVDA dominates portfolio risk. This defeats the purpose of diversification.

### Solution: Volatility-weighted (ERC) allocation

Each symbol's capital budget is inversely proportional to its 20-day rolling realised volatility:

```
w_i  =  (1 / σ_i)  /  Σ_j (1 / σ_j)
```

Where `σ_i` is the annualised 20-day realised vol of symbol `i`.

**Effect:** A high-vol asset (NVDA) receives less capital; a low-vol asset (XOM, UNH) receives more. Each asset contributes roughly equal risk to the total portfolio.

### Implementation location

`src/portfolio/capital_allocator.py` — replace static `max_per_symbol_pct` with the dynamic ERC weight. A hard ceiling of 25% per symbol is retained to prevent extreme concentration during periods of very low individual volatility.

### Sizing hierarchy (unchanged)

```
1. ERC weight         →  symbol capital budget
2. LSTM signal filter →  k_sigma + k_atr threshold check
3. SL-based sizing    →  risk_per_trade_pct × equity / SL distance
4. Gross cap check    →  max 80% of total equity deployed
```

---

## 7. Phased Rollout Plan

### Phase 1 — JPM + XOM (immediate next sprint)
JPM and XOM have the lowest correlation to existing tech positions and the cleanest intraday price action. Start here.

**Implementation steps:**
- [ ] Add `JPM`, `XOM` to `CONTRACT_REGISTRY` in `src/config.py:679`
- [ ] Add to `configs/portfolio.json` symbols list; set `max_per_symbol_pct = 0.20`
- [ ] Assign client IDs: `JPM=12`, `XOM=13`
- [ ] Run data ingestion: requires live IBKR TWS/Gateway connection
- [ ] Train LSTM per symbol: `python src/train.py --symbol JPM`
- [ ] Walk-forward backtest: `python scripts/run_walkforward_backtest.py --symbol JPM`
- [ ] Tune thresholds; save to `configs/symbols/JPM/active.json`
- [ ] Paper-trade for ≥ 4 weeks before live capital

### Phase 2 — UNH + AMZN (after Phase 1 validated)
- UNH: most defensive name in the universe, near-zero tech correlation
- AMZN: already held long-term; distinct vol regime from NVDA
- Same onboarding steps as Phase 1

### Phase 3 — ERC position sizing
- Implement dynamic volatility-weighted allocation in `src/portfolio/capital_allocator.py`
- Replace static `max_per_symbol_pct` with rolling-vol-based weight per bar

### Phase 4 — Portfolio-level optimisation (optional)
- Black-Litterman blending: use LSTM signal strength as "views", historical covariance as prior
- Run combined multi-symbol backtest to evaluate portfolio-level Sharpe vs Phase 3 baseline

---

## 8. Target Metrics

| Metric | Target |
|---|---|
| Portfolio Sharpe (90-day rolling) | ≥ 2.0 |
| Max portfolio drawdown | < individual symbol worst DD |
| Average pairwise PnL correlation | < 0.50 |
| Per-symbol standalone Sharpe | ≥ 1.5 before inclusion |

**Stress test requirement:** run combined backtest over:
- COVID crash: Feb–Mar 2020 (correlated drawdown test)
- Rate hike cycle: 2022 (sector rotation, tech vs energy/financials)
