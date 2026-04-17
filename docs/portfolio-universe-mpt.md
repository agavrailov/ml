# Trading Universe & Portfolio Construction
## Evidence-Driven Framework (post-2026-04-17 pipeline runs)

**Status:** Revised based on hardened-pipeline backtest results  
**Last updated:** 2026-04-17  
**Audience:** Quant/developer — explains *why* these assets were chosen, grounded in actual OOS performance

---

## 1. Context

The LSTM trading system predicts price movements on 60-min bars and executes bracketed orders through IBKR. The original universe was selected via pure MPT sector-diversification reasoning — that framework turned out to be wrong for this specific combination of model architecture + feature set + bar frequency.

After running four candidate symbols through a hardened walk-forward pipeline (min_dd gate, fail-closed fallback, mean_pf sentinel fix), the evidence completely reshapes what we know about where the LSTM has edge. This document reflects the revised universe.

Source documents:
- `docs/robust_param_selection_results_20260417.md`
- `docs/pipeline_audit_and_nvda_longonly_20260417.md`
- `docs/nvda_sellside_optimization_20260417.md`
- `docs/msft_xom_hardened_pipeline_20260417.md`

---

## 2. The Empirical Findings

### Hardened-pipeline results summary

| Symbol | Mode | Honest WF compound | pct pos folds | Trades/fold | Verdict |
|---|---|---:|---:|---:|---|
| **JPM** | two-sided | **+60.2%** | 78% | **25** | ✅ Strong edge — deployed |
| **NVDA** | long-only | **+38.2%** | 67% | 106 | ✅ Edge — deployed |
| **NVDA** | two-sided | -91.0% | 44% | 161 | ❌ Catastrophic whipsaw |
| **MSFT** | two-sided | +7.6% | 67% | 113 | ⚠️ Thin; -44% min DD |
| **MSFT** | long-only | -17% to -58% | 22–44% | — | ❌ Every candidate loses |
| **XOM** | any | < 0% all candidates | — | — | ❌ No alpha at all |

### The single most important finding: trade frequency predicts OOS success

| Symbol | Trades/fold | OOS mean Sharpe | Compound WF |
|---|---:|---:|---:|
| JPM | **25** | **1.09** | **+60%** |
| NVDA long | 106 | 0.77 | +38% |
| MSFT two-sided | 113 | 0.63 | +7.6% |
| NVDA two-sided | 161 | 1.04 (looks good, but...) | -91% |
| XOM | 88+ | -0.78 | < 0% |

**Low-frequency (selective) strategies survive OOS. High-frequency strategies fit noise.** The LSTM + 7-feature set (OHLC + SMA_7/21 + RSI) cannot extract enough signal to support 100+ trades per quarter. Any symbol that needs that frequency to reach profitability is fitting noise in-sample and blowing up OOS.

### Short-side asymmetry is symbol-specific and unpredictable
- **NVDA:** long-only works (+38%), two-sided fails (-91%) — strong upward drift punishes shorts
- **MSFT:** long-only fails (all losing), two-sided works (+7.6%) — shorts carry the entire edge
- **Implication:** can't pre-commit universe-wide to long-only or two-sided; each symbol needs both modes tested

### Sector pattern
- **Financials (JPM):** real edge — suggests LSTM finds genuine mean-reverting structure in bank stocks
- **Tech (NVDA, MSFT):** marginal at best, catastrophic at worst — strong drift + whipsaw risk
- **Energy (XOM):** no edge — macro/OPEC-driven, 7-feature technical signal is insufficient

---

## 3. Why Pure MPT Sector-Diversification Was Wrong

The original universe (NVDA, MSFT, JPM, XOM, UNH, AMZN) assumed that low inter-asset correlation would translate into portfolio benefits. That reasoning has two flaws when combined with an LSTM signal:

1. **Correlation benefit requires each asset to have individual edge.** Adding XOM to the portfolio dilutes capital into a losing strategy — the correlation benefit is swamped by the alpha deficit.
2. **Low tech-correlation ≠ clean technical microstructure.** XOM has low tech-ρ but its price is macro-driven (crude futures, OPEC, USD), so the 7-feature set captures no signal. Diversification on paper, noise in practice.

The revised criterion is: **include symbols where the LSTM has measurable OOS edge first, optimize portfolio correlation second within that set.**

---

## 4. Revised Universe — Three Tiers by Conviction

### Tier 1 — JPM Analogs (high conviction)
Symbols whose microstructure on 60-min bars resembles JPM: mean-reverting, institutional participation, not news-gap-driven, real bidirectional movement.

| Symbol | Role | Why |
|---|---|---|
| **GS** | Investment bank | Most direct JPM analog; higher vol, same family |
| **BAC** | Retail bank | Lower-vol JPM cousin; different business mix |
| **MS** | Investment bank | Similar to GS; correlated — pick one or both |
| **C** | Universal bank | More news-driven, test cautiously |
| **WFC** | Retail bank | Clean price action |
| **SCHW** | Brokerage | Rate-sensitive; diversifies away from pure credit |

**Phase 1 priority:** GS + BAC (sector-effect test with clear correlation separation).

### Tier 2 — Defensive Large-Caps (medium conviction)
Not yet tested, but match the "mean-reverting, institutional, low news-gap" microstructure profile.

| Symbol | Sector | Why |
|---|---|---|
| **UNH** | Managed care | Extremely low tech correlation; deep liquidity; clean 60-min mean reversion |
| **JNJ** | Pharma | β ≈ 0.6; very defensive, clean intraday |
| **LLY** | Pharma | Higher vol; GLP-1 narrative gives regime variety |
| **PG** | Consumer staples | Extremely low beta, mean-reverting |
| **KO** | Consumer staples | Similar to PG; sleepier but clean |
| **WMT** | Consumer disc. | Large, liquid, earnings-driven but not gappy intraday |

**Phase 1 priority:** UNH + JNJ (maximum sector diversification if the pattern generalizes).

### Tier 3 — Semi-Speculative (lower conviction, cheap to test)
Already in `CONTRACT_REGISTRY` or trivially addable. Worth running but expected to behave like NVDA/MSFT.

| Symbol | Expected outcome |
|---|---|
| **AAPL** | Tech; may whipsaw two-sided like NVDA |
| **V / MA** | Payment networks; potentially too steady drift |
| **HD / LOW** | Home improvement; earnings-gap risk |

### Already-deployed symbols (confirmed edge)
- **JPM** — two-sided, `configs/symbols/JPM/active.json` live
- **NVDA** — long-only, `configs/symbols/NVDA/active.json` live with `allow_shorts: false`

### Paper-traded / on-hold
- **MSFT** — two-sided auto-promoted but thin (+7.6%, -44% min DD). Run paper-only until next data refresh confirms edge isn't regime-specific.

---

## 5. Actively Excluded Symbols

| Excluded | Reason |
|---|---|
| **XOM** | Empirically rejected — every candidate loses money. Oil dynamics are macro, not technical |
| **CVX, COP, other energy** | Same class as XOM — macro-driven, 7-feature set has no hook |
| **Commodity producers** (NEM, FCX, miners) | Same structural problem as energy |
| **NVDA two-sided** | Empirically rejected — -91% compound; LSTM cannot predict NVDA down-moves |
| **TSLA, META, NFLX** | High-beta tech with strong drift; same failure mode expected as NVDA two-sided |
| **Semiconductors beyond NVDA** (AMD, INTC, AVGO) | Same microstructure as NVDA; if NVDA two-sided fails, these will too |
| **Small-caps, biotech** (< $20B market cap) | News-gap-driven; insufficient history |
| **Utilities** (NEE, DUK, SO) | Probably too sleepy — insufficient signal variance on 60-min bars |
| **SEHK/international ADRs** (JD, Meituan, Baidu) | Wrong trading hours relative to US LSTM pipeline |
| **Long-term speculative names** (IONQ, RGTI, OKLO, LUNR, JOBY, CRWV, QS, PLUG) | Thin ADV, news-driven, insufficient price history |

---

## 6. Revised Hard Filters

These supersede the original theoretical filters with evidence-backed constraints.

| Filter | Threshold | Rationale |
|---|---|---|
| Exchange | NYSE / NASDAQ | US market hours, IBKR access |
| ADV | ≥ $500M/day | Raised from $300M — liquidity matters more than the earlier estimate |
| Market cap | ≥ $50B | Raised from $20B — smaller caps are too gappy on 60-min |
| Price history | ≥ 3 years | Per-symbol LSTM training + 9-fold walk-forward |
| Expected trade frequency | < 80 trades/quarter in backtests | **New, evidence-based** — selectivity predicts OOS success |
| Business driver | Not primarily macro/commodity/OPEC | XOM is the cautionary example |

### Soft ranking filters (applied only after standalone OOS Sharpe ≥ 0.7)
- Pairwise PnL correlation < 0.50 against already-deployed symbols
- min_dd (worst-fold drawdown) > -30%
- pct_pos_folds ≥ 60%

---

## 7. Position Sizing — Equal Risk Contribution (ERC)

Unchanged from the original plan. With the revised universe leaning toward financials (JPM, GS, BAC) that have more homogeneous volatility than tech+energy, the ERC adjustment is less dramatic but still necessary.

```
w_i  =  (1 / σ_i)  /  Σ_j (1 / σ_j)
```

Where σ_i is the 20-day rolling realised vol. Ceiling 25% per symbol. Implement in `src/portfolio/capital_allocator.py`.

---

## 8. Revised Rollout Plan

### Phase 1 (next sprint) — Test the sector hypothesis
Run these four symbols through the hardened pipeline (`scripts/robust_param_selection.py` with min_dd gate + fail-closed + both modes):

| Symbol | Hypothesis being tested |
|---|---|
| **GS** | Is JPM's edge a bank-sector effect or JPM-specific? |
| **BAC** | Does lower-vol retail banking show the same pattern? |
| **UNH** | Does the "selective strategy" pattern extend outside financials? |
| **JNJ** | Does the pattern generalize to defensive mean-reverters? |

**Decision tree after Phase 1:**
- GS + BAC both pass with JPM-like numbers → edge is sector-wide; add MS, C, WFC, SCHW next
- Only GS passes → edge is "investment banking dynamics" specifically
- Neither bank passes → JPM is idiosyncratic; need to explore outside financials aggressively
- UNH/JNJ pass → pattern generalizes; opens PG, KO, WMT, LLY

**Implementation steps per new symbol:**
- [ ] Add to `CONTRACT_REGISTRY` in `src/config.py`
- [ ] Add to `configs/portfolio.json`; lower `max_per_symbol_pct` if adding many
- [ ] Ingest historical data (requires IBKR connection)
- [ ] Train per-symbol LSTM
- [ ] Run `scripts/robust_param_selection.py --symbol SYM` (both modes, pick winner by oos_score)
- [ ] Paper trade ≥ 4 weeks before live capital

### Phase 2 (parallel, higher-risk bet) — Feature engineering
The recurring pattern in the failure analyses is that the 7-feature set is the bottleneck:
- **XOM** fails because energy needs macro context (crude term structure, XLE, USD index)
- **NVDA two-sided** fails partly because there's no regime feature distinguishing trending vs. mean-reverting periods
- **MSFT long-only** fails because there's no short-bias signal

Candidate feature additions (priority order):
1. **Per-sector ETF context** — XLF for financials, XLV for healthcare, XLK for tech, XLE for energy
2. **VIX level** — regime detection (trending vs. mean-reverting)
3. **Rolling realized volatility** — distinct from ATR; longer window
4. **Relative strength vs SPY** — normalizes absolute price drift

If feature engineering unlocks XOM, MSFT long-only, or NVDA two-sided, those names re-enter the universe.

### Phase 3 — ERC sizing
Unchanged from original plan. Implement in `src/portfolio/capital_allocator.py`.

### Phase 4 (optional) — Black-Litterman
Only worth exploring once the universe has 6+ confirmed-edge symbols.

---

## 9. Target Metrics (revised)

| Metric | Target | Source |
|---|---|---|
| Per-symbol standalone OOS mean Sharpe | ≥ 0.7 | JPM = 1.09, NVDA-long = 0.77 |
| Per-symbol pct positive folds | ≥ 60% | JPM = 78%, NVDA-long = 67% |
| Per-symbol min_dd (worst fold) | > -30% | JPM = -28%, NVDA-long = -22% |
| Per-symbol trades/fold | < 80 | **Hard limit** — anything higher is whipsaw risk |
| Portfolio Sharpe (90-day rolling) | ≥ 2.0 | Aspirational — JPM+NVDA alone should get there |
| Portfolio max DD | < worst individual symbol DD | Diversification validation |

---

## 10. Open Questions

1. **Is JPM's edge sector-wide or idiosyncratic?** Phase 1 GS + BAC runs will answer this.
2. **Can feature engineering rescue rejected symbols?** Per-sector ETF + VIX + rel-strength features are the highest-leverage experiment.
3. **Should we deploy MSFT?** +7.6% / -44% DD is at the deployment bar boundary. Current stance: paper trade, re-evaluate after next data refresh.
4. **How many symbols is "enough"?** JPM + NVDA alone is probably deployable. Adding symbols past 5–6 has diminishing correlation benefit and increasing operational complexity.

---

*This document replaces the original theoretical MPT universe plan. The empirical pipeline results are the primary evidence for every universe claim above.*
