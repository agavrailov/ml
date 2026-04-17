# MSFT / XOM Hardened Pipeline Results — 2026-04-17

## TL;DR

Both MSFT and XOM re-run through the hardened pipeline (min_dd filter + mean_pf sentinel fix + fail-closed fallback) with two-sided enabled:

- **MSFT**: auto-promoted to `configs/symbols/MSFT/active.json`. 5 viable at tier 7/7. Winner compound WF return **+7.6%** / 67% positive folds / -44% worst-fold DD. **Passes filter but thin.**
- **XOM**: pipeline fails closed — **zero viable candidates**. Top candidate by oos_score has mean Sharpe -0.78 and compound WF return -35%. Pipeline correctly refuses to promote.

## MSFT — Passes, Thinly

### Winner (cand #10)

```json
{
  "k_sigma_long":       0.293,
  "k_sigma_short":      0.122,
  "k_atr_long":         0.780,
  "k_atr_short":        0.732,
  "reward_risk_ratio":  1.59,
  "risk_per_trade_pct": 0.0168
}
```

### OOS walk-forward (9 folds, 2024-01 → 2026-04)

| Metric | Value |
|--------|------:|
| mean Sharpe | 0.63 |
| std Sharpe  | **1.25** (stable) |
| min Sharpe  | -0.93 |
| pct positive folds | **67%** (6/9) |
| mean PF | 1.06 |
| mean DD | -19.4% |
| **min DD (worst fold)** | -44.0% |
| mean trades/fold | 113 |
| **compound WF return** | **+7.6%** |

### Per-fold detail

| Fold | Window | Sharpe | PF | DD | Return | n_trades |
|-----:|--------|-------:|---:|---:|-------:|---------:|
| 1 | 2024-Q1 | +1.57 | 1.30 | -10.7% | +9.0% | 32 |
| 2 | 2024-Q2 | +0.33 | 1.03 | -16.0% | +1.4% | 53 |
| 3 | 2024-Q3 | +0.73 | 1.04 | -29.1% | +6.4% | 156 |
| 4 | 2024-Q4 | -0.24 | 0.96 | -24.5% | -6.0% | 143 |
| 5 | 2025-Q1 | +0.33 | 1.00 | -15.8% | +0.6% | 181 |
| 6 | 2025-Q2 | **-0.93** | 0.92 | -24.9% | -15.2% | 172 |
| 7 | 2025-Q3 | +2.71 | 1.28 | -9.2% | +31.7% | 103 |
| 8 | 2025-Q4 | +1.99 | — | 0.0% | +1.1% | 1 (sentinel PF=0 handled) |
| 9 | 2026-Q1 | -0.80 | 0.94 | **-44.0%** | -14.3% | 174 |

### Full-horizon smoke (optimistic — production-model leakage)

- total_return: +24,288% ← wildly inflated, do not trust as deployability metric.
- Sharpe: 4.66, DD: -7.3%, 489 trades (200 long, 289 short).

The smoke Sharpe 4.66 being so divorced from honest WF mean 0.63 confirms the production-model is heavily overfit to the training period. The honest WF number (+7.6% compound) is the deployable expectation.

### Comparison to our validated JPM

| Metric | JPM (deployed) | MSFT (new) |
|--------|---------------:|-----------:|
| Compound WF return | **+60.2%** | +7.6% |
| pct pos folds | 78% | 67% |
| min DD | -27.9% | -44.0% |
| mean trades/fold | 25 | 113 |
| std Sharpe | 1.71 | 1.25 |

MSFT is **dramatically weaker than JPM** (+7.6% vs +60.2% compound) and has worse worst-case risk (-44% vs -28% min DD). Risk-adjusted it's on the edge of "just barely profitable" rather than clearly so.

## XOM — Fails, Pipeline Refuses to Promote

No tier yields any candidates. Pipeline log:

```
filter tier 7/7: mean_sh>0.0 mean_pf>1.0 mean_dd>-0.6 mean_trades>=20 min_sh>-4.0 pct_pos>=0.25 min_dd>-0.5 → 0 viable
no tier yielded viable candidates. Top 5 by oos_score (for inspection only,
not promoted — fix filters or investigate pipeline before rerunning):
ERROR: no viable candidates for XOM. Try loosening filters.
```

Exit code 2. `configs/symbols/XOM/active.json` unchanged.

### Top 5 by oos_score (for diagnostics only — NOT promoted)

| Rank | cand | mean Sh | pct pos | mean PF | min DD | trades/fold | Compound WF |
|-----:|-----:|--------:|--------:|--------:|-------:|------------:|------------:|
| 1 | #21 | -0.78 | 22% | 0.92 | -16.2% |  73 | -34.7% |
| 2 | #31 | -0.83 | 22% | 0.88 | -40.2% | 111 | -65.3% |
| 3 | #39 | -0.76 | 33% | 0.91 | -25.3% |  86 | -37.0% |
| 4 | #63 | -0.61 | 33% | 0.90 | -64.4% | 120 | -90.2% |
| 5 | #10 | -0.87 | 22% | 0.88 | -63.5% | 102 | -90.5% |

**Every XOM candidate is losing money.** Top-5 compound returns span -35% to -91%. No positive-expectancy strategy exists in the 2000 LHS sample space.

Unlike NVDA two-sided (which had 49%+ min-DD across ALL candidates), XOM's issue is not tail risk — 77% of XOM candidates have `min_dd > -50%`. XOM just has **no alpha** — the LSTM's predictions on XOM are not strong enough for any parameter combination to beat transaction costs.

## Hardened-Filter Verdict Across All 4 Symbols

| Symbol | Outcome | Honest WF compound | Deployed? |
|--------|---------|-------------------:|:----------|
| **JPM** (prior, grandfathered) | promoted | **+60.2%** | ✅ live candidate |
| **NVDA** (long-only) | promoted | **+38.2%** | ✅ live candidate |
| **NVDA** (two-sided) | rejected | -91.0% | ❌ pipeline correctly killed |
| **MSFT** (two-sided) | promoted | +7.6%  | ⚠️ thin, see below |
| **XOM** (two-sided) | rejected | < 0% | ❌ pipeline correctly killed |

## MSFT Deployment Recommendation

The auto-promotion is technically correct — MSFT passes every gate in tier 7. But several qualitative concerns:

1. **Thin edge.** +7.6% compound over 2.25 years is ~3.3% annualized. After transaction costs and capital-at-risk discount, the real expected return is marginal.
2. **-44% worst-fold DD.** Right at the tier-7 gate boundary. One bad quarter (fold 9, 2026-Q1) wiped out 44% of the mark-to-market, which in live trading would be psychologically and capital-wise severe.
3. **High trade frequency.** 113 trades per 3-month fold = ~1.4 trades per trading day on 60min bars. More exposure to transaction costs and slippage than the smoke backtest models.

### Options

1. **Deploy MSFT alongside JPM and NVDA-long-only.** Pipeline approved it; the honest WF is positive; portfolio diversification benefit. Risk: -44% worst-fold DD hits real capital.

2. **Hold MSFT in paper-trading only** until a second walk-forward pass on a future data refresh confirms the edge isn't regime-specific. Conservative but misses potential upside.

3. **Re-run MSFT long-only.** NVDA's two-sided failed but long-only worked — maybe MSFT's shorts are dragging on the edge. Worth the 15-min to check. (See next section.)

4. **Don't deploy.** +7.6% compound is below the bar of "clearly worth the operational complexity and tail risk." Focus capital on JPM (+60%) and NVDA-long-only (+38%).

## MSFT Long-Only Re-run (Follow-up — Same Day)

Ran MSFT `--long-only` through the hardened pipeline to test whether shorts were dragging down the edge (as was the case on NVDA).

**Result: zero viable candidates.** Top 5 by oos_score, long-only:

| rank | cand | mean Sh | pct pos | compound WF |
|-----:|-----:|--------:|--------:|------------:|
| 1 | #89  | -0.47 | 33% | -43.7% |
| 2 | #169 | -0.85 | 22% | -58.2% |
| 3 | #92  | -0.70 | 33% | -29.9% |
| 4 | #29  | -0.51 | 44% | -17.1% |
| 5 | #172 | -0.97 | 22% | -32.9% |

**MSFT long-only strategy strictly loses money.** Every candidate in the top 5 has negative mean Sharpe and negative compound returns.

### Critical finding: MSFT and NVDA are directional opposites

| Symbol | Two-sided | Long-only | Conclusion |
|--------|----------:|----------:|:-----------|
| **NVDA** | -91.0% compound | **+38.2% compound** | long-only edge; shorts destroy it |
| **MSFT** | **+7.6% compound** | -17% to -58% (all losing) | short-side edge; long-only doesn't work |

The short-side of MSFT is carrying the entire edge of the two-sided strategy. Disabling shorts flips a marginally-profitable strategy into a reliably-losing one.

### Implication for the universe

The directional asymmetry is symbol-specific and sometimes counter-intuitive. Pre-committing a whole universe to "long-only" (as a conservative default) would have:
- Correctly handled NVDA (avoided -91% compound)
- Mis-handled MSFT (turned +7.6% into a guaranteed loss)

The robust selection pipeline (running both `--long-only` and two-sided and picking the better one) is the right approach per symbol. For production this suggests adding a `run_both_modes` convenience flag that does both runs and picks the winner by oos_score.

### MSFT final config: restored two-sided winner

`configs/symbols/MSFT/active.json.twosided_winner_20260417` was restored as the active config. The long-only attempt is archived at `backtests/robust_selection_msft_longonly/`. MSFT's +7.6% compound / -44% worst DD / 67% positive folds is the best available — thin but genuinely positive.

## Open Follow-ups (Not Executed)

1. **Feature engineering for XOM.** Oil/energy symbols have different drivers than tech/financials. Adding macro features (crude futures term structure, XLE ETF, USD index) may give the LSTM signal strong enough to support strategy params. Not a parameter problem.
2. **`run_both_modes` convenience flag in `robust_param_selection.py`.** Each new symbol should be run both long-only and two-sided by default, winner picked by oos_score, to avoid NVDA-vs-MSFT-style surprises.
3. **Decision: deploy MSFT live or keep paper-only?** Requires judgment call on whether +7.6% compound / -44% DD / 113 trades/fold meets the deployment bar. The alternative (reject MSFT and concentrate capital on JPM + NVDA-long-only) is defensible.
