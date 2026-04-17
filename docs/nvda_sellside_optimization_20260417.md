# NVDA Sell-Side Optimization Report — 2026-04-17

## TL;DR

Enabling short trades and optimizing both-side parameters on NVDA produced a candidate that **looks better on mean Sharpe but loses 91% of capital under honest walk-forward** due to a catastrophic single-quarter blowup. All 200 scored two-sided candidates had at least one fold with ≥49% drawdown; 75% had folds with ≥87% drawdown. **The NVDA model + 60min features do not support profitable short trading on this symbol.**

**Action:** Rolled NVDA back to the long-only winner (mean Sharpe 0.77, 67% positive folds, compound +38% WF return). Added `min_dd` (worst-fold drawdown) to the filter tiers as a structural guard against this class of failure.

## Experimental Setup

- Same clean pipeline from the 2026-04-17 audit: two-stage walk-forward scoring (Stage A on middle fold, Stage B on all 9 folds), no leakage.
- Same 2000 LHS candidates over the 6D parameter space.
- `--long-only` flag dropped: `allow_shorts=True`, short params (`k_sigma_short`, `k_atr_short`) now active in every backtest.
- Comparison baseline: the long-only winner (candidate #3) from the earlier run.

## Headline Metrics

### Winner per configuration

| Metric | Long-only winner (#3) | Two-sided winner (#76) |
|--------|----------------------:|-----------------------:|
| `k_sigma_long`       | 0.248 | 0.760 |
| `k_sigma_short`      | 0.268 | 0.471 |
| `k_atr_long`         | 0.639 | 0.774 |
| `k_atr_short`        | 0.474 | 0.794 |
| `reward_risk_ratio`  | 1.52  | 2.10  |
| `risk_per_trade_pct` | 1.10% | 1.33% |

### Walk-forward OOS aggregates (9 folds, 2024-01 → 2026-04)

| Metric | Long-only | Two-sided | Winner |
|--------|----------:|----------:|:-------|
| mean Sharpe             | 0.77  | **1.04**  | two-sided |
| std Sharpe              | 3.20  | 3.89      | long-only |
| min Sharpe              | -3.14 | -2.71     | two-sided |
| pct positive folds      | **67%** | 44%     | long-only |
| mean profit factor      | 1.07  | **1.25**  | two-sided |
| mean drawdown           | **-9.3%** | -22.8% | long-only |
| **min (worst) drawdown**| **-22.4%** | **-89.3%** | long-only (by a large margin) |
| mean trades per fold    | 106   | 161       | — |
| **compound WF return**  | **+38.2%** | **-91.0%** | **long-only (decisive)** |

### Full-horizon smoke backtest (2024-01 → 2026-04, production model)

| Metric | Long-only | Two-sided |
|--------|----------:|----------:|
| total return       | +85.6%  | **+342.7%** |
| annualized Sharpe  | 1.96    | **2.70**    |
| max drawdown       | -9.6%   | -12.1%     |
| win rate           | 57.6%   | 54.6%      |
| n_trades           | 132     | 174 (80 long, 94 short) |

⚠️ The smoke-backtest numbers are **optimistic** because they use the production NVDA model, which was trained on a 70/30 split of 2023-2026 data and has therefore seen the smoke-backtest window during training. The honest walk-forward numbers above are the defensible OOS performance.

## The Fold-6 Blowup

Two-sided winner, fold 6 (test window **2025-04-01 → 2025-07-01**):

| Metric | Value |
|--------|------:|
| Sharpe | -1.65 |
| max drawdown | **-89.3%** |
| total return | **-87.0%** |
| n_trades | **1022** (≈11 trades/day on 60min bars — 1.7 trades per 60-min bar) |

Reading the trade count: 1022 trades in a 3-month fold is ~16 trades per trading day. On 60min bars that's more than one trade per bar on average — the strategy is re-entering repeatedly as stops get hit, getting whipsawed into oblivion. Classic runaway-loop failure mode.

Contrast with long-only worst fold (fold 4, 2024-10-01 → 2025-01-01): -22.4% DD, 82 trades, -17.6% return. Manageable loss, recoverable position.

## Distribution Across All 200 Two-Sided Candidates

| min_dd threshold (worst-fold DD) | Candidates passing |
|---------------------------------:|-------------------:|
| > -20%                           | 0 |
| > -30%                           | 0 |
| > -40%                           | 0 |
| > -50%                           | 1 |
| > -60%                           | 2 |
| > -80%                           | 27 |

- **Minimum min_dd across 200 candidates: -49.3%.** Even the safest two-sided candidate has a >49% drawdown in its worst fold.
- **Median min_dd: -98.2%.** Most two-sided candidates are one bad quarter away from total loss.
- **The single candidate passing min_dd > -50%** has mean Sharpe **0.73** — lower than the long-only winner's 0.77. So even if we picked the "safest" two-sided candidate, it's worse than long-only.

## Why the Short Side Fails on NVDA

Three plausible, non-exclusive reasons:

1. **Upward drift.** NVDA has a strong positive drift over the test window. Shorts fight a headwind; when they're wrong, they're wrong against a rising stock with ATR-scaled stops that get hit frequently.
2. **LSTM signal is directionally biased.** The model was trained on forward log-returns across a regime that was mostly up. It may have learned a "call for up-moves" bias more reliably than "call for down-moves."
3. **The SL/TP structure compounds losses on whipsaws.** A short with a fixed TP and ATR-based SL gets stopped out repeatedly in a rising market, and the re-entry logic (immediate re-entry when position is None) compounds losses rapidly — the fold-6 1022-trade blowup is the evidence.

## Process Fix: `min_dd` Filter

The old filter gated on `mean_dd` but not `min_dd`. Two-sided winner had mean_dd -22.8% (passes a permissive filter) but min_dd -89.3% (should reject). Added `min_dd_min` as a tier dimension:

```python
tiers = [
    # (mean_sh_min, mean_pf_min, mean_dd_min, mean_trades_min,
    #  min_sh_min, pct_pos_min, min_dd_min)
    (0.50, 1.20, -0.30, 40,  0.00, 0.60, -0.25),  # strict
    (0.40, 1.15, -0.32, 35, -0.10, 0.55, -0.30),
    (0.30, 1.10, -0.35, 30, -0.30, 0.50, -0.35),
    (0.20, 1.05, -0.40, 25, -0.50, 0.45, -0.40),
    (0.00, 1.01, -0.50, 20, -1.00, 0.40, -0.45),
    (0.00, 1.01, -0.55, 20, -2.00, 0.33, -0.50),  # high-vol tolerance
    (0.00, 1.00, -0.60, 20, -4.00, 0.25, -0.50),  # ultra-permissive
]
```

Even the most permissive tier requires `min_dd > -0.50`. With this guard, the two-sided NVDA run would have correctly returned **zero viable candidates** instead of auto-promoting a -91% compound-return strategy.

## Final State

- `configs/symbols/NVDA/active.json`: long-only winner (rolled back, `allow_shorts: false`).
- `configs/symbols/NVDA/active.json.longonly_winner_20260417`: persistent backup of the long-only winner.
- `configs/symbols/NVDA/active.json.twosided_winner_20260417`: persistent backup of the rejected two-sided candidate.
- `backtests/robust_selection_nvda/`: two-sided run artifacts (all 200 scored candidates, fold predictions, winner.json with the pre-rollback pick).
- `backtests/robust_selection_nvda_longonly/`: long-only run artifacts.
- `scripts/robust_param_selection.py`: tier filter now includes `min_dd` gate.

## Conclusion

**NVDA should trade long-only.** The short side fails under honest walk-forward evaluation — not marginally, but catastrophically. The mean-Sharpe improvement from enabling shorts (0.77 → 1.04) is an artifact of aggregating over folds; in compound-return terms (what actually matters for capital preservation), two-sided goes from +38% to -91%.

If a short-side signal on NVDA is desired in the future, the path is not tuning these parameters further — it's improving the underlying model's ability to predict down-moves (feature engineering, regime-aware training, or a separate short-only model).

## JPM Validation (Follow-up Check — Deployed Symbol with `allow_shorts=true`)

JPM is currently deployed with shorts enabled, promoted under the **pre-fix** pipeline. Urgent question: does JPM have a hidden NVDA-style tail-risk blowup?

**Verdict: JPM is safe.** Honest per-fold metrics for the deployed winner (cand #14):

| Metric | Value |
|--------|------:|
| Compound WF return (9 folds, 2024-01 → 2026-04) | **+60.2%** |
| pct positive folds | **78%** (7/9) |
| min_dd (worst-fold drawdown) | **-27.9%** |
| mean trades per fold | 25 (vs NVDA two-sided 161) |

The worst-fold drawdown of -27.9% is a manageable loss, not a capital-destruction event. The 78% positive-fold rate and low trade count (25/fold, 10× more selective than NVDA's 200+) are hallmarks of a strategy trading on real signal rather than whipsaw. **No rollback needed.**

## Uncovered Filter Artifact: `mean_pf=0` Sentinel Corruption

While re-filtering JPM against the new `min_dd` tier, found that JPM's winner had `mean_pf = 0.935` — below the tier-7 gate of 1.00 — despite compounding to +60%. Root cause:

Two of JPM winner's folds (folds 1 and 2) had exactly 1 winning trade each. The backtester stored `profit_factor = 0.0` for these folds (gross losses = 0 → PF undefined → sentinel 0). Those zeros dragged the mean down from the real 1.20 to 0.94.

**Fix** (shipped): `rank_candidates` now replaces 0.0 with NaN in `oos_profit_factor` before aggregating, and the groupby mean skips NaN:

```python
oos_long["oos_profit_factor_clean"] = oos_long["oos_profit_factor"].replace(0.0, float("nan"))
# ...then agg uses oos_profit_factor_clean with fillna(1.0) for all-NaN edge case.
```

With this fix, JPM's mean_pf = 1.20 (correct) and cand #14 is correctly identified as the top-ranked survivor of the new filter (oos_score -0.87, best among 15 viable).

## Additional Hardening: Fail Closed on No-Viable Fallback

Initial fallback design returned "top-N by oos_score for inspection" even when all 7 tiers failed. That still flowed into Phase 4-7 and auto-promoted — defeating the min_dd gate. The NVDA two-sided incident was exactly this: min_dd filter correctly flagged all 200 candidates as too risky, but the fallback still returned a winner.

**Fix** (shipped): when all tiers fail, print top-5 by oos_score to logs for diagnostic purposes, then return **empty** so Phase 3's `if viable.empty: sys.exit(2)` bail fires. No silent auto-promotion of structurally-unsafe candidates.

Demonstrated on the NVDA two-sided data:

```
filter tier 7/7: ... → 0 viable
no tier yielded viable candidates. Top 5 by oos_score (for inspection only,
not promoted — fix filters or investigate pipeline before rerunning):
    cand# 76 mean_sh=+1.04 pct_pos=0.44 min_dd=-89.34% mean_pf=1.25 oos_score=-3.81
    cand#  5 mean_sh=+1.36 pct_pos=0.67 min_dd=-63.01% mean_pf=1.36 oos_score=-3.82
    ...
```

## Final Pipeline State

Three hardening changes landed in `scripts/robust_param_selection.py`:
1. `min_dd` as a filter dimension across all 7 tiers (strictest: -25%, most permissive: -50%).
2. `mean_pf` aggregation treats 0.0 as NaN to avoid sentinel-zero corruption.
3. No-viable fallback fails closed — logs diagnostic but returns empty.

All 7 tests in `tests/test_robust_param_selection.py` still green.

## Open Questions / Follow-ups (Not Executed)

1. **MSFT/XOM rerun** with the hardened filter — their clean-pipeline runs returned zero viable candidates; may be worth a fresh look now that `mean_pf` is no longer sentinel-corrupted.
2. **Feature engineering for short-side signal** on NVDA: bear-flag patterns, VIX term structure, put-call ratio, or a short-specialized LSTM. The evidence says the current 7-feature set cannot support short trades on NVDA — improving the signal is the way in, not tuning strategy params.
3. **Capital-protecting short mode on JPM**: even though JPM passes the new filter, its `min_dd -27.9%` is at the boundary. Worth running JPM through the hardened pipeline (with our fixes) to see if a even-better candidate emerges.
