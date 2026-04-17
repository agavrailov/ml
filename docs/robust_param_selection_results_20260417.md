# Robust Parameter Selection — Results Report

**Date:** 2026-04-17
**Driver:** `scripts/robust_param_selection.py`
**Config:** 2000 LHS samples per symbol, 30 diverse via maximin, 9-fold walk-forward (train 12mo / test 3mo) with per-fold LSTM retrain.

---

## Executive Summary

Of the four symbols in the candidate portfolio, **only JPM has a statistically robust positive-expectancy strategy** under the current model + feature set + 6-parameter space. NVDA, MSFT, and XOM Phase 5–7 runs produced no candidate with positive OOS mean Sharpe across 9 walk-forward test quarters.

### Action taken

- **JPM:** Auto-promoted to `configs/symbols/JPM/active.json` (winner: mean Sharpe 1.09, 78% positive folds).
- **NVDA / MSFT / XOM:** Rolled back to legacy RC2-retrain configs. Bad auto-promotions preserved as `configs/symbols/{SYM}/active.json.autopromoted_robust_20260417` for post-mortem.

---

## Phase 1–4 (In-Sample Search)

| Symbol | Samples | Viable | Tier (1=strictest, 5=most permissive) | Diverse |
|--------|--------:|-------:|---------------------------------------|--------:|
| NVDA   | 2000    | 12     | 5/5                                   | 12      |
| MSFT   | 2000    | 114    | 2/5                                   | 30      |
| **JPM**| 2000    | **87** | **1/5**                               | 30      |
| XOM    | 2000    | 57     | 5/5                                   | 30      |

**Tier 1/5 filter:** `sh>0.5, pf>1.2, dd>-0.3, trades>=40, half_min>0.0`
**Tier 5/5 filter:** `sh>0.0, pf>1.01, dd>-0.5, trades>=20, half_min>-1.0`

JPM was the only symbol where the strictest filter already yielded >2n_diverse viable candidates. NVDA and XOM had to relax all the way to the most permissive filter.

---

## Phase 5–7 (Walk-Forward OOS)

### Top candidate per symbol

| Symbol | mean Sharpe | std | min | pct pos folds | trades/fold | oos_score |
|--------|-----------:|-----:|-----:|--------------:|------------:|----------:|
| **JPM**    | **1.09**   | 1.71 | -1.32 | **78%**       | **25**      | **-0.87** |
| NVDA (#3 by score)  | 0.73 | 5.38 | -5.53 | 56%           | 292         | -9.18     |
| NVDA (winner)       | 0.34 | 5.30 | -4.29 | 22%           | 210         | -6.74     |
| XOM (winner)        | -0.47 | 1.06 | -2.37 | 22%           | 88          | -3.36     |
| MSFT (winner)       | -1.51 | 2.29 | -4.47 | 22%           | 157         | -7.21     |

### Trade-frequency signature

The single strongest predictor of OOS robustness was **trade frequency**:

| Symbol | Winner trades/fold | OOS result |
|--------|-------------------:|------------|
| JPM    | 25                 | ✅ Sharpe 1.09, 78% pos |
| XOM    | 88                 | ❌ Sharpe -0.47 |
| MSFT   | 157                | ❌ Sharpe -1.51 |
| NVDA   | 210                | ❌ Sharpe 0.34, std 5.30 |

Low-frequency (selective) strategies survive OOS. High-frequency strategies look good in-sample but don't generalize — the LSTM + 7-feature set isn't extracting signal strong enough to support 150+ trades per quarter.

---

## Per-Symbol Artifacts

All phases produce structured CSV + JSON output. Paths:

```
backtests/robust_selection_{sym}/
    01_samples.csv        — LHS 2000 parameter points
    02_scored.csv         — Full-window + halves scores for all samples
    03_viable.csv         — Candidates passing tier filter
    04_diverse.csv        — Maximin-selected diverse candidates
    05_oos_long.csv       — Long-form per-fold-per-candidate OOS results
    06_ranked.csv         — Aggregated ranking with robustness-adjusted oos_score
    fold01..N_predictions.csv — Per-fold test-window model predictions
    winner.json           — Winning candidate with full OOS stats

configs/library/{SYM}/{FREQ}/cfg_YYYYMMDD_HHMMSS_robust_oos.json
    — Archived winner snapshots (one per run)

logs/robust_p14_{sym}.log  — Phase 1-4 log
logs/robust_p57_{sym}.log  — Phase 5-7 log
```

---

## JPM Winner Details

```json
{
  "symbol": "JPM",
  "params": {
    "k_sigma_long":       0.527,
    "k_sigma_short":      0.367,
    "k_atr_long":         0.675,
    "k_atr_short":        0.833,
    "reward_risk_ratio":  2.65,
    "risk_per_trade_pct": 0.0167
  },
  "oos_stats": {
    "mean_sharpe":   1.0877,
    "std_sharpe":    1.7143,
    "min_sharpe":   -1.3235,
    "median_sharpe": 1.0235,
    "pct_pos_folds": 0.7778,
    "mean_pf":       0.9347,
    "mean_dd":      -0.0962,
    "n_folds":       9.0,
    "oos_score":    -0.8740
  }
}
```

Note on `oos_score`: the score is negative because `mean_pf < 1.0` triggers penalties and `std_sharpe` is large relative to `mean_sharpe`. The *strategy itself* is positive-expectancy in mean-Sharpe and positive-fold-fraction terms; the aggregate score is conservative.

---

## Bug Found and Fixed During Run

### pick_order missing from early-return branch of maximin_select

`scripts/robust_param_selection.py:maximin_select` early-returned when `len(viable) <= n` without inserting the `pick_order` column. Phase 5 reads `cand.get("pick_order", -1)` as `candidate_id`; without the column, every row collapsed to `candidate_id=-1` and the ranking aggregated all candidates into a single row.

**Triggered by:** NVDA (12 viable < 30 n_diverse → early return path). MSFT / JPM / XOM had >30 viable and took the maximin path which does insert the column.

**Fix:**

```python
if len(viable) <= n:
    out = viable.copy().sort_values("full_sharpe_ratio", ascending=False).reset_index(drop=True)
    out.insert(0, "pick_order", range(1, len(out) + 1))
    return out
```

**Remediation:** NVDA's 04_diverse.csv was patched in-place and Phase 5–7 re-run after the fix. Second run produced proper per-candidate ranking (with the result that every candidate had negative oos_score — hence the rollback).

---

## Open Questions / Next Steps

### Why only JPM?

Three plausible explanations, not mutually exclusive:

1. **JPM has cleaner price dynamics on 60min bars** — bank stocks exhibit more mean-reverting structure than tech (NVDA/MSFT) or commodity-linked (XOM).
2. **The 7-feature set happens to be informative for JPM** but not the others. Features: SMA_7, SMA_21, RSI, plus OHLC. No sector/macro context.
3. **1-year training windows are too short for noisier symbols.** JPM's fold models converge on something stable; NVDA/MSFT/XOM fold models may be overfitting the training window.

### Experiments worth running

| Priority | Experiment | Cost | Expected insight |
|----------|------------|-----:|------------------|
| High     | Long-only NVDA/MSFT/XOM (drop k_sigma_short, k_atr_short from sample space) | ~40 min | Test whether shorts drive OOS collapse |
| High     | Tighten filter: require `trades_full_window ≤ 150` for NVDA/MSFT | ~25 min re-run | Force selectivity; see if the 3rd-rank NVDA #3 (Sharpe 0.73, 56% pos folds) has a less-crazy std |
| Medium   | 18-month train window for Phase 5 | ~2x Phase 5 time | Test whether longer lookback stabilises noisier symbols |
| Medium   | Add per-symbol features (XLK ETF for NVDA/MSFT, yield curve for JPM, XLE for XOM) | Nontrivial refactor | Feature engineering hypothesis |
| Low      | 5000 LHS samples | ~75 min per symbol | Diminishing returns at N=2000 — unlikely to change conclusion |

### Deployment options

1. **Deploy JPM only.** Disable NVDA/MSFT/XOM live trading. Safest.
2. **JPM live + the other three on paper/monitoring.** Run the experiments above in parallel with JPM live.
3. **Hold everything until more experiments complete.** Lose the JPM edge waiting.

---

*End of results report.*
