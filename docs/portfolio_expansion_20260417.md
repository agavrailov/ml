# Portfolio Expansion: GS, BAC, UNH, JNJ — 2026-04-17

## TL;DR

Four new symbols — GS, BAC, UNH, JNJ — were run through the hardened robust-parameter-selection pipeline on 2026-04-17. All four passed and were promoted to `configs/symbols/*/active.json`. The portfolio grows from 2 live-candidate symbols (NVDA, JPM) to 6. GS posts the best `oos_score` of the entire portfolio (0.925). UNH posts the highest compound walk-forward return (159.5%, though this should be treated with caution — see below). BAC and JNJ are weaker but both clear the hardened filter.

A parallel-safety bug in the pipeline was also fixed in this session (race condition in fold-model promotion). See section 3.

---

## 1. Data Acquisition

IBKR TWS/Gateway was offline during this session. Data was sourced via **yfinance** instead of the usual `src/daily_data_agent.py` IBKR pipeline.

| Property | Value |
|----------|-------|
| Source | yfinance |
| Frequency | 60-minute bars |
| Symbols | GS, BAC, UNH, JNJ |
| Bar count per symbol | **5,078** |
| Date range | 2023-05-18 → 2026-04-16 |
| Walk-forward folds | **9** (same as NVDA, JPM, MSFT) |

The 9-fold walk-forward structure is identical to the existing symbol runs: folds cover quarterly windows from 2024-Q1 through 2026-Q1, ensuring a consistent OOS evaluation horizon across the full portfolio universe.

---

## 2. Pipeline Race-Condition Fix

Prior to this session the `generate_fold_predictions` function in `scripts/robust_param_selection.py` had a registry race condition. For each fold, it temporarily promoted the fold-specific model to `symbol_registry.json` via `update_best_model()`. When multiple symbols ran in parallel, these writes collided — a fold model for GS could briefly overwrite the BAC registry entry mid-run, producing corrupted fold predictions.

**Root cause:** `build_prediction_context()` and the backtest layer read the active model from the shared `symbol_registry.json` rather than accepting a model path directly.

**Fix — four changes, no behaviour change for single-symbol runs:**

1. **`src/predict.py` — `build_prediction_context()`**: added `model_path_override` and `bias_path_override` keyword parameters. When supplied, the function uses these paths directly, bypassing the registry lookup entirely.

2. **`src/backtest.py` — `_make_model_prediction_provider()` and `run_backtest_for_ui()`**: both accept and forward the same `model_path_override` / `bias_path_override` parameters down to `build_prediction_context()`.

3. **`scripts/robust_param_selection.py` — `_generate_test_predictions_csv()`**: updated to pass the fold model's `.keras` path and its bias file path directly via the new overrides, rather than writing to the shared registry first.

4. **`scripts/robust_param_selection.py` — fold loop**: the `update_best_model()` call that temporarily promoted fold models to the registry was removed. It is no longer needed.

**Net effect:** parallel runs of `robust_param_selection.py` for different symbols are now safe. Each fold's model is injected directly; the shared `symbol_registry.json` is never touched during fold scoring.

---

## 3. Results

### 3.1 Summary table

| Symbol | Mode | Compound WF | Mean Sharpe | Pct Pos Folds | Min DD | OOS Score |
|--------|------|------------:|------------:|--------------:|-------:|----------:|
| GS | long-only | **+47.1%** | 0.928 | **100%** | -15.7% | **+0.925** |
| BAC | long-only | +38.7% | 0.874 | 78% | -15.0% | -0.323 |
| UNH | two-sided | +159.5% | 0.906 | 67% | -17.6% | -0.672 |
| JNJ | two-sided | +29.7% | 0.666 | 67% | -19.7% | -0.963 |

### 3.2 Mode selection rationale

**GS — long-only.**
Long-only wins decisively: 100% positive folds, compound WF +47.1%, and `oos_score` 0.925. The two-sided run returned only 67% positive folds and a noticeably weaker `oos_score` of 0.486. GS long-only has the highest `oos_score` of any symbol in the entire portfolio — including JPM. There is no case for shorts here.

**BAC — long-only.**
The two-sided candidate for BAC had `min_dd = -33%` and showed three consecutive losing folds in the late-period window — a pattern consistent with a strategy that deteriorates as conditions shift. The long-only winner is materially safer: `min_dd -15.0%` (tied with GS for best in the new cohort), consistent fold performance, and 78% positive folds. Short-side edge on BAC is not present or not reliable enough to deploy.

**UNH — two-sided.**
Two-sided is the clear winner: mean Sharpe 0.906 vs 0.572 for long-only; compound WF 159.5% vs 10.8%. The long-only edge is real but thin; the two-sided strategy captures a short-side signal that more than doubles the expectancy. The 159.5% compound return over 27 months (~71% annualized) is unusually high by the standards of this portfolio. It is attributable to UNH's elevated volatility during 2024-2026 (significant fundamental and regulatory turbulence in the health-insurance sector). Forward expected return should be discounted materially from the historical figure — treat this as a higher-variance symbol and apply position-sizing caution accordingly.

**JNJ — two-sided.**
This is the only viable option. The long-only run failed every filter tier: compound WF -18.8%, 56% positive folds, `min_dd -45.1%` — structurally unprofitable. Two-sided passes because **the short side carries the edge**, analogous to MSFT but considerably stronger (+29.7% compound vs MSFT's +7.6%). JNJ is a low-growth defensive with well-known multi-year price pressure from litigation, biosimilar competition, and sector rotation; the LSTM has apparently learned to anticipate the downside moves more reliably than the upside. Deploy two-sided or do not deploy JNJ at all.

---

## 4. Promoted Configs

### GS — long-only (`configs/symbols/GS/active.json`)

```json
{
  "k_sigma_long":       0.168,
  "k_atr_long":         0.567,
  "k_sigma_short":      0.292,
  "k_atr_short":        0.758,
  "reward_risk_ratio":  2.52,
  "risk_per_trade_pct": 0.0131,
  "enable_longs":       true,
  "allow_shorts":       false
}
```

Metrics: mean_sharpe=0.928, std_sharpe=0.606, pct_pos=1.00, mean_pf=1.251, mean_dd=-7.6%, min_dd=-15.7%, compound_wf=+47.1%, oos_score=+0.925.

### BAC — long-only (`configs/symbols/BAC/active.json`)

```json
{
  "k_sigma_long":       0.483,
  "k_atr_long":         0.266,
  "k_sigma_short":      0.796,
  "k_atr_short":        0.620,
  "reward_risk_ratio":  2.61,
  "risk_per_trade_pct": 0.0129,
  "enable_longs":       true,
  "allow_shorts":       false
}
```

Metrics: mean_sharpe=0.874, std_sharpe=1.371, pct_pos=0.778, mean_pf=1.348, mean_dd=-7.6%, min_dd=-15.0%, compound_wf=+38.7%, oos_score=-0.323.

### UNH — two-sided (`configs/symbols/UNH/active.json`)

```json
{
  "k_sigma_long":       0.222,
  "k_atr_long":         0.398,
  "k_sigma_short":      0.171,
  "k_atr_short":        0.210,
  "reward_risk_ratio":  3.44,
  "risk_per_trade_pct": 0.0093,
  "enable_longs":       true,
  "allow_shorts":       true
}
```

Metrics: mean_sharpe=0.906, std_sharpe=1.312, pct_pos=0.667, mean_pf=1.235, mean_dd=-12.0%, min_dd=-17.6%, compound_wf=+159.5%, oos_score=-0.672.

Note: `risk_per_trade_pct=0.93%` is the lowest of any symbol in the portfolio. This is a pipeline-selected conservative sizing in response to UNH's higher intra-fold volatility.

### JNJ — two-sided (`configs/symbols/JNJ/active.json`)

```json
{
  "k_sigma_long":       0.607,
  "k_atr_long":         0.337,
  "k_sigma_short":      0.754,
  "k_atr_short":        0.640,
  "reward_risk_ratio":  1.85,
  "risk_per_trade_pct": 0.0154,
  "enable_longs":       true,
  "allow_shorts":       true
}
```

Metrics: mean_sharpe=0.666, std_sharpe=1.328, pct_pos=0.667, mean_pf=1.193, mean_dd=-11.4%, min_dd=-19.7%, compound_wf=+29.7%, oos_score=-0.963.

---

## 5. Comparison to Portfolio Baseline

For reference, the two previously-validated symbols:

| Symbol | Mode | Compound WF | Pct Pos Folds | Min DD | OOS Score | Notes |
|--------|------|------------:|--------------:|-------:|----------:|-------|
| JPM | two-sided | +61.8% | 56% | -15.0% | -0.073 | Gold standard; 25 trades/fold |
| NVDA | long-only | +38.2% | 67% | -22.4% | -3.78 | High std_sharpe 3.20; volatile signal |

Full portfolio ordered by compound WF return:

| Rank | Symbol | Mode | Compound WF | Mean Sharpe | Pct Pos | Min DD | OOS Score |
|-----:|--------|------|------------:|------------:|--------:|-------:|----------:|
| 1 | UNH | two-sided | +159.5% | 0.906 | 67% | -17.6% | -0.672 |
| 2 | JPM | two-sided | +61.8% | — | 56% | -15.0% | -0.073 |
| 3 | GS | long-only | +47.1% | 0.928 | 100% | -15.7% | +0.925 |
| 4 | NVDA | long-only | +38.2% | 0.77 | 67% | -22.4% | -3.78 |
| 5 | BAC | long-only | +38.7% | 0.874 | 78% | -15.0% | -0.323 |
| 6 | JNJ | two-sided | +29.7% | 0.666 | 67% | -19.7% | -0.963 |

Key observations:

- **Risk profile is healthy.** Five of six symbols have `min_dd` between -15% and -22%; UNH is the only outlier at -17.6% (still well inside the -50% hard floor). No symbol in the portfolio has a catastrophic tail-risk fold analogous to NVDA two-sided (-89%) or the rejected XOM candidates.
- **GS is the most reliable signal.** 100% positive folds and the best `oos_score` (+0.925 vs the next-best JPM at -0.073) means GS has the most consistent and generalising signal across the test window.
- **NVDA's negative oos_score (-3.78) warrants monitoring.** The oos_score reflects high Sharpe variance (std 3.20); the signal is real but lumpy. It is grandfathered from prior sessions and is not a concern for this expansion.
- **JNJ is the weakest addition.** Mean Sharpe 0.666 and oos_score -0.963 put it at the bottom of the portfolio by both metrics. It passes the filter but only because the short-side edge is present; any degradation in the short signal (e.g., a sustained JNJ recovery rally) would likely flip JNJ to unprofitable quickly. Monitor closely after live deployment.

---

## 6. Updated Portfolio Composition

`configs/portfolio.json` (or equivalent) now reflects the 6-symbol universe:

```json
{
  "symbols": ["NVDA", "JPM", "GS", "BAC", "UNH", "JNJ"],
  "max_per_symbol_pct": 0.20
}
```

`max_per_symbol_pct` was reduced from 0.25 (appropriate for 4 symbols) to **0.20** for 6 symbols. This keeps individual symbol exposure capped at 1/5 of portfolio capital, maintaining adequate diversification and preventing any single position from dominating the equity curve. The ERC (Equal Risk Contribution) sizing layer handles intra-day position sizing; `max_per_symbol_pct` is the portfolio-level cap.

---

## 7. Next Steps for Live Trading

The four new symbols have validated strategy parameters but **no production LSTM model**. The configs in `configs/symbols/*/active.json` specify strategy thresholds only — there is no trained `.keras` model in the registry for GS, BAC, UNH, or JNJ yet. Before any live trading:

### Step 1 — Download data (requires IBKR connection)

```bash
python src/daily_data_agent.py --symbol GS
python src/daily_data_agent.py --symbol BAC
python src/daily_data_agent.py --symbol UNH
python src/daily_data_agent.py --symbol JNJ
```

This writes raw 1-minute OHLCV to `data/raw/` and processes it to `data/processed/{symbol}_60min.csv`.

### Step 2 — Train production models

```bash
python scripts/retrain_symbol.py --symbol GS
python scripts/retrain_symbol.py --symbol BAC
python scripts/retrain_symbol.py --symbol UNH
python scripts/retrain_symbol.py --symbol JNJ
```

Each run trains a full LSTM on the symbol's processed data using the hyperparameters in `best_hyperparameters.json` and registers the resulting `.keras` file in `models/registry/`.

### Step 3 — Smoke backtest

Run a full-history smoke backtest on each new model to verify predictions are sensible before enabling live orders. The smoke numbers will be optimistic (training-data leakage) but should show positive Sharpe and reasonable trade counts.

```bash
python src/backtest.py --symbol GS --mode smoke
# repeat for BAC, UNH, JNJ
```

### Step 4 — Assign IBKR client IDs

Each live daemon needs a unique client ID to connect to TWS/Gateway without collision:

| Symbol | Client ID |
|--------|----------:|
| NVDA | 10 |
| JPM | 11 |
| GS | 12 |
| BAC | 13 |
| UNH | 14 |
| JNJ | 15 |

Update `configs/symbols/{symbol}/active.json` or the relevant daemon launch config with these IDs before starting the live daemons.

### Step 5 — Phased rollout (recommended)

Start GS and BAC first (long-only, simpler operational footprint). Validate P&L and order management for one to two weeks before enabling UNH (two-sided, high volatility) and JNJ (two-sided, short-side dependent). This mirrors the JPM → NVDA phased approach used in the initial portfolio build.
