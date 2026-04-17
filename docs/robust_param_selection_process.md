# Robust Parameter Selection & Multi-Symbol Recovery Process

**Date:** 2026-04-17
**Scope:** NVDA, MSFT, JPM, XOM — 60min LSTM strategy
**Audience:** Future maintainers (including future-me) debugging regressions or re-running this pipeline.

---

## Why This Document Exists

Over a ~24 hour span we (a) recovered a broken NVDA model registry entry, (b) traced a null-hparam poisoning bug affecting three more symbols, (c) back-filled ~400K rows of missing minute data, (d) retrained all four symbols, (e) discovered that in-sample Sharpe ratios (3.9–5.7) were masking catastrophic out-of-sample degradation, and (f) built a robust parameter-selection pipeline to replace the brittle grid-search that produced those overfit strategies.

The process is repeatable but has multiple footguns. This doc is the map.

---

## 1. The Null-Hparam Bug (Root Cause)

### Symptom

NVDA predictions crashed with `ValueError: Unsupported optimizer: None` inside `build_lstm_model`. MSFT/JPM/XOM backtests ran but returned ~2 trades and Sharpe ≈ 0.02 in CSV mode against stale checkpoints.

### Root Cause

`models/best_hyperparameters.json` contained entries like:

```json
"NVDA": { "60min": { "5": {
    "validation_loss": 6.02e-05,
    "model_filename": "...keras",
    "bias_correction_filename": "...json",
    "lstm_units": null,
    "learning_rate": null,
    "optimizer_name": null,
    "loss_function": null,
    ...
}}}
```

`src/predict.py` did `best_hps.get("optimizer_name", "rmsprop")` — `dict.get` with a default **does not** coalesce explicit `None`; it only fills missing keys. So `optimizer_name` became literal `None`, which crashed model construction.

### Entry Point That Wrote The Nulls

`src/ui/page_modules/train_page.py` promote-click handler:

```python
row = {
    "validation_loss": ...,
    "model_filename": ...,
    "bias_correction_filename": ...,
}
promote_training_row(row=row, ...)
```

…and `src/ui/registry.py::promote_training_row` iterated every optional hparam key and wrote `row.get(key)` → explicit `None` for all the missing ones. A single UI click at 00:12 UTC poisoned the file for every symbol that had been promoted through it.

### Two-Layer Fix

**Fix #1 — Defensive read (`src/predict.py`, ~lines 200-320):**

```python
# Fall back through explicit-None values, don't just rely on dict.get defaults.
optimizer_name = best_hps.get("optimizer_name") or optimizer_name_trained or "rmsprop"
loss_function = best_hps.get("loss_function") or loss_function_trained or "mae"
lstm_units    = best_hps.get("lstm_units")     or lstm_units_trained     or LSTM_UNITS
n_lstm_layers = best_hps.get("n_lstm_layers")  or n_lstm_layers_trained  or N_LSTM_LAYERS
```

Also added extraction of `optimizer_name` / `loss_function` from the **registry entry** (previously only read from `best_hps`), so a registry that stores full hparams self-heals.

**Fix #2 — Stop the poisoning at the source (`src/ui/registry.py::promote_training_row`):**

```python
entry: dict = {"validation_loss": float(row.get("validation_loss"))}
optional_keys = ("model_filename", "bias_correction_filename",
                 "lstm_units", "learning_rate", "epochs", "batch_size",
                 "n_lstm_layers", "stateful", "optimizer_name",
                 "loss_function", "features_to_use")
for key in optional_keys:
    value = row.get(key)
    if value is not None:
        entry[key] = value  # drop keys whose source is None — no null-writes
```

Never write explicit `None` into best_hyperparameters — absence is not the same as "null", and downstream `.get(key, default)` depends on that distinction.

---

## 2. NVDA Recovery (Model + Strategy)

### Model

- Found committed good hparams in git `49321f4` (2025-12-08): `lstm_units=64, lr=0.003, rmsprop, mse, 7 features`.
- The matching `.keras` file had been pruned from the registry directory.
- A surviving 2026-04-02 `.keras` file trained on compatible hparams (val_loss 6.02e-05) was promoted as-is into `models/symbol_registry.json`.
- Backup written: `models/symbol_registry.json.bak_before_nvda_promote`.

### Strategy

`configs/symbols/NVDA/active.json` already contained the post-refactor RC2 strategy params:

```json
{
  "k_sigma_long":       0.35,
  "k_sigma_short":      0.5,
  "k_atr_long":         0.45,
  "k_atr_short":        0.5,
  "reward_risk_ratio":  2.5,
  "risk_per_trade_pct": 0.02,
  "enable_longs": true,
  "allow_shorts": false
}
```

### Verification

NVDA backtest reproduced the historical Sharpe within rounding (2.89 vs recorded 2.88) over 2025-01 → 2026-04-02. Total return diverged upward (208% vs 79%) — this is expected: different `risk_per_trade_pct` compounding between the legacy run and current equity-curve code. Sharpe (risk-adjusted, scale-invariant) is the correct comparator.

---

## 3. MSFT / JPM / XOM Data Backfill

### Problem

NVDA's history went back to 2023. The three other symbols only had partial history — backtests over the common window were not apples-to-apples, and their "not great" performance was partly a data-coverage artifact on top of the null-hparam bug.

### Trap: Non-Strict-Range Ignores `--start`

`src/ingestion/tws_historical.py` anchors on `max(timestamp in existing CSV)`. So:

```bash
python -m src.ingestion.tws_historical --symbol MSFT --start 2023-01-01
```

…silently does nothing if `msft_minute.csv` already extends to 2026-04-16 (it only pulls *forward* from the max).

**Fix:** `--strict-range --file-path data/raw/{sym}_minute_2023_gap.csv` into a fresh file, then merge-and-dedupe into the main CSV. This also sidesteps concurrent-write races when multiple symbols pull in parallel.

### Trap: `smart_fill_gaps` Duplicates

`daily_data_agent.py --skip-ingestion` still calls TWS for gap filling and appends the same bars again. Bypassed by calling lower-level functions directly:

```python
clean_raw_minute_data(...)
run_transform_minute_bars(...)
resample_and_add_features(...)
```

### Execution

Three parallel subagents on IBKR client-ids 11 / 12 / 13, each pulling into a distinct `*_gap.csv`:

| Symbol | Gap rows | Final rows | Range                    |
|--------|---------:|-----------:|--------------------------|
| MSFT   | 92,031   | 613,570    | 2022-12-30 → 2026-04-16 |
| JPM    | 102,801  | 389,953    | 2022-12-30 → 2026-04-16 |
| XOM    | 196,086  | 430,783    | 2022-12-30 → 2026-04-16 |

Backups under `data/raw/.backups_20260417/` before dedupe.

---

## 4. Retraining — And the JPM Loss-Function Mislabel

### Helper Script

`scripts/retrain_symbol.py` — reads hparams from `symbol_registry.json`, monkey-patches `src.train.OPTIMIZER_NAME` / `LOSS_FUNCTION` (the training module reads module-level globals, not kwargs), calls `train_model`, force-updates the registry.

### JPM Discovery

JPM's old registry had `validation_loss: 2.10e-05` with `loss_function: "mae"`. That's implausibly low for MAE on normalized returns (typical MAE is ~1e-2, σ ~1e-3).

Diagnostic retrain with MSE produced `val_loss = 5.66e-05` with σ ~7.5e-3 — consistent with NVDA/XOM. Conclusion: the JPM entry had `loss_function: "mae"` mislabeled; it had actually been trained with MSE. Corrected to `"mse"` in registry; new model saved as `my_lstm_model_60min_tsteps5_20260417_075513.keras`.

**Takeaway:** MSE and MAE validation losses are *not* comparable. Any cross-symbol val_loss comparison must first confirm the loss function.

---

## 5. The In-Sample / OOS Revelation

Walk-forward validation (rolling 3-month test windows) on the four retrained models, **strategy-OOS only** (no per-fold model retrain):

| Symbol | In-sample Sharpe | Mean OOS Sharpe |
|--------|-----------------:|----------------:|
| NVDA   | 3.92             | **-0.84**       |
| MSFT   | 5.28             | **-0.21**       |
| JPM    | 4.71             | **+0.33**       |
| XOM    | 5.67             | **+0.01**       |

Classic overfit pattern. The legacy grid search had picked the top-20 by in-sample Sharpe and those top-20 were all near-duplicates clustered at one corner of the parameter space. No robustness test, no diversity constraint, no OOS validation.

This is the motivation for the robust-selection pipeline below.

---

## 6. Robust Parameter Selection Pipeline

### Driver

`scripts/robust_param_selection.py` — ~600 lines, 7 phases, CLI:

```bash
python scripts/robust_param_selection.py --symbol NVDA \
    --n-samples 2000 --n-diverse 30 \
    [--skip-phase5] [--phase5-only] [--auto-promote]
```

### Parameter Space (6D)

```python
PARAM_RANGES = {
    "k_sigma_long":       (0.10, 0.80),
    "k_sigma_short":      (0.10, 0.80),
    "k_atr_long":         (0.15, 0.85),
    "k_atr_short":        (0.15, 0.85),
    "reward_risk_ratio":  (1.5,  3.5),
    "risk_per_trade_pct": (0.005, 0.020),
}
```

### Phases

1. **Sample** — Latin Hypercube Sampling (`scipy.stats.qmc.LatinHypercube`) over the 6D space. Default N=2000. Uniform marginals, near-uniform joint coverage.
2. **Score (two-stage)** — Run full-window backtest → compute in-sample Sharpe + trade count. Top 25% get a second-stage halves-stability check (Sharpe consistency between first and second halves). Compute savings: ~75% of samples skip the expensive half-window runs.
3. **Adaptive tiered filter** — 5 tiers from strict to permissive. Starts at `Sharpe ≥ 2.0, trades ≥ 50, halves-consistency ≥ 0.5`; relaxes until ≥ `2 * n_diverse` viable candidates. Prevents the "zero viable at strict threshold" failure mode seen in early smoke tests.
4. **Diverse select** — Maximin farthest-point sampling in normalized 6D space. Chosen over k-means because:
   - Guaranteed space coverage (no cluster collapse).
   - Deterministic given a seed.
   - Doesn't need to choose `k` for clustering; `n_diverse` is the output size directly.
5. **Walk-forward retrain + backtest** — For each selected candidate, for each fold: retrain the LSTM on the fold's training window, run the candidate's strategy on the fold's test window. Records Sharpe per fold, max DD, trade count. **Backs up `models/symbol_registry.json` and the scaler file** before each run; restores after. This is the bottleneck phase — ~30 candidates × 8 folds × ~90s/fold ≈ 6 hours per symbol.
6. **Rank** — Robustness-adjusted OOS score:

    ```
    oos_score = mean_sharpe
              − 0.5 * std_sharpe
              + 0.3 * pct_pos_folds
              − 1.0 * max(0, -min_sharpe)
              − 0.5 * max(0, -0.25 - min_dd)
    ```

    Rewards consistency (std penalty), positive-fold fraction, and caps downside (min-Sharpe and min-DD penalties only fire when negative).

7. **Promote** — With `--auto-promote`: write winning params to `configs/symbols/{SYM}/active.json` via `save_strategy_defaults`, and archive the full result set to `configs/library/{SYM}/{FREQ}/` for future reference.

### Concurrency Rules (Important)

- Phases 1–4 are **safe to parallelize** across symbols (read-only against registry/scaler).
- Phase 5 **must be serialized** across symbols. It backs up / restores `models/symbol_registry.json` and the global scaler — concurrent writes corrupt them.
- Use `--skip-phase5` to dispatch Phase 1–4 in parallel subagents, then follow up with `--phase5-only --auto-promote` invocations serially per symbol.

### CLI Patterns

```bash
# Parallel Phase 1-4 (safe)
for SYM in NVDA MSFT JPM XOM; do
    dispatch_subagent "python scripts/robust_param_selection.py \
        --symbol $SYM --n-samples 2000 --n-diverse 30 --skip-phase5"
done

# Serial Phase 5-7 (required)
for SYM in NVDA MSFT JPM XOM; do
    python scripts/robust_param_selection.py \
        --symbol $SYM --phase5-only --auto-promote
done
```

---

## 7. Gotchas & Recurring Bugs

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | `ValueError: Unsupported optimizer: None` | `best_hyperparameters.json` contains explicit `null` from UI promote | Two-layer fix in §1 |
| 2 | `--start 2023-01-01` ingests nothing | Non-strict mode anchors on max(CSV ts) | `--strict-range --file-path ..._gap.csv` |
| 3 | Post-backfill rows duplicated | `smart_fill_gaps` re-pulls + appends | Bypass; call lower-level fns directly |
| 4 | JPM val_loss implausibly low | `loss_function: "mae"` mislabeled; actually MSE | Retrain diagnostic, correct label |
| 5 | `UnicodeEncodeError` on Windows | `→` in print on cp1251 console | Use ASCII `to` instead |
| 6 | `rank_candidates` column shape error | `apply(...)` returned 3 cols not 2 | Use `groupby("candidate_id", as_index=False)` |
| 7 | CSV-mode backtest → 2 trades, Sharpe 0.02 | Stale checkpoint from before bug fix | Model-mode + fresh registry |
| 8 | Phase 5 corrupts registry | Concurrent subagents writing same file | Serialize Phase 5 invocations |

---

## 8. Files Created / Modified

### Created

- `scripts/robust_param_selection.py` — the driver.
- `scripts/retrain_symbol.py` — one-off retrain helper that respects registry hparams.
- `data/raw/.backups_20260417/` — pre-dedupe backups.
- `models/symbol_registry.json.bak_before_*` — registry backups.
- `configs/library/{SYM}/{FREQ}/` — archived candidate result sets from Phase 7.

### Modified

- `src/predict.py` — coalesce-None pattern, read hparams from registry entry.
- `src/ui/registry.py::promote_training_row` — skip None values instead of writing `null`.
- `models/symbol_registry.json` — updated entries for all four symbols.
- `data/raw/{msft,jpm,xom}_minute.csv` — backfilled to 2022-12-30.

### To Watch

- Any new writer into `best_hyperparameters.json` / `symbol_registry.json` — must honour the "don't write explicit None" rule. Consider a schema validator.
- Any path that calls `promote_training_row` with partial `row` dicts — the fix is defensive but the UI is still the most likely offender.

---

## 9. Operational Runbook (Re-Running This From Scratch)

**Prerequisites:** IBKR TWS running on 7496, clean working tree, `pytest` green.

1. Verify `best_hyperparameters.json` has no explicit `null` values — `grep ': null' models/best_hyperparameters.json` should return empty.
2. Back up `models/symbol_registry.json` → `.bak_YYYYMMDD`.
3. For each symbol needing fresh data: pull into `*_gap.csv` with `--strict-range`, dedupe-merge into main CSV.
4. Retrain (if needed): `python scripts/retrain_symbol.py --symbol SYM`.
5. Sanity-check: `python src/predict.py --symbol SYM` returns predictions without error; CSV-mode backtest returns a plausible trade count.
6. Dispatch Phase 1–4 in parallel subagents (`--skip-phase5`).
7. When all report back, run Phase 5–7 serially per symbol (`--phase5-only --auto-promote`).
8. Portfolio sanity check: `python src/portfolio/portfolio_backtest.py` with new configs.
9. Commit `configs/symbols/*/active.json` and archive `configs/library/*/` snapshot.

---

## 10. Open Questions / Future Work

- **Schema validator** for `best_hyperparameters.json` / `symbol_registry.json` — reject writes that contain `null` values for required hparam fields. Closes the poisoning vector permanently.
- **Phase 5 parallelism** — the current serialize-for-safety approach leaves CPU/GPU idle. Per-symbol registry files (`symbol_registry.{SYM}.json`) would enable safe parallelism at the cost of a small refactor in `src/core/config_resolver.py`.
- **Robustness score weights** are heuristic — the `-0.5 * std_sharpe` and `+0.3 * pct_pos_folds` coefficients haven't been tuned against a held-out meta-validation window. Candidate for a second-order study once we have enough history of OOS outcomes per strategy.
- **Feature-set sweep** — current pipeline freezes the 7-feature set. Gains may be available from adding per-symbol features (e.g., sector ETF for NVDA, yield curve for JPM, crude futures for XOM).

---

*End of document.*
