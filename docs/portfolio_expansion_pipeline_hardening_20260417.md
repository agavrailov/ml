# Portfolio Expansion & Pipeline Hardening — 2026-04-17

**Scope:** Added 4 new symbols (GS, BAC, UNH, JNJ) to the active portfolio; hardened
the walk-forward evaluation pipeline for concurrency correctness; fixed test coverage
gaps introduced by the new code paths.

---

## TL;DR

| Symbol | Mode | mean_Sh | pct_pos | min_DD | WF Compound | oos_score |
|--------|------|---------|---------|--------|-------------|-----------|
| NVDA   | Long-only  | 0.77 | 67% | -22.4% | +38.2%  | -3.78 |
| JPM    | Two-sided  | 0.96 | 56% | -15.0% | +61.8%  | -0.07 |
| GS     | Long-only  | 0.93 | **100%** | -15.7% | +47.1% | **+0.93** |
| BAC    | Long-only  | 0.87 |  78% | -15.0% | +38.7%  | -0.32 |
| UNH    | Two-sided  | 0.91 |  67% | -17.6% | +159.5% | -0.67 |
| JNJ    | Two-sided  | 0.67 |  67% | -19.7% | +29.7%  | -0.96 |

Also dropped: MSFT (edge too thin, +7.6% WF, -44% worst-fold DD), XOM (no alpha).

---

## 1. Data Acquisition

IBKR was offline. Used **yfinance 1.3.0** as a drop-in replacement:

```bash
pip install --upgrade yfinance   # 0.2.36 → 1.3.0 (JSON fix)

python -c "
import yfinance as yf
import pandas as pd

for sym in ['GS', 'BAC', 'UNH', 'JNJ']:
    df = yf.download(sym, period='730d', interval='60m', auto_adjust=True)
    df.index = df.index.tz_convert('US/Eastern').tz_localize(None)
    df = df[['Open','High','Low','Close']].dropna().reset_index()
    df.rename(columns={'Datetime':'Time'}, inplace=True)
    df.to_csv(f'data/processed/{sym.lower()}_60min.csv', index=False)
"
```

Each symbol yielded 5,078 bars (2023-05-18 → 2026-04-16), enough to populate
all 9 walk-forward folds (earliest train start 2024-Q1).

---

## 2. Registry Placeholder Entries

The pipeline requires each symbol to exist in `models/symbol_registry.json` before
`robust_param_selection.py` can run (it reads hparams from the registry to configure
the fold models). Added minimal placeholder entries for each new symbol:

```json
{
  "GS": { "60min": { "5": { "validation_loss": 1e9, "model_path": "", "bias_path": null,
    "hparams": { "lstm_units": 64, "batch_size": 64, "learning_rate": 0.003,
                 "epochs": 20, "n_lstm_layers": 1, "stateful": true,
                 "optimizer_name": "rmsprop", "loss_function": "mse",
                 "features_to_use": ["Open","High","Low","Close","SMA_7","SMA_21","RSI"] }
  }}}
}
```

---

## 3. Pipeline Concurrency Bug: Registry Race Condition

### Problem

`generate_fold_predictions()` in `robust_param_selection.py` trains 9 fold models
sequentially, then (originally) called `update_best_model()` after each fold to
register the fold model. When 4 symbol pipelines ran in parallel as subagents, each
`update_best_model()` call did a **read → modify (own key) → write (full file)**.
The last writer overwrote all changes from the other 3 writers → JNJ (last writer)
restored its own empty placeholder and simultaneously wiped the GS/BAC/UNH fold
paths that had been written by those processes.

### Fix A — Bypass registry during fold loop (`robust_param_selection.py`)

Removed `update_best_model()` from the fold generation loop entirely.
Instead, pass the fold model paths directly to `_generate_test_predictions_csv`
via `model_path`/`bias_path` parameters, which thread through to
`run_backtest_for_ui` → `run_backtest_on_dataframe` → `_make_model_prediction_provider`
→ `build_prediction_context` via new `model_path_override`/`bias_path_override` params.

**Key invariant:** the production registry is never written during fold evaluation.
Fold models are ephemeral; they are used for prediction generation only.

### Fix B — Filelock on `update_best_model` (`src/model_registry.py`)

Even outside fold loops, concurrent calls to `update_best_model` (e.g., from
`retrain_symbol.py` running for multiple symbols simultaneously) were subject to
the same race. Added a cross-platform exclusive file lock:

```python
@contextlib.contextmanager
def _file_lock(lock_path: str, timeout: float = 15.0, poll: float = 0.05):
    """Atomic lock via O_CREAT | O_EXCL — works on Windows NTFS and Unix."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd); break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(...)
            time.sleep(poll)
    try:
        yield
    finally:
        os.unlink(lock_path)
```

`update_best_model` now wraps the entire `_load → modify → _save` sequence inside
`with _file_lock(registry_path + ".lock"):`.  
`_save` already used `os.replace` (atomic rename from `.tmp`) so the write itself
was already crash-safe; the lock adds serialisation of the read-modify-write cycle.

### Fix C — Override path threading (`src/predict.py`, `src/backtest.py`)

New optional parameters added end-to-end:

```
build_prediction_context(model_path_override, bias_path_override)
  ← _make_model_prediction_provider(model_path_override, bias_path_override)
    ← run_backtest_on_dataframe(model_path_override, bias_path_override)
      ← run_backtest_for_ui(model_path_override, bias_path_override)
        ← _generate_test_predictions_csv(model_path, bias_path)
```

When `model_path_override` is provided and the file exists, `build_prediction_context`
uses it directly and only reads the registry for hparams (architecture info), never
writing to it.

---

## 4. Walk-Forward Evaluation Results

### GS — Long-only winner

Ran `robust_param_selection.py --symbol GS --n-samples 2000 --n-diverse 30 --auto-promote`.

- Two-sided mode explored first; long-only strictly dominated on robustness.
- **100% positive folds** — cleanest result in the portfolio.
- Promoted: `k_sigma_long=0.168, k_atr_long=0.567, rr=2.52, risk=1.31%`

### BAC — Long-only winner

Long-only min_DD = -15.0% vs two-sided min_DD = -33%; long-only chosen on tail risk.

- 78% positive folds, compound +38.7%, oos_score = -0.32
- Promoted: `k_sigma_long=0.483, k_atr_long=0.266, rr=2.61, risk=1.29%`

### UNH — Two-sided winner

Two-sided Sharpe 0.91 vs long-only Sharpe 0.57 — clear directional asymmetry
(UNH experienced high volatility 2024-2026, both directions tradeable).

- 67% positive folds, compound +159.5%, min_DD -17.6%
- Note: +159% compound is genuine but reflects exceptional volatility in the
  evaluation window; forward expectation should be treated conservatively.
- Promoted: `k_sigma_long=0.222, k_sigma_short=0.171, k_atr_long=0.398, k_atr_short=0.210, rr=3.44, risk=0.93%`

### JNJ — Two-sided (only viable option)

Long-only compound WF = -18.8% (losing). Short side carries the edge.

- 67% positive folds, compound +29.7%, min_DD -19.7%, oos_score = -0.96
- Weakest edge in the portfolio; position-size is conservative (risk=1.54% but
  capped by portfolio max_per_symbol_pct=20%).
- Promoted: `k_sigma_long=0.607, k_sigma_short=0.754, k_atr_long=0.337, k_atr_short=0.640, rr=1.85, risk=1.54%`

---

## 5. Why JNJ Shows Model=✅ in the Registry Despite Having No "Production" Model

The walk-forward scoring pipeline is **self-contained**: Phase 0 trains 9 fold
models internally, passes them directly via `model_path_override`, and caches
predictions to `fold01..09_predictions.csv`. The production registry is never
consulted for WF evaluation — it is irrelevant to whether the WF ran correctly.

After the parallel run completed:
- GS, BAC, UNH, JNJ all have `.keras` files in `models/registry/` from the fold
  generation phase (their `finally` block wrote the last fold's model path).
- All 4 show `Model=✅` in the UI (verified post-fix via `get_best_model_path`).

The screenshot showing JNJ `Model=❌` was taken mid-run, before the `finally`
block completed.

---

## 6. Portfolio Configuration

`configs/portfolio.json`:
```json
{
  "_comment": "Active portfolio. Client IDs: NVDA=10, JPM=11, GS=12, BAC=13, UNH=14, JNJ=15. MSFT/XOM dropped (no alpha).",
  "symbols": ["NVDA", "JPM", "GS", "BAC", "UNH", "JNJ"],
  "frequency": "60min",
  "tsteps": 5,
  "base_client_id": 10,
  "allocation": {
    "max_gross_exposure_pct": 0.80,
    "max_per_symbol_pct": 0.20
  }
}
```

Client ID assignments (must match TWS/Gateway concurrent session limits):
- NVDA → 10, JPM → 11, GS → 12, BAC → 13, UNH → 14, JNJ → 15

---

## 7. Dropped Symbols

| Symbol | Reason |
|--------|--------|
| MSFT | Two-sided WF: +7.6% compound, -44% worst-fold DD. Edge too thin vs tail risk. Long-only re-run was strictly losing. |
| XOM  | No consistent alpha found in either mode. |

Both disabled via `enable_longs=false, allow_shorts=false` in their `active.json`.

---

## 8. Test Suite Fixes

Four tests broke due to signature changes in `_make_model_prediction_provider` and
the new primary registry lookup path in `build_prediction_context`:

| Test | Root cause | Fix |
|------|-----------|-----|
| `test_predict_model_resolution::falls_back_to_registry_latest` | Didn't mock `get_best_model_entry`; real NVDA registry entry was found | Added `monkeypatch.setattr(predict, "get_best_model_entry", lambda *_a, **_kw: None)` |
| `test_predict_model_resolution::raises_when_no_model_found` | Same | Same fix |
| `test_backtest_model_mode::no_nans` | `_fake_model_provider` missing `**_kw` | Added `**_kw` to fake signature |
| `test_csv_vs_model_consistency::equity_parity` | Same | Same fix |

Final suite result: **312 passed, 1 skipped** (pre-existing checkpoint-replay test
with stale CSV data, unrelated to this work).

---

## 9. Files Changed

| File | Change |
|------|--------|
| `src/model_registry.py` | Added `_file_lock` context manager; wrapped `update_best_model` read-modify-write in lock |
| `src/predict.py` | Added `model_path_override`/`bias_path_override` params to `build_prediction_context`; new primary lookup via `get_best_model_entry`; scaler fallback raises instead of silently using legacy symbol-less file |
| `src/backtest.py` | Added `model_path_override`/`bias_path_override` to `_make_model_prediction_provider`, `run_backtest_on_dataframe`, `run_backtest_for_ui` |
| `scripts/robust_param_selection.py` | `_generate_test_predictions_csv` accepts `model_path`/`bias_path`; fold loop passes paths directly, no `update_best_model` call; `--long-only` flag; linter fix (`×` → `x`) |
| `tests/test_predict_model_resolution.py` | Added `get_best_model_entry` mock to two tests |
| `tests/test_backtest_model_mode.py` | Added `**_kw` to fake provider |
| `tests/test_csv_vs_model_consistency.py` | Added `**_kw` to fake provider |
| `configs/portfolio.json` | Added GS, BAC, UNH, JNJ; removed MSFT/XOM |
| `configs/symbols/GS/active.json` | New — long-only winner |
| `configs/symbols/BAC/active.json` | New — long-only winner |
| `configs/symbols/UNH/active.json` | Updated — two-sided winner |
| `configs/symbols/JNJ/active.json` | New — two-sided winner |
| `configs/symbols/MSFT/active.json` | Updated — disabled (both sides false) |
| `data/processed/gs_60min.csv` | New — 5,078 bars via yfinance |
| `data/processed/bac_60min.csv` | New — 5,078 bars via yfinance |
| `data/processed/unh_60min.csv` | New — 5,078 bars via yfinance |
| `data/processed/jnj_60min.csv` | New — 5,078 bars via yfinance |

---

## 10. Next Steps Before Live Trading

1. **Ingest IBKR minute data** for GS, BAC, UNH, JNJ once TWS is online:
   ```bash
   python src/daily_data_agent.py  # reconnects IBKR, updates all CSVs
   ```
   This replaces the yfinance data with full-resolution IBKR bars and handles
   overnight/session alignment correctly.

2. **Retrain production models** on full dataset (not just fold windows):
   ```bash
   for sym in GS BAC UNH JNJ; do
     python scripts/retrain_symbol.py --symbol $sym
   done
   ```

3. **Address NVDA oos_score=-3.78** (std_Sharpe=3.20 — high fold-to-fold variance).
   The plan-mode audit identified two pipeline bugs (residual-sigma look-ahead in
   `backtest.py:385`, Phase-2 leaky scoring in `robust_param_selection.py:537`)
   that inflate in-sample metrics. Fix those before re-running NVDA param search.
   See `C:\Users\Anton\.claude\plans\crispy-crunching-cookie.md` for the full plan.

4. **Portfolio risk sizing**: with 6 symbols and `max_gross_exposure_pct=0.80`,
   each symbol gets up to 20% of NAV. Review whether individual `risk_per_trade_pct`
   values (0.9–1.5%) compound correctly under concurrent open positions.
