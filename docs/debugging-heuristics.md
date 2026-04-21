# Debugging Heuristics: Data-Corruption Bugs in the LSTM Pipeline

These heuristics come out of the 2026-04 "flat equity curve from 2024 onwards"
bug, where a silent in-place mutation in `add_features` caused per-bar
predictions to be aligned to the wrong `Time` values. They are expensive
lessons — encode them so the next regression is loud, not silent.

Read alongside the guardrail plan in
`~/.claude/plans/modular-zooming-reef.md`.

---

## The six heuristics

### 1. "Model quality" is the last suspect, not the first.

If the backtest equity curve looks wrong, the overwhelmingly likely cause is
a **pipeline** bug, not the **model**. Pred/Open ratios outside the
`[0.99, 1.01]` band on a next-bar log-return model almost always indicate
misalignment / stale scaler / wrong denormalization. Check the pipeline
first; re-investigate the model only after every transformation stage has
been verified end-to-end.

The guardrail that operationalizes this heuristic is
`_log_pred_ratio_summary` in `src/backtest.py`, which prints p05/p50/p95 of
`predicted_price / Open` on every `_make_model_prediction_provider` call
and warns loudly when `p50` is outside the band.

### 2. Healthy raw log-return head outputs look like:

- `mean ≈ 0`
- `std ≈ 0.003`
- `range ≈ [-0.02, +0.02]`

Anything dramatically different at the **raw** model output — before any
denormalization — means the head itself is the problem. Don't try to
"correct" it downstream; fix the head.

### 3. Prefer Time-based joins over positional alignment.

Every `iloc[:n]` or "pad with NaN to length n" appearing as a **fixup
after** an alignment step is a code smell. Those patches silently turn
upstream off-by-N bugs into "mostly works" checkpoints that are nearly
impossible to diagnose later.

The `_make_model_prediction_provider` fail-loud invariant
(`len(merged) == len(data)` or raise) operationalizes this.

### 4. An on-disk artifact written by older code can mask a current bug.

When debugging a checkpointed pipeline, **always regenerate the checkpoint
from scratch** before drawing any conclusions. In the 2026-04 bug, the
on-disk checkpoint had correct Times because older code did not have the
mutation bug; the bug only became visible on regeneration. The sidecar
`*.meta.json` (see `src/checkpoint_provenance.py`) makes such drift visible
by rejecting checkpoints whose OHLC fingerprint no longer matches.

### 5. NaN patterns carry diagnostic signal. Plot positions, not counts.

"6455 NaN out of 21611 at the **end** of the series" was the "aha" that
pointed at truncation; "6455 NaN scattered throughout" would have pointed
at NaN poisoning; "the first 20 NaN only" would have pointed at warmup.
Always histogram / plot the **positions** of NaNs — count alone is
useless.

### 6. Don't generalize from synthetic to production without matching
the input distribution.

A synthetic NaN reproducer showed 500/500 NaN output (total poisoning) while
real data showed 6455/21611 (concentrated-at-end, different failure mode).
Either make the synthetic match the real distribution before concluding
anything from it, or reproduce directly on real data.

---

## The six corruption patterns

Every guardrail we add is triggered by at least one of these.

### Pattern 1 — In-place mutation aliases the caller's frame

A function that does `df.dropna(inplace=True)` or `df.sort_values(inplace=True)`
and returns the same object silently truncates whatever the caller still
holds a reference to. Fix: **always copy at the top of the function** or
document the mutation loudly and require callers to `.copy()`.

Test: `tests/test_data_processing.py::test_add_features_does_not_mutate_input`.

### Pattern 2 — Positional indexing on a frame that was reduced upstream

Anything that assumes `row[i]` in frame A corresponds to `row[i]` in frame B
breaks the moment either frame loses rows (warmup drop, duplicate filter,
date filter). Fix: **join on Time**; if Time is unavailable, raise rather
than guess.

Test: `tests/test_contracts_and_alignment.py::test_checkpoint_interior_time_alignment_detects_off_by_n_drift`.

### Pattern 3 — NaN / NaT propagation through reductions

`np.mean` / `np.std` / `np.sum` return NaN if **any** element is NaN. A
single warmup NaN poisons the entire rolling-window output. Fix: either
**reject NaN at the stage boundary** (preferred — the scaler does this) or
use NaN-safe reductions (what `apply_rolling_bias_and_amplitude_correction`
does now).

Tests:
- `tests/test_scaler.py::test_apply_scaler_rejects_nan_input`
- `tests/test_bias_correction.py::test_rolling_correction_tolerates_sparse_nan`.

### Pattern 5 — Daily OHLC bars broadcast into a minute feed

An upstream concatenation merged a daily-interval feed (e.g. a Yahoo `1d`
download) into the minute CSV. The result is 1 daily OHLC row repeated across
hundreds of minute slots of the same trading date. After resampling to 60min,
every hour bar of the affected days carries the *full daily range* — open far
from low, low far from high — which causes:

- ATR and sigma-return estimates to inflate on contaminated days;
- tight stops sized from "normal" intraday sigma to be instantly hit;
- backtest equity to crash because realized losses are several× the intended
  `risk_per_trade_pct` budget (the bar's "intraday" range is actually a full
  day of price action).

In the 2026-04 NVDA regression this wiped ~$10k → $0 inside the first ~50 bars
of 2023, producing the equity-curve screenshot that started the investigation.

**Signature in raw data:** the same `(Open, High, Low, Close)` 4-tuple
(with `High > Low`) appears in ≥60 minute rows of the same calendar date.
Real illiquid minutes freeze at `O == H == L == C` (zero range), so the
`High > Low` clause is the discriminator between "daily broadcast" and
"legitimate one-minute freeze."

Fix: `src.data_processing.identify_daily_broadcast_rows` plus the call site
in `clean_raw_minute_data` drop broadcast rows **before** the DateTime
dedup (so co-located real minute rows are the ones kept by the dedup).

Tests:
- `tests/test_data_processing_daily_broadcast.py::test_identify_daily_broadcast_rows_flags_repeated_daily_ohlc_only`
- `tests/test_data_processing_daily_broadcast.py::test_clean_raw_minute_data_drops_daily_broadcast_rows`

### Pattern 4 — Checkpoint / provenance metadata is missing

Every artifact written to disk (scaler JSON, checkpoint CSV, trained model)
should carry a fingerprint of the data it was computed from. Without this,
a downstream run cannot tell whether the checkpoint matches the current
OHLC or was generated against a different frame.

Fix: `src/checkpoint_provenance.py` writes a sidecar `.meta.json` with
`{symbol, frequency, ohlc_sha256, n_rows, first_time, last_time,
model_path, scaler_sha256, generated_at_utc}` next to every prediction
checkpoint.

Test: `tests/test_checkpoint_provenance.py::test_stale_checkpoint_rejected_when_data_changes`.

### Pattern 6 — Duplicate timestamps survive into the resampled feed

**What it looks like:** `_load_predictions_csv` prints `[backtest] WARNING: 380
duplicate Time rows in predictions; keeping first occurrence.` The checkpoint
file has two prediction runs concatenated — the second run appended its rows
without truncating the file first.  After silent dedup the CSV is clean, but
the Time-based merge in `_make_model_prediction_provider` would have produced
doubled rows without the dedup guard, causing the merge invariant
(`len(merged) == n`) to raise.

Separately, a second ingestion run can also append duplicate minute bars into
the raw CSV (`data/raw/nvda_minute.csv`). After `convert_minute_to_timeframe`
those doubles survive into hourly bars as doubled-weight bins, inflating ATR
and skewing any rolling statistic that expects one observation per bar.

**Signatures to recognise:**
- `WARNING: N duplicate Time rows` in backtest or prediction output.
- Two hourly bars sharing the same timestamp after resampling.
- ATR / sigma estimates suddenly 2× larger than neighbouring periods.

**Fix (predictions CSV):** `_load_predictions_csv` warns and deduplicates,
keeping the first occurrence.  Always truncate or overwrite (never append) when
re-generating a predictions checkpoint.

**Fix (raw minute data):** `clean_raw_minute_data` calls `drop_duplicates(subset=['DateTime'])`
after removing daily-broadcast rows.  Run the cleaner after every ingestion
batch.

Tests:
- `tests/test_contracts_and_alignment.py::test_load_predictions_csv_deduplicates_on_duplicate_time`
- `tests/test_contracts_and_alignment.py::test_load_predictions_csv_no_warning_on_clean_input`
- `tests/test_data_processing_daily_broadcast.py::test_clean_raw_minute_data_removes_true_duplicate_timestamps`

### Pattern 7 — Bootstrap cursor stride skips half the history

**What it looks like:** The backtest equity curve crashes in the most recent
period even though NVDA itself has recovered.  The raw minute CSV appears to
end at today's date and span the full ``--start`` → today range, but
per-month row counts alternate between ~20k (full month) and ~100–2k
(boundary overlap only).

The culprit is `scripts/ibkr_bootstrap_history.py` walking backward through
time.  Each request fetches ``durationStr="1 M"`` (≈30 days of bars), so the
cursor must step back by **less than 30 days** between requests.  The pre-fix
logic was:

```python
cursor -= timedelta(days=32)
cursor = cursor.replace(day=1)
```

Starting from a mid-month ``end`` date, ``cursor`` immediately snaps to the
first of the *previous* month, and every subsequent step lands on the first
of the month **two** months earlier.  Concretely for ``end=2026-04-20``:
``2026-04-20 → 2026-03-01 → 2026-01-01 → 2025-11-01 → …``.  Every odd
calendar month is never requested.  The freshly-bootstrapped dataset has
~50% coverage, but the artefact looks healthy because each individual fetched
month is dense.

**Signatures to recognise:**
- `raw.groupby(DateTime.dt.to_period("M")).size()` shows alternating
  full/empty months.
- `analyze_gaps.py` reports many multi-week gaps whose end-times cluster on
  day 1–3 of a month at the market-open hour.
- Equity curve shows a sharp drawdown coincident with the most recent odd
  month (first iteration of the bootstrap, starting mid-month).
- Stateful LSTM predictions near odd-month boundaries look like outliers
  because SMA/RSI/ATR rolling windows straddle a 30-day jump.

**Fix:** Step the cursor back by a fixed ``timedelta(days=28)`` (a few days
of intentional overlap, deduped downstream by ``drop_duplicates``).  Do not
snap to month boundaries — the stride must be driven by the chunk duration,
not the calendar.

Tests:
- `tests/test_ibkr_bootstrap_stride.py` — enumerates chunk end-datetimes and
  asserts every day in ``[start, end]`` is covered by at least one chunk.

---

## The backtest ratio gate

The single most valuable guardrail is the pred/Open ratio summary printed
after every `_make_model_prediction_provider` run. Healthy output:

```
[NVDA 60min] pred/Open ratio: n=15156/21616 mean=1.0001 p05=0.9970 p50=1.0002 p95=1.0028 std=0.0017
```

Bug output (what the 2026-04 regression produced):

```
[NVDA 60min] pred/Open ratio: ... p50=0.7500 ...
  !! WARNING: median pred/Open ratio 0.7500 is outside [0.99, 1.01]. ...
```

If you see the warning line, **stop** and investigate the pipeline —
don't try to tune strategy thresholds until the ratio is back in band.
