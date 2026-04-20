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

## The four corruption patterns

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
