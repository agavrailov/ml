"""Regression test for Pattern 9: double bias correction.

Background
----------
The 2026-04-21 plan (step 5) wired the training-time
``bias_correction_mean_residual`` (log-return space) into the backtest's
model-mode denormalization step.  The intent was backtest/live parity with
``src/live_predictor.py``, which applies the same static residual.

That addition was wrong for the backtest path.  The backtest runs a
*rolling* price-space bias correction (``apply_rolling_bias_and_amplitude_correction``)
immediately after denormalization.  The rolling correction estimates
``mean(acts - preds)`` over a trailing window and subtracts it.  Feeding it
preds that already have ``+mean_residual`` in log space doesn't reduce the
bias twice over the long run, but during the first ``window`` bars (warmup)
the rolling estimate is sparse — leaving a systematic
``+mean_residual`` shift that the strategy reads as a fake uptick.  On
NVDA 60min with ``mean_residual=+0.009`` (log), this produced a cluster of
false long entries in the opening bars that all lost money and — with tight
k_atr thresholds — exhausted the strategy's willingness to fire for the
rest of the dataset.

The fix is to NOT apply ``mean_residual`` in the backtest model-mode path.
Live prediction keeps the static residual because it has no rolling stage.

This regression test asserts the offending line is gone from
``src/backtest.py``'s model-mode prediction provider.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _read_backtest_source() -> str:
    here = Path(__file__).resolve().parent
    src = here.parent / "src" / "backtest.py"
    return src.read_text(encoding="utf-8")


def test_backtest_model_mode_does_not_add_saved_mean_residual() -> None:
    """The model-mode denorm path must not add ``bias_correction_mean_residual``
    on top of the rolling price-space correction.  If this assertion fails,
    Pattern 9 has regressed."""
    source = _read_backtest_source()

    # Find the section between "preds_log_full =" initialisation and the
    # ``base_close`` / denorm block — that's the denorm preparation window
    # where the bug lived.
    marker_start = source.index("preds_log_full = np.full(")
    marker_end = source.index("base_close = df_featured[\"Close\"]")
    window = source[marker_start:marker_end]

    # The specific pattern that would double-correct.
    assert "+ _saved_mean_residual" not in window, (
        "Pattern 9 regression: backtest model-mode path is adding the "
        "training-time mean_residual on top of the rolling bias correction. "
        "This creates a startup bias equal to mean_residual for the first "
        "window bars.  See docs/debugging-heuristics.md Pattern 9."
    )
    # The actual variable binding — comments may mention the attribute to
    # explain *why* we don't use it; an executable binding is the red flag.
    assert "_saved_mean_residual = float(" not in window, (
        "Pattern 9 regression: backtest model-mode path is binding "
        "_saved_mean_residual in the denorm window.  The static residual is "
        "subsumed by the rolling correction — applying both creates the "
        "Pattern 9 startup bias."
    )


def test_live_predictor_still_applies_static_mean_residual() -> None:
    """Live prediction has no rolling-correction stage; it *must* keep the
    static residual.  Guards against the opposite regression — removing
    bias correction everywhere."""
    here = Path(__file__).resolve().parent
    src = (here.parent / "src" / "live_predictor.py").read_text(encoding="utf-8")
    assert "bias_correction_mean_residual" in src, (
        "live_predictor.py must apply the static training-time residual — "
        "it has no rolling-correction stage to stand in for it."
    )


def test_rolling_bias_correction_converges_to_zero_on_unbiased_preds() -> None:
    """Sanity check that the rolling correction does handle bias on its own
    once warmed up — the premise for removing the static offset."""
    from src.bias_correction import apply_rolling_bias_and_amplitude_correction

    rng = np.random.default_rng(0)
    n = 2000
    acts = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    # Preds biased by a constant +1.0 in price space.
    bias = 1.0
    preds = acts + bias + rng.normal(0, 0.2, n)

    corrected = apply_rolling_bias_and_amplitude_correction(
        preds.astype(np.float32),
        acts.astype(np.float32),
        window=200,
        global_mean_residual=0.0,
        lookahead_lag=1,
    )

    # After warmup (window + lookahead_lag bars) the bias should be removed.
    warm = corrected[400:]
    act_warm = acts[400:]
    residual = float((act_warm - warm).mean())
    assert abs(residual) < 0.2, (
        f"rolling correction failed to remove constant price bias; "
        f"residual after warmup = {residual:.3f}"
    )
