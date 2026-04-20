import numpy as np

from src.bias_correction import apply_rolling_bias_and_amplitude_correction


def test_rolling_correction_tolerates_sparse_nan():
    """Sparse NaN in the input must NOT poison the entire rolling window.

    Pattern 3 in docs/debugging-heuristics.md: the pre-fix ``np.mean`` on
    a rolling window returned NaN if any element was NaN, so a single
    warmup NaN contaminated the next `window` predictions.  After the fix
    (using ``np.isfinite`` filters for the mean/std reductions) only the
    row(s) where the window is 100% NaN should remain NaN.
    """
    rng = np.random.default_rng(seed=3)
    n = 500
    actuals = 100.0 + rng.standard_normal(n).cumsum() * 0.1
    predictions = actuals + rng.standard_normal(n) * 0.2

    # Scatter ~1% NaN into predictions.
    nan_idx = rng.choice(n, size=n // 100, replace=False)
    predictions[nan_idx] = np.nan

    corrected = apply_rolling_bias_and_amplitude_correction(
        predictions=predictions,
        actuals=actuals,
        window=50,
        global_mean_residual=0.0,
    )

    # Output length preserved.
    assert corrected.shape == predictions.shape
    # The only NaNs in the output should be at exactly the input-NaN
    # positions (their own window still contains at most 1 NaN but when
    # computing preds[i] directly we use the raw prediction — so the
    # corrected value is NaN wherever the input was NaN).  Crucially,
    # surrounding rows (which had finite predictions) must stay finite.
    finite_input = np.isfinite(predictions)
    assert np.isfinite(corrected[finite_input]).all(), (
        "NaN leaked into rows where the input prediction was finite — "
        "rolling mean is poisoning the window again."
    )


def test_bias_correction_reduces_constant_bias():
    # Model is systematically under-predicting by 5 units.
    actuals = np.linspace(100, 110, 50)
    predictions = actuals - 5.0

    corrected = apply_rolling_bias_and_amplitude_correction(
        predictions=predictions,
        actuals=actuals,
        window=10,
        global_mean_residual=0.0,
    )

    # After correction, mean residual should be much smaller than the original
    # constant bias of 5 units.
    residuals_after = actuals - corrected
    orig_bias = 5.0
    assert abs(residuals_after.mean()) < orig_bias * 0.25


def test_bias_correction_handles_short_series_and_warmup():
    actuals = np.array([100.0, 101.0, 102.0])
    predictions = np.array([99.0, 100.0, 101.0])  # under by ~1

    corrected = apply_rolling_bias_and_amplitude_correction(
        predictions=predictions,
        actuals=actuals,
        window=5,  # larger than series length
        global_mean_residual=1.0,
    )

    # Should return same shape and finite values.
    assert corrected.shape == predictions.shape
    assert np.isfinite(corrected).all()

    # Residuals should be smaller in magnitude than original bias.
    orig_resid = (actuals - predictions).mean()
    new_resid = (actuals - corrected).mean()
    assert abs(new_resid) < abs(orig_resid)
