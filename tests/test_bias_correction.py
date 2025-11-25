import numpy as np

from src.bias_correction import apply_rolling_bias_and_amplitude_correction


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
