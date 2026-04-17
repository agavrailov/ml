"""Regression tests for the residual-sigma look-ahead bug.

See plan ``crispy-crunching-cookie.md`` Step 1a: in
``src/backtest.py::build_model_prediction_provider`` the actuals are aligned
to predictions via ``data["Open"].shift(-ROWS_AHEAD)``. That means
``residuals[i]`` requires ``Open[i + ROWS_AHEAD]``, a price only observable
after bar ``i + ROWS_AHEAD`` closes. The rolling sigma must therefore be
right-shifted by ``ROWS_AHEAD`` bars before it can be used as a decision-time
input without look-ahead.

These tests pin:

1. ``compute_rolling_residual_sigma`` itself uses only the window up to and
   including index ``i`` (a property test on the primitive).
2. ``shift_sigma_to_observable`` moves every sample forward by ``lag`` bars
   and zeros out the leading ``lag`` positions — this is the fix used by
   ``src/backtest.py`` to convert the raw residual-sigma series into a
   look-ahead-free one.
"""
from __future__ import annotations

import numpy as np

from src.bias_correction import (
    compute_rolling_residual_sigma,
    shift_sigma_to_observable,
)
from src.config import ROWS_AHEAD


def test_compute_rolling_residual_sigma_uses_only_past_residuals():
    """sigma[i] must be a function of residuals[0..i] only.

    With predictions == actuals == zeros everywhere except a spike at
    actuals[50], the only non-zero residual is residuals[50] == 10.0. If
    sigma[i] for i < 50 is strictly zero, then the rolling-sigma primitive is
    not peeking into the future.
    """
    n = 100
    predictions = np.zeros(n, dtype=float)
    actuals = np.zeros(n, dtype=float)
    actuals[50] = 10.0

    sigmas = compute_rolling_residual_sigma(
        predictions=predictions,
        actuals=actuals,
        window=20,
    )

    # Every index strictly before the spike must be zero — no residual has
    # contributed anything non-zero yet.
    assert np.all(sigmas[:50] == 0.0), (
        "sigma at indices before the spike must be zero; got "
        f"{sigmas[:50][sigmas[:50] != 0]}"
    )
    # Sanity: at i == 50 the spike is in the window so sigma must be > 0.
    assert sigmas[50] > 0.0


def test_shift_sigma_to_observable_zero_lag_is_identity():
    sigma = np.arange(10, dtype=float)
    shifted = shift_sigma_to_observable(sigma, lag=0)
    np.testing.assert_array_equal(shifted, sigma)
    # Must be a copy, not an alias (callers may mutate).
    assert shifted is not sigma


def test_shift_sigma_to_observable_shifts_by_lag():
    """For lag=L: shifted[L:] == sigma[:-L] and shifted[:L] == 0."""
    lag = 3
    sigma = np.arange(1, 11, dtype=float)  # [1..10]
    shifted = shift_sigma_to_observable(sigma, lag=lag)

    assert shifted.shape == sigma.shape
    np.testing.assert_array_equal(shifted[:lag], np.zeros(lag))
    np.testing.assert_array_equal(shifted[lag:], sigma[:-lag])


def test_shift_sigma_to_observable_when_shorter_than_lag():
    """If the array is shorter than lag, the whole output is zero."""
    sigma = np.array([1.0, 2.0])
    shifted = shift_sigma_to_observable(sigma, lag=5)
    np.testing.assert_array_equal(shifted, np.zeros_like(sigma))


def test_backtest_residual_sigma_series_no_lookahead():
    """Simulate the shift applied in ``src/backtest.py``.

    Start with a 100-long sigma series that has non-zero values at indices
    50..60 (simulating the rolling sigma series coming out of
    ``compute_rolling_residual_sigma``). After applying
    ``shift_sigma_to_observable`` with ``lag == ROWS_AHEAD``, the non-zero
    block must be moved forward by exactly ``ROWS_AHEAD`` bars, and every
    entry before that shifted block must still be zero — importantly,
    the entry at the original spike start (index 50) must now be zero when
    ROWS_AHEAD >= 1 because its value moved to index 50 + ROWS_AHEAD.
    """
    n = 100
    residual_sigma_series = np.zeros(n, dtype=np.float32)
    residual_sigma_series[50:61] = 1.0  # non-zero block at [50, 60]

    shifted = shift_sigma_to_observable(residual_sigma_series, ROWS_AHEAD)

    assert shifted.shape == residual_sigma_series.shape

    if ROWS_AHEAD == 0:
        # Degenerate case: no shift, identity.
        np.testing.assert_array_equal(shifted, residual_sigma_series)
        return

    # Leading ROWS_AHEAD entries are forced to zero (no observable sigma yet).
    np.testing.assert_array_equal(
        shifted[:ROWS_AHEAD], np.zeros(ROWS_AHEAD, dtype=np.float32)
    )

    # The index that used to hold the start of the non-zero block (50) is now
    # strictly before the block — it must be zero after shifting.
    assert shifted[50] == 0.0, (
        f"sigma[50] must be zero after right-shift by ROWS_AHEAD={ROWS_AHEAD}, "
        f"got {shifted[50]}"
    )

    # The non-zero block must have moved forward by exactly ROWS_AHEAD bars.
    expected_start = 50 + ROWS_AHEAD
    expected_end = 60 + ROWS_AHEAD  # inclusive
    if expected_end < n:
        np.testing.assert_array_equal(
            shifted[expected_start : expected_end + 1],
            np.ones(expected_end + 1 - expected_start, dtype=np.float32),
        )

    # And the region strictly before the shifted block is all zeros.
    np.testing.assert_array_equal(
        shifted[:expected_start], np.zeros(expected_start, dtype=np.float32)
    )
