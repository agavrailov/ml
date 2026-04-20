"""Bias and amplitude correction utilities for LSTM price predictions.

This module provides a small, explicit "bias-correction layer" that can be
applied on top of raw model predictions. It re-estimates bias and amplitude
on a rolling recent window of (prediction, actual) pairs.
"""
from __future__ import annotations

import numpy as np


def apply_rolling_bias_and_amplitude_correction(
    predictions: np.ndarray,
    actuals: np.ndarray,
    window: int,
    global_mean_residual: float = 0.0,
    *,
    enable_amplitude: bool = True,
    amp_min: float = 0.5,
    amp_max: float = 2.0,
) -> np.ndarray:
    """Apply rolling bias + optional amplitude correction on recent data.

    Parameters
    ----------
    predictions:
        1D array of model predictions in *price* space.
    actuals:
        1D array of realized prices, same shape as ``predictions``.
    window:
        Rolling window size (in samples) used to re-estimate bias and
        amplitude from the most recent data.
    global_mean_residual:
        Optional global mean residual (e.g. computed on a held-out
        validation set) used as a prior during the warmup period when
        fewer than ``window`` samples are available.
    enable_amplitude:
        When ``False``, only the rolling bias term is applied and amplitude
        scaling is disabled. This is useful when the base model is already
        well-calibrated in price space and additional rescaling can distort
        signals.
    amp_min / amp_max:
        Soft bounds for the amplitude factor when ``enable_amplitude`` is
        ``True``. These prevent extreme rescaling when the windowed standard
        deviations are very different.

    Returns
    -------
    np.ndarray
        1D array of bias- and (optionally) amplitude-corrected predictions.
    """
    if predictions.shape != actuals.shape:
        raise ValueError("predictions and actuals must have the same shape")
    if predictions.ndim != 1:
        raise ValueError("predictions and actuals must be 1D arrays")
    if window <= 0:
        raise ValueError("window must be a positive integer")

    n = predictions.shape[0]
    if n == 0:
        return predictions.astype(float)

    preds = predictions.astype(float)
    acts = actuals.astype(float)
    corrected = np.zeros_like(preds, dtype=float)

    for i in range(n):
        start = max(0, i - window + 1)
        preds_win = preds[start : i + 1]
        acts_win = acts[start : i + 1]

        # Rolling mean residual on the recent window. Use ``nanmean`` so that
        # sparse NaNs in the input (e.g. warmup bars where a prediction is not
        # yet defined) do not poison the entire rolling window via ordinary
        # ``mean``'s NaN propagation.
        residuals_win = acts_win - preds_win
        finite_resid = residuals_win[np.isfinite(residuals_win)]
        local_mean_resid = float(np.mean(finite_resid)) if finite_resid.size > 0 else 0.0

        # During warmup (fewer than ``window`` samples), blend the global
        # mean residual with the local estimate so that the layer is
        # well-defined from the first point but still adapts to new data.
        if i + 1 < window:
            alpha = (i + 1) / float(window)
            mean_resid = (1.0 - alpha) * float(global_mean_residual) + alpha * local_mean_resid
        else:
            mean_resid = local_mean_resid

        # Amplitude scaling: match std(actuals) over the window while
        # preserving the local mean of predictions. When amplitude
        # correction is disabled we fall back to amp = 1.0 so only the bias
        # term is applied.
        # Use NaN-safe reductions for the window stats for the same reason as
        # above: NaNs in the warmup region must not corrupt the whole window.
        finite_preds = preds_win[np.isfinite(preds_win)]
        finite_acts = acts_win[np.isfinite(acts_win)]
        if enable_amplitude:
            std_act = float(np.std(finite_acts)) if finite_acts.size > 0 else 0.0
            std_pred = float(np.std(finite_preds)) if finite_preds.size > 0 else 0.0
            if std_pred > 0.0:
                amp = std_act / std_pred
            else:
                amp = 1.0
            # Clip extreme amplitudes to avoid pathological rescaling.
            amp = max(amp_min, min(amp_max, amp))
        else:
            amp = 1.0

        mean_pred_win = float(np.mean(finite_preds)) if finite_preds.size > 0 else float(preds[i])
        pred_centered = preds[i] - mean_pred_win
        pred_amp_corrected = pred_centered * amp + mean_pred_win

        corrected[i] = pred_amp_corrected + mean_resid

    return corrected


def compute_rolling_residual_sigma(
    predictions: np.ndarray,
    actuals: np.ndarray,
    window: int,
) -> np.ndarray:
    """Compute a rolling std of residuals (actual - prediction).

    This is used to estimate a per-bar ``model_error_sigma`` series in the
    same spirit as :func:`apply_rolling_bias_and_amplitude_correction`, but
    without modifying the predictions.
    """
    if predictions.shape != actuals.shape:
        raise ValueError("predictions and actuals must have the same shape")
    if predictions.ndim != 1:
        raise ValueError("predictions and actuals must be 1D arrays")
    if window <= 0:
        raise ValueError("window must be a positive integer")

    n = predictions.shape[0]
    if n == 0:
        return predictions.astype(float)

    preds = predictions.astype(float)
    acts = actuals.astype(float)
    residuals = acts - preds

    sigmas = np.zeros_like(residuals, dtype=float)
    for i in range(n):
        start = max(0, i - window + 1)
        resid_win = residuals[start : i + 1]
        # Ignore NaNs in the window.
        resid_win = resid_win[np.isfinite(resid_win)]
        if resid_win.size == 0:
            sigmas[i] = 0.0
        else:
            sigmas[i] = float(np.std(resid_win))

    return sigmas


def shift_sigma_to_observable(sigma: np.ndarray, lag: int) -> np.ndarray:
    """Right-shift a rolling residual-sigma series so every value is observable.

    ``compute_rolling_residual_sigma`` returns ``sigma[i]`` computed over the
    residuals ``[start..i]``. When residuals are built from prediction/actual
    pairs where ``residuals[j]`` requires a *future* price (e.g. ``Open[j + lag]``),
    then ``sigma[i]`` implicitly depends on data that is only observable after
    bar ``i + lag`` closes. To use ``sigma`` at decision time without
    look-ahead, it must be right-shifted by ``lag`` bars so that ``sigma[i]``
    only reflects residuals whose actuals were already realized by bar ``i``.
    The first ``lag`` entries have no observable sigma and are set to zero.
    """
    if sigma.ndim != 1:
        raise ValueError("sigma must be a 1D array")
    if lag < 0:
        raise ValueError("lag must be non-negative")

    shifted = np.zeros_like(sigma)
    if lag == 0:
        shifted[:] = sigma
        return shifted
    if len(sigma) > lag:
        shifted[lag:] = sigma[:-lag]
    return shifted
