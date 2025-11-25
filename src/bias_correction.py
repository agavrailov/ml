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
) -> np.ndarray:
    """Apply rolling bias + amplitude correction on recent data.

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

    Returns
    -------
    np.ndarray
        1D array of bias- and amplitude-corrected predictions.
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

        # Rolling mean residual on the recent window.
        residuals_win = acts_win - preds_win
        local_mean_resid = float(np.mean(residuals_win)) if residuals_win.size > 0 else 0.0

        # During warmup (fewer than ``window`` samples), blend the global
        # mean residual with the local estimate so that the layer is
        # well-defined from the first point but still adapts to new data.
        if i + 1 < window:
            alpha = (i + 1) / float(window)
            mean_resid = (1.0 - alpha) * float(global_mean_residual) + alpha * local_mean_resid
        else:
            mean_resid = local_mean_resid

        # Amplitude scaling: match std(actuals) over the window while
        # preserving the local mean of predictions.
        std_act = float(np.std(acts_win))
        std_pred = float(np.std(preds_win))
        if std_pred > 0.0:
            amp = std_act / std_pred
        else:
            amp = 1.0

        pred_centered = preds[i] - float(np.mean(preds_win))
        pred_amp_corrected = pred_centered * amp + float(np.mean(preds_win))

        corrected[i] = pred_amp_corrected + mean_resid

    return corrected
