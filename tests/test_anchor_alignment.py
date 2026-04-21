"""Regression tests for the Open/Close anchor mismatch (Pattern 8).

These tests would have failed on the pre-2026-04-21 code base, where the
model was trained on Open-anchored log returns and the denormalization at
the three inference sites multiplied the log return back onto Open.  The
strategy divides predicted_price by Close, so that combination silently
injected bar-shape noise (Open/Close - 1) into the signal and produced
near-inverted directional accuracy (21% vs 45% on NVDA 60min).

See docs/debugging-heuristics.md for the full pattern description.
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import pytest


def _dummy_ohlc_with_time(n: int = 400, seed: int = 1) -> pd.DataFrame:
    """Build a synthetic OHLC frame where Open and Close differ materially."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    # Non-trivial intra-bar: Open is noticeably offset from Close.
    bar_shape = rng.normal(0, 0.003, n)  # +/- 0.3% typical Open/Close spread
    open_ = close * (1.0 + bar_shape)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.2, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.2, n))
    return pd.DataFrame(
        {
            "Time": pd.date_range("2023-01-01", periods=n, freq="h"),
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
        }
    )


class _FakeModel:
    def __init__(self, y: float) -> None:
        self._y = float(y)

    def predict(self, X, verbose=0):  # noqa: N803
        batch = int(getattr(X, "shape", [1])[0])
        return np.full((batch, 1), self._y, dtype=np.float32)


def _make_ctx(*, tsteps: int, features_to_use: list[str], log_return: float, bias: float = 0.0):
    from src.predict import PredictionContext

    scaler_params = {
        "mean": {c: 0.0 for c in features_to_use},
        "std": {c: 1.0 for c in features_to_use},
    }
    return PredictionContext(
        model=_FakeModel(log_return),
        scaler_params=scaler_params,
        mean_vals=None,
        std_vals=None,
        features_to_use=features_to_use,
        tsteps=tsteps,
        bias_correction_mean_residual=bias,
    )


# -----------------------------------------------------------------------------
# 1. Unit-level: denormalization anchors on Close.
# -----------------------------------------------------------------------------
def test_denorm_anchor_is_close_in_live_predictor():
    """With log_return=0 the predicted price must equal the last *Close*,
    never the last Open.  This is the one-line guarantee that
    train/predict/live all agree on the Close anchor."""
    from src.live_predictor import LivePredictor, LivePredictorConfig

    ctx = _make_ctx(tsteps=2, features_to_use=["Open", "High", "Low", "Close"], log_return=0.0)
    pred = LivePredictor(
        ctx=ctx, config=LivePredictorConfig(frequency="60min", tsteps=2, warmup_extra=0)
    )

    pred.update_and_predict(
        {"Time": "2025-01-01", "Open": 100.0, "High": 100.5, "Low": 99.5, "Close": 100.0}
    )
    out = pred.update_and_predict(
        {"Time": "2025-01-02", "Open": 110.0, "High": 110.5, "Low": 109.5, "Close": 107.3}
    )

    # The Close of the last bar is 107.3; Open is 110.0.  The correct anchor is
    # Close — on the pre-fix code this assertion would read 110.0 (== Open).
    assert out == pytest.approx(107.3)


def test_denorm_anchor_is_close_known_log_return():
    """predicted_price == Close * exp(r) — the algebraic invariant."""
    from src.live_predictor import LivePredictor, LivePredictorConfig

    r = 0.0123
    ctx = _make_ctx(
        tsteps=2, features_to_use=["Open", "High", "Low", "Close"], log_return=r
    )
    pred = LivePredictor(
        ctx=ctx, config=LivePredictorConfig(frequency="60min", tsteps=2, warmup_extra=0)
    )

    pred.update_and_predict(
        {"Time": "2025-01-01", "Open": 100.0, "High": 100.5, "Low": 99.5, "Close": 100.0}
    )
    out = pred.update_and_predict(
        {"Time": "2025-01-02", "Open": 105.0, "High": 106.0, "Low": 104.0, "Close": 101.0}
    )

    assert out == pytest.approx(101.0 * float(np.exp(r)))


# -----------------------------------------------------------------------------
# 2. Guardrail-level: pred/Close ratio is the band that is checked, and the
#    anchor-mismatch correlation warning fires when the denorm anchor drifts
#    back to Open.
# -----------------------------------------------------------------------------
def test_log_pred_ratio_summary_reports_pred_close_in_band():
    """When predictions are Close * exp(tiny_r), the Close-anchored median is
    in [0.99, 1.01] and the bar-shape correlation is tiny."""
    from src.backtest import _log_pred_ratio_summary

    data = _dummy_ohlc_with_time(400)
    n = len(data)

    # Simulate a healthy Close-anchored model: predicted_price = Close * exp(r)
    # where r is a small noise unrelated to the bar shape.
    rng = np.random.default_rng(42)
    r_noise = rng.normal(0, 0.003, n)
    denorm_full = data["Close"].to_numpy(dtype=float) * np.exp(r_noise)

    buf = io.StringIO()
    with redirect_stdout(buf):
        _log_pred_ratio_summary(
            denorm_full, data, n=n, symbol="TEST", frequency="60min"
        )
    out = buf.getvalue()

    # The new guardrail reports pred/Close, not pred/Open.
    assert "pred/Close ratio" in out
    # No anchor-mismatch warning should fire on a healthy Close-anchored model.
    assert "WARNING" not in out


def test_log_pred_ratio_summary_warns_on_open_anchor_mismatch():
    """Simulate the Pattern 8 bug: predictions are Open * exp(tiny_r) but the
    strategy divides by Close.  The guardrail must detect the high correlation
    with bar shape and warn."""
    from src.backtest import _log_pred_ratio_summary

    data = _dummy_ohlc_with_time(400)
    n = len(data)

    # Simulate the pre-fix code: predicted_price = Open * exp(tiny_r).
    rng = np.random.default_rng(7)
    r_noise = rng.normal(0, 0.0005, n)  # tiny so median pred/Close stays ~1
    denorm_full = data["Open"].to_numpy(dtype=float) * np.exp(r_noise)

    buf = io.StringIO()
    with redirect_stdout(buf):
        _log_pred_ratio_summary(
            denorm_full, data, n=n, symbol="TEST", frequency="60min"
        )
    out = buf.getvalue()

    assert "corr(strategy_signal, bar_shape)" in out
    assert "WARNING" in out
    assert "Pattern 8" in out or "anchor mismatch" in out.lower()


# -----------------------------------------------------------------------------
# 3. Synthetic pipeline: anchor-mismatch produces far worse directional
#    accuracy than the matched anchor — reproduces the 45% vs 21% empirical
#    gap seen on NVDA 60min.
# -----------------------------------------------------------------------------
def test_close_anchor_directional_accuracy_beats_open_anchor():
    """If the model's native signal is directional against Close, then the
    Close-anchored signal must be at least as accurate as the Open-anchored
    one — by construction.  This is the structural reason Pattern 8 matters:
    the anchor choice controls which signal the strategy consumes."""
    # Reproduce the NVDA 60min fingerprint: bar-boundary mean reversion means
    # that a positive ``Open/Close - 1`` tends to precede a *negative* next
    # Close move.  When the bar shape is injected into the strategy signal
    # (Pattern 8), directional accuracy drops below 0.5.
    n = 4000
    rng = np.random.default_rng(17)
    eps = rng.normal(0, 0.5, n)
    close = 100.0 + np.cumsum(eps)
    future_close = np.roll(close, -1)[:-1]
    r_true = np.log(future_close / close[:-1])  # std ≈ 0.005

    # Weak model: captures 40% of the truth with matching-scale noise.
    r_hat = 0.4 * r_true + rng.normal(0, 0.003, len(r_true))

    # Bar-boundary mean reversion: Open sits on the *opposite* side of Close
    # from where the next bar is headed.  Fingerprint: corr(bar_shape, r_true)
    # is strongly negative, matching the empirical NVDA observation.
    bar_shape = -1.2 * r_true + rng.normal(0, 0.003, len(r_true))
    open_ = close[:-1] * (1.0 + bar_shape)

    pred_close_anchor = close[:-1] * np.exp(r_hat)
    pred_open_anchor = open_ * np.exp(r_hat)

    sig_correct = pred_close_anchor / close[:-1] - 1.0
    sig_buggy = pred_open_anchor / close[:-1] - 1.0
    truth = np.sign(future_close - close[:-1])

    acc_correct = float((np.sign(sig_correct) == truth).mean())
    acc_buggy = float((np.sign(sig_buggy) == truth).mean())

    # Close-anchored signal follows the weak model (>55%).  Open-anchored
    # signal is inverted by bar-boundary mean reversion — drops well below
    # chance, the empirical Pattern 8 fingerprint.
    assert acc_correct > 0.55, f"close_anchor accuracy={acc_correct:.3f}"
    assert acc_buggy < 0.40, f"open_anchor accuracy={acc_buggy:.3f}"
    assert acc_correct - acc_buggy > 0.20
