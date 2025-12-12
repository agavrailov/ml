import numpy as np
import pytest


class _FakeModel:
    def __init__(self, y: float) -> None:
        self._y = float(y)

    def predict(self, X, verbose=0):  # noqa: N803
        # Return shape (batch, 1)
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
        mean_vals=None,  # not used by LivePredictor
        std_vals=None,  # not used by LivePredictor
        features_to_use=features_to_use,
        tsteps=tsteps,
        bias_correction_mean_residual=bias,
    )


def test_live_predictor_warmup_returns_close():
    from src.live_predictor import LivePredictor, LivePredictorConfig

    ctx = _make_ctx(tsteps=3, features_to_use=["Open", "High", "Low", "Close"], log_return=0.0)
    pred = LivePredictor(ctx=ctx, config=LivePredictorConfig(frequency="60min", tsteps=3, warmup_extra=0))

    out1 = pred.update_and_predict({"Time": "2025-01-01", "Open": 10, "High": 10, "Low": 10, "Close": 11})
    out2 = pred.update_and_predict({"Time": "2025-01-02", "Open": 12, "High": 12, "Low": 12, "Close": 13})

    assert out1 == 11.0
    assert out2 == 13.0


def test_live_predictor_price_mapping_from_log_return():
    from src.live_predictor import LivePredictor, LivePredictorConfig

    # log return=0 => predicted price equals last Open.
    ctx = _make_ctx(tsteps=3, features_to_use=["Open", "High", "Low", "Close"], log_return=0.0)
    pred = LivePredictor(ctx=ctx, config=LivePredictorConfig(frequency="60min", tsteps=3, warmup_extra=0))

    pred.update_and_predict({"Time": "2025-01-01", "Open": 100, "High": 100, "Low": 100, "Close": 100})
    pred.update_and_predict({"Time": "2025-01-02", "Open": 110, "High": 110, "Low": 110, "Close": 110})
    out = pred.update_and_predict({"Time": "2025-01-03", "Open": 120, "High": 120, "Low": 120, "Close": 121})

    assert out == pytest.approx(120.0)


def test_live_predictor_applies_bias_correction_in_log_return_space():
    from src.live_predictor import LivePredictor, LivePredictorConfig

    # model predicts 0, bias=log(2) => predicted price ~ 2 * Open.
    bias = float(np.log(2.0))
    ctx = _make_ctx(
        tsteps=2,
        features_to_use=["Open", "High", "Low", "Close"],
        log_return=0.0,
        bias=bias,
    )
    pred = LivePredictor(ctx=ctx, config=LivePredictorConfig(frequency="60min", tsteps=2, warmup_extra=0))

    pred.update_and_predict({"Time": "2025-01-01", "Open": 10, "High": 10, "Low": 10, "Close": 10})
    out = pred.update_and_predict({"Time": "2025-01-02", "Open": 11, "High": 11, "Low": 11, "Close": 11})

    assert out == pytest.approx(22.0, rel=1e-6)
