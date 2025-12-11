from __future__ import annotations

import json
import os
from typing import Any

import pytest


class _DummyModel:
    def __init__(self) -> None:
        self._weights: list[Any] = []

    def get_weights(self):  # pragma: no cover
        return self._weights

    def set_weights(self, w):  # pragma: no cover
        self._weights = list(w)


def _write_scaler_params(path: str, *, features: list[str]) -> None:
    payload = {
        "mean": {f: 0.0 for f in features if f != "Time"},
        "std": {f: 1.0 for f in features if f != "Time"},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def test_build_prediction_context_falls_back_to_registry_latest(monkeypatch, tmp_path) -> None:
    """If best/active metadata is missing, we still load the newest registry model."""

    from src import predict

    freq = "60min"
    tsteps = 5

    reg = tmp_path / "registry"
    reg.mkdir()

    # Two candidates; lexicographically later timestamp should be selected.
    m1 = reg / "my_lstm_model_60min_tsteps5_20250101_000000.keras"
    m2 = reg / "my_lstm_model_60min_tsteps5_20251211_000000.keras"
    m1.write_text("x", encoding="utf-8")
    m2.write_text("y", encoding="utf-8")

    scaler_path = tmp_path / "scaler_params_60min.json"
    _write_scaler_params(str(scaler_path), features=predict.FEATURES_TO_USE_OPTIONS[0])

    monkeypatch.setattr(predict, "MODEL_REGISTRY_DIR", str(reg))
    monkeypatch.setattr(predict, "get_latest_best_model_path", lambda **_: (None, None, None, None, None))
    monkeypatch.setattr(predict, "get_active_model_path", lambda **_: None)
    monkeypatch.setattr(predict, "get_scaler_params_json_path", lambda _frequency: str(scaler_path))

    loaded_paths: list[str] = []

    def _fake_load_model(path: str):
        loaded_paths.append(os.path.abspath(path))
        return _DummyModel()

    monkeypatch.setattr(predict, "load_model", _fake_load_model)
    monkeypatch.setattr(predict, "build_lstm_model", lambda **_: _DummyModel())
    monkeypatch.setattr(predict, "load_stateful_weights_into_non_stateful_model", lambda *_: None)

    ctx = predict.build_prediction_context(frequency=freq, tsteps=tsteps)

    assert loaded_paths == [os.path.abspath(str(m2))]
    assert ctx.tsteps == tsteps


def test_build_prediction_context_raises_when_no_model_found(monkeypatch, tmp_path) -> None:
    """When no model artifacts exist, keep raising the same FileNotFoundError."""

    from src import predict

    freq = "60min"
    tsteps = 5

    reg = tmp_path / "registry"
    reg.mkdir()

    scaler_path = tmp_path / "scaler_params_60min.json"
    _write_scaler_params(str(scaler_path), features=predict.FEATURES_TO_USE_OPTIONS[0])

    monkeypatch.setattr(predict, "MODEL_REGISTRY_DIR", str(reg))
    monkeypatch.setattr(predict, "get_latest_best_model_path", lambda **_: (None, None, None, None, None))
    monkeypatch.setattr(predict, "get_active_model_path", lambda **_: None)
    monkeypatch.setattr(predict, "get_scaler_params_json_path", lambda _frequency: str(scaler_path))

    with pytest.raises(FileNotFoundError) as e:
        predict.build_prediction_context(frequency=freq, tsteps=tsteps)

    assert "No best model found for frequency" in str(e.value)
