from __future__ import annotations

import json
from pathlib import Path


class _DummyHistory:
    def __init__(self) -> None:
        self.history = {
            "loss": [2.0, 1.0],
            "val_loss": [3.0, 0.25, 0.5],
            "mae": [1.0, 0.7],
        }


class _DummyModel:
    def fit(self, *args, **kwargs):  # pragma: no cover
        return _DummyHistory()

    def save(self, path: str):  # pragma: no cover
        Path(path).write_text("dummy", encoding="utf-8")


def test_train_and_save_model_writes_metrics_sidecar(tmp_path: Path) -> None:
    from src import model as model_mod

    reg = tmp_path / "registry"

    final_loss, model_path, _ts = model_mod.train_and_save_model(
        model=_DummyModel(),
        X_train=None,
        Y_train=None,
        X_val=None,
        Y_val=None,
        epochs=10,
        batch_size=64,
        frequency="60min",
        tsteps=5,
        model_registry_dir=str(reg),
    )

    assert final_loss == 0.25
    assert Path(model_path).exists()

    metrics_path = Path(model_path.replace(".keras", ".metrics.json"))
    assert metrics_path.exists()

    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert data["model_filename"].endswith(".keras")
    assert data["frequency"] == "60min"
    assert data["tsteps"] == 5
    assert data["epochs"] == 10
    assert data["batch_size"] == 64
    assert data["validation_loss"] == 0.25
