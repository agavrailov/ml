from __future__ import annotations

import pytest

from src.core.contracts import TrainRequest
from src.jobs import store
from src.jobs.handlers import train_job


def test_train_job_writes_result_and_copies_artifacts(tmp_path, monkeypatch):
    # Isolate runs/ under tmp_path so the test doesn't touch the repo.
    monkeypatch.setattr(store, "repo_root", lambda: tmp_path)

    registry_dir = tmp_path / "registry"
    registry_dir.mkdir(parents=True, exist_ok=True)

    model_path = registry_dir / "my_model.keras"
    model_path.write_bytes(b"dummy")

    metrics_path = registry_dir / "my_model.metrics.json"
    metrics_path.write_text("{}", encoding="utf-8")

    bias_path = registry_dir / "bias_correction.json"
    bias_path.write_text("{\"mean_residual\": 0.0}", encoding="utf-8")

    def _fake_train_model(**_kwargs):
        return 0.123, str(model_path), str(bias_path)

    monkeypatch.setattr(train_job, "train_model", _fake_train_model)

    job_id = "job123"
    req = TrainRequest(frequency="15min", tsteps=5)

    out = train_job.run(job_id, req)

    assert out.validation_loss == pytest.approx(0.123)
    assert out.model_filename == "my_model.keras"
    assert out.metrics_filename == "my_model.metrics.json"
    assert out.bias_correction_filename == "bias_correction.json"

    # Copied artifacts should exist under runs/<job_id>/artifacts.
    artifacts = tmp_path / "runs" / job_id / "artifacts"
    assert (artifacts / "my_model.keras").exists()
    assert (artifacts / "my_model.metrics.json").exists()
    assert (artifacts / "bias_correction.json").exists()

    # result.json should be present and contain the stable fields.
    result_json = tmp_path / "runs" / job_id / "result.json"
    data = store.read_json(result_json)
    assert data["model_filename"] == "my_model.keras"
    assert data["metrics_filename"] == "my_model.metrics.json"
    assert data["bias_correction_filename"] == "bias_correction.json"


def test_train_job_raises_on_none_result(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "repo_root", lambda: tmp_path)

    monkeypatch.setattr(train_job, "train_model", lambda **_kwargs: None)

    with pytest.raises(RuntimeError):
        train_job.run("job_none", TrainRequest(frequency="15min", tsteps=5))
