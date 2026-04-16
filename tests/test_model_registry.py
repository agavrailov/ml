import json
import os
import pytest
from pathlib import Path


@pytest.fixture
def registry_path(tmp_path):
    return tmp_path / "symbol_registry.json"


def test_get_best_model_empty_registry(registry_path):
    from src.model_registry import get_best_model_path
    result = get_best_model_path("NVDA", "60min", 5, registry_path=str(registry_path))
    assert result is None


def test_update_then_get(registry_path, tmp_path):
    from src.model_registry import update_best_model, get_best_model_path
    model_file = tmp_path / "my_model.keras"
    model_file.write_text("")
    update_best_model(
        symbol="NVDA",
        frequency="60min",
        tsteps=5,
        val_loss=0.0001,
        model_path=str(model_file),
        bias_path=None,
        hparams={"lstm_units": 64, "n_lstm_layers": 1},
        registry_path=str(registry_path),
    )
    result = get_best_model_path("NVDA", "60min", 5, registry_path=str(registry_path))
    assert result == str(model_file)


def test_update_only_replaces_if_better(registry_path, tmp_path):
    from src.model_registry import update_best_model, get_best_model_path
    model_a = tmp_path / "model_a.keras"
    model_b = tmp_path / "model_b.keras"
    model_a.write_text("")
    model_b.write_text("")
    update_best_model("NVDA", "60min", 5, 0.0001, str(model_a), None, {}, registry_path=str(registry_path))
    update_best_model("NVDA", "60min", 5, 0.0002, str(model_b), None, {}, registry_path=str(registry_path))  # worse
    result = get_best_model_path("NVDA", "60min", 5, registry_path=str(registry_path))
    assert result == str(model_a)  # still model_a


def test_migrate_from_legacy(registry_path, tmp_path):
    from src.model_registry import migrate_from_legacy_hps, get_best_model_path
    legacy = tmp_path / "best_hyperparameters.json"
    model_file = tmp_path / "my_model.keras"
    model_file.write_text("")
    legacy.write_text(json.dumps({
        "60min": {"5": {"validation_loss": 0.00005, "model_filename": model_file.name}}
    }))
    migrate_from_legacy_hps(
        legacy_hps_path=str(legacy),
        symbol="NVDA",
        model_registry_dir=str(tmp_path),
        registry_path=str(registry_path),
    )
    result = get_best_model_path("NVDA", "60min", 5, registry_path=str(registry_path))
    assert result == str(model_file)
