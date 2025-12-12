from __future__ import annotations

import json
from pathlib import Path


def test_filter_training_history_by_freq_and_tsteps() -> None:
    import src.app as app

    history = [
        {"timestamp": "2025-01-01T00:00:00", "frequency": "1min", "tsteps": 5, "validation_loss": 2.0},
        {"timestamp": "2025-01-02T00:00:00", "frequency": "1min", "tsteps": 8, "validation_loss": 1.0},
        {"timestamp": "2025-01-03T00:00:00", "frequency": "5min", "tsteps": 5, "validation_loss": 0.5},
        {"timestamp": "2025-01-04T00:00:00", "frequency": "1min", "tsteps": 5, "validation_loss": 0.1},
    ]

    rows = app._filter_training_history(history, frequency="1min", tsteps=5)

    assert [r["timestamp"] for r in rows] == ["2025-01-04T00:00:00", "2025-01-01T00:00:00"]


def test_get_best_training_row_min_validation_loss() -> None:
    import src.app as app

    rows = [
        {"validation_loss": 2.0, "model_filename": "a.keras"},
        {"validation_loss": 0.5, "model_filename": "b.keras"},
        {"validation_loss": 0.8, "model_filename": "c.keras"},
    ]

    best = app._get_best_training_row(rows)
    assert best is not None
    assert best["model_filename"] == "b.keras"


def test_promote_training_row_writes_best_hyperparameters_json(tmp_path: Path) -> None:
    import src.app as app

    best_hps_path = tmp_path / "best_hyperparameters.json"

    row = {
        "validation_loss": 0.123,
        "model_filename": "my_lstm_model_1min_tsteps5_20250101_000000.keras",
        "bias_correction_filename": "bias_correction_1min_tsteps5_20250101_000000.json",
        "lstm_units": 64,
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": 64,
        "n_lstm_layers": 1,
        "stateful": True,
        "optimizer_name": "rmsprop",
        "loss_function": "mse",
        "features_to_use": ["Open"],
    }

    app._promote_training_row(row=row, best_hps_path=best_hps_path, frequency="1min", tsteps=5)

    data = json.loads(best_hps_path.read_text(encoding="utf-8"))
    assert data["1min"]["5"]["validation_loss"] == 0.123
    assert data["1min"]["5"]["model_filename"] == row["model_filename"]


def test_parse_registry_model_filename_extracts_fields() -> None:
    import src.app as app

    info = app._parse_registry_model_filename(
        "my_lstm_model_60min_tsteps5_20251211_235959.keras"
    )

    assert info is not None
    assert info["frequency"] == "60min"
    assert info["tsteps"] == 5
    assert info["timestamp"] == "2025-12-11T23:59:59"
    assert info["stamp"] == "20251211_235959"


def test_list_registry_models_lists_and_sorts_and_adds_bias_filename(tmp_path: Path) -> None:
    import src.app as app

    reg = tmp_path / "registry"
    reg.mkdir()

    # Two models; second is newer.
    m1 = reg / "my_lstm_model_60min_tsteps5_20250101_000000.keras"
    m2 = reg / "my_lstm_model_60min_tsteps5_20251211_000000.keras"
    m1.write_text("x", encoding="utf-8")
    m2.write_text("y", encoding="utf-8")

    # Metrics only for the newer model.
    m2_metrics = reg / "my_lstm_model_60min_tsteps5_20251211_000000.metrics.json"
    m2_metrics.write_text(json.dumps({"validation_loss": 0.42}), encoding="utf-8")

    # Matching bias file only for the newer model.
    bias = reg / "bias_correction_60min_tsteps5_20251211_000000.json"
    bias.write_text("{}", encoding="utf-8")

    rows = app._list_registry_models(reg)

    assert [r["model_filename"] for r in rows] == [m2.name, m1.name]
    assert rows[0]["bias_correction_filename"] == bias.name
    assert rows[0]["validation_loss"] == 0.42
    assert rows[1]["bias_correction_filename"] is None
    assert rows[1].get("validation_loss") is None


def test_format_timestamp_iso_seconds_truncates() -> None:
    import src.app as app

    assert (
        app._format_timestamp_iso_seconds("2025-12-12T10:58:23.123456+00:00")
        == "2025-12-12T10:58:23"
    )
