from __future__ import annotations

import json
from pathlib import Path


def _touch(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_get_latest_best_model_path_falls_back_to_registry_when_best_file_missing(
    monkeypatch, tmp_path: Path
) -> None:
    from src import config

    reg = tmp_path / "models" / "registry"
    reg.mkdir(parents=True)

    # Two candidates; lexicographically later timestamp should be selected.
    m1 = reg / "my_lstm_model_1min_tsteps5_20250101_000000.keras"
    m2 = reg / "my_lstm_model_1min_tsteps5_20251211_000000.keras"
    _touch(m1)
    _touch(m2)

    bias = reg / "bias_correction_1min_tsteps5_20251211_000000.json"
    _touch(bias, content=json.dumps({"mean_residual": 0.0}))

    monkeypatch.setattr(config, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(config, "MODEL_REGISTRY_DIR", str(reg))

    model_path, bias_path, feats, units, layers = config.get_latest_best_model_path(
        target_frequency="1min",
        tsteps=5,
    )

    assert model_path == str(m2)
    assert bias_path == str(bias)
    assert feats is None
    assert units is None
    assert layers is None


def test_get_latest_best_model_path_falls_back_when_best_file_has_no_matching_entry(
    monkeypatch, tmp_path: Path
) -> None:
    from src import config

    reg = tmp_path / "models" / "registry"
    reg.mkdir(parents=True)

    m = reg / "my_lstm_model_1min_tsteps5_20251211_000000.keras"
    _touch(m)

    # best_hyperparameters.json exists but targets a different (frequency, tsteps).
    best_path = tmp_path / "best_hyperparameters.json"
    best_path.write_text(json.dumps({"60min": {"5": {"validation_loss": 0.1, "model_filename": "x.keras"}}}), encoding="utf-8")

    monkeypatch.setattr(config, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(config, "MODEL_REGISTRY_DIR", str(reg))

    model_path, *_ = config.get_latest_best_model_path(target_frequency="1min", tsteps=5)

    assert model_path == str(m)


def test_get_latest_best_model_path_prefers_best_hps_when_present(
    monkeypatch, tmp_path: Path
) -> None:
    from src import config

    reg = tmp_path / "models" / "registry"
    reg.mkdir(parents=True)

    # Latest file exists but best_hps points to the older one with better loss.
    m_old = reg / "my_lstm_model_1min_tsteps5_20250101_000000.keras"
    m_new = reg / "my_lstm_model_1min_tsteps5_20251211_000000.keras"
    _touch(m_old)
    _touch(m_new)

    best_path = tmp_path / "best_hyperparameters.json"
    best_path.write_text(
        json.dumps(
            {
                "1min": {
                    "5": {
                        "validation_loss": 0.0001,
                        "model_filename": m_old.name,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(config, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(config, "MODEL_REGISTRY_DIR", str(reg))

    model_path, *_ = config.get_latest_best_model_path(target_frequency="1min", tsteps=5)

    assert model_path == str(m_old)
