from __future__ import annotations

from src.config import TRAINING, MODEL_REGISTRY_DIR, get_model_config


def test_get_model_config_defaults_match_training() -> None:
    """Default ModelConfig should mirror TRAINING + model registry path.

    This guards against accidental drift between the structured TrainingConfig
    and the concrete ModelConfig used by the model API.
    """
    # src.app._load_strategy_defaults() reloads src.config in some tests, which
    # can invalidate a previously-imported ModelConfig class reference.
    from src.config import ModelConfig

    cfg = get_model_config()
    assert isinstance(cfg, ModelConfig)
    assert cfg.frequency == TRAINING.frequency
    assert cfg.tsteps == TRAINING.tsteps
    assert cfg.n_features == TRAINING.n_features
    assert cfg.lstm_units == TRAINING.lstm_units
    assert cfg.batch_size == TRAINING.batch_size
    assert cfg.learning_rate == TRAINING.learning_rate
    assert cfg.n_lstm_layers == TRAINING.n_lstm_layers
    assert cfg.stateful == TRAINING.stateful
    assert cfg.optimizer_name == TRAINING.optimizer_name
    assert cfg.loss_function == TRAINING.loss_function
    assert cfg.model_registry_dir == MODEL_REGISTRY_DIR


def test_get_model_config_allows_overrides() -> None:
    """Callers can override individual fields without affecting TRAINING."""
    cfg = get_model_config(
        frequency="60min",
        tsteps=16,
        n_features=99,
        lstm_units=128,
        batch_size=256,
        learning_rate=0.123,
        n_lstm_layers=2,
        stateful=False,
        optimizer_name="adam",
        loss_function="mae",
    )

    assert cfg.frequency == "60min"
    assert cfg.tsteps == 16
    assert cfg.n_features == 99
    assert cfg.lstm_units == 128
    assert cfg.batch_size == 256
    assert cfg.learning_rate == 0.123
    assert cfg.n_lstm_layers == 2
    assert cfg.stateful is False
    assert cfg.optimizer_name == "adam"
    assert cfg.loss_function == "mae"

    # TRAINING itself must remain unchanged (immutability / no side effects).
    assert TRAINING.frequency != "60min" or TRAINING.tsteps != 16
