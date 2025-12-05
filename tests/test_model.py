import pytest
import tensorflow as tf
from tensorflow import keras
import numpy as np
from datetime import datetime

# Import necessary components from src.config
from src.config import (
    TSTEPS, BATCH_SIZE, LSTM_UNITS, LEARNING_RATE,
    N_LSTM_LAYERS, STATEFUL, OPTIMIZER_OPTIONS, LOSS_FUNCTION_OPTIONS
)
from unittest.mock import patch

from src.config import get_model_config
from src.model import (
    build_lstm_model,
    load_stateful_weights_into_non_stateful_model,
    build_model,
    load_model,
    train_and_save_model,
)

# Define a default N_FEATURES for testing purposes, as it's now dynamic
DEFAULT_N_FEATURES = 7

def test_build_lstm_model_architecture_single_layer_stateful():
    """
    Tests if a single-layer stateful LSTM model is built with the correct architecture.
    """
    model = build_lstm_model(
        input_shape=(TSTEPS, DEFAULT_N_FEATURES),
        lstm_units=LSTM_UNITS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        n_lstm_layers=1,
        stateful=True,
        optimizer_name='rmsprop',
        loss_function='mae'
    )

    assert isinstance(model, keras.Model)
    # InputLayer, LSTM, Dropout, Dense
    assert len(model.layers) == 4

    # Check InputLayer
    input_layer = model.layers[0]
    assert isinstance(input_layer, keras.layers.InputLayer)
    assert input_layer.batch_shape == (BATCH_SIZE, TSTEPS, DEFAULT_N_FEATURES)

    # Check LSTM layer
    lstm_layer = model.layers[1]
    assert isinstance(lstm_layer, keras.layers.LSTM)
    assert lstm_layer.units == LSTM_UNITS
    assert lstm_layer.return_sequences is False  # For a single layer, return_sequences is False
    assert lstm_layer.stateful is True
    assert lstm_layer.activation.__name__ == "tanh"

    # Check Dropout layer
    dropout = model.layers[2]
    assert isinstance(dropout, keras.layers.Dropout)

    # Check Dense output layer
    dense_layer = model.layers[3]
    assert isinstance(dense_layer, keras.layers.Dense)
    assert dense_layer.units == 1

def test_build_lstm_model_architecture_multi_layer_non_stateful():
    """
    Tests if a multi-layer non-stateful LSTM model is built with the correct architecture.
    """
    model = build_lstm_model(
        input_shape=(TSTEPS, DEFAULT_N_FEATURES),
        lstm_units=LSTM_UNITS,
        batch_size=None, # Non-stateful model
        learning_rate=LEARNING_RATE,
        n_lstm_layers=2,
        stateful=False,
        optimizer_name='adam',
        loss_function='mse'
    )

    assert isinstance(model, keras.Model)
    # InputLayer, LSTM1, Dropout1, LSTM2, Dropout2, Dense
    assert len(model.layers) == 6

    # Check InputLayer
    input_layer = model.layers[0]
    assert isinstance(input_layer, keras.layers.InputLayer)
    assert input_layer.batch_shape == (None, TSTEPS, DEFAULT_N_FEATURES)  # batch_size is None for non-stateful

    # Check first LSTM layer
    lstm_layer_1 = model.layers[1]
    assert isinstance(lstm_layer_1, keras.layers.LSTM)
    assert lstm_layer_1.units == LSTM_UNITS
    assert lstm_layer_1.return_sequences is True  # Intermediate layers return sequences
    assert lstm_layer_1.stateful is False

    # Check second LSTM layer (last LSTM layer)
    lstm_layer_2 = model.layers[3]  # Adjusted index due to single dropout per LSTM
    assert isinstance(lstm_layer_2, keras.layers.LSTM)
    assert lstm_layer_2.units == LSTM_UNITS
    assert lstm_layer_2.return_sequences is False  # Last LSTM layer returns single output
    assert lstm_layer_2.stateful is False

    # Check Dense output layer
    dense_layer = model.layers[5]  # Adjusted index
    assert isinstance(dense_layer, keras.layers.Dense)
    assert dense_layer.units == 1


def test_build_lstm_model_compilation():
    """
    Tests if the LSTM model compiles successfully with the specified loss, optimizer, and metrics.
    """
    model = build_lstm_model(
        input_shape=(TSTEPS, DEFAULT_N_FEATURES),
        lstm_units=LSTM_UNITS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        n_lstm_layers=1,
        stateful=True,
        optimizer_name='adam',
        loss_function='mae'
    )

    assert model.loss == 'mae'
    assert isinstance(model.optimizer, keras.optimizers.Adam)
    assert model.optimizer.learning_rate.numpy() == pytest.approx(LEARNING_RATE)

def test_build_lstm_model_forward_pass():
    """
    Tests if the model can perform a forward pass with dummy data.
    """
    model = build_lstm_model(
        input_shape=(TSTEPS, DEFAULT_N_FEATURES),
        lstm_units=LSTM_UNITS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        n_lstm_layers=1,
        stateful=True,
        optimizer_name='rmsprop',
        loss_function='mae'
    )

    dummy_input = np.random.rand(BATCH_SIZE, TSTEPS, DEFAULT_N_FEATURES).astype(np.float32)

    output = model.predict(dummy_input, batch_size=BATCH_SIZE)

    assert output.shape == (BATCH_SIZE, 1) # Output shape for single-step prediction
    assert output.dtype == np.float32

def test_load_stateful_weights_into_non_stateful_model():
    """
    Tests if weights can be successfully transferred from a stateful to a non-stateful model.
    """
    # Build stateful model
    stateful_model = build_lstm_model(
        input_shape=(TSTEPS, DEFAULT_N_FEATURES),
        lstm_units=LSTM_UNITS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        n_lstm_layers=1,
        stateful=True,
        optimizer_name='rmsprop',
        loss_function='mae'
    )

    # Build non-stateful model
    non_stateful_model = build_lstm_model(
        input_shape=(TSTEPS, DEFAULT_N_FEATURES),
        lstm_units=LSTM_UNITS,
        batch_size=None,  # Non-stateful
        learning_rate=LEARNING_RATE,
        n_lstm_layers=1,
        stateful=False,
        optimizer_name='rmsprop',
        loss_function='mae'
    )

    # Set some dummy weights to the stateful model
    dummy_weights = [np.random.rand(*w.shape) for w in stateful_model.get_weights()]
    stateful_model.set_weights(dummy_weights)

    # Transfer weights
    load_stateful_weights_into_non_stateful_model(stateful_model, non_stateful_model)

    # Check if weights are equal
    for w1, w2 in zip(stateful_model.get_weights(), non_stateful_model.get_weights()):
        np.testing.assert_array_almost_equal(w1, w2)


def test_build_model_uses_model_config_defaults():
    """High-level build_model should honour ModelConfig defaults."""
    cfg = get_model_config()

    model = build_model()

    assert isinstance(model, keras.Model)
    input_layer = model.layers[0]
    assert isinstance(input_layer, keras.layers.InputLayer)

    if cfg.stateful:
        # Fixed batch dimension when using a stateful LSTM.
        assert input_layer.batch_shape == (cfg.batch_size, cfg.tsteps, cfg.n_features)
    else:
        # Dynamic batch dimension for stateless models.
        assert input_layer.batch_shape == (None, cfg.tsteps, cfg.n_features)


def test_load_model_delegates_to_keras_load_model():
    """load_model wrapper must delegate to keras.models.load_model once."""
    dummy_return = object()

    with patch("src.model.keras.models.load_model") as mock_load:
        mock_load.return_value = dummy_return
        result = load_model("dummy_model.keras")

    mock_load.assert_called_once_with("dummy_model.keras", compile=True)
    assert result is dummy_return


def test_train_and_save_model_trains_and_saves_with_expected_name(tmp_path):
    """train_and_save_model should fit once and save with the expected filename."""
    # Dummy arrays; shapes are not important for this test because we mock fit.
    X_train = np.ones((2, 3, 1), dtype=np.float32)
    Y_train = np.ones((2, 1), dtype=np.float32)
    X_val = np.ones((2, 3, 1), dtype=np.float32)
    Y_val = np.ones((2, 1), dtype=np.float32)

    model = keras.Sequential()
    # Patch fit/save and datetime/os.makedirs for deterministic behaviour.
    with (
        patch("src.model.EarlyStopping") as mock_es,
        patch("src.model.datetime") as mock_dt,
        patch("src.model.os.makedirs") as mock_makedirs,
    ):
        mock_es.return_value = object()

        history = type("H", (), {"history": {"val_loss": [0.5, 0.4]}})()
        with patch.object(model, "fit", return_value=history) as mock_fit, patch.object(
            model, "save"
        ) as mock_save:
            mock_dt.now.return_value = datetime(2025, 1, 1, 0, 0, 0)

            final_loss, model_path, timestamp = train_and_save_model(
                model=model,
                X_train=X_train,
                Y_train=Y_train,
                X_val=X_val,
                Y_val=Y_val,
                epochs=3,
                batch_size=2,
                frequency="15min",
                tsteps=5,
                model_registry_dir=str(tmp_path),
            )

    # fit called once with expected arguments (validation_data, callbacks, etc.).
    mock_fit.assert_called_once()

    # Final loss is the best validation loss.
    assert final_loss == 0.4
    assert timestamp == "20250101_000000"

    expected_path = tmp_path / "my_lstm_model_15min_tsteps5_20250101_000000.keras"
    assert model_path == str(expected_path)
    # Ensure the model registry directory is created; experiments logging may
    # create additional directories, so we only assert that this path appears
    # among the makedirs calls, not that it is the only one.
    mock_makedirs.assert_any_call(str(tmp_path), exist_ok=True)
    mock_save.assert_called_once_with(str(expected_path))
