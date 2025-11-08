import pytest
import tensorflow as tf
from tensorflow import keras
import numpy as np

from src.model import build_lstm_model
from src.config import TSTEPS, N_FEATURES, BATCH_SIZE, LSTM_UNITS, LEARNING_RATE

def test_build_lstm_model_architecture():
    """
    Tests if the LSTM model is built with the correct architecture.
    """
    model = build_lstm_model(
        input_shape=(TSTEPS, N_FEATURES),
        lstm_units=LSTM_UNITS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )

    assert isinstance(model, keras.Model)
    assert len(model.layers) == 3 # InputLayer, LSTM, Dense

    # Check InputLayer
    input_layer = model.layers[0]
    assert isinstance(input_layer, keras.layers.InputLayer)
    assert input_layer.batch_shape == (BATCH_SIZE, TSTEPS, N_FEATURES) # Corrected attribute

    # Check LSTM layer
    lstm_layer = model.layers[1]
    assert isinstance(lstm_layer, keras.layers.LSTM)
    assert lstm_layer.units == LSTM_UNITS
    assert lstm_layer.return_sequences is True
    assert lstm_layer.stateful is True
    assert lstm_layer.activation.__name__ == 'tanh' # Corrected assertion

    # Check Dense output layer
    dense_layer = model.layers[2]
    assert isinstance(dense_layer, keras.layers.Dense)
    assert dense_layer.units == 1

def test_build_lstm_model_compilation():
    """
    Tests if the LSTM model compiles successfully with the specified loss, optimizer, and metrics.
    """
    model = build_lstm_model(
        input_shape=(TSTEPS, N_FEATURES),
        lstm_units=LSTM_UNITS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )

    assert model.loss == 'mse'
    assert isinstance(model.optimizer, keras.optimizers.RMSprop)
    assert model.optimizer.learning_rate.numpy() == pytest.approx(LEARNING_RATE) # Use pytest.approx

def test_build_lstm_model_forward_pass():
    """
    Tests if the model can perform a forward pass with dummy data.
    """
    print(f"DEBUG: BATCH_SIZE (from config) = {BATCH_SIZE}")
    model = build_lstm_model(
        input_shape=(TSTEPS, N_FEATURES),
        lstm_units=LSTM_UNITS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )

    dummy_input = np.random.rand(BATCH_SIZE, TSTEPS, N_FEATURES).astype(np.float32)
    print(f"DEBUG: dummy_input.shape = {dummy_input.shape}")

    # Explicitly pass batch_size to model.predict() for stateful models
    output = model.predict(dummy_input, batch_size=BATCH_SIZE)

    assert output.shape == (BATCH_SIZE, TSTEPS, 1)
    assert output.dtype == np.float32
