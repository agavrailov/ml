import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Import necessary components from src.config
from src.config import (
    TSTEPS, BATCH_SIZE, LSTM_UNITS, LEARNING_RATE,
    N_LSTM_LAYERS, STATEFUL, OPTIMIZER_OPTIONS, LOSS_FUNCTION_OPTIONS
)
from src.model import build_lstm_model, load_stateful_weights_into_non_stateful_model

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
    # InputLayer, LSTM, Dropout1, Dropout2, Dense
    assert len(model.layers) == 5

    # Check InputLayer
    input_layer = model.layers[0]
    assert isinstance(input_layer, keras.layers.InputLayer)
    assert input_layer.batch_shape == (BATCH_SIZE, TSTEPS, DEFAULT_N_FEATURES)

    # Check LSTM layer
    lstm_layer = model.layers[1]
    assert isinstance(lstm_layer, keras.layers.LSTM)
    assert lstm_layer.units == LSTM_UNITS
    assert lstm_layer.return_sequences is False # For a single layer, return_sequences is False
    assert lstm_layer.stateful is True
    assert lstm_layer.activation.__name__ == 'tanh'

    # Check Dropout layers
    dropout1 = model.layers[2]
    assert isinstance(dropout1, keras.layers.Dropout)
    dropout2 = model.layers[3]
    assert isinstance(dropout2, keras.layers.Dropout)

    # Check Dense output layer
    dense_layer = model.layers[4]
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
    # InputLayer, LSTM1, Dropout1, Dropout2, LSTM2, Dropout1, Dropout2, Dense
    assert len(model.layers) == 8

    # Check InputLayer
    input_layer = model.layers[0]
    assert isinstance(input_layer, keras.layers.InputLayer)
    assert input_layer.batch_shape == (None, TSTEPS, DEFAULT_N_FEATURES) # batch_size is None for non-stateful

    # Check first LSTM layer
    lstm_layer_1 = model.layers[1]
    assert isinstance(lstm_layer_1, keras.layers.LSTM)
    assert lstm_layer_1.units == LSTM_UNITS
    assert lstm_layer_1.return_sequences is True # Intermediate layers return sequences
    assert lstm_layer_1.stateful is False

    # Check second LSTM layer (last LSTM layer)
    lstm_layer_2 = model.layers[4] # Adjusted index due to dropouts
    assert isinstance(lstm_layer_2, keras.layers.LSTM)
    assert lstm_layer_2.units == LSTM_UNITS
    assert lstm_layer_2.return_sequences is False # Last LSTM layer returns single output
    assert lstm_layer_2.stateful is False

    # Check Dense output layer
    dense_layer = model.layers[7] # Adjusted index
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
        batch_size=None, # Non-stateful
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