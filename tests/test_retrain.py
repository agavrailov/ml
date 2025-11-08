import pytest
import pandas as pd
import os
import json
from unittest.mock import patch, MagicMock
import tensorflow as tf # Added import
from tensorflow import keras
import numpy as np

from src.retrain import retrain_model
from src.config import (
    TSTEPS, ROWS_AHEAD, N_FEATURES, BATCH_SIZE, EPOCHS, LSTM_UNITS,
    PROCESSED_DATA_DIR, TRAINING_DATA_CSV, SCALER_PARAMS_JSON, MODEL_SAVE_PATH
)

@pytest.fixture
def setup_retrain_test_environment(tmp_path):
    """
    Fixture to set up a temporary environment for retraining tests,
    including mock model and data files.
    """
    # Create necessary directories
    temp_processed_dir = tmp_path / PROCESSED_DATA_DIR
    temp_processed_dir.mkdir(parents=True, exist_ok=True)
    
    temp_models_dir = tmp_path / "models"
    temp_models_dir.mkdir(parents=True, exist_ok=True)

    # Mock MODEL_SAVE_PATH
    mock_model_save_path = tmp_path / MODEL_SAVE_PATH
    
    # Mock TRAINING_DATA_CSV
    mock_training_data_csv = tmp_path / TRAINING_DATA_CSV
    
    # Mock SCALER_PARAMS_JSON
    mock_scaler_params_json = tmp_path / SCALER_PARAMS_JSON

    # 1. Create a dummy Keras model and save it
    # Need to build a simple model to save, as load_model expects a valid Keras format
    inputs = keras.Input(batch_shape=(BATCH_SIZE, TSTEPS, N_FEATURES), dtype=tf.float32)
    lstm_layer = keras.layers.LSTM(LSTM_UNITS, return_sequences=True, stateful=True, dtype=tf.float32)(inputs)
    outputs = keras.layers.Dense(1, dtype=tf.float32)(lstm_layer)
    dummy_model = keras.Model(inputs=inputs, outputs=outputs)
    dummy_model.compile(optimizer='adam', loss='mse')
    dummy_model.save(mock_model_save_path)

    # 2. Create mock TRAINING_DATA_CSV
    dummy_data = {
        'DateTime': pd.to_datetime(['2023-01-01T00:00', '2023-01-01T01:00', '2023-01-01T02:00', '2023-01-01T03:00',
                                    '2023-01-01T04:00', '2023-01-01T05:00', '2023-01-01T06:00', '2023-01-01T07:00',
                                    '2023-01-01T08:00', '2023-01-01T09:00', '2023-01-01T10:00', '2023-01-01T11:00',
                                    '2023-01-01T12:00', '2023-01-01T13:00', '2023-01-01T14:00', '2023-01-01T15:00',
                                    '2023-01-01T16:00', '2023-01-01T17:00', '2023-01-01T18:00', '2023-01-01T19:00',
                                    '2023-01-01T20:00', '2023-01-01T21:00', '2023-01-01T22:00', '2023-01-01T23:00']),
        'Open': np.random.rand(24),
        'High': np.random.rand(24),
        'Low': np.random.rand(24),
        'Close': np.random.rand(24)
    }
    # Ensure enough data for sequence creation, considering TSTEPS and ROWS_AHEAD
    # The dummy data needs to be long enough to create at least one batch of sequences.
    # For BATCH_SIZE=1, TSTEPS=3, ROWS_AHEAD=60, we need more than 60+3-1 = 62 rows.
    # Let's create more data to be safe.
    num_rows = BATCH_SIZE + TSTEPS + ROWS_AHEAD + 10 # Ensure enough data for at least one batch
    dummy_data = {
        'DateTime': pd.date_range(start='2023-01-01T00:00', periods=num_rows, freq='h'),
        'Open': np.random.rand(num_rows),
        'High': np.random.rand(num_rows),
        'Low': np.random.rand(num_rows),
        'Close': np.random.rand(num_rows)
    }
    pd.DataFrame(dummy_data).to_csv(mock_training_data_csv, index=False)

    # 3. Create mock SCALER_PARAMS_JSON
    dummy_scaler_params = {
        'mean': {'Open': 0.5, 'High': 0.6, 'Low': 0.4, 'Close': 0.55},
        'std': {'Open': 0.1, 'High': 0.1, 'Low': 0.1, 'Close': 0.1}
    }
    with open(mock_scaler_params_json, 'w') as f:
        json.dump(dummy_scaler_params, f)

    yield mock_model_save_path, mock_training_data_csv, mock_scaler_params_json

def test_retrain_model(setup_retrain_test_environment):
    mock_model_save_path, mock_training_data_csv, mock_scaler_params_json = setup_retrain_test_environment

    with patch('src.retrain.keras.models.load_model') as mock_load_model, \
         patch('src.retrain.create_sequences_for_stateful_lstm') as mock_create_sequences, \
         patch('src.config.TRAINING_DATA_CSV', str(mock_training_data_csv)), \
         patch('src.config.SCALER_PARAMS_JSON', str(mock_scaler_params_json)), \
         patch('keras.layers.RNN') as mock_rnn_class: # Patch keras.layers.RNN
        
        # Configure mock_create_sequences to return valid (but minimal) data
        mock_create_sequences.return_value = (
            np.random.rand(BATCH_SIZE, TSTEPS, N_FEATURES), # X
            np.random.rand(BATCH_SIZE, 1) # Y
        )
        
        # Configure mock_load_model to return a mock model instance
        mock_model_instance = MagicMock()
        # Create a mock layer that satisfies the conditions in retrain.py
        mock_layer = MagicMock()
        mock_layer.__class__ = keras.layers.RNN # Make isinstance(mock_layer, keras.layers.RNN) return True
        mock_layer.stateful = True
        mock_layer.reset_states = MagicMock()
        mock_model_instance.layers = [mock_layer] # Simulate a model with one stateful layer
        mock_model_instance.fit = MagicMock() # Mock fit on the instance
        mock_model_instance.save = MagicMock() # Mock save on the instance
        mock_load_model.return_value = mock_model_instance

        # Configure mock_rnn_class to return our mock_layer
        mock_rnn_class.return_value = mock_layer

        # Call the retrain function
        retrain_model()

        # Assertions
        mock_load_model.assert_called_once_with(MODEL_SAVE_PATH)
        mock_layer.reset_states.assert_called_once() # Assert on the mock layer's reset_states
        mock_model_instance.fit.assert_called_once() # Assert on the instance's mock fit method
        mock_model_instance.save.assert_called_once() # Assert that save was called (path is dynamic)

        # Further checks on fit arguments (optional, but good for robustness)
        args, kwargs = mock_model_instance.fit.call_args
        X_retrain_arg, Y_retrain_arg = args[0], args[1]
        
        assert isinstance(X_retrain_arg, np.ndarray)
        assert isinstance(Y_retrain_arg, np.ndarray)
        assert kwargs['epochs'] == EPOCHS
        assert kwargs['batch_size'] == BATCH_SIZE
        assert kwargs['shuffle'] == False
