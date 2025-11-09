import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import os
from tensorflow.keras.callbacks import EarlyStopping # Added import

from src.model import build_lstm_model
from src.data_processing import convert_minute_to_hourly, prepare_keras_input_data
from datetime import datetime # Added import
from src.config import (
    TSTEPS, ROWS_AHEAD, TR_SPLIT, N_FEATURES, BATCH_SIZE,
    EPOCHS, LEARNING_RATE, LSTM_UNITS,
    PROCESSED_DATA_DIR, TRAINING_DATA_CSV, SCALER_PARAMS_JSON, MODEL_SAVE_PATH, MODEL_REGISTRY_DIR
)

def create_sequences(data, sequence_length, rows_ahead):
    """
    Manually creates X and Y sequences for a stateless LSTM.

    Args:
        data (pd.DataFrame): The input DataFrame with all normalized features.
        sequence_length (int): The length of the input sequences (tsteps).
        rows_ahead (int): How many rows ahead the label should be.

    Returns:
        tuple: (X_sequences, Y_labels) as numpy arrays.
    """
    # Use all columns except 'Time' as features
    feature_cols = [col for col in data.columns if col != 'Time']
    features_array = data[feature_cols].values
    labels_array = data['Open'].values

    shifted_labels = np.full_like(labels_array, np.nan, dtype=float)
    if rows_ahead < len(labels_array):
        shifted_labels[:-rows_ahead] = labels_array[rows_ahead:]
    
    valid_indices = ~np.isnan(shifted_labels)
    features_array = features_array[valid_indices]
    shifted_labels = shifted_labels[valid_indices]

    if len(features_array) < sequence_length:
        print(f"Warning: Not enough data ({len(features_array)} rows) to create sequences of length {sequence_length}. Returning empty arrays.")
        return np.array([]), np.array([])

    X, Y = [], []
    # Iterate to create sequences
    for i in range(len(features_array) - sequence_length + 1):
        X.append(features_array[i : i + sequence_length])
        Y.append(shifted_labels[i + sequence_length - 1]) # Target for the end of the sequence

    X = np.array(X)
    Y = np.array(Y)

    # Reshape Y to be (num_samples, 1) if it's not already
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    return X, Y


def train_model(lstm_units=LSTM_UNITS, learning_rate=LEARNING_RATE, epochs=EPOCHS, current_batch_size=BATCH_SIZE):
    """
    Loads processed data, builds and trains the LSTM model, and saves the trained model.
    """
    # Load processed data
    df_processed = pd.read_csv(TRAINING_DATA_CSV)
    
    # Split data into training and validation sets
    train_size = int(len(df_processed) * TR_SPLIT)
    df_train = df_processed.iloc[:train_size].copy()
    df_val = df_processed.iloc[train_size:].copy()

    # Create X and Y arrays for training and validation
    X_train, Y_train = create_sequences(df_train, TSTEPS, ROWS_AHEAD)
    X_val, Y_val = create_sequences(df_val, TSTEPS, ROWS_AHEAD)
    
    # Build the model
    model = build_lstm_model(
        input_shape=(TSTEPS, N_FEATURES),
        lstm_units=lstm_units,
        batch_size=current_batch_size,
        learning_rate=learning_rate
    )

    # Define Early Stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    print("Starting model training...")
    history = model.fit(X_train, Y_train,
              epochs=epochs,
              batch_size=current_batch_size,
              validation_data=(X_val, Y_val),
              callbacks=[early_stopping],
              shuffle=True) # Shuffle is True for stateless models
    print("Model training finished.")

    # Save the trained model to a versioned path
    os.makedirs(MODEL_REGISTRY_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_model_path = os.path.join(MODEL_REGISTRY_DIR, f"my_lstm_model_{timestamp}.keras")
    model.save(versioned_model_path)
    print(f"Trained model saved to {versioned_model_path}")

    # Return the final training loss
    if history and history.history:
        return history.history['loss'][-1]
    return None

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the LSTM model with specified hyperparameters.")
    parser.add_argument('--lstm_units', type=int, default=LSTM_UNITS,
                        help=f"Number of LSTM units in the layer (default: {LSTM_UNITS})")
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help=f"Learning rate for the optimizer (default: {LEARNING_RATE})")
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f"Number of training epochs (default: {EPOCHS})")
    
    args = parser.parse_args()

    # Ensure data/processed exists and contains necessary files for testing
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
    
    # Create dummy data if not present (for standalone testing)
    if not os.path.exists(TRAINING_DATA_CSV) or not os.path.exists(SCALER_PARAMS_JSON):
        print("Creating dummy processed data for standalone train.py test...")
        
        # Generate a continuous range of minute-level datetimes
        start_time = pd.to_datetime('2023-01-01T00:00')
        num_minutes = 120000 # Increased for more data
        dummy_datetimes = pd.date_range(start=start_time, periods=num_minutes, freq='min')

        dummy_minute_data = {
            'DateTime': dummy_datetimes,
            'Open': np.random.rand(num_minutes) * 100 + 100,
            'High': np.random.rand(num_minutes) * 100 + 101,
            'Low': np.random.rand(num_minutes) * 100 + 99,
            'Close': np.random.rand(num_minutes) * 100 + 100
        }
        dummy_df_minute = pd.DataFrame(dummy_minute_data)
        dummy_input_minute_path = "data/raw/nvda_minute.csv"
        os.makedirs("data/raw", exist_ok=True)
        dummy_df_minute.to_csv(dummy_input_minute_path, index=False)

        hourly_output_path = os.path.join(PROCESSED_DATA_DIR, "nvda_hourly.csv")
        convert_minute_to_hourly(dummy_input_minute_path, hourly_output_path)

        training_output_path = TRAINING_DATA_CSV
        scaler_params_path = SCALER_PARAMS_JSON
        prepare_keras_input_data(hourly_output_path, training_output_path, scaler_params_path)
        print("Dummy processed data created.")

    # Call train_model with parsed arguments
    final_loss = train_model(
        lstm_units=args.lstm_units,
        learning_rate=args.learning_rate,
        epochs=args.epochs
    )
    if final_loss is not None:
        print(f"Final Training Loss: {final_loss:.4f}")
