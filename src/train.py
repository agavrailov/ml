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
    PROCESSED_DATA_DIR, HOURLY_DATA_CSV, TRAINING_DATA_CSV, SCALER_PARAMS_JSON, MODEL_SAVE_PATH, MODEL_REGISTRY_DIR
)

def get_effective_data_length(data, sequence_length, rows_ahead):
    """
    Calculates the effective number of data points available for sequence creation
    after considering sequence_length and rows_ahead.
    """
    # Use all columns except 'Time' as features
    feature_cols = [col for col in data.columns if col != 'Time']
    features_array = data[feature_cols].values
    labels_array = data['Open'].values

    shifted_labels = np.full_like(labels_array, np.nan, dtype=float)
    if rows_ahead < len(labels_array):
        shifted_labels[:-rows_ahead] = labels_array[rows_ahead:]
    
    valid_indices = ~np.isnan(shifted_labels)
    
    # The number of valid feature rows after considering shifted labels
    num_valid_feature_rows = np.sum(valid_indices)

    # The number of sequences that can be formed
    if num_valid_feature_rows < sequence_length:
        return 0
    return num_valid_feature_rows - sequence_length + 1


def create_sequences_for_stateless_lstm(data, sequence_length, rows_ahead):
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


def create_sequences_for_stateful_lstm(data, sequence_length, batch_size, rows_ahead):
    """
    Creates X and Y sequences for a stateful LSTM, ensuring that the number of samples
    is divisible by the batch_size and maintaining temporal order.

    Args:
        data (pd.DataFrame): The input DataFrame with all normalized features.
        sequence_length (int): The length of the input sequences (tsteps).
        batch_size (int): The batch size for the stateful LSTM.
        rows_ahead (int): How many rows ahead the label should be.

    Returns:
        tuple: (X_sequences, Y_labels) as numpy arrays.
    """
    feature_cols = [col for col in data.columns if col != 'Time']
    features_array = data[feature_cols].values
    labels_array = data['Open'].values

    shifted_labels = np.full_like(labels_array, np.nan, dtype=float)
    if rows_ahead < len(labels_array):
        shifted_labels[:-rows_ahead] = labels_array[rows_ahead:]
    
    valid_indices = ~np.isnan(shifted_labels)
    features_array = features_array[valid_indices]
    shifted_labels = shifted_labels[valid_indices]

    # Calculate the number of sequences that can be formed
    num_sequences = len(features_array) - sequence_length + 1

    if num_sequences <= 0:
        print(f"Warning: Not enough data ({len(features_array)} rows) to create sequences of length {sequence_length}. Returning empty arrays.")
        return np.array([]), np.array([])

    # Truncate data to be divisible by batch_size
    num_batches = num_sequences // batch_size
    effective_num_sequences = num_batches * batch_size

    if effective_num_sequences == 0:
        print(f"Warning: Not enough effective sequences ({num_sequences}) for batch_size {batch_size}. Returning empty arrays.")
        return np.array([]), np.array([])

    X, Y = [], []
    for i in range(effective_num_sequences):
        X.append(features_array[i : i + sequence_length])
        Y.append(shifted_labels[i + sequence_length - 1])

    X = np.array(X)
    Y = np.array(Y)

    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    return X, Y


def train_model(lstm_units=LSTM_UNITS, learning_rate=LEARNING_RATE, epochs=EPOCHS, current_batch_size=BATCH_SIZE):
    """
    Loads processed data, builds and trains the LSTM model, and saves the trained model.
    Uses a standard train-validation split.
    """
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # --- Data Preparation ---
    # Get featured data (without normalization)
    df_featured, feature_cols = prepare_keras_input_data(HOURLY_DATA_CSV)

    # Split data into training and validation sets
    split_index = int(len(df_featured) * TR_SPLIT)
    df_train_raw = df_featured.iloc[:split_index].copy()
    df_val_raw = df_featured.iloc[split_index:].copy()

    # --- Normalization ---
    # Calculate mean and standard deviation for normalization ONLY on training data
    mean_vals = df_train_raw[feature_cols].mean()
    std_vals = df_train_raw[feature_cols].std()

    # Handle cases where std_vals might be zero to avoid division by zero
    std_vals = std_vals.replace(0, 1)

    # Save the single, correct set of scaler parameters
    scaler_params = {
        'mean': mean_vals.to_dict(),
        'std': std_vals.to_dict()
    }
    with open(SCALER_PARAMS_JSON, 'w') as f:
        json.dump(scaler_params, f, indent=4)
    print(f"Scaler parameters calculated on training data and saved to {SCALER_PARAMS_JSON}")

    # Normalize both training and validation sets using the scaler fitted on training data
    df_train_normalized = df_train_raw.copy()
    df_val_normalized = df_val_raw.copy()

    df_train_normalized[feature_cols] = (df_train_raw[feature_cols] - mean_vals) / std_vals
    df_val_normalized[feature_cols] = (df_val_raw[feature_cols] - mean_vals) / std_vals
    
    # --- Sequence Creation ---
    # Truncate data to be divisible by the current_batch_size for stateful LSTM
    max_sequences_train = get_effective_data_length(df_train_normalized, TSTEPS, ROWS_AHEAD)
    remainder_train = max_sequences_train % current_batch_size
    if remainder_train > 0:
        df_train_normalized = df_train_normalized.iloc[:-remainder_train]

    max_sequences_val = get_effective_data_length(df_val_normalized, TSTEPS, ROWS_AHEAD)
    remainder_val = max_sequences_val % current_batch_size
    if remainder_val > 0:
        df_val_normalized = df_val_normalized.iloc[:-remainder_val]

    X_train, Y_train = create_sequences_for_stateful_lstm(df_train_normalized, TSTEPS, current_batch_size, ROWS_AHEAD)
    X_val, Y_val = create_sequences_for_stateful_lstm(df_val_normalized, TSTEPS, current_batch_size, ROWS_AHEAD)

    if X_train.shape[0] == 0 or X_val.shape[0] == 0:
        print(f"Warning: Not enough data to create sequences for training or validation. Skipping training.")
        return None

    # --- Model Building and Training ---
    model = build_lstm_model(
        input_shape=(TSTEPS, N_FEATURES),
        lstm_units=lstm_units,
        batch_size=current_batch_size,
        learning_rate=learning_rate
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print("Starting model training...")
    history = model.fit(X_train, Y_train,
              epochs=epochs,
              batch_size=current_batch_size,
              validation_data=(X_val, Y_val),
              callbacks=[early_stopping],
              shuffle=False,
              verbose=1)
    print("Model training finished.")

    final_val_loss = min(history.history['val_loss'])
    
    # --- Model Saving ---
    os.makedirs(MODEL_REGISTRY_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_REGISTRY_DIR, f"my_lstm_model_best_{timestamp}.keras")
    model.save(model_path)
    print(f"Model saved to {model_path} with validation loss {final_val_loss:.4f}")

    return final_val_loss

if __name__ == "__main__":
    import argparse

    # Load best hyperparameters if available
    best_hps = {}
    best_hps_path = 'best_hyperparameters.json'
    if os.path.exists(best_hps_path):
        with open(best_hps_path, 'r') as f:
            best_hps = json.load(f)
        print(f"Loaded best hyperparameters from {best_hps_path}")
    else:
        print("No 'best_hyperparameters.json' found. Using default hyperparameters from config.py.")

    parser = argparse.ArgumentParser(description="Train the LSTM model with specified hyperparameters.")
    parser.add_argument('--lstm_units', type=int, default=best_hps.get('lstm_units', LSTM_UNITS),
                        help=f"Number of LSTM units in the layer (default: {best_hps.get('lstm_units', LSTM_UNITS)})")
    parser.add_argument('--learning_rate', type=float, default=best_hps.get('learning_rate', LEARNING_RATE),
                        help=f"Learning rate for the optimizer (default: {best_hps.get('learning_rate', LEARNING_RATE)})")
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f"Number of training epochs (default: {EPOCHS})")
    parser.add_argument('--batch_size', type=int, default=best_hps.get('batch_size', BATCH_SIZE),
                        help=f"Batch size for training (default: {best_hps.get('batch_size', BATCH_SIZE)})")
    
    args = parser.parse_args()

    # Ensure data/processed exists and contains necessary files for testing
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
    
    # Create dummy data if not present (for standalone testing)
    if not os.path.exists(HOURLY_DATA_CSV):
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
        print("Dummy hourly data created.")

    # Call train_model with parsed arguments
    final_loss = train_model(
        lstm_units=args.lstm_units,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        current_batch_size=args.batch_size # Pass batch_size from args
    )
    if final_loss is not None:
        print(f"Final Validation Loss: {final_loss:.4f}")
