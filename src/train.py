import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import os

from src.model import build_lstm_model
from src.data_processing import convert_minute_to_hourly, prepare_keras_input_data
from src.config import (
    TSTEPS, ROWS_AHEAD, TR_SPLIT, N_FEATURES, BATCH_SIZE,
    EPOCHS, LEARNING_RATE, LSTM_UNITS,
    PROCESSED_DATA_DIR, TRAINING_DATA_CSV, SCALER_PARAMS_JSON, MODEL_SAVE_PATH
)

def get_effective_data_length(data, sequence_length, rows_ahead):
    labels_array = data['Open'].values

    shifted_labels = np.full_like(labels_array, np.nan, dtype=float)
    if rows_ahead < len(labels_array):
        shifted_labels[:-rows_ahead] = labels_array[rows_ahead:]
    
    valid_indices = ~np.isnan(shifted_labels)
    effective_data_len = np.sum(valid_indices)

    # Number of sequences that can be formed from this effective data length
    if effective_data_len < sequence_length:
        return 0
    return effective_data_len - sequence_length + 1


def create_sequences_for_stateful_lstm(data, sequence_length, batch_size, rows_ahead):
    """
    Manually creates X and Y sequences for a stateful LSTM.

    Args:
        data (pd.DataFrame): The input DataFrame (normalized OHLC).
        sequence_length (int): The length of the input sequences (tsteps).
        batch_size (int): The batch size.
        rows_ahead (int): How many rows ahead the label should be.

    Returns:
        tuple: (X_sequences, Y_labels) as numpy arrays.
    """
    features_array = data[['Open', 'High', 'Low', 'Close']].values
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

    # Ensure X and Y are divisible by batch_size
    num_samples = len(X)
    remainder = num_samples % batch_size
    if remainder > 0:
        X = X[:-remainder]
        Y = Y[:-remainder]
    
    if len(X) == 0:
        print(f"Warning: After batching, no samples left. Returning empty arrays.")
        return np.array([]), np.array([])

    # Reshape Y to be (num_samples, 1) if it's not already
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    return X, Y


def train_model(current_batch_size=BATCH_SIZE): # Added current_batch_size parameter
    """
    Loads processed data, builds and trains the LSTM model, and saves the trained model.
    """
    # Load processed data
    df_processed = pd.read_csv(TRAINING_DATA_CSV)
    
    # Load scaler parameters for denormalization (needed for prediction)
    with open(SCALER_PARAMS_JSON, 'r') as f:
        scaler_params = json.load(f)

    # Split data into training and validation sets
    train_size = int(len(df_processed) * TR_SPLIT)
    df_train = df_processed.iloc[:train_size].copy()
    df_val = df_processed.iloc[train_size:].copy()

    # --- Truncation logic to ensure number of sequences is divisible by current_batch_size ---
    
    # For training data
    max_sequences_train = get_effective_data_length(df_train, TSTEPS, ROWS_AHEAD)
    if max_sequences_train < current_batch_size:
        print(f"Warning: Training data ({max_sequences_train} sequences) too short for batch_size={current_batch_size}. Skipping training.")
        return
    
    remainder_sequences_train = max_sequences_train % current_batch_size
    if remainder_sequences_train > 0:
        df_train = df_train.iloc[:-remainder_sequences_train]
        max_sequences_train = get_effective_data_length(df_train, TSTEPS, ROWS_AHEAD)
        if max_sequences_train < current_batch_size: # Check again if it became too short
            print(f"Warning: Training data became too short after truncation. Skipping training.")
            return

    # For validation data
    max_sequences_val = get_effective_data_length(df_val, TSTEPS, ROWS_AHEAD)
    X_val, Y_val = np.array([]), np.array([])
    if max_sequences_val >= current_batch_size:
        remainder_sequences_val = max_sequences_val % current_batch_size
        if remainder_sequences_val > 0:
            df_val = df_val.iloc[:-remainder_sequences_val]
            max_sequences_val = get_effective_data_length(df_val, TSTEPS, ROWS_AHEAD)
            if max_sequences_val < current_batch_size: # Check again if it became too short
                print(f"Warning: Validation data became too short after truncation. Skipping validation.")
            else:
                X_val, Y_val = create_sequences_for_stateful_lstm(df_val, TSTEPS, current_batch_size, ROWS_AHEAD)
        else:
            X_val, Y_val = create_sequences_for_stateful_lstm(df_val, TSTEPS, current_batch_size, ROWS_AHEAD)
    else:
        print(f"Warning: Not enough validation sequences ({max_sequences_val}) for batch_size={current_batch_size}. Skipping validation.")
            

    # Create X and Y arrays for training
    X_train, Y_train = create_sequences_for_stateful_lstm(df_train, TSTEPS, current_batch_size, ROWS_AHEAD)
    
    # Build the model
    model = build_lstm_model(
        input_shape=(TSTEPS, N_FEATURES),
        lstm_units=LSTM_UNITS,
        batch_size=current_batch_size, # Use current_batch_size for model building
        learning_rate=LEARNING_RATE
    )

    # Train the model
    print("Starting model training...")
    if len(X_val) > 0:
        model.fit(X_train, Y_train,
                  epochs=EPOCHS,
                  batch_size=current_batch_size,
                  validation_data=(X_val, Y_val),
                  shuffle=False) # Stateful LSTMs typically require shuffle=False
    else:
        model.fit(X_train, Y_train,
                  epochs=EPOCHS,
                  batch_size=current_batch_size,
                  shuffle=False) # Stateful LSTMs typically require shuffle=False
    print("Model training finished.")

    # Save the trained model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True) # Ensure directory exists
    model.save(MODEL_SAVE_PATH)
    print(f"Trained model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    # Ensure data/processed exists and contains necessary files for testing
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
    
    # Create dummy data if not present (for standalone testing)
    if not os.path.exists(TRAINING_DATA_CSV) or not os.path.exists(SCALER_PARAMS_JSON):
        print("Creating dummy processed data for standalone train.py test...")
        
        # Generate a continuous range of minute-level datetimes
        start_time = pd.to_datetime('2023-01-01T00:00')
        num_minutes = 60000 # For 1000 hours of data
        dummy_datetimes = pd.date_range(start=start_time, periods=num_minutes, freq='min')

        dummy_minute_data = {
            'DateTime': dummy_datetimes,
            'Open': np.random.rand(num_minutes) * 100 + 100,
            'High': np.random.rand(num_minutes) * 100 + 101,
            'Low': np.random.rand(num_minutes) * 100 + 99,
            'Close': np.random.rand(num_minutes) * 100 + 100
        }
        dummy_df_minute = pd.DataFrame(dummy_minute_data)
        dummy_input_minute_path = "data/raw/xagusd_minute.csv"
        os.makedirs("data/raw", exist_ok=True)
        dummy_df_minute.to_csv(dummy_input_minute_path, index=False)

        hourly_output_path = os.path.join(PROCESSED_DATA_DIR, "xagusd_hourly.csv")
        convert_minute_to_hourly(dummy_input_minute_path, hourly_output_path)

        training_output_path = TRAINING_DATA_CSV
        scaler_params_path = SCALER_PARAMS_JSON
        prepare_keras_input_data(hourly_output_path, training_output_path, scaler_params_path)
        print("Dummy processed data created.")

    # Call train_model without current_batch_size to use global BATCH_SIZE (500)
    train_model()
