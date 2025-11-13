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
from src.data_processing import prepare_keras_input_data
from datetime import datetime # Added import
from src.config import (
    ROWS_AHEAD, TR_SPLIT, N_FEATURES, BATCH_SIZE,
    EPOCHS, LEARNING_RATE, LSTM_UNITS,
    PROCESSED_DATA_DIR, MODEL_REGISTRY_DIR,
    FREQUENCY, TSTEPS, get_training_data_csv_path, get_scaler_params_json_path
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
    training_data_path = get_training_data_csv_path()
    scaler_params_path = get_scaler_params_json_path()

    if not os.path.exists(training_data_path):
        print(f"Error: Training data not found for frequency {FREQUENCY} at {training_data_path}. Skipping training.")
        return None

    df_featured, feature_cols = prepare_keras_input_data(training_data_path)

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
    with open(scaler_params_path, 'w') as f:
        json.dump(scaler_params, f, indent=4)
    print(f"Scaler parameters calculated on training data and saved to {scaler_params_path}")

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
        print(f"Warning: Not enough data to create sequences for training or validation for {FREQUENCY} with TSTEPS={TSTEPS}. Skipping training.")
        return None

    # --- Model Building and Training ---
    model = build_lstm_model(
        input_shape=(TSTEPS, N_FEATURES),
        lstm_units=lstm_units,
        batch_size=current_batch_size,
        learning_rate=learning_rate
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print(f"Starting model training for {FREQUENCY} with TSTEPS={TSTEPS}...")
    history = model.fit(X_train, Y_train,
              epochs=epochs,
              batch_size=current_batch_size,
              validation_data=(X_val, Y_val),
              callbacks=[early_stopping],
              shuffle=False,
              verbose=1)
    print(f"Model training finished for {FREQUENCY} with TSTEPS={TSTEPS}.")

    final_val_loss = min(history.history['val_loss'])
    
    # --- Model Saving ---
    os.makedirs(MODEL_REGISTRY_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_REGISTRY_DIR, f"my_lstm_model_{FREQUENCY}_tsteps{TSTEPS}_{timestamp}.keras")
    model.save(model_path)
    print(f"Model saved to {model_path} with validation loss {final_val_loss:.4f}")

    return final_val_loss, model_path

if __name__ == "__main__":
    # Ensure data/processed exists
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    print(f"\n--- Training model for frequency: {FREQUENCY}, TSTEPS: {TSTEPS} ---")
    
    final_loss_model_path = train_model(
        lstm_units=LSTM_UNITS,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        current_batch_size=BATCH_SIZE
    )
    
    if final_loss_model_path is not None:
        final_loss, model_path = final_loss_model_path
        model_filename = os.path.basename(model_path)

        # Load best hyperparameters if available
        best_hps_overall = {}
        best_hps_path = 'best_hyperparameters.json'
        if os.path.exists(best_hps_path):
            with open(best_hps_path, 'r') as f:
                best_hps_overall = json.load(f)
        else:
            best_hps_overall = {}

        if FREQUENCY not in best_hps_overall:
            best_hps_overall[FREQUENCY] = {}

        if str(TSTEPS) not in best_hps_overall[FREQUENCY] or \
           final_loss < best_hps_overall[FREQUENCY][str(TSTEPS)].get('validation_loss', float('inf')):
            
            best_hps_overall[FREQUENCY][str(TSTEPS)] = {
                'validation_loss': final_loss,
                'model_filename': model_filename,
                'lstm_units': LSTM_UNITS,
                'learning_rate': LEARNING_RATE,
                'epochs': EPOCHS,
                'batch_size': BATCH_SIZE
            }
            print(f"Updated best hyperparameters for Frequency: {FREQUENCY}, TSTEPS: {TSTEPS} with validation loss {final_loss:.4f}")
        
        # Save the updated best_hyperparameters.json
        with open(best_hps_path, 'w') as f:
            json.dump(best_hps_overall, f, indent=4)
        print(f"\nUpdated best hyperparameters saved to {best_hps_path}")

        print("\n--- Training Summary ---")
        print(f"Frequency: {FREQUENCY}, TSTEPS: {TSTEPS}, Validation Loss: {final_loss:.4f}, Model: {model_filename}")
    else:
        print("Model training was not successful.")
