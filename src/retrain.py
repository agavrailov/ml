import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
from tensorflow.keras.callbacks import EarlyStopping # Added import
from datetime import datetime

from src.model import build_lstm_model
from src.train import create_sequences_for_stateful_lstm, get_effective_data_length
from src.data_processing import convert_minute_to_hourly, prepare_keras_input_data
from src.config import (
    TSTEPS, ROWS_AHEAD, TR_SPLIT, N_FEATURES, BATCH_SIZE,
    EPOCHS, LEARNING_RATE, LSTM_UNITS,
    PROCESSED_DATA_DIR, HOURLY_DATA_CSV, TRAINING_DATA_CSV, SCALER_PARAMS_JSON, MODEL_REGISTRY_DIR, get_latest_model_path, get_active_model_path
)

ACTIVE_MODEL_PATH_FILE = os.path.join("models", "active_model.txt")

def retrain_model():
    """
    Loads the existing trained model, loads the latest processed data,
    retrains the model with the new data, evaluates its performance,
    and saves the newly retrained model.
    """
    # Load best hyperparameters if available
    best_hps = {}
    best_hps_path = 'best_hyperparameters.json'
    if os.path.exists(best_hps_path):
        with open(best_hps_path, 'r') as f:
            best_hps = json.load(f)
        print(f"Loaded best hyperparameters from {best_hps_path}")
    else:
        print("No 'best_hyperparameters.json' found. Using default hyperparameters from config.py.")

    # Use loaded hyperparameters or fall back to config defaults
    current_lstm_units = best_hps.get('lstm_units', LSTM_UNITS)
    current_learning_rate = best_hps.get('learning_rate', LEARNING_RATE)
    current_batch_size = best_hps.get('batch_size', BATCH_SIZE)
    current_epochs = EPOCHS # Epochs are not tuned, use config default

    # --- Data Preparation ---
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)

    # Ensure HOURLY_DATA_CSV exists for processing
    if not os.path.exists(HOURLY_DATA_CSV):
        print("Creating/updating dummy hourly data for retraining...")
        start_time = pd.to_datetime('2023-01-01T00:00')
        num_minutes = 120000
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
        convert_minute_to_hourly(dummy_input_minute_path, HOURLY_DATA_CSV)
        print("Dummy hourly data created/updated.")

    # Get featured data (without normalization)
    df_featured, feature_cols = prepare_keras_input_data(HOURLY_DATA_CSV)
    
    # Split data into raw training and validation sets
    train_size = int(len(df_featured) * TR_SPLIT)
    df_train_raw = df_featured.iloc[:train_size].copy()
    df_val_raw = df_featured.iloc[train_size:].copy()

    # Calculate mean and standard deviation for normalization ONLY on training data
    mean_vals = df_train_raw[feature_cols].mean()
    std_vals = df_train_raw[feature_cols].std()

    # Handle cases where std_vals might be zero to avoid division by zero
    std_vals = std_vals.replace(0, 1) # Replace 0 with 1 to prevent division by zero

    # Normalize both training and validation sets using the scaler fitted on training data
    df_train_normalized = df_train_raw.copy()
    df_val_normalized = df_val_raw.copy()

    df_train_normalized[feature_cols] = (df_train_raw[feature_cols] - mean_vals) / std_vals
    df_val_normalized[feature_cols] = (df_val_raw[feature_cols] - mean_vals) / std_vals

    # Save scaler parameters
    scaler_params = {
        'mean': mean_vals.to_dict(),
        'std': std_vals.to_dict()
    }
    with open(SCALER_PARAMS_JSON, 'w') as f:
        json.dump(scaler_params, f, indent=4)
    print(f"Scaler parameters saved to {SCALER_PARAMS_JSON}")

    # --- Model Loading ---
    latest_model_path = get_latest_model_path()
    if latest_model_path is None:
        print(f"Error: No trained models found in {MODEL_REGISTRY_DIR}. Please train a model first using train.py.")
        return

    print(f"Loading model for retraining: {latest_model_path}")
    # Load the model with custom_objects if needed, though for simple LSTM it might not be
    model = keras.models.load_model(latest_model_path)

    # Create sequences for training and validation using normalized data
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

    if len(X_train) == 0:
        print("Not enough training data to create sequences. Aborting retraining.")
        return

    # Define Early Stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # --- Model Retraining ---
    print("Starting model retraining...")
    history = None
    if len(X_val) > 0:
        history = model.fit(X_train, Y_train,
                  epochs=current_epochs,
                  batch_size=current_batch_size,
                  validation_data=(X_val, Y_val),
                  callbacks=[early_stopping], # Add early stopping
                  shuffle=False)
    else:
        history = model.fit(X_train, Y_train,
                  epochs=current_epochs,
                  batch_size=current_batch_size,
                  callbacks=[early_stopping], # Add early stopping
                  shuffle=False)
    print("Model retraining finished.")

    # --- Model Saving ---
    os.makedirs(MODEL_REGISTRY_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_model_path = os.path.join(MODEL_REGISTRY_DIR, f"my_lstm_model_{timestamp}.keras")
    model.save(versioned_model_path)
    print(f"Retrained model saved to {versioned_model_path}")

    final_loss = history.history['loss'][-1] if history and history.history else None
    if final_loss is not None:
        print(f"Final Training Loss: {final_loss:.4f}")

if __name__ == "__main__":
    retrain_model()