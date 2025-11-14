import os

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
from tensorflow.keras.callbacks import EarlyStopping # Added import
from datetime import datetime

from src.model import build_lstm_model
from src.data_utils import (
    fit_standard_scaler,
    apply_standard_scaler,
    get_effective_data_length,
    create_sequences_for_stateful_lstm,
)
from src.data_processing import convert_minute_to_hourly, prepare_keras_input_data
from src.config import (
    TSTEPS, ROWS_AHEAD, TR_SPLIT, N_FEATURES, BATCH_SIZE,
    EPOCHS, LEARNING_RATE, LSTM_UNITS,
    PROCESSED_DATA_DIR, HOURLY_DATA_CSV, TRAINING_DATA_CSV, SCALER_PARAMS_JSON, MODEL_REGISTRY_DIR, get_latest_model_path, get_active_model_path
)

ACTIVE_MODEL_PATH_FILE = os.path.join("models", "active_model.txt")

def retrain_model(lstm_units=LSTM_UNITS, learning_rate=LEARNING_RATE, epochs=EPOCHS, current_batch_size=BATCH_SIZE, train_window_size=None, validation_window_size=None):
    """
    Loads the existing trained model, loads the latest processed data,
    retrains the model with the new data using a rolling window strategy,
    evaluates its performance, and saves the newly retrained model.
    """
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

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
    lstm_units = best_hps.get('lstm_units', LSTM_UNITS)
    learning_rate = best_hps.get('learning_rate', LEARNING_RATE)
    current_batch_size = best_hps.get('batch_size', BATCH_SIZE)
    epochs = EPOCHS # Epochs are not tuned, use config default

    # --- Data Preparation ---
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)

    # Get featured data (without normalization)
    df_featured, feature_cols = prepare_keras_input_data(HOURLY_DATA_CSV)
    
    if train_window_size is None or validation_window_size is None:
        raise ValueError("train_window_size and validation_window_size must be provided for rolling window retraining.")

    # Ensure enough data for at least one full training and validation window
    if len(df_featured) < train_window_size + validation_window_size:
        print(f"Warning: Not enough data ({len(df_featured)} rows) for the specified train ({train_window_size}) and validation ({validation_window_size}) window sizes. Skipping retraining.")
        return None

    best_val_loss = float('inf')
    best_model_path = None

    # Load the base model for retraining once
    latest_model_path = get_latest_model_path()
    if latest_model_path is None:
        print(f"Error: No trained models found in {MODEL_REGISTRY_DIR}. Please train a model first using train.py.")
        return None
    print(f"Loading base model for retraining: {latest_model_path}")
    base_model = keras.models.load_model(latest_model_path)

    # Rolling window loop
    for i in range(len(df_featured) - train_window_size - validation_window_size + 1):
        print(f"\n--- Retraining Window {i+1} ---")
        
        # Create a fresh model instance for each window to avoid state issues,
        # or load weights into a new model. For simplicity, we'll build and load weights.
        # Alternatively, could clone the base_model and reset states.
        model = build_lstm_model(
            input_shape=(TSTEPS, N_FEATURES),
            lstm_units=lstm_units,
            batch_size=current_batch_size,
            learning_rate=learning_rate
        )
        model.set_weights(base_model.get_weights()) # Start with weights from the latest trained model

        # Define current training and validation windows
        df_train_raw = df_featured.iloc[i : i + train_window_size].copy()
        df_val_raw = df_featured.iloc[i + train_window_size : i + train_window_size + validation_window_size].copy()

        # Fit scaler on current training window and normalize train/val
        mean_vals, std_vals, scaler_params = fit_standard_scaler(df_train_raw, feature_cols)
        df_train_normalized = apply_standard_scaler(df_train_raw, feature_cols, scaler_params)
        df_val_normalized = apply_standard_scaler(df_val_raw, feature_cols, scaler_params)

        # Save scaler parameters (overwriting for each window, or save versioned if needed)
        with open(SCALER_PARAMS_JSON, 'w') as f:
            json.dump(scaler_params, f, indent=4)
        print(f"Scaler parameters saved to {SCALER_PARAMS_JSON}")

        # Create X and Y arrays for training and validation using normalized data
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
            print(f"Warning: Not enough data to create sequences for training or validation in window {i+1}. Skipping this window.")
            continue

        # Define Early Stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        # Retrain the model
        print("Starting model retraining...")
        history = model.fit(X_train, Y_train,
                  epochs=epochs,
                  batch_size=current_batch_size,
                  validation_data=(X_val, Y_val),
                  callbacks=[early_stopping],
                  shuffle=False, # Shuffle is False for time series data to preserve temporal order
                  verbose=0) # Suppress verbose output for each epoch
        print("Model retraining finished.")

        current_val_loss = history.history['val_loss'][-1]
        print(f"Validation Loss for window {i+1}: {current_val_loss:.4f}")

        # Save the best model
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            os.makedirs(MODEL_REGISTRY_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            best_model_path = os.path.join(MODEL_REGISTRY_DIR, f"my_lstm_model_retrained_best_{timestamp}.keras")
            model.save(best_model_path)
            print(f"New best retrained model saved with validation loss {best_val_loss:.4f} to {best_model_path}")

    if best_model_path:
        print(f"\nOverall best retrained model saved at: {best_model_path} with validation loss: {best_val_loss:.4f}")
        return best_val_loss
    else:
        print("\nNo models were retrained successfully.")
        return None

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

    parser = argparse.ArgumentParser(description="Retrain the LSTM model with specified hyperparameters.")
    parser.add_argument('--lstm_units', type=int, default=best_hps.get('lstm_units', LSTM_UNITS),
                        help=f"Number of LSTM units in the layer (default: {best_hps.get('lstm_units', LSTM_UNITS)})")
    parser.add_argument('--learning_rate', type=float, default=best_hps.get('learning_rate', LEARNING_RATE),
                        help=f"Learning rate for the optimizer (default: {best_hps.get('learning_rate', LEARNING_RATE)})")
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f"Number of training epochs (default: {EPOCHS})")
    parser.add_argument('--batch_size', type=int, default=best_hps.get('batch_size', BATCH_SIZE),
                        help=f"Batch size for training (default: {best_hps.get('batch_size', BATCH_SIZE)})")
    parser.add_argument('--train_window_size', type=int, default=2000, # Example default, will need tuning
                        help="Number of hourly data points in the training window.")
    parser.add_argument('--validation_window_size', type=int, default=500, # Example default, will need tuning
                        help="Number of hourly data points in the validation window.")
    
    args = parser.parse_args()

    # Call retrain_model with parsed arguments
    final_loss = retrain_model(
        lstm_units=args.lstm_units,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        current_batch_size=args.batch_size,
        train_window_size=args.train_window_size,
        validation_window_size=args.validation_window_size
    )
    if final_loss is not None:
        print(f"Overall Best Retraining Validation Loss: {final_loss:.4f}")