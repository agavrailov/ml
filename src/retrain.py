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
    PROCESSED_DATA_DIR, TRAINING_DATA_CSV, SCALER_PARAMS_JSON, MODEL_REGISTRY_DIR, get_latest_model_path, get_active_model_path
)

ACTIVE_MODEL_PATH_FILE = os.path.join("models", "active_model.txt")

def retrain_model():
    """
    Loads the existing trained model, loads the latest processed data,
    retrains the model with the new data, evaluates its performance,
    and saves the newly retrained model.
    """
    # --- Data Preparation ---
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)

    min_required_rows = int((BATCH_SIZE + TSTEPS + ROWS_AHEAD) / (1 - TR_SPLIT)) + 100
    
    df_processed_check = None
    if os.path.exists(TRAINING_DATA_CSV):
        df_processed_check = pd.read_csv(TRAINING_DATA_CSV)

    if df_processed_check is None or len(df_processed_check) < min_required_rows or not os.path.exists(SCALER_PARAMS_JSON):
        print("Creating/updating dummy processed data for retraining...")
        # (Dummy data generation logic remains the same)
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
        hourly_output_path = os.path.join(PROCESSED_DATA_DIR, "nvda_hourly.csv")
        convert_minute_to_hourly(dummy_input_minute_path, hourly_output_path)
        prepare_keras_input_data(hourly_output_path, TRAINING_DATA_CSV, SCALER_PARAMS_JSON)
        print("Dummy processed data created/updated.")

    # --- Model Loading ---
    latest_model_path = get_latest_model_path()
    if latest_model_path is None:
        print(f"Error: No trained models found in {MODEL_REGISTRY_DIR}. Please train a model first using train.py.")
        return

    print(f"Loading model for retraining: {latest_model_path}")
    model = keras.models.load_model(latest_model_path)

    # --- Data Loading and Preparation for Retraining ---
    df_processed = pd.read_csv(TRAINING_DATA_CSV)
    
    # For retraining, we use all available data. The split is conceptual for validation.
    train_size = int(len(df_processed) * TR_SPLIT)
    df_train = df_processed.iloc[:train_size].copy()
    df_val = df_processed.iloc[train_size:].copy()

    # Create sequences for training and validation
    X_train, Y_train = create_sequences_for_stateful_lstm(df_train, TSTEPS, BATCH_SIZE, ROWS_AHEAD)
    X_val, Y_val = create_sequences_for_stateful_lstm(df_val, TSTEPS, BATCH_SIZE, ROWS_AHEAD)

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
                  epochs=EPOCHS,
                  batch_size=BATCH_SIZE,
                  validation_data=(X_val, Y_val),
                  callbacks=[early_stopping], # Add early stopping
                  shuffle=False)
    else:
        history = model.fit(X_train, Y_train,
                  epochs=EPOCHS,
                  batch_size=BATCH_SIZE,
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