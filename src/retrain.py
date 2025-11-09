import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json

from src.model import build_lstm_model # Although we load, we might need this for reference
from src.train import create_sequences_for_stateful_lstm # Reusing data preparation logic
from datetime import datetime # Added import
from src.config import (
    TSTEPS, ROWS_AHEAD, TR_SPLIT, N_FEATURES, BATCH_SIZE,
    EPOCHS, LEARNING_RATE, LSTM_UNITS,
    PROCESSED_DATA_DIR, TRAINING_DATA_CSV, SCALER_PARAMS_JSON, MODEL_SAVE_PATH, MODEL_REGISTRY_DIR, get_latest_model_path
)

def retrain_model():
    """
    Loads the existing trained model, loads the latest processed data,
    and retrains the model with the new data.
    """
    # 1. Load the existing trained model
    latest_model_path = get_latest_model_path()
    if latest_model_path is None:
        print(f"Error: No trained models found in the registry at {MODEL_REGISTRY_DIR}. Please train a model first.")
        return

    try:
        model = keras.models.load_model(latest_model_path)
        print(f"Existing model loaded from {latest_model_path}")
    except Exception as e:
        print(f"Error loading model from {latest_model_path}: {e}")
        print("Attempting to build a new model and load weights (if compatible).")
        # Fallback: build model and load weights if direct load fails
        model = build_lstm_model(
            input_shape=(TSTEPS, N_FEATURES),
            lstm_units=LSTM_UNITS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE
        )
        model.load_weights(latest_model_path)
        print("Model built and weights loaded.")


    # 2. Load the latest processed data
    if not os.path.exists(TRAINING_DATA_CSV) or not os.path.exists(SCALER_PARAMS_JSON):
        print(f"Error: Processed data not found. Please run data processing first.")
        return

    df_processed = pd.read_csv(TRAINING_DATA_CSV)
    
    # Load scaler parameters (not directly used for retraining, but good practice)
    with open(SCALER_PARAMS_JSON, 'r') as f:
        scaler_params = json.load(f)

    # 3. Prepare data for retraining
    # For retraining, we typically use the entire available processed dataset
    # or a new batch of data. For simplicity, we'll use the whole dataset here.
    # In a real-world scenario, you might only retrain on new data or a sliding window.
    X_retrain, Y_retrain = create_sequences_for_stateful_lstm(df_processed, TSTEPS, BATCH_SIZE, ROWS_AHEAD)

    if len(X_retrain) == 0:
        print("No data available for retraining after sequence creation. Exiting.")
        return

    # 4. Retrain the model
    print("Starting model retraining...")
    # Reset states before retraining if the model is stateful and we are training on new data
    # or a full pass.
    # Reset states of stateful LSTM layers before retraining
    for layer in model.layers:
        if isinstance(layer, keras.layers.RNN) and layer.stateful:
            layer.reset_states()
    
    model.fit(X_retrain, Y_retrain,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              shuffle=False) # Stateful LSTMs typically require shuffle=False
    print("Model retraining finished.")

    # 5. Evaluate performance (Placeholder for Task 3.3/3.4)
    # In a real scenario, you would evaluate the model on a separate test set
    # and decide whether to save it based on performance metrics.
    print("Placeholder: Model evaluation would happen here.")

    # 6. Save the retrained model to a versioned path
    os.makedirs(MODEL_REGISTRY_DIR, exist_ok=True) # Ensure registry directory exists
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_model_path = os.path.join(MODEL_REGISTRY_DIR, f"my_lstm_model_{timestamp}.keras")
    model.save(versioned_model_path)
    print(f"Retrained model saved to {versioned_model_path}")

if __name__ == "__main__":
    print("Running model retraining...")
    retrain_model()
