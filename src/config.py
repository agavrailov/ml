# src/config.py

import os
import json # Added import

# --- Model Hyperparameters ---
TSTEPS = 3  # window size a.k.a. time steps
ROWS_AHEAD = 60  # prediction Labels are n rows ahead of the current
TR_SPLIT = 0.7   # part of data used for training
N_FEATURES = 9    # Number of features (OHLC + technical indicators + time features)
BATCH_SIZE = 256 # Number of samples per gradient update. Found by KerasTuner.
EPOCHS = 20
LEARNING_RATE = 0.01
LSTM_UNITS = 96  # number of neurons in a LSTM layer. Found by KerasTuner.

# --- Paths ---
PROCESSED_DATA_DIR = "data/processed"
TRAINING_DATA_CSV = os.path.join(PROCESSED_DATA_DIR, "training_data.csv")
SCALER_PARAMS_JSON = os.path.join(PROCESSED_DATA_DIR, "scaler_params.json")
MODEL_SAVE_PATH = "models/my_lstm_model.keras" # Keras native format
MODEL_REGISTRY_DIR = "models/registry" # Directory to store versioned models
ACTIVE_MODEL_PATH_FILE = os.path.join("models", "active_model.txt") # Added active model pointer file

def get_latest_model_path():
    """Returns the path to the latest saved model in the registry."""
    if not os.path.exists(MODEL_REGISTRY_DIR):
        return None
    
    model_files = [f for f in os.listdir(MODEL_REGISTRY_DIR) if f.endswith('.keras')]
    if not model_files:
        return None
    
    # Assuming models are named with a timestamp, e.g., my_lstm_model_YYYYMMDD_HHMMSS.keras
    latest_model_file = sorted(model_files, reverse=True)[0]
    return os.path.join(MODEL_REGISTRY_DIR, latest_model_file)

def get_active_model_path():
    """
    Returns the path to the currently active model based on active_model.txt.
    Returns None if no active model is designated or file is missing/corrupted.
    """
    if not os.path.exists(ACTIVE_MODEL_PATH_FILE):
        return None
    try:
        with open(ACTIVE_MODEL_PATH_FILE, 'r') as f:
            active_model_info = json.load(f)
            return active_model_info.get("path")
    except json.JSONDecodeError:
        print(f"Warning: {ACTIVE_MODEL_PATH_FILE} is corrupted. No active model found.")
        return None

# --- TWS API Connection Parameters ---
TWS_HOST = os.getenv('TWS_HOST', '127.0.0.1')
TWS_PORT = int(os.getenv('TWS_PORT', '7496')) # Default for TWS, 4001 for Gateway
TWS_CLIENT_ID = int(os.getenv('TWS_CLIENT_ID', '1'))

# --- NVDA Stock Contract Details ---
NVDA_CONTRACT_DETAILS = {
    'symbol': 'NVDA',
    'secType': 'STK',
    'exchange': 'SMART',
    'currency': 'USD',
}

RAW_DATA_CSV = "data/raw/nvda_minute.csv"
HOURLY_DATA_CSV = os.path.join(PROCESSED_DATA_DIR, "nvda_hourly.csv")
