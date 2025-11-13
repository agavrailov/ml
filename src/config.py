# src/config.py

import os
import json
from datetime import datetime, time

# --- Model Hyperparameters ---
# Default values for a single run
FREQUENCY = '30min' # Resampling frequency for the data
TSTEPS = 8 # Number of time steps to look back. Found by KerasTuner.
ROWS_AHEAD = 1  # prediction Labels are n rows ahead of the current
TR_SPLIT = 0.7   # part of data used for training
BATCH_SIZE = 128 # Number of samples per gradient update. Found by KerasTuner.
EPOCHS = 20
LEARNING_RATE = 0.01
LSTM_UNITS = 128  # number of neurons in a LSTM layer. Found by KerasTuner.
DROPOUT_RATE_1 = 0.1
DROPOUT_RATE_2 = 0.1
N_LSTM_LAYERS = 1 # Number of LSTM layers
STATEFUL = True # Whether the LSTM model is stateful

# --- Hyperparameter Tuning Options ---
RESAMPLE_FREQUENCIES = ['15min', '30min', '60min', '240min']
TSTEPS_OPTIONS = [5, 8, 16, 24, 48]
LSTM_UNITS_OPTIONS = [64, 128, 256]
BATCH_SIZE_OPTIONS = [64, 128, 256]
DROPOUT_RATE_OPTIONS = [0.0, 0.1, 0.2, 0.3]
N_LSTM_LAYERS_OPTIONS = [1, 2] # 1 or 2 LSTM layers
STATEFUL_OPTIONS = [True, False] # Stateful or non-stateful LSTM
OPTIMIZER_OPTIONS = ['rmsprop', 'adam']
LOSS_FUNCTION_OPTIONS = ['mae', 'mse']
FEATURES_TO_USE_OPTIONS = [
    ['Open', 'High', 'Low', 'Close', 'SMA_7', 'SMA_21', 'RSI'],
    ['Open', 'High', 'Low', 'Close', 'SMA_7', 'SMA_21', 'RSI', 'Hour', 'DayOfWeek']
]

# --- Paths ---
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
MODEL_SAVE_PATH = "models/my_lstm_model.keras" # Keras native format
MODEL_REGISTRY_DIR = "models/registry" # Directory to store versioned models
ACTIVE_MODEL_PATH_FILE = os.path.join("models", "active_model.txt") # Added active model pointer file
RAW_DATA_CSV = os.path.join(RAW_DATA_DIR, "nvda_minute.csv")

def get_hourly_data_csv_path(frequency, processed_data_dir=PROCESSED_DATA_DIR):
    """Generates the path for the resampled data CSV based on the given frequency."""
    return os.path.join(processed_data_dir, f"nvda_{frequency}.csv")

def get_training_data_csv_path(frequency, processed_data_dir=PROCESSED_DATA_DIR):
    """Generates the path for the training data CSV based on the given frequency."""
    return os.path.join(processed_data_dir, f"training_data_{frequency}.csv")

def get_scaler_params_json_path(frequency, processed_data_dir=PROCESSED_DATA_DIR):
    """Generates the path for the scaler parameters JSON based on the given frequency."""
    return os.path.join(processed_data_dir, f"scaler_params_{frequency}.json")

def get_active_model_path():
    """
    Reads the path of the currently active model from a file.
    """
    if os.path.exists(ACTIVE_MODEL_PATH_FILE):
        with open(ACTIVE_MODEL_PATH_FILE, 'r') as f:
            return f.read().strip()
    return None

def get_latest_best_model_path(target_frequency=None, tsteps=None):
    """
    Finds the path to the model with the lowest validation loss for a given
    frequency and TSTEPS, or the overall best model, by consulting best_hyperparameters.json.
    """
    best_hps_path = 'best_hyperparameters.json'
    if not os.path.exists(best_hps_path):
        return None

    with open(best_hps_path, 'r') as f:
        best_hps_data = json.load(f)

    best_loss = float('inf')
    best_model_filename = None
    
    for freq, tsteps_data in best_hps_data.items():
        if target_frequency and freq != target_frequency:
            continue
        for tstep_val_str, metrics in tsteps_data.items():
            current_tsteps = int(tstep_val_str)
            if tsteps and current_tsteps != tsteps:
                continue
            
            if metrics['validation_loss'] < best_loss:
                best_loss = metrics['validation_loss']
                best_model_filename = metrics['model_filename'] # Assuming filename is stored here
    
    if best_model_filename:
        return os.path.join(MODEL_REGISTRY_DIR, best_model_filename)
    return None

# --- Data Ingestion Configuration ---
TWS_HOST = '127.0.0.1'
TWS_PORT = 7497
TWS_CLIENT_ID = 1
TWS_MAX_CONCURRENT_REQUESTS = 3 # Max concurrent historical data requests to TWS
DATA_BATCH_SAVE_SIZE = 7 # Save raw data in batches of N days

# Contract details for NVDA
NVDA_CONTRACT_DETAILS = {
    'symbol': 'NVDA',
    'secType': 'STK',
    'exchange': 'SMART',
    'currency': 'USD'
}

# Market hours for gap analysis (New York time)
MARKET_TIMEZONE = 'America/New_York'
MARKET_OPEN_TIME = time(9, 30)
MARKET_CLOSE_TIME = time(16, 0)
EXCHANGE_CALENDAR_NAME = 'XNYS' # New York Stock Exchange
