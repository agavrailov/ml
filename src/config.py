# src/config.py

import os

# --- Model Hyperparameters ---
TSTEPS = 3  # window size a.k.a. time steps
ROWS_AHEAD = 60  # prediction Labels are n rows ahead of the current
TR_SPLIT = 0.7   # part of data used for training
N_FEATURES = 4    # Number of features (OHLC)
BATCH_SIZE = 1 # Number of samples per gradient update. Reduced for smaller datasets.
EPOCHS = 20
LEARNING_RATE = 0.001
LSTM_UNITS = 50  # number of neurons in a LSTM layer

# --- Paths ---
PROCESSED_DATA_DIR = "data/processed"
TRAINING_DATA_CSV = os.path.join(PROCESSED_DATA_DIR, "training_data.csv")
SCALER_PARAMS_JSON = os.path.join(PROCESSED_DATA_DIR, "scaler_params.json")
MODEL_SAVE_PATH = "models/my_lstm_model.keras" # Keras native format

RAW_DATA_CSV = "data/raw/qqq_minute.csv"
HOURLY_DATA_CSV = os.path.join(PROCESSED_DATA_DIR, "xagusd_hourly.csv")

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
