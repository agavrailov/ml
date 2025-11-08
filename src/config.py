# src/config.py

import os

# --- Model Hyperparameters ---
TSTEPS = 3  # window size a.k.a. time steps
ROWS_AHEAD = 60  # prediction Labels are n rows ahead of the current
TR_SPLIT = 0.7   # part of data used for training
N_FEATURES = 4    # Number of features (OHLC)
BATCH_SIZE = 500 # Original batch size from R code
EPOCHS = 20
LEARNING_RATE = 0.001
LSTM_UNITS = 50  # number of neurons in a LSTM layer

# --- Paths ---
PROCESSED_DATA_DIR = "data/processed"
TRAINING_DATA_CSV = os.path.join(PROCESSED_DATA_DIR, "training_data.csv")
SCALER_PARAMS_JSON = os.path.join(PROCESSED_DATA_DIR, "scaler_params.json")
MODEL_SAVE_PATH = "models/my_lstm_model.keras" # Keras native format
