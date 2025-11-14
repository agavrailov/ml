import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import os
from tensorflow.keras.callbacks import EarlyStopping
import importlib
from datetime import datetime

from src.model import build_lstm_model, load_stateful_weights_into_non_stateful_model
from src.data_processing import prepare_keras_input_data
from src.data_utils import (
    fit_standard_scaler,
    apply_standard_scaler,
    get_effective_data_length,
    create_sequences_for_stateful_lstm,
)
from src.config import (
    ROWS_AHEAD, TR_SPLIT, BATCH_SIZE,
    EPOCHS, LEARNING_RATE, LSTM_UNITS,
    PROCESSED_DATA_DIR, MODEL_REGISTRY_DIR,
    FREQUENCY, TSTEPS, get_training_data_csv_path, get_scaler_params_json_path,
    get_hourly_data_csv_path, N_LSTM_LAYERS, STATEFUL,
    FEATURES_TO_USE_OPTIONS, OPTIMIZER_NAME, LOSS_FUNCTION,
    get_run_hyperparameters,
)
import src.config # Import src.config as a module

from src.data_utils import create_sequences_for_stateless_lstm  # re-export for backwards compatibility


def train_model(
    frequency: str = FREQUENCY,
    tsteps: int = TSTEPS,
    lstm_units: int | None = None,
    learning_rate: float | None = None,
    epochs: int | None = None,
    current_batch_size: int | None = None,
    n_lstm_layers: int | None = None,
    stateful: bool | None = None,
    features_to_use: list[str] | None = None,
):
    """Train an LSTM model for the given frequency/tsteps using tuned defaults.

    Callers may override individual hyperparameters; otherwise tuned values
    (if available) or TRAINING defaults are used via ``get_run_hyperparameters``.
    """
    importlib.reload(src.config)  # Ensure latest config is loaded
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Resolve effective hyperparameters for this run
    hps = get_run_hyperparameters(frequency=frequency, tsteps=tsteps)
    lstm_units = lstm_units or hps["lstm_units"]
    learning_rate = learning_rate or hps["learning_rate"]
    epochs = epochs or hps["epochs"]
    current_batch_size = current_batch_size or hps["batch_size"]
    n_lstm_layers = n_lstm_layers or hps["n_lstm_layers"]
    stateful = stateful if stateful is not None else hps["stateful"]
    if features_to_use is None:
        features_to_use = hps["features_to_use"]

    n_features_dynamic = len(features_to_use)

    # --- Data Preparation ---
    hourly_data_path = get_hourly_data_csv_path(frequency) # Get path to the hourly data
    scaler_params_path = get_scaler_params_json_path(frequency)

    if not os.path.exists(hourly_data_path):
        print(f"Error: Hourly data not found for frequency {frequency} at {hourly_data_path}. Skipping training.")
        return None

    df_featured, feature_cols = prepare_keras_input_data(hourly_data_path, features_to_use)

    # Split data into training and validation sets
    split_index = int(len(df_featured) * TR_SPLIT)
    df_train_raw = df_featured.iloc[:split_index].copy()
    df_val_raw = df_featured.iloc[split_index:].copy()

    # --- Normalization ---
    # Fit scaler on training data and persist parameters
    mean_vals, std_vals, scaler_params = fit_standard_scaler(df_train_raw, feature_cols)
    with open(scaler_params_path, 'w') as f:
        json.dump(scaler_params, f, indent=4)
    print(f"Scaler parameters calculated on training data and saved to {scaler_params_path}")

    # Normalize both training and validation sets using the scaler fitted on training data
    df_train_normalized = apply_standard_scaler(df_train_raw, feature_cols, scaler_params)
    df_val_normalized = apply_standard_scaler(df_val_raw, feature_cols, scaler_params)
    
    X_train, Y_train = create_sequences_for_stateful_lstm(df_train_normalized, tsteps, current_batch_size, ROWS_AHEAD)
    X_val, Y_val = create_sequences_for_stateful_lstm(df_val_normalized, tsteps, current_batch_size, ROWS_AHEAD)

    if X_train.shape[0] == 0 or X_val.shape[0] == 0:
        print(f"Warning: Not enough data to create sequences for training or validation for {frequency} with TSTEPS={tsteps}. Skipping training.")
        return None

    # --- Model Building and Training ---
    model = build_lstm_model(
        input_shape=(tsteps, n_features_dynamic),
        lstm_units=lstm_units,
        batch_size=current_batch_size if stateful else None, # Pass batch_size only if stateful
        learning_rate=learning_rate,
        n_lstm_layers=n_lstm_layers,
        stateful=stateful,
        optimizer_name=OPTIMIZER_NAME,
        loss_function=LOSS_FUNCTION
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print(f"Starting model training for {frequency} with TSTEPS={tsteps}...")
    history = model.fit(X_train, Y_train,
              epochs=epochs,
              batch_size=current_batch_size,
              validation_data=(X_val, Y_val),
              callbacks=[early_stopping],
              shuffle=False,
              verbose=1)
    print(f"Model training finished for {frequency} with TSTEPS={tsteps}.")

    final_val_loss = min(history.history['val_loss'])
    
    # --- Model Saving ---
    os.makedirs(MODEL_REGISTRY_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_REGISTRY_DIR, f"my_lstm_model_{frequency}_tsteps{tsteps}_{timestamp}.keras")
    model.save(model_path)
    print(f"Model saved to {model_path} with validation loss {final_val_loss:.4f}")

    # --- Bias Correction Calculation ---
    print("Calculating bias correction from validation set...")
    # Create a non-stateful model for making predictions on the validation set
    non_stateful_model_for_bias = build_lstm_model(
        input_shape=(tsteps, n_features_dynamic),
        lstm_units=lstm_units,
        batch_size=None, # Non-stateful model does not require batch_size
        learning_rate=learning_rate, # Learning rate is not used for prediction model compilation
        n_lstm_layers=n_lstm_layers,
        stateful=False, # Always non-stateful for evaluation
        optimizer_name=OPTIMIZER_NAME,
        loss_function=LOSS_FUNCTION
    )
    # Load weights from the trained stateful model into the non-stateful model
    load_stateful_weights_into_non_stateful_model(model, non_stateful_model_for_bias)

    # Make predictions on the validation set
    predictions_val_normalized = non_stateful_model_for_bias.predict(X_val, batch_size=current_batch_size)

    # Denormalize predictions and actuals
    mean_open = scaler_params['mean']['Open']
    std_open = scaler_params['std']['Open']

    predictions_val_denormalized = (predictions_val_normalized * std_open) + mean_open
    actuals_val_denormalized = (Y_val * std_open) + mean_open

    # Calculate mean residual
    mean_residual = np.mean(actuals_val_denormalized - predictions_val_denormalized)
    print(f"Mean residual on validation set: {mean_residual:.4f}")

    # Save mean residual to a JSON file
    bias_correction_path = os.path.join(MODEL_REGISTRY_DIR, f"bias_correction_{frequency}_tsteps{tsteps}_{timestamp}.json")
    with open(bias_correction_path, 'w') as f:
        json.dump({"mean_residual": float(mean_residual)}, f, indent=4)
    print(f"Bias correction (mean residual) saved to {bias_correction_path}")

    return final_val_loss, model_path, bias_correction_path

if __name__ == "__main__":
    # Ensure data/processed exists
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    importlib.reload(src.config) # Ensure latest config is loaded
    print(f"\n--- Training model for frequency: {FREQUENCY}, TSTEPS: {TSTEPS} ---")
    
    training_results = train_model(
        frequency=FREQUENCY,
        tsteps=TSTEPS,
        lstm_units=LSTM_UNITS,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        current_batch_size=BATCH_SIZE,
        n_lstm_layers=N_LSTM_LAYERS, # Pass N_LSTM_LAYERS from config
        stateful=STATEFUL,
        features_to_use=FEATURES_TO_USE_OPTIONS[0] # Pass the default feature set
    )
    
    if training_results is not None:
        final_loss, model_path, bias_correction_path = training_results
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
                'batch_size': BATCH_SIZE,
                'n_lstm_layers': N_LSTM_LAYERS, # Use N_LSTM_LAYERS from config
                'stateful': STATEFUL,
                'optimizer_name': OPTIMIZER_NAME,
                'loss_function': LOSS_FUNCTION,
                'bias_correction_filename': os.path.basename(bias_correction_path) # Save bias correction filename
            }
            print(f"Updated best hyperparameters for Frequency: {FREQUENCY}, TSTEPS: {TSTEPS} with validation loss {final_loss:.4f}")
        
        # Save the updated best_hyperparameters.json
        with open(best_hps_path, 'w') as f:
            json.dump(best_hps_overall, f, indent=4)
        print(f"\nUpdated best hyperparameters saved to {best_hps_path}")

        print("\n--- Training Summary ---")
        print(f"Frequency: {FREQUENCY}, TSTEPS: {TSTEPS}, Validation Loss: {final_loss:.4f}, Model: {model_filename}, Bias Correction: {os.path.basename(bias_correction_path)}")
    else:
        print("Model training was not successful.")
