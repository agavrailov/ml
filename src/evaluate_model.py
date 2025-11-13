import os
import sys
import pandas as pd
import numpy as np
import json
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error # Added import
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import (
    get_latest_best_model_path,
    get_hourly_data_csv_path,
    get_scaler_params_json_path,
    BATCH_SIZE,
    ROWS_AHEAD,
    LSTM_UNITS,
    FREQUENCY,
    TSTEPS
)

from src.data_processing import prepare_keras_input_data, add_features # Import add_features
from src.model import build_lstm_model, load_stateful_weights_into_non_stateful_model # Import new functions

from src.train import create_sequences_for_stateless_lstm # Import the sequence creation function

def evaluate_model_performance(model_path, validation_window_size=500,
                             frequency=FREQUENCY, tsteps=TSTEPS, n_features=None, # n_features will be passed dynamically
                             lstm_units=LSTM_UNITS, n_lstm_layers=None, # n_lstm_layers will be passed dynamically
                             stateful=None, # stateful will be passed dynamically
                             optimizer_name='rmsprop', loss_function='mae',
                             features_to_use=None): # Added features_to_use
    """
    Evaluates the performance of a trained LSTM model.
    """
    if features_to_use is None:
        from src.config import FEATURES_TO_USE_OPTIONS
        features_to_use = FEATURES_TO_USE_OPTIONS[0] # Default to the first option if not provided

    # --- 1. Load Data and Model ---
    if model_path is None:
        print("Error: No model path provided for evaluation.")
        return

    hourly_data_csv = get_hourly_data_csv_path(frequency)
    scaler_params_json = get_scaler_params_json_path(frequency)

    if not os.path.exists(hourly_data_csv):
        print(f"Error: Hourly data not found for frequency {frequency} at '{hourly_data_csv}'. Please process data first.")
        return

    if not os.path.exists(scaler_params_json):
        print(f"Error: Scaler parameters not found at '{scaler_params_json}'.")
        return

    print(f"Loading trained (stateful) model from: {model_path}")
    stateful_model = keras.models.load_model(model_path)
    
    # Hyperparameters are passed directly, no need to load from best_hps.json
    # lstm_units, n_lstm_layers, stateful, optimizer_name, loss_function are already arguments


    # Create a non-stateful model for prediction and transfer weights
    print(f"Creating non-stateful model with {lstm_units} LSTM units for evaluation...")
    non_stateful_model = build_lstm_model( # Changed to build_lstm_model
        input_shape=(tsteps, n_features),
        lstm_units=lstm_units,
        batch_size=None, # Non-stateful model does not require batch_size
        learning_rate=0.001, # Learning rate is not used for prediction model compilation
        n_lstm_layers=n_lstm_layers,
        stateful=False, # Always non-stateful for evaluation
        optimizer_name=optimizer_name,
        loss_function=loss_function
    )
    load_stateful_weights_into_non_stateful_model(stateful_model, non_stateful_model)
    
    # Use the non_stateful_model for predictions
    model = non_stateful_model
    
    print("Loading hourly data and scaler parameters...")
    df_hourly = pd.read_csv(hourly_data_csv, parse_dates=['Time'])
    
    with open(scaler_params_json, 'r') as f:
        scaler_params = json.load(f)
    
    # --- 2. Filter and Prepare Data ---
    df_full_featured, _ = prepare_keras_input_data(hourly_data_csv, features_to_use) # Pass features_to_use

    if len(df_full_featured) < validation_window_size:
        print(f"Warning: Not enough data ({len(df_full_featured)} rows) for the specified validation window size ({validation_window_size}). Using all available data.")
        validation_window_size = len(df_full_featured)

    df_eval_featured = df_full_featured.iloc[-validation_window_size:].copy()

    if len(df_eval_featured) < tsteps + ROWS_AHEAD:
        print(f"Error: Not enough data in the evaluation window ({len(df_eval_featured)} rows) to generate a prediction.")
        return
        
    print(f"Evaluating on {len(df_eval_featured)} data points from the last validation window.")

    feature_cols = [col for col in df_eval_featured.columns if col != 'Time']
    
    mean_vals = pd.Series(scaler_params['mean'])
    std_vals = pd.Series(scaler_params['std'])
    
    df_eval_normalized = df_eval_featured.copy()
    df_eval_normalized[feature_cols] = (df_eval_featured[feature_cols] - mean_vals[feature_cols]) / std_vals[feature_cols]
    
    mean_open = scaler_params['mean']['Open']
    std_open = scaler_params['std']['Open']

    # --- 3. Generate Predictions using Sequence Method ---
    print("Generating predictions using sequence method...")
    
    # Create sequences from the normalized evaluation data
    X_eval, Y_eval_normalized = create_sequences_for_stateless_lstm(df_eval_normalized, tsteps, ROWS_AHEAD)

    if X_eval.shape[0] == 0:
        print("Could not generate any evaluation sequences with the available data.")
        return

    # Get the corresponding dates for the predictions
    # The actual values (Y_eval) correspond to the end of each sequence
    prediction_dates = df_eval_featured['Time'].iloc[tsteps + ROWS_AHEAD - 1 : tsteps + ROWS_AHEAD - 1 + len(Y_eval_normalized)].values

    # Make predictions on the entire set of sequences
    predictions_normalized = model.predict(X_eval, batch_size=BATCH_SIZE)

    # --- DIAGNOSTIC: Compare standard deviations ---
    print("\n--- Diagnostic Stats ---")
    print(f"Std Dev of Normalized Predictions: {np.std(predictions_normalized):.4f}")
    print(f"Std Dev of Normalized Actuals (Y_eval): {np.std(Y_eval_normalized):.4f}")
    print("------------------------\n")

    # Denormalize predictions and actuals
    predictions = (predictions_normalized * std_open) + mean_open
    actuals = (Y_eval_normalized * std_open) + mean_open

    # --- 4. Calculate and Print Evaluation Metrics ---
    mae = mean_absolute_error(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    correlation = np.corrcoef(actuals.flatten(), predictions.flatten())[0, 1] # Calculate correlation

    print("\n--- Model Performance Metrics ---")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Correlation (Actual vs. Predicted): {correlation:.4f}") # Print correlation
    print("---------------------------------\n")

    # --- 5. Plot the Results ---
    print("Plotting results...")
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(15, 7))

    ax.plot(prediction_dates, actuals, label='Actual Prices', color='royalblue', linewidth=2)
    ax.plot(prediction_dates, predictions, label='Predicted Prices', color='orangered', linestyle='--', linewidth=2)
    
    ax.set_title('Model Prediction vs. Actual Prices', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price (USD)', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True)
    
    fig.autofmt_xdate()
    plt.tight_layout()

    plot_filename = 'evaluation_plot.png'
    plt.savefig(plot_filename)
    print(f"Evaluation plot saved as '{plot_filename}'")

    return mae, correlation # Return MAE and correlation

if __name__ == "__main__":
    try:
        from src.config import (
            FREQUENCY, TSTEPS, N_FEATURES, LSTM_UNITS, N_LSTM_LAYERS, STATEFUL,
            FEATURES_TO_USE_OPTIONS
        )
        evaluate_model_performance(
            frequency=FREQUENCY,
            tsteps=TSTEPS,
            n_features=None, # n_features will be determined dynamically by the caller
            lstm_units=LSTM_UNITS,
            n_lstm_layers=None, # n_lstm_layers will be determined dynamically by the caller
            stateful=None, # stateful will be determined dynamically by the caller
            optimizer_name='rmsprop', # Default for now
            loss_function='mae', # Default for now
            features_to_use=FEATURES_TO_USE_OPTIONS[0] # Default for now
        )
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during model evaluation: {e}")