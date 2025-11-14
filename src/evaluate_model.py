import os
import sys
import pandas as pd
import numpy as np
import json
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error # Added import
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import importlib # Added import for reloading modules

from src.config import (
    get_latest_best_model_path,
    get_hourly_data_csv_path,
    get_scaler_params_json_path,
    BATCH_SIZE,
    ROWS_AHEAD,
    LSTM_UNITS,
    FREQUENCY,
    TSTEPS,
    OPTIMIZER_NAME,
    LOSS_FUNCTION
)
import src.config # Import src.config as a module

from src.data_processing import prepare_keras_input_data, add_features
from src.model import build_lstm_model, load_stateful_weights_into_non_stateful_model
from src.data_utils import (
    apply_standard_scaler,
    create_sequences_for_stateless_lstm,
)

from typing import Optional, Tuple

...

def evaluate_model_performance(
    model_path: str,
    validation_window_size: int = 500,
    correction_window_size: int = 100,
    frequency: str = FREQUENCY,
    tsteps: int = TSTEPS,
    n_features: Optional[int] = None,
    lstm_units: Optional[int] = None,
    n_lstm_layers: Optional[int] = None,
    stateful: Optional[bool] = None,
    features_to_use: Optional[list[str]] = None,
    bias_correction_path: Optional[str] = None,
) -> Tuple[float, float]:
    """Evaluate a trained LSTM model on the most recent validation window.

    Loads the trained *stateful* model from ``model_path``, creates a
    non-stateful clone for inference, generates rolling-window predictions,
    applies optional bias/amplitude correction, and computes error metrics.

    Args:
        model_path: Path to the saved Keras model (stateful).
        validation_window_size: Number of most-recent hourly samples to use for
            evaluation.
        correction_window_size: Window length for rolling bias/amplitude
            correction applied on the denormalized predictions.
        frequency: Resampling frequency (e.g. ``"15min"``) for locating data
            and scaler files.
        tsteps: Number of timesteps per input sequence.
        n_features: Number of input features used by the model (if known).
        lstm_units: Number of LSTM units in each layer.
        n_lstm_layers: Number of stacked LSTM layers.
        stateful: Whether the original model was trained as stateful (kept for
            completeness, not used directly here).
        features_to_use: Feature names the model was trained on.
        bias_correction_path: Optional path to a JSON file containing
            precomputed bias-correction statistics.

    Returns:
        A tuple ``(mae, correlation)`` with the mean absolute error and Pearson
        correlation between actual and predicted prices.
    """
    importlib.reload(src.config) # Ensure latest config is loaded
    if features_to_use is None:
        from src.config import FEATURES_TO_USE_OPTIONS
        features_to_use = FEATURES_TO_USE_OPTIONS[0] # Default to the first option if not provided

    n_features_dynamic = len(features_to_use) # Calculate n_features dynamically

    mean_residual = 0.0
    if bias_correction_path and os.path.exists(bias_correction_path):
        with open(bias_correction_path, 'r') as f:
            bias_data = json.load(f)
            mean_residual = bias_data.get('mean_residual', 0.0)
        print(f"Loaded bias correction: mean_residual = {mean_residual:.4f}")
    else:
        print("No bias correction path provided or file not found. Predictions will not be bias-corrected.")

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
        input_shape=(tsteps, n_features_dynamic),
        lstm_units=lstm_units,
        batch_size=None, # Non-stateful model does not require batch_size
        learning_rate=0.001, # Learning rate is not used for prediction model compilation
        n_lstm_layers=n_lstm_layers,
        stateful=False, # Always non-stateful for evaluation
        optimizer_name=OPTIMIZER_NAME,
        loss_function=LOSS_FUNCTION
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
    
    # Normalize evaluation window using the stored scaler parameters
    df_eval_normalized = apply_standard_scaler(df_eval_featured, feature_cols, scaler_params)
    
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

    print(f"Mean 'Open' for denormalization: {mean_open:.4f}")
    print(f"Std 'Open' for denormalization: {std_open:.4f}")

    # Denormalize predictions and actuals
    predictions = (predictions_normalized * std_open) + mean_open
    actuals = (Y_eval_normalized * std_open) + mean_open

    # Initialize corrected predictions array
    predictions_corrected = np.zeros_like(predictions)

    # Apply rolling bias correction and amplitude scaling
    for i in range(len(predictions)):
        if i < correction_window_size:
            # For the initial period, use the global mean residual and amplitude scaling
            current_predictions_window = predictions[:i+1]
            current_actuals_window = actuals[:i+1]
        else:
            current_predictions_window = predictions[i-correction_window_size+1 : i+1]
            current_actuals_window = actuals[i-correction_window_size+1 : i+1]

        # Calculate rolling mean residual
        rolling_mean_residual = np.mean(current_actuals_window - current_predictions_window)

        # Calculate rolling amplitude scaling factor
        std_actuals_window = np.std(current_actuals_window)
        std_predictions_window = np.std(current_predictions_window)

        amplitude_scaling_factor = 1.0
        if std_predictions_window > 0:
            amplitude_scaling_factor = std_actuals_window / std_predictions_window
        
        # Apply rolling amplitude scaling
        prediction_amplitude_corrected = (predictions[i] - np.mean(current_predictions_window)) * amplitude_scaling_factor + np.mean(current_predictions_window)
        
        # Apply rolling bias correction
        predictions_corrected[i] = prediction_amplitude_corrected + rolling_mean_residual
    
    print(f"Applied rolling bias correction and amplitude scaling with window size: {correction_window_size}")

    # Calculate residuals (using corrected predictions)
    residuals = actuals - predictions_corrected

    # --- 4. Calculate and Print Evaluation Metrics ---
    mae = mean_absolute_error(actuals, predictions_corrected)
    mse = mean_squared_error(actuals, predictions_corrected)
    rmse = np.sqrt(mse)
    correlation = np.corrcoef(actuals.flatten(), predictions_corrected.flatten())[0, 1] # Calculate correlation

    print("\n--- Model Performance Metrics ---")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Correlation (Actual vs. Predicted): {correlation:.4f}") # Print correlation
    print("---------------------------------\n")

    # --- 5. Plot the Results ---
    print("Plotting results...")
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True) # Two subplots

    # Plot Actual vs. Predicted
    ax1.plot(prediction_dates, actuals, label='Actual Prices', color='royalblue', linewidth=2)
    ax1.plot(prediction_dates, predictions_corrected, label='Predicted Prices (Corrected)', color='orangered', linestyle='--', linewidth=2)
    ax1.set_title('Model Prediction vs. Actual Prices (Bias Corrected)', fontsize=16)
    ax1.set_ylabel('Price (USD)', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True)

    # Plot Residuals
    ax2.plot(prediction_dates, residuals, label='Residuals (Actual - Predicted)', color='green', linewidth=1)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=0.8)
    ax2.set_title('Prediction Residuals', fontsize=16)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Error (USD)', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True)
    
    fig.autofmt_xdate()
    plt.tight_layout()

    plot_filename = 'evaluation_plot.png'
    plt.savefig(plot_filename)
    print(f"Evaluation plot saved as '{plot_filename}'")

    # Plot histogram of residuals
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, color='skyblue', edgecolor='black')
    plt.title('Histogram of Prediction Residuals', fontsize=16)
    plt.xlabel('Residual (Actual - Predicted)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True)
    plt.savefig('residuals_histogram.png')
    print("Residuals histogram saved as 'residuals_histogram.png'")

    return mae, correlation # Return MAE and correlation

if __name__ == "__main__":
    try:
        importlib.reload(src.config) # Ensure latest config is loaded
        from src.config import (
            FREQUENCY, TSTEPS, N_FEATURES, LSTM_UNITS, N_LSTM_LAYERS, STATEFUL,
            FEATURES_TO_USE_OPTIONS
        )
        
        model_path, bias_correction_path, features_to_use_trained, lstm_units_trained, n_lstm_layers_trained = get_latest_best_model_path(target_frequency=FREQUENCY, tsteps=TSTEPS)
        print(f"DEBUG: model_path from get_latest_best_model_path: {model_path}")
        print(f"DEBUG: bias_correction_path from get_latest_best_model_path: {bias_correction_path}")
        print(f"DEBUG: features_to_use_trained from get_latest_best_model_path: {features_to_use_trained}")
        print(f"DEBUG: lstm_units_trained from get_latest_best_model_path: {lstm_units_trained}")
        print(f"DEBUG: n_lstm_layers_trained from get_latest_best_model_path: {n_lstm_layers_trained}")

        if model_path:
            if features_to_use_trained is None:
                print("WARNING: features_to_use_trained not found in best_hyperparameters.json. Using default from config.")
                features_to_use_trained = FEATURES_TO_USE_OPTIONS[0] # Fallback to default
            
            n_features_trained = len(features_to_use_trained)
            print(f"DEBUG: n_features_trained: {n_features_trained}")

            # Use trained hyperparameters if available, otherwise fallback to config defaults
            effective_lstm_units = lstm_units_trained if lstm_units_trained is not None else LSTM_UNITS
            effective_n_lstm_layers = n_lstm_layers_trained if n_lstm_layers_trained is not None else N_LSTM_LAYERS

            evaluate_model_performance(
                model_path=model_path,
                frequency=FREQUENCY,
                tsteps=TSTEPS,
                n_features=n_features_trained, # Pass n_features_trained
                lstm_units=effective_lstm_units, # Pass effective_lstm_units
                n_lstm_layers=effective_n_lstm_layers, # Pass effective_n_lstm_layers
                stateful=STATEFUL,
                features_to_use=features_to_use_trained, # Pass features_to_use_trained
                bias_correction_path=bias_correction_path, # Pass bias correction path
                correction_window_size=100 # Default correction window size
            )
        else:
            print(f"ERROR: No best model found for frequency={FREQUENCY}, tsteps={TSTEPS}. Please train a model first.")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during model evaluation: {e}")