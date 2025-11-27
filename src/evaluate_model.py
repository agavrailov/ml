import os
import sys
import pandas as pd
import numpy as np
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error  # Added import
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
    LOSS_FUNCTION,
)
import src.config  # Import src.config as a module

from src.data import load_hourly_features
from src.model import (
    build_lstm_model,
    load_stateful_weights_into_non_stateful_model,
    load_model,
)
from src.data_utils import (
    apply_standard_scaler,
    create_sequences_for_stateless_lstm,
)
from src.bias_correction import apply_rolling_bias_and_amplitude_correction

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
    # Load the trained *stateful* model via the unified model loader.
    stateful_model = load_model(model_path)
    
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
    
    print(f"Loading hourly data and scaler parameters...")

    with open(scaler_params_json, 'r') as f:
        scaler_params = json.load(f)
    
    # --- 2. Filter and Prepare Data ---
    # Use centralized data helper to obtain the feature-engineered frame.
    df_full_featured, _ = load_hourly_features(frequency, features_to_use)

    # Use the last 20% of the available data as the validation/evaluation window.
    total_rows = len(df_full_featured)
    if total_rows == 0:
        print("Error: No data available in hourly feature set for evaluation.")
        return

    validation_window_size = max(1, int(total_rows * 0.2))
    df_eval_featured = df_full_featured.iloc[-validation_window_size:].copy()

    if len(df_eval_featured) < tsteps + ROWS_AHEAD:
        print(
            f"Error: Not enough data in the evaluation window ({len(df_eval_featured)} rows) "
            "to generate a prediction."
        )
        return

    print(f"Evaluating on {len(df_eval_featured)} data points from the last validation window.")

    feature_cols = [col for col in df_eval_featured.columns if col != "Time"]

    # Normalize evaluation window using the stored scaler parameters
    df_eval_normalized = apply_standard_scaler(df_eval_featured, feature_cols, scaler_params)

    # --- 3. Generate Predictions using Sequence Method (log returns) ---
    print("Generating predictions using sequence method (log returns)...")

    # Create input sequences from the normalized evaluation data. We ignore the
    # price labels returned by this helper and instead construct log-return
    # targets from the raw Open prices.
    X_eval, _ = create_sequences_for_stateless_lstm(df_eval_normalized, tsteps, ROWS_AHEAD)

    if X_eval.shape[0] == 0:
        print("Could not generate any evaluation sequences with the available data.")
        return

    # Build log-return targets aligned to the end of each window.
    from src.data_utils import compute_log_return_labels

    prices_eval = df_eval_featured["Open"].to_numpy(dtype=float)
    log_returns = compute_log_return_labels(prices_eval, rows_ahead=ROWS_AHEAD)

    n = len(log_returns)
    if n == 0:
        print("No prices available to compute log-return targets.")
        return

    valid_len = max(0, n - ROWS_AHEAD)
    if valid_len <= 0:
        print("Not enough data to compute forward log returns for evaluation.")
        return

    total_sequences = max(0, valid_len - tsteps + 1)
    if total_sequences <= 0:
        print("Not enough data to align log-return targets with sequences.")
        return

    effective_sequences = min(X_eval.shape[0], total_sequences)
    y_eval = []
    for i in range(effective_sequences):
        t_end = i + tsteps - 1
        y_eval.append(log_returns[t_end])

    if not y_eval:
        print("No aligned log-return targets available for evaluation.")
        return

    Y_eval = np.asarray(y_eval, dtype=float)
    if Y_eval.ndim == 1:
        Y_eval = Y_eval.reshape(-1, 1)

    # Truncate X_eval if necessary so that X and Y lengths match.
    if X_eval.shape[0] > Y_eval.shape[0]:
        X_eval = X_eval[: Y_eval.shape[0]]

    # Get the corresponding dates for the predictions (use horizon timestamp).
    prediction_dates = df_eval_featured["Time"].iloc[
        tsteps + ROWS_AHEAD - 1 : tsteps + ROWS_AHEAD - 1 + len(Y_eval)
    ].values

    # Make predictions on the entire set of sequences; outputs are log-return
    # predictions in the same units as ``Y_eval``.
    predictions_log = model.predict(X_eval, batch_size=BATCH_SIZE)

    # --- DIAGNOSTIC: Compare standard deviations in log-return space ---
    print("\n--- Diagnostic Stats (log returns) ---")
    print(f"Std Dev of Predicted Log Returns: {np.std(predictions_log):.6f}")
    print(f"Std Dev of Actual Log Returns (Y_eval): {np.std(Y_eval):.6f}")
    print("------------------------\n")

    # Ensure shapes are compatible
    predictions_log = predictions_log.reshape(-1, 1)
    if predictions_log.shape[0] > Y_eval.shape[0]:
        predictions_log = predictions_log[: Y_eval.shape[0]]

    # Apply explicit rolling bias-correction layer in log-return space.
    predictions_corrected_log = apply_rolling_bias_and_amplitude_correction(
        predictions=predictions_log.flatten(),
        actuals=Y_eval.flatten(),
        window=correction_window_size,
        global_mean_residual=mean_residual,
    ).reshape(predictions_log.shape)
    print(
        f"Applied rolling bias correction and amplitude scaling in log-return space "
        f"with window size: {correction_window_size}"
    )

    # Calculate residuals (using corrected predictions) in log-return space.
    residuals_log = Y_eval - predictions_corrected_log

    # --- 4. Calculate and Print Evaluation Metrics (log returns) ---
    mae = mean_absolute_error(Y_eval, predictions_corrected_log)
    mse = mean_squared_error(Y_eval, predictions_corrected_log)
    rmse = np.sqrt(mse)
    correlation = np.corrcoef(Y_eval.flatten(), predictions_corrected_log.flatten())[0, 1]

    print("\n--- Model Performance Metrics (log returns) ---")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"Correlation (Actual vs. Predicted): {correlation:.4f}")
    print("---------------------------------------------\n")

    # --- 5. Plot the Results ---
    print("Plotting results...")
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)  # Two subplots

    # Plot Actual vs. Predicted log returns
    ax1.plot(
        prediction_dates,
        Y_eval,
        label='Actual Log Returns',
        color='royalblue',
        linewidth=2,
    )
    ax1.plot(
        prediction_dates,
        predictions_corrected_log,
        label='Predicted Log Returns (Corrected)',
        color='orangered',
        linestyle='--',
        linewidth=2,
    )
    ax1.set_title('Model Prediction vs. Actual Log Returns (Bias Corrected)', fontsize=16)
    ax1.set_ylabel('Log Return', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True)

    # Plot Residuals in log-return space
    ax2.plot(
        prediction_dates,
        residuals_log,
        label='Residuals (Actual - Predicted)',
        color='green',
        linewidth=1,
    )
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=0.8)
    ax2.set_title('Prediction Residuals (Log Returns)', fontsize=16)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Error (Log Return)', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True)
    
    fig.autofmt_xdate()
    plt.tight_layout()

    # Ensure plots are saved into the top-level 'models' directory
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)

    evaluation_plot_path = os.path.join(models_dir, 'evaluation_plot.png')
    plt.savefig(evaluation_plot_path)
    print(f"Evaluation plot saved as '{evaluation_plot_path}'")

    # Plot histogram of residuals (log returns)
    plt.figure(figsize=(10, 6))
    plt.hist(residuals_log, bins=50, color='skyblue', edgecolor='black')
    plt.title('Histogram of Prediction Residuals (Log Returns)', fontsize=16)
    plt.xlabel('Residual (Actual - Predicted Log Return)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True)

    residuals_hist_path = os.path.join(models_dir, 'residuals_histogram.png')
    plt.savefig(residuals_hist_path)
    print(f"Residuals histogram saved as '{residuals_hist_path}'")

    return mae, correlation  # Return MAE and correlation

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