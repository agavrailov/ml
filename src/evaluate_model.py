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
    get_active_model_path,
    SCALER_PARAMS_JSON,
    HOURLY_DATA_CSV,
    TSTEPS,
    N_FEATURES,
    BATCH_SIZE,
    ROWS_AHEAD
)

from src.data_processing import add_features # Import add_features

def evaluate_model_performance():
    """
    Loads the active model, generates predictions over the last 2 months of data,
    and plots the predicted prices against the actual prices.
    """
    # --- 1. Load Data and Model ---
    active_model_path = get_active_model_path()
    if active_model_path is None:
        print("Error: No active model found. Please promote a model first using 'src/promote_model.py'.")
        return

    if not os.path.exists(HOURLY_DATA_CSV):
        print(f"Error: Hourly data not found at '{HOURLY_DATA_CSV}'. Please process data first.")
        return

    if not os.path.exists(SCALER_PARAMS_JSON):
        print(f"Error: Scaler parameters not found at '{SCALER_PARAMS_JSON}'.")
        return

    print(f"Loading active model: {active_model_path}")
    model = keras.models.load_model(active_model_path)
    
    print("Loading hourly data and scaler parameters...")
    df_hourly = pd.read_csv(HOURLY_DATA_CSV, parse_dates=['Time'])
    df_hourly.rename(columns={'Time': 'DateTime'}, inplace=True)
    
    with open(SCALER_PARAMS_JSON, 'r') as f:
        scaler_params = json.load(f)
    
    # --- 2. Filter and Prepare Data ---
    # Load the entire dataset first to calculate features correctly
    df_full = pd.read_csv(HOURLY_DATA_CSV)
    df_full_featured = add_features(df_full.copy())

    # Now, filter for the last 2 months from the featured dataset
    two_months_ago = datetime.now() - timedelta(days=60)
    # Ensure the 'Time' column is in datetime format for comparison
    df_full_featured['Time'] = pd.to_datetime(df_full_featured['Time'])
    df_eval_featured = df_full_featured[df_full_featured['Time'] >= two_months_ago].copy()

    if len(df_eval_featured) < TSTEPS + ROWS_AHEAD:
        print(f"Error: Not enough data in the last 2 months ({len(df_eval_featured)} rows) to generate a prediction after feature calculation.")
        return
        
    print(f"Evaluating on {len(df_eval_featured)} data points from the last 2 months.")

    # Normalize the evaluation data using the saved scaler params
    feature_cols = [col for col in df_eval_featured.columns if col != 'Time']
    
    # Ensure the order of columns matches the scaler
    ordered_mean = np.array([scaler_params['mean'][col] for col in feature_cols])
    ordered_std = np.array([scaler_params['std'][col] for col in feature_cols])

    df_eval_normalized = (df_eval_featured[feature_cols] - ordered_mean) / ordered_std
    
    mean_open = scaler_params['mean']['Open']
    std_open = scaler_params['std']['Open']

    # --- 3. Generate Predictions ---
    predictions = []
    actuals = []
    prediction_dates = []
    
    print("Generating predictions...")
    # We can't predict for the last (TSTEPS + ROWS_AHEAD) data points
    for i in range(len(df_eval_normalized) - TSTEPS - ROWS_AHEAD):
        # Get the input sequence
        input_sequence = df_eval_normalized.iloc[i : i + TSTEPS].values
        
        # Reshape for the model (add batch dimension for a single sample)
        input_sequence = input_sequence.reshape(1, TSTEPS, N_FEATURES)
        
        # Make prediction
        predicted_normalized = model.predict(input_sequence, verbose=0)[0][-1][0]
        
        # Denormalize the prediction
        predicted_price = (predicted_normalized * std_open) + mean_open
        predictions.append(predicted_price)
        
        # Get the actual future price from the *un-normalized* featured dataframe
        actual_price = df_eval_featured.iloc[i + TSTEPS + ROWS_AHEAD]['Open']
        actuals.append(actual_price)
        
        # Get the date for which the prediction is made
        prediction_dates.append(df_eval_featured.iloc[i + TSTEPS + ROWS_AHEAD]['Time'])

    if not predictions:
        print("Could not generate any predictions with the available data.")
        return

    # --- 4. Calculate and Print Evaluation Metrics ---
    mae = mean_absolute_error(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)

    print("\n--- Model Performance Metrics ---")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print("---------------------------------\n")

    # --- 5. Plot the Results ---
    print("Plotting results...")
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(15, 7))

    ax.plot(prediction_dates, actuals, label='Actual Prices', color='royalblue', linewidth=2)
    ax.plot(prediction_dates, predictions, label='Predicted Prices', color='orangered', linestyle='--', linewidth=2)
    
    ax.set_title('Model Prediction vs. Actual Prices (Last 2 Months)', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price (USD)', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True)
    
    # Improve formatting
    fig.autofmt_xdate()
    plt.tight_layout()

    # Save the plot
    plot_filename = 'evaluation_plot.png'
    plt.savefig(plot_filename)
    print(f"Evaluation plot saved as '{plot_filename}'")

if __name__ == "__main__":
    evaluate_model_performance()
