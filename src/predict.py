import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import os

from src.model import build_lstm_model, build_non_stateful_lstm_model, load_stateful_weights_into_non_stateful_model
from src.data_processing import add_features # Import add_features
from src.config import (
    TSTEPS, N_FEATURES, BATCH_SIZE, FREQUENCY,
    PROCESSED_DATA_DIR, get_scaler_params_json_path,
    LSTM_UNITS, LEARNING_RATE, get_latest_best_model_path
)

def predict_future_prices(input_data_df):
    """
    Loads the latest trained LSTM model, prepares and normalizes input data,
    makes predictions, and denormalizes the predictions using a non-stateful model.

    Args:
        input_data_df (pd.DataFrame): DataFrame containing the raw OHLC input data for prediction.
                                      It must contain enough historical data to allow for feature
                                      engineering and still provide `TSTEPS` data points.

    Returns:
        np.array: Denormalized predicted prices.
    """
    # Get the path to the best model for the current FREQUENCY and TSTEPS
    active_model_path = get_latest_best_model_path(target_frequency=FREQUENCY, tsteps=TSTEPS)
    if not active_model_path or not os.path.exists(active_model_path):
        raise FileNotFoundError(f"No best model found for frequency {FREQUENCY} and TSTEPS {TSTEPS}. Please train a model first.")

    print(f"Loading stateful model from: {active_model_path}")
    stateful_model = keras.models.load_model(active_model_path)

    # Build a non-stateful model for prediction and transfer weights
    print("Creating non-stateful model for prediction and transferring weights...")
    prediction_model = build_non_stateful_lstm_model(
        input_shape=(TSTEPS, N_FEATURES),
        lstm_units=LSTM_UNITS # Assuming LSTM_UNITS is consistent or loaded from model metadata
    )
    load_stateful_weights_into_non_stateful_model(stateful_model, prediction_model)

    # Load scaler parameters for the current FREQUENCY
    scaler_params_path = get_scaler_params_json_path()
    if not os.path.exists(scaler_params_path):
        raise FileNotFoundError(f"Scaler parameters not found at {scaler_params_path}.")
    
    with open(scaler_params_path, 'r') as f:
        scaler_params = json.load(f)
    mean_vals = pd.Series(scaler_params['mean'])
    std_vals = pd.Series(scaler_params['std'])

    # Prepare input data
    input_data_df_copy = input_data_df.copy()
    df_featured_input = add_features(input_data_df_copy)

    if len(df_featured_input) < TSTEPS:
        raise ValueError(f"Not enough data after feature engineering to form {TSTEPS} timesteps.")
    
    df_featured_input = df_featured_input.tail(TSTEPS)

    feature_cols = [col for col in df_featured_input.columns if col != 'Time']
    
    # Normalize the input features
    input_normalized_features = (df_featured_input[feature_cols] - mean_vals[feature_cols]) / std_vals[feature_cols]

    # Reshape for non-stateful LSTM: (1, TSTEPS, N_FEATURES)
    input_reshaped = input_normalized_features.values[np.newaxis, :, :]

    # Make prediction
    predictions_normalized = prediction_model.predict(input_reshaped)
    single_prediction_normalized = predictions_normalized[0, 0]

    # Denormalize prediction
    denormalized_prediction = single_prediction_normalized * std_vals['Open'] + mean_vals['Open']

    return denormalized_prediction

if __name__ == "__main__":
    # Example usage for prediction
    print("Running prediction example...")

    # Ensure an active model and scaler params exist
    active_model_path = get_latest_best_model_path(target_frequency=FREQUENCY, tsteps=TSTEPS)
    scaler_params_path = get_scaler_params_json_path()

    if not active_model_path or not os.path.exists(scaler_params_path):
        print(f"Error: Active model for {FREQUENCY} (TSTEPS={TSTEPS}) or scaler parameters not found. Please train a model first.")
    else:
        # Define the minimum number of data points required for feature engineering
        # Max rolling window (SMA_21) is 21.
        # So, we need at least (21 - 1) + TSTEPS = 20 + TSTEPS data points.
        MIN_DATA_FOR_FEATURES = 20 + TSTEPS

        # Create dummy raw OHLC input data for prediction
        # This should represent the last MIN_DATA_FOR_FEATURES of your actual raw hourly data.
        dummy_input_data = pd.DataFrame(np.random.rand(MIN_DATA_FOR_FEATURES, 4) * 100 + 100,
                                        columns=['Open', 'High', 'Low', 'Close'])
        # Add a dummy 'Time' column for add_features to work
        dummy_input_data['Time'] = pd.to_datetime(pd.date_range(end=pd.Timestamp.now(), periods=MIN_DATA_FOR_FEATURES, freq='H'))
        
        print(f"Input data for prediction (raw OHLC):\n{dummy_input_data}")
        predicted_price = predict_future_prices(dummy_input_data)
        print(f"Predicted future price: {predicted_price}")
