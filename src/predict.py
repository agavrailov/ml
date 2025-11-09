import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import os

from src.model import build_lstm_model
from src.config import (
    TSTEPS, N_FEATURES, BATCH_SIZE,
    PROCESSED_DATA_DIR, SCALER_PARAMS_JSON, MODEL_SAVE_PATH,
    LSTM_UNITS, LEARNING_RATE, get_latest_model_path, get_active_model_path # Added get_active_model_path
)

def predict_future_prices(input_data_df, scaler_params_path=SCALER_PARAMS_JSON):
    """
    Loads the latest trained LSTM model, normalizes input data, makes predictions,
    and denormalizes the predictions. Handles stateful model prediction for single sequences.

    Args:
        input_data_df (pd.DataFrame): DataFrame containing the input data for prediction.
                                      Should have 'Open', 'High', 'Low', 'Close' columns.
                                      It must contain exactly TSTEPS data points.
        scaler_params_path (str): Path to the JSON file containing scaler parameters (mean, std).

    Returns:
        np.array: Denormalized predicted prices.
    """
    active_model_path = get_active_model_path()
    if not active_model_path:
        raise FileNotFoundError("No active model found. Please train and promote a model first.")

    # Load the trained model (which was trained with BATCH_SIZE)
    trained_model = keras.models.load_model(active_model_path)

    # Build a prediction model with batch_size=1, but same architecture
    # This is crucial for stateful LSTMs when predicting single samples
    prediction_model = build_lstm_model(
        input_shape=(TSTEPS, N_FEATURES),
        lstm_units=LSTM_UNITS,
        batch_size=1, # Prediction model uses batch_size=1
        learning_rate=LEARNING_RATE # Learning rate doesn't matter for prediction model
    )
    # Copy weights from the trained model to the prediction model
    prediction_model.set_weights(trained_model.get_weights())

    # Load scaler parameters
    with open(scaler_params_path, 'r') as f:
        scaler_params = json.load(f)
    mean_vals = pd.Series(scaler_params['mean'])
    std_vals = pd.Series(scaler_params['std'])

    # Prepare input data
    # Ensure input_data_df has the correct number of timesteps (TSTEPS)
    if len(input_data_df) != TSTEPS:
        raise ValueError(f"Input data for prediction must have {TSTEPS} timesteps, but got {len(input_data_df)}.")
    
    # Select OHLC columns and normalize
    ohlc_cols = ['Open', 'High', 'Low', 'Close']
    input_ohlc = input_data_df[ohlc_cols]
    input_normalized = (input_ohlc - mean_vals) / std_vals

    # Reshape input for LSTM: (batch_size, TSTEPS, N_FEATURES)
    # For prediction with batch_size=1 model, input shape is (1, TSTEPS, N_FEATURES)
    input_reshaped = input_normalized.values[np.newaxis, :, :] # Add batch dimension

    # Make prediction
    # Use the prediction_model with batch_size=1
    predictions_normalized = prediction_model.predict(input_reshaped, batch_size=1)

    # The model returns (1, TSTEPS, 1) because return_sequences=True.
    # We are interested in the prediction for the last timestep.
    single_prediction_normalized = predictions_normalized[0, -1, 0]

    # Denormalize prediction
    denormalized_prediction = single_prediction_normalized * std_vals['Open'] + mean_vals['Open']

    return denormalized_prediction

if __name__ == "__main__":
    # Example usage for prediction
    print("Running prediction example...")

    # Ensure an active model and scaler params exist
    active_model_path = get_active_model_path()
    if not active_model_path or not os.path.exists(SCALER_PARAMS_JSON):
        print("Error: Active model or scaler parameters not found. Please train and promote a model first.")
    else:
        # Create dummy input data for prediction
        # This should be the last TSTEPS of your actual data, normalized.
        # For this example, we'll create random data.
        dummy_input_data = pd.DataFrame(np.random.rand(TSTEPS, N_FEATURES) * 100 + 100,
                                        columns=['Open', 'High', 'Low', 'Close'])
        
        print(f"Input data for prediction:\n{dummy_input_data}")
        predicted_price = predict_future_prices(dummy_input_data)
        print(f"Predicted future price: {predicted_price}")