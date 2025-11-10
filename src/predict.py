import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import os

from src.model import build_lstm_model
from src.data_processing import add_features # Import add_features
from src.config import (
    TSTEPS, N_FEATURES, BATCH_SIZE,
    PROCESSED_DATA_DIR, SCALER_PARAMS_JSON, MODEL_SAVE_PATH,
    LSTM_UNITS, LEARNING_RATE, get_latest_model_path, get_active_model_path
)

def predict_future_prices(input_data_df, scaler_params_path=SCALER_PARAMS_JSON):
    """
    Loads the latest trained LSTM model, prepares and normalizes input data,
    makes predictions, and denormalizes the predictions.

    Args:
        input_data_df (pd.DataFrame): DataFrame containing the raw OHLC input data for prediction.
                                      It must contain enough historical data to allow for feature
                                      engineering (e.g., rolling means) and still provide `TSTEPS`
                                      data points for the model's input.
        scaler_params_path (str): Path to the JSON file containing scaler parameters (mean, std).

    Returns:
        np.array: Denormalized predicted prices.
    """
    active_model_path = get_active_model_path()
    if not active_model_path:
        raise FileNotFoundError("No active model found. Please train and promote a model first.")

    # Load the trained model
    trained_model = keras.models.load_model(active_model_path)

    # Load scaler parameters
    with open(scaler_params_path, 'r') as f:
        scaler_params = json.load(f)
    mean_vals = pd.Series(scaler_params['mean'])
    std_vals = pd.Series(scaler_params['std'])

    # Prepare input data
    # Ensure input_data_df has the correct number of timesteps (TSTEPS)
    if len(input_data_df) != TSTEPS:
        raise ValueError(f"Input data for prediction must have {TSTEPS} timesteps, but got {len(input_data_df)}.")
    
    # Apply feature engineering to the input data
    # Make a copy to avoid SettingWithCopyWarning
    input_data_df_copy = input_data_df.copy() 
    df_featured_input = add_features(input_data_df_copy)

    # Ensure the featured input has enough data after adding features (e.g., for rolling means)
    if len(df_featured_input) < TSTEPS:
        raise ValueError(f"Not enough data after feature engineering to form {TSTEPS} timesteps. "
                         f"Consider providing more historical data for feature calculation.")
    
    # Select only the last TSTEPS rows after feature engineering
    df_featured_input = df_featured_input.tail(TSTEPS)

    # Select feature columns for normalization
    feature_cols = [col for col in df_featured_input.columns if col != 'Time']
    
    # Normalize the input features using the loaded scaler parameters
    input_normalized_features = (df_featured_input[feature_cols] - mean_vals[feature_cols]) / std_vals[feature_cols]

    # Reshape input for LSTM: (batch_size, TSTEPS, N_FEATURES)
    # Add batch dimension (batch_size=1 for single prediction)
    input_reshaped = input_normalized_features.values[np.newaxis, :, :]

    # Make prediction using the trained model
    predictions_normalized = trained_model.predict(input_reshaped)

    # The model returns (batch_size, 1). Extract the single prediction.
    single_prediction_normalized = predictions_normalized[0, 0]

    # Denormalize prediction using the 'Open' price's mean and std
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
        # Create dummy raw OHLC input data for prediction
        # This should represent the last TSTEPS of your actual raw hourly data.
        # For this example, we'll create random data with 4 OHLC columns.
        dummy_input_data = pd.DataFrame(np.random.rand(TSTEPS, 4) * 100 + 100,
                                        columns=['Open', 'High', 'Low', 'Close'])
        # Add a dummy 'Time' column for add_features to work
        dummy_input_data['Time'] = pd.to_datetime(pd.date_range(end=pd.Timestamp.now(), periods=TSTEPS, freq='H'))
        
        print(f"Input data for prediction (raw OHLC):\n{dummy_input_data}")
        predicted_price = predict_future_prices(dummy_input_data)
        print(f"Predicted future price: {predicted_price}")
