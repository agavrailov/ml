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

    # Load the trained model (this model was trained with a specific batch_size)
    trained_model_full = keras.models.load_model(active_model_path)

    # Load best hyperparameters if available, otherwise use defaults from config
    best_hps = {}
    best_hps_path = 'best_hyperparameters.json'
    if os.path.exists(best_hps_path):
        with open(best_hps_path, 'r') as f:
            best_hps = json.load(f)
    
    # Get the LSTM units and learning rate from best_hps or config
    lstm_units = best_hps.get('lstm_units', LSTM_UNITS)
    learning_rate = best_hps.get('learning_rate', LEARNING_RATE)

    # Build a new model for prediction with batch_size=1
    # This model will have the same architecture but a batch_size of 1
    prediction_model = build_lstm_model(
        input_shape=(TSTEPS, N_FEATURES),
        lstm_units=lstm_units,
        batch_size=1, # Batch size of 1 for prediction
        learning_rate=learning_rate
    )
    # Copy weights from the trained model to the prediction model
    prediction_model.set_weights(trained_model_full.get_weights())

    # Load scaler parameters
    with open(scaler_params_path, 'r') as f:
        scaler_params = json.load(f)
    mean_vals = pd.Series(scaler_params['mean'])
    std_vals = pd.Series(scaler_params['std'])

    # Prepare input data
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
    # For prediction, batch_size is 1.
    input_reshaped = input_normalized_features.values[np.newaxis, :, :]

    # Reset model states before prediction for independent sequences
    for layer in prediction_model.layers: # Call reset_states on the prediction_model
        if hasattr(layer, 'reset_states') and layer.stateful:
            layer.reset_states()

    # Make prediction using the prediction model
    predictions_normalized = prediction_model.predict(input_reshaped)

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
        # Define the minimum number of data points required for feature engineering
        # Max rolling window (SMA_21) is 21. TSTEPS is 3.
        # So, we need at least 21 + TSTEPS - 1 = 23 data points.
        MIN_DATA_FOR_FEATURES = 21 + TSTEPS - 1 

        # Create dummy raw OHLC input data for prediction
        # This should represent the last MIN_DATA_FOR_FEATURES of your actual raw hourly data.
        dummy_input_data = pd.DataFrame(np.random.rand(MIN_DATA_FOR_FEATURES, 4) * 100 + 100,
                                        columns=['Open', 'High', 'Low', 'Close'])
        # Add a dummy 'Time' column for add_features to work
        dummy_input_data['Time'] = pd.to_datetime(pd.date_range(end=pd.Timestamp.now(), periods=MIN_DATA_FOR_FEATURES, freq='H'))
        
        print(f"Input data for prediction (raw OHLC):\n{dummy_input_data}")
        predicted_price = predict_future_prices(dummy_input_data)
        print(f"Predicted future price: {predicted_price}")
