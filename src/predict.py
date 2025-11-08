import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import os

from src.model import build_lstm_model # Not directly used for loading, but for context
from src.config import (
    TSTEPS, N_FEATURES, BATCH_SIZE,
    PROCESSED_DATA_DIR, SCALER_PARAMS_JSON, MODEL_SAVE_PATH
)

def predict_future_prices(input_data_df, model_path=MODEL_SAVE_PATH, scaler_params_path=SCALER_PARAMS_JSON):
    """
    Loads a trained LSTM model, normalizes input data, makes predictions,
    and denormalizes the predictions.

    Args:
        input_data_df (pd.DataFrame): DataFrame containing the input data for prediction.
                                      Should have 'Open', 'High', 'Low', 'Close' columns.
                                      It should represent the last TSTEPS data points.
        model_path (str): Path to the trained Keras model.
        scaler_params_path (str): Path to the JSON file containing scaler parameters (mean, std).

    Returns:
        np.array: Denormalized predicted prices.
    """
    # Load the trained model
    model = keras.models.load_model(model_path)

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
    # For prediction, we typically predict one sample at a time, so batch_size=1.
    # However, for stateful LSTMs, the batch_size must match the one used during training.
    # So, we need to reshape the single input sequence to match the training batch_size.
    # This means we'll have to pad the input with dummy data if batch_size > 1.
    # A more practical approach for stateful prediction is to reset states after each prediction
    # or predict in batches of 1 if the model is built with batch_size=1.
    # Given the R code's `batch_size=500`, we need to be careful.

    # Let's assume for now that we are predicting a single sequence,
    # and the model was trained with a batch_size that allows for this.
    # If the model was trained with batch_size > 1 and stateful=True,
    # predicting a single sample requires careful handling (e.g., model.reset_states()).
    # For simplicity in this initial port, let's assume the input is a single batch.

    # Reshape input for LSTM: (batch_size, TSTEPS, N_FEATURES)
    # The model was built with a fixed BATCH_SIZE.
    # For prediction, we need to provide input with that exact batch_size.
    # If we want to predict for a single sequence, we need to create a batch of BATCH_SIZE
    # where our actual sequence is the first one, and the rest are dummy.
    # This is not ideal for real-time prediction.

    # A better approach for stateful prediction with a model trained with BATCH_SIZE > 1
    # is to build a separate prediction model with batch_size=1, or reset states.
    # However, the R code implies using the same model for prediction.

    # Let's assume the input_data_df is already a single sequence of TSTEPS.
    # We need to expand its dimensions to (BATCH_SIZE, TSTEPS, N_FEATURES)
    # by repeating the single sequence BATCH_SIZE times.
    # This is a workaround for stateful models with fixed batch_size > 1.
    # The actual prediction will be from the first sequence.

    input_reshaped = np.tile(input_normalized.values[np.newaxis, :, :], (BATCH_SIZE, 1, 1))

    # Make prediction
    predictions_normalized = model.predict(input_reshaped, batch_size=BATCH_SIZE)

    # The model returns (BATCH_SIZE, TSTEPS, 1). We only care about the first sequence's prediction.
    # And the R code's `return_sequences = TRUE` means it predicts for each timestep in the sequence.
    # We are interested in the prediction for the last timestep, or the next step.
    # The R code's `Y <- Models[[model]] %>% predict(X)` returns `Y` with `dim(Y) {20500,5}`.
    # It then uses `Y.pred[,1]` which is the first column.
    # The `layer_dense(units = 1)` means the output is a single value per timestep.
    # So, `predictions_normalized` will be `(BATCH_SIZE, TSTEPS, 1)`.
    # We want the prediction for the last timestep of the first sequence.

    # Get the prediction for the first sequence, last timestep
    single_prediction_normalized = predictions_normalized[0, -1, 0]

    # Denormalize prediction
    # The R code denormalizes using `r*attr(XY_norm,'scaled:scale') + attr(XY_norm, 'scaled:center')`
    # This implies denormalizing the 'Open' price.
    # So, we use the mean and std of the 'Open' column.
    denormalized_prediction = single_prediction_normalized * std_vals['Open'] + mean_vals['Open']

    return denormalized_prediction

if __name__ == "__main__":
    # Example usage for prediction
    print("Running prediction example...")

    # Ensure a trained model and scaler params exist
    if not os.path.exists(MODEL_SAVE_PATH) or not os.path.exists(SCALER_PARAMS_JSON):
        print("Error: Trained model or scaler parameters not found. Please run train.py first.")
    else:
        # Create dummy input data for prediction
        # This should be the last TSTEPS of your actual data, normalized.
        # For this example, we'll create random data.
        dummy_input_data = pd.DataFrame(np.random.rand(TSTEPS, N_FEATURES) * 100 + 100,
                                        columns=['Open', 'High', 'Low', 'Close'])
        
        print(f"Input data for prediction:\n{dummy_input_data}")
        predicted_price = predict_future_prices(dummy_input_data)
        print(f"Predicted future price: {predicted_price}")
