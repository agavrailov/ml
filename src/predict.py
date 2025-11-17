import os
import json
from typing import Optional

# Silence most TensorFlow C++ and Python-level logs (info/warning)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # 0=all,1=filter INFO,2=filter WARNING,3=filter ERROR

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging

# Reduce Python-level TF logging (e.g. retracing warnings)
tf.get_logger().setLevel(logging.ERROR)

from src.model import build_lstm_model, load_stateful_weights_into_non_stateful_model
from src.data_processing import add_features  # Import add_features
from src.data_utils import apply_standard_scaler
from src.config import (
    TSTEPS,
    N_FEATURES,
    BATCH_SIZE,
    FREQUENCY,
    PROCESSED_DATA_DIR,
    get_scaler_params_json_path,
    LSTM_UNITS,
    LEARNING_RATE,
    get_latest_best_model_path,
    N_LSTM_LAYERS,
    STATEFUL,
    FEATURES_TO_USE_OPTIONS,
)

def predict_future_prices(
    input_data_df: pd.DataFrame,
    frequency: str = FREQUENCY,
    tsteps: int = TSTEPS,
    n_features: int = N_FEATURES,
    lstm_units: int = LSTM_UNITS,
    n_lstm_layers: int = N_LSTM_LAYERS,
    stateful: bool = STATEFUL,
    optimizer_name: str = "rmsprop",
    loss_function: str = "mae",
    features_to_use: Optional[list[str]] = None,
) -> float:
    """Predict a future price using the latest best LSTM model.

    The function reuses the training-time scaler, rebuilds a non-stateful
    version of the best stateful model, and returns a single-step price
    prediction.

    Args:
        input_data_df: Raw OHLC input data. Must contain enough history to
            compute all engineered features and still provide ``tsteps`` rows.
        frequency: Resampling frequency for which the model was trained.
        tsteps: Number of timesteps in the model input.
        n_features: Number of features expected by the model.
        lstm_units: Number of LSTM units used by the model.
        n_lstm_layers: Number of LSTM layers in the model.
        stateful: Whether the original training model was stateful.
        optimizer_name: Optimizer name used to compile the prediction model.
        loss_function: Loss name used to compile the prediction model.
        features_to_use: Features to engineer and feed to the model. If ``None``
            the first entry from ``FEATURES_TO_USE_OPTIONS`` is used.

    Returns:
        The denormalized single-step price prediction.
    """
    if features_to_use is None:
        features_to_use = FEATURES_TO_USE_OPTIONS[0] # Default to the first option if not provided

    # Get the path to the best model for the current frequency and TSTEPS.
    (
        model_path,
        _bias_correction_path,
        features_to_use_trained,
        lstm_units_trained,
        n_lstm_layers_trained,
    ) = get_latest_best_model_path(target_frequency=frequency, tsteps=tsteps)

    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No best model found for frequency {frequency} and TSTEPS {tsteps}. "
            "Please train a model first."
        )

    stateful_model = keras.models.load_model(model_path)

    # Load best hyperparameters to get the correct lstm_units, n_lstm_layers, etc.
    best_hps = {}
    best_hps_path = "best_hyperparameters.json"
    if os.path.exists(best_hps_path):
        try:
            with open(best_hps_path, 'r') as f:
                content = f.read().strip()
                if content:
                    best_hps_data = json.loads(content)
                    if frequency in best_hps_data and str(tsteps) in best_hps_data[frequency]:
                        best_hps = best_hps_data[frequency][str(tsteps)]
        except json.JSONDecodeError:
            # If the file exists but is empty or malformed, fall back to
            # training-time defaults without failing prediction.
            best_hps = {}

    # Prefer tuned/trained hyperparameters when available.
    lstm_units = best_hps.get("lstm_units", lstm_units_trained or LSTM_UNITS)
    n_lstm_layers = best_hps.get("n_lstm_layers", n_lstm_layers_trained or N_LSTM_LAYERS)
    optimizer_name = best_hps.get("optimizer_name", "rmsprop")
    loss_function = best_hps.get("loss_function", "mae")

    # Prefer the features recorded with the best model if not explicitly given.
    if features_to_use is None and features_to_use_trained:
        features_to_use = features_to_use_trained

    # Build a non-stateful model for prediction and transfer weights
    prediction_model = build_lstm_model(
        input_shape=(tsteps, n_features),
        lstm_units=lstm_units,
        batch_size=None, # Non-stateful model does not require batch_size
        learning_rate=0.001, # Learning rate is not used for prediction model compilation
        n_lstm_layers=n_lstm_layers,
        stateful=False, # Always non-stateful for prediction
        optimizer_name=optimizer_name,
        loss_function=loss_function
    )
    load_stateful_weights_into_non_stateful_model(stateful_model, prediction_model)

    # Load scaler parameters for the current frequency.
    scaler_params_path = get_scaler_params_json_path(frequency)
    if not os.path.exists(scaler_params_path):
        raise FileNotFoundError(f"Scaler parameters not found at {scaler_params_path}.")
    
    with open(scaler_params_path, "r") as f:
        scaler_params = json.load(f)
    mean_vals = pd.Series(scaler_params["mean"])
    std_vals = pd.Series(scaler_params["std"])

    # Prepare input data
    input_data_df_copy = input_data_df.copy()
    df_featured_input = add_features(input_data_df_copy, features_to_use) # Pass features_to_use

    if len(df_featured_input) < tsteps:
        raise ValueError(f"Not enough data after feature engineering to form {tsteps} timesteps.")
    
    df_featured_input = df_featured_input.tail(tsteps)

    feature_cols = [col for col in df_featured_input.columns if col != 'Time']

    # Normalize the input features using the stored scaler
    input_normalized_df = apply_standard_scaler(df_featured_input, feature_cols, scaler_params)
    input_normalized_features = input_normalized_df[feature_cols]

    # Reshape for non-stateful LSTM: (1, TSTEPS, N_FEATURES)
    input_reshaped = input_normalized_features.values[np.newaxis, :, :]

    # Make prediction (silent)
    predictions_normalized = prediction_model.predict(input_reshaped, verbose=0)
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
        predicted_price = predict_future_prices(
            dummy_input_data,
            frequency=FREQUENCY,
            tsteps=TSTEPS,
            n_features=N_FEATURES,
            lstm_units=LSTM_UNITS,
            n_lstm_layers=N_LSTM_LAYERS,
            stateful=STATEFUL,
            optimizer_name='rmsprop', # Default for now
            loss_function='mae', # Default for now
            features_to_use=FEATURES_TO_USE_OPTIONS[0] # Default for now
        )
        print(f"Predicted future price: {predicted_price}")
