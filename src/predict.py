import os
import json
from dataclasses import dataclass
from typing import Optional, List

# Silence most TensorFlow C++ and Python-level logs (info/warning)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # 0=all,1=filter INFO,2=filter WARNING,3=filter ERROR

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging

# Reduce Python-level TF logging (e.g. retracing warnings)
tf.get_logger().setLevel(logging.ERROR)

from src.model import (
    build_lstm_model,
    load_stateful_weights_into_non_stateful_model,
    load_model,
)
from src.data_processing import add_features  # Import add_features
from src.data_utils import apply_standard_scaler
from src.config import (
    TSTEPS,
    N_FEATURES,
    BATCH_SIZE,
    FREQUENCY,
    PROCESSED_DATA_DIR,
    MODEL_REGISTRY_DIR,
    get_scaler_params_json_path,
    LSTM_UNITS,
    LEARNING_RATE,
    get_latest_best_model_path,
    get_active_model_path,
    N_LSTM_LAYERS,
    STATEFUL,
    FEATURES_TO_USE_OPTIONS,
)


def _find_latest_registry_model_path(*, frequency: str, tsteps: int) -> Optional[str]:
    """Best-effort fallback to a saved model when metadata isn't available.

    When ``best_hyperparameters.json`` hasn't been updated (or is missing), we
    can still often run predictions if a model file exists in the registry.

    We only consider models that match both ``frequency`` and ``tsteps`` using
    the training naming convention:

        my_lstm_model_{frequency}_tsteps{tsteps}_YYYYMMDD_HHMMSS.keras

    Returns an absolute filesystem path or ``None`` if no candidate exists.
    """

    if not MODEL_REGISTRY_DIR or not os.path.isdir(MODEL_REGISTRY_DIR):
        return None

    prefix = f"my_lstm_model_{frequency}_tsteps{tsteps}_"
    exts = (".keras", ".h5")

    try:
        names = [
            n
            for n in os.listdir(MODEL_REGISTRY_DIR)
            if n.startswith(prefix) and n.lower().endswith(exts)
        ]
    except OSError:
        return None

    if not names:
        return None

    # Filenames embed a lexicographically sortable timestamp (YYYYMMDD_HHMMSS).
    names.sort()
    return os.path.abspath(os.path.join(MODEL_REGISTRY_DIR, names[-1]))


@dataclass
class PredictionContext:
    """Reusable context for batched LSTM predictions.

    Holds the non-stateful prediction model, scaler parameters, and feature
    metadata so we can run many predictions without rebuilding the model.

    Notes
    -----
    The model is trained to predict *forward log returns* on Open prices.
    ``bias_correction_mean_residual`` (when available) is also in log-return
    space and should be *added* to raw model outputs before mapping back to a
    price.
    """

    model: keras.Model
    scaler_params: dict
    mean_vals: Optional[pd.Series]
    std_vals: Optional[pd.Series]
    features_to_use: List[str]
    tsteps: int
    bias_correction_mean_residual: float = 0.0


def build_prediction_context(
    frequency: str,
    tsteps: int,
) -> PredictionContext:
    """Build and return a PredictionContext for a given (frequency, tsteps)."""

    script_dir = os.path.dirname(__file__)

    model_path = None
    bias_correction_path = None
    features_to_use_trained = None
    lstm_units_trained = None
    n_lstm_layers_trained = None

    # Try to get the latest best model and its associated parameters
    (
        best_model_path_candidate,
        best_bias_correction_path_candidate,
        best_features_to_use_trained_candidate,
        best_lstm_units_trained_candidate,
        best_n_lstm_layers_trained_candidate,
    ) = get_latest_best_model_path(target_frequency=frequency, tsteps=tsteps)

    if best_model_path_candidate:
        abs_best_model_path = os.path.abspath(os.path.join(script_dir, best_model_path_candidate))
        if os.path.exists(abs_best_model_path):
            model_path = abs_best_model_path
            bias_correction_path = best_bias_correction_path_candidate
            features_to_use_trained = best_features_to_use_trained_candidate
            lstm_units_trained = best_lstm_units_trained_candidate
            n_lstm_layers_trained = best_n_lstm_layers_trained_candidate

    # Fallback to active model if no best model found or path doesn't exist
    if not model_path:
        active_model_path_candidate = get_active_model_path(frequency=frequency, tsteps=tsteps)
        if active_model_path_candidate:
            abs_active_model_path = os.path.abspath(os.path.join(script_dir, active_model_path_candidate))
            if os.path.exists(abs_active_model_path):
                model_path = abs_active_model_path
                # When using active model, the trained parameters are not available,
                # so they should default to None and be picked up from best_hps or global defaults.
                features_to_use_trained = None
                lstm_units_trained = None
                n_lstm_layers_trained = None
                bias_correction_path = None  # tied to best model metadata

    # Final fallback: look for the newest matching model in the registry.
    if not model_path:
        registry_model_path = _find_latest_registry_model_path(frequency=frequency, tsteps=tsteps)
        if registry_model_path and os.path.exists(registry_model_path):
            model_path = registry_model_path
            features_to_use_trained = None
            lstm_units_trained = None
            n_lstm_layers_trained = None
            bias_correction_path = None

    if not model_path or not os.path.exists(model_path):
        # Message includes the legacy substring "No best model found for frequency"
        # so that existing tests can detect and skip when no trained model is
        # available yet.
        raise FileNotFoundError(
            f"No best model found for frequency {frequency} (no best model or active model found for TSTEPS {tsteps}). "
            "Please train a model or update models/active_model.txt."
        )

    # Load the trained *stateful* model via the unified model loader.
    stateful_model = load_model(model_path)

    best_hps: dict = {}
    # Ensure best_hps_path is an absolute path for robust checking
    best_hps_path = os.path.abspath(os.path.join(script_dir, "best_hyperparameters.json"))

    if os.path.exists(best_hps_path):
        try:
            with open(best_hps_path, 'r') as f:
                content = f.read().strip()
                if content:
                    best_hps_data = json.loads(content)
                    if frequency in best_hps_data and str(tsteps) in best_hps_data[frequency]:
                        best_hps = best_hps_data[frequency][str(tsteps)]
        except json.JSONDecodeError:
            best_hps = {}

    lstm_units = best_hps.get("lstm_units", lstm_units_trained or LSTM_UNITS)
    n_lstm_layers = best_hps.get("n_lstm_layers", n_lstm_layers_trained or N_LSTM_LAYERS)
    optimizer_name = best_hps.get("optimizer_name", "rmsprop")
    loss_function = best_hps.get("loss_function", "mae")

    if features_to_use_trained:
        features_to_use = features_to_use_trained
    else:
        features_to_use = FEATURES_TO_USE_OPTIONS[0]

    n_features = len(features_to_use)
    prediction_model = build_lstm_model(
        input_shape=(tsteps, n_features),
        lstm_units=lstm_units,
        batch_size=None,
        learning_rate=0.001,
        n_lstm_layers=n_lstm_layers,
        stateful=False,
        optimizer_name=optimizer_name,
        loss_function=loss_function,
    )
    load_stateful_weights_into_non_stateful_model(stateful_model, prediction_model)

    scaler_params_path = get_scaler_params_json_path(frequency)
    if not os.path.exists(scaler_params_path):
        raise FileNotFoundError(f"Scaler parameters not found at {scaler_params_path}.")

    with open(scaler_params_path, "r") as f:
        scaler_params = json.load(f)
    mean_vals = pd.Series(scaler_params["mean"])
    std_vals = pd.Series(scaler_params["std"])

    bias_mean_residual = 0.0
    if bias_correction_path and os.path.exists(bias_correction_path):
        try:
            with open(bias_correction_path, "r") as f:
                data = json.load(f)
            bias_mean_residual = float(data.get("mean_residual", 0.0))
        except Exception:
            bias_mean_residual = 0.0

    return PredictionContext(
        model=prediction_model,
        scaler_params=scaler_params,
        mean_vals=mean_vals,
        std_vals=std_vals,
        features_to_use=features_to_use,
        tsteps=tsteps,
        bias_correction_mean_residual=bias_mean_residual,
    )


def predict_sequence_batch(
    ctx: PredictionContext,
    df_featured: pd.DataFrame,
) -> np.ndarray:
    """Vectorized prediction over all valid sliding windows.

    Returns an array of normalized predictions aligned to the *end* of each
    window. If there are N rows and tsteps=T, the result has length
    max(N - T + 1, 0).
    """

    feature_cols = [c for c in df_featured.columns if c != "Time"]
    values = df_featured[feature_cols].to_numpy(dtype=np.float32)
    n = len(values)
    tsteps = ctx.tsteps
    if n < tsteps:
        return np.array([], dtype=np.float32)

    n_windows = n - tsteps + 1
    X = np.stack(
        [values[i : i + tsteps] for i in range(n_windows)],
        axis=0,
    )

    preds = ctx.model.predict(X, batch_size=256, verbose=0)
    return preds.reshape(-1)


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

    The underlying LSTM is trained to predict *forward log returns* on Open
    prices rather than absolute prices. This wrapper runs feature engineering +
    scaling on the input data, obtains a single-step log-return prediction for
    the last available window, and maps it back to a price using the most
    recent Open price:

        price_pred = Open_t * exp(r_t_pred).

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
        features_to_use = FEATURES_TO_USE_OPTIONS[0]  # Default feature set if not provided

    # Build a reusable prediction context (model + scaler + feature metadata).
    ctx = build_prediction_context(frequency=frequency, tsteps=tsteps)

    # Prepare input data and features.
    input_data_df_copy = input_data_df.copy()
    df_featured_input = add_features(input_data_df_copy, ctx.features_to_use)

    if len(df_featured_input) < tsteps:
        raise ValueError(f"Not enough data after feature engineering to form {tsteps} timesteps.")

    # Keep only the last tsteps rows for a single-window prediction.
    df_featured_input = df_featured_input.tail(tsteps)

    # Normalize using the stored scaler.
    feature_cols = [col for col in df_featured_input.columns if col != 'Time']
    input_normalized_df = apply_standard_scaler(df_featured_input, feature_cols, ctx.scaler_params)
    input_normalized_features = input_normalized_df[feature_cols]

    # Shape (1, TSTEPS, N_FEATURES)
    input_reshaped = input_normalized_features.values[np.newaxis, :, :]

    # Model outputs a log-return prediction for the horizon defined by ROWS_AHEAD.
    predictions_log = ctx.model.predict(input_reshaped, verbose=0)
    single_log_return = float(predictions_log[0, 0])

    # Map log-return back to a price using the last raw Open price.
    base_open = float(df_featured_input['Open'].iloc[-1])
    predicted_price = base_open * float(np.exp(single_log_return))

    return predicted_price

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
