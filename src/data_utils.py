import numpy as np
import pandas as pd
from typing import Tuple, List, Dict


def fit_standard_scaler(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[pd.Series, pd.Series, Dict[str, dict]]:
    """Fit a simple mean/std scaler on the given feature columns.

    Any zero standard deviations are replaced with 1.0 to avoid division by
    zero during normalization.

    Returns both the raw ``mean``/``std`` series and a JSON-serializable
    dictionary.
    """
    mean_vals = df[feature_cols].mean()
    std_vals = df[feature_cols].std()

    # Avoid division by zero
    std_vals = std_vals.replace(0, 1)

    scaler_params = {
        "mean": mean_vals.to_dict(),
        "std": std_vals.to_dict(),
    }
    return mean_vals, std_vals, scaler_params


def apply_standard_scaler(
    df: pd.DataFrame,
    feature_cols: List[str],
    scaler_params: Dict[str, dict],
) -> pd.DataFrame:
    """Apply a previously-fitted mean/std scaler to the given DataFrame.

    ``scaler_params`` must contain ``"mean"`` and ``"std"`` mappings from
    feature name to value.
    """
    mean_vals = pd.Series(scaler_params["mean"])
    std_vals = pd.Series(scaler_params["std"])

    df_norm = df.copy()
    df_norm[feature_cols] = (df[feature_cols] - mean_vals[feature_cols]) / std_vals[feature_cols]
    return df_norm


def compute_log_return_labels(prices: np.ndarray, rows_ahead: int) -> np.ndarray:
    """Compute forward log-return labels for a 1D price series.

    For each index ``t`` where ``t + rows_ahead`` exists, we define::

        r_t = log(price_{t+rows_ahead}) - log(price_t)

    The last ``rows_ahead`` positions are set to ``NaN`` because they lack a
    future price. The returned array has the same length as ``prices``.
    """
    prices = np.asarray(prices, dtype=float)

    log_returns = np.full_like(prices, np.nan, dtype=float)
    if rows_ahead < len(prices):
        log_returns[:-rows_ahead] = np.log(prices[rows_ahead:]) - np.log(prices[:-rows_ahead])

    return log_returns


def get_effective_data_length(
    data: pd.DataFrame,
    sequence_length: int,
    rows_ahead: int,
) -> int:
    """Return how many LSTM sequences can be formed from a feature DataFrame.

    This takes into account the look-back ``sequence_length`` and label shift
    ``rows_ahead`` (for predicting future values).
    """
    # Use all columns except 'Time' as features
    feature_cols = [col for col in data.columns if col != "Time"]
    features_array = data[feature_cols].values
    labels_array = data["Open"].values

    shifted_labels = np.full_like(labels_array, np.nan, dtype=float)
    if rows_ahead < len(labels_array):
        shifted_labels[:-rows_ahead] = labels_array[rows_ahead:]

    valid_indices = ~np.isnan(shifted_labels)

    # The number of valid feature rows after considering shifted labels
    num_valid_feature_rows = np.sum(valid_indices)

    # The number of sequences that can be formed
    if num_valid_feature_rows < sequence_length:
        return 0
    return int(num_valid_feature_rows - sequence_length + 1)


def create_sequences_for_stateless_lstm(
    data: pd.DataFrame,
    sequence_length: int,
    rows_ahead: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create ``(X, y)`` sequences suitable for a stateless LSTM model.

    Args:
        data: DataFrame with normalized feature columns and a ``"Time"`` column.
        sequence_length: Length of the input sequence (TSTEPS).
        rows_ahead: Number of rows ahead that the prediction target is taken.

    Returns:
        A tuple ``(X, y)`` where ``X`` has shape ``(n_samples, sequence_length,
        n_features)`` and ``y`` has shape ``(n_samples, 1)``.
    """
    feature_cols = [col for col in data.columns if col != "Time"]
    features_array = data[feature_cols].values
    labels_array = data["Open"].values

    shifted_labels = np.full_like(labels_array, np.nan, dtype=float)
    if rows_ahead < len(labels_array):
        shifted_labels[:-rows_ahead] = labels_array[rows_ahead:]

    valid_indices = ~np.isnan(shifted_labels)
    features_array = features_array[valid_indices]
    shifted_labels = shifted_labels[valid_indices]

    if len(features_array) < sequence_length:
        return np.array([]), np.array([])

    X, Y = [], []
    for i in range(len(features_array) - sequence_length + 1):
        X.append(features_array[i : i + sequence_length])
        Y.append(shifted_labels[i + sequence_length - 1])

    X_arr = np.array(X)
    Y_arr = np.array(Y)

    if Y_arr.ndim == 1:
        Y_arr = Y_arr.reshape(-1, 1)

    return X_arr, Y_arr


def create_sequences_for_stateful_lstm(
    data: pd.DataFrame,
    sequence_length: int,
    batch_size: int,
    rows_ahead: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create ``(X, y)`` sequences for a stateful LSTM, respecting batch size.

    The number of samples is truncated so that it is divisible by ``batch_size``
    while preserving temporal ordering.
    """
    feature_cols = [col for col in data.columns if col != "Time"]
    features_array = data[feature_cols].values
    labels_array = data["Open"].values

    shifted_labels = np.full_like(labels_array, np.nan, dtype=float)
    if rows_ahead < len(labels_array):
        shifted_labels[:-rows_ahead] = labels_array[rows_ahead:]

    valid_indices = ~np.isnan(shifted_labels)
    features_array = features_array[valid_indices]
    shifted_labels = shifted_labels[valid_indices]

    num_sequences = len(features_array) - sequence_length + 1
    if num_sequences <= 0:
        return np.array([]), np.array([])

    num_batches = num_sequences // batch_size
    effective_num_sequences = num_batches * batch_size

    if effective_num_sequences == 0:
        return np.array([]), np.array([])

    X, Y = [], []
    for i in range(effective_num_sequences):
        X.append(features_array[i : i + sequence_length])
        Y.append(shifted_labels[i + sequence_length - 1])

    X_arr = np.array(X)
    Y_arr = np.array(Y)

    if Y_arr.ndim == 1:
        Y_arr = Y_arr.reshape(-1, 1)

    return X_arr, Y_arr
