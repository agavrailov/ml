"""Centralized data access helpers for the model pipeline.

This module provides a small, explicit API around the existing data
processing utilities so that training, evaluation, and backtests can
share the same code path for loading hourly OHLC data and engineered
features.

The goal is to *wrap* existing behaviour from :mod:`src.data_processing`
without changing notebooks or CLIs.
"""

from __future__ import annotations

from typing import Tuple, List

import pandas as pd

from src.config import get_hourly_data_csv_path
from src.data_processing import prepare_keras_input_data
from src.data_quality import validate_hourly_ohlc, validate_feature_frame

__all__ = [
    "load_hourly_ohlc",
    "load_hourly_features",
]


def load_hourly_ohlc(frequency: str) -> pd.DataFrame:
    """Load resampled OHLC data for the given frequency.

    This uses :func:`src.config.get_hourly_data_csv_path` to resolve the
    CSV location (e.g. ``data/processed/nvda_15min.csv``) and parses the
    ``"Time"`` column as a datetime index column.
    """

    csv_path = get_hourly_data_csv_path(frequency)
    df = pd.read_csv(csv_path, parse_dates=["Time"])
    validate_hourly_ohlc(df, context=f"load_hourly_ohlc[{frequency}]")
    return df


def load_hourly_features(
    frequency: str,
    features_to_use: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Load hourly OHLC data and engineer features for a given frequency.

    This is a thin wrapper around :func:`prepare_keras_input_data` that
    first resolves the hourly CSV path via config and then prepares the
    feature frame used by the model.

    Parameters
    ----------
    frequency:
        Resampling frequency string, e.g. ``"15min"`` or ``"60min"``.
    features_to_use:
        List of feature names to keep in the final frame.

    Returns
    -------
    df_filtered:
        DataFrame containing a ``"Time"`` column plus the requested
        feature columns.
    feature_cols:
        The list of feature column names actually present.
    """

    csv_path = get_hourly_data_csv_path(frequency)
    df_filtered, feature_cols = prepare_keras_input_data(csv_path, features_to_use)
    validate_feature_frame(
        df_filtered,
        feature_cols,
        context=f"load_hourly_features[{frequency}]",
    )
    return df_filtered, feature_cols
