import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import json
import os
import numpy as np # Added numpy import

from src.config import (
    PROCESSED_DATA_DIR, HOURLY_DATA_CSV, TRAINING_DATA_CSV, SCALER_PARAMS_JSON, RAW_DATA_CSV
)

def convert_minute_to_hourly(input_csv_path, output_csv_path):
    """
    Converts minute-level OHLC data from a CSV file to hourly OHLC data.

    Args:
        input_csv_path (str): Path to the input minute-level CSV file.
        output_csv_path (str): Path to save the output hourly-level CSV file.
    """
    # Load input timeseries file
    # Assuming the CSV has columns like "DateTime", "Open", "High", "Low", "Close"
    # and "DateTime" is in "%Y-%m-%dT%H:%M" format.
    df = pd.read_csv(input_csv_path, parse_dates=['DateTime'], index_col='DateTime')

    # Convert to hourly OHLC data
    # 'first' for Open, 'max' for High, 'min' for Low, 'last' for Close
    df_hourly = df.resample('h').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    })

    # Drop any rows that might have been created by resampling but have no data
    df_hourly.dropna(inplace=True)

    # Reset index to make 'DateTime' a column again and rename it to 'Time'
    df_hourly = df_hourly.reset_index().rename(columns={'DateTime': 'Time'})
    
    # Ensure DateTime is in the correct format
    df_hourly['Time'] = df_hourly['Time'].dt.strftime('%Y-%m-%dT%H:%M')

    # Save the hourly data to a new CSV file
    df_hourly.to_csv(output_csv_path, index=False)
    print(f"Successfully converted minute data to hourly and saved to {output_csv_path}")

def add_features(df):
    """
    Adds technical indicators and time-based features to the DataFrame.
    """
    # Ensure 'Time' is a datetime object
    df['Time'] = pd.to_datetime(df['Time'])

    # --- Technical Indicators ---
    # Simple Moving Averages (SMA)
    df['SMA_7'] = df['Close'].rolling(window=7).mean()
    df['SMA_21'] = df['Close'].rolling(window=21).mean()

    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # --- Time-Based Features ---
    df['Hour'] = df['Time'].dt.hour
    df['DayOfWeek'] = df['Time'].dt.dayofweek

    # Drop rows with NaN values created by rolling indicators
    df.dropna(inplace=True)
    
    return df

def prepare_keras_input_data(input_hourly_csv_path):
    """
    Prepares data for Keras input by adding features.
    Normalization is handled separately in the training script to prevent data leakage.

    Args:
        input_hourly_csv_path (str): Path to the input hourly-level CSV file.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: DataFrame with added features.
            - list: List of feature column names.
    """
    df = pd.read_csv(input_hourly_csv_path)

    # Add features
    df_featured = add_features(df)

    # Select all feature columns for normalization
    # Exclude the 'Time' column as it's not a feature for the model
    feature_cols = [col for col in df_featured.columns if col != 'Time']
    
    return df_featured, feature_cols


if __name__ == "__main__":
    # Ensure data/processed exists
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # Process the actual raw data
    print(f"Processing raw minute data from {RAW_DATA_CSV}...")
    convert_minute_to_hourly(RAW_DATA_CSV, HOURLY_DATA_CSV)
    
    # Prepare features, normalization will be handled in training script
    df_featured, feature_cols = prepare_keras_input_data(HOURLY_DATA_CSV)
    print("Features added to hourly data. Normalization will be performed during model training.")
    
    # Optionally save the featured data before normalization for inspection if needed
    # df_featured.to_csv(os.path.join(PROCESSED_DATA_DIR, "featured_hourly_data.csv"), index=False)
    print("Data processing complete.")