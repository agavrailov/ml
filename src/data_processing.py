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

def prepare_keras_input_data(input_hourly_csv_path, output_training_csv_path, output_scaler_params_path):
    """
    Prepares data for Keras input by normalizing OHLC values.

    Args:
        input_hourly_csv_path (str): Path to the input hourly-level CSV file.
        output_training_csv_path (str): Path to save the normalized training data CSV.
        output_scaler_params_path (str): Path to save the scaler parameters (mean, std).
    """
    df = pd.read_csv(input_hourly_csv_path)

    # Select OHLC columns for normalization
    ohlc_cols = ['Open', 'High', 'Low', 'Close']
    df_ohlc = df[ohlc_cols]

    # Calculate mean and standard deviation for normalization
    mean_vals = df_ohlc.mean()
    std_vals = df_ohlc.std()

    # Normalize OHLC values (Z-score normalization)
    df_normalized = (df_ohlc - mean_vals) / std_vals

    # Combine normalized OHLC with other columns (e.g., 'Time')
    df_processed = df.copy()
    df_processed[ohlc_cols] = df_normalized

    # Save normalized data
    df_processed.to_csv(output_training_csv_path, index=False)
    print(f"Successfully normalized data and saved to {output_training_csv_path}")

    # Save scaler parameters for denormalization during prediction
    scaler_params = {
        'mean': mean_vals.to_dict(),
        'std': std_vals.to_dict()
    }
    with open(output_scaler_params_path, 'w') as f:
        json.dump(scaler_params, f, indent=4)
    print(f"Scaler parameters saved to {output_scaler_params_path}")


if __name__ == "__main__":
    # Ensure data/processed exists
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # Process the actual raw data
    print(f"Processing raw minute data from {RAW_DATA_CSV}...")
    convert_minute_to_hourly(RAW_DATA_CSV, HOURLY_DATA_CSV)
    prepare_keras_input_data(HOURLY_DATA_CSV, TRAINING_DATA_CSV, SCALER_PARAMS_JSON)
    print("Data processing complete.")