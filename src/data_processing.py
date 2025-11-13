import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import json
import os
import numpy as np
import subprocess # Added for running external scripts

from src.config import (
    PROCESSED_DATA_DIR, RAW_DATA_CSV, FREQUENCY, get_hourly_data_csv_path,
    FEATURES_TO_USE_OPTIONS
)

GAP_ANALYSIS_OUTPUT_JSON = os.path.join(PROCESSED_DATA_DIR, "missing_trading_days.json")

def convert_minute_to_timeframe(input_csv_path, frequency, processed_data_dir=PROCESSED_DATA_DIR):
    """
    Converts minute-level OHLC data from a CSV file to a specified timeframe OHLC data.

    Args:
        input_csv_path (str): Path to the input minute-level CSV file.
        frequency (str): The resampling frequency (e.g., '60min', '30min').
        processed_data_dir (str): The directory where processed data will be saved.
    """
    output_csv_path = get_hourly_data_csv_path(frequency, processed_data_dir)
    
    # Load input timeseries file
    # Assuming the CSV has columns like "DateTime", "Open", "High", "Low", "Close"
    # and "DateTime" is in "%Y-%m-%dT%H:%M" format.
    df = pd.read_csv(input_csv_path, parse_dates=['DateTime'], index_col='DateTime')

    # Convert to specified timeframe OHLC data
    df_resampled = df.resample(frequency).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    })

    # Drop any rows that might have been created by resampling but have no data
    df_resampled.dropna(inplace=True)

    # Reset index to make 'DateTime' a column again and rename it to 'Time'
    df_resampled = df_resampled.reset_index().rename(columns={'DateTime': 'Time'})
    
    # Ensure DateTime is in the correct format
    df_resampled['Time'] = df_resampled['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Save the resampled data to a new CSV file
    df_resampled.to_csv(output_csv_path, index=False)
    print(f"Successfully converted minute data to {frequency} and saved to {output_csv_path}")

def add_features(df, features_to_generate):
    """
    Adds technical indicators and time-based features to the DataFrame based on the provided list.
    """
    # Ensure 'Time' is a datetime object
    df['Time'] = pd.to_datetime(df['Time'])

    # --- Technical Indicators ---
    if 'SMA_7' in features_to_generate:
        df['SMA_7'] = df['Close'].rolling(window=7).mean()
    if 'SMA_21' in features_to_generate:
        df['SMA_21'] = df['Close'].rolling(window=21).mean()

    if 'RSI' in features_to_generate:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

    # --- Time-Based Features ---
    if 'Hour' in features_to_generate:
        df['Hour'] = df['Time'].dt.hour
    if 'DayOfWeek' in features_to_generate:
        df['DayOfWeek'] = df['Time'].dt.dayofweek # Monday=0, Sunday=6

    # Drop rows with NaN values created by rolling indicators
    df.dropna(inplace=True)
    
    return df

def prepare_keras_input_data(input_hourly_csv_path, features_to_use):
    """
    Prepares data for Keras input by adding features and selecting only the specified ones.
    Normalization is handled separately in the training script to prevent data leakage.

    Args:
        input_hourly_csv_path (str): Path to the input hourly-level CSV file.
        features_to_use (list): A list of feature names to be used by the model.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: DataFrame with selected features.
            - list: List of selected feature column names.
    """
    df = pd.read_csv(input_hourly_csv_path)

    # Define all possible features that can be generated
    all_possible_features_to_generate = [
        'Open', 'High', 'Low', 'Close', 'SMA_7', 'SMA_21', 'RSI', 'Hour', 'DayOfWeek'
    ]
    
    # Add all possible features to the DataFrame
    df_featured = add_features(df, all_possible_features_to_generate)

    # Filter to only include the features specified in features_to_use
    # Ensure 'Time' is not included in features_to_use if it's not meant to be a model feature
    final_features = [f for f in features_to_use if f in df_featured.columns]
    df_filtered = df_featured[['Time'] + final_features] # Keep 'Time' for indexing/merging if needed

    return df_filtered, final_features


def clean_raw_minute_data(input_csv_path):
    """
    Loads raw minute data, sorts it by DateTime, removes duplicates, and saves it back.

    Args:
        input_csv_path (str): Path to the raw minute-level CSV file.
    """
    if not os.path.exists(input_csv_path):
        print(f"Warning: Raw data file not found at {input_csv_path}. Skipping cleaning.")
        return

    print(f"Cleaning raw minute data at {input_csv_path}...")
    df = pd.read_csv(input_csv_path, parse_dates=['DateTime'])
    
    # Sort by DateTime
    df.sort_values('DateTime', inplace=True)
    
    # Drop duplicates based on DateTime
    initial_rows = len(df)
    df.drop_duplicates(subset=['DateTime'], inplace=True)
    if len(df) < initial_rows:
        print(f"Removed {initial_rows - len(df)} duplicate rows.")
    
    # Save cleaned data back
    df.to_csv(input_csv_path, index=False)
    print(f"Raw minute data cleaned and saved to {input_csv_path}")

def fill_gaps(df, identified_gaps):
    """
    Placeholder function to fill identified gaps in the raw minute data.
    Currently, it does not perform any filling but serves as an integration point.

    Args:
        df (pd.DataFrame): The raw minute-level DataFrame.
        identified_gaps (list): A list of dictionaries, each describing a gap.

    Returns:
        pd.DataFrame: The DataFrame, potentially with gaps filled.
    """
    if identified_gaps:
        print(f"Identified {len(identified_gaps)} gaps for potential filling. No filling performed yet.")
        # Future implementation will go here based on user's chosen strategy
    else:
        print("No gaps identified for filling.")
    return df

if __name__ == "__main__":
    # Ensure data/processed exists
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # Clean raw minute data before processing
    clean_raw_minute_data(RAW_DATA_CSV)

    # --- New: Analyze and potentially fill gaps ---
    print("Analyzing gaps in raw data...")
    # Run analyze_gaps.py as a subprocess
    analyze_gaps_script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'analyze_gaps.py'))
    command = [
        sys.executable, # Use the current python interpreter
        analyze_gaps_script_path,
        RAW_DATA_CSV,
        GAP_ANALYSIS_OUTPUT_JSON
    ]
    
    identified_gaps = []
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Gap analysis script output: {result.stdout}")
        if result.stderr:
            print(f"Gap analysis script errors: {result.stderr}")
        
        if os.path.exists(GAP_ANALYSIS_OUTPUT_JSON):
            with open(GAP_ANALYSIS_OUTPUT_JSON, 'r') as f:
                identified_gaps = json.load(f)
    except subprocess.CalledProcessError as e:
        print(f"Error running gap analysis script: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
    except FileNotFoundError:
        print(f"Error: analyze_gaps.py not found at {analyze_gaps_script_path}. Please ensure the script exists.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {GAP_ANALYSIS_OUTPUT_JSON}. File might be empty or corrupted.")


    # Load the raw data to pass to fill_gaps
    raw_df_for_filling = pd.read_csv(RAW_DATA_CSV, parse_dates=['DateTime'])
    processed_df = fill_gaps(raw_df_for_filling, identified_gaps)
    
    # Save the processed_df back to RAW_DATA_CSV if fill_gaps actually modified it
    # For now, since fill_gaps does nothing, this step is not strictly necessary but good for future proofing
    processed_df.to_csv(RAW_DATA_CSV, index=False)
    # --- End New ---

    # Process the actual raw data (now potentially with gaps filled)
    print(f"Processing raw minute data from {RAW_DATA_CSV}...")
    convert_minute_to_timeframe(RAW_DATA_CSV, FREQUENCY, PROCESSED_DATA_DIR)
    
    # Prepare features, normalization will be handled in training script
    # Use a default set of features for standalone execution
    from src.config import FEATURES_TO_USE_OPTIONS
    default_features_to_use = FEATURES_TO_USE_OPTIONS[0] # Use the first set of features as default
    df_featured, feature_cols = prepare_keras_input_data(get_hourly_data_csv_path(FREQUENCY, PROCESSED_DATA_DIR), default_features_to_use)
    print(f"Features added to hourly data: {feature_cols}. Normalization will be performed during model training.")
    
    # Optionally save the featured data before normalization for inspection if needed
    # df_featured.to_csv(os.path.join(PROCESSED_DATA_DIR, "featured_hourly_data.csv"), index=False)
    print("Data processing complete.")