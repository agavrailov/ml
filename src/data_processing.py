import pandas as pd
import json
import os

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
    df_hourly = df.resample('H').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    })

    # Drop any rows that might have been created by resampling but have no data
    df_hourly.dropna(inplace=True)

    # Reset index to make 'DateTime' a column again and rename it to 'Time'
    df_hourly = df_hourly.reset_index().rename(columns={'DateTime': 'Time'})

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
    # --- Test convert_minute_to_hourly function ---
    dummy_data = {
        'DateTime': pd.to_datetime(['2023-01-01T00:00', '2023-01-01T00:01', '2023-01-01T00:02', '2023-01-01T00:03',
                                    '2023-01-01T00:59', '2023-01-01T01:00', '2023-01-01T01:01', '2023-01-01T01:02',
                                    '2023-01-01T01:59', '2023-01-01T02:00']),
        'Open': np.random.rand(10) * 100 + 100,
        'High': np.random.rand(10) * 100 + 101,
        'Low': np.random.rand(10) * 100 + 99,
        'Close': np.random.rand(10) * 100 + 100
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_input_minute_path = "data/raw/xagusd_minute.csv"
    os.makedirs("data/raw", exist_ok=True)
    dummy_df.to_csv(dummy_input_minute_path, index=False)
    print(f"Created dummy minute data at {dummy_input_minute_path}")

    hourly_output_path = "data/processed/xagusd_hourly.csv"
    convert_minute_to_hourly(dummy_input_minute_path, hourly_output_path)

    # --- Test prepare_keras_input_data function ---
    training_output_path = "data/processed/training_data.csv"
    scaler_params_path = "data/processed/scaler_params.json"
    prepare_keras_input_data(hourly_output_path, training_output_path, scaler_params_path)