import os
import sys

import pandas as pd
import json
import os
import numpy as np
import subprocess # Added for running external scripts

from src.config import (
    PROCESSED_DATA_DIR, RAW_DATA_CSV, FREQUENCY,
    FEATURES_TO_USE_OPTIONS
)

GAP_ANALYSIS_OUTPUT_JSON = os.path.join(PROCESSED_DATA_DIR, "missing_trading_days.json")

def convert_minute_to_timeframe(
    input_csv_path: str,
    frequency: str,
    processed_data_dir: str = PROCESSED_DATA_DIR,
) -> None:
    """Resample minute-level OHLC data to a coarser timeframe and save as CSV.

    Args:
        input_csv_path: Path to the input minute-level CSV file. Must contain a
            ``"DateTime"`` column parseable as a datetime index.
        frequency: Resampling rule (e.g. ``"60min"``, ``"30min"``) understood by
            ``pandas.DataFrame.resample``.
        processed_data_dir: Target directory where the resampled CSV will be
            written.
    """
    # Build output path from the provided processed_data_dir to keep tests and runtime aligned
    output_csv_path = os.path.join(processed_data_dir, f"nvda_{frequency}.csv")
    print(f"Converting minute data from {input_csv_path} to {frequency} frequency.")
    print(f"Output will be saved to {output_csv_path}")
    
    # Load input timeseries file
    # Assuming the CSV has columns like "DateTime", "Open", "High", "Low", "Close"
    # and "DateTime" is in "%Y-%m-%dT%H:%M" format.
    df = pd.read_csv(input_csv_path, parse_dates=['DateTime'], index_col='DateTime')
    print(f"Initial DataFrame shape after reading CSV: {df.shape}")

    # Convert to specified timeframe OHLC data
    df_resampled = df.resample(frequency).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    })
    print(f"DataFrame shape after resampling to {frequency}: {df_resampled.shape}")

    # Drop any rows that might have been created by resampling but have no data
    df_resampled.dropna(inplace=True)
    print(f"DataFrame shape after dropping NaNs: {df_resampled.shape}")

    if df_resampled.empty:
        print(f"Warning: Resampled DataFrame for {frequency} is empty after dropping NaNs. No file will be saved.")
        return


    # Reset index to make 'DateTime' a column again and rename it to 'Time'
    df_resampled = df_resampled.reset_index().rename(columns={'DateTime': 'Time'})
    
    # Ensure DateTime is in the correct format
    df_resampled['Time'] = df_resampled['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Save the resampled data to a new CSV file
    df_resampled.to_csv(output_csv_path, index=False)
    print(f"Successfully converted minute data to {frequency} and saved to {output_csv_path}")

def add_features(df: pd.DataFrame, features_to_generate: list[str]) -> pd.DataFrame:
    """Add technical indicators and time-based features to a price DataFrame.

    The function operates in-place on ``df`` and returns the same DataFrame for
    convenience.

    Args:
        df: Input DataFrame with at least ``"Time"`` and OHLC columns.
        features_to_generate: Names of features to generate (subset of
            ``["SMA_7", "SMA_21", "RSI", "Hour", "DayOfWeek"]``).

    Returns:
        The DataFrame with requested feature columns added and rows with NA
        values from rolling operations dropped.
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

def prepare_keras_input_data(
    input_hourly_csv_path: str,
    features_to_use: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Prepare hourly OHLC data for Keras by engineering and selecting features.

    Normalization is intentionally *not* performed here to avoid data leakage;
    it is handled later in the training pipeline.

    Args:
        input_hourly_csv_path: Path to the hourly-level CSV file (as produced by
            :func:`convert_minute_to_timeframe`).
        features_to_use: Names of features to keep for model input (e.g.
            ``["Open", "High", "Low", "Close", "SMA_7", "RSI", "Hour"]``).

    Returns:
        A tuple ``(df_filtered, feature_cols)`` where ``df_filtered`` contains a
        ``"Time"`` column and the selected features, and ``feature_cols`` is the
        list of feature column names actually present.
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
    Loads raw minute data, ensures a ``DateTime`` column, sorts it, removes duplicates,
    and saves it back.

    Args:
        input_csv_path (str): Path to the raw minute-level CSV file.
    """
    if not os.path.exists(input_csv_path):
        print(f"Warning: Raw data file not found at {input_csv_path}. Skipping cleaning.")
        return

    print(f"Cleaning raw minute data at {input_csv_path}...")
    df = pd.read_csv(input_csv_path)

    # Handle legacy files where the datetime column header is "index" rather than "DateTime".
    if 'DateTime' not in df.columns:
        if 'index' in df.columns:
            print("Detected 'index' column, renaming to 'DateTime'.")
            df.rename(columns={'index': 'DateTime'}, inplace=True)
        else:
            raise ValueError("Raw data CSV must contain a 'DateTime' or 'index' column.")

    # Ensure DateTime is parsed as datetime dtype
    df['DateTime'] = pd.to_datetime(df['DateTime'])

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

def fill_gaps(df, identified_gaps, max_fill_hours: int = 48):
    """Fill identified gaps in the raw minute data using a simple rule-based strategy.

    Strategy:
        * For each gap in ``identified_gaps`` whose duration is <= ``max_fill_hours``,
          we synthesize missing 1-minute bars between the gap start and end times
          and forward-fill the OHLC values.
        * Larger gaps are left untouched (they may correspond to holidays or
          extended outages) but are logged.

    Args:
        df: The raw minute-level DataFrame.
        identified_gaps: A list of dictionaries with ``start``, ``end`` and
            ``duration`` keys (as produced by ``analyze_gaps.py``).
        max_fill_hours: Maximum gap duration (in hours) to fill.

    Returns:
        The DataFrame with small gaps filled and re-saved in chronological order.
    """
    if not identified_gaps:
        print("No gaps identified for filling.")
        return df

    df = df.sort_values("DateTime").copy()
    df.set_index("DateTime", inplace=True)

    for gap in identified_gaps:
        try:
            start = pd.to_datetime(gap["start"])
            end = pd.to_datetime(gap["end"])
            duration = pd.to_timedelta(gap["duration"])
        except Exception as e:
            print(f"Warning: could not parse gap entry {gap}: {e}")
            continue

        if duration > pd.Timedelta(hours=max_fill_hours):
            print(f"Skipping gap from {start} to {end} (duration {duration}) – exceeds max_fill_hours={max_fill_hours}.")
            continue

        print(f"Filling gap from {start} to {end} (duration {duration}) via 1-minute forward fill.")

        # Build a complete minute-level index for the gap window.
        # We include both endpoints to avoid off-by-one issues, then rely on
        # sorting and ffill to propagate prior values.
        missing_index = pd.date_range(start=start, end=end, freq="1min")

        # Union the existing index with the gap index and sort.
        df = df.reindex(df.index.union(missing_index)).sort_index()
        # Ensure the index retains the "DateTime" name so that reset_index()
        # produces a "DateTime" column in the output CSV.
        df.index.name = "DateTime"

        # Forward-fill OHLC values across the gap. Volume/other columns, if any,
        # are **not** filled here to avoid fabricating volume.
        ohlc_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
        if ohlc_cols:
            df[ohlc_cols] = df[ohlc_cols].ffill()

    df = df.reset_index()
    return df

if __name__ == "__main__":
    # Ensure data/processed exists
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # Clean raw minute data before processing
    clean_raw_minute_data(RAW_DATA_CSV)

    # --- New: Analyze and potentially fill gaps ---
    print("Analyzing gaps in raw data...")
    # Run analyze_gaps.py as a subprocess
    analyze_gaps_script_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), 'analyze_gaps.py')
    )
    command = [
        sys.executable,  # Use the current python interpreter
        analyze_gaps_script_path,
        RAW_DATA_CSV,
        GAP_ANALYSIS_OUTPUT_JSON,
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