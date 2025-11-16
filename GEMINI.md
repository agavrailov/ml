# Machine Learning LSTM for Time Series Prediction (Python Refactor)

This project has been refactored from its original R-based implementation to a modern Python-based architecture. It utilizes deep learning techniques with TensorFlow/Keras for time series prediction, specifically focusing on the Silver SPOT price against USD (XAGUSD). The core model employs a Long Short-Term Memory (LSTM) neural network.

## Project Overview

The goal of this project is to predict future silver prices using historical data. The refactored architecture emphasizes modularity, reproducibility, and maintainability, aligning with current MLOps best practices.

**Key Features:**
*   **Data Preparation:** Automated pipeline to convert minute-level data to hourly, and then normalize OHLC (Open, High, Low, Close) values.
*   **LSTM Model:** A single-layer LSTM neural network for time series forecasting.
*   **Training & Evaluation:** Scripts for training the model and saving its state.
*   **Configuration Management:** Centralized configuration for hyperparameters and paths.
*   **Testing:** Unit tests for data processing and model architecture.

**Key Technologies:**
*   **Python 3.x**
*   **Pandas:** For data manipulation and analysis.
*   **TensorFlow/Keras:** For building and training the deep learning model.
*   **NumPy:** For numerical operations.
*   **Pytest:** For unit testing.

## Setup and Installation

To set up and run this project, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd ml_lstm
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Project Structure

The project follows a modular structure:

```
ml_lstm/
├── .gitignore
├── ARCHITECTURE.md         # Detailed architectural design document
├── REFACTOR_PLAN.md        # Plan for the refactoring process
├── update_data.bat         # Script to run data ingestion and processing
├── data/
│   ├── raw/                # Raw, immutable input data (e.g., nvda_minute.csv)
│   └── processed/          # Processed data ready for modeling (e.g., training_data.csv)
├── notebooks/              # Jupyter notebooks for EDA and experimentation
├── src/
│   ├── data_ingestion.py   # Historical TWS ingestion CLI and wrapper around the ingestion core
│   ├── daily_data_agent.py # Orchestrates daily ingestion, gap handling, curation, and feature generation
│   ├── data_processing.py  # Logic for data loading, conversion, and normalization
│   ├── model.py            # LSTM model definition
│   ├── train.py            # Script for model training
│   ├── predict.py          # Script for making predictions (TODO)
│   └── config.py           # Centralized configuration for hyperparameters and paths
├── tests/
│   ├── test_data_processing.py # Unit tests for data processing
│   └── test_model.py       # Unit tests for model architecture and functionality
└── requirements.txt        # Python project dependencies
```

## Usage

### 1. Data Ingestion and Update

The project now includes an automated data ingestion and update pipeline. This process connects to Interactive Brokers (TWS/Gateway) to fetch minute-level data for NVDA, handles initial historical data fetching, continuous updates, and gap filling.

The recommended daily entrypoint is:

```bash
python src/daily_data_agent.py
```

On Windows, you can also use the helper batch script:

```bash
.\update_data.bat
```

With the current implementation, these will:
*   Connect to IB TWS/Gateway.
*   Fetch new minute-level data for NVDA from the last recorded timestamp up to the current time.
*   Identify and fill any historical gaps within market hours (excluding weekends and holidays).
*   Sort and deduplicate the entire raw dataset (`data/raw/nvda_minute.csv`).
*   Convert raw/curated minute data into hourly format and add features under `data/processed/`.

### 2. Data Preparation

After the data ingestion and update process, the `data/raw/nvda_minute.csv` file will contain the latest sorted and deduplicated minute-level data. The `update_data.bat` script automatically calls `src/data_processing.py` which will then convert this into hourly data and normalize it, saving the results in `data/processed/`.

### 3. Model Training

To train the LSTM model:

```bash
python src/train.py
```
This script will:
*   Load the processed data from `data/processed/`.
*   Build the LSTM model as defined in `src/model.py`.
*   Train the model using the hyperparameters specified in `src/config.py`.
*   Save the trained model to `models/my_lstm_model.keras`.

### 3. Making Predictions (TODO)

The prediction logic will be implemented in `src/predict.py`.

## Development

### Running Tests

To run the unit tests for the project:

```bash
# Ensure your virtual environment is active
pytest tests/
```

## Further Documentation

*   **`ARCHITECTURE.md`**: Provides a detailed explanation of the project's design principles, proposed structure, and technology choices.
*   **`REFACTOR_PLAN.md`**: Outlines the step-by-step plan followed during the refactoring process.

# Data Flow Description

The data flow in the refactored Python-based project is designed to be clear, modular, and reproducible, moving through distinct stages from raw input to processed features and finally to model predictions.

## 1. Raw Data Ingestion (`src/data_ingestion.py`)

*   **Purpose:** The `src/data_ingestion.py` script is responsible for connecting to the Interactive Brokers (IB) Trader Workstation (TWS) or IB Gateway, fetching historical minute-level stock data for a specified instrument (currently NVDA), and saving this raw data to a CSV file. It is primarily used for initial historical data fetching or fetching specific date ranges, while `src/data_updater.py` orchestrates the continuous update and gap-filling process.
*   **Key Components:**
    *   **`ib_insync` library:** This Python library provides a convenient and asynchronous way to interact with the TWS API.
    *   **`src/config.py`:** This file stores configuration parameters such as:
        *   `TWS_HOST`, `TWS_PORT`, `TWS_CLIENT_ID`: For connecting to TWS/IB Gateway.
        *   `TWS_MAX_CONCURRENT_REQUESTS`: Maximum concurrent requests to TWS.
        *   `DATA_BATCH_SAVE_SIZE`: Number of days to fetch before saving a batch to CSV.
        *   `NVDA_CONTRACT_DETAILS`: A dictionary defining the instrument (symbol, security type, exchange, currency).
        *   `RAW_DATA_CSV`: The path where the fetched raw data will be saved (`data/raw/nvda_minute.csv`).
    *   **`fetch_historical_data` function:** The core asynchronous function in `src/data_ingestion.py` that orchestrates the data fetching.
*   **Process Flow:**
    *   **Connection:** The `fetch_historical_data` function initializes an `IB` object from `ib_insync` and attempts to connect to TWS/IB Gateway using the configured `TWS_HOST`, `TWS_PORT`, and `TWS_CLIENT_ID`.
    *   **Contract Definition:** It constructs an `ib_insync` `Contract` object (e.g., `Stock` for NVDA) using the `NVDA_CONTRACT_DETAILS`.
    *   **Contract Qualification:** The contract is then "qualified" with IB to ensure its details are correct and unambiguous.
    *   **Parallelized Data Request & Batch Saving:** To optimize retrieval and manage API limits, the script fetches historical data in daily chunks. These daily requests are highly parallelized using `asyncio.gather`, respecting the `TWS_MAX_CONCURRENT_REQUESTS` limit. Fetched data is then saved to `data/raw/nvda_minute.csv` in batches of `DATA_BATCH_SAVE_SIZE` days, improving resilience and reducing I/O overhead.
    *   **Data Cleaning/Formatting:** The fetched bars are converted into a Pandas DataFrame, sorted by date, and columns are renamed to `DateTime`, `Open`, `High`, `Low`, `Close`. The `DateTime` column is formatted to `YYYY-MM-DDTHH:%M`.
    *   **Data Storage:** The processed DataFrame is then saved to `RAW_DATA_CSV` (`data/raw/nvda_minute.csv`). If the file already exists, new data is appended without writing the header again.
    *   **Disconnection:** The script attempts to disconnect from TWS/IB Gateway.
*   **Execution:** The script can be executed directly using `python -m src.data_ingestion` for initial full fetches, but for day-to-day operations you should prefer the daily pipeline agent described below.

## 1.5. Daily Data Pipeline and Gap Filling (`src/daily_data_agent.py`)

*   **Purpose:** The `src/daily_data_agent.py` module orchestrates daily ingestion, gap analysis/filling, curated-minute snapshotting, and resampling/feature engineering so that `data/raw/` and `data/processed/` stay in sync.
*   **Key Components:**
    *   **`src/ingestion/tws_historical.py`:** Implements the IB/TWS historical ingestion core, used by the agent to pull new minute data.
    *   **`src/ingestion/curated_minute.py`:** Provides `run_transform_minute_bars` and the curated-minute CSV snapshot.
    *   **`src/data_processing.py`:** Provides `clean_raw_minute_data`, `convert_minute_to_timeframe`, `prepare_keras_input_data`, and `fill_gaps`.
    *   **`analyze_gaps.py`:** Standalone script invoked by the agent to detect long weekday gaps in raw minute data.
    *   **`src/config.py`:** Provides paths, IB connection settings, and market parameters.
*   **Process Flow (high level):**
    *   **Ingest New Data:** Calls the ingestion core to append minute-level NVDA data since a configured start date into `data/raw/nvda_minute.csv`.
    *   **Clean & Deduplicate:** Runs `clean_raw_minute_data` on the raw CSV.
    *   **Gap Analysis & Filling:** Executes `analyze_gaps.py` to produce a gaps JSON, then applies `fill_gaps` to synthesize small missing intervals via forward fill.
    *   **Curated-Minute Snapshot:** Calls `run_transform_minute_bars` to write a curated-minute CSV under `data/processed/`.
    *   **Resample & Add Features:** Resamples curated minutes to the configured hourly frequency and engineers features for training.
*   **Execution:** This module is designed to be run periodically (e.g., via Task Scheduler or cron) using `python src/daily_data_agent.py`. It is also a good target for Windows helper scripts like `update_data.bat`.

## 2. Minute-to-Hourly Conversion (`src/data_processing.py`):

*   The `convert_minute_to_hourly` function in `src/data_processing.py` reads the raw minute-level data from `data/raw/nvda_minute.csv`.
*   It then resamples this data to an hourly frequency, aggregating the OHLC (Open, High, Low, Close) values (e.g., first for Open, max for High, min for Low, last for Close within each hour).
*   The resulting hourly data is saved as `nvda_hourly.csv` in the `data/processed/` directory.

## 3. Data Normalization (`src/data_processing.py`):

*   The `prepare_keras_input_data` function, also in `src/data_processing.py`, takes the hourly data (`data/processed/nvda_hourly.csv`) as input.
*   It calculates the mean and standard deviation for each OHLC column.
*   It then performs Z-score normalization on these OHLC values, transforming them to have a mean of 0 and a standard deviation of 1.
*   The normalized data is saved as `training_data.csv` in `data/processed/`.
*   Crucially, the calculated mean and standard deviation values (scaler parameters) are saved as `scaler_params.json` in `data/processed/`. These parameters are essential for denormalizing predictions back to their original scale.

## 4. Sequence Creation and Splitting (`src/train.py` and `src/retrain.py`):

*   The `train_model` function in `src/train.py` (and `retrain_model` in `src/retrain.py`) loads the `training_data.csv` (normalized hourly data) and `scaler_params.json`.
*   It splits the data into training and validation sets based on `TR_SPLIT` (e.g., 70% for training, 30% for validation).
*   The `create_sequences_for_stateful_lstm` function is then used to transform the dataframes into `X` (features) and `Y` (labels) numpy arrays. This involves:
    *   Creating sequences of `TSTEPS` length for `X`.
    *   Generating lagged labels `Y` by shifting the 'Open' price `ROWS_AHEAD` steps into the future.
    *   Ensuring that the number of samples in `X` and `Y` is divisible by the `BATCH_SIZE` for stateful LSTM training.

## 5. Model Training and Retraining (`src/train.py` and `src/retrain.py`):

*   The `train_model` function in `src/train.py` (and `retrain_model` in `src/retrain.py`) builds the LSTM model using `build_lstm_model` from `src/model.py`, configured with parameters from `src/config.py`.
*   The model is then trained (or retrained) using the prepared `X_train` and `Y_train` (and `X_val`, `Y_val` for validation) arrays.
*   **Model Versioning:** After training, the model is saved to a versioned path within `models/registry/` using a timestamped filename (e.g., `my_lstm_model_YYYYMMDD_HHMMSS.keras`).

## 6. Prediction (`src/predict.py`):

*   The `predict_future_prices` function in `src/predict.py` loads the *latest* trained model from the `models/registry/` using the `get_latest_model_path()` function from `src/config.py`, and the `scaler_params.json`.
*   It takes new input data (a DataFrame representing the last `TSTEPS` data points) for which a prediction is desired.
*   This input data is normalized using the loaded `mean` and `std` values.
*   The normalized input is reshaped to match the model's expected input shape (`BATCH_SIZE`, `TSTEPS`, `N_FEATURES`).
*   The model makes a prediction on this prepared input.
*   The normalized prediction is then denormalized back to the original price scale using the `mean` and `std` of the 'Open' price from `scaler_params.json`.
*   The final denormalized predicted price is returned.

This structured data flow ensures that each step is clearly defined, testable, and contributes to a robust and maintainable machine learning pipeline.
