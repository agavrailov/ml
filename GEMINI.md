# Project Overview

This project uses the R language with Keras and TensorFlow to predict silver spot prices (XAG/USD) using an LSTM (Long Short-Term Memory) neural network. The project is structured into data preparation, model training, and prediction components.

The data pipeline starts with minute-based price data, which is converted to hourly data. This hourly data is then normalized and used to train the LSTM model. The model architecture consists of a single LSTM layer with 100 neurons.

## Key Files

*   `src/main.R`: The main script that defines and orchestrates the neural network model. It includes functions for generating time series data, training the model, making predictions, and plotting the results.
*   `data_preparation/convert data to hourly.R`: This script converts minute-based historical data into an hourly format.
*   `data_preparation/Keras_input_data_prep.R`: This script takes the hourly data, normalizes the OHLC (Open, High, Low, Close) values, and saves the prepared data for training.
*   `data/training_data`: The output of the data preparation scripts, used as input for the model.
*   `models/MyModels`: The saved, trained Keras models.

# Building and Running

This is an R project. To run it, you will need to have R installed with the necessary libraries (`keras`, `xts`).

1.  **Data Preparation:**
    *   Run `data_preparation/convert data to hourly.R` to convert minute data to hourly data.
    *   Run `data_preparation/Keras_input_data_prep.R` to normalize the hourly data and create the `training_data` file.

2.  **Training and Prediction:**
    *   Run `src/main.R` to train the model, make predictions, and visualize the results. The script will also save the trained model to the `models` directory.

*TODO: Add specific commands to run the R scripts from the command line.*

# Development Conventions

*   The project uses a functional approach, with separate functions for different stages of the machine learning pipeline.
*   The main script `src/main.R` contains both function definitions and the main execution logic.
*   Constants for the model and training process are defined at the top of `src/main.R`.
*   The project includes a `.gitignore` file, which is a good practice for version control.

---

# Refactoring Overview

The project, originally an R-based solution for LSTM time series prediction of XAG/USD prices, has been completely refactored into a modern Python-based architecture. The primary motivation was to address issues of complexity, maintainability, and scalability inherent in the previous setup, bringing the project up to current software engineering and MLOps standards.

**What is New:**

1.  **Language Transition (R to Python):**
    *   The entire codebase has been migrated from R to Python, leveraging Python's robust ecosystem for machine learning.

2.  **Modular Architecture:**
    *   **Clear Separation of Concerns:** The monolithic `main.R` script has been broken down into distinct, modular Python files:
        *   `src/data_processing.py`: Handles all data loading, minute-to-hourly conversion, and normalization.
        *   `src/model.py`: Defines the Keras LSTM model architecture.
        *   `src/train.py`: Orchestrates the data preparation, model building, training, and saving.
        *   `src/predict.py`: Manages loading the trained model, preparing new data, and making predictions.
    *   **Standard Project Structure:** The project now adheres to a conventional Python project layout, including `src/`, `data/raw`, `data/processed`, `models/`, `tests/`, and `notebooks/` directories.

3.  **Improved Data Pipeline:**
    *   **Automated Data Flow:** The previously manual data preparation steps are now encapsulated in `src/data_processing.py`, allowing for a more automated and reproducible pipeline.
    *   **Pandas for Data Handling:** Replaced R's data manipulation with Python's Pandas library, offering powerful and flexible data processing capabilities.

4.  **Centralized Configuration:**
    *   **`src/config.py`:** All hyperparameters, file paths, and other configurable settings are now centralized in `src/config.py`, making the project easier to configure, experiment with, and deploy across different environments.

5.  **Robust Model Definition and Training:**
    *   **TensorFlow/Keras Functional API:** The LSTM model is now defined using Keras's Functional API, providing more flexibility and clarity compared to the R Keras wrapper.
    *   **Stateful LSTM Handling:** The data preparation for stateful LSTMs during training and prediction has been carefully implemented to ensure correct batching and sequence generation.

6.  **Comprehensive Testing:**
    *   **`pytest` Integration:** Unit tests have been introduced using `pytest` to verify the correctness of:
        *   `src/data_processing.py` (data conversion and normalization).
        *   `src/model.py` (model architecture, compilation, and forward pass).
    *   This significantly improves code quality and reduces the risk of regressions.

7.  **Enhanced Documentation:**
    *   **`README.md`:** Updated to provide a clear overview of the new Python project, setup instructions, and usage guidelines.
    *   **`ARCHITECTURE.md`:** A new document detailing the design principles, proposed structure, and technology choices of the modernized project.
    *   **`REFACTOR_PLAN.md`:** A new document outlining the step-by-step plan followed during the refactoring process.

8.  **Dependency Management:**
    *   **`requirements.txt`:** Explicitly lists all Python dependencies, ensuring a reproducible development environment.

In essence, the project has evolved from a research-oriented R script to a production-ready Python machine learning application, emphasizing best practices in code organization, testing, and MLOps.

---

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
    *   **Data Cleaning/Formatting:** The fetched bars are converted into a Pandas DataFrame, sorted by date, and columns are renamed to `DateTime`, `Open`, `High`, `Low`, `Close`. The `DateTime` column is formatted to `YYYY-MM-DDTHH:MM`.
    *   **Data Storage:** The processed DataFrame is then saved to `RAW_DATA_CSV` (`data/raw/nvda_minute.csv`). If the file already exists, new data is appended without writing the header again.
    *   **Disconnection:** The script attempts to disconnect from TWS/IB Gateway.
*   **Execution:** The script can be executed directly using `python src/data_ingestion.py` for initial full fetches, but is primarily called by `src/data_updater.py` for continuous operations.

## 1.5. Continuous Data Update and Gap Filling (`src/data_updater.py`)

*   **Purpose:** The `src/data_updater.py` script orchestrates the continuous updating and gap-filling of the `data/raw/nvda_minute.csv` dataset. It ensures the dataset remains current, complete, and free of gaps within market hours.
*   **Key Components:**
    *   **`src/data_ingestion.py`:** Utilizes the `fetch_historical_data` function to retrieve new and missing data.
    *   **`src/data_processing.py`:** Employs the `clean_raw_minute_data` function for final sorting and deduplication.
    *   **`exchange_calendars` library:** Used to accurately identify market trading days and holidays for gap analysis.
    *   **`src/config.py`:** Provides market-specific parameters like `MARKET_OPEN_TIME`, `MARKET_CLOSE_TIME`, `MARKET_TIMEZONE`, and `EXCHANGE_CALENDAR_NAME`.
*   **Process Flow:**
    *   **Load Existing Data:** Reads the current `data/raw/nvda_minute.csv` to establish the existing data range.
    *   **Fetch Recent Data:** Calls `fetch_historical_data` to retrieve any new minute-level data from the last recorded timestamp in the file up to the current moment.
    *   **Identify Gaps:** Iterates through the entire historical range, considering market trading days and hours (excluding weekends and holidays), to pinpoint any missing minute bars.
    *   **Fill Gaps:** For each identified gap, it makes targeted calls to `fetch_historical_data` to retrieve the missing data. These requests are also parallelized.
    *   **Merge and Clean:** After all new and missing data is fetched, the entire dataset is reloaded, merged, and then passed to `clean_raw_minute_data` for a final sort and deduplication, ensuring data integrity.
    *   **Save:** The fully updated, sorted, and deduplicated dataset is saved back to `data/raw/nvda_minute.csv`.
*   **Execution:** This script is designed to be run periodically (e.g., via a scheduled task or cron job) using `python src/data_updater.py`, typically orchestrated by `update_data.bat`.

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