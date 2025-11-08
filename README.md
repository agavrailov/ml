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
├── data/
│   ├── raw/                # Raw, immutable input data (e.g., xagusd_minute.csv)
│   └── processed/          # Processed data ready for modeling (e.g., training_data.csv)
├── notebooks/              # Jupyter notebooks for EDA and experimentation
├── src/
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

### 1. Data Preparation

Place your raw minute-level data (e.g., `xagusd_minute.csv`) into the `data/raw/` directory. The `src/data_processing.py` script will then convert this into hourly data and normalize it, saving the results in `data/processed/`.

### 2. Model Training

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