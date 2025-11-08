# Architecture Document: ML LSTM Project Modernization

## 1. Overview

This document outlines the proposed modern architecture for the XAG/USD time series forecasting project. The goal is to refactor the existing R-based project into a more robust, scalable, and maintainable system using Python and industry-standard MLOps practices. The new architecture will be modular, automated, and reproducible.

## 2. Architectural Principles

*   **Modularity:** Each component of the ML pipeline (data processing, modeling, training, prediction) will be separated into its own module.
*   **Reproducibility:** The entire pipeline, from data preparation to model training, will be fully automated and reproducible. This includes managing dependencies and data versions.
*   **Configuration-driven:** All hyperparameters, file paths, and other settings will be managed in a central configuration file, not hardcoded in scripts.
*   **Testability:** The codebase will be designed to be easily testable, with a dedicated testing suite.

## 3. Proposed Architecture

The project will be migrated to a Python-based environment, leveraging its extensive ecosystem for machine learning and data science.

### 3.1. Core Technologies

*   **Language:** Python 3.x
*   **Data Manipulation:** Pandas
*   **ML Framework:** TensorFlow/Keras
*   **Dependency Management:** Pip with `requirements.txt`
*   **Testing:** Pytest
*   **Experiment Tracking (Optional but Recommended):** MLflow or Weights & Biases
*   **API Serving (Optional):** FastAPI or Flask

### 3.2. Project Structure

The project will be organized into the following directory structure:

```
ml_lstm/
├── .gitignore
├── ARCHITECTURE.md
├── REFACTOR_PLAN.md
├── data/
│   ├── raw/              # Raw, immutable data
│   └── processed/        # Processed data ready for modeling
├── notebooks/
│   └── exploratory_analysis.ipynb # For EDA and experimentation
├── src/
│   ├── data_processing.py # Data loading and transformation
│   ├── model.py           # Model definition
│   ├── train.py           # Script for model training
│   ├── predict.py         # Script for making predictions
│   └── config.py          # Central configuration
├── tests/
│   ├── test_data_processing.py
│   └── test_model.py
└── requirements.txt
```

### 3.3. Component Descriptions

*   **`data/`**: This directory is split into `raw` and `processed` subdirectories. The `raw` directory should be treated as immutable. The `processed` directory will contain the data that is ready to be fed into the model.
*   **`notebooks/`**: Jupyter notebooks for exploratory data analysis (EDA), visualization, and initial model prototyping.
*   **`src/`**: The main source code for the project.
    *   `data_processing.py`: Contains all logic for loading data from `data/raw`, converting it from minute to hourly, normalizing it, and saving it to `data/processed`.
    *   `model.py`: Defines the Keras LSTM model architecture.
    *   `train.py`: The main entry point for training the model. It will use the modules for data processing and modeling, and it will be driven by the configuration file.
    *   `predict.py`: A script to load a trained model and make predictions on new data.
    *   `config.py`: A file to store all configurations, such as hyperparameters, file paths, and other settings.
*   **`tests/`**: Contains all unit and integration tests for the project.
*   **`requirements.txt`**: A file that lists all Python dependencies for the project, ensuring a reproducible environment.

## 4. Data Flow

1.  Raw minute-level data is placed in the `data/raw` directory.
2.  The `train.py` script is executed.
3.  It calls the `data_processing.py` module to process the raw data and save the result in `data/processed`.
4.  The `train.py` script then loads the processed data, instantiates the model from `model.py`, and begins the training loop.
5.  (Optional) During training, metrics and model artifacts are logged to an experiment tracking tool like MLflow.
6.  The trained model is saved to a designated `models` directory (not shown in the structure, but could be added).

This architecture provides a solid foundation for a modern machine learning project, enabling better code organization, easier maintenance, and more reliable results.
