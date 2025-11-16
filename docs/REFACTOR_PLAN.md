# Refactoring Plan: ML LSTM Project

This document outlines the tasks required to refactor the project from its current R-based implementation to the new Python-based architecture.

## Phase 1: Project Scaffolding and Setup

*   [ ] **Task 1.1: Create the new directory structure.**
    *   Create the `data/raw`, `data/processed`, `notebooks`, `src`, and `tests` directories.
*   [ ] **Task 1.2: Move existing data.**
    *   Move the raw data files (e.g., `xagusd_minute.csv`) to `data/raw`.
*   [ ] **Task 1.3: Create initial Python files.**
    *   Create empty files: `src/data_processing.py`, `src/model.py`, `src/train.py`, `src/predict.py`, `src/config.py`, and `requirements.txt`.

## Phase 2: Porting the Data Preparation Logic

*   [ ] **Task 2.1: Port `convert data to hourly.R`.**
    *   Implement the logic to convert minute data to hourly data in `src/data_processing.py` using the Pandas library.
*   [ ] **Task 2.2: Port `Keras_input_data_prep.R`.**
    *   Implement the data normalization logic in `src/data_processing.py`.
*   [ ] **Task 2.3: Add tests for data processing.**
    *   Write unit tests in `tests/test_data_processing.py` to verify the data conversion and normalization logic.

## Phase 3: Porting the Model and Training Logic

*   [ ] **Task 3.1: Port the Keras model.**
    *   Define the LSTM model in `src/model.py` using TensorFlow/Keras.
*   [ ] **Task 3.2: Port the training and prediction logic.**
    *   Implement the training loop from `src/main.R` in `src/train.py`.
    *   Implement the prediction logic in `src/predict.py`.
*   [ ] **Task 3.3: Centralize configuration.**
    *   Move all hardcoded hyperparameters and settings from the original scripts to `src/config.py`.
*   [ ] **Task 3.4: Add tests for the model.**
    *   Write unit tests in `tests/test_model.py` to verify the model's architecture and functionality.

## Phase 4: Finalization and Documentation

*   [ ] **Task 4.1: Create `requirements.txt`.**
    *   List all Python dependencies in the `requirements.txt` file.
*   [ ] **Task 4.2: Update `README.md`.**
    *   Update the main `README.md` to reflect the new architecture and provide instructions on how to set up and run the project.
*   [ ] **Task 4.3: (Optional) Add experiment tracking.**
    *   Integrate MLflow or Weights & Biases into the `train.py` script.

This plan provides a structured approach to modernizing the project. Each phase builds upon the previous one, ensuring a smooth transition.
