# Project Roadmap: ML LSTM for Time Series Prediction

This roadmap outlines the planned phases for developing and operationalizing the ML LSTM project, building upon the completed refactoring to a Python-based architecture (Phase 1).

## Phase 1: Core ML Pipeline (Completed)

**Objective:** Establish a modular, testable, and reproducible Python-based machine learning pipeline for LSTM time series prediction.

**Key Achievements:**
*   **Language Transition:** Full migration from R to Python.
*   **Modular Codebase:** Separation of concerns into `src/data_processing.py`, `src/model.py`, `src/train.py`, `src/predict.py`.
*   **Data Pipeline:** Implemented minute-to-hourly conversion and normalization using Pandas.
*   **Model Definition:** Keras LSTM model defined using the Functional API.
*   **Configuration Management:** Centralized hyperparameters and paths in `src/config.py`.
*   **Unit Testing:** Comprehensive tests for data processing and model components using `pytest`.
*   **Documentation:** Updated `README.md`, created `ARCHITECTURE.md`, and `REFACTOR_PLAN.md`.
*   **Dependency Management:** `requirements.txt` for reproducible environments.

## Phase 2: Prediction Service & Basic UI

**Objective:** Expose the trained prediction model via a robust API and provide a simple, interactive user interface for manual predictions and visualization.

**Key Tasks:**

*   **2.1 Develop Prediction API (Completed):**
    *   Implemented a REST API endpoint (using FastAPI) that accepts historical OHLC data as input.
    *   The API internally calls `src/predict.py` to get predictions.
    *   Implemented input validation using Pydantic models.
    *   Implemented basic error handling.
    *   Added unit/integration tests for the API endpoint.
*   **2.2 Build Basic Web UI (Completed):**
    *   Created a simple web interface (using Streamlit) to interact with the Prediction API.
    *   Implemented manual input and CSV upload for historical data.
    *   Displays historical prices and overlays the predicted future price using Altair.
    *   The UI guides users to provide the correct number of data points (`TSTEPS`).
*   **2.3 Refine Prediction Logic for API (Completed):**
    *   Adjusted `src/predict.py` to handle single-sequence predictions more robustly by building a separate prediction model with `batch_size=1` and copying weights from the trained model.

## Phase 3: Automated Data Pipeline & Retraining

**Objective:** Automate the ingestion of new data from TWS, integrate it into the processing pipeline, and establish a mechanism for periodic model retraining.

**Key Tasks:**

*   **3.1 TWS Data Ingestion Integration:**
    *   Develop a module (e.g., `src/data_ingestion.py`) to connect to the Interactive Brokers (TWS) API.
    *   Implement logic to fetch historical XAG/USD minute-level data.
    *   Store newly ingested raw data in `data/raw/`.
*   **3.2 Automated Data Processing:**
    *   Create a scheduled job (e.g., using Airflow, Prefect, or a simple cron job) that:
        *   Triggers `src/data_ingestion.py` to fetch new data.
        *   Runs `src/data_processing.py` to convert new minute data to hourly and normalize it, updating `data/processed/`.
*   **3.3 Model Retraining Mechanism:**
    *   Implement a retraining script (e.g., `src/retrain.py`) that:
        *   Loads the latest processed data.
        *   Retrains the model (potentially using transfer learning or training from scratch).
        *   Evaluates the retrained model's performance against a held-out validation set.
        *   Saves the new model version if performance metrics meet predefined criteria.
*   **3.4 Model Versioning & Registry (Initial):**
    *   Establish a simple system for versioning trained models (e.g., saving with timestamped filenames).
    *   Update the prediction service to load the latest (or best-performing) model version.

## Phase 4: MLOps & Production Readiness

**Objective:** Implement advanced MLOps practices to ensure the model is robust, scalable, and performs reliably in a production environment.

**Key Tasks:**

*   **4.1 Model Monitoring:**
    *   Implement monitoring for model performance (e.g., prediction accuracy, loss) over time.
    *   Monitor data drift (changes in input data distribution) and concept drift (changes in the relationship between inputs and outputs).
    *   Integrate with monitoring tools (e.g., Prometheus/Grafana, MLflow Tracking).
*   **4.2 Automated Deployment:**
    *   Set up a CI/CD pipeline to automate testing, building, and deploying the prediction service and updated models.
    *   Implement A/B testing or canary deployments for new model versions.
*   **4.3 Alerting System:**
    *   Configure alerts for:
        *   Significant drops in model performance.
        *   Data pipeline failures.
        *   Data quality issues.
        *   System resource utilization.
*   **4.4 Scalability & Resilience:**
    *   Optimize the prediction service for high throughput and low latency.
    *   Implement fault tolerance and recovery mechanisms.
    *   Consider cloud deployment strategies (e.g., AWS SageMaker, Google AI Platform, Azure ML).
*   **4.5 Comprehensive Logging:**
    *   Implement detailed logging for all components (data ingestion, processing, training, prediction, API calls) for debugging and auditing.

This roadmap provides a structured approach to evolving the project from a core ML pipeline to a fully operational and production-ready system. Each phase builds upon the previous one, adding critical functionalities and MLOps capabilities.