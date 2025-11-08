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
