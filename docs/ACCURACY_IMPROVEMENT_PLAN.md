# Accuracy Improvement Plan

This document outlines the strategies to be implemented to improve the prediction accuracy of the LSTM model.

## 1. Feature Engineering

The model's inputs will be enriched by adding more context beyond the raw OHLC (Open, High, Low, Close) prices.

- **Technical Indicators:**
  - **Moving Averages (SMA):** Add 7-period and 21-period Simple Moving Averages based on the 'Close' price to capture short and medium-term trends.
  - **Relative Strength Index (RSI):** Add a 14-period RSI to help the model identify overbought or oversold conditions.
  - **Moving Average Convergence Divergence (MACD):** Add MACD to reveal changes in the strength, direction, momentum, and duration of a trend.
- **Volume Data:**
  - Incorporate trading volume to give the model a sense of the market's conviction behind price movements.
- **Time-Based Features:**
  - **Hour of Day:** Extract the hour from the timestamp.
  - **Day of Week:** Extract the day of the week from the timestamp.

## 2. Hyperparameter Tuning

A systematic search for the optimal model hyperparameters will be conducted using a library like `KerasTuner` or `Optuna`. This will automate the process of finding the best combination of parameters such as:
- `TSTEPS` (window size)
- `ROWS_AHEAD` (prediction horizon)
- `LEARNING_RATE`
- `LSTM_UNITS`
- `BATCH_SIZE`

## 3. Advanced Model Architecture

The model's architecture will be enhanced to better capture complex temporal patterns.
- **Stacked LSTMs:** Increase the model's depth by stacking two or more LSTM layers.
- **Bidirectional LSTMs:** Experiment with Bidirectional LSTMs to process sequences in both forward and backward directions.

## 4. Data Preprocessing

Alternative data preprocessing techniques will be explored.
- **Scaling:** Experiment with `MinMaxScaler` as an alternative to `StandardScaler`.
- **Stationarity:** Investigate methods to make the time series stationary, such as predicting price changes (returns) instead of absolute price levels.
