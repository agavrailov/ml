import sys
import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import (
    TRAINING_DATA_CSV, SCALER_PARAMS_JSON, TSTEPS, N_FEATURES,
    ROWS_AHEAD, TR_SPLIT, EPOCHS
)
from src.train import create_sequences_for_stateful_lstm, get_effective_data_length

def build_tuned_model(hp):
    """
    Builds a tunable LSTM model for KerasTuner.
    Defines the search space for hyperparameters.
    """
    # --- Hyperparameter Search Space ---
    hp_units = hp.Int('lstm_units', min_value=32, max_value=128, step=32)
    hp_learning_rate = hp.Choice('learning_rate', values=[0.01, 0.005, 0.001])
    batch_size = hp.Choice('batch_size', values=[128, 256, 500])

    # --- Model Architecture ---
    inputs = keras.Input(batch_shape=(batch_size, TSTEPS, N_FEATURES), dtype=tf.float32)
    lstm_layer = keras.layers.LSTM(
        units=hp_units,
        return_sequences=False, # Changed to False for single output
        stateful=True,
        activation='tanh'
    )(inputs)
    outputs = keras.layers.Dense(units=1)(lstm_layer)
    
    model = keras.Model(inputs=inputs, outputs=outputs)

    # --- Compilation ---
    optimizer = keras.optimizers.RMSprop(learning_rate=hp_learning_rate)
    model.compile(loss='mae', optimizer=optimizer, metrics=['mae'])
    
    return model

class MyTuner(kt.Hyperband):
    """
    Custom KerasTuner class to handle dynamic batch sizes during hyperparameter tuning.
    """
    def run_trial(self, trial, **kwargs):
        # Get the batch_size for the current trial
        hp = trial.hyperparameters
        batch_size = hp.get('batch_size')

        # --- Data Preparation for the current batch_size ---
        df_processed = pd.read_csv(TRAINING_DATA_CSV)
        train_size = int(len(df_processed) * TR_SPLIT)
        df_train = df_processed.iloc[:train_size].copy()
        df_val = df_processed.iloc[train_size:].copy()

        # Truncate data to be divisible by the current batch_size
        max_sequences_train = get_effective_data_length(df_train, TSTEPS, ROWS_AHEAD)
        if max_sequences_train < batch_size:
            print(f"Skipping trial: Training data too short for batch_size={batch_size}.")
            return
        remainder_train = max_sequences_train % batch_size
        if remainder_train > 0:
            df_train = df_train.iloc[:-remainder_train]

        max_sequences_val = get_effective_data_length(df_val, TSTEPS, ROWS_AHEAD)
        if max_sequences_val < batch_size:
            # Instead of skipping, just run without validation data if it's too short
            print(f"Warning: Validation data too short for batch_size={batch_size}. Proceeding without validation.")
            X_val, Y_val = np.array([]), np.array([])
        else:
            remainder_val = max_sequences_val % batch_size
            if remainder_val > 0:
                df_val = df_val.iloc[:-remainder_val]
            X_val, Y_val = create_sequences_for_stateful_lstm(df_val, TSTEPS, batch_size, ROWS_AHEAD)

        X_train, Y_train = create_sequences_for_stateful_lstm(df_train, TSTEPS, batch_size, ROWS_AHEAD)

        if len(X_train) == 0:
            print("Skipping trial due to insufficient training data after processing.")
            return
            
        # Update kwargs with the prepared data and the current batch_size
        kwargs['x'] = X_train
        kwargs['y'] = Y_train
        kwargs['batch_size'] = batch_size # Explicitly set batch_size for fit
        if len(X_val) > 0:
            kwargs['validation_data'] = (X_val, Y_val)
        else:
            kwargs['validation_data'] = None

        # Pass the prepared data to the original run_trial method
        return super(MyTuner, self).run_trial(trial, **kwargs)

if __name__ == "__main__":
    if not os.path.exists(TRAINING_DATA_CSV):
        print(f"Error: Training data not found at '{TRAINING_DATA_CSV}'.")
        print("Please run 'python src/data_processing.py' first.")
        sys.exit(1)

    tuner = MyTuner(
        build_tuned_model,
        objective='val_loss',
        max_epochs=EPOCHS,
        factor=3,
        directory='hyperparameter_tuning',
        project_name='lstm_tuning'
    )

    print("\n--- Starting Hyperparameter Tuning ---")
    # The data passed here is just a placeholder; the actual data is prepared in run_trial
    tuner.search(x=None, y=None, epochs=EPOCHS, shuffle=False)
    print("\n--- Hyperparameter Tuning Finished ---")

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    Best hyperparameters found:
    - LSTM Units: {best_hps.get('lstm_units')}
    - Learning Rate: {best_hps.get('learning_rate')}
    - Batch Size: {best_hps.get('batch_size')}
    """)

    # Save the best hyperparameters to a file
    best_hps_dict = best_hps.values
    with open('best_hyperparameters.json', 'w') as f:
        json.dump(best_hps_dict, f, indent=4)
    
    print("Best hyperparameters saved to 'best_hyperparameters.json'")
