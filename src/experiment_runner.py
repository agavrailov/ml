import sys
import os
import json
import itertools
import pandas as pd
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import (
    RESAMPLE_FREQUENCIES, TSTEPS_OPTIONS, LSTM_UNITS_OPTIONS, BATCH_SIZE_OPTIONS,
    DROPOUT_RATE_OPTIONS, N_LSTM_LAYERS_OPTIONS, STATEFUL_OPTIONS,
    OPTIMIZER_OPTIONS, LOSS_FUNCTION_OPTIONS, FEATURES_TO_USE_OPTIONS,
    PROCESSED_DATA_DIR, RAW_DATA_CSV,
    get_hourly_data_csv_path, get_training_data_csv_path, get_scaler_params_json_path
)
from src.data_processing import convert_minute_to_timeframe, prepare_keras_input_data
from src.train import train_model
from src.evaluate_model import evaluate_model_performance

def run_experiments():
    """
    Orchestrates running experiments with different hyperparameter combinations.
    """
    results = []
    best_hps_overall = {}
    best_hps_path = 'best_hyperparameters.json'

    # Load existing best hyperparameters if available
    if os.path.exists(best_hps_path):
        with open(best_hps_path, 'r') as f:
            best_hps_overall = json.load(f)

    # Iterate through all combinations of hyperparameters
    # For simplicity, we'll iterate through a subset of options first.
    # Full Cartesian product can be very large.
    # Let's focus on frequency, tsteps, lstm_units, batch_size, n_lstm_layers, stateful, features_to_use
    
    experiment_params = itertools.product(
        RESAMPLE_FREQUENCIES,
        TSTEPS_OPTIONS,
        LSTM_UNITS_OPTIONS,
        BATCH_SIZE_OPTIONS,
        N_LSTM_LAYERS_OPTIONS,
        STATEFUL_OPTIONS,
        FEATURES_TO_USE_OPTIONS,
        OPTIMIZER_OPTIONS,
        LOSS_FUNCTION_OPTIONS
    )

    for (frequency, tsteps, lstm_units, batch_size, n_lstm_layers, stateful,
         features_to_use, optimizer_name, loss_function) in experiment_params:
        
        print(f"\n--- Running Experiment ---")
        print(f"Frequency: {frequency}, TSTEPS: {tsteps}, LSTM Units: {lstm_units}, "
              f"Batch Size: {batch_size}, N_LSTM_Layers: {n_lstm_layers}, Stateful: {stateful}, "
              f"Features: {features_to_use}, Optimizer: {optimizer_name}, Loss: {loss_function}")

        try:
            # Convert minute data to current frequency
            convert_minute_to_timeframe(RAW_DATA_CSV)
            
            # Prepare Keras input data (features will be selected in train/eval)
            # N_FEATURES will be determined by len(features_to_use)
            n_features = len(features_to_use)

            # --- Train Model ---
            final_val_loss, model_path = train_model(
                frequency=frequency,
                tsteps=tsteps,
                n_features=n_features,
                lstm_units=lstm_units,
                learning_rate=0.01, # Using a fixed learning rate for now
                epochs=20, # Using fixed epochs for now
                current_batch_size=batch_size,
                n_lstm_layers=n_lstm_layers,
                stateful=stateful,
                optimizer_name=optimizer_name,
                loss_function=loss_function
            )

            if model_path:
                # --- Evaluate Model ---
                mae, correlation = evaluate_model_performance(
                    frequency=frequency,
                    tsteps=tsteps,
                    n_features=n_features,
                    lstm_units=lstm_units,
                    n_lstm_layers=n_lstm_layers,
                    stateful=stateful,
                    optimizer_name=optimizer_name,
                    loss_function=loss_function,
                    features_to_use=features_to_use
                )

                experiment_result = {
                    'timestamp': datetime.now().isoformat(),
                    'frequency': frequency,
                    'tsteps': tsteps,
                    'lstm_units': lstm_units,
                    'batch_size': batch_size,
                    'n_lstm_layers': n_lstm_layers,
                    'stateful': stateful,
                    'features_to_use': features_to_use,
                    'optimizer_name': optimizer_name,
                    'loss_function': loss_function,
                    'validation_loss': final_val_loss,
                    'mae': mae,
                    'correlation': correlation,
                    'model_path': model_path
                }
                results.append(experiment_result)

                # Update best_hps_overall
                if frequency not in best_hps_overall:
                    best_hps_overall[frequency] = {}
                if str(tsteps) not in best_hps_overall[frequency]:
                    best_hps_overall[frequency][str(tsteps)] = {}
                
                current_best_loss = best_hps_overall[frequency][str(tsteps)].get('validation_loss', float('inf'))
                if final_val_loss < current_best_loss:
                    best_hps_overall[frequency][str(tsteps)] = {
                        'validation_loss': final_val_loss,
                        'model_filename': os.path.basename(model_path),
                        'lstm_units': lstm_units,
                        'learning_rate': 0.01, # Fixed for now
                        'epochs': 20, # Fixed for now
                        'batch_size': batch_size,
                        'n_lstm_layers': n_lstm_layers,
                        'stateful': stateful,
                        'optimizer_name': optimizer_name,
                        'loss_function': loss_function,
                        'features_to_use': features_to_use
                    }
                    print(f"Updated best hyperparameters for {frequency}, TSTEPS={tsteps} with new best loss: {final_val_loss:.4f}")

        except Exception as e:
            print(f"Error running experiment for {frequency}, {tsteps}, {lstm_units}, {batch_size}: {e}")


    # Save all experiment results
    with open('experiment_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nAll experiment results saved to 'experiment_results.json'")

    # Save updated best hyperparameters
    with open(best_hps_path, 'w') as f:
        json.dump(best_hps_overall, f, indent=4)
    print(f"Updated best hyperparameters saved to '{best_hps_path}'")

if __name__ == "__main__":
    run_experiments()
