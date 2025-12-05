import sys
import os
import json
import itertools
from datetime import datetime

from src.config import (
    RESAMPLE_FREQUENCIES,
    TSTEPS_OPTIONS,
    LSTM_UNITS_OPTIONS,
    BATCH_SIZE_OPTIONS,
    N_LSTM_LAYERS_OPTIONS,
    STATEFUL_OPTIONS,
    OPTIMIZER_OPTIONS,
    LOSS_FUNCTION_OPTIONS,
    FEATURES_TO_USE_OPTIONS,
    RAW_DATA_CSV,
    get_run_hyperparameters,
)
from src.data_processing import convert_minute_to_timeframe
from src.train import train_model
from src.evaluate_model import evaluate_model_performance

def load_experiment_parameters(experiment_id, results_file='experiment_results.json'):
    """
    Loads the parameters of a specific experiment from the results file.
    """
    if not os.path.exists(results_file):
        print(f"Error: Results file '{results_file}' not found.")
        return None

    with open(results_file, 'r') as f:
        results = json.load(f)

    for exp in results:
        if exp.get('experiment_id') == experiment_id:
            return exp
    
    print(f"Error: Experiment with ID '{experiment_id}' not found.")
    return None

def run_single_experiment(params):
    """Run a single experiment with the current train/eval APIs.

    If some hyperparameters are missing from ``params``, fall back to tuned or
    default values via :func:`get_run_hyperparameters`.
    """
    frequency = params['frequency']
    tsteps = params['tsteps']

    # Resolve hyperparameters, letting explicit params override tuned/defaults
    hps = get_run_hyperparameters(frequency=frequency, tsteps=tsteps)
    lstm_units = params.get('lstm_units', hps['lstm_units'])
    batch_size = params.get('batch_size', hps['batch_size'])
    n_lstm_layers = params.get('n_lstm_layers', hps['n_lstm_layers'])
    stateful = params.get('stateful', hps['stateful'])
    features_to_use = params.get('features_to_use', hps['features_to_use'])

    experiment_id = params.get('experiment_id', 'N/A')

    print(f"\n--- Re-running Experiment ID: {experiment_id} ---")
    print(
        f"Frequency: {frequency}, TSTEPS: {tsteps}, LSTM Units: {lstm_units}, "
        f"Batch Size: {batch_size}, N_LSTM_Layers: {n_lstm_layers}, Stateful: {stateful}, "
        f"Features: {features_to_use}"
    )

    try:
        # Convert minute data to current frequency
        convert_minute_to_timeframe(RAW_DATA_CSV, frequency)

        # --- Train Model ---
        train_result = train_model(
            frequency=frequency,
            tsteps=tsteps,
            lstm_units=lstm_units,
            learning_rate=0.01,  # Fixed for now
            epochs=20,  # Fixed for now
            current_batch_size=batch_size,
            n_lstm_layers=n_lstm_layers,
            stateful=stateful,
            features_to_use=features_to_use,
        )

        if not train_result:
            return None

        final_val_loss, model_path, bias_correction_path = train_result

        # --- Evaluate Model ---
        mae, correlation = evaluate_model_performance(
            model_path=model_path,
            frequency=frequency,
            tsteps=tsteps,
            lstm_units=lstm_units,
            n_lstm_layers=n_lstm_layers,
            stateful=stateful,
            features_to_use=features_to_use,
            bias_correction_path=bias_correction_path,
        )

        print(f"--- Re-run Summary for Experiment ID: {experiment_id} ---")
        print(f"  Validation Loss: {final_val_loss:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  Correlation: {correlation:.4f}")
        print(f"  Model Path: {model_path}")
        print("------------------------------------")
        return {
            'mae': mae,
            'correlation': correlation,
            'validation_loss': final_val_loss,
            'model_path': model_path,
        }

    except Exception as e:
        print(f"Error re-running experiment {experiment_id}: {e}", file=sys.stderr)
        return None

def run_experiments():
    """
    Orchestrates running experiments with different hyperparameter combinations.
    """
    results = []
    best_hps_overall = {}
    best_hps_path = 'best_hyperparameters.json'

    # Load existing best hyperparameters if available
    if os.path.exists(best_hps_path):
        try:
            with open(best_hps_path, 'r') as f:
                best_hps_overall = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: '{best_hps_path}' is empty or contains invalid JSON. Initializing best hyperparameters as empty.")
            best_hps_overall = {}

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
        
        experiment_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f") # Unique ID for each experiment
        
        print(f"\n--- Running Experiment ID: {experiment_id} ---")
        print(f"Frequency: {frequency}, TSTEPS: {tsteps}, LSTM Units: {lstm_units}, "
              f"Batch Size: {batch_size}, N_LSTM_Layers: {n_lstm_layers}, Stateful: {stateful}, "
              f"Features: {features_to_use}, Optimizer: {optimizer_name}, Loss: {loss_function}")

        try:
            # Convert minute data to current frequency
            convert_minute_to_timeframe(RAW_DATA_CSV, frequency)
            
            # Prepare Keras input data (features will be selected in train/eval)
            n_features = len(features_to_use)

        # --- Train Model ---
            train_result = train_model(
                frequency=frequency,
                tsteps=tsteps,
                lstm_units=lstm_units,
                learning_rate=0.01,  # Fixed for now
                epochs=20,  # Fixed for now
                current_batch_size=batch_size,
                n_lstm_layers=n_lstm_layers,
                stateful=stateful,
                features_to_use=features_to_use,
            )

            if not train_result:
                continue

            final_val_loss, model_path, bias_correction_path = train_result

            if model_path:
                # --- Evaluate Model ---
                mae, correlation = evaluate_model_performance(
                    model_path=model_path,
                    frequency=frequency,
                    tsteps=tsteps,
                    lstm_units=lstm_units,
                    n_lstm_layers=n_lstm_layers,
                    stateful=stateful,
                    features_to_use=features_to_use,
                    bias_correction_path=bias_correction_path,
                )

                experiment_result = {
                    'experiment_id': experiment_id,
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

                print(f"--- Experiment {experiment_id} Summary ---")
                print(f"  Validation Loss: {final_val_loss:.4f}")
                print(f"  MAE: {mae:.4f}")
                print(f"  Correlation: {correlation:.4f}")
                print(f"  Model Path: {model_path}")
                print("------------------------------------")

                # Update best_hps_overall
                if frequency not in best_hps_overall:
                    best_hps_overall[frequency] = {}
                if str(tsteps) not in best_hps_overall[frequency]:
                    best_hps_overall[frequency][str(tsteps)] = {}
                
                current_best_loss = best_hps_overall[frequency][str(tsteps)].get('validation_loss', float('inf'))
                if final_val_loss < current_best_loss:
                    best_hps_overall[frequency][str(tsteps)] = {
                        'experiment_id': experiment_id, # Store the ID of the best experiment
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
            print(f"Error running experiment for {frequency}, {tsteps}, {lstm_units}, {batch_size}: {e}", file=sys.stderr)


    # Save all experiment results
    with open('experiment_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nAll experiment results saved to 'experiment_results.json'")

    # Save updated best hyperparameters
    with open(best_hps_path, 'w') as f:
        json.dump(best_hps_overall, f, indent=4)
    print(f"Updated best hyperparameters saved to '{best_hps_path}'")

if __name__ == "__main__":
    # Example of how to run all experiments
    run_experiments()

    # Example of how to re-run a specific experiment by ID
    # experiment_id_to_rerun = "20251113-220000-123456" # Replace with an actual ID from experiment_results.json
    # params_to_rerun = load_experiment_parameters(experiment_id_to_rerun)
    # if params_to_rerun:
    #     run_single_experiment(params_to_rerun)