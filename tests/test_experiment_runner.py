import pytest
import os
import json
from unittest.mock import patch, mock_open
from datetime import datetime
import pandas as pd # Added import for pandas

from src.experiment_runner import (
    load_experiment_parameters,
    run_single_experiment,
    run_experiments
)
from src.config import (
    RESAMPLE_FREQUENCIES, TSTEPS_OPTIONS, LSTM_UNITS_OPTIONS, BATCH_SIZE_OPTIONS,
    N_LSTM_LAYERS_OPTIONS, STATEFUL_OPTIONS, FEATURES_TO_USE_OPTIONS,
    OPTIMIZER_OPTIONS, LOSS_FUNCTION_OPTIONS, RAW_DATA_CSV
)

# Mock external dependencies
@pytest.fixture
def mock_dependencies():
    with patch('src.experiment_runner.convert_minute_to_timeframe') as mock_convert_minute_to_timeframe, \
         patch('src.experiment_runner.train_model') as mock_train_model, \
         patch('src.experiment_runner.evaluate_model_performance') as mock_evaluate_model_performance, \
         patch('os.path.exists') as mock_os_path_exists, \
         patch('json.dump') as mock_json_dump, \
         patch('json.load') as mock_json_load, \
         patch('src.experiment_runner.datetime') as mock_datetime:
        
        # Configure mocks
        mock_train_model.return_value = (0.5, '/path/to/model.keras') # final_val_loss, model_path
        mock_evaluate_model_performance.return_value = (10.0, 0.8) # mae, correlation
        mock_os_path_exists.return_value = True # Assume files exist by default
        mock_json_load.return_value = {} # Default empty for best_hps_overall
        mock_datetime.now.return_value = datetime(2025, 1, 1, 12, 0, 0, 123456)
        mock_datetime.strftime.return_value = "20250101-120000-123456"
        mock_datetime.isoformat.return_value = "2025-01-01T12:00:00.123456"

        yield {
            'mock_convert_minute_to_timeframe': mock_convert_minute_to_timeframe,
            'mock_train_model': mock_train_model,
            'mock_evaluate_model_performance': mock_evaluate_model_performance,
            'mock_os_path_exists': mock_os_path_exists,
            'mock_json_dump': mock_json_dump,
            'mock_json_load': mock_json_load,
            'mock_datetime': mock_datetime
        }

# Test load_experiment_parameters
def test_load_experiment_parameters_file_not_found(mock_dependencies, capsys):
    mock_dependencies['mock_os_path_exists'].return_value = False
    with patch('builtins.open', side_effect=FileNotFoundError): # Mock open to raise FileNotFoundError
        result = load_experiment_parameters("some_id", "non_existent_file.json")
    assert result is None
    captured = capsys.readouterr()
    assert "Error: Results file 'non_existent_file.json' not found." in captured.out

def test_load_experiment_parameters_id_not_found(mock_dependencies, capsys):
    with patch('src.experiment_runner.json.load') as mock_json_load_func:
        mock_json_load_func.return_value = [{'experiment_id': 'other_id', 'param': 'value'}]
        result = load_experiment_parameters("some_id")
    assert result is None
    captured = capsys.readouterr()
    assert "Error: Experiment with ID 'some_id' not found." in captured.out

def test_load_experiment_parameters_success(mock_dependencies):
    expected_params = {'experiment_id': 'found_id', 'param': 'value'}
    with patch('src.experiment_runner.json.load') as mock_json_load_func:
        mock_json_load_func.return_value = [expected_params]
        result = load_experiment_parameters("found_id")
    assert result == expected_params

# Test run_single_experiment
def test_run_single_experiment_success(mock_dependencies, capsys):
    params = {
        'frequency': '15min', 'tsteps': 5, 'lstm_units': 128, 'batch_size': 64,
        'n_lstm_layers': 1, 'stateful': True, 'features_to_use': ['Open', 'High'],
        'optimizer_name': 'adam', 'loss_function': 'mse', 'experiment_id': 'test_id'
    }
    mock_dependencies['mock_train_model'].return_value = (0.1, '/path/to/model.keras', '/path/to/bias.json')
    mock_dependencies['mock_evaluate_model_performance'].return_value = (5.0, 0.9)

    result = run_single_experiment(params)

    mock_dependencies['mock_convert_minute_to_timeframe'].assert_called_once_with(RAW_DATA_CSV, '15min')
    mock_dependencies['mock_train_model'].assert_called_once_with(
        frequency='15min',
        tsteps=5,
        lstm_units=128,
        learning_rate=0.01,
        epochs=20,
        current_batch_size=64,
        n_lstm_layers=1,
        stateful=True,
        features_to_use=['Open', 'High'],
    )
    mock_dependencies['mock_evaluate_model_performance'].assert_called_once_with(
        model_path='/path/to/model.keras',
        frequency='15min',
        tsteps=5,
        lstm_units=128,
        n_lstm_layers=1,
        stateful=True,
        features_to_use=['Open', 'High'],
        bias_correction_path='/path/to/bias.json',
    )
    assert result == {'mae': 5.0, 'correlation': 0.9, 'validation_loss': 0.1, 'model_path': '/path/to/model.keras'}
    captured = capsys.readouterr()
    assert "Re-running Experiment ID: test_id" in captured.out
    assert "Validation Loss: 0.1000" in captured.out
    assert "MAE: 5.0000" in captured.out
    assert "Correlation: 0.9000" in captured.out

def test_run_single_experiment_error_handling(mock_dependencies, capsys):
    params = {
        'frequency': '15min', 'tsteps': 5, 'lstm_units': 128, 'batch_size': 64,
        'n_lstm_layers': 1, 'stateful': True, 'features_to_use': ['Open', 'High'],
        'optimizer_name': 'adam', 'loss_function': 'mse', 'experiment_id': 'error_id'
    }
    mock_dependencies['mock_train_model'].side_effect = Exception("Training failed")

    result = run_single_experiment(params)
    assert result is None
    captured = capsys.readouterr()
    assert "Error re-running experiment error_id: Training failed" in captured.err

# Test run_experiments
def test_run_experiments_basic_flow(mock_dependencies, capsys):
    # Reduce the number of combinations for a quick test
    with patch('src.experiment_runner.RESAMPLE_FREQUENCIES', ['15min']), \
         patch('src.experiment_runner.TSTEPS_OPTIONS', [5]), \
         patch('src.experiment_runner.LSTM_UNITS_OPTIONS', [128]), \
         patch('src.experiment_runner.BATCH_SIZE_OPTIONS', [64]), \
         patch('src.experiment_runner.N_LSTM_LAYERS_OPTIONS', [1]), \
         patch('src.experiment_runner.STATEFUL_OPTIONS', [True]), \
         patch('src.experiment_runner.FEATURES_TO_USE_OPTIONS', [['Open']]), \
         patch('src.experiment_runner.OPTIMIZER_OPTIONS', ['adam']), \
         patch('src.experiment_runner.LOSS_FUNCTION_OPTIONS', ['mae']):
        
        mock_dependencies['mock_train_model'].return_value = (0.1, '/path/to/model.keras', '/path/to/bias.json')
        mock_dependencies['mock_evaluate_model_performance'].return_value = (5.0, 0.9)
        mock_dependencies['mock_os_path_exists'].return_value = False # No existing best_hps.json

        run_experiments()

        # Verify calls
        mock_dependencies['mock_convert_minute_to_timeframe'].assert_called_once_with(RAW_DATA_CSV, '15min')
        mock_dependencies['mock_train_model'].assert_called_once()
        mock_dependencies['mock_evaluate_model_performance'].assert_called_once()

        # Verify json.dump calls
        assert mock_dependencies['mock_json_dump'].call_count == 2
        
        # Check experiment_results.json content
        args, kwargs = mock_dependencies['mock_json_dump'].call_args_list[0]
        results_dumped = args[0]
        assert len(results_dumped) == 1
        assert results_dumped[0]['frequency'] == '15min'
        assert results_dumped[0]['tsteps'] == 5
        assert results_dumped[0]['validation_loss'] == 0.1
        assert 'experiment_id' in results_dumped[0]
        
        # Check best_hyperparameters.json content
        args, kwargs = mock_dependencies['mock_json_dump'].call_args_list[1]
        best_hps_dumped = args[0]
        assert '15min' in best_hps_dumped
        assert '5' in best_hps_dumped['15min']
        assert best_hps_dumped['15min']['5']['validation_loss'] == 0.1
        assert 'experiment_id' in best_hps_dumped['15min']['5']

        # Verify print statements
        captured = capsys.readouterr()
        assert "Running Experiment ID: 20250101-120000-123456" in captured.out
        assert "Validation Loss: 0.1000" in captured.out
        assert "MAE: 5.0000" in captured.out
        assert "Correlation: 0.9000" in captured.out
        assert "All experiment results saved to 'experiment_results.json'" in captured.out
        assert "Updated best hyperparameters saved to 'best_hyperparameters.json'" in captured.out

def test_run_single_experiment_train_model_returns_none(mock_dependencies, capsys):
    params = {
        'frequency': '15min', 'tsteps': 5, 'lstm_units': 128, 'batch_size': 64,
        'n_lstm_layers': 1, 'stateful': True, 'features_to_use': ['Open', 'High'],
        'optimizer_name': 'adam', 'loss_function': 'mse', 'experiment_id': 'none_model_id'
    }
    mock_dependencies['mock_train_model'].return_value = None # Simulate train_model returning None

    result = run_single_experiment(params)

    mock_dependencies['mock_convert_minute_to_timeframe'].assert_called_once_with(RAW_DATA_CSV, '15min')
    mock_dependencies['mock_train_model'].assert_called_once()
    mock_dependencies['mock_evaluate_model_performance'].assert_not_called() # Should not call evaluate_model_performance
    assert result is None
    captured = capsys.readouterr()
    assert "Error re-running experiment none_model_id: NoneType object is not iterable" not in captured.out # Ensure no NoneType error
    assert "--- Re-run Summary for Experiment ID: none_model_id ---" not in captured.out # No summary printed

def test_run_experiments_train_model_returns_none_for_one_experiment(mock_dependencies, capsys):
    with patch('src.experiment_runner.RESAMPLE_FREQUENCIES', ['15min']), \
         patch('src.experiment_runner.TSTEPS_OPTIONS', [5, 10]), \
         patch('src.experiment_runner.LSTM_UNITS_OPTIONS', [128]), \
         patch('src.experiment_runner.BATCH_SIZE_OPTIONS', [64]), \
         patch('src.experiment_runner.N_LSTM_LAYERS_OPTIONS', [1]), \
         patch('src.experiment_runner.STATEFUL_OPTIONS', [True]), \
         patch('src.experiment_runner.FEATURES_TO_USE_OPTIONS', [['Open']]), \
         patch('src.experiment_runner.OPTIMIZER_OPTIONS', ['adam']), \
         patch('src.experiment_runner.LOSS_FUNCTION_OPTIONS', ['mae']):
        
        # Make the first experiment's train_model return None
        mock_dependencies['mock_train_model'].side_effect = [
            None,  # First experiment (tsteps=5) train_model returns None
            (0.1, '/path/to/model.keras', '/path/to/bias.json'),  # Second experiment (tsteps=10) succeeds
        ]
        mock_dependencies['mock_evaluate_model_performance'].return_value = (5.0, 0.9)
        mock_dependencies['mock_os_path_exists'].return_value = False

        run_experiments()

        # Verify that the second experiment still ran despite the first returning None
        captured = capsys.readouterr()
        assert "Running Experiment ID" in captured.out
        assert "Running Experiment ID" in captured.out # Should see output for both experiments
        assert mock_dependencies['mock_train_model'].call_count == 2
        assert mock_dependencies['mock_evaluate_model_performance'].call_count == 1 # Only called for the successful one
        
        # Check json.dump calls
        args, kwargs = mock_dependencies['mock_json_dump'].call_args_list[0]
        results_dumped = args[0]
        assert len(results_dumped) == 1 # Only the successful experiment should be in results
        assert results_dumped[0]['tsteps'] == 10 # The second experiment was tsteps=10

