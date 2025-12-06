from __future__ import annotations

import argparse
import itertools
import json
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List

import numpy as np
import tensorflow as tf

# Disable eager execution so training uses compiled graph mode (faster,
# and avoids extra tf.data / TensorArray warnings).
tf.config.run_functions_eagerly(False)

from src.config import (
    TRAINING,
    get_model_config,
    MODEL_REGISTRY_DIR,
    FEATURES_TO_USE_OPTIONS,
)
from src.data_processing import prepare_keras_input_data
from src.data_utils import (
    fit_standard_scaler,
    apply_standard_scaler,
    create_sequences_for_stateless_lstm,
)
from src.model import build_lstm_model

EXPERIMENTS_LOG = os.path.join("experiments", "experiments_log.jsonl")


def log_experiment(record: Dict[str, Any]) -> None:
    """Append a single experiment record as JSON to the experiments log.

    The record is shallow-copied and enriched with a UTC timestamp if it does
    not already contain one. The log format is JSON Lines (one JSON object per
    line) for easy ingestion into pandas or other tools.
    """

    os.makedirs(os.path.dirname(EXPERIMENTS_LOG), exist_ok=True)

    enriched = dict(record)
    enriched.setdefault("timestamp", datetime.utcnow().isoformat())

    with open(EXPERIMENTS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(enriched) + "\n")
def _product(dict_of_iterables: Dict[str, Iterable[Any]]) -> Iterable[Dict[str, Any]]:
    """Cartesian product over a mapping of name -> iterable of values.

    Returns an iterator of dicts mapping each name to a concrete choice.
    """

    keys = list(dict_of_iterables.keys())
    for values in itertools.product(*(dict_of_iterables[k] for k in keys)):
        yield dict(zip(keys, values))
def _load_all_experiments() -> List[Dict[str, Any]]:
    """Load all experiment records from the JSONL log, if it exists."""

    if not os.path.exists(EXPERIMENTS_LOG):
        return []

    records: List[Dict[str, Any]] = []
    with open(EXPERIMENTS_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip malformed lines to keep the log robust over time.
                continue
    return records


def _update_global_best_from_log() -> None:
    """Scan the log for the globally best grid run.

    This helper currently has no side-effects; global best information is read
    from and written to ``best_hyperparameters.json`` by training scripts and
    UIs. The log remains useful for offline analysis.
    """

    records = _load_all_experiments()
    if not records:
        return

    # Filter to completed grid runs that have the fields we expect.
    candidates: List[Dict[str, Any]] = []
    for r in records:
        if r.get("phase") != "grid_search_initial":
            continue
        if "final_val_loss" in r and "config" in r and "model_path" in r:
            candidates.append(r)

    if not candidates:
        return

    # We still compute the best candidate for logging/debugging purposes, but
    # do not persist a separate best_config.json file anymore.
    _best = min(candidates, key=lambda r: r["final_val_loss"])


def run_lr_units_batch_grid(
    *,
    learning_rates: List[float] | None = None,
    lstm_units_list: List[int] | None = None,
    batch_sizes: List[int] | None = None,
) -> None:
    """Run a small grid over (learning_rate, lstm_units, batch_size).

    This uses the current TRAINING defaults for all other hyperparameters and
    logs a compact record for each configuration.
    """

    # Defaults from TRAINING config
    learning_rates = learning_rates or [1e-4, 3e-4, 1e-3]
    lstm_units_list = lstm_units_list or [32, 64]
    batch_sizes = batch_sizes or [32, 64]

    freq = TRAINING.frequency

    # Choose a default feature set for experimentation (first configured option)
    features_to_use = FEATURES_TO_USE_OPTIONS[0]

    # Prepare data once per run
    df_featured, feature_cols = prepare_keras_input_data(
        os.path.join("data", "processed", f"nvda_{freq}.csv"),
        features_to_use,
    )

    results = []

    for combo in _product(
        {
            "learning_rate": learning_rates,
            "lstm_units": lstm_units_list,
            "batch_size": batch_sizes,
        }
    ):
        lr = combo["learning_rate"]
        units = combo["lstm_units"]
        batch_size = combo["batch_size"]

        print(f"\n=== Running config: lr={lr}, units={units}, batch_size={batch_size} ===")

        # Deterministic seeds for comparability
        np.random.seed(42)
        tf.random.set_seed(42)

        # Fit scaler on full data for this coarse search run
        _, _, scaler_params = fit_standard_scaler(df_featured, feature_cols)
        df_norm = apply_standard_scaler(df_featured, feature_cols, scaler_params)

        # Build sequences for a *stateless* LSTM during coarse experiments.
        # This avoids the strict batch-size divisibility constraint that
        # stateful RNNs impose, which simplifies experimentation.
        tsteps = TRAINING.tsteps
        rows_ahead = TRAINING.rows_ahead

        X, Y = create_sequences_for_stateless_lstm(
            df_norm,
            tsteps,
            rows_ahead,
        )

        if X.shape[0] == 0:
            print("Not enough data to form sequences for this configuration; skipping.")
            continue

        # Simple train/validation split along the time axis
        tr_split = TRAINING.tr_split
        n_samples = X.shape[0]
        n_train = int(n_samples * tr_split)
        X_train, Y_train = X[:n_train], Y[:n_train]
        X_val, Y_val = X[n_train:], Y[n_train:]

        if X_val.shape[0] == 0:
            print("No validation samples after split; skipping.")
            continue

        # Build model config and model: use a *stateless* LSTM for experiments.
        model_cfg = get_model_config(
            frequency=freq,
            tsteps=tsteps,
            n_features=TRAINING.n_features,
            lstm_units=units,
            batch_size=None,  # None for stateless
            learning_rate=lr,
            n_lstm_layers=TRAINING.n_lstm_layers,
            stateful=False,
            optimizer_name=TRAINING.optimizer_name,
            loss_function=TRAINING.loss_function,
        )

        model = build_lstm_model(
            input_shape=(model_cfg.tsteps, model_cfg.n_features),
            lstm_units=model_cfg.lstm_units,
            batch_size=None,
            learning_rate=model_cfg.learning_rate,
            n_lstm_layers=model_cfg.n_lstm_layers,
            stateful=False,
            optimizer_name=model_cfg.optimizer_name,
            loss_function=model_cfg.loss_function,
        )

        # Simple training loop with early stopping handled by model.train_and_save_model
        from src.model import train_and_save_model  # local import to avoid cycles at module import time

        final_val_loss, model_path, timestamp = train_and_save_model(
            model=model,
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val,
            epochs=TRAINING.epochs,
            batch_size=batch_size,
            frequency=freq,
            tsteps=tsteps,
            model_registry_dir=MODEL_REGISTRY_DIR,
        )

        combo_result = {
            "phase": "grid_search_initial",
            "config": {
                "frequency": freq,
                "tsteps": tsteps,
                "lstm_units": units,
                "batch_size": batch_size,
                "learning_rate": lr,
                "n_lstm_layers": TRAINING.n_lstm_layers,
                "stateful": TRAINING.stateful,
                "optimizer_name": TRAINING.optimizer_name,
                "loss_function": TRAINING.loss_function,
            },
            "final_val_loss": float(final_val_loss),
            "model_path": model_path,
            "timestamp": timestamp,
        }

        results.append(combo_result)
        log_experiment(combo_result)

    # Also log a summary record for *this* grid run, then recompute the
    # global best across all logged experiments.
    if results:
        best = min(results, key=lambda r: r["final_val_loss"])

        summary = {
            "phase": "grid_search_initial_summary",
            "num_runs": len(results),
            "best_final_val_loss": best["final_val_loss"],
            "best_config": best["config"],
            "best_model_path": best["model_path"],
        }
        log_experiment(summary)

        # Recompute and persist the global best based on the full JSONL log.
        _update_global_best_from_log()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a small hyperparameter grid over (learning_rate, lstm_units, "
            "batch_size) and log validation losses."
        )
    )

    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="*",
        default=[1e-4, 3e-4, 1e-3],
        help="Learning rates to try (space-separated).",
    )
    parser.add_argument(
        "--lstm-units",
        type=int,
        nargs="*",
        default=[32, 64],
        help="LSTM unit sizes to try (space-separated).",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="*",
        default=[32, 64],
        help="Batch sizes to try (space-separated).",
    )

    args = parser.parse_args()

    run_lr_units_batch_grid(
        learning_rates=args.learning_rates,
        lstm_units_list=args.lstm_units,
        batch_sizes=args.batch_sizes,
    )


if __name__ == "__main__":
    main()
