import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

from src.config import DROPOUT_RATE_1, DROPOUT_RATE_2, ModelConfig, get_model_config

def build_lstm_model(
    input_shape: Tuple[int, int],
    lstm_units: int,
    batch_size: Optional[int],
    learning_rate: float,
    n_lstm_layers: int,
    stateful: bool,
    optimizer_name: str,
    loss_function: str,
) -> keras.Model:
    """Build and compile an LSTM model.

    The model can be configured as stateful or stateless and with one or more
    stacked LSTM layers.

    Args:
        input_shape: Tuple ``(tsteps, n_features)`` describing the sequence input.
        lstm_units: Number of units in each LSTM layer.
        batch_size: Fixed batch size for stateful models, or ``None`` for stateless.
        learning_rate: Learning rate for the optimizer.
        n_lstm_layers: Number of stacked LSTM layers (>= 1).
        stateful: Whether to build a stateful LSTM (fixed batch dimension).
        optimizer_name: Name of the optimizer (e.g. ``"rmsprop"``, ``"adam"``).
        loss_function: Name of the loss function (e.g. ``"mse"``, ``"mae"``).

    Returns:
        A compiled Keras ``Model`` ready for training or inference.
    """

    # --- Input configuration ---
    # For stateful=True, we must fix the batch dimension at build time and then
    # consistently use the same batch_size during training.
    if stateful:
        if batch_size is None or batch_size <= 0:
            raise ValueError(
                "For a stateful LSTM, 'batch_size' must be a positive integer "
                "(got None or non-positive)."
            )
        inputs = keras.Input(
            batch_shape=(batch_size, input_shape[0], input_shape[1]),
            dtype=tf.float32,
        )
    else:
        # For non-stateful models the batch dimension is left dynamic.
        inputs = keras.Input(shape=(input_shape[0], input_shape[1]), dtype=tf.float32)

    x = inputs
    for i in range(n_lstm_layers):
        return_sequences = True if i < n_lstm_layers - 1 else False
        x = layers.LSTM(
            units=lstm_units,
            return_sequences=return_sequences,
            stateful=stateful,
            activation="tanh",
        )(x)
        # Apply a single dropout layer after each LSTM block.
        # DROPOUT_RATE_1 controls this post-LSTM dropout; DROPOUT_RATE_2 is kept
        # available for future use (e.g. recurrent/temporal dropout) but is not
        # applied as a second sequential Dropout layer here.
        x = layers.Dropout(DROPOUT_RATE_1)(x)

    outputs = layers.Dense(units=1, dtype=tf.float32)(x)

    model = Model(inputs=inputs, outputs=outputs)

    # Configure optimizer
    if optimizer_name == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer_name == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    model.compile(loss=loss_function, optimizer=optimizer, metrics=['mae'])

    return model

def load_stateful_weights_into_non_stateful_model(
    stateful_model: keras.Model,
    non_stateful_model: keras.Model,
) -> None:
    """Copy weights from a stateful LSTM model into a non-stateful model.

    Both models must share the same architecture (same layers and parameter
    shapes), otherwise setting weights will raise a value error.

    Args:
        stateful_model: The trained, stateful Keras model.
        non_stateful_model: The non-stateful model to receive the weights.
    """
    # Get weights from the stateful model
    stateful_weights = stateful_model.get_weights()

    # Set weights to the non-stateful model
    non_stateful_model.set_weights(stateful_weights)


def build_model(model_cfg: Optional[ModelConfig] = None) -> keras.Model:
    """Build a compiled LSTM model from :class:`ModelConfig`.

    This is a thin convenience wrapper around :func:`build_lstm_model` that
    reads defaults from :func:`get_model_config` when ``model_cfg`` is omitted.
    """

    cfg = model_cfg or get_model_config()

    return build_lstm_model(
        input_shape=(cfg.tsteps, cfg.n_features),
        lstm_units=cfg.lstm_units,
        batch_size=cfg.batch_size if cfg.stateful else None,
        learning_rate=cfg.learning_rate,
        n_lstm_layers=cfg.n_lstm_layers,
        stateful=cfg.stateful,
        optimizer_name=cfg.optimizer_name,
        loss_function=cfg.loss_function,
    )


def load_model(model_path: str | Path, *, compile: bool = True) -> keras.Model:
    """Load a saved Keras model from disk.

    Centralizes usage of ``keras.models.load_model`` so higher-level code can
    depend on a single entrypoint for deserialization.
    """

    return keras.models.load_model(str(model_path), compile=compile)

from src.experiments import log_experiment


def train_and_save_model(
    *,
    model: keras.Model,
    X_train,
    Y_train,
    X_val,
    Y_val,
    epochs: int,
    batch_size: int,
    frequency: str,
    tsteps: int,
    model_registry_dir: str,
    patience: int = 5,
):
    """Fit a model on (X_train, Y_train), evaluate on validation data, and save it.

    This helper centralizes the Keras ``fit`` + early-stopping behaviour and the
    naming convention for saved models used during training.

    Returns
    -------
    final_val_loss : float
        The best (minimum) validation loss observed during training.
    model_path : str
        Filesystem path where the trained model was saved.
    timestamp : str
        Timestamp string used in the filename, e.g. ``"20250101_000000"``.
    """

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, Y_val),
        callbacks=[early_stopping],
        shuffle=False,
        verbose=2,
    )

    final_val_loss = float(min(history.history["val_loss"]))

    os.makedirs(model_registry_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(
        model_registry_dir,
        f"my_lstm_model_{frequency}_tsteps{tsteps}_{timestamp}.keras",
    )
    model.save(model_path)

    # Persist per-model metrics next to the model artifact so UIs can show
    # validation loss for *every* registry model (not just the promoted/best one).
    metrics_path = model_path[: -len(".keras")] + ".metrics.json"
    try:
        Path(metrics_path).write_text(
            json.dumps(
                {
                    "model_filename": os.path.basename(model_path),
                    "frequency": frequency,
                    "tsteps": int(tsteps),
                    "epochs": int(epochs),
                    "batch_size": int(batch_size),
                    "validation_loss": float(final_val_loss),
                    "saved_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        # Best-effort: metrics are helpful for the UI but should not fail training.
        pass

    # Log a compact summary record for this training run.
    log_experiment(
        {
            "phase": "initial_train",
            "model_path": model_path,
            "frequency": frequency,
            "tsteps": tsteps,
            "batch_size": batch_size,
            "epochs": epochs,
            "patience": patience,
            "final_val_loss": final_val_loss,
            "history": {
                "loss": history.history.get("loss"),
                "val_loss": history.history.get("val_loss"),
                "mae": history.history.get("mae"),
            },
        }
    )

    return final_val_loss, model_path, timestamp


if __name__ == "__main__":
    # Example usage and basic test
    print("Building a dummy LSTM model...")
    
    # Define dummy parameters
    dummy_tsteps = 3
    dummy_n_features = 4 # Assuming OHLC
    dummy_batch_size = 500
    dummy_lstm_units = 50
    dummy_learning_rate = 0.001
    dummy_n_lstm_layers = 2
    dummy_stateful = True
    dummy_optimizer = 'rmsprop'
    dummy_loss = 'mae'

    # Build the stateful model
    dummy_stateful_model = build_lstm_model(
        input_shape=(dummy_tsteps, dummy_n_features),
        lstm_units=dummy_lstm_units,
        batch_size=dummy_batch_size,
        learning_rate=dummy_learning_rate,
        n_lstm_layers=dummy_n_lstm_layers,
        stateful=dummy_stateful,
        optimizer_name=dummy_optimizer,
        loss_function=dummy_loss
    )
    print("\nStateful Model Summary:")
    dummy_stateful_model.summary()

    # Build a non-stateful model for comparison
    dummy_non_stateful_model = build_lstm_model(
        input_shape=(dummy_tsteps, dummy_n_features),
        lstm_units=dummy_lstm_units,
        batch_size=None, # None for non-stateful
        learning_rate=dummy_learning_rate,
        n_lstm_layers=dummy_n_lstm_layers,
        stateful=False,
        optimizer_name=dummy_optimizer,
        loss_function=dummy_loss
    )
    print("\nNon-Stateful Model Summary:")
    dummy_non_stateful_model.summary()
    
    print("\nDummy LSTM models built successfully.")
