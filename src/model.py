import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from typing import Optional, Tuple

from src.config import DROPOUT_RATE_1, DROPOUT_RATE_2

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
    print(f"DEBUG: build_lstm_model received input_shape: {input_shape}")

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
    print("Weights successfully transferred from stateful to non-stateful model.")

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
