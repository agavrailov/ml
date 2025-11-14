import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from src.config import DROPOUT_RATE_1, DROPOUT_RATE_2

def build_lstm_model(input_shape, lstm_units, batch_size, learning_rate,
                     n_lstm_layers, stateful, optimizer_name, loss_function):
    """
    Builds and compiles a Keras LSTM model, configurable for stateful/non-stateful
    and multiple LSTM layers.
    """
    print(f"DEBUG: build_lstm_model received input_shape: {input_shape}")
    if stateful:
        inputs = keras.Input(batch_shape=(batch_size, input_shape[0], input_shape[1]), dtype=tf.float32)
    else:
        inputs = keras.Input(shape=(input_shape[0], input_shape[1]), dtype=tf.float32)

    x = inputs
    for i in range(n_lstm_layers):
        return_sequences = True if i < n_lstm_layers - 1 else False
        x = layers.LSTM(units=lstm_units,
                        return_sequences=return_sequences,
                        stateful=stateful,
                        activation='tanh')(x)
        # Always apply dropout after each LSTM layer
        x = layers.Dropout(DROPOUT_RATE_1)(x)
        x = layers.Dropout(DROPOUT_RATE_2)(x)

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

def load_stateful_weights_into_non_stateful_model(stateful_model, non_stateful_model):
    """
    Transfers weights from a stateful LSTM model to a non-stateful LSTM model
    with the same architecture.

    Args:
        stateful_model (keras.Model): The trained stateful model.
        non_stateful_model (keras.Model): The non-stateful model to load weights into.
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
