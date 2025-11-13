import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model # Import Model

from tensorflow.keras.layers import Bidirectional
from src.config import DROPOUT_RATE_1, DROPOUT_RATE_2

def build_lstm_model(input_shape, lstm_units, batch_size, learning_rate):
    """
    Builds and compiles a Keras Stateful LSTM model.

    Args:
        input_shape (tuple): The shape of the input data (timesteps, features).
        lstm_units (int): Number of neurons in the LSTM layer.
        batch_size (int): The batch size for training (required for stateful model).
        learning_rate (float): The learning rate for the RMSprop optimizer.

    Returns:
        keras.Model: A compiled Keras LSTM model.
    """
    # Define the input layer for a stateful model
    inputs = keras.Input(batch_shape=(batch_size, input_shape[0], input_shape[1]), dtype=tf.float32)

    # Add the LSTM layer
    lstm_layer = layers.LSTM(units=lstm_units,
                             return_sequences=False, # False for single output
                             stateful=True,
                             activation='tanh')(inputs)
    dropout1 = layers.Dropout(DROPOUT_RATE_1)(lstm_layer)
    dropout2 = layers.Dropout(DROPOUT_RATE_2)(dropout1)

    # Add the Dense output layer
    outputs = layers.Dense(units=1, dtype=tf.float32)(dropout2)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    model.compile(loss='mae', optimizer=optimizer, metrics=['mae'])

    return model

def build_non_stateful_lstm_model(input_shape, lstm_units):
    """
    Builds a non-stateful Keras LSTM model with the same architecture as the stateful one.
    This is suitable for prediction/evaluation where state does not need to be maintained
    across batches.

    Args:
        input_shape (tuple): The shape of the input data (timesteps, features).
        lstm_units (int): Number of neurons in the LSTM layer.

    Returns:
        keras.Model: A non-stateful Keras LSTM model.
    """
    # Define the input layer for a non-stateful model (batch_size can be None)
    inputs = keras.Input(shape=(input_shape[0], input_shape[1]), dtype=tf.float32)

    # Add the LSTM layer (non-stateful)
    lstm_layer = layers.LSTM(units=lstm_units,
                             return_sequences=False,
                             stateful=False, # Key difference: non-stateful
                             activation='tanh')(inputs)
    dropout1 = layers.Dropout(DROPOUT_RATE_1)(lstm_layer)
    dropout2 = layers.Dropout(DROPOUT_RATE_2)(dropout1)

    # Add the Dense output layer
    outputs = layers.Dense(units=1, dtype=tf.float32)(dropout2)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
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

    # Build the stateful model
    dummy_stateful_model = build_lstm_model(
        input_shape=(dummy_tsteps, dummy_n_features),
        lstm_units=dummy_lstm_units,
        batch_size=dummy_batch_size,
        learning_rate=dummy_learning_rate
    )
    print("\nStateful Model Summary:")
    dummy_stateful_model.summary()

    # Build the non-stateful model
    dummy_non_stateful_model = build_non_stateful_lstm_model(
        input_shape=(dummy_tsteps, dummy_n_features),
        lstm_units=dummy_lstm_units
    )
    print("\nNon-Stateful Model Summary:")
    dummy_non_stateful_model.summary()

    # Test weight transfer (optional, for verification)
    # dummy_stateful_model.save_weights("temp_stateful_weights.h5")
    # dummy_non_stateful_model.load_weights("temp_stateful_weights.h5")
    # print("\nWeights transferred successfully (example).")
    
    print("\nDummy LSTM models built successfully.")