import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model # Import Model

from tensorflow.keras.layers import Bidirectional

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
    dropout = layers.Dropout(0.2)(lstm_layer)

    # Add the Dense output layer
    outputs = layers.Dense(units=1, dtype=tf.float32)(dropout)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    model.compile(loss='mae', optimizer=optimizer, metrics=['mae'])

    return model

if __name__ == "__main__":
    # Example usage and basic test
    print("Building a dummy LSTM model...")
    
    # Define dummy parameters
    dummy_tsteps = 3
    dummy_n_features = 4 # Assuming OHLC
    dummy_batch_size = 500
    dummy_lstm_units = 50
    dummy_learning_rate = 0.001

    # Build the model
    dummy_model = build_lstm_model(
        input_shape=(dummy_tsteps, dummy_n_features),
        lstm_units=dummy_lstm_units,
        batch_size=dummy_batch_size,
        learning_rate=dummy_learning_rate
    )

    # Print model summary
    dummy_model.summary()
    print("Dummy LSTM model built successfully.")