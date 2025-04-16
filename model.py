from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam
import os

class LSTMAutoencoder:
    def __init__(self, window_size: int, n_channels: int):
        self.window_size = window_size
        self.n_channels = n_channels
        self.model = self._build_model()

    def _build_model(self) -> Model:
        # Define input shape
        inputs = Input(shape=(self.window_size, self.n_channels))

        # Encoder: maps the input sequence to a latent vector
        encoded = LSTM(64, activation='relu', return_sequences=True)(inputs)
        encoded = LSTM(32, activation='relu')(encoded)

        # Latent representation is repeated for decoding
        decoded = RepeatVector(self.window_size)(encoded)

        # Decoder: reconstructs the sequence
        decoded = LSTM(32, activation='relu', return_sequences=True)(decoded)
        decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)

        # Output layer: reconstructs original features for each time step
        output = TimeDistributed(Dense(self.n_channels))(decoded)

        # Build model
        model = Model(inputs, output)
        model.compile(
            optimizer=Adam(learning_rate=1e-4, clipvalue=1.0),
            loss='mse',
            metrics=['mae']
        )

        return model

    def summary(self):
        self.model.summary()

    def train(self, X_train, epochs=10, batch_size=32, validation_split=0.1, **kwargs):
        return self.model.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            shuffle=True,
            **kwargs
        )

    def predict(self, X, batch_size=32, verbose=1):
        return self.model.predict(X, batch_size=batch_size, verbose=verbose)

    def save(self, path="autoencoder_model.keras"):
        # Create directory if it doesn't exist
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        self.model.save(path)

    def load_weights(self, path):
        self.model.load_weights(path)
