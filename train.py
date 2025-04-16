from model import LSTMAutoencoder
from data_preprocessor import DataPreprocessor
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import tensorflow as tf
import numpy as np
import os

# Load and preprocess data
print("Przetwarzanie danych...")
pre = DataPreprocessor(
    train_path="data/train.parquet",
    test_path="data/test.parquet",
    target_channels_path="data/target_channels.csv",
    mode="subset",
    window_size=100
)

print("Dane przetworzone")
print("Ładowanie danych...")
X_train, y_train, X_test = pre.prepare()
print("Dane załadowane")

# Reduce memory usage
X_train = X_train[:1_000_000]
y_train = y_train[:1_000_000]

# Check for NaN or Inf in data
print("Sprawdzanie danych wejściowych na NaN i Inf...")
print("NaN in X_train:", np.isnan(X_train).any())
print("Inf in X_train:", np.isinf(X_train).any())
print("NaN in y_train:", np.isnan(y_train).any())
print("Inf in y_train:", np.isinf(y_train).any())

# Build model
print("Budowanie modelu...")
model = LSTMAutoencoder(window_size=100, n_channels=X_train.shape[2])
print("Model zbudowany")
print("Podsumowanie modelu:")
model.summary()

# Train model
print("Trenowanie modelu...")
history = model.train(X_train, epochs=10, batch_size=32, validation_split=0.2)
print("Model wytrenowany")

# Save model
print("Zapisywanie modelu...")
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model.save(f"models/autoencoder_model_{timestamp}.keras")

# Save training history
history_df = pd.DataFrame(history.history)
history_df.to_csv(f"models/training_history_{timestamp}.csv", index=False)
print("Model zapisany")

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
plt.plot(history.history['mae'], label='MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title('Training History')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('training_history.png')
plt.show()
