import numpy as np
import pandas as pd
import tensorflow as tf
from model import LSTMAutoencoder
from data_preprocessor import DataPreprocessor
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

class AnomalyDetector:
    def __init__(self, model_path, window_size=100, mode="subset"):
        self.model = tf.keras.models.load_model(model_path)
        self.window_size = window_size
        self.mode = mode

    def compute_reconstruction_errors(self, X, batch_size=64):
        errors = []

        for i in range(0, len(X), batch_size):
            print(f"Processing batch {i // batch_size + 1}/{len(X) // batch_size + 1}")
            batch = X[i:i+batch_size]
            X_pred = self.model.predict(batch, verbose=0)
            batch_errors = np.mean(np.mean((batch - X_pred) ** 2, axis=2), axis=1)
            errors.extend(batch_errors)

        return np.array(errors)

    def plot_error_distribution(self, errors):
        plt.figure(figsize=(10, 5))
        plt.hist(errors, bins=100, color='skyblue', edgecolor='black')
        plt.title("Reconstruction Error Distribution")
        plt.xlabel("MSE")
        plt.ylabel("Count")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def get_threshold(self, errors, quantile=0.995):
        return np.quantile(errors, quantile)

    def detect_anomalies(self, errors, threshold):
        return (errors > threshold).astype(int)

    def generate_submission(self, test_ids, y_pred, output_path="submission.parquet"):
        timestamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
        output_path = output_path.replace(".parquet", f"_{timestamp}.parquet")
        submission = pd.DataFrame({"id": test_ids[self.window_size:], "is_anomaly": y_pred})
        submission.to_parquet(output_path, index=False)
        print(f"✅ Saved submission to {output_path}")
