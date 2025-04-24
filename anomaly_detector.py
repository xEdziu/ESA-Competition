import numpy as np
import pandas as pd
import tensorflow as tf
from model import LSTMAutoencoder
from data_preprocessor import DataPreprocessor
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os

class AnomalyDetector:
    def __init__(self, model_path, window_size=100, mode="subset"):
        self.model = tf.keras.models.load_model(model_path)
        self.window_size = window_size
        self.mode = mode

    def compute_reconstruction_errors(self, X, batch_size=64, verbose=1, use_gpu=True, save_temp=False, temp_dir='temp', num_prefetch=2):
        # Sprawdź, czy GPU jest dostępne
        if use_gpu and tf.config.list_physical_devices('GPU'):
            print("Using GPU for computation")
        else:
            print("Using CPU for computation")
            
        # Przygotuj miejsce do zapisywania wyników tymczasowych
        if save_temp:
            os.makedirs(temp_dir, exist_ok=True)
            temp_file = os.path.join(temp_dir, 'temp_errors.npy')
        
        total_batches = (len(X) + batch_size - 1) // batch_size
        if verbose > 0:
            print(f"Processing {total_batches} batches")
        
        errors = []
        
        # Zamiast przetwarzać cały dataset na raz, dzielimy go na mniejsze części ("chunks")
        # aby lepiej kontrolować użycie pamięci
        chunk_size = min(10000, len(X))  # Nie większe niż 10000 próbek na raz
        for chunk_start in range(0, len(X), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(X))
            X_chunk = X[chunk_start:chunk_end]
            
            if verbose > 0:
                print(f"Processing chunk {chunk_start//chunk_size + 1}/{(len(X) + chunk_size - 1)//chunk_size}, "
                      f"samples {chunk_start} to {chunk_end-1}")
            
            # Korzystaj z tf.data.Dataset tylko dla aktualnego fragmentu danych
            # Dodatkowo używamy opcji prefetch dla lepszej wydajności, ale z ograniczoną liczbą buforowanych elementów
            dataset = tf.data.Dataset.from_tensor_slices(X_chunk).batch(batch_size).prefetch(num_prefetch)
            
            chunk_errors = []
            for i, batch in enumerate(dataset):
                if verbose > 0 and i % 10 == 0:
                    batch_num = i + (chunk_start // batch_size)
                    print(f"  Processing batch {batch_num + 1}/{total_batches}")
                
                # Wykonaj predykcję
                X_pred = self.model.predict(batch, verbose=0)
                
                # Oblicz błędy rekonstrukcji
                batch_errors = np.mean(np.mean((batch.numpy() - X_pred) ** 2, axis=2), axis=1)
                chunk_errors.extend(batch_errors)
                
                # Opcjonalnie zwolnij pamięć
                tf.keras.backend.clear_session()
            
            # Zapisz wyniki fragmentu
            if save_temp:
                if chunk_start == 0:
                    np.save(temp_file, np.array(chunk_errors))
                else:
                    try:
                        existing = np.load(temp_file)
                        np.save(temp_file, np.concatenate([existing, np.array(chunk_errors)]))
                    except Exception as e:
                        print(f"Błąd przy zapisie tymczasowym: {e}")
                        errors.extend(chunk_errors)
            else:
                errors.extend(chunk_errors)
        
        # Wczytaj wszystkie błędy, jeśli zapisano je do pliku
        if save_temp and not errors:
            try:
                errors = np.load(temp_file)
                if verbose > 0:
                    print(f"Loaded errors from temporary file: {temp_file}")
            except Exception as e:
                print(f"Błąd przy wczytywaniu z pliku tymczasowego: {e}")
        
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
