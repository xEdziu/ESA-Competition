from anomaly_detector import AnomalyDetector
from data_preprocessor import DataPreprocessor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# === CONFIGURATION ===
MODEL_PATH = "models/autoencoder_model_20250416-142029.keras"
TRAIN_PATH = "data/train.parquet"
TEST_PATH = "data/test.parquet"
CHANNELS_PATH = "data/target_channels.csv"
OUTPUT_PATH = "submission.parquet"
WINDOW_SIZE = 100
MODE = "subset"  # or "full"

# === LOAD DATA ===
print("📥 Wczytywanie danych...")
pre = DataPreprocessor(
    train_path=TRAIN_PATH,
    test_path=TEST_PATH,
    target_channels_path=CHANNELS_PATH,
    mode=MODE,
    window_size=WINDOW_SIZE
)
X_train, y_train, X_test = pre.prepare()
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

# === INIT DETECTOR ===
detector = AnomalyDetector(model_path=MODEL_PATH, window_size=WINDOW_SIZE, mode=MODE)

# === TRAIN ERROR ANALYSIS ===
print("🧠 Obliczanie błędów rekonstrukcji na zbiorze treningowym...")
errors_train = detector.compute_reconstruction_errors(X_train)
detector.plot_error_distribution(errors_train)

# === THRESHOLD ===
threshold = detector.get_threshold(errors_train, quantile=0.995)
print(f"📉 Ustalony threshold (0.5% kwantyl): {threshold:.6f}")

# === WYKRES: błąd vs. indeks ===
plt.figure(figsize=(10, 4))
plt.plot(errors_train, label="Reconstruction Error")
plt.axhline(y=threshold, color='red', linestyle='--', label=f"Threshold = {threshold:.4f}")
plt.title("Reconstruction Error per Window (Train)")
plt.xlabel("Sample Index")
plt.ylabel("Error")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()

# === METRYKI DLA X_TRAIN ===
y_train_eval = y_train[WINDOW_SIZE:]  # wyrównanie z błędami
y_pred_train = detector.detect_anomalies(errors_train, threshold)

precision = precision_score(y_train_eval, y_pred_train)
recall = recall_score(y_train_eval, y_pred_train)
f1 = f1_score(y_train_eval, y_pred_train)
print("\n🔬 Metryki na X_train:")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# === CONFUSION MATRIX ===
cm = confusion_matrix(y_train_eval, y_pred_train)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomaly"])
plt.figure(figsize=(5, 5))
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix (Train)")
plt.grid(False)
plt.tight_layout()
plt.show()

# === TEST PREDICTION ===
print("🔍 Przewidywanie anomalii na zbiorze testowym...")
errors_test = detector.compute_reconstruction_errors(X_test)
detector.plot_error_distribution(errors_test)

y_pred = detector.detect_anomalies(errors_test, threshold)
test_ids = test_df['id'].values

# === ZAPIS WYNIKÓW ===
detector.generate_submission(test_ids=test_ids, y_pred=y_pred, output_path=OUTPUT_PATH)
print("✅ Wszystko gotowe!")
