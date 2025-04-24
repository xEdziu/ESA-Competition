import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from anomaly_detector import AnomalyDetector

# === KONFIGURACJA ===
TEMP_FILE = "temp/temp_errors.npy"
WINDOW_SIZE = 100
Y_TRAIN_FILE = "data/train.parquet"
QUANTILES_TO_TEST = [0.950, 0.960, 0.970, 0.980, 0.985, 0.990, 0.995, 0.997, 0.999, 0.9995]
MODEL_PATH = "models/autoencoder_model_20250416-142029.keras"

# === Wczytaj dane ===
print("📥 Wczytywanie danych błędów i etykiet...")
errors = np.load(TEMP_FILE)
train_df = pd.read_parquet(Y_TRAIN_FILE)

# Sprawdź długości i dokonaj odpowiedniego wyrównania
print(f"Długość errors: {len(errors)}")
print(f"Długość train_df: {len(train_df)}")

# Usuń pierwsze WINDOW_SIZE próbek z train_df - te, dla których nie mamy błędów rekonstrukcji
y_true = train_df['is_anomaly'].values[WINDOW_SIZE:]

# Jeśli nadal długości się nie zgadzają, dostosuj odpowiednio
if len(y_true) > len(errors):
    print(f"⚠️ Usuwam ostatnie {len(y_true) - len(errors)} próbki z y_true dla dopasowania")
    y_true = y_true[:len(errors)]
elif len(errors) > len(y_true):
    print(f"⚠️ Usuwam ostatnie {len(errors) - len(y_true)} próbki z errors dla dopasowania")
    errors = errors[:len(y_true)]

print(f"Po wyrównaniu: długość y_true: {len(y_true)}, długość errors: {len(errors)}")

# === Inicjalizacja detektora (tylko do progu / predykcji) ===
detector = AnomalyDetector(model_path=MODEL_PATH, window_size=WINDOW_SIZE)

# === Testuj różne progi ===
results = []
for q in QUANTILES_TO_TEST:
    threshold = detector.get_threshold(errors, quantile=q)
    y_pred = detector.detect_anomalies(errors, threshold)
    
    # Upewnij się, że długości są zgodne
    assert len(y_true) == len(y_pred), f"Niezgodność długości: y_true: {len(y_true)}, y_pred: {len(y_pred)}"
    
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    results.append((q, threshold, precision, recall, f1))

    print(f"\n🔎 Quantile: {q:.4f}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# === Wykres F1-score vs quantile ===
quantiles, thresholds, precisions, recalls, f1s = zip(*results)
plt.figure(figsize=(8, 5))
plt.plot(quantiles, f1s, marker='o', label='F1 Score')
plt.plot(quantiles, precisions, marker='x', linestyle='--', label='Precision')
plt.plot(quantiles, recalls, marker='s', linestyle=':', label='Recall')
plt.xlabel("Quantile Threshold")
plt.ylabel("Score")
plt.title("Precision, Recall, F1 vs. Threshold Quantile")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Najlepszy wynik? ===
best_idx = np.argmax(f1s)
print(f"\n🏆 Najlepszy F1: {f1s[best_idx]:.4f} przy quantile {quantiles[best_idx]} (threshold={thresholds[best_idx]:.2f})")

best_threshold = thresholds[best_idx]
y_best = detector.detect_anomalies(errors, best_threshold)

# np. zapis do pliku .npy
np.save("temp/y_pred_best.npy", y_best)
