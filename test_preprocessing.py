from data_preprocessor import DataPreprocessor

if __name__ == "__main__":
    print("🔄 Przetwarzanie danych...")
    pre = DataPreprocessor(
        train_path="data/train.parquet",
        test_path="data/test.parquet",
        target_channels_path="data/target_channels.csv",
        mode="subset",  # albo "full"
        window_size=100
    )

    X_train, y_train, X_test = pre.prepare()

    print("✅ Dane przetworzone")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)

    print("X_train sample:", X_train[0].shape)
    print("Pierwsze 5 etykiet:", y_train[:5])
else:
    print("Ten skrypt jest tylko testem i nie powinien być importowany.")
