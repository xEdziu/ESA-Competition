import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List

class DataPreprocessor:
    def __init__(self,
                 train_path: str,
                 test_path: str,
                 target_channels_path: str,
                 mode: str = "subset",  # "subset" (41–46) or "full"
                 window_size: int = 100):
        self.train_path = train_path
        self.test_path = test_path
        self.target_channels_path = target_channels_path
        self.mode = mode
        self.window_size = window_size

        self.target_channels = self._load_target_channels()
        self.scaler = StandardScaler()

    def _load_target_channels(self) -> List[str]:
        df = pd.read_csv(self.target_channels_path)
        full_list = df.values.flatten().tolist()
        if self.mode == "subset":
            # Kanały 41–46 (numerycznie, ale jako channel_XX)
            return [f"channel_{i}" for i in range(41, 47)]
        return full_list

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df = pd.read_parquet(self.train_path)
        test_df = pd.read_parquet(self.test_path)
        return train_df, test_df

    def scale_data(self,
                   train_df: pd.DataFrame,
                   test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X_train = train_df[self.target_channels].values
        X_test = test_df[self.target_channels].values

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled

    def create_sequences(self,
                         data: np.ndarray,
                         labels: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(data) - self.window_size):
            seq_x = data[i:i + self.window_size]
            X.append(seq_x)
            if labels is not None:
                y.append(labels[i + self.window_size - 1])  # label na końcu okna
        return np.array(X), np.array(y) if labels is not None else None

    def prepare(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        train_df, test_df = self.load_data()
        X_train_scaled, X_test_scaled = self.scale_data(train_df, test_df)

        y_train = train_df['is_anomaly'].values

        X_train_seq, y_train_seq = self.create_sequences(X_train_scaled, y_train)
        X_test_seq, _ = self.create_sequences(X_test_scaled)

        return X_train_seq, y_train_seq, X_test_seq
