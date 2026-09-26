"""Pré-processamento comum: remoção de constantes do treino e StandardScaler ajustado só em train."""

from __future__ import annotations

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

F32_MAX = np.finfo(np.float32).max


class Preprocessor:
    def __init__(self, feature_names: list[str], drop_train_constants: bool = True):
        self.feature_names = list(feature_names)
        self.drop_train_constants = drop_train_constants

    def fit(self, X_train: np.ndarray) -> "Preprocessor":
        X = np.asarray(X_train, dtype=np.float64)
        if not np.isfinite(X).all():
            raise ValueError("treino contém valores não finitos")
        constant = np.ptp(X, axis=0) == 0 if self.drop_train_constants else np.zeros(X.shape[1], bool)
        self.kept_idx = np.flatnonzero(~constant)
        self.dropped = [self.feature_names[i] for i in np.flatnonzero(constant)]
        self.kept = [self.feature_names[i] for i in self.kept_idx]
        # zero variance is handled above by dropping; sklearn would otherwise silently use scale 1
        self.scaler = StandardScaler().fit(X[:, self.kept_idx])
        if (self.scaler.scale_ == 0).any():
            raise ValueError("escala nula após remoção de constantes")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = self.scaler.transform(np.asarray(X, dtype=np.float64)[:, self.kept_idx])
        if not np.isfinite(Z).all():
            raise ValueError("valores não finitos após padronização")
        if np.abs(Z).max(initial=0) > F32_MAX:
            raise OverflowError("valor padronizado excede o intervalo de float32")
        return Z.astype(np.float32)

    @property
    def n_features_out(self) -> int:
        return len(self.kept_idx)

    def summary(self) -> dict:
        return {"input_features": self.feature_names, "kept": self.kept, "dropped_train_constants": self.dropped,
                "mean": dict(zip(self.kept, self.scaler.mean_.tolist())),
                "scale": dict(zip(self.kept, self.scaler.scale_.tolist())),
                "fit_rows": int(self.scaler.n_samples_seen_)}

    def save(self, path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "Preprocessor":
        return joblib.load(path)
