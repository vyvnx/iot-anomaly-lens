from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from .base import AnomalyModel


class IsolationForestModel(AnomalyModel):
    name = "isolation_forest"

    def __init__(self, params: dict, seed: int, threads: int, n_train: int):
        self.model = IsolationForest(
            n_estimators=params["n_estimators"], max_samples=min(params["max_samples"], n_train),
            max_features=params["max_features"], bootstrap=params["bootstrap"],
            contamination=params["contamination"], random_state=seed, n_jobs=threads,
        )

    def fit(self, X_train, X_validation_fit=None):
        self.model.fit(X_train)
        return self

    def anomaly_score(self, X):
        # score_samples: higher = more normal; negate so higher = more anomalous
        return -self.model.score_samples(X).astype(np.float64)

    def save(self, path: Path):
        p = Path(path) / "model.joblib"
        joblib.dump(self, p)
        return [p]

    @classmethod
    def load(cls, path: Path):
        return joblib.load(Path(path) / "model.joblib")

    def resolved_params(self):
        return {k: v for k, v in self.model.get_params().items()}
