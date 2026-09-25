"""Interface comum mínima. anomaly_score: maior valor = mais anômalo. Limiar sempre externo."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class AnomalyModel:
    name: str = ""

    def fit(self, X_train: np.ndarray, X_validation_fit: np.ndarray | None = None) -> "AnomalyModel":
        raise NotImplementedError

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: Path) -> list[Path]:
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path) -> "AnomalyModel":
        raise NotImplementedError

    def resolved_params(self) -> dict:
        """Effective parameters of the installed library version, for the protocol record."""
        raise NotImplementedError

    def fit_info(self) -> dict:
        return {}


def build(name: str, cfg: dict, seed: int, threads: int, n_train: int, n_features: int) -> AnomalyModel:
    from .autoencoder import Autoencoder
    from .isolation_forest import IsolationForestModel
    from .sgd_ocsvm import SGDOneClassSVMModel

    m = cfg["models"][name]
    if name == "isolation_forest":
        return IsolationForestModel(m, seed, threads, n_train)
    if name == "sgd_one_class_svm":
        return SGDOneClassSVMModel(m, seed, n_train, n_features)
    if name == "autoencoder":
        return Autoencoder(m, seed, n_features, cfg["experiment"]["device"])
    raise ValueError(f"modelo desconhecido: {name}")


def load(name: str, path: Path) -> AnomalyModel:
    from .autoencoder import Autoencoder
    from .isolation_forest import IsolationForestModel
    from .sgd_ocsvm import SGDOneClassSVMModel

    return {"isolation_forest": IsolationForestModel, "sgd_one_class_svm": SGDOneClassSVMModel,
            "autoencoder": Autoencoder}[name].load(path)
