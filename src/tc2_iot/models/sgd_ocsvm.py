from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.pipeline import make_pipeline

from .base import AnomalyModel


class SGDOneClassSVMModel(AnomalyModel):
    """SGD One-Class SVM with RBF approximation by Nyström (main) or linear (optional variant)."""

    name = "sgd_one_class_svm"

    def __init__(self, params: dict, seed: int, n_train: int, n_features: int):
        self.variant = params["variant"]
        gamma = 1.0 / n_features if params["gamma"] == "inverse_feature_count" else float(params["gamma"])
        svm = SGDOneClassSVM(nu=params["nu"], max_iter=params["max_iter"], tol=params["tol"],
                             shuffle=params["shuffle"], average=params["average"], random_state=seed)
        if self.variant == "nystroem_rbf":
            nys = Nystroem(kernel="rbf", gamma=gamma, n_components=min(params["components"], n_train), random_state=seed)
            self.model = make_pipeline(nys, svm)
        else:
            self.model = make_pipeline(svm)
        self.warnings: list[str] = []

    def fit(self, X_train, X_validation_fit=None):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            self.model.fit(X_train)
        self.warnings = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
        return self

    def anomaly_score(self, X):
        # decision_function: positive = inlier; negate so higher = more anomalous
        return -self.model.decision_function(X).astype(np.float64)

    def save(self, path: Path):
        p = Path(path) / "model.joblib"
        joblib.dump(self, p)
        return [p]

    @classmethod
    def load(cls, path: Path):
        return joblib.load(Path(path) / "model.joblib")

    def resolved_params(self):
        steps = {name: est.get_params() for name, est in self.model.steps}
        return {"variant": self.variant, "steps": steps,
                "display_name": "SGD One-Class SVM com aproximação RBF por Nyström" if self.variant == "nystroem_rbf"
                else "SGD One-Class SVM linear (variante adicional)"}

    def fit_info(self):
        svm = self.model.steps[-1][1]
        return {"n_iter_": int(svm.n_iter_), "convergence_warnings": self.warnings,
                "converged": not self.warnings}
