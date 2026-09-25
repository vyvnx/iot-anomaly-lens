"""Limiar por quantil dos escores de benignos de validation_calibration."""

from __future__ import annotations

import numpy as np


def calibrate(scores: np.ndarray, target_fpr: float, min_warning: int = 1000) -> dict:
    s = np.asarray(scores, dtype=np.float64)
    if s.size == 0 or not np.isfinite(s).all():
        raise ValueError("escores de calibração vazios ou não finitos")
    q = 1.0 - target_fpr
    thr = float(np.quantile(s, q, method="higher"))
    above = int((s > thr).sum())
    out = {
        "rule": f"threshold = quantile(scores, {q}, method='higher'); pred = score > threshold",
        "target_fpr": target_fpr,
        "n_calibration": int(s.size),
        "threshold": thr,
        "ties_at_threshold": int((s == thr).sum()),
        "calibration_fp": above,
        "calibration_fpr": above / s.size,
        "warnings": [],
    }
    # ties are never broken randomly: the empirical fpr can end up below the target
    if s.size < min_warning:
        out["warnings"].append(f"calibração com {s.size} < {min_warning} registros: não é estimativa precisa de FPR {target_fpr}")
    return out


def predict(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(scores) > threshold).astype(np.int8)
