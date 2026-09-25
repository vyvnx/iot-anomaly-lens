"""Métricas binárias (malicioso = positivo) e detecção por categoria. Denominador nulo -> None + motivo."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import AMBIGUOUS, CATEGORIES


def _ratio(num: int, den: int, why: str) -> tuple[float | None, str | None]:
    return (num / den, None) if den else (None, why)


def binary_metrics(y: np.ndarray, score: np.ndarray, pred: np.ndarray) -> dict:
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    out: dict = {"n": int(y.size), "n_benign": tn + fp, "n_attack": tp + fn, "TN": tn, "FP": fp, "FN": fn, "TP": tp}
    undefined = {}
    for name, num, den, why in [
        ("recall", tp, tp + fn, "sem ataques"),
        ("fpr", fp, fp + tn, "sem benignos"),
        ("precision", tp, tp + fp, "nenhuma predição positiva"),
        ("f1", 2 * tp, 2 * tp + fp + fn, "sem positivos reais nem preditos"),
        ("tnr", tn, tn + fp, "sem benignos"),
    ]:
        out[name], reason = _ratio(num, den, why)
        if reason:
            undefined[name] = reason
    if out["recall"] is not None and out["tnr"] is not None:
        out["balanced_accuracy"] = (out["recall"] + out["tnr"]) / 2
    else:
        out["balanced_accuracy"] = None
        undefined["balanced_accuracy"] = "requer benignos e ataques"
    if len(np.unique(y)) == 2:
        out["auroc"] = float(roc_auc_score(y, score))
        # average precision (step-wise sum), not the trapezoidal area under the pr curve
        out["average_precision"] = float(average_precision_score(y, score))
    else:
        out["auroc"] = out["average_precision"] = None
        undefined["auroc"] = undefined["average_precision"] = "requer as duas classes"
    out["undefined_reasons"] = "; ".join(f"{k}: {v}" for k, v in undefined.items())
    return out


def category_metrics(df: pd.DataFrame) -> list[dict]:
    """df: rows of one model/seed/split with columns stratum, category, label, benign, pred.

    The 7 categories use the vectors of their stratum (single-label and same-category multi-label).
    attack_category_ambiguous is reported separately and excluded from the 7-category macro.
    """
    rows = []
    att = df[~df["benign"]]

    def row(level, key, cat, g):
        n, tp = len(g), int((g["pred"] == 1).sum())
        return {"nivel": level, "grupo": key, "categoria": cat, "N": n, "TP": tp, "FN": n - tp,
                "recall": tp / n if n else None, "motivo_indefinido": "" if n else "sem registros"}

    for c in CATEGORIES:
        rows.append(row("categoria", c, c, att[att["stratum"] == c]))
    cats = [r["recall"] for r in rows]
    rows.append({"nivel": "macro_categorias", "grupo": "média simples do recall das 7 categorias",
                 "categoria": None, "N": int(att["stratum"].isin(CATEGORIES).sum()), "TP": None, "FN": None,
                 "recall": float(np.mean(cats)) if all(c is not None for c in cats) else None,
                 "motivo_indefinido": "" if all(c is not None for c in cats) else "categoria sem registros"})
    rows.append(row("categoria_ambigua", AMBIGUOUS, None, att[att["stratum"] == AMBIGUOUS]))
    for label in sorted(att["label"].unique()):
        g = att[att["label"] == label]
        rows.append(row("rotulo_original", label, g["stratum"].iloc[0], g))
    return rows
