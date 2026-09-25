"""Dados 2D de brinquedo: dois grupos benignos e alguns "ataques" de três tipos."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import yaml

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "initial.yaml"
CENTERS = np.array([(-1.2, -0.8), (1.2, 0.9)])
STD = 0.35


class Toy(NamedTuple):
    train: np.ndarray    # benign, used to fit
    calib: np.ndarray    # benign, used only for the 99th-percentile threshold
    attacks: np.ndarray
    kinds: list[str]     # "far" (flood-like), "gap" (normal per feature, odd combination), "edge" (subtle)
    seed: int


def _benign(rng: np.random.Generator, n: int) -> np.ndarray:
    return np.vstack([rng.normal(c, STD, (n // 2, 2)) for c in CENTERS]).astype(np.float32)


def make_toy(seed: int) -> Toy:
    rng = np.random.default_rng(seed)
    train, calib = _benign(rng, 600), _benign(rng, 400)
    far = np.array([(3.2, -3.0), (-3.3, 3.0), (3.4, 3.2), (-0.2, -3.5)])
    gap = np.array([(0.0, 0.0), (-0.4, 0.4), (0.5, -0.3)])
    # just beyond ~3 std of a blob, like recon traffic that looks almost normal
    edge = CENTERS[[0, 1, 1]] + np.array([(-1.1, 0.4), (1.1, -0.3), (0.2, 1.15)])
    attacks = (np.vstack([far, gap, edge]) + rng.normal(0, 0.08, (10, 2))).astype(np.float32)
    return Toy(train, calib, attacks, ["far"] * 4 + ["gap"] * 3 + ["edge"] * 3, seed)


def params(model: str) -> dict:
    """Hyperparameters the thesis uses, from configs/initial.yaml."""
    return dict(yaml.safe_load(CONFIG.read_text())["models"][model])
