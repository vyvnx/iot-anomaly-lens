"""Visualizador didático (visualizer/): lógica das cenas e teste de fumaça sem janela."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame = pytest.importorskip("pygame")  # optional `viz` group
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visualizer.data import make_toy  # noqa: E402
from visualizer.draw import View  # noqa: E402


def test_toy_is_deterministic_and_shaped():
    a, b = make_toy(7), make_toy(7)
    assert a.train.shape == (600, 2) and a.calib.shape == (400, 2)
    assert np.array_equal(a.attacks, b.attacks) and len(a.kinds) == len(a.attacks)
    assert not np.array_equal(make_toy(8).train, a.train)


def test_view_roundtrip():
    v = View(pygame.Rect(10, 20, 400, 400))
    x, y = v.to_world(*v.to_screen((1.5, -2.0)))
    assert abs(x - 1.5) < 0.03 and abs(y + 2.0) < 0.03
