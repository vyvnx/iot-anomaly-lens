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


from visualizer.draw import grid  # noqa: E402
from visualizer.scene_if import IFScene, forest_score, tree_cuts  # noqa: E402


def test_forest_score_matches_sklearn():
    s = IFScene(make_toy(0))
    X = grid(30)
    assert np.allclose(forest_score(s.model.model, X, 200), s.model.anomaly_score(X))


def test_tree_cuts_are_internal_nodes_inside_their_region():
    s = IFScene(make_toy(0))
    est, feats = s.model.model.estimators_[0], s.model.model.estimators_features_[0]
    cuts = tree_cuts(est, feats)
    assert len(cuts) == int((est.tree_.children_left != -1).sum())
    for (x0, y0, x1, y1), f, t, _ in cuts:
        assert (x0 <= t <= x1) if f == 0 else (y0 <= t <= y1)


def test_if_step_after_done_is_noop():
    s = IFScene(make_toy(0))
    while not s.done():
        s.step()
    s.step()
    assert s.done() and s.caught.sum() >= 4  # the four far attacks at least


from visualizer.scene_svm import SVMScene  # noqa: E402


def test_svm_final_snapshot_equals_full_fit():
    s = SVMScene(make_toy(0))
    X = grid(20)
    assert np.allclose(s.snapshot(s.n_iter).anomaly_score(X), s.snapshot(2000).anomaly_score(X))


def test_svm_param_change_restarts_training():
    s = SVMScene(make_toy(0))
    for _ in range(40):
        s.step()
    s.set_params(nu=0.1)
    assert s.nu == 0.1 and s.epoch == 0 and s.phase == "train" and s.caught is None
    while not s.done():
        s.step()
    s.step()
    assert s.done() and s.caught.sum() >= 4
