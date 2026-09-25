"""Isolation Forest: cortes de uma árvore real do scikit-learn, caminho de isolamento e floresta."""

from __future__ import annotations

import math

import numpy as np
import pygame

from tc2_iot.models.isolation_forest import IsolationForestModel

from . import draw as d
from .data import Toy, params

N_GRID = 100
TREES_PER_STEP = 5


def c(n: int) -> float:
    """Average path length of an unsuccessful BST search over n points (normalizes depths)."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2 * (math.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n


def _depths(tree) -> np.ndarray:
    depth = np.zeros(tree.node_count, dtype=np.int64)
    for node in range(tree.node_count):  # children always have larger ids than parents
        for child in (tree.children_left[node], tree.children_right[node]):
            if child != -1:
                depth[child] = depth[node] + 1
    return depth


def path_lengths(est, feats, X: np.ndarray) -> np.ndarray:
    """h(x) = depth of the leaf reached + c(points still together in that leaf)."""
    t = est.tree_
    leaves = est.apply(np.asarray(X, dtype=np.float32)[:, feats])
    return _depths(t)[leaves] + np.array([c(n) for n in t.n_node_samples[leaves]])


def forest_score(forest, X: np.ndarray, k: int) -> np.ndarray:
    """Score after averaging the first k trees; with k = all trees it equals -score_samples."""
    h = sum(path_lengths(e, f, X) for e, f in zip(forest.estimators_[:k], forest.estimators_features_[:k]))
    return 2.0 ** (-(h / k) / c(forest.max_samples_))


def tree_cuts(est, feats) -> list[tuple[tuple[float, float, float, float], int, float, int]]:
    """Cuts in breadth-first order: (region x0, y0, x1, y1), feature, threshold, depth."""
    t, out = est.tree_, []
    queue = [(0, (-d.LIM, -d.LIM, d.LIM, d.LIM), 0)]
    while queue:
        node, (x0, y0, x1, y1), depth = queue.pop(0)
        left, right = t.children_left[node], t.children_right[node]
        if left == -1:
            continue
        f, thr = int(feats[t.feature[node]]), float(t.threshold[node])
        out.append(((x0, y0, x1, y1), f, thr, depth))
        if f == 0:
            queue += [(left, (x0, y0, thr, y1), depth + 1), (right, (thr, y0, x1, y1), depth + 1)]
        else:
            queue += [(left, (x0, y0, x1, thr), depth + 1), (right, (x0, thr, x1, y1), depth + 1)]
    return out


def _probe_path(est, feats, p) -> tuple[list[int], tuple, int]:
    """Indices (into tree_cuts order) of the cuts p passes, its leaf region and leaf sample count."""
    cuts, t = tree_cuts(est, feats), est.tree_
    region, node, used = (-d.LIM, -d.LIM, d.LIM, d.LIM), 0, []
    while t.children_left[node] != -1:
        f, thr = int(feats[t.feature[node]]), float(t.threshold[node])
        used.append(next(i for i, (r, cf, ct, _) in enumerate(cuts) if r == region and cf == f and ct == thr))
        x0, y0, x1, y1 = region
        go_left = p[f] <= thr
        node = t.children_left[node] if go_left else t.children_right[node]
        if f == 0:
            region = (x0, y0, thr, y1) if go_left else (thr, y0, x1, y1)
        else:
            region = (x0, y0, x1, thr) if go_left else (x0, thr, x1, y1)
    return used, region, int(t.n_node_samples[node])


class IFScene:
    title = "Isolation Forest"

    def __init__(self, toy: Toy):
        self.toy = toy
        self.model = IsolationForestModel(params("isolation_forest"), toy.seed, 1, len(toy.train)).fit(toy.train)
        self.forest = self.model.model
        self.grid = d.grid(N_GRID)
        self.probe: tuple[float, float] | None = None
        self.caught = None
        self._tree(0)

    # phases: "tree" (one tree's cuts appear) -> "forest" (trees averaged) -> "final" (threshold)
    def _tree(self, i: int) -> None:
        self.phase, self.tree_i, self.shown = "tree", i % len(self.forest.estimators_), 0
        self.est, self.feats = self.forest.estimators_[self.tree_i], self.forest.estimators_features_[self.tree_i]
        self.cuts = tree_cuts(self.est, self.feats)
        self.sub = self.toy.train[self.forest.estimators_samples_[self.tree_i]]

    def _forest(self) -> None:
        self.phase, self.k, self.h_sum = "forest", 0, np.zeros(len(self.grid))

    @property
    def interval(self) -> float:
        return 0.25 if self.phase == "tree" else 0.06

    def done(self) -> bool:
        return self.phase == "final"

    def step(self) -> None:
        if self.phase == "tree":
            if self.shown < len(self.cuts):
                self.shown += 1
            else:
                self._forest()
        elif self.phase == "forest":
            est, feats = self.forest.estimators_, self.forest.estimators_features_
            for e, f in zip(est[self.k:self.k + TREES_PER_STEP], feats[self.k:self.k + TREES_PER_STEP]):
                self.h_sum += path_lengths(e, f, self.grid)
            self.k = min(self.k + TREES_PER_STEP, len(est))
            if self.k == len(est):
                self.phase = "final"
                self.thr = d.threshold(self.model.anomaly_score(self.toy.calib))
                self.caught = self.model.anomaly_score(self.toy.attacks) > self.thr

    def handle(self, event, view: d.View) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and view.contains(*event.pos):
            self.probe = view.to_world(*event.pos)
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
            self._tree(self.tree_i + 1 if self.phase == "tree" else 0)
            self.shown = len(self.cuts)
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_f and self.phase == "tree":
            self._forest()

    def _grid_score(self) -> np.ndarray:
        return 2.0 ** (-(self.h_sum / max(self.k, 1)) / c(self.forest.max_samples_))

    def draw(self, surf, view: d.View) -> None:
        if self.phase == "tree":
            d.points(surf, view, self.toy.train, d.DIM, 1)
            d.points(surf, view, self.sub, d.BENIGN, 3)
            path = set()
            if self.probe:
                used, (x0, y0, x1, y1), _ = _probe_path(self.est, self.feats, self.probe)
                path = set(used)
                a, b = view.to_screen((x0, y1)), view.to_screen((x1, y0))
                leaf = pygame.Surface((b[0] - a[0], b[1] - a[1]), pygame.SRCALPHA)
                leaf.fill((*d.ACCENT, 60))
                surf.blit(leaf, a)
            for i, ((x0, y0, x1, y1), f, t, _) in enumerate(self.cuts[:self.shown]):
                seg = ((t, y0), (t, y1)) if f == 0 else ((x0, t), (x1, t))
                hot = i in path or (not path and i == self.shown - 1)
                pygame.draw.line(surf, d.ACCENT if hot else (150, 150, 170),
                                 view.to_screen(seg[0]), view.to_screen(seg[1]), 3 if hot else 1)
        else:
            d.heatmap(surf, view, self._grid_score(), N_GRID)
            if self.phase == "final":
                d.contour(surf, view, (self._grid_score() > self.thr).reshape(N_GRID, N_GRID), d.ACCENT)
            d.points(surf, view, self.toy.train, d.BENIGN, 1)
        d.attacks(surf, view, self.toy.attacks, self.caught)
        if self.probe:
            pygame.draw.circle(surf, d.ACCENT, view.to_screen(self.probe), 6, 2)

    def caption(self) -> list[str]:
        cap = self.forest.max_samples_
        if self.phase == "tree":
            lines = [f"Tree {self.tree_i + 1}/{len(self.forest.estimators_)}: cut {self.shown}/{len(self.cuts)}",
                     f"The tree only sees {cap} random benign points (bright). Each cut picks a random feature and a "
                     "random value between that region's min and max, until every point is alone or depth "
                     f"{math.ceil(math.log2(cap))} is reached.",
                     "Dense areas need many cuts; lonely points are isolated after a few."]
            if self.probe:
                used, _, n = _probe_path(self.est, self.feats, self.probe)
                lines += [f"Clicked point: {len(used)} cuts to reach its leaf, which holds {n} training point(s). "
                          f"h(x) = {len(used)} + c({n}) = {len(used) + c(n):.2f}. Shorter = more anomalous."]
            else:
                lines += ["Click anywhere to follow a point down this tree."]
            return lines + ["", "N: next tree   F: jump to forest"]
        lines = [f"Forest: {self.k}/{len(self.forest.estimators_)} trees averaged",
                 f"score = 2^(-mean h(x) / c({cap})). Bright = short average path = anomalous.",
                 "Look for the rectangular (axis-aligned) streaks, and how the score stops growing once you "
                 "leave the benign data: a point far away lands in the same leaf as the most extreme benign point."]
        if self.probe:
            s = forest_score(self.forest, np.array([self.probe]), max(self.k, 1))[0]
            lines += [f"Clicked point score: {s:.3f}" + (f" (threshold {self.thr:.3f})" if self.done() else "")]
        if self.done():
            lines += ["Yellow line: threshold at the 99th percentile of separate benign calibration scores "
                      "(same rule as the thesis, FPR about 1%).", d.caught_line(self.toy, self.caught)]
        return lines
