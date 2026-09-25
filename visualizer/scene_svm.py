"""SGD One-Class SVM com Nyström: pontos de referência, similaridade RBF e fronteira época a época."""

from __future__ import annotations

import warnings

import numpy as np
import pygame
from sklearn.exceptions import ConvergenceWarning

from tc2_iot.models.sgd_ocsvm import SGDOneClassSVMModel

from . import draw as d
from .data import Toy, params

N_GRID = 80
NUS = (0.01, 0.05, 0.1, 0.2)
LANDMARKS_PER_STEP = 16


class SVMScene:
    title = "SGD One-Class SVM (Nystrom RBF)"

    def __init__(self, toy: Toy):
        self.toy, self.base = toy, params("sgd_one_class_svm")
        self.nu = self.base["nu"]
        self.gamma = 1.0 / toy.train.shape[1]  # "inverse_feature_count", as in the thesis
        self.grid, self.hover = d.grid(N_GRID), None
        self._restart()
        self.phase, self.shown = "landmarks", 0

    def snapshot(self, k: int) -> SGDOneClassSVMModel:
        """The thesis model after fit() with max_iter = k (same seed, so the same trajectory)."""
        p = dict(self.base, nu=self.nu, gamma=self.gamma, max_iter=k)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)  # early snapshots stop before tol by design
            return SGDOneClassSVMModel(p, self.toy.seed, len(self.toy.train), self.toy.train.shape[1]).fit(self.toy.train)

    def _restart(self) -> None:
        full = self.snapshot(self.base["max_iter"])
        self.n_iter = full.fit_info()["n_iter_"]
        self.landmarks = full.model.steps[0][1].components_
        self.phase, self.epoch, self.cur, self.caught, self.shown = "train", 0, None, None, len(self.landmarks)

    def set_params(self, nu: float | None = None, gamma: float | None = None) -> None:
        self.nu = self.nu if nu is None else nu
        self.gamma = self.gamma if gamma is None else gamma
        self._restart()

    @property
    def interval(self) -> float:
        return 0.08 if self.phase == "landmarks" else 0.6

    def done(self) -> bool:
        return self.phase == "final"

    def step(self) -> None:
        if self.phase == "landmarks":
            self.shown = min(self.shown + LANDMARKS_PER_STEP, len(self.landmarks))
            if self.shown == len(self.landmarks):
                self.phase = "train"
        elif self.phase == "train":
            self.epoch += 1
            self.cur = self.snapshot(self.epoch)
            self.scores = self.cur.anomaly_score(self.grid)
            if self.epoch == self.n_iter:
                self.phase = "final"
                self.thr = d.threshold(self.cur.anomaly_score(self.toy.calib))
                self.caught = self.cur.anomaly_score(self.toy.attacks) > self.thr

    def handle(self, event, view: d.View) -> None:
        if event.type == pygame.MOUSEMOTION:
            self.hover = view.to_world(*event.pos) if view.contains(*event.pos) else None
        elif event.type == pygame.KEYDOWN:
            i = NUS.index(self.nu) if self.nu in NUS else 0
            if event.key == pygame.K_RIGHTBRACKET:
                self.set_params(nu=NUS[min(i + 1, len(NUS) - 1)])
            elif event.key == pygame.K_LEFTBRACKET:
                self.set_params(nu=NUS[max(i - 1, 0)])
            elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                self.set_params(gamma=min(self.gamma * 2, 8.0))
            elif event.key == pygame.K_MINUS:
                self.set_params(gamma=max(self.gamma / 2, 1 / 32))

    def draw(self, surf, view: d.View) -> None:
        if self.cur is not None:
            d.heatmap(surf, view, self.scores, N_GRID)
            d.contour(surf, view, (self.scores > 0).reshape(N_GRID, N_GRID), d.NATIVE)
            if self.done():
                d.contour(surf, view, (self.scores > self.thr).reshape(N_GRID, N_GRID), d.ACCENT)
        d.points(surf, view, self.toy.train, d.BENIGN if self.cur is not None else d.DIM, 1)
        sim = None
        if self.hover is not None:
            sim = np.exp(-self.gamma * ((self.landmarks[:self.shown] - self.hover) ** 2).sum(1))
        for i, lm in enumerate(self.landmarks[:self.shown]):
            k = 0.0 if sim is None else float(sim[i])
            color = tuple(int(a + (b - a) * k) for a, b in zip((90, 90, 110), d.NATIVE))
            pygame.draw.circle(surf, color, view.to_screen(lm), 3 + round(7 * k), 0 if k > 0.5 else 1)
        d.attacks(surf, view, self.toy.attacks, self.caught)
        if self.hover is not None:
            pygame.draw.circle(surf, d.ACCENT, view.to_screen(self.hover), 6, 2)

    def caption(self) -> list[str]:
        g = f"nu = {self.nu}   gamma = {self.gamma:.3g}"
        if self.phase == "landmarks":
            return [f"Step 1, Nystrom: {self.shown}/{len(self.landmarks)} landmarks picked from the benign data", g,
                    "An RBF kernel measures similarity exp(-gamma * distance^2): 1 when two points coincide, "
                    "near 0 when far apart. Hover to see one point's similarity to every landmark "
                    "(bigger, brighter circle = more similar).",
                    "The new 256 coordinates of a point are these similarities times a fixed 256x256 matrix "
                    "(K_LL^-1/2). A straight boundary in that space is a curved one here.",
                    "", "[ ]: change nu   - =: change gamma (restarts training)"]
        lines = [f"Step 2, SGD one-class SVM: epoch {self.epoch}/{self.n_iter} "
                 f"(the thesis fit stops here via tol)", g,
                 "Each epoch is one pass of stochastic gradient descent pushing a linear boundary in the landmark "
                 "space around the benign points. Blue line: the SVM's own boundary (score = 0). "
                 "nu is the fraction of training points allowed outside; gamma sets how local the similarity is."]
        if self.cur is not None:
            outside = float((self.cur.anomaly_score(self.toy.train) > 0).mean())
            lines += [f"Training points outside the blue boundary: {outside:.1%} (nu asks for at most {self.nu:.0%})."]
        if self.done():
            lines += ["Yellow line: calibrated threshold (99th percentile of separate benign scores). "
                      "The thesis decides with this line, not the blue one.", d.caught_line(self.toy, self.caught)]
        return lines + ["", "[ ]: change nu   - =: change gamma (restarts training)"]
