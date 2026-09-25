"""Autoencoder: reconstrução dos pontos, curva aprendida no gargalo e erro de reconstrução como escore."""

from __future__ import annotations

import numpy as np
import pygame
import torch
from torch import nn

from tc2_iot.models.autoencoder import Autoencoder

from . import draw as d
from .data import Toy, params

N_GRID = 80
EPOCHS = 300  # fixed so the animation has a known length; the thesis uses early stopping instead
PER_STEP = 5
HIDDEN = [32, 1, 32]  # thesis: [32, 8, 32] on 36 features; in 2D the bottleneck must be 1


class AEScene:
    title = "Autoencoder"
    interval = 0.03

    def __init__(self, toy: Toy):
        self.toy, self.p = toy, dict(params("autoencoder"), hidden_dimensions=HIDDEN)
        self.X = torch.as_tensor(toy.train)
        # a 1-unit relu bottleneck can start dead (every code = 0) and then never learns: skip those inits
        torch_seed, self.skipped_inits = toy.seed, 0
        while True:
            self.model = Autoencoder(self.p, torch_seed, 2)
            if self._codes().gt(0).float().mean() >= 0.5:
                break
            torch_seed, self.skipped_inits = torch_seed + 1000, self.skipped_inits + 1
        self.net = self.model.net
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.p["learning_rate"],
                                    weight_decay=self.p["weight_decay"])
        self.gen = torch.Generator().manual_seed(torch_seed)
        self.epoch, self.losses, self.caught = 0, [self._loss()], None
        self.grid = d.grid(N_GRID)

    @torch.no_grad()
    def _codes(self, X: torch.Tensor | None = None) -> torch.Tensor:
        return self.model.net[:4](self.X if X is None else X)

    def _loss(self) -> float:
        return float(self.model.anomaly_score(self.toy.train).mean())

    def done(self) -> bool:
        return self.epoch >= EPOCHS

    def step(self) -> None:
        if self.done():
            return
        # same loop as Autoencoder.fit: shuffled mini-batches, mse, adam
        for _ in range(PER_STEP):
            self.net.train()
            perm = torch.randperm(len(self.X), generator=self.gen)
            for i in range(0, len(self.X), self.p["batch_size"]):
                xb = self.X[perm[i:i + self.p["batch_size"]]]
                self.opt.zero_grad()
                nn.functional.mse_loss(self.net(xb), xb).backward()
                self.opt.step()
            self.epoch += 1
            self.losses.append(self._loss())
        if self.done():
            self.scores = self.model.anomaly_score(self.grid)
            self.thr = d.threshold(self.model.anomaly_score(self.toy.calib))
            self.caught = self.model.anomaly_score(self.toy.attacks) > self.thr

    def handle(self, event, view: d.View) -> None:
        pass

    @torch.no_grad()
    def _recon(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        return self.net(torch.as_tensor(X)).numpy()

    @torch.no_grad()
    def _curve(self) -> np.ndarray:
        """Decoder output over the range of codes the benign data uses: the learned 1D manifold."""
        z = self._codes()
        sweep = torch.linspace(float(z.min()), float(z.max()), 200).unsqueeze(1)
        return self.net[4:](sweep).numpy()

    def draw(self, surf, view: d.View) -> None:
        if self.done():
            d.heatmap(surf, view, self.scores, N_GRID)
            d.contour(surf, view, (self.scores > self.thr).reshape(N_GRID, N_GRID), d.ACCENT)
        d.points(surf, view, self.toy.train, d.DIM if not self.done() else d.BENIGN, 1)
        sub = self.toy.train[::3]
        for x, r in zip(sub, self._recon(sub)):
            pygame.draw.line(surf, (110, 110, 140), view.to_screen(x), view.to_screen(r), 1)
        for x, r in zip(self.toy.attacks, self._recon(self.toy.attacks)):
            pygame.draw.line(surf, d.ATTACK, view.to_screen(x), view.to_screen(r), 2)
        pygame.draw.lines(surf, d.NATIVE, False, [view.to_screen(p) for p in self._curve()], 3)
        d.attacks(surf, view, self.toy.attacks, self.caught)
        self._loss_chart(surf, pygame.Rect(view.rect.right - 230, view.rect.y + 10, 220, 110))

    def _loss_chart(self, surf, r: pygame.Rect) -> None:
        box = pygame.Surface(r.size, pygame.SRCALPHA)
        box.fill((0, 0, 0, 170))
        surf.blit(box, r)
        y = np.log10(np.array(self.losses))
        lo, hi = y.min(), max(y.max(), y.min() + 1e-6)
        pts = [(r.x + 10 + (r.width - 20) * i / EPOCHS, r.bottom - 10 - (r.height - 35) * (v - lo) / (hi - lo))
               for i, v in enumerate(y)]
        if len(pts) > 1:
            pygame.draw.lines(surf, d.ACCENT, False, pts, 2)
        d.text(surf, [f"train MSE (log): {self.losses[-1]:.4f}"], r.x + 8, r.y + 5, r.width, 18)

    def caption(self) -> list[str]:
        lines = [f"Epoch {self.epoch}/{EPOCHS}",
                 "Network 2 -> 32 -> 1 -> 32 -> 2 (thesis: 36 -> 32 -> 8 -> 32 -> 36). It must squeeze each "
                 "point through a single number and rebuild it, so it can only keep what benign data has in common.",
                 "Grey/red lines join each point to its reconstruction. Blue curve: everything the decoder can "
                 "output. It bends to follow the benign data as training goes.",
                 "Score = reconstruction error (squared length of the line). Red lines = attacks."]
        if self.skipped_inits:
            lines += [f"Skipped {self.skipped_inits} initialization(s) where the 1-unit ReLU bottleneck was dead "
                      "(all codes 0, so it can never learn)."]
        lines += [f"Fixed {EPOCHS} epochs here; the thesis stops early when the validation MSE stops improving."]
        if self.done():
            lines += ["Yellow line: calibrated threshold (99th percentile of separate benign errors).",
                      "Gap attacks sit where the curve passes between the two groups, so they are rebuilt almost "
                      "perfectly: the autoencoder only knows the curve, not where the data actually is.",
                      d.caught_line(self.toy, self.caught)]
        return lines
