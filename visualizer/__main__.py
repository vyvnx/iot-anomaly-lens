"""Visualizador didático dos três modelos do TC. Uso: uv run --group viz python -m visualizer"""

from __future__ import annotations

import pygame

from . import draw as d
from .data import make_toy
from .scene_ae import AEScene
from .scene_if import IFScene
from .scene_svm import SVMScene

SIZE = (1280, 800)
PLOT = pygame.Rect(20, 20, 760, 760)
PANEL_X, PANEL_W = 810, 440
SCENES = [
    (IFScene, "Isolation Forest", "Random cuts isolate lonely points quickly."),
    (SVMScene, "SGD One-Class SVM (Nystrom RBF)", "A boundary drawn around the benign data."),
    (AEScene, "Autoencoder", "Compress, rebuild, and measure what got lost."),
]
LEGEND = ["Dots: benign traffic   Red X: attack   Green ring: attack caught",
          "Yellow line: calibrated threshold (FPR about 1%)"]


class App:
    def __init__(self, seed: int = 0):
        pygame.init()
        self.seed, self.menu_index, self.scene, self.running = seed, 0, None, True
        self.view, self.playing, self.acc = d.View(PLOT), True, 0.0

    def _open(self) -> None:
        self.scene, self.playing, self.acc = SCENES[self.menu_index][0](make_toy(self.seed)), True, 0.0

    def _menu_item_at(self, pos) -> int | None:
        for i in range(len(SCENES)):
            if pygame.Rect(340, 250 + i * 110, 600, 90).collidepoint(pos):
                return i
        return None

    def handle(self, event) -> None:
        if event.type == pygame.QUIT:
            self.running = False
        elif self.scene is None:
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_UP, pygame.K_DOWN):
                    self.menu_index = (self.menu_index + (1 if event.key == pygame.K_DOWN else -1)) % len(SCENES)
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    self._open()
                elif event.key == pygame.K_ESCAPE:
                    self.running = False
            elif event.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN) and self._menu_item_at(event.pos) is not None:
                self.menu_index = self._menu_item_at(event.pos)
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._open()
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.scene = None
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            self.playing = not self.playing
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
            self.playing = False
            self.scene.step()
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            self.seed += 1
            self._open()
        else:
            self.scene.handle(event, self.view)

    def update(self, dt: float) -> None:
        if self.scene is None or not self.playing or self.scene.done():
            return
        self.acc += dt
        if self.acc >= self.scene.interval:  # at most one step per frame, so slow steps never pile up
            self.acc = 0.0
            self.scene.step()

    def draw(self, surf) -> None:
        surf.fill(d.BG)
        if self.scene is None:
            d.text(surf, ["How the anomaly detectors work"], 340, 120, 700, 48)
            d.text(surf, ["2D toy data, real thesis models and hyperparameters. Up/Down + Enter, or click. Esc quits."],
                   340, 180, 700, 22, d.DIM)
            for i, (_, name, desc) in enumerate(SCENES):
                r = pygame.Rect(340, 250 + i * 110, 600, 90)
                pygame.draw.rect(surf, d.PANEL, r, border_radius=10)
                if i == self.menu_index:
                    pygame.draw.rect(surf, d.ACCENT, r, 3, border_radius=10)
                d.text(surf, [name], r.x + 20, r.y + 18, r.width - 40, 32)
                d.text(surf, [desc], r.x + 20, r.y + 54, r.width - 40, 22, d.DIM)
            return
        pygame.draw.rect(surf, (10, 10, 14), PLOT)
        surf.set_clip(PLOT)
        self.scene.draw(surf, self.view)
        surf.set_clip(None)
        pygame.draw.rect(surf, d.PANEL, (PANEL_X - 10, 20, PANEL_W + 20, 760), border_radius=10)
        y = d.text(surf, [self.scene.title], PANEL_X, 32, PANEL_W, 34)
        y = d.text(surf, self.scene.caption(), PANEL_X, y + 8, PANEL_W)
        d.text(surf, LEGEND, PANEL_X, 680, PANEL_W, 20, d.DIM)
        state = "finished" if self.scene.done() else ("playing" if self.playing else "paused")
        d.text(surf, [f"seed {self.seed} | {state} | Space play/pause  Right step  R new data  Esc menu"],
               PANEL_X, 745, PANEL_W, 20, d.ACCENT)


def main() -> None:
    app = App()
    screen = pygame.display.set_mode(SIZE)
    pygame.display.set_caption("tc2-iot model visualizer")
    clock = pygame.time.Clock()
    while app.running:
        for event in pygame.event.get():
            app.handle(event)
        app.update(clock.tick(60) / 1000)
        app.draw(screen)
        pygame.display.flip()
    pygame.quit()


if __name__ == "__main__":
    main()
