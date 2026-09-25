"""Utilidades de desenho compartilhadas pelas cenas: transformação de coordenadas, mapa de calor e texto."""

from __future__ import annotations

from functools import lru_cache

import matplotlib
import numpy as np
import pygame

from tc2_iot.calibration import calibrate

LIM = 4.0  # world is [-LIM, LIM]^2
BG, PANEL, FG, DIM = (18, 18, 24), (30, 30, 40), (230, 230, 235), (120, 120, 135)
BENIGN, ATTACK, CAUGHT, ACCENT, NATIVE = (200, 220, 255), (255, 80, 80), (80, 255, 140), (255, 210, 60), (90, 200, 255)
_CMAP = matplotlib.colormaps["magma"]


class View:
    """Maps world coordinates to a square screen rect (y grows upwards in the world)."""

    def __init__(self, rect: pygame.Rect):
        self.rect = rect
        self.scale = rect.width / (2 * LIM)

    def to_screen(self, xy) -> tuple[int, int]:
        return (round(self.rect.x + (xy[0] + LIM) * self.scale), round(self.rect.y + (LIM - xy[1]) * self.scale))

    def to_world(self, px, py) -> tuple[float, float]:
        return ((px - self.rect.x) / self.scale - LIM, LIM - (py - self.rect.y) / self.scale)

    def contains(self, px, py) -> bool:
        return self.rect.collidepoint(px, py)


def grid(n: int) -> np.ndarray:
    """Cell centres, row-major, first row at the top of the plot."""
    c = (np.arange(n) + 0.5) / n * 2 * LIM - LIM
    xx, yy = np.meshgrid(c, c[::-1])
    return np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)


def threshold(calib_scores: np.ndarray) -> float:
    return calibrate(calib_scores, 0.01, min_warning=0)["threshold"]


def heatmap(surf, view: View, values: np.ndarray, n: int) -> None:
    v = values.reshape(n, n)
    lo, hi = np.percentile(v, [1, 99])
    rgb = (_CMAP(np.clip((v - lo) / (hi - lo + 1e-12), 0, 1))[..., :3] * 200).astype(np.uint8)
    img = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))  # surfarray is (x, y)
    surf.blit(pygame.transform.scale(img, view.rect.size), view.rect)


def contour(surf, view: View, mask: np.ndarray, color, width: int = 2) -> None:
    """Draws the edges between cells where the boolean mask changes."""
    n, cell = mask.shape[0], view.rect.width / mask.shape[0]
    x0, y0 = view.rect.topleft
    for i, j in zip(*np.nonzero(mask[:, 1:] != mask[:, :-1])):
        x = x0 + (j + 1) * cell
        pygame.draw.line(surf, color, (x, y0 + i * cell), (x, y0 + (i + 1) * cell), width)
    for i, j in zip(*np.nonzero(mask[1:, :] != mask[:-1, :])):
        y = y0 + (i + 1) * cell
        pygame.draw.line(surf, color, (x0 + j * cell, y), (x0 + (j + 1) * cell, y), width)


def points(surf, view: View, X: np.ndarray, color, r: int = 2) -> None:
    for p in X:
        pygame.draw.circle(surf, color, view.to_screen(p), r)


def attacks(surf, view: View, X: np.ndarray, flagged: np.ndarray | None = None) -> None:
    """Attacks as red crosses; once a threshold exists, green ring = caught, none = missed."""
    for i, p in enumerate(X):
        x, y = view.to_screen(p)
        pygame.draw.line(surf, ATTACK, (x - 5, y - 5), (x + 5, y + 5), 3)
        pygame.draw.line(surf, ATTACK, (x - 5, y + 5), (x + 5, y - 5), 3)
        if flagged is not None and flagged[i]:
            pygame.draw.circle(surf, CAUGHT, (x, y), 10, 2)


def caught_line(toy, caught: np.ndarray) -> str:
    kinds = np.array(toy.kinds)
    per = ", ".join(f"{k} {int(caught[kinds == k].sum())}/{int((kinds == k).sum())}" for k in ("far", "gap", "edge"))
    return f"Attacks caught (green ring): {per}."


@lru_cache(maxsize=4)
def font(size: int = 22) -> pygame.font.Font:
    return pygame.font.Font(None, size)


def text(surf, lines: list[str], x: int, y: int, width: int, size: int = 22, color=FG) -> int:
    """Word-wrapped lines; returns the y after the last line."""
    f = font(size)
    for line in lines:
        words, cur = line.split(" "), ""
        for w in words:
            if cur and f.size(cur + " " + w)[0] > width:
                surf.blit(f.render(cur, True, color), (x, y))
                y, cur = y + f.get_linesize(), w
            else:
                cur = f"{cur} {w}" if cur else w
        surf.blit(f.render(cur, True, color), (x, y))
        y += f.get_linesize() + (6 if line else 0)
    return y
