"""Cross-sections of a stack through depth, along a line drawn on its image."""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, Field

MAX_WIDTH = 200


class SectionLine(BaseModel):
    """A line in the image's pixel coordinates (pixel i spans [i, i + 1); y
    counts from the top), and the width (pixels) averaged across it."""

    x0: float
    y0: float
    x1: float
    y1: float
    width: float = Field(1.0, ge=1, le=MAX_WIDTH)


def section(stack: np.ndarray, line: SectionLine) -> tuple[np.ndarray, int]:
    """Every plane of ``stack`` ([plane, row, column]) sampled along ``line``:
    [plane, position], positions at the centres of equal steps along it. The
    image repeats beyond its edges, as a simulation's cell does. Also returns
    the number of positions."""
    from scipy.ndimage import map_coordinates

    dx, dy = line.x1 - line.x0, line.y1 - line.y0
    length = math.hypot(dx, dy)
    if length < 1:
        raise ValueError("The line is shorter than a pixel")
    n = max(2, round(length))
    t = (np.arange(n) + 0.5) / n
    # Pixel centres are at i + 0.5.
    xs = line.x0 + dx * t - 0.5
    ys = line.y0 + dy * t - 0.5
    across = max(1, round(line.width))
    offsets = np.arange(across) - (across - 1) / 2
    nx, ny = -dy / length, dx / length
    planes = stack.shape[0]
    cols = xs[None, :] + offsets[:, None] * nx
    rows = ys[None, :] + offsets[:, None] * ny
    coords = np.stack([
        np.broadcast_to(np.arange(planes, dtype=float)[:, None, None], (planes, across, n)),
        np.broadcast_to(rows, (planes, across, n)),
        np.broadcast_to(cols, (planes, across, n)),
    ])
    values = map_coordinates(np.asarray(stack, np.float32), coords, order=1, mode="grid-wrap")
    return values.mean(axis=1).astype(np.float32), n
