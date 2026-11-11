"""A line scan cut from a spectrum image along a line (a line profile), at
any angle: positions about one pixel apart along the line, each the mean of
the spectra across the line's width, sampled as the viewer samples a
profile (frontend ``viewer/profile.ts``)."""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, Field


class LineSpec(BaseModel):
    """A line in the scan's pixels (pixel i spans i to i + 1; its centre is
    i + 0.5) and the width averaged across it, in pixels."""

    x0: float
    y0: float
    x1: float
    y1: float
    width: float = Field(1.0, ge=1, le=200)


def sample_points(line: LineSpec) -> tuple[np.ndarray, np.ndarray]:
    """The points sampled: x and y of shape (positions, across). Positions
    are ``floor(length) + 1`` (at least 2) from one end to the other; across
    the line, ``round(width)`` points 1 pixel apart centred on it."""
    dx, dy = line.x1 - line.x0, line.y1 - line.y0
    length = math.hypot(dx, dy)
    if length < 1:
        raise ValueError("The line is shorter than a pixel")
    n = max(2, math.floor(length) + 1)
    m = max(1, round(line.width))
    t = np.linspace(0, 1, n)[:, None]
    offset = (np.arange(m) - (m - 1) / 2)[None, :]
    nx, ny = -dy / length, dx / length
    return line.x0 + dx * t + nx * offset, line.y0 + dy * t + ny * offset


def extract(data: np.ndarray, line: LineSpec) -> np.ndarray:
    """The spectra along ``line`` of ``data`` (height, width, channels):
    (positions, channels) float32, each the mean of the spectra at the points
    across the line, bilinear between pixel centres. Points outside the scan
    are left out; ValueError if a whole position is outside."""
    h, w, c = data.shape
    xs, ys = sample_points(line)
    n, m = xs.shape
    total = np.zeros((n, c), np.float64)
    count = np.zeros(n, np.int64)
    for j in range(m):
        fx, fy = xs[:, j] - 0.5, ys[:, j] - 0.5
        inside = (fx >= -0.5) & (fy >= -0.5) & (fx <= w - 0.5) & (fy <= h - 0.5)
        ix = np.clip(np.floor(fx), 0, w - 1).astype(int)
        iy = np.clip(np.floor(fy), 0, h - 1).astype(int)
        jx, jy = np.minimum(ix + 1, w - 1), np.minimum(iy + 1, h - 1)
        ax = np.clip(fx - ix, 0, 1)[:, None]
        ay = np.clip(fy - iy, 0, 1)[:, None]
        top = data[iy, ix] * (1 - ax) + data[iy, jx] * ax
        bottom = data[jy, ix] * (1 - ax) + data[jy, jx] * ax
        total[inside] += (top * (1 - ay) + bottom * ay)[inside]
        count += inside
    if not count.all():
        raise ValueError("The line leaves the spectrum image")
    return (total / count[:, None]).astype(np.float32)


def line_scan_signal(signal, line: LineSpec, title: str):
    """A line scan of the spectrum image ``signal`` (in memory, two
    navigation axes) along ``line``: its spectra (``extract``) on its energy
    axis, with its metadata and a "Distance" axis in its scan's units."""
    am = signal.axes_manager
    ax_x, ax_y = am.navigation_axes  # natural order: x, y
    data = extract(np.asarray(signal.data), line)
    sx, sy = float(ax_x.scale), float(ax_y.scale)
    units = str(ax_x.units) if str(ax_x.units) not in ("", "<undefined>", "None") else ""
    if not units or str(ax_y.units) != units:
        sx = sy = 1.0
        units = "px"
    length = math.hypot((line.x1 - line.x0) * sx, (line.y1 - line.y0) * sy)
    distance = {
        "name": "Distance", "units": units, "scale": length / (len(data) - 1), "offset": 0.0,
        "size": len(data), "navigate": True,
    }
    energy = am.signal_axes[0].get_axis_dictionary()
    metadata = signal.metadata.deepcopy().as_dictionary()
    metadata.setdefault("General", {})["title"] = title
    out = type(signal)(data, axes=[distance, energy], metadata=metadata)
    out.metadata.set_item(
        "Processing.si_viewer.line_scan", {**line.model_dump(), "positions": len(data)},
    )
    return out
