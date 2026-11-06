"""Spectra from points and regions of a spectrum image, and energy-window maps.

Coordinates are in navigation pixels with x to the right and y down; pixel
(i, j) spans [i, i+1) × [j, j+1), and a pixel belongs to a region when its
centre lies inside it.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel


class Region(BaseModel):
    kind: Literal["rect", "ellipse", "polygon"]
    # rect/ellipse: bounding box corners (any order)
    x0: float = 0
    y0: float = 0
    x1: float = 0
    y1: float = 0
    # polygon: vertices
    points: list[tuple[float, float]] = []


def _bbox(region: Region) -> tuple[float, float, float, float]:
    if region.kind == "polygon":
        xs, ys = zip(*region.points) if region.points else ((0,), (0,))
        return min(xs), min(ys), max(xs), max(ys)
    return (
        min(region.x0, region.x1), min(region.y0, region.y1),
        max(region.x0, region.x1), max(region.y0, region.y1),
    )


def region_mask(shape: tuple[int, int], region: Region) -> tuple[slice, slice, np.ndarray]:
    """(row slice, column slice, boolean mask within that crop)."""
    h, w = shape
    x0, y0, x1, y1 = _bbox(region)
    # Pixels whose centres (i + 0.5) fall within the bounding box.
    i0 = max(0, int(np.ceil(x0 - 0.5)))
    i1 = min(w - 1, int(np.floor(x1 - 0.5)))
    j0 = max(0, int(np.ceil(y0 - 0.5)))
    j1 = min(h - 1, int(np.floor(y1 - 0.5)))
    if i1 < i0 or j1 < j0:
        return slice(0, 0), slice(0, 0), np.zeros((0, 0), bool)
    ys, xs = slice(j0, j1 + 1), slice(i0, i1 + 1)
    cx = np.arange(i0, i1 + 1) + 0.5
    cy = np.arange(j0, j1 + 1) + 0.5

    if region.kind == "rect":
        mask = np.ones((len(cy), len(cx)), bool)
    elif region.kind == "ellipse":
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        rx, ry = max((x1 - x0) / 2, 1e-9), max((y1 - y0) / 2, 1e-9)
        mask = ((cx[None, :] - mx) / rx) ** 2 + ((cy[:, None] - my) / ry) ** 2 <= 1
    else:
        from matplotlib.path import Path

        if len(region.points) < 3:
            return slice(0, 0), slice(0, 0), np.zeros((0, 0), bool)
        gx, gy = np.meshgrid(cx, cy)
        inside = Path(region.points).contains_points(np.column_stack([gx.ravel(), gy.ravel()]))
        mask = inside.reshape(gx.shape)
    return ys, xs, mask


def _accumulator(dtype: np.dtype, n: int) -> type:
    """Exact and fast: uint32 when n integer counts can't overflow it."""
    if np.issubdtype(dtype, np.unsignedinteger) and n * np.iinfo(dtype).max < 2**32:
        return np.uint32
    return np.float64


def region_spectrum(data: np.ndarray, region: Region) -> tuple[np.ndarray, int]:
    """Summed spectrum over the region's pixels, and the number of pixels.

    ``data`` has shape (height, width, channels).
    """
    ys, xs, mask = region_mask(data.shape[:2], region)
    n = int(mask.sum())
    if n == 0:
        return np.zeros(data.shape[-1]), 0
    sub = data[ys, xs]
    acc_type = _accumulator(data.dtype, n)
    if mask.all():
        return sub.sum(axis=(0, 1), dtype=acc_type).astype(np.float64), n
    # Sum each row's contiguous runs of pixels: no copies and about twice as
    # fast as a masked (where=) reduction.
    acc = np.zeros(data.shape[-1], acc_type)
    padded = np.zeros(mask.shape[1] + 2, np.int8)
    for j, row in enumerate(mask):
        if not row.any():
            continue
        padded[1:-1] = row
        edges = np.flatnonzero(np.diff(padded))
        for a, b in zip(edges[::2], edges[1::2]):
            acc += sub[j, a:b].sum(axis=0, dtype=acc_type)
    return acc.astype(np.float64), n


def pixel_spectrum(data: np.ndarray, x: int, y: int, size: int = 1) -> tuple[np.ndarray, int]:
    """Summed spectrum of the size × size block centred on pixel (x, y)."""
    h, w = data.shape[:2]
    half = (size - 1) // 2
    x0, y0 = max(0, x - half), max(0, y - half)
    x1, y1 = min(w, x0 + size), min(h, y0 + size)
    block = data[y0:y1, x0:x1]
    n = block.shape[0] * block.shape[1]
    if n == 0:
        raise IndexError(f"Pixel ({x}, {y}) is outside the {w}×{h} scan")
    return block.sum(axis=(0, 1), dtype=_accumulator(data.dtype, n)).astype(np.float64), n


def energy_map(data: np.ndarray, offset: float, scale: float, lo: float, hi: float) -> np.ndarray:
    """Counts summed over channels from energy ``lo`` to ``hi`` (axis units)."""
    lo, hi = sorted((lo, hi))
    n = data.shape[-1]
    i0 = int(np.clip(np.floor((lo - offset) / scale), 0, n - 1))
    i1 = int(np.clip(np.ceil((hi - offset) / scale), i0 + 1, n))
    return data[..., i0:i1].sum(axis=-1, dtype=np.float32)
