"""4D-STEM: virtual detectors, virtual images, and patterns at a scan position
or over a region.

The data is ``[y, x, ky, kx]`` (scan rows and columns, then the pattern's).
Detectors and regions are in pixel coordinates, pixel i spanning [i, i + 1):
a pixel belongs to a shape when its centre does.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

from .regions import Region, region_mask

Shape = Literal["circle", "annulus", "rectangle", "point", "apertures"]
Response = Literal["sum", "mean", "max"]
MAX_APERTURES = 2000
# Hot pixels aren't looked for within this × the bright-field disk's radius (+1 px).
BF_MARGIN = 1.3


class Point(BaseModel):
    x: float
    y: float


class Detector(BaseModel):
    """A virtual detector on the pattern (pattern pixels). Circle and annulus:
    centre ``cx``, ``cy`` and radius ``r`` (annulus: inner radius ``r_inner``);
    rectangle: corners ``x0``, ``y0``, ``x1``, ``y1``; point: ``cx``, ``cy``;
    apertures: discs of radius ``r`` at ``apertures`` (their union), with the
    g-vector array that placed some of them kept in ``lattice`` (as the view
    made it: it isn't used here)."""

    shape: Shape = "circle"
    apertures: list[Point] = Field(default_factory=list, max_length=MAX_APERTURES)
    lattice: dict | None = None
    cx: float = 0.0
    cy: float = 0.0
    r: float = Field(1.0, ge=0)
    r_inner: float = Field(0.0, ge=0)
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 1.0
    y1: float = 1.0

    @model_validator(mode="after")
    def _ordered(self):
        if self.shape == "annulus" and self.r_inner >= self.r:
            raise ValueError("The inner radius must be below the outer radius")
        return self



def detector_mask(shape: tuple[int, int], d: Detector) -> np.ndarray:
    """The pattern pixels (``shape``: rows, columns) inside ``d``."""
    h, w = shape
    if d.shape == "point":
        mask = np.zeros(shape, bool)
        x, y = int(np.floor(d.cx)), int(np.floor(d.cy))
        if 0 <= x < w and 0 <= y < h:
            mask[y, x] = True
        return mask
    yy, xx = np.mgrid[0:h, 0:w] + 0.5
    if d.shape == "apertures":
        mask = np.zeros(shape, bool)
        for a in d.apertures:
            # Only the pixels near each aperture are tested.
            x0, x1 = max(0, int(a.x - d.r) - 1), min(w, int(a.x + d.r) + 2)
            y0, y1 = max(0, int(a.y - d.r) - 1), min(h, int(a.y + d.r) + 2)
            if x0 < x1 and y0 < y1:
                mask[y0:y1, x0:x1] |= (xx[y0:y1, x0:x1] - a.x) ** 2 + (yy[y0:y1, x0:x1] - a.y) ** 2 <= d.r**2
        return mask
    if d.shape == "rectangle":
        x0, x1 = sorted((d.x0, d.x1))
        y0, y1 = sorted((d.y0, d.y1))
        return (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
    r2 = (xx - d.cx) ** 2 + (yy - d.cy) ** 2
    if d.shape == "annulus":
        return (r2 >= d.r_inner**2) & (r2 <= d.r**2)
    return r2 <= d.r**2


def _numpy(a) -> np.ndarray:
    return np.asarray(a.compute() if hasattr(a, "compute") else a)


def virtual_image(data, d: Detector, response: Response = "sum") -> np.ndarray:
    """The virtual image of ``data`` ([y, x, ky, kx], numpy or dask) through
    ``d``: per scan position, the sum, mean or maximum of the pattern pixels
    inside it (zeros if it covers none). float32, [y, x]."""
    return _numpy(_virtual(data, d, response)).reshape(data.shape[:2]).astype(np.float32)


def _virtual(data, d: Detector, response: Response):
    """``virtual_image`` flattened over the scan, not yet computed (dask)."""
    ny, nx, kh, kw = data.shape
    idx = np.flatnonzero(detector_mask((kh, kw), d))
    if idx.size == 0:
        return np.zeros(ny * nx, np.float32)
    picked = data.reshape(ny * nx, kh * kw)[:, idx]
    if response == "max":
        return picked.max(axis=1)
    out = picked.sum(axis=1, dtype=np.float64)
    return out / idx.size if response == "mean" else out


def virtual_images(data, items: list[tuple[Detector, Response]]) -> np.ndarray:
    """The virtual images through several detectors, float32 [n, y, x]. Lazy
    (dask) data is read once for all of them."""
    ny, nx = data.shape[:2]
    if not items:
        return np.zeros((0, ny, nx), np.float32)
    parts = [_virtual(data, d, r) for d, r in items]
    if any(hasattr(p, "compute") for p in parts):
        import dask

        parts = dask.compute(*parts)
    return np.stack([np.asarray(p).reshape(ny, nx) for p in parts]).astype(np.float32)


def pattern_at(data, x: float, y: float) -> np.ndarray:
    """The pattern at scan pixel (``x``, ``y``), float32 [ky, kx]. IndexError
    outside the scan."""
    ny, nx = data.shape[:2]
    ix, iy = int(np.floor(x)), int(np.floor(y))
    if not (0 <= ix < nx and 0 <= iy < ny):
        raise IndexError(f"({ix}, {iy}) is outside the {nx} × {ny} scan")
    return _numpy(data[iy, ix]).astype(np.float32)


def pattern_over(data, r: Region, reduce: Literal["mean", "max"] = "mean") -> tuple[np.ndarray, int]:
    """The mean or maximum pattern over the scan positions whose centres are
    inside ``r`` (a rectangle, ellipse or polygon of the scan), float32
    [ky, kx], and how many positions. ValueError if none."""
    ys, xs, mask = region_mask(data.shape[:2], r)
    n = int(mask.sum())
    if n == 0:
        raise ValueError("The region covers no scan position")
    block = data[ys, xs]
    # The positions inside, gathered by index (numpy and dask alike).
    picked = block.reshape((-1, *block.shape[2:]))[np.flatnonzero(mask.ravel())]
    out = picked.max(axis=0) if reduce == "max" else picked.mean(axis=0, dtype=np.float64)
    return _numpy(out).astype(np.float32), n


def describe(d: Detector, response: Response, scale: float = 1.0, units: str = "px") -> str:
    """A short title for a virtual image, e.g. "Virtual annulus 12–30 1/nm"
    (radii in the pattern's calibration; ``scale`` per pixel)."""
    n = lambda v: f"{v * scale:.3g}"  # noqa: E731
    what = {
        "circle": f"circle r {n(d.r)} {units}",
        "annulus": f"annulus {n(d.r_inner)}–{n(d.r)} {units}",
        "rectangle": f"rectangle {n(abs(d.x1 - d.x0))} × {n(abs(d.y1 - d.y0))} {units}",
        "point": f"point ({int(np.floor(d.cx))}, {int(np.floor(d.cy))})",
        "apertures": f"{len(d.apertures)} aperture{'s' if len(d.apertures) != 1 else ''} r {n(d.r)} {units}",
    }[d.shape]
    return f"Virtual {what}" + ("" if response == "sum" else f", {response}")


def bright_field_disk(pattern: np.ndarray) -> dict:
    """The bright-field disk of a pattern (the mean pattern, usually): the
    connected pixels around the brightest one that are brighter than halfway
    between the pattern's median and that maximum, after a 3 × 3 median filter
    (so a hot pixel brighter than the beam isn't taken for it). Returns its
    centre (``cx``, ``cy``: intensity-weighted, pattern pixels) and the radius
    ``r`` of a disc of the same area. ValueError if the pattern is flat."""
    from scipy import ndimage

    raw = np.asarray(pattern, dtype=np.float64)
    smooth = ndimage.median_filter(raw, size=3, mode="nearest")
    top, floor = float(smooth.max()), float(np.median(smooth))
    if not top > floor:
        raise ValueError("The pattern is flat: no bright-field disk to find")
    # The disk is measured on the pattern itself, grown from the smoothed maximum.
    labels, _ = ndimage.label(raw > (top + floor) / 2)
    seed = labels[np.unravel_index(np.argmax(smooth), smooth.shape)]
    if seed == 0:
        raise ValueError("No bright-field disk found")
    disk = labels == seed
    weights = np.where(disk, raw - floor, 0.0)
    cy, cx = ndimage.center_of_mass(weights)
    return {"cx": float(cx) + 0.5, "cy": float(cy) + 0.5, "r": float(np.sqrt(disk.sum() / np.pi)), "pixels": int(disk.sum())}


def bad_pixels(mean_pattern: np.ndarray, hot: float, dead: float) -> dict:
    """Hot and dead pixels of the detector, found in the mean pattern. Hot:
    brighter than the median of the 8 pixels around them by more than ``hot``
    times the pattern's mean intensity (pyxem's ``find_hot_pixels``, its
    threshold given relative to the pattern), outside the bright-field disk
    (its edge is steep, not hot). Dead: dimmer than ``dead`` times that median
    (and the median above 0). Returns boolean arrays ``hot`` and ``dead`` (0
    to turn either off)."""
    from pyxem.utils.diffraction import find_hot_pixels
    from scipy import ndimage

    p = np.asarray(mean_pattern, dtype=np.float64)
    if hot > 0:
        try:
            disk = bright_field_disk(p)
            yy, xx = np.mgrid[0 : p.shape[0], 0 : p.shape[1]] + 0.5
            beam = (xx - disk["cx"]) ** 2 + (yy - disk["cy"]) ** 2 <= (disk["r"] * BF_MARGIN + 1) ** 2
        except ValueError:
            beam = None
        hot_mask = find_hot_pixels(p, threshold_multiplier=hot * float(p.mean()), mask=beam)
    else:
        hot_mask = np.zeros(p.shape, bool)
    if dead > 0:
        around = ndimage.median_filter(p, footprint=[[1, 1, 1], [1, 0, 1], [1, 1, 1]], mode="nearest")
        dead_mask = (around > 0) & (p < dead * around)
    else:
        dead_mask = np.zeros(p.shape, bool)
    return {"hot": np.asarray(hot_mask, bool), "dead": dead_mask & ~hot_mask}


def interior(mask: np.ndarray) -> np.ndarray:
    """``mask`` without its border pixels (pyxem's repair averages the 3 × 3
    pixels around each, which a border pixel doesn't have)."""
    out = mask.copy()
    out[0, :] = out[-1, :] = out[:, 0] = out[:, -1] = False
    return out


def repair(signal, mask: np.ndarray):
    """``signal`` (4D-STEM, in memory) with the pixels of ``mask`` (one
    pattern's shape) replaced in every pattern by the mean of the 8 around
    them (pyxem's ``correct_bad_pixels``). Returns a new signal, float32: the
    means of sparse counts rounded back to integers would leave the repaired
    pixels low (dead-looking) in the mean pattern."""
    import pyxem

    s = pyxem.signals.Diffraction2D(np.asarray(signal.data, dtype=np.float32))
    fixed = np.asarray(s.correct_bad_pixels(np.asarray(mask, bool), inplace=False, show_progressbar=False).data)
    return signal._deepcopy_with_new_data(fixed.astype(np.float32, copy=False))
