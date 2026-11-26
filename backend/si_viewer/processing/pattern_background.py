"""Background subtraction of diffraction patterns (diffuse scattering): pyxem's
``Diffraction2D.subtract_diffraction_background`` methods, pattern by pattern.

- "median kernel": the pattern minus its median filter (``footprint`` px).
- "difference of gaussians": the pattern minus its wide blur (``max_sigma``),
  kept where the narrow blur (``min_sigma``) is above it (≥ 0).
- "radial median": the pattern minus, at each radius from the centre, the
  median there. The centre is given per pattern (the beam's: the Centring's
  origin, else the bright-field disk of the mean pattern), not pyxem's 128, 128.
- "h-dome": local maxima higher than ``h`` (of the pattern's maximum) above
  their surroundings, normalised to 1.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Literal

import numpy as np
from pydantic import BaseModel, Field

Method = Literal["median kernel", "difference of gaussians", "radial median", "h-dome"]


class PatternBackgroundOptions(BaseModel):
    method: Method = "median kernel"
    footprint: int = Field(19, ge=3, le=255)
    min_sigma: float = Field(1.0, gt=0, le=100)
    max_sigma: float = Field(55.0, gt=0, le=500)
    h: float = Field(0.25, gt=0, le=1)


def subtract(frame: np.ndarray, opts: PatternBackgroundOptions, centre: tuple[float, float] | None = None) -> np.ndarray:
    """One pattern with its background subtracted (pyxem's per-pattern
    functions). ``centre``: (x, y) pattern px (pixel i spanning [i, i + 1)),
    for the radial median; None: the pattern's middle. float32."""
    from pyxem.utils import _background_subtraction as bg

    z = np.asarray(frame, dtype=np.float64)
    if opts.method == "median kernel":
        out = bg._subtract_median(z, footprint=opts.footprint)
    elif opts.method == "difference of gaussians":
        out = bg._subtract_dog(z, min_sigma=opts.min_sigma, max_sigma=max(opts.max_sigma, opts.min_sigma))
    elif opts.method == "radial median":
        h, w = z.shape
        cx, cy = centre if centre is not None else (w / 2, h / 2)
        # pyxem measures radii from pixel indices (pixel i's centre at i).
        out = bg._subtract_radial_median(z, center_x=cx - 0.5, center_y=cy - 0.5)
    else:
        if not np.max(z) > 0:
            return np.zeros(z.shape, np.float32)
        out = bg._subtract_hdome(z, h=opts.h)
    return np.asarray(out, dtype=np.float32)


def subtract_all(
    data, opts: PatternBackgroundOptions, centres: np.ndarray | None = None,
    report: Callable[[float, str], None] | None = None,
) -> np.ndarray:
    """Every pattern of ``data`` ([y, x, ky, kx], numpy or dask) with its
    background subtracted: float32, same shape. ``centres``: [y, x, 2] of
    (x, y) px per position, or [2] for all, or None (each pattern's middle).
    Scan rows in parallel threads; ``report`` (fraction, message) after each,
    which may raise to cancel."""
    ny, nx = data.shape[:2]
    out = np.empty(data.shape, np.float32)
    centres = None if centres is None else np.asarray(centres, dtype=float)

    def centre_at(iy: int, ix: int):
        if centres is None:
            return None
        c = centres if centres.ndim == 1 else centres[iy, ix]
        return float(c[0]), float(c[1])

    def row(iy: int) -> None:
        block = np.asarray(data[iy].compute() if hasattr(data[iy], "compute") else data[iy])
        for ix in range(nx):
            out[iy, ix] = subtract(block[ix], opts, centre_at(iy, ix))

    with ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as pool:
        futures = [pool.submit(row, iy) for iy in range(ny)]
        done = 0
        try:
            for f in futures:
                f.result()
                done += 1
                if report:
                    report(done / ny, f"Row {done} of {ny}")
        except BaseException:
            for f in futures:
                f.cancel()
            raise
    return out


def describe(opts: PatternBackgroundOptions) -> str:
    """e.g. "median kernel 19 px", for titles and the record."""
    return {
        "median kernel": f"median kernel {opts.footprint} px",
        "difference of gaussians": f"difference of Gaussians σ {opts.min_sigma:g}–{opts.max_sigma:g} px",
        "radial median": "radial median",
        "h-dome": f"h-dome h {opts.h:g}",
    }[opts.method]
