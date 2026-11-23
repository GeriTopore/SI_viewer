"""Masks made from images: the pixels of an image (smoothed first, if asked)
between thresholds, cleaned up with scikit-image's morphology. A mask is an
image of 0 and 1 (origin "mask") under the image it was made from, and can be
used by any process that takes one (so far: sublattices of atomic columns).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel


class MaskOptions(BaseModel):
    # Gaussian blur (px) before thresholding; 0: none.
    smooth: float = 0.0
    # The pixels kept: above ``lo``, below ``hi``, or between them.
    keep: Literal["above", "below", "between"] = "above"
    lo: float | None = None
    hi: float | None = None
    invert: bool = False
    # Opening then closing with a disk of this radius (px); 0: none.
    clean: int = 0
    # Regions and holes smaller than this (px) are removed / filled; 0: none.
    min_area: int = 0

    def check(self) -> "MaskOptions":
        if self.smooth < 0 or self.clean < 0 or self.min_area < 0:
            raise ValueError("Smoothing, cleaning and the smallest area can't be negative")
        if self.keep in ("above", "between") and self.lo is None:
            raise ValueError("A lower threshold is needed")
        if self.keep in ("below", "between") and self.hi is None:
            raise ValueError("An upper threshold is needed")
        if self.keep == "between" and self.lo > self.hi:
            raise ValueError("The lower threshold is above the upper one")
        return self


def smoothed(image: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    a = np.asarray(image, dtype=np.float64)
    if sigma > 0:
        a = gaussian_filter(np.nan_to_num(a, nan=float(np.nanmean(a))), sigma)
    return a


def make_mask(image: np.ndarray, opts: MaskOptions) -> np.ndarray:
    """The mask (bool) of ``image`` with ``opts``."""
    from skimage import morphology

    opts.check()
    a = smoothed(image, opts.smooth)
    with np.errstate(invalid="ignore"):
        if opts.keep == "above":
            m = a >= opts.lo
        elif opts.keep == "below":
            m = a <= opts.hi
        else:
            m = (a >= opts.lo) & (a <= opts.hi)
    if opts.invert:
        m = ~m
    if opts.clean > 0:
        disk = morphology.disk(opts.clean)
        m = morphology.closing(morphology.opening(m, disk), disk)
    if opts.min_area > 0:
        m = morphology.remove_small_objects(m, max_size=opts.min_area - 1)
        m = morphology.remove_small_holes(m, max_size=opts.min_area - 1)
    return m


def threshold_values(image: np.ndarray, smooth: float, sample: int = 20000, seed: int = 0) -> dict:
    """A sample of the (smoothed) image's values, for a histogram, and its
    Otsu threshold (scikit-image)."""
    from skimage.filters import threshold_otsu

    a = smoothed(image, smooth).ravel()
    a = a[np.isfinite(a)]
    if not a.size:
        raise ValueError("The image has no values")
    rng = np.random.default_rng(seed)
    pick = a if a.size <= sample else rng.choice(a, sample, replace=False)
    return {"values": np.round(pick, 6).tolist(), "otsu": float(threshold_otsu(a)),
            "min": float(a.min()), "max": float(a.max())}


def inside(mask: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Whether each (y, x) position (px, centres at integers) falls in the
    mask; NaN positions or those off it: False."""
    h, w = mask.shape
    out = np.zeros(len(positions), bool)
    ok = np.all(np.isfinite(positions), axis=1)
    iy = np.rint(positions[ok, 0]).astype(np.int64)
    ix = np.rint(positions[ok, 1]).astype(np.int64)
    on = (iy >= 0) & (iy < h) & (ix >= 0) & (ix < w)
    hit = np.zeros(ok.sum(), bool)
    hit[on] = mask[iy[on], ix[on]] > 0.5
    out[ok] = hit
    return out
