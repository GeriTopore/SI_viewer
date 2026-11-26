"""Bragg disk detection in 4D-STEM data: every pattern is matched with a
template of the probe (pyxem's normalised cross-correlation), the maxima of
the correlation are the disks (HyperSpy's local-maximum peak finder), each
refined to sub-pixel precision by the centre of mass of the correlation
around it.

Positions are in pattern pixels with pixel i spanning [i, i + 1) (the
viewer's convention): a disk centred on pixel (i, j) is at (i + 0.5, j + 0.5).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Literal

import numpy as np
from pydantic import BaseModel, Field

# The columns of a Bragg vectors table: one row per disk.
VECTOR_FIELDS = ("scan_y", "scan_x", "ky", "kx", "score")


class DiskOptions(BaseModel):
    # "probe": the bright-field disk of the mean pattern (the measured probe);
    # "disc": a flat disc of ``radius``.
    template: Literal["probe", "disc"] = "probe"
    # Disc radius (pattern px); None: the bright-field disk's.
    radius: float | None = Field(None, gt=0, le=1000)
    # Lowest correlation a disk may have (normalised cross-correlation, −1…1).
    threshold: float = Field(0.4, ge=-1, le=1)
    # Closest two disks may be (pattern px).
    min_spacing: int = Field(5, ge=1, le=1000)
    # Disks this close to the pattern's edge (px) are left out.
    edge: int = Field(2, ge=0, le=1000)
    max_disks: int = Field(100, ge=1, le=10000)
    subpixel: bool = True


def probe_template(mean: np.ndarray, disk: dict, margin: float = 1.4) -> np.ndarray:
    """The probe as a template: the mean pattern around its bright-field disk
    (``fourd.bright_field_disk``), zero beyond ``margin`` × its radius."""
    cy, cx, r = disk["cy"], disk["cx"], disk["r"]
    half = int(np.ceil(margin * r)) + 1
    iy, ix = int(np.floor(cy)), int(np.floor(cx))
    h, w = mean.shape
    if iy - half < 0 or ix - half < 0 or iy + half >= h or ix + half >= w:
        raise ValueError("The bright-field disk is too close to the pattern's edge for a template")
    from scipy.ndimage import shift

    crop = np.asarray(mean[iy - half : iy + half + 1, ix - half : ix + half + 1], dtype=np.float32)
    # The disk moved to the template's centre (its middle pixel's centre),
    # so the matches aren't off by the fraction of a pixel it was off.
    crop = shift(crop, (iy + 0.5 - cy, ix + 0.5 - cx), order=1, mode="nearest")
    yy, xx = np.mgrid[-half : half + 1, -half : half + 1]
    crop = crop - float(np.median(mean))
    crop[np.hypot(yy, xx) > margin * r] = 0
    return crop


def disc_template(r: float) -> np.ndarray:
    """A flat disc of radius ``r`` (px) on a slightly larger square."""
    half = int(np.ceil(r)) + 2
    yy, xx = np.mgrid[-half : half + 1, -half : half + 1]
    return (np.hypot(yy, xx) <= r).astype(np.float32)


def correlation(pattern: np.ndarray, template: np.ndarray) -> np.ndarray:
    """The normalised cross-correlation of ``pattern`` with ``template``
    (−1…1), the same size as the pattern (pyxem)."""
    from pyxem.utils.diffraction import normalize_template_match

    # Padded with the edge's values: zeros around the pattern correlate with
    # the template's edge and give false disks along the border.
    return normalize_template_match(np.asarray(pattern, dtype=np.float32), template, subtract_min=False, mode="edge")


def _refine(cc: np.ndarray, iy: int, ix: int, w: int = 1) -> tuple[float, float]:
    """The centre of mass of the correlation (above its local minimum) in the
    (2w + 1)² window around pixel (iy, ix), as a pixel-centre position."""
    h, wd = cc.shape
    y0, y1, x0, x1 = max(0, iy - w), min(h, iy + w + 1), max(0, ix - w), min(wd, ix + w + 1)
    win = cc[y0:y1, x0:x1]
    win = win - win.min()
    s = win.sum()
    if s <= 0:
        return iy + 0.5, ix + 0.5
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return float((win * yy).sum() / s) + 0.5, float((win * xx).sum() / s) + 0.5


def find_disks(pattern: np.ndarray, template: np.ndarray, opts: DiskOptions) -> np.ndarray:
    """The disks in one pattern: float32 [n, 3] of (ky, kx, score), the
    score being the correlation there; strongest first."""
    from hyperspy.utils.peakfinders2D import find_local_max

    cc = correlation(pattern, template)
    peaks = find_local_max(
        cc, min_distance=opts.min_spacing, threshold_abs=opts.threshold,
        exclude_border=max(opts.edge, 1), num_peaks=opts.max_disks,
    )
    peaks = np.asarray(peaks, dtype=float).reshape(-1, 2)
    peaks = peaks[np.isfinite(peaks).all(axis=1)].astype(int)
    out = np.empty((len(peaks), 3), np.float32)
    for k, (iy, ix) in enumerate(peaks):
        y, x = _refine(cc, iy, ix) if opts.subpixel else (iy + 0.5, ix + 0.5)
        out[k] = (y, x, cc[iy, ix])
    return out[np.argsort(-out[:, 2])] if len(out) else out


def find_all(
    data, template: np.ndarray, opts: DiskOptions, report: Callable[[float, str], None] | None = None,
) -> np.ndarray:
    """The disks of every pattern of ``data`` ([y, x, ky, kx], numpy or
    dask): a Bragg vectors table, float32 [n, 5] (``VECTOR_FIELDS``), rows
    by scan position. Scan rows are done in parallel threads; ``report``
    (fraction, message) after each, which may raise to cancel."""
    ny, nx = data.shape[:2]

    def row(iy: int) -> np.ndarray:
        block = np.asarray(data[iy].compute() if hasattr(data[iy], "compute") else data[iy])
        parts = []
        for ix in range(nx):
            d = find_disks(block[ix], template, opts)
            parts.append(np.column_stack([np.full(len(d), iy + 0.5), np.full(len(d), ix + 0.5), d]))
        return np.concatenate(parts) if parts else np.empty((0, 5))

    rows: list[np.ndarray | None] = [None] * ny
    with ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as pool:
        futures = {pool.submit(row, iy): iy for iy in range(ny)}
        done = 0
        try:
            for f in futures:
                rows[futures[f]] = f.result()
                done += 1
                if report:
                    report(done / ny, f"Row {done} of {ny}")
        except BaseException:
            for f in futures:
                f.cancel()
            raise
    table = np.concatenate([r for r in rows if r is not None]) if ny else np.empty((0, 5))
    return np.ascontiguousarray(table, dtype=np.float32).reshape(-1, 5)


def disks_at(table: np.ndarray, x: float, y: float) -> np.ndarray:
    """The rows of a Bragg vectors table at scan pixel (x, y)."""
    sel = (np.floor(table[:, 0]) == np.floor(y)) & (np.floor(table[:, 1]) == np.floor(x))
    return table[sel]


def vector_map(table: np.ndarray, shape: tuple[int, int], upsample: int = 2, weighted: bool = False) -> np.ndarray:
    """The Bragg vector map: a histogram of every disk's position on the
    pattern, ``upsample`` × finer than its pixels (each disk split between the
    4 bins around it); counts, or the disks' scores if ``weighted``. float32
    [ky × upsample, kx × upsample]."""
    h, w = shape[0] * upsample, shape[1] * upsample
    out = np.zeros((h, w), np.float64)
    if len(table):
        y = table[:, 2] * upsample - 0.5
        x = table[:, 3] * upsample - 0.5
        wt = table[:, 4] if weighted else np.ones(len(table))
        y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
        fy, fx = y - y0, x - x0
        for dy, dx, f in ((0, 0, (1 - fy) * (1 - fx)), (0, 1, (1 - fy) * fx), (1, 0, fy * (1 - fx)), (1, 1, fy * fx)):
            yy, xx = y0 + dy, x0 + dx
            ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
            np.add.at(out, (yy[ok], xx[ok]), (wt * f)[ok])
    return out.astype(np.float32)
