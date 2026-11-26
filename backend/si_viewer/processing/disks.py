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


def to_vectors(table: np.ndarray, cube):
    """A Bragg vectors table as pyxem ``DiffractionVectors``: over ``cube``'s
    scan (its navigation axes), at each position its disks as (ky, kx,
    intensity) in the pattern's calibration (value = offset + scale × pixel
    index, as HyperSpy's axes), the intensity being the correlation."""
    import hyperspy.api as hs
    from pyxem.signals import DiffractionVectors

    ny, nx = cube.data.shape[:2]
    iy = np.floor(table[:, 0]).astype(int)
    ix = np.floor(table[:, 1]).astype(int)
    order = np.lexsort((ix, iy))
    flat = iy[order] * nx + ix[order]
    rows = table[order]
    bounds = np.searchsorted(flat, np.arange(ny * nx + 1))
    peaks = np.empty((ny, nx), dtype=object)
    for k in range(ny * nx):
        r = rows[bounds[k] : bounds[k + 1]]
        # Pixel indices (pixel i's centre at index i), then the score.
        peaks[k // nx, k % nx] = np.column_stack([r[:, 2] - 0.5, r[:, 3] - 0.5, r[:, 4]]).astype(np.float64)
    signal = hs.signals.BaseSignal(peaks, ragged=True)
    for axis, src in zip(signal.axes_manager.navigation_axes, cube.axes_manager.navigation_axes):
        axis.name, axis.scale, axis.offset, axis.units = src.name, src.scale, src.offset, src.units
    ky_ax, kx_ax = cube.axes_manager.signal_axes[::-1]
    with _quiet():
        vectors = DiffractionVectors.from_peaks(
            signal,
            center=[-ky_ax.offset / ky_ax.scale, -kx_ax.offset / kx_ax.scale],
            calibration=[ky_ax.scale, kx_ax.scale],
            column_names=[str(ky_ax.name), str(kx_ax.name)],
            units=[str(ky_ax.units), str(kx_ax.units)],
        )
    vectors.metadata.General.title = "Bragg vectors"
    return vectors


class _quiet:
    """HyperSpy's progress bar and its warning that the function mapped
    ignores the axes' scales off for a while (pyxem's ``from_peaks`` maps a
    per-vector calibration, which doesn't use them)."""

    def __enter__(self):
        import logging

        import hyperspy.api as hs

        self.was = hs.preferences.General.show_progressbar
        hs.preferences.General.show_progressbar = False
        self.logger = logging.getLogger("hyperspy.signal")
        self.level = self.logger.level
        self.logger.setLevel(logging.ERROR)

    def __exit__(self, *exc):
        import hyperspy.api as hs

        hs.preferences.General.show_progressbar = self.was
        self.logger.setLevel(self.level)


def vectors_at(vectors, x: float, y: float) -> np.ndarray:
    """The disks of ``DiffractionVectors`` (``to_vectors``) at scan pixel
    (x, y) as pattern pixel positions (pixel i spanning [i, i + 1)) and
    intensities: float32 [n, 3] of (ky, kx, intensity)."""
    ny, nx = vectors.data.shape[:2]
    ix, iy = int(np.floor(x)), int(np.floor(y))
    if not (0 <= ix < nx and 0 <= iy < ny):
        raise IndexError(f"({ix}, {iy}) is outside the {nx} × {ny} scan")
    v = np.asarray(vectors.data[iy, ix], dtype=np.float64).reshape(-1, 3)
    off, sc = np.asarray(vectors.offsets, dtype=float), np.asarray(vectors.scales, dtype=float)
    out = v.copy()
    out[:, :2] = (v[:, :2] - off[:2]) / sc[:2] + 0.5
    return out.astype(np.float32)


def table_of(vectors) -> np.ndarray:
    """``DiffractionVectors`` (``to_vectors``) back as a Bragg vectors table
    (``VECTOR_FIELDS``, pattern pixels)."""
    ny, nx = vectors.data.shape[:2]
    parts = []
    for iy in range(ny):
        for ix in range(nx):
            d = vectors_at(vectors, ix, iy)
            parts.append(np.column_stack([np.full(len(d), iy + 0.5), np.full(len(d), ix + 0.5), d]))
    return np.concatenate(parts).astype(np.float32) if parts else np.empty((0, 5), np.float32)


# ---------- calibrated vectors (py4DSTEM-style: applied when read) ----------


def direct_beam_origins(table: np.ndarray, scan_shape: tuple[int, int], guess: dict) -> np.ndarray:
    """Each scan position's direct beam from its own disks: the disk nearest
    ``guess`` (the bright-field disk of the mean pattern) within 2 × its
    radius. [y, x, 2] of (x, y) pattern px; positions without one: NaN."""
    ny, nx = scan_shape
    out = np.full((ny, nx, 2), np.nan)
    if not len(table):
        return out
    d = np.hypot(table[:, 3] - guess["cx"], table[:, 2] - guess["cy"])
    near = d <= 2 * guess["r"]
    rows, dist = table[near], d[near]
    iy, ix = np.floor(rows[:, 0]).astype(int), np.floor(rows[:, 1]).astype(int)
    order = np.argsort(-dist)  # nearest written last: it stays
    out[iy[order], ix[order], 0] = rows[order, 3]
    out[iy[order], ix[order], 1] = rows[order, 2]
    return out


def fill_origins(origins: np.ndarray) -> np.ndarray:
    """Origins with the missing (NaN) ones set to the median of the others."""
    out = origins.copy()
    for k in range(2):
        ch = out[..., k]
        if np.isnan(ch).all():
            raise ValueError("No direct beam was found among the disks")
        ch[np.isnan(ch)] = np.nanmedian(ch)
    return out


def ellipse_matrix(ring: dict) -> np.ndarray:
    """The 2 × 2 matrix (on (x, y) offsets) that makes ``ring``'s ellipse a
    circle of radius √(a b): its a axis scaled by √(b / a), b by √(a / b)."""
    c, s = np.cos(ring["angle"]), np.sin(ring["angle"])
    r = np.array([[c, -s], [s, c]])
    k = np.diag([np.sqrt(ring["b"] / ring["a"]), np.sqrt(ring["a"] / ring["b"])])
    return r @ k @ r.T


def calibrated(table: np.ndarray, origins: np.ndarray | None, centre: tuple[float, float],
               ellipse: dict | None = None, scale: float = 1.0) -> np.ndarray:
    """The disks' positions calibrated: relative to their scan position's
    origin (``origins`` [y, x, 2] of (x, y) px; None: ``centre`` for all),
    the ellipse made circular, times ``scale``. [n, 2] of (kx, ky)."""
    x, y = table[:, 3].astype(float), table[:, 2].astype(float)
    if origins is not None:
        iy, ix = np.floor(table[:, 0]).astype(int), np.floor(table[:, 1]).astype(int)
        ox, oy = origins[iy, ix, 0], origins[iy, ix, 1]
    else:
        ox, oy = centre
    q = np.column_stack([x - ox, y - oy])
    if ellipse is not None:
        q = q @ ellipse_matrix(ellipse).T
    return q * scale


def ddf_image(table: np.ndarray, scan_shape: tuple[int, int], detector, response: str = "sum",
              weight: str = "intensity", origins: np.ndarray | None = None) -> np.ndarray:
    """A digital dark-field image (py4DSTEM's ``DDFimage``): made from the
    disks found, not the patterns. A disk counts where its position lies
    inside ``detector`` (``fourd.Detector``, pattern px; apertures: within
    an aperture's radius, the match tolerance, of its centre; once, however
    many apertures it is in). With ``origins`` ([y, x, 2] of x, y px, the
    centring), each position's disks are moved by its origin's offset from
    the mean origin first (descan), so apertures drawn on any pattern match.
    Per scan position: the sum, mean or maximum of the matched disks'
    intensity (their correlation score), or with ``weight`` "count" their
    number. float32 [y, x]."""
    ny, nx = scan_shape
    out = np.zeros(ny * nx, np.float64)
    if not len(table):
        return out.reshape(ny, nx).astype(np.float32)
    x, y = table[:, 3].astype(float), table[:, 2].astype(float)
    iy, ix = np.floor(table[:, 0]).astype(int), np.floor(table[:, 1]).astype(int)
    if origins is not None:
        ref = np.nanmean(origins.reshape(-1, 2), axis=0)
        x = x - origins[iy, ix, 0] + ref[0]
        y = y - origins[iy, ix, 1] + ref[1]
    d = detector
    if d.shape == "apertures":
        inside = np.zeros(len(x), bool)
        for a in d.apertures:
            r = d.r if a.r is None else a.r
            inside |= (x - a.x) ** 2 + (y - a.y) ** 2 <= r**2
    elif d.shape == "point":
        inside = (np.floor(x) == np.floor(d.cx)) & (np.floor(y) == np.floor(d.cy))
    elif d.shape == "rectangle":
        x0, x1 = sorted((d.x0, d.x1))
        y0, y1 = sorted((d.y0, d.y1))
        inside = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
    else:
        r2 = (x - d.cx) ** 2 + (y - d.cy) ** 2
        inside = (r2 <= d.r**2) & ((r2 >= d.r_inner**2) if d.shape == "annulus" else True)
    pos = (iy * nx + ix)[inside]
    ok = (pos >= 0) & (pos < ny * nx)
    pos = pos[ok]
    value = np.ones(len(pos)) if weight == "count" else table[inside, 4].astype(float)[ok]
    if response == "max":
        np.maximum.at(out, pos, value)
    else:
        np.add.at(out, pos, value)
        if response == "mean":
            n = np.bincount(pos, minlength=ny * nx)
            out = np.divide(out, n, out=np.zeros_like(out), where=n > 0)
    return out.reshape(ny, nx).astype(np.float32)


def vector_map_of(q: np.ndarray, weights: np.ndarray, half: float, bins: int) -> np.ndarray:
    """A Bragg vector map of calibrated positions ``q`` ([n, 2] of kx, ky):
    ``bins`` × ``bins`` over −``half`` … ``half`` on both axes, each disk
    split between the 4 bins around it. float32 [ky, kx]."""
    out = np.zeros((bins, bins), np.float64)
    if len(q):
        step = 2 * half / bins
        x = (q[:, 0] + half) / step - 0.5
        y = (q[:, 1] + half) / step - 0.5
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        fx, fy = x - x0, y - y0
        for dy, dx, f in ((0, 0, (1 - fy) * (1 - fx)), (0, 1, (1 - fy) * fx), (1, 0, fy * (1 - fx)), (1, 1, fy * fx)):
            yy, xx = y0 + dy, x0 + dx
            ok = (yy >= 0) & (yy < bins) & (xx >= 0) & (xx < bins)
            np.add.at(out, (yy[ok], xx[ok]), (weights * f)[ok])
    return out.astype(np.float32)


def map_of(table: np.ndarray, shape: tuple[int, int], origins: np.ndarray | None = None,
           centre: tuple[float, float] | None = None, ring: dict | None = None, scale: float | None = None,
           upsample: int = 2) -> tuple[np.ndarray, float, float]:
    """The Bragg vector map of a table on patterns of ``shape`` (h, w),
    ``upsample`` × finer: raw (no calibration given: pattern px, from the
    pattern's corner), or calibrated (relative to ``origins`` per position
    or ``centre``, the ellipse made circular, × ``scale``; 0 at the centre).
    Returns (map, step, offset) of both axes (the first bin's centre)."""
    h, w = shape
    if origins is None and ring is None and scale is None:
        step = 1 / upsample
        return vector_map(table, shape, upsample=upsample), step, step / 2
    s = scale or 1.0
    q = calibrated(table, origins, centre if centre is not None else (w / 2, h / 2), ring, s)
    half = max(w, h) / 2 * s
    bins = upsample * max(w, h)
    step = 2 * half / bins
    return vector_map_of(q, np.ones(len(q)), half, bins), step, -half + step / 2


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
