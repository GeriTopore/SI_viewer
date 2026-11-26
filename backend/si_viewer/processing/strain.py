"""Strain from 4D-STEM Bragg vectors (py4DSTEM's workflow, pyxem's maths).

Step 1, the lattice: the Bragg vector map's peaks (on the pattern's own
pixels, as the pattern pane shows it) are found, and the origin O and the
basis vectors g1 and g2 are placed on them (proposed here, moved by the user).
Every lattice point O + h g1 + k g2 within the tolerance of a peak is
"matched"; the later steps index each pattern's disks to these points.

Positions are in pattern pixels with pixel i spanning [i, i + 1).
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field


class BvmPeakOptions(BaseModel):
    # Gaussian blur of the map before the maxima are found (pattern px; 0: none).
    sigma: float = Field(1.0, ge=0, le=50)
    # Lowest peak kept: this fraction of the brightest (after the blur).
    threshold: float = Field(0.02, ge=0, le=1)
    # Closest two peaks may be (pattern px).
    min_spacing: float = Field(3.0, ge=0.5, le=1000)
    max_peaks: int = Field(200, ge=3, le=5000)


# The map is made this many times finer than the pattern, so peaks are found
# to a fraction of a pixel.
UPSAMPLE = 2


def bvm_peaks(table: np.ndarray, shape: tuple[int, int], opts: BvmPeakOptions) -> np.ndarray:
    """The Bragg vector map's peaks: float [n, 3] of (x, y, intensity) in
    pattern px, brightest first. ``table``: the Bragg vectors (``disks``'
    rows; moved by the centring already if wanted) on patterns of ``shape``."""
    from hyperspy.utils.peakfinders2D import find_local_max
    from scipy.ndimage import gaussian_filter

    from .disks import _refine, vector_map

    up = UPSAMPLE
    m = vector_map(table, shape, upsample=up).astype(np.float64)
    if opts.sigma > 0:
        m = gaussian_filter(m, opts.sigma * up)
    top = float(m.max()) if m.size else 0.0
    if top <= 0:
        return np.zeros((0, 3))
    found = find_local_max(
        m, min_distance=max(1, int(round(opts.min_spacing * up))), threshold_abs=opts.threshold * top,
        exclude_border=1, num_peaks=opts.max_peaks,
    )
    found = np.asarray(found, dtype=float).reshape(-1, 2)
    found = found[np.isfinite(found).all(axis=1)].astype(int)
    out = np.empty((len(found), 3))
    for k, (iy, ix) in enumerate(found):
        y, x = _refine(m, iy, ix, w=up)
        out[k] = (x / up, y / up, m[iy, ix])
    return out[np.argsort(-out[:, 2])] if len(out) else out


def guess_lattice(peaks: np.ndarray, centre: tuple[float, float], min_angle: float = 30.0) -> dict | None:
    """O, g1 and g2 proposed from the peaks (pattern px, g1 and g2 as vectors
    from O): O the peak nearest ``centre`` (the direct beam: the map counts
    disks, so it needn't be the brightest), g1 the nearest peak to it, g2 the
    nearest at least ``min_angle``° off g1's line. None with fewer than three
    peaks or all on one line."""
    if len(peaks) < 3:
        return None
    at = int(np.argmin(np.hypot(peaks[:, 0] - centre[0], peaks[:, 1] - centre[1])))
    o = peaks[at, :2]
    rest = np.delete(peaks[:, :2], at, axis=0) - o
    d = np.hypot(rest[:, 0], rest[:, 1])
    order = np.argsort(d)
    g1 = rest[order[0]]
    g2 = None
    for i in order[1:]:
        v = rest[i]
        cos = abs(float(v @ g1)) / (np.hypot(*v) * np.hypot(*g1))
        if math.degrees(math.acos(min(1.0, cos))) >= min_angle:
            g2 = v
            break
    if g2 is None:
        return None
    # Right-handed in the view (y down): g2 counter-clockwise from g1 on screen.
    if g1[0] * g2[1] - g1[1] * g2[0] > 0:
        g1, g2 = g2, g1
    return {"ox": float(o[0]), "oy": float(o[1]), "g1x": float(g1[0]), "g1y": float(g1[1]),
            "g2x": float(g2[0]), "g2y": float(g2[1])}


# ---------- step 2: index every pattern's disks and fit the lattice ----------


class StrainLattice(BaseModel):
    """O, g1 and g2 on the Bragg vector map (pattern px, as the pattern pane
    shows it; g1 and g2 from O)."""

    ox: float
    oy: float
    g1x: float
    g1y: float
    g2x: float
    g2y: float


class FitOptions(BaseModel):
    # Largest distance (px) from a lattice point at which a disk is indexed to it.
    tolerance: float = Field(2.0, gt=0, le=1000)
    # Fewest indexed disks a position needs to be fitted (and not all on one line).
    min_disks: int = Field(4, ge=3, le=10000)
    # Disks weighted by their intensity (the correlation), as py4DSTEM does.
    weighted: bool = True


# The lattice fit's components, in order (calibrated frame, pattern px).
FIT_COMPONENTS = ("g1 x", "g1 y", "g2 x", "g2 y", "O x", "O y")


def frame_matrix(ring: dict | None = None, rotation: dict | None = None) -> np.ndarray:
    """The 2 × 2 matrix taking offsets on the pattern (px) into the calibrated
    frame: the ring's ellipse made circular (``disks.ellipse_matrix``), then
    the QR rotation (``processing.dpc.rotate``: y flipped if mirrored, then
    turned by θ). Lengths stay in pattern px."""
    from .disks import ellipse_matrix

    m = np.eye(2)
    if ring is not None:
        m = ellipse_matrix(ring) @ m
    if rotation is not None:
        t = math.radians(rotation["theta"])
        r = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
        flip = np.diag([1.0, -1.0 if rotation.get("mirror") else 1.0])
        m = r @ flip @ m
    return m


def index_disks(table: np.ndarray, lattice: StrainLattice, tolerance: float, frame: np.ndarray):
    """Each disk (``table``'s rows, on the Bragg vector map's pixels) indexed
    to the nearest lattice point: (q [n, 2] its offset from O in the
    calibrated frame, hk [n, 2] int, ok [n] within ``tolerance`` px)."""
    xy = np.column_stack([table[:, 3], table[:, 2]]).astype(np.float64)
    q = (xy - (lattice.ox, lattice.oy)) @ frame.T
    g = frame @ np.array([[lattice.g1x, lattice.g2x], [lattice.g1y, lattice.g2y]])
    if abs(np.linalg.det(g)) < 1e-9:
        raise ValueError("g1 and g2 are on one line")
    hk = np.rint(np.linalg.solve(g, q.T).T).astype(int)
    d = np.hypot(*(q - hk @ g.T).T)
    return q, hk, d <= tolerance


def fit_lattice(table: np.ndarray, scan_shape: tuple[int, int], lattice: StrainLattice, opts: FitOptions,
                frame: np.ndarray) -> dict:
    """py4DSTEM's lattice fit at every scan position: its indexed disks
    (``index_disks``) fitted by least squares (weighted by intensity if
    asked) as q = O' + h g1 + k g2, in the calibrated frame (pattern px).
    Returns {"fit": [6, y, x] (``FIT_COMPONENTS``; NaN where not fitted),
    "count": [y, x] indexed disks, "residual": [y, x] RMS distance (px) of
    the indexed disks from the fitted lattice}."""
    ny, nx = scan_shape
    n = ny * nx
    q, hk, ok = index_disks(table, lattice, opts.tolerance, frame)
    pos = (np.floor(table[:, 0]).astype(int) * nx + np.floor(table[:, 1]).astype(int))
    keep = ok & (pos >= 0) & (pos < n)
    q, hk, pos = q[keep], hk[keep], pos[keep]
    w = np.clip(table[keep, 4].astype(np.float64), 1e-6, None) if opts.weighted else np.ones(len(pos))
    a = np.column_stack([np.ones(len(pos)), hk[:, 0], hk[:, 1]])
    # Normal equations per position: Σ w a aᵀ, Σ w a x, Σ w a y.
    ata = np.zeros((n, 3, 3))
    np.add.at(ata, pos, w[:, None, None] * a[:, :, None] * a[:, None, :])
    atx = np.zeros((n, 3))
    aty = np.zeros((n, 3))
    np.add.at(atx, pos, w[:, None] * a * q[:, :1])
    np.add.at(aty, pos, w[:, None] * a * q[:, 1:])
    count = np.bincount(pos, minlength=n)
    # Enough disks, and not all on one lattice line (the matrix well conditioned).
    det = np.linalg.det(ata)
    wsum = ata[:, 0, 0]
    good = (count >= opts.min_disks) & (det > 1e-9 * np.maximum(wsum, 1e-12) ** 3)
    cx = np.full((n, 3), np.nan)
    cy = np.full((n, 3), np.nan)
    if good.any():
        cx[good] = np.linalg.solve(ata[good], atx[good][..., None])[..., 0]
        cy[good] = np.linalg.solve(ata[good], aty[good][..., None])[..., 0]
    # The indexed disks' distances from the fitted lattice.
    ex = q[:, 0] - (a * cx[pos]).sum(axis=1)
    ey = q[:, 1] - (a * cy[pos]).sum(axis=1)
    r2 = np.zeros(n)
    np.add.at(r2, pos, w * np.nan_to_num(ex * ex + ey * ey))
    with np.errstate(invalid="ignore", divide="ignore"):
        residual = np.where(good, np.sqrt(r2 / np.where(wsum > 0, wsum, 1)), np.nan)
    # (O', g1, g2) → components: g1 x, g1 y, g2 x, g2 y, O x, O y.
    fit = np.stack([cx[:, 1], cy[:, 1], cx[:, 2], cy[:, 2], cx[:, 0], cy[:, 0]]).reshape(6, ny, nx)
    return {"fit": fit.astype(np.float32), "count": count.reshape(ny, nx).astype(np.float32),
            "residual": residual.reshape(ny, nx).astype(np.float32)}


# ---------- steps 3 and 4: the reference, the strain axes and the maps ----------


class StrainAxes(BaseModel):
    """The strain maps' x axis: the scan's x (the calibrated frame's: the
    scan's when the QR rotation is applied), along the reference g1, or at
    ``angle``° from the scan's x, counter-clockwise as seen on screen."""

    along: Literal["scan", "g1", "angle"] = "scan"
    angle: float = Field(0.0, ge=-360, le=360)


# The maps made, in order: (output, title), as kemstem's strain tensor names them.
STRAIN_OUTPUTS = (("exx", "εxx"), ("eyy", "εyy"), ("exy", "εxy"), ("oxy", "ωxy"))


def fitted_basis(fit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The fitted g1 and g2 at every position (``fit_lattice``'s ``fit``,
    [6, y, x]) as rows: ([y, x, 2, 2], [y, x] where fitted)."""
    vs = np.moveaxis(np.asarray(fit[:4], dtype=np.float64), 0, -1).reshape(*fit.shape[1:], 2, 2)
    return vs, np.isfinite(vs).all(axis=(-2, -1))


def reference_basis(fit: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """The unstrained g1 and g2 (rows, [2, 2]): the median of the fitted
    ones over the scan, or over ``mask`` ([y, x] bool, a region of it);
    with how many positions it is from."""
    vs, ok = fitted_basis(fit)
    if mask is not None:
        ok = ok & mask
    if not ok.any():
        raise ValueError("No position in the reference was fitted")
    ref = np.median(vs[ok], axis=0)
    if abs(np.linalg.det(ref)) < 1e-12:
        raise ValueError("The reference g1 and g2 are on one line")
    return ref, int(ok.sum())


def axis_direction(axes: StrainAxes, ref: np.ndarray) -> tuple[float, float]:
    """The maps' x axis as a unit vector in the calibrated frame (pattern px,
    y down)."""
    if axes.along == "g1":
        g = ref[0] / np.hypot(*ref[0])
        return float(g[0]), float(g[1])
    if axes.along == "angle":
        a = math.radians(axes.angle)
        return math.cos(a), -math.sin(a)
    return 1.0, 0.0


def strain_maps(fit: np.ndarray, ref: np.ndarray, x_axis: tuple[float, float]) -> dict[str, np.ndarray]:
    """εxx, εyy, εxy and ωxy at every position (``STRAIN_OUTPUTS``; NaN where
    not fitted) from the fitted g1 and g2 against ``ref``, by pyxem
    (``get_DisplacementGradientMap`` → ``get_strain_maps`` →
    ``StrainMap.rotate_strain_basis`` onto ``x_axis``). Signs as kemstem's
    strain tensor (image axes, y down): pyxem's e12 is its reciprocal-space
    stretch's, the opposite of the real-space shear, and its θ turns x
    toward y (clockwise on screen), so εxy = −e12 and ωxy = −θ (positive
    counter-clockwise on screen, radians)."""
    import hyperspy.api as hs
    from pyxem.generators.displacement_gradient_tensor_generator import get_DisplacementGradientMap

    vs, ok = fitted_basis(fit)
    # Positions not fitted take the reference (pyxem's maths needs numbers), then NaN.
    vs[~ok] = ref
    gradient = get_DisplacementGradientMap(hs.signals.Signal2D(vs), ref, show_progressbar=False)
    sm = gradient.get_strain_maps()
    sm.data[2] *= -1
    if abs(x_axis[1]) > 1e-12 or x_axis[0] < 0:
        sm = sm.rotate_strain_basis([x_axis[0], x_axis[1]])
    data = np.asarray(sm.data, dtype=np.float64)
    out = {"exx": data[0], "eyy": data[1], "exy": data[2], "oxy": -data[3]}
    return {k: np.where(ok, v, np.nan).astype(np.float32) for k, v in out.items()}
