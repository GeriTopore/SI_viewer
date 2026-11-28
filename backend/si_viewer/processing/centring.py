"""Centring 4D-STEM data (pyxem): the direct beam found in every pattern,
the scan's descan as a linear plane fitted to those positions, and the
patterns shifted so the beam is at their centre.

Shifts are pyxem's: per scan position (x-shift, y-shift), the pattern pixels
that move the beam to the pattern's centre (index w/2, h/2). The beam itself
is then at index (w/2 − x-shift, h/2 − y-shift), drawn at + 0.5 in the
viewer's convention (pixel i spanning [i, i + 1)).
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np
from pydantic import BaseModel, Field


class BeamSeed(BaseModel):
    """Where the direct beam is, given by hand (an aperture placed on it):
    centre and radius in pattern px (pixel i spanning [i, i + 1))."""

    cx: float
    cy: float
    r: float = Field(gt=0)


class BeamOptions(BaseModel):
    # "center_of_mass": the centre of mass (above ``threshold`` × the mean);
    # "blur": the maximum after a Gaussian blur of ``sigma`` (whole pixels).
    # (pyxem's "cross_correlate" isn't offered: its shifts come out on
    # another basis with a search box.)
    method: Literal["center_of_mass", "blur"] = "center_of_mass"
    threshold: float = Field(1.0, ge=0, le=1000)
    sigma: float = Field(2.0, gt=0, le=100)
    # Half the side of the square searched, around the bright-field disk of
    # the mean pattern (px); None: 3 × its radius (1 × an aperture's).
    half_width: int | None = Field(None, ge=2, le=4096)
    # The beam given by an aperture instead of the mean pattern's brightest
    # disk (patterns with no disk that stands out, e.g. overlapping CBED disks).
    seed: BeamSeed | None = None


class PlaneOptions(BaseModel):
    # The descan: none (each pattern's own beam), a linear plane fitted to
    # every position's, or to the scan's corners only (a fraction of each side).
    mode: Literal["none", "plane", "corners"] = "plane"
    corners: float = Field(0.1, gt=0, le=0.5)


def _diffraction(signal):
    """``signal`` as a pyxem Diffraction2D (the same data, not copied)."""
    s = signal._deepcopy_with_new_data(signal.data)
    s.set_signal_type("electron_diffraction")
    return s


def find_beam(signal, opts: BeamOptions, bf: dict) -> np.ndarray:
    """The shifts that centre the beam in every pattern of ``signal`` (4D-STEM,
    numpy or dask): float32 [y, x, 2] (x-shift, y-shift; pattern px). The
    search is limited to a square around ``bf`` (``fourd.bright_field_disk``
    of the mean pattern), so brighter spots elsewhere don't take the beam's
    place. ``bf`` is the aperture when ``opts.seed`` is given."""
    if opts.seed is not None and opts.method == "center_of_mass":
        return _follow_aperture(signal, opts.seed, opts.threshold)
    s = _diffraction(signal)
    h, w = signal.data.shape[2:]
    # An aperture marks the beam itself: search only it (not its neighbours).
    hw = opts.half_width or max(2 if opts.seed else 4, int(np.ceil((1 if opts.seed else 3) * bf["r"])))
    cx, cy = int(np.floor(bf["cx"])), int(np.floor(bf["cy"]))
    box = (max(0, cx - hw), min(w, cx + hw), max(0, cy - hw), min(h, cy + hw))
    kwargs: dict = {}
    if opts.method == "center_of_mass":
        kwargs["threshold"] = opts.threshold
    else:
        kwargs["sigma"] = opts.sigma
    shifts = s.get_direct_beam_position(method=opts.method, signal_slice=box, lazy_output=False, **kwargs)
    return np.asarray(shifts.data, dtype=np.float32)


def _follow_aperture(signal, seed: BeamSeed, threshold: float, passes: int = 4) -> np.ndarray:
    """``find_beam`` from an aperture on the beam: in every pattern, the centre
    of mass inside a circle of the aperture's radius, the circle moved onto
    it and the centre of mass taken again (``passes`` times), so a beam that
    moves over the scan stays inside and its neighbours outside. Pixels below
    ``threshold`` × the pattern's mean are left out (pyxem's convention)."""
    data = signal.data
    ny, nx, h, w = data.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    out = np.empty((ny, nx, 2), np.float32)
    # Index coordinates: pixel i is centred on i (the seed's are i + 0.5).
    x0, y0 = seed.cx - 0.5, seed.cy - 0.5
    for y in range(ny):
        rows = np.asarray(data[y], np.float32)  # nx × h × w
        floor = threshold * rows.mean(axis=(1, 2), keepdims=True)
        weights = np.where(rows >= floor, rows, 0)
        cx = np.full(nx, x0, np.float32)
        cy = np.full(nx, y0, np.float32)
        for _ in range(passes):
            inside = (xx[None] - cx[:, None, None]) ** 2 + (yy[None] - cy[:, None, None]) ** 2 <= seed.r**2
            wsum = (weights * inside).sum(axis=(1, 2))
            ok = wsum > 0
            cx = np.where(ok, (weights * inside * xx[None]).sum(axis=(1, 2)) / np.where(ok, wsum, 1), cx)
            cy = np.where(ok, (weights * inside * yy[None]).sum(axis=(1, 2)) / np.where(ok, wsum, 1), cy)
        # pyxem's shifts: what moves the beam to index (w/2, h/2).
        out[y, :, 0] = w / 2 - cx
        out[y, :, 1] = h / 2 - cy
    return out


def plane(shifts: np.ndarray, opts: PlaneOptions) -> np.ndarray:
    """The shifts used for centring: as found ("none"), or the linear planes
    fitted to them by pyxem (the whole scan, or its corners)."""
    if opts.mode == "none":
        return shifts
    from pyxem.signals import BeamShift

    bs = BeamShift(np.asarray(shifts, dtype=np.float64))
    fitted = bs.get_linear_plane(fit_corners=opts.corners if opts.mode == "corners" else None)
    return np.asarray(fitted.data, dtype=np.float32)


def beam_positions(shifts: np.ndarray, pattern_shape: tuple[int, int]) -> np.ndarray:
    """Where the beam is ([y, x, 2]: x, y in pattern px, pixel i spanning
    [i, i + 1)) for ``shifts`` on patterns of ``pattern_shape`` (h, w)."""
    h, w = pattern_shape
    out = np.empty_like(shifts, dtype=np.float32)
    out[..., 0] = w / 2 - shifts[..., 0] + 0.5
    out[..., 1] = h / 2 - shifts[..., 1] + 0.5
    return out


def centre(signal, shifts: np.ndarray, report: Callable[[float, str], None] | None = None):
    """``signal`` with every pattern shifted by ``shifts`` (sub-pixel, pyxem's
    ``center_direct_beam``): the beam at the pattern's centre. float32 (so
    the shifts' interpolation keeps the intensity); the pattern axes'
    offsets set so that the centre is at 0."""
    from pyxem.signals import BeamShift

    s = _diffraction(signal)
    if s.data.dtype != np.float32:
        s.change_dtype("float32", rechunk=False)
    if report:
        report(0.1, "Shifting the patterns")
    bs = BeamShift(np.asarray(shifts, dtype=np.float64))
    out = s.center_direct_beam(shifts=bs, subpixel=True, inplace=False)
    if out._lazy:
        out.compute()
    out.set_signal_type("")
    for axis in out.axes_manager.signal_axes:
        axis.offset = -axis.scale * axis.size / 2
    return out


def shift_stats(shifts: np.ndarray) -> dict:
    """How far the beam moves over the scan: each shift's range (px)."""
    if not shifts.size:
        return {}
    lo, hi = shifts.reshape(-1, 2).min(axis=0), shifts.reshape(-1, 2).max(axis=0)
    return {"x": [float(lo[0]), float(hi[0])], "y": [float(lo[1]), float(hi[1])]}
