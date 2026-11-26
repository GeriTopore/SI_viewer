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


class BeamOptions(BaseModel):
    # "center_of_mass": the centre of mass (above ``threshold`` × the mean);
    # "blur": the maximum after a Gaussian blur of ``sigma`` (whole pixels).
    # (pyxem's "cross_correlate" isn't offered: its shifts come out on
    # another basis with a search box.)
    method: Literal["center_of_mass", "blur"] = "center_of_mass"
    threshold: float = Field(1.0, ge=0, le=1000)
    sigma: float = Field(2.0, gt=0, le=100)
    # Half the side of the square searched, around the bright-field disk of
    # the mean pattern (px); None: 3 × its radius.
    half_width: int | None = Field(None, ge=2, le=4096)


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
    place."""
    s = _diffraction(signal)
    h, w = signal.data.shape[2:]
    hw = opts.half_width or max(4, int(np.ceil(3 * bf["r"])))
    cx, cy = int(np.floor(bf["cx"])), int(np.floor(bf["cy"]))
    box = (max(0, cx - hw), min(w, cx + hw), max(0, cy - hw), min(h, cy + hw))
    kwargs: dict = {}
    if opts.method == "center_of_mass":
        kwargs["threshold"] = opts.threshold
    else:
        kwargs["sigma"] = opts.sigma
    shifts = s.get_direct_beam_position(method=opts.method, signal_slice=box, lazy_output=False, **kwargs)
    return np.asarray(shifts.data, dtype=np.float32)


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
