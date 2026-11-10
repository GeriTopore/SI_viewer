"""Sub-scan images: an image acquired together with a spectrum image on a
finer grid of the same field of view (Gatan DM's ADF "sub-scan": the image is
k × k times finer than the spectrum image, each spectrum pixel summing a k × k
block of image pixels).

Only the arithmetic lives here, on plain numbers (:class:`Plane`), so it can be
tested without any file. :func:`plane_of` reads a dataset's plane.
"""

from __future__ import annotations

from dataclasses import dataclass

# Fields of view that differ by more than this (relative) are not the same.
FOV_TOLERANCE = 0.01

# Length units in metres. Anything else (empty, "px", "keV", ...) cannot be
# compared, and then the field of view is not checked.
_METRES = {
    "m": 1.0,
    "mm": 1e-3,
    "um": 1e-6,
    "µm": 1e-6,
    "μm": 1e-6,
    "nm": 1e-9,
    "pm": 1e-12,
    "å": 1e-10,
    "angstrom": 1e-10,
}


@dataclass(frozen=True)
class Plane:
    """A 2D grid: size in pixels and calibration of its x and y axes."""

    width: int
    height: int
    scale_x: float = 1.0
    scale_y: float = 1.0
    units_x: str = ""
    units_y: str = ""


def _metres(units: str | None) -> float | None:
    return _METRES.get((units or "").strip().lower())


def _same_extent(n_a: int, scale_a: float, units_a: str, n_b: int, scale_b: float, units_b: str) -> bool:
    """The extents ``n × scale`` agree within the tolerance; True if the units
    cannot be compared."""
    ma, mb = _metres(units_a), _metres(units_b)
    if ma is None or mb is None or not scale_a > 0 or not scale_b > 0:
        return True
    a = n_a * scale_a * ma
    b = n_b * scale_b * mb
    return abs(a - b) <= FOV_TOLERANCE * max(a, b)


def scan_factor(scan: Plane, image: Plane) -> int | None:
    """k if ``image`` is the spectrum image's scan sampled k times more finely
    in both directions (1: the same grid), else None.

    The size must be a whole multiple of the scan, the same on both axes. For
    k > 1 the calibrated field of view must also agree to within 1 % wherever
    both axes have comparable length units; a different field of view is a
    different image.
    """
    if scan.width < 1 or scan.height < 1 or image.width % scan.width or image.height % scan.height:
        return None
    k = image.width // scan.width
    if k < 1 or image.height // scan.height != k:
        return None
    if k > 1:
        if not _same_extent(scan.width, scan.scale_x, scan.units_x, image.width, image.scale_x, image.units_x):
            return None
        if not _same_extent(scan.height, scan.scale_y, scan.units_y, image.height, image.scale_y, image.units_y):
            return None
    return k


def plane_of(ds) -> Plane:
    """The plane a dataset is shown on: a spectrum image's scan, an image's or
    a stack's frame (a stack as loaded, before any alignment)."""
    from .session import DatasetKind

    if ds.kind is DatasetKind.SPECTRUM_IMAGE:
        axes = ds.signal.axes_manager.navigation_axes
    else:
        axes = ds.loaded_signal.axes_manager.signal_axes
    x, y = axes[-2:]  # natural order: x, y
    return Plane(
        int(x.size), int(y.size), float(x.scale), float(y.scale), str(x.units or ""), str(y.units or "")
    )


def scan_factor_of(si, image) -> int | None:
    """:func:`scan_factor` for datasets: the image ``image`` against the scan of ``si``."""
    try:
        return scan_factor(plane_of(si), plane_of(image))
    except (ValueError, IndexError, AttributeError):  # not a 2D plane
        return None
