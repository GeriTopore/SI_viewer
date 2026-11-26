"""Calibrating 4D-STEM patterns: a ring fitted as an ellipse (its centre,
semi-axes and angle: the ellipticity), the reciprocal pixel size from that
ring and a known reflection (or entered), and both kept with the data.

The ellipse is fitted directly (Halir & Flusser's numerically stable
least-squares conic fit) to points on the ring: at each angle, the
intensity-weighted radius within the annulus searched. (pyxem's
``determine_ellipse`` uses scikit-image's ``EllipseModel``, which fails with
scikit-image 0.26 here.) Positions are pattern pixels, pixel i spanning
[i, i + 1).
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

# d-spacings (Å) of common calibration standards' strongest rings (fcc/diamond, room temperature).
STANDARDS: dict[str, dict[str, float]] = {
    "Au": {"111": 2.355, "200": 2.039, "220": 1.442, "311": 1.230},
    "Ag": {"111": 2.359, "200": 2.043, "220": 1.445, "311": 1.232},
    "Al": {"111": 2.338, "200": 2.025, "220": 1.432, "311": 1.221},
    "Cu": {"111": 2.087, "200": 1.808, "220": 1.278, "311": 1.090},
    "Si": {"111": 3.135, "220": 1.920, "311": 1.638, "400": 1.358},
}


class RingOptions(BaseModel):
    # The annulus searched for the ring (pattern px from the centre guess).
    r_min: float = Field(..., gt=0)
    r_max: float = Field(..., gt=0)
    # Where the ring's centre is guessed (pattern px); None: the bright-field disk's.
    cx: float | None = None
    cy: float | None = None
    angles: int = Field(180, ge=12, le=3600)


def ring_points(image: np.ndarray, cx: float, cy: float, r_min: float, r_max: float, angles: int = 180) -> np.ndarray:
    """Points on the ring between ``r_min`` and ``r_max`` around (cx, cy):
    at each of ``angles`` directions, the intensity-weighted radius above the
    profile's minimum (sub-pixel). [n, 2] of (x, y); directions with nothing
    above the minimum are left out."""
    from scipy.ndimage import map_coordinates

    if r_max <= r_min:
        raise ValueError("The ring's outer radius must be larger than its inner")
    radii = np.arange(r_min, r_max + 0.25, 0.25)
    theta = np.linspace(0, 2 * np.pi, angles, endpoint=False)
    xs = cx + radii[None, :] * np.cos(theta)[:, None]
    ys = cy + radii[None, :] * np.sin(theta)[:, None]
    # Pixel i's centre is at i + 0.5: sample at index (position − 0.5).
    prof = map_coordinates(np.asarray(image, dtype=np.float64), [ys - 0.5, xs - 0.5], order=1, mode="nearest")
    w = prof - prof.min(axis=1, keepdims=True)
    s = w.sum(axis=1)
    # Directions with real contrast (not interpolation's rounding on a flat image).
    ok = w.max(axis=1) > 1e-6 * max(float(np.abs(prof).max()), 1e-30)
    r = (w[ok] * radii).sum(axis=1) / s[ok]
    return np.column_stack([cx + r * np.cos(theta[ok]), cy + r * np.sin(theta[ok])])


def fit_ellipse(points: np.ndarray) -> dict:
    """The ellipse through ``points`` ([n, 2] of x, y), by Halir & Flusser's
    direct least-squares fit: {"cx", "cy", "a", "b" (semi-axes, a ≥ b),
    "angle" (of the a axis from x, radians), "rms" (points' distance from it,
    approximately, px)}. ValueError if they don't fit an ellipse."""
    x, y = points[:, 0].astype(float), points[:, 1].astype(float)
    if len(x) < 6:
        raise ValueError("Too few points on the ring to fit an ellipse")
    # Centred and scaled, for conditioning.
    mx, my = x.mean(), y.mean()
    sc = max(np.std(x), np.std(y), 1e-9)
    u, v = (x - mx) / sc, (y - my) / sc
    D1 = np.column_stack([u * u, u * v, v * v])
    D2 = np.column_stack([u, v, np.ones_like(u)])
    S1, S2, S3 = D1.T @ D1, D1.T @ D2, D2.T @ D2
    T = -np.linalg.solve(S3, S2.T)
    M = S1 + S2 @ T
    M = np.array([M[2] / 2, -M[1], M[0] / 2])
    w, vec = np.linalg.eig(M)
    vec = np.real(vec)
    cond = 4 * vec[0] * vec[2] - vec[1] ** 2
    good = np.flatnonzero(cond > 0)
    if not len(good):
        raise ValueError("The points on the ring don't make an ellipse")
    a1 = vec[:, good[0]]
    A, B, C, Dd, E, F = np.concatenate([a1, T @ a1])
    # Back to the image's coordinates: substitute u = (x − mx)/sc, v = (y − my)/sc.
    A2, B2, C2 = A / sc**2, B / sc**2, C / sc**2
    D2c = Dd / sc - 2 * A2 * mx - B2 * my
    E2c = E / sc - 2 * C2 * my - B2 * mx
    F2c = F + A2 * mx**2 + B2 * mx * my + C2 * my**2 - Dd * mx / sc - E * my / sc
    den = B2**2 - 4 * A2 * C2
    cx = (2 * C2 * D2c - B2 * E2c) / den
    cy = (2 * A2 * E2c - B2 * D2c) / den
    num = 2 * (A2 * E2c**2 + C2 * D2c**2 - B2 * D2c * E2c + den * F2c)
    root = np.sqrt((A2 - C2) ** 2 + B2**2)
    ax1 = -np.sqrt(num * (A2 + C2 + root)) / den
    ax2 = -np.sqrt(num * (A2 + C2 - root)) / den
    if not (np.isfinite(ax1) and np.isfinite(ax2)):
        raise ValueError("The points on the ring don't make an ellipse")
    # The angle of the (A2 + C2 + root) axis, then a ≥ b.
    angle = 0.5 * np.arctan2(-B2, C2 - A2)
    a, b = float(ax1), float(ax2)
    if b > a:
        a, b, angle = b, a, angle + np.pi / 2
    angle = float((angle + np.pi / 2) % np.pi - np.pi / 2)
    # Distances (to first order): the radius at each point's angle vs the point's.
    t = np.arctan2(y - cy, x - cx) - angle
    r_e = a * b / np.sqrt((b * np.cos(t)) ** 2 + (a * np.sin(t)) ** 2)
    rms = float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r_e) ** 2)))
    return {"cx": float(cx), "cy": float(cy), "a": a, "b": b, "angle": angle, "rms": rms}


def fit_ring(image: np.ndarray, opts: RingOptions, guess: dict) -> dict:
    """The ring in ``image`` between ``opts.r_min`` and ``r_max`` as an
    ellipse (``fit_ellipse``), around ``guess`` ({"cx", "cy"}) unless the
    options give a centre; twice, the second time around the first fit's
    centre. Adds "radius" (√(a b), px) and "ellipticity" (a / b − 1)."""
    cx = opts.cx if opts.cx is not None else guess["cx"]
    cy = opts.cy if opts.cy is not None else guess["cy"]
    for _ in range(2):
        pts = ring_points(image, cx, cy, opts.r_min, opts.r_max, opts.angles)
        el = fit_ellipse(pts)
        cx, cy = el["cx"], el["cy"]
    return {**el, "radius": float(np.sqrt(el["a"] * el["b"])), "ellipticity": float(el["a"] / el["b"] - 1)}


def affine_of(ellipse: dict) -> list[list[float]]:
    """The affine transform that makes the ring circular, as pyxem's
    calibration takes it (``ransac_ellipse_tools._ellipse_to_affine``)."""
    from pyxem.utils.ransac_ellipse_tools import _ellipse_to_affine

    # pyxem measures the angle the other way round (checked: with −angle the
    # affine maps the ellipse's (x, y) points onto a circle, of radius b).
    return np.asarray(_ellipse_to_affine(ellipse["a"], ellipse["b"], -ellipse["angle"]), dtype=float).tolist()


def wavelength(kv: float) -> float:
    """The relativistic electron wavelength (Å) at ``kv`` kV."""
    h, m, e, c = 6.62607015e-34, 9.1093837015e-31, 1.602176634e-19, 299792458.0
    v = kv * 1e3
    return h / np.sqrt(2 * m * e * v * (1 + e * v / (2 * m * c**2))) * 1e10


def scale_from_ring(radius_px: float, d_spacing: float) -> float:
    """Reciprocal pixel size (1/Å per px) that puts a ring of ``radius_px``
    at 1/d."""
    return (1 / d_spacing) / radius_px


def scale_from_camera(camera_length_mm: float, detector_pixel_um: float, kv: float) -> float:
    """Reciprocal pixel size (1/Å per px) of a detector with pixels of
    ``detector_pixel_um`` at camera length ``camera_length_mm``, at ``kv``:
    θ ≈ p / L per pixel, k = θ / λ."""
    return (detector_pixel_um * 1e-3 / camera_length_mm) / wavelength(kv)
