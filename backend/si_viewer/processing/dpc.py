"""DPC and iDPC images from the four quadrants of a segmented detector.

Each input image is the sum of segments recorded at acquisition (e.g. DF-A is
segments 0 and 4 of the DF-S detector). A segment's quadrant is its number
modulo 4 (inner and outer rings share them), and opposite quadrants differ by
two: the differential signals are A − C (quadrants 0 and 2) and B − D
(quadrants 1 and 3), optionally divided by the sum of all four.

The detector's axes are rotated (and possibly mirrored) relative to the
image. The rotation is found from the data: a gradient field (the deflection
of the beam by the projected potential) has no curl, so the right rotation is
the one that leaves the least. Rotating by θ mixes the field's curl and
divergence, curl(θ) = cos θ · curl₀ + sin θ · div₀, so the least-squares θ
has a closed form (up to 180°, which is resolved by making the integrated
image's sharp extremes positive: atom columns are maxima of the potential).

iDPC integrates the rotated field in Fourier space (least squares;
Lazić et al., Ultramicroscopy 160 (2016) 265, Eq. 46), as periodic as Velox
does: with Velox's rotationalOffset (the DetectorRotation recorded in the
file) as θ, no mirror and the plain differences, this is Velox's iDPC up to a
constant factor.
"""

from __future__ import annotations

import math

import numpy as np

LETTERS = "ABCD"


def quadrant(segments: list[int]) -> int:
    """The quadrant (0–3) of an image summed from ``segments``. ValueError
    if they are not all in one quadrant."""
    if not segments:
        raise ValueError("No detector segments recorded for this image")
    quads = {int(s) % 4 for s in segments}
    if len(quads) != 1:
        raise ValueError(f"Segments {', '.join(map(str, segments))} are not all in one quadrant")
    return quads.pop()


def differences(a, b, c, d, normalise: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(A − C, B − D) in the detector's frame; divided by A + B + C + D if
    ``normalise``. Images, or stacks (frames first)."""
    a, b, c, d = (np.asarray(x, dtype=np.float64) for x in (a, b, c, d))
    if not (a.shape == b.shape == c.shape == d.shape) or a.ndim not in (2, 3):
        raise ValueError("The four images must be of one size")
    dx, dy = a - c, b - d
    if normalise:
        total = a + b + c + d
        scale = np.abs(total)
        floor = max(float(np.median(scale)) * 1e-6, 1e-12)
        total = np.where(scale < floor, floor, total)
        dx, dy = dx / total, dy / total
    return dx, dy


def rotate(dx, dy, theta_deg: float, mirror: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """The field in the image's frame: mirrored (y flipped) if ``mirror``,
    then rotated by ``theta_deg`` (counter-clockwise, x along columns and y
    along rows)."""
    if mirror:
        dy = -dy
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    return c * dx - s * dy, s * dx + c * dy


def _curl_div(dx, dy) -> tuple[np.ndarray, np.ndarray]:
    gx_y, gx_x = np.gradient(dx)
    gy_y, gy_x = np.gradient(dy)
    return gy_x - gx_y, gx_x + gy_y


def _smooth(x, sigma: float):
    """``x`` blurred by a Gaussian of ``sigma`` pixels (periodic, by FFT)."""
    if sigma <= 0:
        return x
    ky = np.fft.fftfreq(x.shape[0])[:, None]
    kx = np.fft.fftfreq(x.shape[1])[None, :]
    g = np.exp(-2 * (math.pi * sigma) ** 2 * (kx * kx + ky * ky))
    return np.real(np.fft.ifft2(np.fft.fft2(x) * g))


def _band(x, fine: float, coarse: float):
    """``x`` with detail finer than ``fine`` and coarser than ``coarse``
    pixels removed (difference of Gaussians)."""
    return _smooth(x, fine) - _smooth(x, coarse)


def find_rotation(dx, dy, fine: float = 2.0, coarse: float = 50.0) -> dict:
    """The rotation (degrees, 0–360) and mirror that leave the field with the
    least curl, and how much is left relative to its divergence
    (``curl_ratio``). Fitted on the field band-passed between ``fine`` and
    ``coarse`` pixels: pixel noise has as much curl as divergence, and slow
    variations (descan, segment gains) are not the specimen's. Of the two
    angles 180° apart (the same curl), the one whose iDPC has its sharp
    extremes positive: atom columns are maxima of the potential."""
    sx = _band(np.asarray(dx, np.float64), fine, coarse)
    sy = _band(np.asarray(dy, np.float64), fine, coarse)
    best = None
    for mirror in (False, True):
        y = -sy if mirror else sy
        curl0, div0 = _curl_div(sx, y)
        cc, dd, cd = float(np.mean(curl0 * curl0)), float(np.mean(div0 * div0)), float(np.mean(curl0 * div0))
        # E(θ) = (cc + dd)/2 + (cc − dd)/2 · cos 2θ + cd · sin 2θ, least at:
        t = 0.5 * (math.atan2(2 * cd, cc - dd) + math.pi)
        energy = (cc + dd) / 2 + (cc - dd) / 2 * math.cos(2 * t) + cd * math.sin(2 * t)
        rest = (cc + dd) / 2 - (cc - dd) / 2 * math.cos(2 * t) - cd * math.sin(2 * t)  # the divergence left
        ratio = math.sqrt(max(energy, 0.0) / rest) if rest > 0 else float("nan")
        if best is None or energy < best["energy"]:
            best = {"theta": math.degrees(t) % 180.0, "mirror": mirror, "energy": energy, "curl_ratio": ratio}
    theta = best["theta"]
    phase = integrate(*rotate(sx, sy, theta, best["mirror"]))
    if skew(phase) < 0:
        theta += 180.0
    return {"theta": theta % 360.0, "mirror": best["mirror"], "curl_ratio": best["curl_ratio"]}


def skew(x) -> float:
    """The skewness of ``x``'s values: positive for sharp maxima on a broad floor."""
    v = np.asarray(x, np.float64).ravel()
    v = v - v.mean()
    s = v.std()
    return float(np.mean(v**3) / s**3) if s > 0 else 0.0


def integrate(gx, gy, pad: bool = False) -> np.ndarray:
    """The potential whose gradient is (gx, gy), least squares in Fourier
    space (the field taken as periodic, as Velox does; ``pad``: from a
    mirror-padded copy instead, so the edges do not streak). In units of the
    field × pixels. Stacks (frames first): each frame."""
    gx, gy = np.asarray(gx, np.float64), np.asarray(gy, np.float64)
    if gx.ndim == 3:
        return np.stack([integrate(x, y, pad) for x, y in zip(gx, gy)])
    # A constant offset of the field (segment gains) would integrate to a ramp.
    gx, gy = gx - gx.mean(), gy - gy.mean()
    h, w = gx.shape
    if pad:
        # Mirror padding makes the field periodic: x flips sign where
        # mirrored along x, y where mirrored along y.
        px = np.block([[gx, -gx[:, ::-1]], [gx[::-1, :], -gx[::-1, ::-1]]])
        py = np.block([[gy, gy[:, ::-1]], [-gy[::-1, :], -gy[::-1, ::-1]]])
    else:
        px, py = gx, gy
    ky = np.fft.fftfreq(px.shape[0])[:, None]
    kx = np.fft.fftfreq(px.shape[1])[None, :]
    k2 = kx * kx + ky * ky
    k2[0, 0] = 1.0
    phi = (kx * np.fft.fft2(px) + ky * np.fft.fft2(py)) / (2j * math.pi * k2)
    phi[0, 0] = 0.0
    return np.real(np.fft.ifft2(phi))[:h, :w]
