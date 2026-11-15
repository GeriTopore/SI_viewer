"""Atomic-resolution image analysis with kemstem (M22): glue only.

kemstem works on plain arrays: images normalised to 0–1, points as (n, 2)
arrays of (y, x) pixels, FFTs centred (``fftshift``). FFT peaks are kept here
and in the frontend as spatial frequencies in cycles per image pixel
``(fy, fx)``: pixel (r, c) of the centred FFT of an H × W image is
``((r - H // 2) / H, (c - W // 2) / W)``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import numpy as np

# kemstem's loops print tqdm bars; keep the server log clean.
os.environ.setdefault("TQDM_DISABLE", "1")

Method = Literal["gauss", "com"]
Edge = Literal["none", "hann", "pplus"]
Pattern = Literal["log", "abs"]


def _kemstem():
    import kemstem

    return kemstem


@dataclass(frozen=True)
class PeakOptions:
    method: Method = "gauss"
    # Full width of the fit window (Gaussian) or of the centre-of-mass crop (px of the FFT).
    window: int = 9
    # Gaussian blur (σ, px) of the pattern before refining, as in kemstem's notebooks.
    blur: float = 1.0
    edge: Edge = "none"
    # The pattern refined: log(1 + |F|) (positions) or |F| (backgrounds, for damping).
    pattern: Pattern = "log"

    def check(self) -> "PeakOptions":
        if self.window < 3 or self.window % 2 == 0:
            raise ValueError("The window must be an odd number of pixels, 3 or more")
        if self.blur < 0:
            raise ValueError("The blur must be 0 or more")
        return self


def fourier_pattern(image: np.ndarray, edge: Edge = "none") -> np.ndarray:
    """kemstem's centred complex FFT of the normalised image."""
    k = _kemstem()
    return k.prepare_fourier_pattern(k.normalize(np.asarray(image, dtype=np.float64)),
                                     edge_reduction=None if edge == "none" else edge)


def to_pixels(freqs: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """(n, 2) frequencies (fy, fx; cycles/px) → (y, x) pixels of the centred FFT."""
    h, w = shape
    f = np.asarray(freqs, dtype=np.float64).reshape(-1, 2)
    return np.stack([f[:, 0] * h + h // 2, f[:, 1] * w + w // 2], axis=1)


def to_freqs(pixels: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    p = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    return np.stack([(p[:, 0] - h // 2) / h, (p[:, 1] - w // 2) / w], axis=1)


def refine_peaks(image: np.ndarray, freqs: np.ndarray, opts: PeakOptions) -> list[dict]:
    """Refine FFT peaks of ``image`` (frequencies) with kemstem: 2D Gaussian
    fits (``refine_peaks_gf``) or centre of mass (``refine_peaks_com``) on the
    blurred log (or amplitude) pattern. One dict per peak, in order: ``fy``,
    ``fx`` and ``failed`` (the input kept), and for Gaussian fits
    ``amplitude``, ``background``, ``sigma`` (px of the FFT)."""
    from scipy.ndimage import gaussian_filter

    opts.check()
    k = _kemstem()
    ft = fourier_pattern(image, opts.edge)
    pattern = np.abs(ft) if opts.pattern == "abs" else np.log1p(np.abs(ft))
    if opts.blur > 0:
        pattern = gaussian_filter(pattern, opts.blur)
    shape = pattern.shape
    p0 = to_pixels(freqs, shape)
    if not len(p0):
        return []
    half = opts.window // 2
    inside = (p0[:, 0] >= half + 1) & (p0[:, 0] < shape[0] - half - 1) & (p0[:, 1] >= half + 1) & (p0[:, 1] < shape[1] - half - 1)
    out = [{"fy": float(f[0]), "fx": float(f[1]), "failed": True} for f in to_freqs(p0, shape)]
    if not inside.any():
        return out
    if opts.method == "com":
        refined = k.refine_peaks_com(pattern, p0[inside], half)
        failed = ~np.isfinite(refined).all(axis=1)
        params = None
    else:
        refined, failed, params, _ = k.refine_peaks_gf(pattern, p0[inside], window_dimension=opts.window,
                                                       remove_unfit=False, verbose=False)
    for j, i in enumerate(np.flatnonzero(inside)):
        if failed[j]:
            continue
        fy, fx = to_freqs(refined[j], shape)[0]
        peak = {"fy": float(fy), "fx": float(fx), "failed": False}
        if params is not None:
            amp, _, _, s1, s2, _, offset = params[j]
            peak.update(amplitude=float(amp), background=float(offset), sigma=[float(s1), float(s2)])
        out[i] = peak
    return out
