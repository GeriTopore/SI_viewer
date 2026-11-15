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


# ---------- M22b: Fourier filtering, phase and strain ----------

PhaseMethod = Literal["lockin", "gpa"]
# What can be made: per peak (amplitude, grating, phase, strain from one
# peak) and from two peaks (the strain tensor).
PER_PEAK = ("amplitude", "grating", "phase", "strain")


@dataclass(frozen=True)
class FourierOptions:
    # Gaussian filter round each peak (σ, FFT px).
    sigma: float = 20.0
    phase: PhaseMethod = "lockin"
    # The lock-in's low-pass (σ, FFT px); None: twice the filter's.
    lock_sigma: float | None = None
    edge: Edge = "none"
    # Strain values beyond this (|ε|) are replaced by their median-filtered
    # surroundings (kemstem's mask_threshold); None: kept.
    mask_threshold: float | None = None

    def check(self) -> "FourierOptions":
        if not self.sigma > 0:
            raise ValueError("The filter's σ must be more than 0")
        if self.lock_sigma is not None and not self.lock_sigma > 0:
            raise ValueError("The lock-in σ must be more than 0")
        if self.mask_threshold is not None and not self.mask_threshold > 0:
            raise ValueError("The outlier limit must be more than 0")
        return self


def coarsening(shape: tuple[int, int], sigma: float) -> float:
    """Real-space coarsening radius (image px) of a Gaussian of σ FFT px
    (kemstem's ``coarsening_length``, with the image's first axis as it uses)."""
    return float(_kemstem().coarsening_length(shape[0], sigma))


def _array_pixel(f, shape):
    """Frequency → pixel of the centred FFT array (zero at N // 2): where the
    filter goes."""
    return to_pixels([f], shape)[0]


def _half_pixel(f, shape):
    """Frequency → kemstem's pixel convention for references and strain
    (zero at N / 2): ``(f·H + H/2, f·W + W/2)``. With these, the lock-in
    reference, the GPA carrier and the strain scale are the peak's true
    frequency also for odd sizes."""
    h, w = shape
    return np.array([f[0] * h + h / 2, f[1] * w + w / 2])


def fourier_maps(image: np.ndarray, peaks: dict[int, tuple[float, float]], outputs: set[str],
                 opts: FourierOptions, per_peak: list[int] | None = None,
                 tensor: tuple[int, int] | None = None, report=None) -> list[dict]:
    """kemstem's Fourier filtering of the normalised image round each peak
    (``fourier_filter``), and from it: the amplitude |filtered|, the grating
    Re(filtered), the phase (``phaselock`` or ``gpa``, wrapped to 0–2π),
    ε∥ and ε⊥ (``phase_to_strain``), and, for the two peaks ``tensor``, εxx,
    εyy, εxy and ωxy (``calculate_strain_tensor``). ``peaks``: number (1-based,
    as listed) → frequency, for every peak used; ``per_peak``: the peaks
    ``outputs`` are made for (default all). Returns dicts ``output``,
    ``peaks`` (numbers), ``title`` and ``data`` (float32)."""
    opts.check()
    per_peak = list(peaks) if per_peak is None else per_peak
    if tensor is not None:
        g = np.array([peaks[tensor[0]], peaks[tensor[1]]])
        if tensor[0] == tensor[1] or abs(np.linalg.det(g)) < 1e-12:
            raise ValueError("The strain tensor needs two peaks in different directions")
    k = _kemstem()
    ft = fourier_pattern(image, opts.edge)
    shape = ft.shape
    lock = opts.lock_sigma or 2 * opts.sigma
    need_phase = bool({"phase", "strain"} & outputs)
    numbers = sorted(set(per_peak) | set(tensor or ()))
    phases: dict[int, np.ndarray] = {}
    out: list[dict] = []
    steps = len(numbers) + (1 if tensor else 0)

    def add(output, nums, title, data):
        out.append({"output": output, "peaks": list(nums), "title": title,
                    "data": np.ascontiguousarray(data, dtype=np.float32)})

    for i, n in enumerate(numbers):
        if report:
            report(i / steps, f"Peak {n}")
        f = peaks[n]
        filtered, _, _ = k.fourier_filter(ft, _array_pixel(f, shape), opts.sigma)
        mine = n in per_peak
        if "amplitude" in outputs and mine:
            add("amplitude", [n], f"Amplitude g{n}", np.abs(filtered))
        if "grating" in outputs and mine:
            add("grating", [n], f"Grating g{n}", np.real(filtered))
        if (need_phase and mine) or (tensor and n in tensor):
            ph = (k.phaselock(filtered, _half_pixel(f, shape), lock) if opts.phase == "lockin"
                  else np.mod(k.gpa(filtered, _half_pixel(f, shape)), 2 * np.pi))
            phases[n] = ph
            if "phase" in outputs and mine:
                add("phase", [n], f"Phase g{n}", ph)
            if "strain" in outputs and mine:
                par, perp = k.phase_to_strain(ph, _half_pixel(f, shape), opts.mask_threshold)
                add("strain-par", [n], f"ε∥ g{n}", par)
                add("strain-perp", [n], f"ε⊥ g{n}", perp)
    if tensor:
        if report:
            report(len(numbers) / steps, "Strain tensor")
        a, b = tensor
        exx, eyy, exy, oxy = k.calculate_strain_tensor(
            phases[a], phases[b], _half_pixel(peaks[a], shape), _half_pixel(peaks[b], shape), opts.mask_threshold)
        label = f"g{a}, g{b}"
        for output, title, data in (("exx", "εxx", exx), ("eyy", "εyy", eyy), ("exy", "εxy", exy), ("oxy", "ωxy", oxy)):
            add(output, [a, b], f"{title} ({label})", data)
    return out
