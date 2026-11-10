"""Plural-scattering deconvolution of EELS spectrum images (M10d).

Three methods, each with the low loss of the spectrum image's link group:

- **Fourier-log** (exSpy's ``fourier_log_deconvolution``): the single
  scattering distribution of a low loss, with its own zero-loss peak.
- **Fourier-ratio** (exSpy's ``fourier_ratio_deconvolution``): a core loss
  with its background removed, divided by the low loss in Fourier space (and
  convolved with a Gaussian of the zero-loss peak's width).
- **Richardson–Lucy**: iterative deconvolution by a point-spread function, the
  whole low loss (plural scattering) or its zero-loss peak alone (energy
  resolution). Each pixel by its own low loss, centred on its maximum and
  normalised to a sum of 1, so the counts keep their scale; written for many
  pixels at once with FFT convolutions. Negative counts are taken as zero.

The zero-loss peak is the low loss up to a threshold: one energy for all
pixels, or estimated at every pixel as for the thickness (exSpy's
``estimate_elastic_scattering_threshold``; a pixel with no estimate uses the
estimate for the summed low loss). Everything runs on chunks of pixels so that
a job can report progress and be cancelled.
"""

from __future__ import annotations

import math
from typing import Callable, Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator

from .background import EnergyAxis
from .thickness import ThicknessOptions, _spectra, _tolerance, check_options, quiet

Method = Literal["fourier_log", "fourier_ratio", "richardson_lucy"]
METHOD_NAMES = {"fourier_log": "Fourier-log", "fourier_ratio": "Fourier-ratio", "richardson_lucy": "Richardson–Lucy"}

# Pixels per step.
CHUNK = 256
# Energy scales of the spectrum and the low loss must agree this well.
SCALE_TOLERANCE = 1e-3

Report = Callable[..., None]


class DeconvolutionOptions(BaseModel):
    method: Method
    # End of the zero-loss peak: "auto" (estimated at every pixel, searching
    # ``window`` above the peak) or one energy (the low loss' energy units).
    threshold: Literal["auto"] | float = "auto"
    window: float = Field(10.0, gt=0, le=1000)
    # Fourier-log: add the zero-loss peak back to the result.
    add_zlp: bool = False
    # Fourier-ratio: width of the Gaussian the result is convolved with (None:
    # the zero-loss peak's, measured on the summed low loss), and whether both
    # spectra are extrapolated with a power law before transforming.
    fwhm: float | None = Field(None, gt=0)
    extrapolate: bool = True
    # Richardson–Lucy: iterations, and the point-spread function.
    iterations: int = Field(15, ge=1, le=500)
    psf: Literal["low_loss", "zlp"] = "low_loss"

    @field_validator("threshold")
    @classmethod
    def _finite(cls, v):
        if v != "auto" and not math.isfinite(v):
            raise ValueError("The threshold must be a number")
        return v

    @property
    def auto(self) -> bool:
        return self.threshold == "auto"

    @property
    def needs_threshold(self) -> bool:
        """Whether the zero-loss peak's end is used."""
        return self.method != "richardson_lucy" or self.psf == "zlp"


def describe(options: DeconvolutionOptions, units: str = "") -> str:
    """e.g. "Richardson–Lucy, low loss, 15 iterations"."""
    u = f" {units}" if units else ""
    parts = [METHOD_NAMES[options.method]]
    if options.method == "richardson_lucy":
        parts += ["low loss" if options.psf == "low_loss" else "zero-loss peak", f"{options.iterations} iterations"]
    if options.needs_threshold:
        parts.append(f"threshold auto ({options.window:g}{u})" if options.auto else f"threshold {options.threshold:g}{u}")
    if options.method == "fourier_ratio" and options.fwhm is not None:
        parts.append(f"FWHM {options.fwhm:g}{u}")
    return ", ".join(parts)


def check(src, ll, options: DeconvolutionOptions) -> None:
    """Raise ValueError (in plain English) unless ``src`` can be deconvolved
    with the low loss ``ll`` (two signals of the same scan)."""
    a, b = EnergyAxis.of(src), EnergyAxis.of(ll)
    if not (a.scale > 0 and b.scale > 0):
        raise ValueError("Deconvolution needs energy axes that increase")
    if a.units != b.units or not math.isclose(a.scale, b.scale, rel_tol=SCALE_TOLERANCE):
        raise ValueError(
            f"The spectrum ({a.scale:g} {a.units}/channel) and the low loss ({b.scale:g} {b.units}/channel) "
            "need the same energy scale"
        )
    if tuple(src.axes_manager.navigation_shape) != tuple(ll.axes_manager.navigation_shape):
        raise ValueError("The spectrum and the low loss need the same scan")
    if options.needs_threshold:
        check_options(ll, ThicknessOptions(threshold=options.threshold, window=options.window))


# ---------- the zero-loss peak ----------


def _thresholds(ll_spectra, options: DeconvolutionOptions, tol: float | None, fallback: float | None) -> np.ndarray:
    """The end of the zero-loss peak of each spectrum (energy units)."""
    n = ll_spectra.axes_manager.navigation_size or 1
    if not options.auto:
        return np.full(n, float(options.threshold))
    t = ll_spectra.estimate_elastic_scattering_threshold(window=options.window, tol=tol)
    out = np.asarray(t.data, dtype=float).ravel()
    bad = ~np.isfinite(out)
    if bad.any():
        if fallback is None:
            raise ValueError("The end of the zero-loss peak could not be found: set the threshold by hand")
        out[bad] = fallback
    return out


def zero_loss_peak(ll: np.ndarray, axis: EnergyAxis, thresholds: np.ndarray) -> np.ndarray:
    """The low loss spectra (n, channels) cut to zero from each one's threshold on."""
    energies = axis.energies(0, axis.size)
    return np.where(energies[None, :] < thresholds[:, None], ll, 0.0)


# ---------- Richardson–Lucy ----------


def richardson_lucy(signal: np.ndarray, kernel: np.ndarray, iterations: int) -> np.ndarray:
    """Richardson–Lucy deconvolution of each spectrum (n, N) by its own kernel
    (n, K), centred on the kernel's maximum and normalised to a sum of 1."""
    from scipy.fft import irfft, next_fast_len, rfft

    n, size = signal.shape
    k = np.clip(np.asarray(kernel, dtype=float), 0, None)
    total = k.sum(axis=1, keepdims=True)
    k = k / np.where(total > 0, total, 1.0)
    length = next_fast_len(size + k.shape[1], real=True)
    # Each kernel with its maximum at lag 0 (earlier channels at negative lags,
    # wrapped to the end); the padding keeps the circular convolution linear.
    lags = (np.arange(k.shape[1])[None, :] - k.argmax(axis=1)[:, None]) % length
    padded = np.zeros((n, length))
    np.put_along_axis(padded, lags, k, axis=1)
    f = rfft(padded, axis=1)

    y = np.clip(np.nan_to_num(np.asarray(signal, dtype=float)), 0, None)
    x = y.copy()
    eps = 1e-12 * max(float(y.max(initial=0.0)), 1.0)
    for _ in range(iterations):
        estimate = irfft(rfft(x, length, axis=1) * f, length, axis=1)[:, :size]
        ratio = y / np.maximum(estimate, eps)
        x *= np.clip(irfft(rfft(ratio, length, axis=1) * np.conj(f), length, axis=1)[:, :size], 0, None)
    return x


# ---------- one block of pixels ----------


def deconvolve_block(
    src, ll, src_data: np.ndarray, ll_data: np.ndarray, options: DeconvolutionOptions,
    tol: float | None = None, fallback: float | None = None, fwhm: float | None = None,
) -> np.ndarray:
    """Deconvolve the spectra ``src_data`` (n, N; on ``src``'s energy axis)
    with the matching low loss spectra ``ll_data`` (n, M; on ``ll``'s). A
    pixel without counts in either gives zeros."""
    src_data = np.nan_to_num(np.asarray(src_data, dtype=float))
    ll_data = np.nan_to_num(np.asarray(ll_data, dtype=float))
    out = np.zeros_like(src_data)
    live = (np.abs(src_data).sum(axis=1) > 0) & (np.abs(ll_data).sum(axis=1) > 0)
    if not live.any():
        return out
    with quiet(), np.errstate(all="ignore"):
        thresholds = (
            _thresholds(_spectra(ll, ll_data[live]), options, tol, fallback) if options.needs_threshold else None
        )
        out[live] = _deconvolve(src, ll, src_data[live], ll_data[live], options, thresholds, tol, fwhm)
    out[~np.isfinite(out)] = 0.0
    return out


def _deconvolve(src, ll, src_data, ll_data, options: DeconvolutionOptions, thresholds, tol, fwhm) -> np.ndarray:
    """``deconvolve_block`` of spectra that all have counts."""
    ll_axis = EnergyAxis.of(ll)
    size = src_data.shape[1]
    if options.method == "richardson_lucy":
        kernel = ll_data if options.psf == "low_loss" else zero_loss_peak(ll_data, ll_axis, thresholds)
        return richardson_lucy(src_data, kernel, options.iterations)
    if options.method == "fourier_log":
        zlp = _spectra(ll, zero_loss_peak(ll_data, ll_axis, thresholds))
        out = _spectra(src, src_data).fourier_log_deconvolution(zlp, add_zlp=options.add_zlp).data
    else:
        low = _spectra(ll, ll_data)
        if options.auto:
            threshold = low.estimate_elastic_scattering_threshold(window=options.window, tol=tol)
            threshold.data = thresholds.reshape(threshold.data.shape)
        else:
            threshold = float(options.threshold)
        core = _spectra(src, src_data)
        # exSpy's own extrapolation gives up on a whole block when one spectrum
        # has no counts at its end: the same estimate, per spectrum, instead.
        for s in (core, low):
            s.power_law_extrapolation = _robust_extrapolation.__get__(s)
        out = core.fourier_ratio_deconvolution(
            low, fwhm=options.fwhm or fwhm, threshold=threshold,
            extrapolate_lowloss=options.extrapolate, extrapolate_coreloss=options.extrapolate,
        ).data
    return np.asarray(out, dtype=float).reshape(len(out), -1)[:, :size]


def _robust_extrapolation(self, window_size=20, extrapolation_size=1024, **_):
    """``EELSSpectrum.power_law_extrapolation`` of one of ``_spectra``'s
    signals (n spectra on one navigation axis), by ``power_law_extrapolation``."""
    s = self.deepcopy()
    s.data = power_law_extrapolation(
        np.asarray(self.data, dtype=float), EnergyAxis.of(self), window_size, extrapolation_size
    )
    s.get_dimensions_from_data()
    return s


def power_law_extrapolation(data: np.ndarray, axis: EnergyAxis, window: int, extra: int) -> np.ndarray:
    """The spectra (n, channels; counts per channel) continued by ``extra``
    channels of the power law fitted to their last ``window`` channels by
    HyperSpy's two-area estimate (exSpy's ``power_law_extrapolation``). A
    spectrum whose estimate fails (no counts, or negative ones, at its end) is
    continued with zeros."""
    size = data.shape[1]
    i1, i4 = size - window, size - 1
    if (i4 + i1) % 2:
        i4 -= 1
    if i4 == i1:
        i4 += 2
    i3 = (i4 + i1) // 2
    x1, x2, x3, x4 = (axis.offset + axis.scale * i for i in (i1, i3, i3, i4))
    a = data[:, i1:i3].sum(axis=1)
    b = data[:, i3:i4].sum(axis=1)
    with np.errstate(all="ignore"):
        r = 2 * np.log(a / b * (x4 - x3) / (x2 - x1)) / np.log(x4 * x3 / x2 / x1)
        k = 1 - r
        amp = (k * a / (x2**k - x1**k) * a + k * b / (x4**k - x3**k) * b) / (a + b)
        r, amp = np.nan_to_num(r), np.nan_to_num(amp)
        x = axis.energies(size, extra)
        tail = axis.scale * amp[:, None] * x[None, :] ** (-r[:, None])
    tail[~np.isfinite(tail)] = 0.0
    return np.concatenate([data, tail], axis=1)


def summed_threshold(ll, ll_sum: np.ndarray, options: DeconvolutionOptions) -> float | None:
    """The end of the zero-loss peak of the summed low loss (None if not found)."""
    if not options.needs_threshold:
        return None
    if not options.auto:
        return float(options.threshold)
    with quiet():
        t = _spectra(ll, np.asarray(ll_sum, dtype=float)[None, :]).estimate_elastic_scattering_threshold(
            window=options.window
        )
    value = float(np.ravel(t.data)[0])
    return value if math.isfinite(value) else None


def zlp_width(ll, ll_sum: np.ndarray) -> float | None:
    """The FWHM of the summed low loss' zero-loss peak (energy units)."""
    with quiet(), np.errstate(all="ignore"):
        w = _spectra(ll, np.asarray(ll_sum, dtype=float)[None, :]).estimate_peak_width()
    value = float(np.ravel(w.data)[0])
    return value if math.isfinite(value) and value > 0 else None


def preview(src, ll, pairs: list[tuple[np.ndarray, np.ndarray]], options: DeconvolutionOptions) -> dict:
    """The deconvolved spectrum of each (spectrum, low loss) pair (e.g. summed
    over a region), and the threshold and FWHM used for the first."""
    check(src, ll, options)
    if not pairs:
        return {"spectra": [], "threshold": None, "fwhm": None}
    s = np.stack([p[0] for p in pairs])
    lows = np.stack([p[1] for p in pairs])
    fallback = summed_threshold(ll, lows.sum(axis=0), options)
    fwhm = zlp_width(ll, lows.sum(axis=0)) if options.method == "fourier_ratio" and options.fwhm is None else None
    out = np.vstack([deconvolve_block(src, ll, s[i : i + 1], lows[i : i + 1], options, None, fallback, fwhm) for i in range(len(s))])
    first = _thresholds(_spectra(ll, lows[:1]), options, None, fallback)[0] if options.needs_threshold else None
    return {"spectra": out, "threshold": first, "fwhm": options.fwhm or fwhm}


def deconvolve(src, ll, options: DeconvolutionOptions, report: Report = lambda f, m=None: None):
    """A copy of the EELS spectra ``src`` (in memory; one or two navigation
    axes) deconvolved with the low loss ``ll`` of the same scan.
    ``report(fraction, message)`` is called after each chunk."""
    check(src, ll, options)
    n_src = src.data.shape[-1]
    nav = tuple(src.data.shape[:-1])
    flat = np.asarray(src.data).reshape(-1, n_src)
    low = np.asarray(ll.data).reshape(-1, ll.data.shape[-1])
    n = len(flat)
    starts = range(0, n, CHUNK)
    ll_sum = np.nan_to_num(low.astype(float)).sum(axis=0)
    fallback = summed_threshold(ll, ll_sum, options)
    fwhm = zlp_width(ll, ll_sum) if options.method == "fourier_ratio" and options.fwhm is None else None
    share = 0.0
    tol = None
    if options.needs_threshold and options.auto:
        share = 0.2
        tolerances = []
        with quiet():
            for i, start in enumerate(starts):
                chunk = _spectra(ll, np.nan_to_num(low[start : start + CHUNK].astype(float)))
                tolerances.append(_tolerance(chunk, options.window))
                report(share * (i + 1) / len(starts), "Finding the zero-loss peak")
        tol = max(tolerances)
    out = np.empty((n, n_src), np.float32)
    for i, start in enumerate(starts):
        stop = start + CHUNK
        out[start:stop] = deconvolve_block(src, ll, flat[start:stop], low[start:stop], options, tol, fallback, fwhm)
        report(share + (1 - share) * (i + 1) / len(starts), "Deconvolving")
    result = src.deepcopy()
    result.data = out.reshape(*nav, n_src)
    result.metadata.set_item("Processing.si_viewer.deconvolution", describe(options, EnergyAxis.of(src).units))
    return result
