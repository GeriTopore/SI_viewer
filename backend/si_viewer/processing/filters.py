"""Non-destructive pre-processing filters (M5).

A dataset's filters are applied whenever its data is read for display, so the
loaded data stays untouched and a filter can be changed or removed at any time.

- **Spatial** (images, stacks, spectrum images): Gaussian, median or mean
  (box) blur, or a Fourier bandpass. On a spectrum image only the linear ones
  (Gaussian, mean, bandpass) are offered: they commute with summing channels,
  so a map of the filtered spectrum image is simply the filtered map, and the
  summed spectrum of any pixels of it is a weighted sum of the raw spectra
  (``spatial_weights``).
- The **bandpass** is a radially symmetric Fourier-space Gaussian filter, like
  ImageJ's "FFT Bandpass Filter": structures smaller than ``small`` px and
  larger than ``large`` px are suppressed, the mean level is kept. See
  ``bandpass_transfer`` for the exact transfer function.
- **Spectral** (spectra, spectrum images): Gaussian or Savitzky–Golay smoothing
  along the energy axis. A weighted sum of the channels of smoothed spectra is
  a weighted sum of the raw channels (``spectral_adjoint``), so maps and
  per-pixel fits of a smoothed spectrum image need no smoothed copy either.

A spectrum image's filters are its processing: its spectra, maps and fits all
use the processed data.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator


class Filters(BaseModel):
    spatial: Literal["none", "gaussian", "median", "mean", "bandpass"] = "none"
    # Gaussian σ in pixels.
    sigma: float = Field(1.0, gt=0, le=50)
    # Median / mean window in pixels (odd).
    size: int = Field(3, ge=3, le=51)
    # Bandpass: structures smaller than ``small`` px and larger than ``large``
    # px are removed.
    small: float = Field(2.0, gt=0, le=10000)
    large: float = Field(40.0, gt=0, le=10000)
    spectral: Literal["none", "gaussian", "savgol"] = "none"
    # Gaussian σ in channels.
    spectral_sigma: float = Field(1.0, gt=0, le=50)
    # Savitzky–Golay window (channels, odd) and polynomial order.
    window: int = Field(7, ge=3, le=101)
    order: int = Field(2, ge=0, le=6)

    @model_validator(mode="after")
    def _check(self) -> "Filters":
        if self.size % 2 == 0:
            raise ValueError("The filter size must be odd")
        if self.window % 2 == 0:
            raise ValueError("The Savitzky–Golay window must be odd")
        if self.order >= self.window:
            raise ValueError("The Savitzky–Golay order must be less than its window")
        if self.spatial == "bandpass" and self.small >= self.large:
            raise ValueError("The bandpass's small size must be less than its large size")
        return self

    def spatial_params(self) -> dict:
        """The spatial filter's settings (what linked datasets share)."""
        return {"spatial": self.spatial, "sigma": self.sigma, "size": self.size, "small": self.small, "large": self.large}

    @property
    def has_spatial(self) -> bool:
        return self.spatial != "none"

    @property
    def has_spectral(self) -> bool:
        return self.spectral != "none"

    def describe(self) -> str:
        """e.g. "Gaussian σ 1.5 px; Savitzky–Golay 7 ch, order 2" ("" if none)."""
        parts = []
        if self.spatial == "gaussian":
            parts.append(f"Gaussian σ {self.sigma:g} px")
        elif self.spatial in ("median", "mean"):
            parts.append(f"{self.spatial.capitalize()} {self.size}×{self.size} px")
        elif self.spatial == "bandpass":
            parts.append(f"Bandpass {self.small:g}–{self.large:g} px")
        if self.spectral == "gaussian":
            parts.append(f"spectral Gaussian σ {self.spectral_sigma:g} ch")
        elif self.spectral == "savgol":
            parts.append(f"Savitzky–Golay {self.window} ch, order {self.order}")
        return "; ".join(parts)


# Which filters each kind of dataset accepts.
SPATIAL_KINDS = {"image", "stack", "spectrum_image"}
SPECTRAL_KINDS = {"spectrum", "spectrum_image"}


def check_applicable(kind: str, f: Filters, line_scan: bool = False) -> None:
    if f.has_spatial and line_scan:
        raise ValueError("Spatial filters don't apply to line scans (only spectral smoothing does)")
    if f.has_spatial and kind not in SPATIAL_KINDS:
        raise ValueError(f"Spatial filters don't apply to {kind.replace('_', ' ')} datasets")
    if f.has_spectral and kind not in SPECTRAL_KINDS:
        raise ValueError(f"Spectral smoothing doesn't apply to {kind.replace('_', ' ')} datasets")
    if kind == "spectrum_image" and f.spatial == "median":
        raise ValueError("Spectrum images take a Gaussian or mean filter (a median isn't linear)")


def filter_image(arr: np.ndarray, f: Filters) -> np.ndarray:
    """``arr`` (2D) with the spatial filter applied. NaN pixels (e.g. masked
    composition) stay NaN and don't spread into their neighbours."""
    from scipy import ndimage

    if not f.has_spatial:
        return arr
    arr = np.asarray(arr, dtype=np.float32)
    nan = np.isnan(arr)
    if f.spatial == "bandpass":
        if nan.all():
            return arr
        # NaN pixels take the mean of the valid ones for the transform.
        filled = np.where(nan, arr[~nan].mean(), arr) if nan.any() else arr
        out = bandpass(filled, f.small, f.large)
    elif f.spatial == "median":
        if nan.any():
            out = ndimage.generic_filter(arr, np.nanmedian, size=f.size, mode="reflect")
        else:
            out = ndimage.median_filter(arr, size=f.size, mode="reflect")
    else:
        smooth = (
            (lambda a: ndimage.gaussian_filter(a, f.sigma, mode="reflect"))
            if f.spatial == "gaussian"
            else (lambda a: ndimage.uniform_filter(a, f.size, mode="reflect"))
        )
        if nan.any():
            # Normalised convolution: average over the valid neighbours only.
            with np.errstate(invalid="ignore", divide="ignore"):
                out = smooth(np.where(nan, 0, arr)) / smooth((~nan).astype(np.float32))
        else:
            out = smooth(arr)
    out = out.astype(np.float32, copy=False)
    out[nan] = np.nan
    return out


# Bandpass cutoffs: the constants make a sinusoid whose period equals the
# given size pass at half power (amplitude 1/√2) through its own Gaussian.
#   low-pass   exp(-c_low·(f·small)²)        = 2^-½ at f = 1/small
#   high-pass  1 - exp(-c_high·(f·large)²)   = 2^-½ at f = 1/large
_C_LOW = 0.5 * np.log(2.0)
_C_HIGH = -np.log(1.0 - 2.0**-0.5)


def bandpass_transfer(shape: tuple[int, int], small: float, large: float, dtype=np.float64) -> np.ndarray:
    """The bandpass's transfer function H on the ``rfft2`` grid of an image of
    ``shape``. With f the radial spatial frequency in cycles per pixel::

        H(f) = exp(-c_low·(f·small)²) · (1 - exp(-c_high·(f·large)²)),   H(0) = 1

    where c_low = ½·ln 2 and c_high = -ln(1 - 2^-½), so a structure with a
    period of ``small`` (``large``) px is at half power (amplitude 2^-½) of
    the low-pass (high-pass) part. H(0) is set to 1, so the mean level is
    kept. H is real and even, so the filter is a symmetric operator."""
    f2 = np.fft.fftfreq(shape[0])[:, None] ** 2 + np.fft.rfftfreq(shape[1])[None, :] ** 2
    h = np.exp(-_C_LOW * small**2 * f2) * -np.expm1(-_C_HIGH * large**2 * f2)
    h[0, 0] = 1.0
    return h.astype(dtype, copy=False)


def _reflect_plan(shape: tuple[int, int], large: float):
    """Padding for the bandpass: per axis, the source index of every padded
    pixel (scipy's "reflect" edges, ``d c b a | a b c d | d c b a``) and the
    offset of the image in the padded array. Padded to about ``large`` px
    each side (at most one image), and to a length that is fast to transform."""
    from scipy import fft

    src, start = [], []
    for n in shape:
        before = min(int(np.ceil(large)), n)
        total = fft.next_fast_len(n + 2 * before, real=True)
        p = np.mod(np.arange(-before, total - before), 2 * n)
        src.append(np.where(p < n, p, 2 * n - 1 - p))
        start.append(before)
    return src, start


def _apply_h(a: np.ndarray, small: float, large: float) -> np.ndarray:
    from scipy import fft

    h = bandpass_transfer(a.shape, small, large, a.dtype)
    return fft.irfft2(fft.rfft2(a) * h, s=a.shape)


def bandpass(arr: np.ndarray, small: float, large: float) -> np.ndarray:
    """``arr`` (2D, no NaN) with the Fourier bandpass applied (see
    ``bandpass_transfer``), reflect-padded before the transform and cropped
    after. Keeps the dtype (float32 or float64)."""
    arr = np.asarray(arr)
    if arr.dtype not in (np.float32, np.float64):
        arr = arr.astype(np.float64)
    (sy, sx), (y0, x0) = _reflect_plan(arr.shape, large)
    out = _apply_h(arr[np.ix_(sy, sx)], small, large)
    return out[y0 : y0 + arr.shape[0], x0 : x0 + arr.shape[1]]


def bandpass_adjoint(arr: np.ndarray, small: float, large: float) -> np.ndarray:
    """Transpose of ``bandpass``: crop and pad are each other's transpose
    (zero-fill; fold the padding back onto the pixels it was copied from),
    and the filter itself is symmetric."""
    arr = np.asarray(arr, dtype=np.float64)
    (sy, sx), (y0, x0) = _reflect_plan(arr.shape, large)
    z = np.zeros((len(sy), len(sx)))
    z[y0 : y0 + arr.shape[0], x0 : x0 + arr.shape[1]] = arr
    z = _apply_h(z, small, large)
    rows = np.zeros((arr.shape[0], z.shape[1]))
    np.add.at(rows, sy, z)
    out = np.zeros(arr.shape)
    np.add.at(out.T, sx, rows.T)
    return out


def _kernel1d(f: Filters) -> np.ndarray:
    """The (separable, linear) spatial filter along one axis, as scipy applies it."""
    if f.spatial == "gaussian":
        r = int(4.0 * f.sigma + 0.5)  # scipy's truncate=4
        t = np.arange(-r, r + 1)
        k = np.exp(-0.5 * (t / f.sigma) ** 2)
        return k / k.sum()
    if f.spatial == "mean":
        return np.full(f.size, 1.0 / f.size)
    raise ValueError(f"No linear kernel for {f.spatial}")


def _adjoint_1d(a: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
    """Transpose of filtering along ``axis`` with kernel ``k`` (symmetric) and
    scipy's "reflect" edges: spread each value over the padded axis, then fold
    the padding back onto the pixels it was copied from."""
    from scipy.ndimage import convolve1d

    r = len(k) // 2
    n = a.shape[axis]
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r, r)
    ext = convolve1d(np.pad(a, pad), k, axis=axis, mode="constant")
    p = np.mod(np.arange(-r, n + r), 2 * n)
    src = np.where(p < n, p, 2 * n - 1 - p)  # (d c b a | a b c d | d c b a)
    out = np.zeros_like(a)
    np.add.at(np.moveaxis(out, axis, 0), src, np.moveaxis(ext, axis, 0))
    return out


def spatial_weights(mask: np.ndarray, f: Filters) -> np.ndarray:
    """Weights w (same shape as ``mask``) such that summing the spatially
    filtered image over ``mask`` equals Σ w·(raw image), for any image (and so
    for every channel of a spectrum image)."""
    if f.spatial == "bandpass":
        return bandpass_adjoint(mask, f.small, f.large)
    k = _kernel1d(f)
    return _adjoint_1d(_adjoint_1d(np.asarray(mask, np.float64), k, 0), k, 1)


def filtered_sum(data: np.ndarray, mask: np.ndarray, raw_sum: np.ndarray, f: Filters, chunk: int = 2048) -> np.ndarray:
    """Summed spectrum over ``mask`` (full scan shape) of the spatially
    filtered spectrum image, from the raw sum over it: only pixels whose
    weight differs from the mask (near its edge) need adding."""
    delta = spatial_weights(mask, f) - mask
    if f.spatial == "bandpass":
        # The weights reach every pixel: sum over the scan in blocks of rows.
        out = np.asarray(raw_sum, np.float64).copy()
        h, w = delta.shape
        rows = max(1, chunk // w)
        for y in range(0, h, rows):
            block = np.asarray(data[y : y + rows], np.float32)
            out += delta[y : y + rows].reshape(-1) @ block.reshape(-1, block.shape[-1])
        return out
    ys, xs = np.nonzero(np.abs(delta) > 1e-12)
    out = np.asarray(raw_sum, np.float64).copy()
    for i in range(0, len(ys), chunk):
        y, x = ys[i : i + chunk], xs[i : i + chunk]
        out += delta[y, x] @ data[y, x].astype(np.float32)
    return out


def spectral_adjoint(weights: np.ndarray, f: Filters, n_channels: int, offset: int = 0, chunk: int = 256) -> np.ndarray:
    """Channel weights for the raw spectrum equivalent to ``weights`` (n × k,
    on channels offset … offset + n) applied to the smoothed spectrum: the
    (n_channels × k) matrix Sᵀ·W, where S is the smoothing."""
    weights = np.asarray(weights, np.float64)
    n = weights.shape[0]
    out = np.empty((n_channels, weights.shape[1]))
    for i in range(0, n_channels, chunk):
        rows = np.zeros((min(chunk, n_channels - i), n_channels))
        rows[np.arange(len(rows)), np.arange(i, i + len(rows))] = 1
        smoothed = filter_spectrum(rows, f)  # row j: S applied to unit channel i + j
        out[i : i + len(rows)] = smoothed[:, offset : offset + n] @ weights
    return out


def weighted_channel_map(data, weights: np.ndarray, f: Filters) -> np.ndarray:
    """Σ_c weights[c]·(spectrally smoothed spectrum)[c] at every pixel: a
    window map of the smoothed spectrum image, from the raw channels near the
    windows only."""
    v = spectral_adjoint(np.asarray(weights)[:, None], f, len(weights))[:, 0] if f.has_spectral else np.asarray(weights)
    nz = np.flatnonzero(np.abs(v) > 1e-9 * np.abs(v).max()) if np.any(v) else np.array([0])
    lo, hi = nz[0], nz[-1] + 1
    result = np.tensordot(data[..., lo:hi], v[lo:hi].astype(np.float32), axes=([-1], [0]))
    return np.asarray(result.compute() if hasattr(result, "compute") else result, np.float32)


def filter_spectrum(spectrum: np.ndarray, f: Filters) -> np.ndarray:
    """``spectrum`` smoothed along its last (energy) axis."""
    if not f.has_spectral:
        return spectrum
    spectrum = np.asarray(spectrum, dtype=np.float64)
    if f.spectral == "gaussian":
        from scipy.ndimage import gaussian_filter1d

        return gaussian_filter1d(spectrum, f.spectral_sigma, axis=-1, mode="nearest")
    from scipy.signal import savgol_filter

    if spectrum.shape[-1] < f.window:
        return spectrum
    return savgol_filter(spectrum, f.window, f.order, axis=-1, mode="interp")
