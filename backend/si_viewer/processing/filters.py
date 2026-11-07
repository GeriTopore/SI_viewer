"""Non-destructive pre-processing filters (M5).

A dataset's filters are applied whenever its data is read for display, so the
loaded data stays untouched and a filter can be changed or removed at any time.

- **Spatial** (images, stacks, spectrum images): Gaussian, median or mean
  (box) blur. On a spectrum image only the linear ones (Gaussian, mean) are
  offered: they commute with summing channels, so a map of the blurred
  spectrum image is simply the blurred map, and a pixel spectrum is a weighted
  sum of its neighbours. Nothing needs a blurred copy of the whole cube.
- **Spectral** (spectra, spectrum images): Gaussian or Savitzky–Golay smoothing
  along the energy axis, applied to the spectra shown.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator


class Filters(BaseModel):
    spatial: Literal["none", "gaussian", "median", "mean"] = "none"
    # Gaussian σ in pixels.
    sigma: float = Field(1.0, gt=0, le=50)
    # Median / mean window in pixels (odd).
    size: int = Field(3, ge=3, le=51)
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
        return self

    @property
    def has_spatial(self) -> bool:
        return self.spatial != "none"

    @property
    def has_spectral(self) -> bool:
        return self.spectral != "none"

    def spatial_key(self) -> tuple:
        """Changes only when the spatial filter does."""
        if self.spatial == "gaussian":
            return ("gaussian", self.sigma)
        if self.spatial in ("median", "mean"):
            return (self.spatial, self.size)
        return ("none",)

    def describe(self) -> str:
        """e.g. "Gaussian σ 1.5 px; Savitzky–Golay 7 ch, order 2" ("" if none)."""
        parts = []
        if self.spatial == "gaussian":
            parts.append(f"Gaussian σ {self.sigma:g} px")
        elif self.spatial in ("median", "mean"):
            parts.append(f"{self.spatial.capitalize()} {self.size}×{self.size} px")
        if self.spectral == "gaussian":
            parts.append(f"spectral Gaussian σ {self.spectral_sigma:g} ch")
        elif self.spectral == "savgol":
            parts.append(f"Savitzky–Golay {self.window} ch, order {self.order}")
        return "; ".join(parts)


# Which filters each kind of dataset accepts.
SPATIAL_KINDS = {"image", "stack", "spectrum_image"}
SPECTRAL_KINDS = {"spectrum", "spectrum_image"}


def check_applicable(kind: str, f: Filters) -> None:
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
    if f.spatial == "median":
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


def spatial_kernel(f: Filters) -> np.ndarray:
    """The (linear) spatial filter as a normalised 2D kernel."""
    if f.spatial == "gaussian":
        r = max(1, int(np.ceil(4 * f.sigma)))  # as scipy's gaussian_filter
        t = np.arange(-r, r + 1)
        g = np.exp(-0.5 * (t / f.sigma) ** 2)
        k = np.outer(g, g)
    elif f.spatial == "mean":
        k = np.ones((f.size, f.size))
    else:
        raise ValueError(f"No linear kernel for {f.spatial}")
    return k / k.sum()


def filtered_pixel_spectrum(
    data: np.ndarray, x: int, y: int, size: int, f: Filters
) -> tuple[np.ndarray, int]:
    """Summed spectrum of the size × size block centred on (x, y) of the
    spatially filtered spectrum image: a weighted sum of the raw spectra
    around it. Weights falling outside the scan are dropped and the rest
    rescaled, so the block still counts as ``n`` pixels."""
    from scipy.signal import convolve2d

    h, w = data.shape[:2]
    half = (size - 1) // 2
    bx0, by0 = max(0, x - half), max(0, y - half)
    bx1, by1 = min(w, bx0 + size), min(h, by0 + size)
    n = (bx1 - bx0) * (by1 - by0)
    if n == 0 or not (0 <= x < w and 0 <= y < h):
        raise IndexError(f"Pixel ({x}, {y}) is outside the {w}×{h} scan")
    kernel = spatial_kernel(f)
    r = kernel.shape[0] // 2
    weights = convolve2d(np.ones((by1 - by0, bx1 - bx0)), kernel)  # block ⊛ kernel
    wx0, wy0 = bx0 - r, by0 - r
    cx0, cy0 = max(0, wx0), max(0, wy0)
    cx1, cy1 = min(w, wx0 + weights.shape[1]), min(h, wy0 + weights.shape[0])
    weights = weights[cy0 - wy0 : cy1 - wy0, cx0 - wx0 : cx1 - wx0]
    weights *= n / weights.sum()
    window = data[cy0:cy1, cx0:cx1].astype(np.float64)
    return np.tensordot(weights, window, axes=([0, 1], [0, 1])), n


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
