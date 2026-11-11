"""Background removal for EELS spectra (M10e).

The background under an edge is fitted in a *background* window before it and
subtracted over a *signal* window after it. The estimates are HyperSpy's own
(``Signal1D.remove_background``: the two-area power law, the geometric-mean
exponential, polynomial and offset fits), written for many spectra at once and
per pixel, so that a single dead pixel doesn't spoil the estimate for the rest.
A full fit refines the power law or exponential of every pixel by least squares
with HyperSpy's own model, starting from the estimates. A log fit is a linear
least-squares fit of ln I to ln E (power law) or E (exponential), at every
pixel at once.

LBA (local background averaging): before the fit, each channel of the
background window is smoothed across the scan with a Gaussian (its FWHM in
pixels), and each pixel's smoothed window is scaled back to that pixel's own
mean; the background is fitted to that and subtracted from the raw spectrum.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

from .regions import channel_span, window_label

MODELS = ("power_law", "exponential", "polynomial", "offset")
MODEL_NAMES = {
    "power_law": "power law", "exponential": "exponential", "polynomial": "polynomial", "offset": "offset",
}

# Pixels fitted per step: the fast estimates are vectorised; the full fits
# take a few milliseconds per pixel.
FAST_CHUNK = 4096
FULL_CHUNK = 128


class BackgroundOptions(BaseModel):
    model: Literal["power_law", "exponential", "polynomial", "offset"] = "power_law"
    # Polynomial only.
    order: int = Field(2, ge=1, le=6)
    # Fast: the closed-form estimate; else a least-squares fit at every pixel.
    fast: bool = True
    # Power law and exponential: a linear fit of the logarithm (overrides fast).
    log: bool = False
    # LBA: FWHM in pixels of the Gaussian the background window is averaged
    # with across the scan; 0 for none.
    lba: float = Field(0.0, ge=0, le=50)
    # [lo, hi] in the energy axis' units: the window the background is fitted
    # in, and the window kept (with the background subtracted).
    background: tuple[float, float]
    signal: tuple[float, float]

    @field_validator("background", "signal")
    @classmethod
    def _numbers(cls, window: tuple[float, float]) -> tuple[float, float]:
        if not all(math.isfinite(v) for v in window):
            raise ValueError("The window's limits must be numbers")
        lo, hi = sorted(window)
        if not lo < hi:
            raise ValueError("A window needs a start below its end")
        return (lo, hi)

    @model_validator(mode="after")
    def _ordered(self) -> "BackgroundOptions":
        if self.background[1] > self.signal[0]:
            raise ValueError("The background window must end before the signal window starts")
        return self


@dataclass(frozen=True)
class EnergyAxis:
    offset: float
    scale: float
    size: int
    binned: bool
    units: str

    @classmethod
    def of(cls, signal) -> "EnergyAxis":
        axis = signal.axes_manager.signal_axes[0]
        units = axis.units if isinstance(axis.units, str) and axis.units != "None" else ""
        return cls(float(axis.offset), float(axis.scale), int(axis.size), bool(axis.is_binned), units)

    @property
    def end(self) -> float:
        return self.offset + self.scale * self.size

    def energies(self, first: int, count: int) -> np.ndarray:
        return self.offset + self.scale * (first + np.arange(count))


@dataclass(frozen=True)
class Windows:
    """The channels of the two windows: ``[b0, b1)`` is fitted, ``[s0, s1)`` kept."""

    b0: int
    b1: int
    s0: int
    s1: int

    @property
    def span(self) -> int:
        """Channels from the start of the background to the end of the signal window."""
        return self.s1 - self.b0


def uses_log(options: BackgroundOptions) -> bool:
    return options.log and options.model in ("power_law", "exponential")


def min_background_channels(options: BackgroundOptions) -> int:
    if options.model in ("power_law", "exponential"):
        return 3
    return max(2, options.order + 1) if options.model == "polynomial" else 2


def automatic_title(options: BackgroundOptions, axis: EnergyAxis) -> str:
    return f"{window_label(*options.signal, axis.units, axis.scale)}, bg removed"


def describe(options: BackgroundOptions, units: str = "") -> str:
    """e.g. "power law (fast), background 800–920 eV, signal 925–1010 eV"."""
    name = MODEL_NAMES[options.model] + (f" (order {options.order})" if options.model == "polynomial" else "")
    fit = "log fit" if uses_log(options) else "fast" if options.fast else "full"
    lba = f", LBA {options.lba:g} px" if options.lba > 0 else ""
    return (
        f"{name} ({fit}){lba}, background {window_label(*options.background, units)}"
        f", signal {window_label(*options.signal, units)}"
    )


def channel_windows(axis: EnergyAxis, options: BackgroundOptions) -> Windows:
    """The channels of ``options``' windows on ``axis``. Raises ValueError (in
    plain English) when a window is outside the spectrum or too narrow."""
    if not axis.scale > 0:
        raise ValueError("Background removal needs an energy axis that increases")
    available = window_label(axis.offset, axis.end, axis.units)
    eps = 1e-9 * max(1.0, abs(axis.offset), abs(axis.end))
    spans = {}
    for name, (lo, hi) in (("background", options.background), ("signal", options.signal)):
        if lo < axis.offset - eps or hi > axis.end + eps:
            raise ValueError(f"The {name} window must lie inside the spectrum's energy range ({available})")
        spans[name] = channel_span(axis.size, axis.offset, axis.scale, lo, hi)
    (b0, b1), (s0, s1) = spans["background"], spans["signal"]
    b1 = min(b1, s0)  # windows meeting inside one channel don't share it
    need = min_background_channels(options)
    if b1 - b0 < need:
        what = MODEL_NAMES[options.model]
        raise ValueError(f"The background window needs at least {need} channels for a {what} fit")
    if s1 - s0 < 2:
        raise ValueError("The signal window needs at least 2 channels")
    return Windows(b0, b1, s0, s1)


# ---------- the estimates (HyperSpy's, for many spectra at once) ----------


def _power_law_estimate(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(A, r, ok) of A·x^-r by the two-area method on the spectra ``y`` (n, nb)
    at energies ``x`` (from the first channel; at least nb + 1 of them): the
    window is split in two halves, as in HyperSpy."""
    i1, i4 = 0, y.shape[1]
    if not (i4 + i1) % 2 == 0:
        i4 -= 1
    if i4 == i1:
        i4 += 2
    i3 = i2 = (i4 + i1) // 2
    x1, x2, x3, x4 = x[i1], x[i2], x[i3], x[i4]
    i_a, i_b = y[:, i1:i2].sum(axis=1), y[:, i3:i4].sum(axis=1)
    with np.errstate(all="ignore"):
        r = 2 * np.log(i_a / i_b * (x4 - x3) / (x2 - x1)) / np.log(x4 * x3 / x2 / x1)
        k = 1 - r
        a2 = k * i_b / (x4**k - x3**k)
        a1 = k * i_a / (x2**k - x1**k)
        amp = (a1 * i_a + a2 * i_b) / (i_a + i_b)
    ok = np.isfinite(r) & np.isfinite(amp) & (amp != 0)
    return np.where(ok, amp, 0.0), np.where(ok, r, np.nan), ok


def _exponential_estimate(y: np.ndarray, x: np.ndarray, axis: EnergyAxis) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(A, tau, ok) of A·exp(-x/tau) from the geometric means of the two
    halves of the window (A in HyperSpy's convention: per unit energy when the
    axis is binned). ``x`` as for the power law."""
    i1, i2 = 0, y.shape[1]
    i_mid = (i1 + i2) // 2
    x_start, x_mid, x_end = x[i1], x[i_mid], x[i2]
    with np.errstate(all="ignore"):
        first, second = y[:, i1:i_mid], y[:, i_mid:i2]
        positive = (first > 0).all(axis=1) & (second > 0).all(axis=1)
        log_g1, log_g2 = np.log(first).mean(axis=1), np.log(second).mean(axis=1)
        xa, xb = (x_start + x_mid) / 2, (x_mid + x_end) / 2
        amp = np.exp((log_g1 - (xa / xb) * log_g2) / (1 - xa / xb))
        tau = -xb / (log_g2 - np.log(amp))
        if axis.binned:
            amp = amp / axis.scale
    ok = positive & np.isfinite(amp) & np.isfinite(tau) & (amp != 0) & (tau != 0)
    return np.where(ok, amp, 0.0), np.where(ok, tau, 1.0), ok


def _log_line(u: np.ndarray, y: np.ndarray, use: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(intercept, slope, ok) of the least-squares line ln y = intercept +
    slope·u of every row of ``y`` (n, nb), over the channels ``use`` (n, nb)
    where y > 0 (at least 2 of them, at 2 different u)."""
    w = (use & (y > 0)).astype(float)
    with np.errstate(all="ignore"):
        v = np.where(w > 0, np.log(np.where(y > 0, y, 1.0)), 0.0)
        s0, su, suu = w.sum(1), w @ u, w @ (u * u)
        sv, suv = (w * v).sum(1), (w * v) @ u
        det = s0 * suu - su * su
        slope = (s0 * suv - su * sv) / det
        intercept = (sv - slope * su) / s0
    ok = (s0 >= 2) & (np.abs(det) > 1e-12 * np.maximum(1.0, s0 * suu)) & np.isfinite(slope) & np.isfinite(intercept)
    return np.where(ok, intercept, 0.0), np.where(ok, slope, 0.0), ok


def _power_law_log(y: np.ndarray, x: np.ndarray, axis: EnergyAxis) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(A, r, ok) of A·x^-r by a linear fit of ln y to ln x; ``x`` the
    window's energies (all must be positive)."""
    if not (x > 0).all():
        raise ValueError("A log fit of a power law needs energies above 0 in the background window")
    c, slope, ok = _log_line(np.log(x), y, np.ones(y.shape, bool))
    with np.errstate(all="ignore"):
        amp = np.exp(c) / (axis.scale if axis.binned else 1.0)
    ok &= np.isfinite(amp)
    return np.where(ok, amp, 0.0), np.where(ok, -slope, np.nan), ok


def _exponential_log(y: np.ndarray, x: np.ndarray, axis: EnergyAxis) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(A, tau, ok) of A·exp(-x/tau) by a linear fit of ln y to x."""
    c, slope, ok = _log_line(x, y, np.ones(y.shape, bool))
    with np.errstate(all="ignore"):
        amp = np.exp(c) / (axis.scale if axis.binned else 1.0)
        tau = -1.0 / slope
    ok &= np.isfinite(amp) & np.isfinite(tau) & (slope != 0)
    return np.where(ok, amp, 0.0), np.where(ok, tau, 1.0), ok


def lba_window(window: np.ndarray, fwhm: float) -> np.ndarray:
    """LBA of the background-window channels ``window`` (scan axes…, nb):
    each channel smoothed across the scan by a Gaussian of ``fwhm`` pixels,
    then each pixel's window scaled to its own mean (left as it was where
    the smoothed mean is not positive). float64, same shape."""
    from scipy.ndimage import gaussian_filter

    raw = np.nan_to_num(np.asarray(window, dtype=float))
    sigma = fwhm / (2 * math.sqrt(2 * math.log(2)))
    smooth = gaussian_filter(raw, sigma=(sigma,) * (raw.ndim - 1) + (0,), mode="nearest")
    with np.errstate(all="ignore"):
        k = raw.mean(-1, keepdims=True) / smooth.mean(-1, keepdims=True)
    good = np.isfinite(k) & (smooth.mean(-1, keepdims=True) > 0)
    return np.where(good, smooth * np.where(good, k, 1.0), raw)


def lba_margin(fwhm: float) -> int:
    """Pixels on each side that the LBA of a pixel depends on."""
    return int(4.0 * fwhm / (2 * math.sqrt(2 * math.log(2))) + 0.5) + 1


def _multifit(component, y: np.ndarray, x: np.ndarray, axis: EnergyAxis, start: dict[str, np.ndarray]) -> None:
    """Least-squares fit of ``component`` (a HyperSpy component) to every row
    of ``y`` (n, nb) from the ``start`` values of its parameters; the fitted
    values are left in the component's parameter maps."""
    import hyperspy.api as hs

    sig = hs.signals.Signal1D(np.ascontiguousarray(y, dtype=float))
    ax = sig.axes_manager.signal_axes[0]
    ax.offset, ax.scale, ax.is_binned = float(x[0]), axis.scale, axis.binned
    model = sig.create_model()
    model.append(component)
    for name, values in start.items():
        parameter = getattr(component, name)
        parameter.map["values"][:] = values
        parameter.map["is_set"][:] = True
    component.fetch_stored_values()
    model.multifit(show_progressbar=False, iterpath="serpentine")


def _power_law_curve(amp, r, x, scale) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.where(x > 0, amp[:, None] * np.power(x[None, :], -r[:, None]), 0.0) * scale


def _exponential_curve(amp, tau, x, scale) -> np.ndarray:
    with np.errstate(all="ignore"):
        return amp[:, None] * np.exp(-x[None, :] / tau[:, None]) * scale


def fit_block(
    block: np.ndarray, axis: EnergyAxis, win: Windows, options: BackgroundOptions
) -> tuple[np.ndarray, np.ndarray]:
    """The fitted background of every spectrum in ``block`` (n, win.span: the
    channels from the start of the background window to the end of the signal
    window). Returns (background, same shape; r (n,), the power-law exponent,
    NaN for other models and for pixels where the fit failed).

    A pixel whose estimate fails (e.g. no counts in the window) gets a zero
    background, as HyperSpy's does.
    """
    block = np.asarray(block, dtype=float)
    n, m = block.shape
    nb = win.b1 - win.b0
    x = axis.energies(win.b0, m)
    xb, yb = x[:nb], block[:, :nb]
    scale = axis.scale if axis.binned else 1.0
    r = np.full(n, np.nan)

    log = uses_log(options)
    if options.model == "power_law":
        amp, r, ok = _power_law_log(yb, xb, axis) if log else _power_law_estimate(yb, x)
        if not log and not options.fast and ok.any():
            from hyperspy._components.power_law import PowerLaw

            comp = PowerLaw()
            _multifit(comp, yb, xb, axis, {"A": amp, "r": np.where(ok, r, 0.0)})
            fa, fr = np.asarray(comp.A.map["values"], float), np.asarray(comp.r.map["values"], float)
            good = ok & np.isfinite(fa) & np.isfinite(fr)
            amp, r = np.where(good, fa, amp), np.where(good, fr, r)
        bg = _power_law_curve(amp, np.nan_to_num(r), x, scale)
    elif options.model == "exponential":
        amp, tau, ok = _exponential_log(yb, xb, axis) if log else _exponential_estimate(yb, x, axis)
        if not log and not options.fast and ok.any():
            from hyperspy._components.exponential import Exponential

            comp = Exponential()
            _multifit(comp, yb, xb, axis, {"A": amp, "tau": tau})
            fa, ft = np.asarray(comp.A.map["values"], float), np.asarray(comp.tau.map["values"], float)
            good = ok & np.isfinite(fa) & np.isfinite(ft) & (ft != 0)
            amp, tau = np.where(good, fa, amp), np.where(good, ft, tau)
        bg = _exponential_curve(amp, tau, x, scale)
    elif options.model == "polynomial":
        # Fitted on the window centred and scaled to [-1, 1]: the same
        # polynomial as on the raw energies, but well conditioned.
        centre, half = (xb[0] + xb[-1]) / 2, (xb[-1] - xb[0]) / 2
        poly = np.polynomial.polynomial
        coefficients = poly.polyfit((xb - centre) / half, yb.T, options.order)
        bg = poly.polyval((x - centre) / half, coefficients)
    else:  # offset
        bg = np.repeat(yb.mean(axis=1, keepdims=True), m, axis=1)
    return np.nan_to_num(bg, nan=0.0, posinf=0.0, neginf=0.0), r


def preview(
    spectra: np.ndarray, axis: EnergyAxis, options: BackgroundOptions
) -> tuple[int, np.ndarray, np.ndarray]:
    """The background of each spectrum (n, axis.size) over the channels from the
    start of the background window to the end of the signal window: (that first
    channel, backgrounds, r)."""
    win = channel_windows(axis, options)
    bg, r = fit_block(spectra[:, win.b0 : win.s1], axis, win, options)
    return win.b0, bg, r


def remove_background(
    signal, options: BackgroundOptions, report: Callable[[float], None] = lambda f: None
) -> tuple[object, np.ndarray | None]:
    """A copy of the EELS spectra ``signal`` (loaded in memory; any navigation)
    cut to the signal window, with the background subtracted. Also the
    power-law exponent r at every pixel (navigation shape; None for the other
    models). ``report`` gets the fraction done after each chunk."""
    axis = EnergyAxis.of(signal)
    win = channel_windows(axis, options)
    flat = np.asarray(signal.data).reshape(-1, axis.size)
    n = len(flat)
    keep = win.s1 - win.s0
    out = np.empty((n, keep), np.float32)
    r = np.full(n, np.nan, np.float32)
    nb = win.b1 - win.b0
    nav = tuple(signal.data.shape[:-1])
    averaged = None
    if options.lba > 0 and nav:
        averaged = lba_window(np.asarray(signal.data)[..., win.b0 : win.b1], options.lba).reshape(n, nb)
    fast = options.fast or uses_log(options) or options.model in ("polynomial", "offset")
    step = FAST_CHUNK if fast else FULL_CHUNK
    for start in range(0, n, step):
        stop = min(start + step, n)
        block = flat[start:stop, win.b0 : win.s1].astype(float)
        fitted = block
        if averaged is not None:
            fitted = block.copy()
            fitted[:, :nb] = averaged[start:stop]
        bg, r[start:stop] = fit_block(fitted, axis, win, options)
        out[start:stop] = (block - bg)[:, win.s0 - win.b0 :]
        report(stop / n)

    result = signal.isig[win.s0 : win.s1].deepcopy()
    result.data = out.reshape(*nav, keep)
    result.metadata.set_item("Processing.si_viewer.background_removal", describe(options, axis.units))
    return result, (r.reshape(nav) if options.model == "power_law" and nav else None)
