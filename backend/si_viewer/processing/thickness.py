"""Thickness of EELS low-loss spectrum images (M10d).

The log-ratio method (exSpy's ``estimate_thickness``): t/λ = ln(I_total / I_0),
with I_0 the intensity of the zero-loss peak, taken as the counts below a
threshold. The threshold is one energy for every pixel, or estimated at every
pixel (``estimate_elastic_scattering_threshold``: the first inflexion of the
spectrum's derivative after the peak). Run on chunks of pixels, so that a
job can report progress and be cancelled.
"""

from __future__ import annotations

import contextlib
import logging
import math
from typing import Callable, Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

from .background import EnergyAxis
from .eels import _number
from .filters import Filters, filter_image, filter_spectrum
from .regions import window_label

# Where exSpy's search for the end of the zero-loss peak starts (energy loss),
# and the width of the smoothing it does before differentiating.
SEARCH_START = 1.0
SAVGOL_WINDOW = 5
SAVGOL_ORDER = 3
# exSpy refuses a search window narrower than this many channels.
MIN_SEARCH_CHANNELS = 10

# Spectra per step.
CHUNK = 2048

RELATIVE_TITLE = "t/λ"
ABSOLUTE_TITLE = "Thickness (nm)"
TITLES = (RELATIVE_TITLE, ABSOLUTE_TITLE)

Report = Callable[..., None]


class ThicknessOptions(BaseModel):
    # "auto": estimated at every pixel; else one energy for all (the energy
    # axis' units).
    threshold: Literal["auto"] | float = "auto"
    # Auto only: how far from the peak the search for its end goes (energy units).
    window: float = Field(10.0, gt=0, le=1000)
    # t/λ, or thickness in nm from a mean free path or a density.
    output: Literal["relative", "absolute"] = "relative"
    mean_free_path: float | None = Field(None, gt=0)  # nm
    density: float | None = Field(None, gt=0)  # g/cm³

    @field_validator("threshold")
    @classmethod
    def _finite(cls, v):
        if v != "auto" and not math.isfinite(v):
            raise ValueError("The threshold must be a number")
        return v

    @field_validator("window", "mean_free_path", "density")
    @classmethod
    def _numbers(cls, v):
        if v is not None and not math.isfinite(v):
            raise ValueError("Enter a number")
        return v

    @model_validator(mode="after")
    def _output(self) -> "ThicknessOptions":
        if self.output == "relative":
            self.mean_free_path = self.density = None
        elif (self.mean_free_path is None) == (self.density is None):
            raise ValueError("Absolute thickness needs a mean free path or a density, not both or neither")
        return self

    @property
    def auto(self) -> bool:
        return self.threshold == "auto"

    @property
    def title(self) -> str:
        return RELATIVE_TITLE if self.output == "relative" else ABSOLUTE_TITLE


def microscope_parameters(signal) -> dict[str, float | None]:
    """The beam energy (keV) and the convergence and collection angles (mrad)
    in the metadata (None where missing)."""
    return {
        "beam_energy": _number(signal, "Acquisition_instrument.TEM.beam_energy"),
        "convergence_angle": _number(signal, "Acquisition_instrument.TEM.convergence_angle"),
        "collection_angle": _number(signal, "Acquisition_instrument.TEM.Detector.EELS.collection_angle"),
    }


def missing_parameters(signal) -> list[str]:
    names = {"beam_energy": "beam energy", "convergence_angle": "convergence angle", "collection_angle": "collection angle"}
    return [names[k] for k, v in microscope_parameters(signal).items() if v is None]


def check_options(signal, options: ThicknessOptions) -> None:
    """Raise ValueError (in plain English) if ``options`` can't be used on the
    low-loss ``signal``."""
    axis = EnergyAxis.of(signal)
    if not axis.scale > 0 or axis.size < 2:
        raise ValueError("Thickness needs an energy axis that increases")
    last = axis.offset + axis.scale * (axis.size - 1)
    available = window_label(axis.offset, last, axis.units)
    eps = 1e-9 * max(1.0, abs(axis.offset), abs(last))
    if options.auto:
        lo, hi = SEARCH_START, SEARCH_START + options.window
        if lo < axis.offset - eps or hi > last + eps:
            raise ValueError(
                f"The automatic threshold searches {window_label(lo, hi, axis.units)}, "
                f"outside the spectrum's energy range ({available})"
            )
        if options.window / axis.scale < MIN_SEARCH_CHANNELS:
            least = math.ceil(MIN_SEARCH_CHANNELS * axis.scale * 10 - 1e-6) / 10
            raise ValueError(f"The search window must be at least {least:g} {axis.units}".rstrip())
    else:
        t = options.threshold
        if not axis.offset - eps <= t <= last + eps:
            raise ValueError(f"The threshold must lie inside the spectrum's energy range ({available})")
        if round((t - axis.offset) / axis.scale) < 1:
            raise ValueError("The threshold must be above the first channel")
    if options.density is not None:
        missing = missing_parameters(signal)
        if missing:
            raise ValueError(f"A density needs the {', '.join(missing)}")


def describe(options: ThicknessOptions, units: str = "") -> str:
    """e.g. "threshold auto (10 eV), λ 120 nm"."""
    u = f" {units}" if units else ""
    threshold = f"auto ({options.window:g}{u})" if options.auto else f"{options.threshold:g}{u}"
    out = f"threshold {threshold}"
    if options.mean_free_path is not None:
        out += f", mean free path {options.mean_free_path:g} nm"
    if options.density is not None:
        out += f", density {options.density:g} g/cm³"
    return out


@contextlib.contextmanager
def quiet():
    """No progress bars, and no log lines about the missing parameters, from
    HyperSpy and exSpy while it runs."""
    import hyperspy.api as hs

    loggers = [logging.getLogger("exspy"), logging.getLogger("hyperspy")]
    levels = [lg.level for lg in loggers]
    bar = hs.preferences.General.show_progressbar
    for lg in loggers:
        lg.setLevel(logging.ERROR)
    hs.preferences.General.show_progressbar = False
    try:
        yield
    finally:
        hs.preferences.General.show_progressbar = bar
        for lg, level in zip(loggers, levels):
            lg.setLevel(level)


def _spectra(src, data: np.ndarray):
    """An exSpy EELS signal of ``data`` (n, channels) on ``src``'s energy axis
    and with its metadata (so the microscope parameters follow); the n spectra
    are one navigation axis."""
    import hyperspy.api as hs

    # Floating-point counts keep their precision (exSpy's results then match
    # a direct call on the signal); integers are converted.
    data = np.ascontiguousarray(data, dtype=data.dtype if data.dtype.kind == "f" else float)
    s = hs.signals.Signal1D(data)
    s.metadata.add_dictionary(src.metadata.as_dictionary())
    s.set_signal_type("EELS")
    ax, a = s.axes_manager.signal_axes[0], src.axes_manager.signal_axes[0]
    ax.offset, ax.scale, ax.name = float(a.offset), float(a.scale), a.name
    if isinstance(a.units, str) and a.units != "None":
        ax.units = a.units
    # The counts are summed: the ratio of two sums is the ratio of the two
    # integrals whether the axis is binned or not (exSpy's integration of an
    # unbinned axis mixes sums and integrals).
    ax.is_binned = True
    return s


def _tolerance(spectra, window: float) -> float:
    """The largest, over the spectra, of the smallest |derivative| in the
    search window: exSpy's automatic ``tol`` (it needs every spectrum, so a
    chunked run finds it first)."""
    axis = spectra.axes_manager.signal_axes[0]
    first, last = axis.value_range_to_indices(SEARCH_START, SEARCH_START + window)
    s = spectra.isig[first:last].deepcopy()
    s.smooth_savitzky_golay(polynomial_order=SAVGOL_ORDER, window_length=SAVGOL_WINDOW, differential_order=1)
    return float(np.max(np.abs(s.data).min(axis=-1)))


def estimate_threshold(spectrum: np.ndarray, src, window: float) -> float | None:
    """exSpy's elastic-scattering threshold (in the energy axis' units) of one
    spectrum on ``src``'s energy axis; None if there is no inflexion."""
    check = ThicknessOptions(window=window)
    check_options(src, check)
    with quiet():
        s = _spectra(src, np.asarray(spectrum, dtype=float)[None, :])
        value = float(np.ravel(s.estimate_elastic_scattering_threshold(window=window).data)[0])
    return value if math.isfinite(value) else None


def _chunk_values(spectra, options: ThicknessOptions, tol: float | None) -> np.ndarray:
    """t/λ (or the thickness in nm) of every spectrum, from exSpy."""
    if options.auto:
        threshold = spectra.estimate_elastic_scattering_threshold(window=options.window, tol=tol)
    else:
        threshold = float(options.threshold)
    with np.errstate(all="ignore"):
        t = spectra.estimate_thickness(
            threshold=threshold, density=options.density, mean_free_path=options.mean_free_path
        )
    out = np.asarray(t.data, dtype=float).ravel()
    out[~np.isfinite(out)] = np.nan
    return out


def thickness_map(
    signal, options: ThicknessOptions, filters: Filters | None = None, report: Report = lambda f, m=None: None
) -> np.ndarray:
    """The thickness of every pixel of the EELS low loss ``signal`` (in memory;
    one or two navigation axes) as a float32 array of its navigation shape,
    NaN where there is none (no counts, or no threshold could be estimated).

    The processed data is used: the spectra are smoothed spectrally by
    ``filters`` first and the map is blurred spatially (a 2D scan only).
    ``report(fraction, message)`` is called after each chunk.
    """
    filters = filters or Filters()
    check_options(signal, options)
    nav = tuple(signal.data.shape[:-1])
    flat = np.asarray(signal.data).reshape(-1, signal.data.shape[-1])
    n = len(flat)

    def chunk(start: int):
        data = np.nan_to_num(np.asarray(flat[start : start + CHUNK]))
        return _spectra(signal, filter_spectrum(data, filters))

    starts = range(0, n, CHUNK)
    tol = None
    share = 0.0
    with quiet():
        if options.auto:
            share = 0.3
            tolerances = []
            for i, start in enumerate(starts):
                tolerances.append(_tolerance(chunk(start), options.window))
                report(share * (i + 1) / len(starts), "Finding the threshold")
            tol = max(tolerances)
        out = np.empty(n, np.float32)
        for i, start in enumerate(starts):
            out[start : start + CHUNK] = _chunk_values(chunk(start), options, tol)
            report(share + (1 - share) * (i + 1) / len(starts), "Estimating the thickness")
    result = out.reshape(nav)
    if filters.has_spatial and len(nav) == 2:
        result = filter_image(result, filters)
    return result


def result_signal(si_signal, data: np.ndarray, options: ThicknessOptions):
    """The thickness ``data`` as a dataset on the spectrum image's scan: an
    image for a 2D scan, a profile along the scan for a line scan."""
    import hyperspy.api as hs

    from .element_maps import scan_image

    nav = si_signal.axes_manager.navigation_axes
    if len(nav) == 2:
        out = scan_image(si_signal, data, options.title)
    else:
        axis = nav[0].get_axis_dictionary()
        axis["navigate"] = False
        out = hs.signals.Signal1D(np.asarray(data, np.float32), axes=[axis])
        out.metadata.General.title = options.title
    out.metadata.set_item("Signal.quantity", options.title)
    return out
