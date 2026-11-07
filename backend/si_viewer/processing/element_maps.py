"""Element maps computed from an EDS spectrum image.

A map is the net intensity of an element's analysis line at every pixel:
counts in an integration window around the line, minus the background from
windows on either side, exactly as exSpy's ``get_lines_intensity`` with
``background_windows`` but computed directly with NumPy (~30× faster).
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

MIN_LINE_ENERGY = 0.1  # keV: lines below this sit in the noise peak

_GREEK = {"a": "α", "b": "β", "g": "γ", "z": "ζ", "n": "η", "l": "ℓ"}


class MapOptions(BaseModel):
    """How element maps of a spectrum image are computed."""

    background: bool = True
    # Integration window width in units of the line's FWHM (exSpy's default: 2).
    width: float = Field(2.0, gt=0, le=10)
    # n × n box smoothing of the map (1 = none); keeps the map size.
    smoothing: int = Field(1, ge=1, le=15)


def line_label(line: str) -> str:
    """"Fe_Ka" → "Fe Kα"."""
    element, family = line.split("_")
    return f"{element} {family[0]}{_GREEK.get(family[1:2], family[1:2])}{family[2:]}"


def analysis_line(signal, element: str) -> str | None:
    """exSpy's choice of line for ``element`` (e.g. "Fe_Ka"), or None if it has
    no usable line inside the spectrum's energy range."""
    lines = signal._get_lines_from_elements([element], only_one=True, only_lines=("a",))
    if not lines:
        return None
    axis = signal.axes_manager.signal_axes[0]
    energy = signal._get_line_energy(lines[0])
    if energy < max(MIN_LINE_ENERGY, axis.low_value) or energy > axis.high_value:
        return None
    return lines[0]


def line_windows(signal, line: str, options: MapOptions) -> tuple[list[float], list[float] | None]:
    """Integration window [lo, hi] and background windows [b0, b1, b2, b3] (keV)."""
    integration = list(signal.estimate_integration_windows(windows_width=options.width, xray_lines=[line])[0])
    background = None
    if options.background:
        background = [float(v) for v in signal.estimate_background_windows(xray_lines=[line])[0]]
    return [float(v) for v in integration], background


def line_intensity(signal, integration: list[float], background: list[float] | None) -> np.ndarray:
    """Net counts per pixel, with exSpy's channel conventions (value2index)."""
    axis = signal.axes_manager.signal_axes[0]
    data = signal.data
    idx = lambda e: int(axis.value2index(min(max(e, axis.low_value), axis.high_value)))  # noqa: E731

    def window_sum(lo: float, hi: float) -> tuple[np.ndarray, int]:
        i0, i1 = idx(lo), idx(hi)
        if i0 == i1:  # zero-width window: the single channel (as exSpy)
            return data[..., i0].astype(np.float32), 0
        return data[..., i0:i1].sum(axis=-1, dtype=np.float32), i1 - i0

    counts, n = window_sum(*integration)
    result = counts
    if background is not None:
        b1, n1 = window_sum(background[0], background[1])
        b2, n2 = window_sum(background[2], background[3])
        if n1 + n2:
            result = counts - (b1 + b2) * (n / (n1 + n2))
    # Lazy (dask) spectrum images: evaluate now.
    return np.asarray(result.compute() if hasattr(result, "compute") else result)


def scan_image(si_signal, data: np.ndarray, title: str):
    """A Signal2D on the spectrum image's scan grid (same calibration)."""
    import hyperspy.api as hs

    axes = []
    for ax in si_signal.axes_manager.navigation_axes[::-1]:  # array order y, x
        d = ax.get_axis_dictionary()
        d["navigate"] = False
        axes.append(d)
    out = hs.signals.Signal2D(np.asarray(data, np.float32), axes=axes)
    out.metadata.General.title = title
    return out


def compute_map(signal, element: str, options: MapOptions):
    """The element's map as a HyperSpy Signal2D (None if it has no usable line)."""
    import hyperspy.api as hs
    from scipy.ndimage import uniform_filter

    line = analysis_line(signal, element)
    if line is None:
        return None
    integration, background = line_windows(signal, line, options)
    data = line_intensity(signal, integration, background)
    if options.smoothing > 1:
        data = uniform_filter(data, size=options.smoothing, mode="nearest")

    out = scan_image(signal, data, line_label(line))
    out.metadata.set_item(
        "Sample.xray_line_map",
        {"line": line, "integration_window": integration, "background_windows": background or []},
    )
    return out
