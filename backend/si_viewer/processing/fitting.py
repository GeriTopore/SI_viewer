"""Fitting one EDS spectrum with exSpy's model (M4, phase 1).

Recipe (tested on exSpy's FePt nanoparticle spectrum): ignore the noise peak
below ``min_energy``, fit the polynomial background first, then a bounded fit
(no negative intensities), then calibrate the detector resolution and the
energies of the fitted α lines (Kα/Lα/Mα) and refit. Calibrating every minor
line too is ~25× slower for a spectrum image's sum, with near-identical
intensities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from pydantic import BaseModel, Field

from .filters import filter_spectrum
from .regions import Region, processed_pixel_spectrum, processed_region_spectrum


class FitOptions(BaseModel):
    background_order: int = Field(6, ge=1, le=12)
    # Channels below this energy (keV) are ignored: the zero/noise peak.
    min_energy: float = Field(0.5, ge=0)
    calibrate_resolution: bool = True
    calibrate_energy: bool = True


class FitSource(BaseModel):
    """Which spectrum of a spectrum image to fit."""

    type: Literal["all", "pixel", "region"] = "all"
    x: int = 0
    y: int = 0
    size: int = Field(1, ge=1, le=64)
    region: Region | None = None


def source_spectrum(dataset, source: FitSource):
    """The 1D EDS signal to fit (summed counts of the processed data: the
    dataset's filters applied) and how many pixels it sums."""
    from ..session import DatasetKind, si_cube

    f = dataset.filter_spec
    if dataset.kind is DatasetKind.SPECTRUM:
        dataset.materialise()
        spec = dataset.signal.deepcopy()
        spec.change_dtype("float64")
        spec.data[:] = filter_spectrum(spec.data, f)
        return spec, 1
    if dataset.kind is not DatasetKind.SPECTRUM_IMAGE:
        raise ValueError(f"{dataset.title} is not a spectrum or spectrum image")

    dataset.materialise()
    data = si_cube(dataset)
    if source.type == "pixel":
        counts, n = processed_pixel_spectrum(data, source.x, source.y, source.size, f)
    elif source.type == "region":
        if source.region is None:
            raise ValueError("A region source needs a region")
        counts, n = processed_region_spectrum(data, source.region, f)
        if n == 0:
            raise ValueError("The region contains no pixels")
    else:
        # A spatial filter keeps the total counts (reflecting edges).
        counts = filter_spectrum(data.sum(axis=(0, 1), dtype=np.float64), f)
        n = data.shape[0] * data.shape[1]
    # A 1D signal with the spectrum image's metadata (elements, detector, …).
    spec = dataset.signal.inav[(0,) * dataset.signal.axes_manager.navigation_dimension].deepcopy()
    spec.change_dtype("float64")
    spec.data[:] = counts
    return spec, n


def analysis_lines(spec) -> dict[str, str]:
    """exSpy's choice of line per element (as used for maps and markers)."""
    elements = list(spec.metadata.get_item("Sample.elements", []))
    out = {}
    for el in elements:
        lines = spec._get_lines_from_elements([el], only_one=True, only_lines=("a",))
        if lines:
            out[el] = lines[0]
    return out


@dataclass
class FittedModel:
    model: object  # exSpy EDSModel, fitted
    spectrum: object  # the (cropped) spectrum it was fitted to
    elements: list[str]
    lines: list[str]  # all α lines in the fitted range
    chosen: dict[str, str]  # analysis line per element


def fit_model(spec, options: FitOptions, report: Callable[[float, str], None] | None = None) -> FittedModel:
    """Fit exSpy's EDS model to ``spec`` (a 1D EDS signal with elements set).
    ``report(fraction, step)`` is called before each step (it may raise to stop)."""
    report = report or (lambda f, step: None)
    elements = list(spec.metadata.get_item("Sample.elements", []))
    if not elements:
        raise ValueError("Add elements in the periodic table before fitting")

    axis = spec.axes_manager.signal_axes[0]
    lo = max(options.min_energy, axis.low_value)
    if lo >= axis.high_value:
        raise ValueError("Nothing left to fit above the minimum energy")
    cropped = spec.isig[lo:].deepcopy()
    # All α lines of the elements inside the fitted range.
    cropped.set_lines([], only_one=False)
    if "Sample.xray_lines" in cropped.metadata:
        del cropped.metadata.Sample.xray_lines
    cropped.add_lines(only_one=False)
    lines = list(cropped.metadata.Sample.xray_lines)
    chosen = analysis_lines(cropped)

    report(0.05, "building the model")
    m = cropped.create_model(auto_background=False)
    m.add_polynomial_background(order=options.background_order)
    report(0.15, "fitting the background")
    m.fit_background()
    report(0.2, "fitting the lines")
    m.fit(bounded=True)
    if options.calibrate_resolution:
        report(0.3, "calibrating the resolution")
        m.calibrate_energy_axis("resolution")
    if options.calibrate_energy and chosen:
        report(0.7, "calibrating line energies")
        m.calibrate_xray_lines("energy", xray_lines=lines, bound=10)
    if options.calibrate_resolution or options.calibrate_energy:
        report(0.9, "refitting")
        m.fit(bounded=True)
    return FittedModel(m, cropped, elements, lines, chosen)


def fit_spectrum(spec, options: FitOptions) -> dict:
    """Fit ``spec`` (a 1D EDS signal with elements set) and return curves,
    line intensities and calibration results."""
    fitted = fit_model(spec, options)
    m, cropped, elements, lines, chosen = (
        fitted.model, fitted.spectrum, fitted.elements, fitted.lines, fitted.chosen,
    )
    x = cropped.axes_manager.signal_axes[0].axis
    data = cropped.data
    model = m.as_signal(show_progressbar=False).data
    background_parts = [c for c in m if c.name.startswith("background")]
    background = m.as_signal(component_list=background_parts, show_progressbar=False).data
    per_element = {}
    for el in elements:
        parts = [c for c in m if c.name.split("_")[0] == el]
        if parts:
            per_element[el] = m.as_signal(component_list=parts, show_progressbar=False).data

    intensities = m.get_lines_intensity(xray_lines=lines, plot_result=False)
    line_rows = []
    for line, sig in zip(lines, intensities):
        el = line.split("_")[0]
        line_rows.append({
            "line": line,
            "element": el,
            "energy": float(cropped._get_line_energy(line)),
            "intensity": float(np.asarray(sig.data).ravel()[0]),
            "analysis": chosen.get(el) == line,
        })

    resid = data - model
    resolution = cropped.metadata.get_item("Acquisition_instrument.TEM.Detector.EDS.energy_resolution_MnKa")
    return {
        "x": x.tolist(),
        "data": data.tolist(),
        "model": model.tolist(),
        "background": background.tolist(),
        "elements": {el: curve.tolist() for el, curve in per_element.items()},
        "lines": line_rows,
        "resolution_MnKa": float(resolution) if resolution is not None else None,
        # Residual size relative to the signal (0 = perfect).
        "residual_ratio": float(resid.std() / data.std()) if data.std() else None,
        # Elements with no α line in the fitted range (e.g. C below 0.5 keV).
        "skipped": [el for el in elements if el not in chosen],
    }
