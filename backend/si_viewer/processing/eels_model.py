"""Fitting EELS core-loss edges with computed cross-sections (M13).

exSpy's ``EELSModel``: a power-law background plus one ``EELSCLEdge`` per
subshell of each element inside the fitted range, shaped by its cross-section
(a GOS table, the beam energy and the convergence and collection angles). An
element's subshells are tied to one of them (the *master* edge: L2 and L1
follow L3), so each element has one intensity. With a low loss the model is
convolved with it. The fit is exSpy's ``smart_fit``: the background, then the
edges in order of energy.

The intensities are divided by the cross-sections already, so their ratios
are atomic ratios.
"""

from __future__ import annotations

import logging
import math
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator

GOS_NAMES = {"dft": "DFT", "dirac": "Dirac", "hydrogenic": "hydrogenic"}

# Pixels fitted between two progress reports (about 10 ms each; 30 ms convolved).
MAP_CHUNK = 64

_PARAMETERS = (
    ("beam energy", "Acquisition_instrument.TEM.beam_energy"),
    ("convergence angle", "Acquisition_instrument.TEM.convergence_angle"),
    ("collection angle", "Acquisition_instrument.TEM.Detector.EELS.collection_angle"),
)


class EelsFitOptions(BaseModel):
    # The cross-sections: exSpy's GOS tables ("dft": Segger, Guzzinati & Kohl;
    # "dirac") or the hydrogenic model (K and L edges only).
    gos: Literal["dft", "dirac", "hydrogenic"] = "dft"
    # Fit the fine structure after each onset with exSpy's spline, over
    # ``fine_structure_width`` (eV; narrowed where edges overlap).
    fine_structure: bool = False
    fine_structure_width: float = Field(30.0, gt=0, le=500)
    # Element → the onset (eV) of its master edge, where the database's is off.
    onsets: dict[str, float] = {}
    # The fitted range (eV); None: to the end of the spectrum.
    lo: float | None = None
    hi: float | None = None
    # Convolve the model with the low loss, when there is one.
    convolve: bool = True
    # The background window (eV): the background is fitted there alone (the
    # edges off), then fixed while the edges are fitted over the whole range.
    # None: exSpy's smart fit (the background up to the first edge, refitted
    # with each edge).
    bg_lo: float | None = None
    bg_hi: float | None = None

    @field_validator("onsets")
    @classmethod
    def _finite(cls, onsets: dict[str, float]) -> dict[str, float]:
        if not all(math.isfinite(v) and v > 0 for v in onsets.values()):
            raise ValueError("An onset must be a positive energy")
        return onsets


# ---------- elements and their edges ----------


def element_edges(symbol: str) -> list[dict]:
    """The ionisation edges of an element in exSpy's database, by onset:
    ``{edge: "L3", onset (eV), major}``."""
    from exspy.material import elements as db

    props = db[symbol].Atomic_properties
    if "Binding_energies" not in props:
        return []
    edges = [
        {"edge": name, "onset": float(e["onset_energy (eV)"]), "major": e["relevance"] == "Major"}
        for name, e in props.Binding_energies.as_dictionary().items()
        if name[-1] != "a"  # as exSpy: "K1a" and the like are not subshells
    ]
    return sorted(edges, key=lambda e: (e["onset"], e["edge"]))


def edges_info(signal) -> dict:
    """The ``edges`` part of an EELS dataset's summary: its elements and their
    subshells inside the energy range (the ones a model would get), by onset."""
    elements = sorted(signal.metadata.get_item("Sample.elements", []) or [])
    axis = signal.axes_manager.signal_axes[0]
    lo, hi = sorted((float(axis.low_value), float(axis.high_value)))
    subshells = [
        {"name": f"{el}_{e['edge']}", "element": el, **e}
        for el in elements
        for e in element_edges(el)
        if lo <= e["onset"] <= hi
    ]
    return {"elements": elements, "subshells": sorted(subshells, key=lambda s: s["onset"])}


def set_elements(signal, elements: list[str]) -> None:
    """Replace an EELS signal's elements (exSpy's ``add_elements`` only adds)."""
    from exspy.material import elements as db

    unknown = [e for e in elements if e not in db.keys()]
    if unknown:
        raise ValueError(f"Unknown element(s): {', '.join(unknown)}")
    signal.elements, signal.subshells = set(), set()
    if elements:
        signal.add_elements(tuple(sorted(set(elements))))
    elif signal.metadata.get_item("Sample.elements") is not None:
        del signal.metadata.Sample.elements


def missing_parameters(signal) -> list[str]:
    """The microscope parameters the cross-sections need and the metadata lacks."""
    missing = []
    for name, path in _PARAMETERS:
        value = signal.metadata.get_item(path, None)
        try:
            ok = math.isfinite(float(value)) and float(value) > 0
        except (TypeError, ValueError):
            ok = False
        if not ok:
            missing.append(name)
    return missing


# ---------- the model ----------


@contextmanager
def _quiet():
    """exSpy and HyperSpy log a line or two per edge: not while this runs."""
    loggers = [logging.getLogger(name) for name in ("exspy", "hyperspy")]
    levels = [log.level for log in loggers]
    for log in loggers:
        log.setLevel(logging.ERROR)
    try:
        yield
    finally:
        for log, level in zip(loggers, levels):
            log.setLevel(level)


@dataclass
class EelsModel:
    model: object  # exSpy EELSModel
    signal: object  # the spectra it fits (cropped to the fitted range, float)
    masters: dict[str, object]  # element → its master edge (the one with the intensity)
    skipped: list[str]  # elements with no edge in the fitted range
    convolved: bool


def _crop(signal, options: EelsFitOptions):
    axis = signal.axes_manager.signal_axes[0]
    lo = axis.low_value if options.lo is None else max(float(options.lo), axis.low_value)
    hi = axis.high_value if options.hi is None else min(float(options.hi), axis.high_value)
    if not lo < hi:
        raise ValueError("The fit range is outside the spectrum's energy range")
    i0, i1 = int(axis.value2index(lo)), int(axis.value2index(hi)) + 1
    if i1 - i0 < 8:
        raise ValueError("The fit range is too narrow")
    return signal.isig[i0:i1]


def build_model(signal, options: EelsFitOptions, low_loss=None) -> EelsModel:
    """The model of the EELS spectra ``signal`` (any navigation; elements in
    its metadata), not yet fitted. ``low_loss``: spectra of the same
    navigation shape to convolve it with. Raises ValueError (in plain English)
    for what the user can put right."""
    elements = sorted(signal.metadata.get_item("Sample.elements", []) or [])
    if not elements:
        raise ValueError("Add elements in the periodic table before fitting")
    missing = missing_parameters(signal)
    if missing:
        raise ValueError(f"Set the {', '.join(missing)} in the Info tab")

    s = _crop(signal, options).deepcopy()
    s.change_dtype("float64" if s.axes_manager.navigation_size <= 1 else "float32")
    set_elements(s, elements)  # the subshells inside the fitted range
    if not s.subshells:
        raise ValueError("None of the elements has an edge in the fitted range")

    if low_loss is not None and not options.convolve:
        low_loss = None
    if low_loss is not None:
        a, b = s.axes_manager.signal_axes[0], low_loss.axes_manager.signal_axes[0]
        if not math.isclose(float(a.scale), float(b.scale), rel_tol=1e-4):
            raise ValueError("The low loss needs the core loss' energy step to be convolved with it")

    try:
        with _quiet():  # one line per edge about its GOS
            m = s.create_model(low_loss=low_loss, GOS=options.gos)
    except Exception as e:  # e.g. no hydrogenic GOS for an M edge
        if options.gos == "hydrogenic":
            raise ValueError("Hydrogenic cross-sections cover K and L edges only") from e
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"No {GOS_NAMES[options.gos]} cross-section for one of the edges: {e}") from e

    masters = {}
    for c in m:
        if hasattr(c, "onset_energy") and c.intensity.twin is None:
            masters[c.element] = c
    for el, onset in options.onsets.items():
        if el in masters:
            m.set_parameters_value("onset_energy", float(onset), component_list=[masters[el]], only_current=False)
    if options.fine_structure:
        m.enable_fine_structure()
        for c in m:
            if hasattr(c, "fine_structure_width"):
                c.fine_structure_width = options.fine_structure_width
    return EelsModel(m, s, masters, [el for el in elements if el not in masters], low_loss is not None)


def run_fit(built: EelsModel, options: EelsFitOptions, mask: np.ndarray | None = None) -> None:
    """Fit the model (``mask``: every pixel it leaves False, with
    ``multifit``; None: the current one). Without a background window this is
    exSpy's smart fit; with one, the background components are fitted over
    the window alone with the edges switched off, then fixed while the edges
    are fitted over the whole range (and freed again afterwards)."""
    m = built.model

    def fit():
        if mask is None:
            m.fit()
        else:
            m.multifit(mask=mask, show_progressbar=False)

    if options.bg_lo is None or options.bg_hi is None:
        if mask is None:
            m.smart_fit()
        else:
            m.multifit(kind="smart", mask=mask, show_progressbar=False)
        return
    lo, hi = sorted((float(options.bg_lo), float(options.bg_hi)))
    axis = built.signal.axes_manager.signal_axes[0]
    if lo < axis.low_value - axis.scale / 2 or hi > axis.high_value + axis.scale / 2:
        raise ValueError("The background window is outside the fitted range")
    if int(axis.value2index(min(hi, axis.high_value))) - int(axis.value2index(max(lo, axis.low_value))) < 3:
        raise ValueError("The background window is too narrow")
    edges = [c for c in m if hasattr(c, "onset_energy") and c.active]
    background = [c for c in m if not hasattr(c, "onset_energy")]
    m.disable_edges(edges)
    try:
        m.set_signal_range(max(lo, axis.low_value), min(hi, axis.high_value))
        fit()
    finally:
        m.reset_signal_range()
        m.enable_edges(edges)
    was = [(p, p.free) for c in background for p in c.parameters]
    for p, _ in was:
        p.free = False
    try:
        fit()
    finally:
        for p, free in was:
            p.free = free


def _component_curve(m, components) -> np.ndarray:
    with _quiet():  # "Using slow as_signal implementation"
        return np.asarray(m.as_signal(component_list=components, show_progressbar=False).data, float).ravel()


def fit_spectrum(spec, options: EelsFitOptions, low_loss=None) -> dict:
    """Fit one EELS spectrum ``spec`` (1D, elements in its metadata; convolved
    with the 1D ``low_loss`` if given) and return the curves over the fitted
    range and each element's intensity."""
    built = build_model(spec, options, low_loss)
    m, s = built.model, built.signal
    try:
        run_fit(built, options)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"The fit failed: {e}") from e

    x = s.axes_manager.signal_axes[0].axis
    data = np.asarray(s.data, float).ravel()
    background = _component_curve(m, [c for c in m if not hasattr(c, "onset_energy")])
    curves = {
        el: _component_curve(m, [c for c in m if hasattr(c, "onset_energy") and c.element == el])
        for el in built.masters
    }
    # The parts added up: ``as_signal()`` leaves out whatever the smart fit
    # left switched off.
    model = background + sum(curves.values())

    total = sum(max(float(c.intensity.value), 0.0) for c in built.masters.values())
    rows = []
    for el, c in built.masters.items():
        value, std = float(c.intensity.value), c.intensity.std
        rows.append({
            "element": el,
            "edge": c.name,
            "onset": float(c.onset_energy.value),
            "intensity": value,
            "std": float(std) if std is not None and math.isfinite(std) else None,
            "atomic_percent": 100 * max(value, 0.0) / total if total > 0 else None,
            "fine_structure_width": float(c.fine_structure_width) if options.fine_structure else None,
        })
    rows.sort(key=lambda r: r["onset"])
    resid = data - model
    return {
        "x": x.tolist(),
        "data": data.tolist(),
        "model": np.nan_to_num(model).tolist(),
        "background": np.nan_to_num(background).tolist(),
        "elements": {el: np.nan_to_num(curve).tolist() for el, curve in curves.items()},
        "rows": rows,
        "background_r": next((float(c.r.value) for c in m if hasattr(c, "r")), None),
        # Residual size relative to the signal (0 = perfect).
        "residual_ratio": float(resid.std() / data.std()) if data.std() else None,
        "convolved": built.convolved,
        "gos": options.gos,
        "skipped": built.skipped,
    }


@dataclass
class PixelFits:
    """The fit of every pixel of a spectrum image (``fit_maps``), kept to show
    the fit at any pixel (``pixel_fit``)."""

    built: EelsModel
    failed: np.ndarray  # (height, width): pixels whose fit failed
    options: EelsFitOptions
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


def fit_maps(
    signal, options: EelsFitOptions, low_loss=None, report: Callable[[float], None] = lambda f: None,
) -> tuple[dict[str, np.ndarray], list[str], PixelFits]:
    """Fit every spectrum of the spectrum image ``signal`` (``run_fit`` at
    each pixel: exSpy's smart fit, or the background window then the edges) and return each element's intensity at every pixel
    (navigation shape; NaN where the fit failed), the elements skipped and
    the fits themselves. ``report`` gets the fraction done after each chunk
    (and may raise to stop)."""
    built = build_model(signal, options, low_loss)
    m = built.model
    nav = tuple(signal.data.shape[:-1])
    n = int(np.prod(nav))
    failed = np.zeros(n, bool)
    for start in range(0, n, MAP_CHUNK):
        stop = min(start + MAP_CHUNK, n)
        mask = np.ones(n, bool)
        mask[start:stop] = False
        try:
            run_fit(built, options, mask.reshape(nav))
        except ValueError as e:
            if "background window" in str(e):
                raise
            failed_chunk = True
        except Exception:
            failed_chunk = True
        else:
            failed_chunk = False
        if failed_chunk:
            # One spectrum that can't be fitted (e.g. no counts) stops exSpy's
            # loop: fit the chunk's pixels one by one and leave that one out.
            for i in range(start, stop):
                one = np.ones(n, bool)
                one[i] = False
                try:
                    run_fit(built, options, one.reshape(nav))
                except Exception:
                    failed[i] = True
        report(stop / n)

    maps = {}
    for el, c in built.masters.items():
        values = np.array(c.intensity.map["values"], np.float32).reshape(n)
        values[failed | ~np.isfinite(values)] = np.nan
        maps[el] = values.reshape(nav)
    return maps, built.skipped, PixelFits(built, failed.reshape(nav), options)


def pixel_fit(fits: PixelFits, x: int, y: int, size: int = 1) -> dict:
    """The fit of every pixel at the ``size`` × ``size`` block at (x, y) (as
    the cursor's: its top-left ``(size - 1) // 2`` before it), added up: the
    curves and intensities of ``fit_spectrum`` for the pixels whose fit
    worked, with their number in ``pixels``. ValueError if none did."""
    built = fits.built
    m, s = built.model, built.signal
    h, w = fits.failed.shape
    half = (size - 1) // 2
    x0, y0 = max(0, x - half), max(0, y - half)
    block = [
        (i, j) for j in range(y0, min(h, y0 + size)) for i in range(x0, min(w, x0 + size)) if not fits.failed[j, i]
    ]
    if not block:
        raise ValueError("The fit failed here")
    edges = {el: [c for c in m if hasattr(c, "onset_energy") and c.element == el] for el in built.masters}
    background_parts = [c for c in m if not hasattr(c, "onset_energy")]
    n = s.axes_manager.signal_axes[0].size
    data, background = np.zeros(n), np.zeros(n)
    curves = {el: np.zeros(n) for el in edges}
    intensity = dict.fromkeys(built.masters, 0.0)
    exponents = []
    with fits.lock:  # the model is moved from pixel to pixel
        for i, j in block:
            m.axes_manager.indices = (i, j)
            m.fetch_stored_values()
            data += np.asarray(s.data[j, i], float)
            background += m._get_current_data(component_list=background_parts, ignore_channel_switches=True)
            for el, parts in edges.items():
                curves[el] += m._get_current_data(component_list=parts, ignore_channel_switches=True)
            for el, c in built.masters.items():
                intensity[el] += float(c.intensity.value)
            exponents += [float(c.r.value) for c in m if hasattr(c, "r")]
        onsets = {el: float(c.onset_energy.value) for el, c in built.masters.items()}
        names = {el: c.name for el, c in built.masters.items()}
        single_std = {el: c.intensity.std for el, c in built.masters.items()} if len(block) == 1 else {}
    model = background + sum(curves.values())
    total = sum(max(v, 0.0) for v in intensity.values())
    rows = sorted(
        (
            {
                "element": el,
                "edge": names[el],
                "onset": onsets[el],
                "intensity": value,
                "std": float(single_std[el]) if single_std.get(el) is not None and math.isfinite(single_std[el]) else None,
                "atomic_percent": 100 * max(value, 0.0) / total if total > 0 else None,
                "fine_structure_width": fits.options.fine_structure_width if fits.options.fine_structure else None,
            }
            for el, value in intensity.items()
        ),
        key=lambda r: r["onset"],
    )
    resid = data - model
    return {
        "x": s.axes_manager.signal_axes[0].axis.tolist(),
        "data": data.tolist(),
        "model": np.nan_to_num(model).tolist(),
        "background": np.nan_to_num(background).tolist(),
        "elements": {el: np.nan_to_num(curve).tolist() for el, curve in curves.items()},
        "rows": rows,
        "background_r": float(np.mean(exponents)) if exponents else None,
        "residual_ratio": float(resid.std() / data.std()) if data.std() else None,
        "convolved": built.convolved,
        "gos": fits.options.gos,
        "skipped": built.skipped,
        "pixels": len(block),
    }


def atomic_percent_maps(maps: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """at% of each element from the fitted intensities (negative ones count as
    0; NaN where no element has any)."""
    positive = {el: np.clip(np.nan_to_num(v), 0, None) for el, v in maps.items()}
    total = sum(positive.values())
    with np.errstate(all="ignore"):
        return {el: np.where(total > 0, 100 * v / total, np.nan).astype(np.float32) for el, v in positive.items()}
