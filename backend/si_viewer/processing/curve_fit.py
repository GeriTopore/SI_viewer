"""Curve fitting (M21) with HyperSpy's ``Model1D``: a model of the user's
components built on one spectrum, its parameters' starting values estimated
by the components themselves, and HyperSpy's ``fit``; and the fit of every
pixel of a spectrum image (``multifit``). Glue only: the model is described
by a plain spec (components, their parameters) and rebuilt for every
request; only the fit of every pixel is kept (``PixelFits``), to draw it at
the cursor."""

from __future__ import annotations

import math
import threading
import warnings
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
from pydantic import BaseModel, Field

# HyperSpy's 1D components offered (the ones that take no required
# argument), grouped for the Add menu.
GROUPS: dict[str, list[str]] = {
    "Peaks": ["Gaussian", "GaussianHF", "Lorentzian", "Voigt", "SplitVoigt", "SkewNormal", "Doniach"],
    "Backgrounds": ["PowerLaw", "Exponential", "Polynomial", "Offset"],
    "Steps": ["Arctan", "Erf", "HeavisideStep", "Logistic"],
    "Other": ["Bleasdale", "RC"],
}
KINDS = [k for ks in GROUPS.values() for k in ks]
MAX_ORDER = 10


class TwinSpec(BaseModel):
    """A parameter following another (HyperSpy's ``twin``): its value is
    ``expr`` of the other's (``x``), e.g. "x + 5.4" or "0.5*x"."""

    component: str
    param: str
    expr: str = "x"


class ParamSpec(BaseModel):
    value: float = 0.0
    free: bool = True
    bmin: float | None = None
    bmax: float | None = None
    twin: TwinSpec | None = None


class ComponentSpec(BaseModel):
    # Unique within the model (HyperSpy requires it).
    name: str
    kind: str
    active: bool = True
    # Polynomial only.
    order: int = Field(2, ge=0, le=MAX_ORDER)
    params: dict[str, ParamSpec] = {}


class ModelSpec(BaseModel):
    components: list[ComponentSpec] = []
    # The fit range (energies; None: that end of the spectrum).
    lo: float | None = None
    hi: float | None = None
    # The background window (energies): the background components (PowerLaw,
    # Exponential, Polynomial, Offset) are fitted there alone, then fixed
    # while the others are fitted over the whole range. None: all together.
    bg_lo: float | None = None
    bg_hi: float | None = None


def _component_class(kind: str):
    import hyperspy.api as hs

    if kind not in KINDS:
        raise ValueError(f"Unknown component {kind}")
    return getattr(hs.model.components1D, kind)


def make_component(spec: ComponentSpec):
    cls = _component_class(spec.kind)
    comp = cls(order=spec.order) if spec.kind == "Polynomial" else cls()
    comp.name = spec.name
    return comp


def catalogue() -> dict:
    """The components offered: their group and parameters (name, default
    value, free) as HyperSpy makes them."""
    out = []
    for group, kinds in GROUPS.items():
        for kind in kinds:
            comp = make_component(ComponentSpec(name=kind, kind=kind))
            doc = (_component_class(kind).__doc__ or "").strip().splitlines()
            out.append({
                "kind": kind,
                "group": group,
                "summary": doc[0].strip() if doc else kind,
                "estimate": hasattr(comp, "estimate_parameters"),
                "params": [_param_out(p) for p in comp.parameters],
            })
    return {"components": out, "max_order": MAX_ORDER}


def _finite(v) -> float | None:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _param_out(p) -> dict:
    return {
        "name": p.name,
        "value": _finite(p.value) or 0.0,
        "free": bool(p.free),
        "bmin": _finite(p.bmin),
        "bmax": _finite(p.bmax),
        "std": _finite(p.std),
    }


def _plain(spec):
    """``spec`` (a 1D signal) as a plain ``Signal1D``: an EELS or EDS spectrum
    would make exSpy's own model (its background, its microscope checks)."""
    s = spec.deepcopy()
    s.set_signal_type("")
    s.change_dtype("float64")
    # Counts: Poisson noise (for the errors and the reduced χ²).
    if np.all(np.isfinite(s.data)) and np.nanmin(s.data) >= 0:
        s.estimate_poissonian_noise_variance()
        # A channel with no counts still has a variance (about one count),
        # else its weight is infinite and the fit fails.
        var = s.metadata.Signal.Noise_properties.variance
        if hasattr(var, "data"):
            var.data = np.maximum(np.asarray(var.data, float), 1.0)
        else:
            s.metadata.Signal.Noise_properties.variance = max(float(var), 1.0)
    return s


def build(spec_signal, model: ModelSpec, scale: float = 1.0):
    """The ``Model1D`` of ``model`` on ``spec_signal`` (a spectrum, or a
    spectrum image: every pixel starts from the values set), its parameters
    set, its signal range set. ``scale``: the linear parameters (amplitudes,
    areas, polynomial coefficients; HyperSpy's ``_linear``) and their limits
    multiplied by it, e.g. 1 / pixels for a model made on a summed spectrum
    and fitted pixel by pixel. Returns (model, signal, (lo, hi))."""
    s = _plain(spec_signal)
    names = [c.name for c in model.components]
    if len(set(names)) != len(names):
        raise ValueError("Each component needs its own name")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = s.create_model()
    for c in model.components:
        comp = make_component(c)
        m.append(comp)
        comp.active = c.active
        for p in comp.parameters:
            ps = c.params.get(p.name)
            if ps is None:
                continue
            k = scale if getattr(p, "_linear", False) else 1.0
            p.value = ps.value * k
            p.free = ps.free
            p.bmin = None if ps.bmin is None else ps.bmin * k
            p.bmax = None if ps.bmax is None else ps.bmax * k
        # Values set by hand count as set (``as_signal`` draws only those).
        for p in comp.parameters:
            p.store_current_value_in_array()
    _twin(m, model)
    if s.axes_manager.navigation_dimension:
        m.assign_current_values_to_all()
    axis = s.axes_manager.signal_axes[0]
    lo = axis.low_value if model.lo is None else max(float(model.lo), axis.low_value)
    hi = axis.high_value if model.hi is None else min(float(model.hi), axis.high_value)
    if not lo < hi:
        raise ValueError("The fit range is outside the spectrum's energy range")
    m.set_signal_range(lo, hi)
    return m, s, (lo, hi)


def _fit_with_background_window(m, model: ModelSpec, fit_range: tuple[float, float], bounded, run=None):
    """The background components fitted over the window alone (the others
    switched off), then fixed while the others are fitted over ``fit_range``
    (their free parameters restored afterwards). ``run(bounded)`` fits
    (default: ``m.fit``). Returns the last fit's info."""
    if run is None:
        def run(b):
            return m.fit(bounded=b, return_info=True)
    bw_lo, bw_hi = sorted((float(model.bg_lo), float(model.bg_hi)))
    lo, hi = fit_range
    if bw_lo < lo or bw_hi > hi:
        raise ValueError("The background window is outside the spectrum")
    axis = m.signal.axes_manager.signal_axes[0]
    if int(axis.value2index(bw_hi)) - int(axis.value2index(bw_lo)) < 3:
        raise ValueError("The background window is too narrow")
    background = [c for c in m if c.active and c.__class__.__name__ in GROUPS["Backgrounds"]]
    if not background:
        raise ValueError("Add a background component (PowerLaw, Exponential, Polynomial or Offset) to fit over the window")
    others = [c for c in m if c.active and c not in background]
    for c in others:
        c.active = False
    try:
        m.set_signal_range(bw_lo, bw_hi)
        run(bounded())
    finally:
        m.set_signal_range(lo, hi)
        for c in others:
            c.active = True
    if not others:
        return None
    was = [(p, p.free) for c in background for p in c.parameters]
    for p, _ in was:
        p.free = False
    try:
        if not any(p.free for c in others for p in c.parameters):
            return None
        return run(bounded())
    finally:
        for p, free in was:
            p.free = free


def _twin(m, model: ModelSpec) -> None:
    """Link the twinned parameters (HyperSpy's ``twin`` and
    ``twin_function_expr``); refuses a missing target, a parameter following
    itself and a loop."""
    by_name = {c.name: c for c in m}
    links: dict[tuple[str, str], tuple[str, str]] = {}
    for c in model.components:
        for pname, ps in c.params.items():
            if ps.twin is not None:
                links[(c.name, pname)] = (ps.twin.component, ps.twin.param)
    for start in links:
        seen, at = {start}, links[start]
        while at in links:
            if at in seen:
                raise ValueError(f"{start[0]} {start[1]} follows itself through its twins")
            seen.add(at)
            at = links[at]
    for c in model.components:
        for pname, ps in c.params.items():
            t = ps.twin
            if t is None:
                continue
            target = by_name.get(t.component)
            other = getattr(target, t.param, None) if target is not None else None
            if other is None or not hasattr(other, "twin"):
                raise ValueError(f"{c.name} {pname} follows {t.component} {t.param}, which doesn't exist")
            if (t.component, t.param) == (c.name, pname):
                raise ValueError(f"{c.name} {pname} can't follow itself")
            p = getattr(by_name[c.name], pname)
            p.twin = other
            expr = (t.expr or "x").strip()
            if expr != "x":
                try:
                    p.twin_function_expr = expr
                except Exception as e:
                    raise ValueError(f"{c.name} {pname}: can't read the formula \"{expr}\" (write it in x, e.g. x + 5.4)") from e


def estimate(spec_signal, model: ModelSpec, name: str) -> dict:
    """The starting values of component ``name`` from its own
    ``estimate_parameters`` over the fit range. Returns its parameters."""
    m, s, (lo, hi) = build(spec_signal, model)
    comp = next((c for c in m if c.name == name), None)
    if comp is None:
        raise ValueError(f"No component {name}")
    if not hasattr(comp, "estimate_parameters"):
        raise ValueError(f"{comp.__class__.__name__} can't estimate its parameters")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ok = comp.estimate_parameters(s, lo, hi, only_current=True)
    if ok is False:
        raise ValueError(f"Could not estimate {name} over {lo:g}–{hi:g}")
    return {"params": [_param_out(p) for p in comp.parameters]}


def fit(spec_signal, model: ModelSpec, run: bool = True) -> dict:
    """Fit ``model`` to the 1D signal (``run``; otherwise just evaluate it
    with its current values) with HyperSpy's default optimizer, bounded when
    a free parameter has a limit. Returns the curves over the fit range, each
    component's parameters (with errors after a fit) and the goodness of fit."""
    m, s, (lo, hi) = build(spec_signal, model)
    if not len(m):
        raise ValueError("Add a component to fit")
    info = None
    if run:
        if not any(p.free for c in m if c.active for p in c.parameters):
            raise ValueError("Every parameter is fixed: free one to fit")
        def bounded() -> bool:
            return any(
                p.free and (p.bmin is not None or p.bmax is not None) for c in m if c.active for p in c.parameters
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                if model.bg_lo is not None and model.bg_hi is not None:
                    info = _fit_with_background_window(m, model, (lo, hi), bounded)
                else:
                    info = m.fit(bounded=bounded(), return_info=True)
            except ValueError:
                raise
            except Exception as e:
                raise ValueError(f"The fit failed: {e}") from e
    axis = s.axes_manager.signal_axes[0]
    i0, i1 = int(axis.value2index(lo)), int(axis.value2index(hi)) + 1
    x = axis.axis[i0:i1]
    data = np.asarray(s.data, float)[i0:i1]

    def curve(components) -> list[float]:
        if not components:
            return [0.0] * (i1 - i0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = np.asarray(m.as_signal(component_list=components, show_progressbar=False).data, float)[i0:i1]
        return np.nan_to_num(out).tolist()

    active = [c for c in m if c.active]
    model_curve = curve(active)
    resid = data - np.asarray(model_curve)
    out = {
        "x": x.tolist(),
        "model": model_curve,
        "components": [
            {"name": c.name, "curve": curve([c]) if c.active else None, "params": [_param_out(p) for p in c.parameters]}
            for c in m
        ],
        "range": [float(lo), float(hi)],
        # Plotted counts per channel = the components' function × this (the
        # channel width when the axis is binned, HyperSpy's convention).
        "factor": float(axis.scale) if axis.is_binned else 1.0,
        "residual_ratio": float(resid.std() / data.std()) if data.std() else None,
        "fitted": run,
    }
    if run:
        out.update({
            "chisq": _finite(np.asarray(m.chisq.data).ravel()[0]),
            "red_chisq": _finite(np.asarray(m.red_chisq.data).ravel()[0]),
            "dof": int(np.asarray(m.dof.data).ravel()[0]),
            "poisson": s.metadata.has_item("Signal.Noise_properties.variance"),
            "success": bool(getattr(info, "success", True)),
            "message": str(getattr(info, "message", "") or "").strip(),
        })
    return out


# ---------- every pixel (M21c) ----------

def check(spec_signal, model: ModelSpec) -> None:
    """ValueError if ``model`` can't be fitted to ``spec_signal``: no
    component, nothing free, or a set-up HyperSpy refuses."""
    m, _, _ = build(spec_signal, model)
    if not len(m):
        raise ValueError("Add a component to fit")
    if not any(p.free for c in m if c.active for p in c.parameters):
        raise ValueError("Every parameter is fixed: free one to fit")


# Progress is reported (and the job can be cancelled) every this many pixels.
REPORT_EVERY = 64


def _each_pixel(m, failed: np.ndarray, fit_one: Callable[[], None], report: Callable[[float], None]) -> None:
    """``fit_one()`` at every pixel of the multi-dimensional model ``m``
    (serpentine), each starting from its stored values (HyperSpy's ``fit``
    stores the result, χ² and degrees of freedom there). A pixel whose fit
    raises is marked in ``failed`` (flat, array order) and skipped by later
    passes. ``report`` gets the fraction done every ``REPORT_EVERY`` pixels."""
    am = m.axes_manager
    shape = tuple(am._navigation_shape_in_array)
    n = failed.size
    done = 0
    with am.events.indices_changed.suppress_callback(m.fetch_stored_values):
        with am.switch_iterpath("serpentine"), m.suspend_update(update_on_resume=True):
            for index in am:
                i = int(np.ravel_multi_index(index[::-1], shape))
                if not failed[i]:
                    m.fetch_stored_values()
                    try:
                        fit_one()
                    except Exception:
                        failed[i] = True
                done += 1
                if done % REPORT_EVERY == 0 or done == n:
                    report(done / n)
    am.events.indices_changed.trigger(am)


def _bounded(m) -> bool:
    """Whether a free parameter has a limit (the fit is then bounded)."""
    return any(p.free and (p.bmin is not None or p.bmax is not None) for c in m if c.active for p in c.parameters)


@dataclass
class PixelFits:
    """The fit of every pixel of a spectrum image (``fit_pixels``), kept to
    draw the fit at any pixel (``pixel_fit``)."""

    model: object  # HyperSpy's Model1D, every pixel's values stored
    signal: object
    spec: ModelSpec
    failed: np.ndarray  # (height, width): pixels whose fit failed
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


def map_parameters(spec: ModelSpec) -> list[tuple[str, str]]:
    """The (component, parameter) pairs mapped after fitting every pixel:
    the ones that vary, i.e. not fixed (free, limited or twinned), of the
    components switched on."""
    out = []
    for c in spec.components:
        if not c.active:
            continue
        for p in make_component(c).parameters:
            ps = c.params.get(p.name)
            if (p.free if ps is None else ps.free or ps.twin is not None):
                out.append((c.name, p.name))
    return out


def fit_pixels(
    signal, spec: ModelSpec, pixels: int = 1, report: Callable[[float], None] = lambda f: None,
) -> tuple[dict[str, np.ndarray], PixelFits]:
    """Fit ``spec`` at every pixel of the spectrum image ``signal``
    (HyperSpy's ``multifit``, chunk by chunk; with the background window as
    the single fit), every pixel starting from the values set. ``pixels``:
    how many pixels the spectrum the model was made on adds up (its linear
    parameters are divided by it). ``report`` gets the fraction done after
    each chunk (and may raise to stop). Returns maps (navigation shape; NaN
    where the fit failed): "<component> · <parameter>" for each parameter
    that varies, and "Reduced χ²"; and the fits themselves."""
    if signal.axes_manager.navigation_dimension != 2:
        raise ValueError("Fitting every pixel needs a 2D spectrum image")
    m, s, (lo, hi) = build(signal, spec, scale=1.0 / max(pixels, 1))
    per_pixel = scaled(spec, 1.0 / max(pixels, 1))
    if not len(m):
        raise ValueError("Add a component to fit")
    if not any(p.free for c in m if c.active for p in c.parameters):
        raise ValueError("Every parameter is fixed: free one to fit")
    window = spec.bg_lo is not None and spec.bg_hi is not None
    nav = tuple(s.data.shape[:-1])
    n = int(np.prod(nav))
    failed = np.zeros(n, bool)

    # One pass through the pixels per fit (two with the background window:
    # the background alone, then the rest), as HyperSpy's ``multifit`` does,
    # but reporting progress and leaving out a pixel whose fit fails.
    passes = 2 if window else 1
    done_passes = [0]

    def each_pixel(bounded: bool) -> None:
        k = done_passes[0]
        done_passes[0] += 1
        _each_pixel(m, failed, lambda: m.fit(bounded=bounded), lambda f: report((k + f) / passes))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if window:
            _fit_with_background_window(m, spec, (lo, hi), lambda: _bounded(m), each_pixel)
        else:
            each_pixel(_bounded(m))

    by_name = {c.name: c for c in m}
    maps: dict[str, np.ndarray] = {}
    for cname, pname in map_parameters(spec):
        values = np.array(getattr(by_name[cname], pname).map["values"], np.float32).reshape(n)
        values[failed | ~np.isfinite(values)] = np.nan
        maps[f"{cname} · {pname}"] = values.reshape(nav)
    chi = np.array(m.red_chisq.data, np.float32).reshape(n)
    chi[failed | ~np.isfinite(chi)] = np.nan
    maps["Reduced χ²"] = chi.reshape(nav)
    return maps, PixelFits(m, s, per_pixel, failed.reshape(nav))


def model_signal(fits: PixelFits):
    """The fitted model at every pixel as a spectrum image (HyperSpy's
    ``as_signal``; zero where the fit failed)."""
    with fits.lock, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = fits.model.as_signal(show_progressbar=False)
    out.change_dtype("float32")
    out.data[fits.failed] = 0
    return out


def scaled(spec: ModelSpec, k: float) -> ModelSpec:
    """``spec`` with its linear parameters (and their limits) multiplied by
    ``k``, as ``build(..., scale=k)`` sets them."""
    out = spec.model_copy(deep=True)
    for c in out.components:
        for p in make_component(c).parameters:
            ps = c.params.get(p.name)
            if ps is None or not getattr(p, "_linear", False):
                continue
            ps.value *= k
            ps.bmin = None if ps.bmin is None else ps.bmin * k
            ps.bmax = None if ps.bmax is None else ps.bmax * k
    return out


def model_at(signal, spec: ModelSpec, maps: dict[str, np.ndarray], x: int, y: int, size: int = 1) -> dict:
    """The fitted model of a spectrum image at the ``size`` × ``size`` block
    at (x, y) (as the cursor's), rebuilt from its maps (``fit_pixels``: the
    model per pixel ``spec``, "<component> · <parameter>" and "Reduced χ²"):
    each parameter set from its map at every pixel, the curves added up over
    the pixels whose fit worked, the parameters their mean. As ``fit``
    returns a fit, with the data (added up). ValueError if none worked."""
    chi_map = maps["Reduced χ²"]
    h, w = chi_map.shape
    half = (size - 1) // 2
    x0, y0 = max(0, x - half), max(0, y - half)
    x1, y1 = min(w, x0 + size), min(h, y0 + size)
    ok = np.isfinite(chi_map[y0:y1, x0:x1])
    if not ok.any():
        raise ValueError("The fit failed here")
    block = signal.inav[x0:x1, y0:y1]
    m, s, _ = build(block, spec)
    for c in m:
        for p in c.parameters:
            values = maps.get(f"{c.name} · {p.name}")
            if values is not None:
                p.map["values"][:] = np.nan_to_num(values[y0:y1, x0:x1])
                p.map["is_set"][:] = True
    axis = s.axes_manager.signal_axes[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        curves = {
            c.name: np.asarray(m.as_signal(component_list=[c], show_progressbar=False).data, float)[ok].sum(axis=0)
            for c in m if c.active
        }
    data = np.asarray(s.data, float)[ok].sum(axis=0)
    model = sum(curves.values(), np.zeros(axis.size))

    def params(c) -> list[dict]:
        out = []
        for p in c.parameters:
            values = maps.get(f"{c.name} · {p.name}")
            v = float(np.nanmean(values[y0:y1, x0:x1][ok])) if values is not None else float(p.value)
            out.append({**_param_out(p), "value": v, "std": None})
        return out

    resid = data - model
    chi = chi_map[y0:y1, x0:x1][ok]
    return {
        "x": axis.axis.tolist(),
        "data": data.tolist(),
        "model": np.nan_to_num(model).tolist(),
        "components": [
            {"name": c.name, "curve": np.nan_to_num(curves[c.name]).tolist() if c.active else None, "params": params(c)}
            for c in m
        ],
        "range": [float(axis.low_value), float(axis.high_value)],
        "factor": float(axis.scale) if axis.is_binned else 1.0,
        "residual_ratio": float(resid.std() / data.std()) if data.std() else None,
        "fitted": True,
        "chisq": None,
        "red_chisq": float(np.mean(chi)),
        "dof": None,
        "poisson": s.metadata.has_item("Signal.Noise_properties.variance"),
        "success": True,
        "message": "",
        "pixels": int(ok.sum()),
    }


def pixel_fit(fits: PixelFits, x: int, y: int, size: int = 1) -> dict:
    """The fit of every pixel at the ``size`` × ``size`` block at (x, y) (as
    the cursor's: its top-left ``(size - 1) // 2`` before it), as ``fit``
    returns a fit: the curves added up over the pixels whose fit worked, the
    parameters (their mean over the block; errors for one pixel) and the
    reduced χ² (mean). ValueError if no pixel's fit worked."""
    m, s = fits.model, fits.signal
    h, w = fits.failed.shape
    half = (size - 1) // 2
    x0, y0 = max(0, x - half), max(0, y - half)
    block = [
        (i, j) for j in range(y0, min(h, y0 + size)) for i in range(x0, min(w, x0 + size)) if not fits.failed[j, i]
    ]
    if not block:
        raise ValueError("The fit failed here")
    axis = s.axes_manager.signal_axes[0]
    nch = axis.size
    data, model = np.zeros(nch), np.zeros(nch)
    comps = list(m)
    curves = {c.name: np.zeros(nch) for c in comps if c.active}
    sums: dict[tuple[str, str], float] = {}
    chis: list[float] = []
    one: dict[tuple[str, str], dict] = {}
    with fits.lock, warnings.catch_warnings():  # the model is moved from pixel to pixel
        warnings.simplefilter("ignore")
        for i, j in block:
            m.axes_manager.indices = (i, j)
            m.fetch_stored_values()
            data += np.asarray(s.data[j, i], float)
            for c in comps:
                if c.active:
                    curves[c.name] += np.nan_to_num(
                        m._get_current_data(component_list=[c], ignore_channel_switches=True)
                    )
                for p in c.parameters:
                    sums[(c.name, p.name)] = sums.get((c.name, p.name), 0.0) + float(p.value)
            chis.append(float(np.asarray(m.red_chisq.data)[j, i]))
        if len(block) == 1:
            one = {(c.name, p.name): _param_out(p) for c in comps for p in c.parameters}
        meta = {(c.name, p.name): _param_out(p) for c in comps for p in c.parameters}
    model = sum(curves.values(), np.zeros(nch))
    params_of = lambda c: [  # noqa: E731
        one.get((c.name, p.name)) or {**meta[(c.name, p.name)], "value": sums[(c.name, p.name)] / len(block), "std": None}
        for p in c.parameters
    ]
    resid = data - model
    chi = [v for v in chis if math.isfinite(v)]
    return {
        "x": axis.axis.tolist(),
        "model": model.tolist(),
        "components": [
            {"name": c.name, "curve": curves[c.name].tolist() if c.active else None, "params": params_of(c)}
            for c in comps
        ],
        "range": [float(axis.low_value), float(axis.high_value)],
        "factor": float(axis.scale) if axis.is_binned else 1.0,
        "residual_ratio": float(resid.std() / data.std()) if data.std() else None,
        "fitted": True,
        "chisq": None,
        "red_chisq": float(np.mean(chi)) if chi else None,
        "dof": None,
        "poisson": s.metadata.has_item("Signal.Noise_properties.variance"),
        "success": True,
        "message": "",
        "pixels": len(block),
    }
