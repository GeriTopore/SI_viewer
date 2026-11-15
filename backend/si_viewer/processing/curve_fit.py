"""Curve fitting (M21) with HyperSpy's ``Model1D``: a model of the user's
components built on one spectrum, its parameters' starting values estimated
by the components themselves, and HyperSpy's ``fit``. Glue only: the model is
described by a plain spec (components, their parameters, the fit range) and
rebuilt for every request, so nothing of HyperSpy's is kept between calls."""

from __future__ import annotations

import math
import warnings
from typing import Literal

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


def build(spec_signal, model: ModelSpec):
    """The ``Model1D`` of ``model`` on the 1D signal ``spec_signal``, its
    parameters set, its signal range set. Returns (model, signal, (lo, hi))."""
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
            p.value = ps.value
            p.free = ps.free
            p.bmin, p.bmax = ps.bmin, ps.bmax
        # Values set by hand count as set (``as_signal`` draws only those).
        for p in comp.parameters:
            p.store_current_value_in_array()
    _twin(m, model)
    axis = s.axes_manager.signal_axes[0]
    lo = axis.low_value if model.lo is None else max(float(model.lo), axis.low_value)
    hi = axis.high_value if model.hi is None else min(float(model.hi), axis.high_value)
    if not lo < hi:
        raise ValueError("The fit range is outside the spectrum's energy range")
    m.set_signal_range(lo, hi)
    return m, s, (lo, hi)


def _fit_with_background_window(m, model: ModelSpec, fit_range: tuple[float, float], bounded):
    """The background components fitted over the window alone (the others
    switched off), then fixed while the others are fitted over ``fit_range``
    (their free parameters restored afterwards). Returns the last fit's info."""
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
        m.fit(bounded=bounded())
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
        return m.fit(bounded=bounded(), return_info=True)
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
