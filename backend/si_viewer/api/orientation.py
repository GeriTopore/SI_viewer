"""Orientation and phase mapping as pyxem does it: a Structure's templates
(diffsims), their views (orix), and the polar transform of 4D-STEM data."""

from __future__ import annotations


import numpy as np

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..processing.orientation import MatchOptions, PolarOptions, TemplateOptions
from .fourd import _cube

router = APIRouter(tags=["orientation"])

# 1/Å in the pattern axis' units.
PER_ANGSTROM = {"1/Å": 1.0, "1/A": 1.0, "1/nm": 10.0}


def _structure(request: Request, dataset_id: str):
    """A Structure and its orix phase (named after the Structure)."""
    from ..processing.structure import StructureError, is_structure, phase_of

    try:
        ds = request.app.state.session.get_dataset(dataset_id)
    except KeyError:
        raise HTTPException(404, f"Unknown dataset id: {dataset_id}")
    if not is_structure(ds.signal):
        raise HTTPException(400, f"{ds.title} is not a crystal structure (open a .cif)")
    try:
        phase = phase_of(ds.signal)
    except StructureError as e:
        raise HTTPException(422, str(e))
    phase.name = ds.title
    return ds, phase


def _zones(o: TemplateOptions) -> str:
    return " ".join(f"[{' '.join(f'{x:g}' for x in z.uvw)}]" for z in o.zone_axes)


class TemplatesBody(BaseModel):
    options: TemplateOptions = TemplateOptions()


@router.post("/api/datasets/{dataset_id}/templates/make")
def make_templates(dataset_id: str, body: TemplatesBody, request: Request) -> dict:
    """Simulate a Structure's templates (as a job): a new Templates object
    under it. The job's result is ``{file, id, templates}``."""
    from ..processing import orientation

    session = request.app.state.session
    structure, phase = _structure(request, dataset_id)
    o = body.options
    try:
        rotations = orientation.template_rotations(phase, o)
    except ValueError as e:
        raise HTTPException(422, str(e))

    def run(job) -> dict:
        job.report(0.05, f"Simulating {rotations.size} templates of {phase.name}")
        sim = orientation.simulate(phase, o, rotations)
        title = f"Templates · {o.resolution:g}°" + (f", {_zones(o)}" if o.zone_axes else "")
        record = {"name": phase.name, "templates": int(rotations.size), "options": o.model_dump()}
        ds = session.add_templates(structure, record, title)
        session.template_sims[ds.id] = (record["options"], sim)
        return {"file": session.files[structure.file_id].summary(session), "id": ds.id, "templates": record["templates"]}

    return request.app.state.jobs.submit("templates", f"Templates of {structure.title}", run).summary()


class PolarBody(BaseModel):
    options: PolarOptions = PolarOptions()
    in_place: bool = False


@router.post("/api/datasets/{dataset_id}/4d/polar")
def polar_transform(dataset_id: str, body: PolarBody, request: Request) -> dict:
    """The polar transform of every pattern (pyxem's, as its orientation
    mapping tutorial makes it), as a copy under the data or in place (a job).
    The job's result is ``{file, id}`` (the data transformed)."""
    session = request.app.state.session
    cube = _cube(request, dataset_id)
    if str(cube.signal.axes_manager.signal_axes[0].units) not in PER_ANGSTROM:
        raise HTTPException(409, "Calibrate the pattern's pixel size first (1/Å or 1/nm)")

    def run(job) -> dict:
        job.report(0.02, "Polar transform")
        try:
            made = session.polar_transform(cube, body.options, body.in_place, lambda f, m: job.report(f, m))
        except ValueError as e:
            raise RuntimeError(str(e))
        return {"file": session.file_of(cube).summary(session), "id": made.id}

    return request.app.state.jobs.submit("polar", f"Polar transform of {cube.title}", run).summary()


class MatchBody(BaseModel):
    templates: str  # a Templates object's id
    positions: list[tuple[int, int]] = Field(min_length=1, max_length=64)  # scan (x, y)
    # The polar transform matched (a dataset made by Polar transform); None:
    # the data itself when it is one, else its newest polar transform.
    polar: str | None = None
    options: MatchOptions = MatchOptions()


def _polar_of(request: Request, ds, polar_id: str | None):
    """The polar transform matched for ``ds`` (4D-STEM data or a polar transform)."""
    session = request.app.state.session

    def is_polar(d):
        return d.signal.metadata.get_item("Signal.signal_type") == "polar_diffraction"

    if polar_id:
        try:
            p = session.get_dataset(polar_id)
        except KeyError:
            raise HTTPException(404, f"Unknown dataset id: {polar_id}")
        if not is_polar(p):
            raise HTTPException(422, f"{p.title} is not a polar transform")
        return p
    if is_polar(ds):
        return ds
    made = [d for d in session.siblings(ds) if d.parent_id == ds.id and is_polar(d)]
    if not made:
        raise HTTPException(409, "Make a polar transform of the data first (Polar transform)")
    return made[-1]


@router.post("/api/datasets/{dataset_id}/4d/orientation/match")
def match_at(dataset_id: str, body: MatchBody, request: Request) -> dict:
    """Template matching (pyxem's ``get_orientation``, all templates kept by
    default, as the tutorial at a few positions) of the polar transform at
    scan ``positions``: ``{polar, matches}`` (see ``orientation.match``)."""
    from ..processing import orientation

    ds = _cube(request, dataset_id)
    polar = _polar_of(request, ds, body.polar)
    nx, ny = polar.signal.axes_manager.navigation_shape
    for x, y in body.positions:
        if not (0 <= x < nx and 0 <= y < ny):
            raise HTTPException(422, f"({x}, {y}) is outside the scan")
    _, sim, phase = _templates(request, body.templates)
    patterns = orientation.polar_patterns(polar.signal, body.positions)
    matches = orientation.match(patterns, sim, body.options, phase)
    for m, (x, y) in zip(matches, body.positions):
        m["x"], m["y"] = x, y
        m["best"]["uvw"] = orientation.direction_of(sim.rotations[m["best"]["index"]], phase)
    return {"polar": polar.id, "matches": matches}


# ---------- Templates objects (under their Structures) and orix's plots ----------


def _templates(request: Request, dataset_id: str):
    """A Templates object, its simulation (made again from its record when
    not kept: after opening a saved analysis) and its Structure's phase."""
    from ..processing import orientation

    session = request.app.state.session
    try:
        ds = session.get_dataset(dataset_id)
    except KeyError:
        raise HTTPException(404, f"Unknown dataset id: {dataset_id}")
    if ds.origin != "templates" or not ds.templates:
        raise HTTPException(400, f"{ds.title} is not a set of templates")
    _, phase = _structure(request, ds.parent_id)
    options = ds.templates["options"]
    kept = session.template_sims.get(ds.id)
    if kept is None or kept[0] != options:
        try:
            sim = orientation.simulate(phase, TemplateOptions(**options))
        except ValueError as e:
            raise HTTPException(422, str(e))
        with session._lock:
            session.template_sims[ds.id] = (options, sim)
        kept = (options, sim)
    return ds, kept[1], phase


@router.get("/api/datasets/{dataset_id}/templates")
def templates(dataset_id: str, request: Request) -> dict:
    """The templates' inverse pole figure (orix): the sector, its colour key
    and each template's beam direction (``points``: x, y, rgb)."""
    from ..processing import orientation

    ds, sim, phase = _templates(request, dataset_id)
    if "ipf" not in ds.cache:
        ds.cache["ipf"] = {
            "sector": orientation.ipf_sector(phase),
            "points": orientation.template_directions(sim.rotations, phase),
        }
    return ds.cache["ipf"]


@router.get("/api/datasets/{dataset_id}/templates/nearest")
def nearest(dataset_id: str, request: Request, u: float, v: float, w: float) -> dict:
    """The template whose beam direction is nearest [u v w]."""
    from ..processing import orientation

    _, sim, phase = _templates(request, dataset_id)
    if u == v == w == 0:
        raise HTTPException(422, "[0 0 0] is not a direction")
    return {"index": orientation.nearest_template(sim.rotations, phase, (u, v, w))}


@router.get("/api/datasets/{dataset_id}/templates/toward")
def toward(dataset_id: str, request: Request, u: float, v: float, w: float) -> dict:
    """The template nearest the beam direction [u v w], its reflections
    taken onto [u v w]'s zone (``template_toward``)."""
    from ..processing import orientation

    _, sim, phase = _templates(request, dataset_id)
    if u == v == w == 0:
        raise HTTPException(422, "[0 0 0] is not a direction")
    return orientation.template_toward(sim, phase, (u, v, w))


@router.get("/api/datasets/{dataset_id}/templates/{index}")
def template(dataset_id: str, index: int, request: Request) -> dict:
    """Template ``index``: its reflections (x, y in 1/Å, intensity, hkl), the
    beam direction ([u, v, w]) and its rotation (Euler angles, degrees)."""
    from ..processing import orientation

    ds, sim, phase = _templates(request, dataset_id)
    n = sim.rotations.size
    if not 0 <= index < n:
        raise HTTPException(404, f"There are {n} templates")
    rot = sim.rotations[index]
    return {
        **orientation.template_pattern(sim.irot[index] if n > 1 else sim),
        "index": index,
        "uvw": orientation.direction_of(rot, phase),
        "euler": np.degrees(np.asarray(rot.to_euler()).ravel()).round(3).tolist(),
        "reciprocal_radius": float(ds.templates["options"]["reciprocal_radius"]),
    }


@router.get("/api/datasets/{dataset_id}/structure/plots")
def structure_plots(dataset_id: str, request: Request) -> dict:
    """orix's plots of a Structure: its point group's symmetry in the
    stereographic projection, and its inverse pole figure's colour key."""
    from ..processing import orientation
    from ..processing.structure import StructureError, is_structure, phase_of

    try:
        ds = request.app.state.session.get_dataset(dataset_id)
    except KeyError:
        raise HTTPException(404, f"Unknown dataset id: {dataset_id}")
    if not is_structure(ds.signal):
        raise HTTPException(400, f"{ds.title} is not a crystal structure")
    if "plots" not in ds.cache:
        try:
            phase = phase_of(ds.signal)
        except StructureError as e:
            raise HTTPException(422, str(e))
        ds.cache["plots"] = {"symmetry": orientation.symmetry_poles(phase), "sector": orientation.ipf_sector(phase)}
    return ds.cache["plots"]


@router.get("/api/datasets/{dataset_id}/structure/atoms")
def structure_atoms(dataset_id: str, request: Request) -> dict:
    """A Structure's unit cell to draw in 3D (see ``structure_atoms``)."""
    from ..processing.structure import StructureError, is_structure
    from ..processing.structure import structure_atoms as atoms_of

    try:
        ds = request.app.state.session.get_dataset(dataset_id)
    except KeyError:
        raise HTTPException(404, f"Unknown dataset id: {dataset_id}")
    if not is_structure(ds.signal):
        raise HTTPException(400, f"{ds.title} is not a crystal structure")
    if "atoms" not in ds.cache:
        try:
            ds.cache["atoms"] = atoms_of(ds.signal)
        except StructureError as e:
            raise HTTPException(422, str(e))
    return ds.cache["atoms"]
