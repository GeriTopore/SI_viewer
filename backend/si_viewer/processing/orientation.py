"""Orientation and phase mapping as pyxem does it (the workflow of its
SPED-Ag orientation mapping tutorial): a Structure's templates (diffsims, over
orix's sampling of its reduced fundamental zone, optionally only near some
zone axes) matched against the polar transform of the patterns
(``PolarDiffraction2D.get_orientation``)."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field


class ZoneAxis(BaseModel):
    """Orientations kept within ``angle`` (degrees) of [``uvw``]."""

    uvw: tuple[float, float, float]
    angle: float = Field(5.0, gt=0, le=90)


class TemplateOptions(BaseModel):
    """diffsims' ``SimulationGenerator`` and ``calculate_diffraction2d``
    settings, and orix's orientation sampling (defaults: the tutorial's)."""

    kv: float = Field(200.0, gt=0)  # accelerating voltage
    # Angular step of the orientations (degrees) over the phase's reduced
    # fundamental zone (orix ``get_sample_reduced_fundamental``).
    resolution: float = Field(2.0, ge=0.1, le=10)
    # Only the orientations near these zone axes (the tutorial's
    # ``reduce_beam_directions``); none: all of them.
    zone_axes: list[ZoneAxis] = []
    reciprocal_radius: float = Field(1.5, gt=0, le=10)  # 1/Å
    max_excitation_error: float = Field(0.001, gt=0, le=1)  # 1/Å
    precession_angle: float = Field(1.0, ge=0, le=5)  # degrees
    minimum_intensity: float = Field(1e-5, ge=0)


def _generator(o: TemplateOptions):
    from diffsims.generators.simulation_generator import SimulationGenerator

    return SimulationGenerator(
        accelerating_voltage=o.kv, precession_angle=o.precession_angle, minimum_intensity=o.minimum_intensity,
    )


def reflections_1d(phase, reciprocal_radius: float) -> list[dict]:
    """diffsims' powder reflections of ``phase`` (``calculate_diffraction1d``):
    [{"hkl", "k" (1/Å), "intensity"}] by k, out to ``reciprocal_radius``."""
    from diffsims.generators.simulation_generator import SimulationGenerator

    sim = SimulationGenerator().calculate_diffraction1d(phase=phase, reciprocal_radius=reciprocal_radius)
    out = [{"hkl": [int(round(float(h))) for h in hkl], "k": float(k), "intensity": float(i)}
           for hkl, k, i in zip(sim.hkl, sim.reciprocal_spacing, sim.intensities)
           if float(k) <= reciprocal_radius + 1e-9]
    return sorted(out, key=lambda r: r["k"])


def orientations(phase, resolution: float):
    """orix's sampling of ``phase``'s reduced fundamental zone."""
    from orix.sampling import get_sample_reduced_fundamental

    return get_sample_reduced_fundamental(resolution=resolution, point_group=phase.point_group)


def template_rotations(phase, o: TemplateOptions):
    """The rotations templates are made at: orix's sampling, only those whose
    beam direction is within each zone axis' angle when there are any (the
    tutorial's ``reduce_beam_directions``, the zone axes' sets joined)."""
    from orix.quaternion import Orientation, Rotation
    from orix.vector import Miller, Vector3d

    grid = orientations(phase, o.resolution)
    if not o.zone_axes:
        return grid
    beam = Orientation(grid, symmetry=phase.point_group) * Vector3d.zvector()
    keep = []
    for z in o.zone_axes:
        if not any(z.uvw):
            raise ValueError("[0 0 0] is not a zone axis")
        angles = np.asarray(beam.angle_with(Miller(uvw=list(z.uvw), phase=phase))).ravel()
        keep.append(np.asarray(grid.data).reshape(-1, 4)[angles <= np.deg2rad(z.angle)])
    data = np.concatenate(keep)
    if not len(data):
        raise ValueError("No orientation is that close to the zone axes: widen the angle or refine the resolution")
    return Rotation(data)


def simulate(phase, o: TemplateOptions, rotations=None):
    """A Structure's templates (a diffsims ``Simulation2D``). ValueError if
    an orientation has no reflection: diffsims (0.7) can't simulate it."""
    try:
        return _generator(o).calculate_diffraction2d(
            phase,
            rotation=template_rotations(phase, o) if rotations is None else rotations,
            reciprocal_radius=o.reciprocal_radius,
            max_excitation_error=o.max_excitation_error,
            with_direct_beam=False,
        )
    except ValueError as e:
        # diffsims takes the maximum intensity of each orientation's
        # reflections, and fails where there are none.
        if "zero-size array" not in str(e):
            raise
        how = "raise the excitation error" + ("" if o.precession_angle else " (0.01 1/Å or more without precession) or add precession")
        raise ValueError(f"Some orientations have no reflection within the excitation error "
                         f"({o.max_excitation_error:g} 1/Å): {how}") from None


def zone_rotation(phase, uvw):
    """The rotation putting the crystal direction ``uvw`` along the beam."""
    from orix.quaternion import Rotation
    from orix.vector import Miller, Vector3d

    v = Vector3d(Miller(uvw=list(uvw), phase=phase).data)
    if not np.all(np.isfinite(v.data)) or np.linalg.norm(v.data) == 0:
        raise ValueError(f"[{' '.join(str(x) for x in uvw)}] is not a direction")
    return ~Rotation.from_align_vectors(Vector3d.zvector(), v)


def zone_template(phase, uvw, o: TemplateOptions) -> dict:
    """The template of ``phase`` with ``uvw`` along the beam: ``x``, ``y``
    (1/Å), ``intensity`` (the brightest 1) and ``hkl`` of each reflection."""
    sim = _generator(o).calculate_diffraction2d(
        phase, rotation=zone_rotation(phase, uvw), reciprocal_radius=o.reciprocal_radius,
        max_excitation_error=o.max_excitation_error, with_direct_beam=False,
    )
    c = sim.coordinates
    xy = np.asarray(c.data)[:, :2]
    inten = np.asarray(c.intensity, float)
    top = inten.max() if inten.size else 1.0
    hkl = np.rint(np.asarray(c.hkl)).astype(int) if len(xy) else np.zeros((0, 3), int)
    return {
        "x": xy[:, 0].round(6).tolist(),
        "y": xy[:, 1].round(6).tolist(),
        "intensity": (inten / top).round(5).tolist() if top > 0 else inten.tolist(),
        "hkl": hkl.tolist(),
    }


class PolarOptions(BaseModel):
    """pyxem's ``get_azimuthal_integral2d`` (``mean=True``) of the patterns,
    then a gamma (pyxem's advice: 0.5) before matching."""

    npt: int = Field(112, ge=10, le=1000)  # radial points
    npt_azim: int = Field(360, ge=36, le=3600)  # azimuthal points
    r_min: float = Field(0.0, ge=0)  # 1/Å
    r_max: float | None = Field(None, gt=0)  # 1/Å; None: the whole pattern (pyxem's default)
    gamma: float = Field(1.0, gt=0, le=2)  # 1: none (the tutorial's)
    # Make the ring circular with the calibration's ellipse (pyxem's affine).
    ellipse: bool = True


def diffraction2d(data: np.ndarray, axes, affine=None):
    """``data`` (…, ky, kx) as a pyxem ``Diffraction2D`` with the pattern
    axes ``axes`` ([(scale, offset, units)] for ky, kx): its centre where
    they are 0, and the ellipse's affine."""
    import pyxem

    s = pyxem.signals.Diffraction2D(np.asarray(data, dtype=np.float32))
    for ax, (scale, offset, units) in zip(s.axes_manager.signal_axes, list(axes)[::-1]):
        ax.scale, ax.offset, ax.units = float(scale), float(offset), str(units)
    if affine is not None:
        s.calibration.affine = np.asarray(affine, float)
    return s


# pyxem's integration slices of recent polar transforms (the slow part:
# the same for every pattern of a data with the same axes and settings).
_SLICES: dict[tuple, tuple] = {}
_SLICES_KEPT = 8


def _slices(s, o: PolarOptions, per: float):
    """pyxem's integration slices of ``s``'s patterns (kept: the slow part),
    and the radial range (axis units)."""
    cal = s.calibration
    hi = o.r_max * per if o.r_max is not None else cal._get_radial_range()[1]
    lo = o.r_min * per
    if not hi > lo:
        raise ValueError("The radial range is empty: its end must be past its start")
    affine = cal.affine
    key = (
        tuple(s.data.shape[-2:]),
        tuple((ax.scale, ax.offset) for ax in s.axes_manager.signal_axes),
        None if affine is None else np.asarray(affine, float).round(12).tobytes(),
        o.npt, o.npt_azim, float(lo), float(hi),
    )
    if key not in _SLICES:
        if len(_SLICES) >= _SLICES_KEPT:
            _SLICES.pop(next(iter(_SLICES)))
        _SLICES[key] = cal.get_slices2d(o.npt, o.npt_azim, radial_range=(lo, hi), azimuthal_range=(-np.pi, np.pi))
    return _SLICES[key]


def polar_of(s, o: PolarOptions, per: float) -> dict:
    """The polar transform of one pattern, pyxem ``Diffraction2D`` ``s``
    (axes in 1/Å × ``per``), as the tutorial makes it
    (``get_azimuthal_integral2d``: pyxem's slices, kept, and its
    ``_slice_radial_integrate``), then the gamma: ``data`` (radius ×
    azimuth), ``radius`` and ``azimuth`` axes ({offset, scale}, 1/Å and
    radians). ``s`` must be fresh: pyxem changes its calibration's affine
    in place whenever it makes the slices (only the first use is as set)."""
    from pyxem.utils._azimuthal_integrations import _slice_radial_integrate

    cal = s.calibration
    slices, factors, factors_slice, radial_range = _slices(s, o, per)
    data = _slice_radial_integrate(
        np.asarray(s.data, np.float32), factors=factors, factors_slice=factors_slice, slices=slices,
        npt_rad=o.npt, npt_azim=o.npt_azim, mask=cal.mask, mean=False,
    )
    data = np.clip(np.nan_to_num(np.asarray(data, np.float32)), 0, None) ** o.gamma
    r0, r1 = radial_range
    return {
        "data": data,
        "radius": {"offset": float(r0) / per, "scale": float(r1 - r0) / o.npt / per},
        "azimuth": {"offset": -np.pi, "scale": 2 * np.pi / o.npt_azim},
    }


# ---------- what orix draws, as numbers the app draws ----------


def _xy(vectors) -> tuple[np.ndarray, np.ndarray]:
    """Stereographic x, y (orix's projection of the upper hemisphere)."""
    from orix.projections import StereographicProjection

    x, y = StereographicProjection().vector2xy(vectors)
    return np.asarray(x, float).ravel(), np.asarray(y, float).ravel()


def _uvw_labels(vectors, phase) -> list[str]:
    """Crystal directions of ``vectors`` as orix labels them (indices up to 2
    for the sector's corners, as ``_get_ipf_axes_labels``)."""
    from orix.plot.inverse_pole_figure_plot import _get_ipf_axes_labels

    return [s.replace("$", "").replace("\\bar{", "-").replace("}", "").strip("[]")
            for s in _get_ipf_axes_labels(vectors, symmetry=phase.point_group)]


def colour_key_png(phase) -> tuple[str, list[float]]:
    """orix's IPF colour key (``DirectionColorKeyTSL``, as pyxem's IPF
    markers use) as a PNG data URL, and its extent [x0, x1, y0, y1] in the
    stereographic projection (first row at y1)."""
    import base64
    import io

    from matplotlib.image import imsave
    from orix.plot import DirectionColorKeyTSL

    grid, ext = DirectionColorKeyTSL(phase.point_group)._create_rgba_grid(return_extent=True)
    buf = io.BytesIO()
    imsave(buf, np.clip(np.asarray(grid, float), 0, 1), format="png")
    (x0, x1), (y0, y1) = ext
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), [float(x0), float(x1), float(y0), float(y1)]


def ipf_sector(phase) -> dict:
    """The inverse pole figure of ``phase`` as orix draws it: the fundamental
    sector's outline (``x``, ``y``), its corners (``x``, ``y``, ``label``) and
    the colour key (``key``: PNG data URL, ``extent``)."""
    from orix.vector.fundamental_sector import _closed_edges_in_hemisphere

    sector = phase.point_group.fundamental_sector
    ex, ey = _xy(_closed_edges_in_hemisphere(sector.edges, sector))
    vx, vy = _xy(sector.vertices)
    key, extent = colour_key_png(phase)
    return {
        "outline": {"x": ex.round(5).tolist(), "y": ey.round(5).tolist()},
        "corners": [{"x": round(float(a), 5), "y": round(float(b), 5), "label": s}
                    for a, b, s in zip(vx, vy, _uvw_labels(sector.vertices, phase))],
        "key": key,
        "extent": extent,
        "point_group": phase.point_group.name,
    }


def template_directions(rotations, phase) -> dict:
    """Where each template's beam direction (rotation × z, as pyxem's IPF
    markers) falls in ``phase``'s inverse pole figure: ``x``, ``y`` and its
    IPF colour (``rgb``, 0–255) per template."""
    from orix.plot import DirectionColorKeyTSL
    from orix.vector import Vector3d

    v = (rotations * Vector3d.zvector()).in_fundamental_sector(phase.point_group)
    x, y = _xy(v)
    rgb = np.rint(np.asarray(DirectionColorKeyTSL(phase.point_group).direction2color(v)).reshape(-1, 3) * 255)
    return {"x": x.round(5).tolist(), "y": y.round(5).tolist(), "rgb": rgb.astype(int).tolist()}


def direction_of(rotation, phase) -> list[float]:
    """The crystal direction along the beam of a template (rotation × z) in
    ``phase``'s fundamental sector, as [u, v, w] (largest 1)."""
    from orix.vector import Miller, Vector3d

    v = (rotation * Vector3d.zvector()).in_fundamental_sector(phase.point_group)
    uvw = np.asarray(Miller(xyz=v.data, phase=phase).uvw, float).ravel()
    return (uvw / np.abs(uvw).max()).round(4).tolist()


def view_of(orientation, phase) -> dict:
    """How the crystal is seen in the pattern at ``orientation`` (orix, as
    pyxem's ``to_single_phase_orientations``): the direction towards the
    viewer (orientation × z) and the screen's up (orientation × −y): a
    reflection g is drawn at x = −g·(orientation × x) across and
    y = g·(orientation × y) down the rows. Each as [u, v, w] of ``phase``'s
    cell (largest 1), for the 3D view to look the same way. (Checked on a
    pattern off a zone axis, where a half turn about the beam isn't a
    symmetry of the crystal.)"""
    from orix.vector import Miller, Vector3d

    out = {}
    for key, xyz in (("view", (0, 0, 1)), ("up", (0, -1, 0))):
        v = orientation * Vector3d(xyz)
        uvw = np.asarray(Miller(xyz=v.data, phase=phase).uvw, float).ravel()
        out[key] = (uvw / np.abs(uvw).max()).round(5).tolist()
    return out


def _crystal_hkl(xy: np.ndarray, seen: dict, phase) -> np.ndarray:
    """The Miller indices of reflections drawn at ``xy`` (1/Å, x right, y
    down) when the crystal is seen as ``seen`` (``view_of``)."""
    A = np.asarray(phase.structure.lattice.base, float)  # rows a, b, c (Å)
    up = np.asarray(seen["up"], float) @ A
    view = np.asarray(seen["view"], float) @ A
    up, view = up / np.linalg.norm(up), view / np.linalg.norm(view)
    right = np.cross(up, view)
    g = np.outer(xy[:, 0], right) - np.outer(xy[:, 1], up)
    return np.rint(g @ A.T).astype(int)


def nearest_template(rotations, phase, uvw) -> int:
    """The template whose beam direction is nearest [uvw] (up to symmetry)."""
    from orix.vector import Miller, Vector3d

    target = Vector3d(Miller(uvw=list(uvw), phase=phase).data).in_fundamental_sector(phase.point_group).unit
    v = (rotations * Vector3d.zvector()).in_fundamental_sector(phase.point_group).unit
    return int(np.argmax(np.asarray(v.data) @ np.asarray(target.data).ravel()))


def template_pattern(sim) -> dict:
    """A single template's reflections (a diffsims simulation of one phase
    and rotation): ``x``, ``y`` (1/Å), ``intensity`` (brightest 1), ``hkl``."""
    c = sim.coordinates
    xy = np.asarray(c.data)[:, :2] if c.size else np.zeros((0, 2))
    inten = np.asarray(c.intensity, float).ravel()
    top = inten.max() if inten.size else 1.0
    hkl = np.rint(np.asarray(c.hkl)).astype(int).reshape(-1, 3)
    return {
        "x": xy[:, 0].round(6).tolist(), "y": xy[:, 1].round(6).tolist(),
        "intensity": (inten / top if top > 0 else inten).round(5).tolist(), "hkl": hkl.tolist(),
    }


def symmetry_poles(phase) -> dict:
    """orix's stereographic projection of the point group (``Symmetry.plot``):
    its operations applied to one general orientation, the poles of z on the
    upper hemisphere (``upper``) and those on the lower one reprojected
    (``lower``), each {x, y}."""
    from orix.quaternion import Rotation
    from orix.vector import Vector3d

    pg = phase.point_group
    general = Rotation.from_axes_angles((-1, 8, 1), np.deg2rad(65))
    v = np.asarray((pg.outer(general) * Vector3d.zvector()).data).reshape(-1, 3)
    up = v[:, 2] >= -1e-9
    ux, uy = _xy(Vector3d(v[up]))
    lx, ly = _xy(Vector3d(v[~up] * [1, 1, -1]))
    return {
        "name": pg.name,
        "upper": {"x": ux.round(5).tolist(), "y": uy.round(5).tolist()},
        "lower": {"x": lx.round(5).tolist(), "y": ly.round(5).tolist()},
    }


def template_toward(sim, phase, uvw) -> dict:
    """The template nearest the beam direction [uvw] (up to symmetry), its
    reflections taken onto [uvw]'s own zone by the symmetry operation that
    takes the template's beam direction nearest it: ``index``, ``angle``
    (degrees off [uvw]), ``hkl`` (taken onto it) and ``intensity``
    (brightest 1). Drawn turned as the crystal is, they are the pattern of
    the crystal seen down [uvw] (as near as the sampling allows)."""
    from orix.vector import Miller, Vector3d

    rotations = sim.rotations
    target = Vector3d(Miller(uvw=list(uvw), phase=phase).data).unit
    beams = (rotations * Vector3d.zvector()).unit
    pg = phase.point_group
    # Every template's beam under every symmetry operation, against [uvw].
    ops = np.asarray((pg.outer(beams)).data).reshape(pg.size, -1, 3)  # ops × templates × 3
    cos = ops @ np.asarray(target.data).ravel()
    op, index = np.unravel_index(int(np.argmax(cos)), cos.shape)
    angle = float(np.degrees(np.arccos(np.clip(cos[op, index], -1, 1))))
    one = sim.irot[int(index)] if rotations.size > 1 else sim
    c = one.coordinates
    inten = np.asarray(c.intensity, float).ravel()
    top = inten.max() if inten.size else 1.0
    hkl = np.asarray(c.hkl, float).reshape(-1, 3)
    if len(hkl):
        g = Vector3d(Miller(hkl=hkl, phase=phase).data)
        turned = Miller(xyz=np.asarray((pg[int(op)] * g).data), phase=phase)
        hkl = np.asarray(turned.hkl, float).reshape(-1, 3)
    return {
        "index": int(index),
        "angle": round(angle, 3),
        "hkl": np.rint(hkl).astype(int).tolist(),
        "intensity": (inten / top if top > 0 else inten).round(5).tolist(),
    }


def polar_scan(signal, axes, affine, o: PolarOptions, per: float, report=None):
    """The polar transform of every pattern of 4D-STEM ``signal`` (pattern
    axes ``axes``: [(scale, offset, units)] for ky, kx) as pyxem makes it for
    matching: a pyxem ``PolarDiffraction2D`` (the scan's axes; Radians and
    Radius in 1/Å, as pyxem's ``get_azimuthal_integral2d``)."""
    import pyxem
    from pyxem.utils._azimuthal_integrations import _slice_radial_integrate

    ny, nx, ky, kx = signal.data.shape
    s0 = diffraction2d(np.zeros((ky, kx), np.float32), axes, affine)
    slices, factors, factors_slice, (r0, r1) = _slices(s0, o, per)
    mask = s0.calibration.mask
    out = np.empty((ny, nx, o.npt, o.npt_azim), np.float32)
    for y in range(ny):
        rows = np.asarray(signal.data[y], np.float32)
        for x in range(nx):
            out[y, x] = _slice_radial_integrate(
                rows[x], factors=factors, factors_slice=factors_slice, slices=slices,
                npt_rad=o.npt, npt_azim=o.npt_azim, mask=mask, mean=False,
            )
        if report and y % 4 == 0:
            report(0.05 + 0.9 * (y + 1) / ny, f"Row {y + 1} of {ny}")
    np.nan_to_num(out, copy=False)
    np.clip(out, 0, None, out=out)
    if o.gamma != 1:
        out **= o.gamma
    polar = pyxem.signals.PolarDiffraction2D(out)
    for new, old in zip(polar.axes_manager.navigation_axes, signal.axes_manager.navigation_axes):
        new.name, new.scale, new.offset, new.units = old.name, old.scale, old.offset, old.units
    az, rad = polar.axes_manager.signal_axes
    az.name, az.units, az.offset, az.scale = "Radians", "Rad", -np.pi, 2 * np.pi / o.npt_azim
    rad.name, rad.units, rad.offset, rad.scale = "Radius", "1/Å", float(r0) / per, float(r1 - r0) / o.npt / per
    polar.metadata.General.title = signal.metadata.General.title
    polar.metadata.set_item("Processing.si_viewer.polar", o.model_dump())
    return polar


# ---------- matching (pyxem's get_orientation) ----------


class MatchOptions(BaseModel):
    """``PolarDiffraction2D.get_orientation``: how many results are kept per
    pattern (-1: all, as the tutorial does at a few positions) and the
    fraction of templates fully matched after the fast 1D pass."""

    n_best: int = Field(-1, ge=-1)
    frac_keep: float = Field(1.0, gt=0, le=1)


def polar_patterns(polar_signal, positions) -> object:
    """The polar patterns at scan ``positions`` ([(x, y)]) of a polar
    transform (pyxem ``PolarDiffraction2D``), as one with a 1D navigation."""
    import pyxem

    data = np.stack([np.asarray(polar_signal.data[y, x], np.float32) for x, y in positions])
    out = pyxem.signals.PolarDiffraction2D(data)
    for new, old in zip(out.axes_manager.signal_axes, polar_signal.axes_manager.signal_axes):
        new.name, new.scale, new.offset, new.units = old.name, old.scale, old.offset, old.units
    return out


def match(polar, sim, o: MatchOptions, phase=None) -> list[dict]:
    """pyxem's template matching of ``polar`` (``PolarDiffraction2D``, 1D
    navigation) against ``sim``: per pattern, the best match (template
    ``index``, ``correlation``, in-plane ``rotation`` in degrees, ``mirror``,
    Bunge ``euler`` in degrees), its reflections as pyxem draws them
    (``vectors``: ``x``, ``y`` in 1/Å on the pattern's axes, ``intensity``,
    ``hkl``; ``polar``: ``theta`` in radians, ``r`` in 1/Å, as
    ``to_polar_markers``), and the correlation of every template matched
    (``correlations``, by template index; null where not matched)."""
    return describe(polar.get_orientation(sim, n_best=o.n_best, frac_keep=o.frac_keep), sim, phase)


def describe(res, sim, phase=None) -> list[dict]:
    """``match``'s description of each pattern of an ``OrientationMap``
    (pyxem's, 1D navigation) against ``sim``."""
    data = np.asarray(res.data, float)  # patterns × kept × [index, correlation, rotation, factor]
    oris = res.to_single_phase_orientations()
    euler = np.degrees(np.asarray(oris.to_euler())).reshape(data.shape[0], -1, 3)[:, 0]
    vectors = res.to_vectors(n_best_index=0)
    n = sim.rotations.size
    out = []
    for i in range(data.shape[0]):
        v = vectors.data[i]
        xy = np.asarray(v.data, float).reshape(-1, 3)[:, :2]
        inten = np.asarray(v.intensity, float).ravel()
        top = inten.max() if inten.size else 1.0
        corr = [None] * n
        ranked = []  # every template kept, best first (as pyxem orders them)
        for idx, c, rot, factor in data[i]:
            if np.isfinite(c) and 0 <= int(idx) < n:
                corr[int(idx)] = round(float(c), 6)
                ranked.append({"index": int(idx), "correlation": round(float(c), 6), "rotation": float(rot), "mirror": bool(factor < 0)})
        best = data[i, 0]
        seen = view_of(oris[i, 0], phase) if phase is not None else {}
        out.append({
            "best": {
                "index": int(best[0]), "correlation": float(best[1]), "rotation": float(best[2]),
                "mirror": bool(best[3] < 0), "euler": [round(float(a), 3) for a in euler[i]], **seen,
            },
            "vectors": {
                "x": xy[:, 0].round(6).tolist(), "y": xy[:, 1].round(6).tolist(),
                "intensity": (inten / top if top > 0 else inten).round(5).tolist(),
                # pyxem's are the turned template's, rounded: the crystal's
                # come from where each is seen (the orientation's view).
                "hkl": (_crystal_hkl(xy, seen, phase) if seen else np.rint(np.asarray(v.hkl, float)).astype(int).reshape(-1, 3)).tolist(),
            },
            "polar": {
                "theta": (-np.arctan2(xy[:, 1], xy[:, 0])).round(6).tolist(),
                "r": np.hypot(xy[:, 0], xy[:, 1]).round(6).tolist(),
            },
            "correlations": corr,
            "ranked": ranked,
        })
    return out


def describe_one(sim, phase, index: int, rotation: float, mirror: bool, correlation: float = 0.0) -> dict:
    """``match``'s description of one template at an in-plane rotation (one
    of a match's ``ranked``): its reflections on the pattern and its view."""
    res = orientation_map(np.array([[[index, correlation, rotation, -1.0 if mirror else 1.0]]]), sim)
    m = describe(res, sim, phase)[0]
    del m["correlations"], m["ranked"]
    return m


def template_uvws(rotations, phase) -> np.ndarray:
    """Every template's beam direction as ``direction_of`` gives it: [n, 3]."""
    from orix.vector import Miller, Vector3d

    v = (rotations * Vector3d.zvector()).in_fundamental_sector(phase.point_group)
    uvw = np.asarray(Miller(xyz=v.data, phase=phase).uvw, float).reshape(-1, 3)
    return (uvw / np.abs(uvw).max(axis=1, keepdims=True)).round(4)


# ---------- the whole scan: the orientation map (pyxem's OrientationMap) ----------


class MapOptions(BaseModel):
    """``get_orientation`` over the whole scan, as the tutorial does: the best
    match kept per pattern, the full match on the half of the templates that
    did best in the fast 1D pass."""

    n_best: int = Field(1, ge=1, le=10)
    frac_keep: float = Field(0.5, gt=0, le=1)


def match_scan(polar, sim, o: MapOptions, report=None, rows: int = 8) -> np.ndarray:
    """pyxem's ``get_orientation`` of every pattern of ``polar`` (a
    ``PolarDiffraction2D`` with a 2D scan) against ``sim``, a block of
    ``rows`` scan rows at a time (``report(fraction, message)`` after each,
    which may raise to cancel): float32 [y, x, n_best, 4] (template index,
    correlation, in-plane rotation °, mirror factor), as OrientationMap holds."""
    import pyxem

    if not isinstance(polar, pyxem.signals.PolarDiffraction2D):  # e.g. opened from an analysis
        data = polar
        polar = pyxem.signals.PolarDiffraction2D(np.asarray(data.data))
        for new, old in zip(polar.axes_manager._axes, data.axes_manager._axes):
            new.name, new.scale, new.offset, new.units = old.name, old.scale, old.offset, old.units
    ny = polar.axes_manager.navigation_shape[1]
    parts = []
    for y0 in range(0, ny, rows):
        block = polar.inav[:, y0:min(ny, y0 + rows)]
        res = block.get_orientation(sim, n_best=o.n_best, frac_keep=o.frac_keep)
        data = res.data.compute() if hasattr(res.data, "compute") else res.data
        parts.append(np.asarray(data, np.float32).reshape(block.data.shape[0], block.data.shape[1], o.n_best, 4))
        if report:
            report(min(ny, y0 + rows) / ny, f"Rows {min(ny, y0 + rows)} of {ny}")
    return np.concatenate(parts, axis=0)


def orientation_map(results: np.ndarray, sim, scan_axes=None):
    """pyxem's ``OrientationMap`` of ``results`` (``match_scan``) against
    ``sim``, on ``scan_axes`` ([(name, scale, offset, units)] x then y)."""
    from pyxem.signals import OrientationMap

    omap = OrientationMap(np.asarray(results, float))
    omap.metadata.set_item("simulation", sim)
    for axis, (name, scale, offset, units) in zip(omap.axes_manager.navigation_axes, scan_axes or []):
        axis.name, axis.scale, axis.offset, axis.units = name, scale, offset, units
    return omap


IPF_DIRECTIONS = ("x", "y", "z")


def ipf_colours(omap, direction: str) -> np.ndarray:
    """The best orientation at each position coloured by orix's
    ``IPFColorKeyTSL`` along the sample's ``direction`` ("x", "y", "z": z
    along the beam), as pyxem's ``to_ipf_colormap``: float32 [y, x, 3], 0–1."""
    from orix.plot import IPFColorKeyTSL
    from orix.vector import Vector3d

    oris = omap.to_single_phase_orientations()[:, :, 0]
    v = {"x": Vector3d.xvector(), "y": Vector3d.yvector(), "z": Vector3d.zvector()}[direction]
    rgb = IPFColorKeyTSL(oris.symmetry, v).orientation2color(oris)
    return np.asarray(rgb, np.float32).reshape(*oris.shape, 3)


def map_at(results: np.ndarray, sim, phase, positions) -> list[dict]:
    """The orientation map ``results`` (``match_scan``) at scan ``positions``
    ([(x, y)]), described as ``match`` describes a match (the best template's
    reflections, labelled; the view): the tutorial's ``to_markers``."""
    picked = np.stack([np.asarray(results[y, x], float) for x, y in positions])
    out = describe(orientation_map(picked, sim), sim, phase)
    for m, (x, y) in zip(out, positions):
        m["x"], m["y"] = int(x), int(y)
        del m["correlations"], m["ranked"]  # only the kept ones: see the Match tab for all
    return out


# ---------- several phases (a phase map) ----------

# A colour per phase (matplotlib's tab10, as orix's default phase colours).
PHASE_COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


def phase_blocks(results: np.ndarray) -> np.ndarray:
    """An orientation map's results as [phases, y, x, n_best, 4] (a
    single-phase map, [y, x, n_best, 4], as one phase)."""
    r = np.asarray(results)
    return r[None] if r.ndim == 4 else r


def best_phase(results: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each position's phase (the one whose best template correlates best,
    as pyxem's multi-phase matching and the two-phase SPED work choose it):
    the phase index [y, x] and that phase's results [y, x, n_best, 4]."""
    blocks = phase_blocks(results)
    which = np.argmax(blocks[..., 0, 1], axis=0)
    picked = np.take_along_axis(blocks, which[None, ..., None, None], axis=0)[0]
    return which.astype(np.int32), picked


def combine_matches(per_phase: list[list[dict]], phases: list[dict]) -> list[dict]:
    """Matches of the same positions against several phases (``phases``:
    [{name, templates}], in order) as one per position: the phase whose best
    match correlates best, with ``phase`` (its name), ``phase_index``,
    ``templates`` (its Templates object) and ``phases`` (every phase's best:
    name, templates, correlation, uvw)."""
    out = []
    for i in range(len(per_phase[0])):
        rows = [p[i] for p in per_phase]
        k = int(np.argmax([r["best"]["correlation"] for r in rows]))
        m = dict(rows[k])
        m["phase"], m["phase_index"], m["templates"] = phases[k]["name"], k, phases[k]["templates"]
        m["phases"] = [{"name": ph["name"], "templates": ph["templates"], "correlation": r["best"]["correlation"],
                        "uvw": r["best"].get("uvw")} for ph, r in zip(phases, rows)]
        # Every phase's templates in one list, best first, each with its phase's index.
        m["ranked"] = sorted(({**row, "phase": j} for j, r in enumerate(rows) for row in r.get("ranked", [])),
                             key=lambda row: -row["correlation"])
        out.append(m)
    return out
