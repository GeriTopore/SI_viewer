"""4D-STEM: the pattern at a scan position or over a region, virtual images
through a detector, and virtual images kept as datasets."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from ..processing import fourd
from ..processing.pattern_background import PatternBackgroundOptions
from ..processing.calibration import RingOptions
from ..processing.centring import BeamOptions, PlaneOptions
from ..processing.disks import DiskOptions
from ..processing.strain import BvmPeakOptions, FitOptions, StrainAxes, StrainLattice
from .. import session as _session
from ..session import DatasetKind
from .datasets import array_response

router = APIRouter(tags=["4dstem"])


def _cube(request: Request, dataset_id: str):
    try:
        ds = request.app.state.session.get_dataset(dataset_id)
    except KeyError:
        raise HTTPException(404, f"Unknown dataset id: {dataset_id}")
    if ds.kind is not DatasetKind.DIFFRACTION4D:
        raise HTTPException(400, f"{ds.title} is not 4D-STEM data")
    return ds


@router.get("/api/datasets/{dataset_id}/4d/pattern")
def pattern(dataset_id: str, request: Request, x: float, y: float) -> Response:
    """The diffraction pattern at scan pixel (``x``, ``y``)."""
    ds = _cube(request, dataset_id)
    try:
        return array_response(fourd.pattern_at(ds.signal.data, x, y), x=int(x), y=int(y))
    except IndexError as e:
        raise HTTPException(416, str(e))


class RegionBody(BaseModel):
    region: fourd.Region
    reduce: Literal["mean", "max"] = "mean"


@router.post("/api/datasets/{dataset_id}/4d/pattern")
def region_pattern(dataset_id: str, body: RegionBody, request: Request) -> Response:
    """The mean or maximum pattern over a region (rectangle, ellipse or
    polygon) of the scan."""
    ds = _cube(request, dataset_id)
    try:
        out, n = fourd.pattern_over(ds.signal.data, body.region, body.reduce)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return array_response(out, positions=n)


@router.get("/api/datasets/{dataset_id}/4d/bvm-pattern")
def bvm_pattern(dataset_id: str, request: Request) -> Response:
    """The Bragg vector map on the pattern's own pixels (the pattern's size,
    so detectors drawn on it are where they are on the patterns): every disk
    found, each position's moved by its origin's offset from the mean origin
    when the centring is switched on (as a digital dark field does)."""
    from ..processing import disks

    ds = _cube(request, dataset_id)
    table, centred = _centred_table(request, ds)
    data = disks.vector_map(table, tuple(ds.signal.data.shape[2:]), upsample=1)
    return array_response(data, disks=len(table), centred=centred)


def _centred_table(request: Request, ds) -> tuple[np.ndarray, bool]:
    """The Bragg vectors table on the pattern's pixels, each position's disks
    moved by its origin's offset from the mean origin while the centring is
    on (as a digital dark field does); and whether they were."""
    from ..processing import disks

    table = _table(request, ds)
    origins = request.app.state.session.centring_origins(ds)
    if origins is None:
        return table, False
    mean = np.nanmean(origins.reshape(-1, 2), axis=0)
    q = disks.calibrated(table, origins, (0, 0)) + mean
    return np.column_stack([table[:, :2], q[:, 1], q[:, 0], table[:, 4]]).astype(np.float32), True


class BvmPeaksBody(BaseModel):
    options: BvmPeakOptions = BvmPeakOptions()


@router.post("/api/datasets/{dataset_id}/4d/strain/bvm-peaks")
def strain_bvm_peaks(dataset_id: str, body: BvmPeaksBody, request: Request) -> dict:
    """The Bragg vector map's peaks (on the pattern's pixels, as ``bvm-pattern``
    shows it): ``{peaks: [[x, y, intensity], …] brightest first, lattice:
    {ox, oy, g1x, g1y, g2x, g2y} proposed (None if it can't be; O at the
    peak nearest the direct beam: the mean origin while centred, else the
    mean pattern's bright-field disk), centred}``."""
    from ..processing import strain
    from ..processing.frames import get_frame

    session = request.app.state.session
    ds = _cube(request, dataset_id)
    table, centred = _centred_table(request, ds)
    peaks = strain.bvm_peaks(table, tuple(ds.signal.data.shape[2:]), body.options)
    origins = session.centring_origins(ds)
    if origins is not None:
        centre = tuple(float(v) for v in np.nanmean(origins.reshape(-1, 2), axis=0))
    else:
        try:
            bf = fourd.bright_field_disk(get_frame(ds, projection="mean-pattern"))
            centre = (bf["cx"], bf["cy"])
        except ValueError:
            h, w = ds.signal.data.shape[2:]
            centre = (w / 2, h / 2)
    return {
        "peaks": np.round(peaks, 3).tolist(),
        "lattice": strain.guess_lattice(peaks, centre),
        "centred": centred,
    }


class StrainFitBody(BaseModel):
    lattice: StrainLattice
    options: FitOptions = FitOptions()


def _strain_frame(session, ds) -> tuple[np.ndarray, dict]:
    """The calibrated frame the lattice is fitted in (the ellipse and the QR
    rotation where switched on), and what it is."""
    from ..processing import strain

    cal = ds.calibration or {}
    on = cal.get("enabled", {})
    ring = cal.get("ring") if on.get("ellipse") else None
    rotation = cal.get("rotation") if on.get("rotation") else None
    return strain.frame_matrix(ring, rotation), {
        "centred": session.centring_origins(ds) is not None, "ellipse": ring is not None,
        "rotation": {"theta": rotation["theta"], "mirror": bool(rotation.get("mirror"))} if rotation else None,
    }


@router.post("/api/datasets/{dataset_id}/4d/strain/fit")
def strain_fit(dataset_id: str, body: StrainFitBody, request: Request) -> dict:
    """Index every pattern's disks to the lattice and fit it at every scan
    position (as a job; py4DSTEM's indexing and weighted fit), kept as the
    "Lattice fit" stack with "Indexed disks" and "Fit residual" under the
    Bragg vectors. The job's result is ``{file, ids, fitted, positions,
    indexed, disks}``."""
    from ..processing import strain

    session = request.app.state.session
    ds = _cube(request, dataset_id)
    table, _ = _centred_table(request, ds)
    frame, how = _strain_frame(session, ds)
    ny, nx = ds.signal.data.shape[:2]

    def run(job) -> dict:
        job.report(0.1, "Indexing the disks and fitting the lattice")
        try:
            result = strain.fit_lattice(table, (ny, nx), body.lattice, body.options, frame)
        except ValueError as e:
            raise RuntimeError(str(e))
        job.report(0.9, "Making the maps")
        meta = {"lattice": body.lattice.model_dump(), "options": body.options.model_dump(), "frame": how}
        made = session.set_strain_fit(ds, result, meta)
        fitted = int(np.isfinite(result["fit"][0]).sum())
        return {"file": session.file_of(ds).summary(session), "ids": [d.id for d in made], "fitted": fitted,
                "positions": int(ny * nx), "indexed": int(result["count"].sum()), "disks": int(len(table))}

    return request.app.state.jobs.submit("strain-fit", f"Lattice fit of {ds.title}", run).summary()


class StrainReferenceBody(BaseModel):
    # A scan region the reference is the median over; None: the whole scan.
    region: fourd.Region | None = None
    axes: StrainAxes = StrainAxes()


def _strain_reference(request: Request, ds, body: StrainReferenceBody):
    """The data's Lattice fit, the reference g1 and g2 (rows) and how many
    positions it is from, and the maps' x axis."""
    from ..processing import strain
    from ..processing.regions import region_mask

    session = request.app.state.session
    fit = session.strain_fit_of(ds)
    if fit is None:
        raise HTTPException(422, "Fit the lattice first (Fit every pattern)")
    data = np.asarray(fit.signal.data)
    mask = None
    if body.region is not None:
        ny, nx = data.shape[1:]
        rows, cols, inside = region_mask((ny, nx), body.region)
        mask = np.zeros((ny, nx), dtype=bool)
        mask[rows, cols] = inside
    try:
        ref, n = strain.reference_basis(data, mask)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return fit, data, ref, n, strain.axis_direction(body.axes, ref)


def _reference_summary(ref: np.ndarray, n: int, x_axis: tuple[float, float]) -> dict:
    return {"g1": [float(v) for v in ref[0]], "g2": [float(v) for v in ref[1]], "positions": n,
            # The x axis counter-clockwise on screen from the frame's x (degrees).
            "x_angle": math.degrees(math.atan2(-x_axis[1], x_axis[0]))}


@router.post("/api/datasets/{dataset_id}/4d/strain/reference")
def strain_reference(dataset_id: str, body: StrainReferenceBody, request: Request) -> dict:
    """The unstrained lattice (step 3): the median of the fitted g1 and g2
    over the scan or a region, ``{g1: [x, y], g2: [x, y]}`` (calibrated
    frame, pattern px), ``positions`` it is from, and the maps' x axis
    (``x_angle``)."""
    _, _, ref, n, x_axis = _strain_reference(request, _cube(request, dataset_id), body)
    return _reference_summary(ref, n, x_axis)


@router.post("/api/datasets/{dataset_id}/4d/strain/maps")
def strain_maps(dataset_id: str, body: StrainReferenceBody, request: Request) -> dict:
    """εxx, εyy, εxy and ωxy against the reference, on the axes (step 4; a
    job, by pyxem), kept as images under the Lattice fit. The job's result
    is ``{file, ids, reference}``."""
    from ..processing import strain

    session = request.app.state.session
    ds = _cube(request, dataset_id)
    fit, data, ref, n, x_axis = _strain_reference(request, ds, body)
    reference = _reference_summary(ref, n, x_axis)
    meta = {"reference": {**reference, "region": body.region.model_dump() if body.region else None},
            "axes": body.axes.model_dump(), "fit": {k: fit.strain.get(k) for k in ("lattice", "options", "frame")}}

    def run(job) -> dict:
        job.report(0.1, "Strain from the lattice fit (pyxem)")
        maps = strain.strain_maps(data, ref, x_axis)
        job.report(0.9, "Making the maps")
        made = session.set_strain_maps(ds, maps, meta)
        return {"file": session.file_of(ds).summary(session), "ids": [d.id for d in made], "reference": reference}

    return request.app.state.jobs.submit("strain-maps", f"Strain maps of {ds.title}", run).summary()


class StrainPreviewBody(BaseModel):
    lattice: StrainLattice
    options: FitOptions = FitOptions()
    # Scan positions (px) whose patterns are shown.
    positions: list[tuple[int, int]] = Field(default_factory=list, max_length=64)


@router.post("/api/datasets/{dataset_id}/4d/strain/preview")
def strain_preview(dataset_id: str, body: StrainPreviewBody, request: Request) -> dict:
    """The indexing and fit at a few scan positions, on their own patterns
    (pattern px, not centred): ``{"x,y": {"disks": [[x, y, h, k, indexed], …],
    "points": [[x, y, h, k], …] the fitted lattice, "count", "residual"}}``
    (no "points" where it can't be fitted)."""
    from ..processing import strain

    session = request.app.state.session
    ds = _cube(request, dataset_id)
    table, _ = _centred_table(request, ds)
    raw = _table(request, ds)
    frame, _ = _strain_frame(session, ds)
    ny, nx = ds.signal.data.shape[:2]
    h, w = ds.signal.data.shape[2:]
    inv = np.linalg.inv(frame)
    lat = body.lattice
    out = {}
    for x, y in body.positions:
        if not (0 <= x < nx and 0 <= y < ny):
            continue
        sel = (np.floor(table[:, 0]) == y) & (np.floor(table[:, 1]) == x)
        rows, rows_raw = table[sel], raw[sel]
        # Where this pattern's disks are on the map, less where they are on the pattern.
        shift = (np.column_stack([rows_raw[:, 3], rows_raw[:, 2]]) - np.column_stack([rows[:, 3], rows[:, 2]])).mean(
            axis=0) if len(rows) else np.zeros(2)
        entry: dict = {"disks": [], "count": 0, "residual": None}
        if len(rows):
            try:
                _, hk, ok = strain.index_disks(rows, lat, body.options.tolerance, frame)
            except ValueError as e:
                raise HTTPException(422, str(e))
            entry["disks"] = [[round(float(r[3]), 2), round(float(r[2]), 2), int(a), int(b), bool(k)]
                              for r, (a, b), k in zip(rows_raw, hk, ok)]
            one = rows.copy()
            one[:, 0], one[:, 1] = 0, 0
            res = strain.fit_lattice(one, (1, 1), lat, body.options, frame)
            entry["count"] = int(res["count"][0, 0])
            fit = res["fit"][:, 0, 0]
            if np.isfinite(fit).all():
                entry["residual"] = float(res["residual"][0, 0])
                g = np.array([[fit[0], fit[2]], [fit[1], fit[3]]])
                o = np.array([fit[4], fit[5]])
                hks = sorted({(int(a), int(b)) for a, b in hk[ok]})
                pts = []
                for a in range(min(p[0] for p in hks) - 1, max(p[0] for p in hks) + 2):
                    for b in range(min(p[1] for p in hks) - 1, max(p[1] for p in hks) + 2):
                        p = inv @ (o + g @ (a, b)) + (lat.ox, lat.oy) + shift
                        if 0 <= p[0] <= w and 0 <= p[1] <= h:
                            pts.append([round(float(p[0]), 2), round(float(p[1]), 2), a, b])
                entry["points"] = pts
        out[f"{x},{y}"] = entry
    return out


class VirtualBody(BaseModel):
    detector: fourd.Detector
    response: fourd.Response = "sum"
    # "pixels": the patterns' pixels inside the detector; "vectors": a
    # digital dark field from the Bragg vectors (the disks inside it).
    source: Literal["pixels", "vectors"] = "pixels"
    # Digital dark field: the disks' intensity, or their number.
    weight: Literal["intensity", "count"] = "intensity"


def _scale(ds) -> float:
    """The pattern's pixel size (its calibration; 1 uncalibrated)."""
    return float(ds.signal.axes_manager.signal_axes[0].scale or 1.0)


def _ddf(request: Request, ds, body: VirtualBody) -> np.ndarray:
    """A digital dark field (``disks.ddf_image``) from the data's Bragg
    vectors, centred per position when its centring is switched on."""
    from ..processing import disks

    if body.response in fourd.COM_RESPONSES:
        raise HTTPException(422, "A centre of mass is of the patterns' pixels, not of the disks found")
    session = request.app.state.session
    table = session.vectors_table(ds)
    if table is None:
        raise HTTPException(409, f"The disks of {ds.title} haven't been found (Disk detection)")
    return disks.ddf_image(table, ds.signal.data.shape[:2], body.detector, body.response, body.weight,
                           session.centring_origins(ds))


def _image(request: Request, ds, body: VirtualBody) -> np.ndarray:
    if body.source == "vectors":
        return _ddf(request, ds, body)
    return fourd.virtual_image(ds.signal.data, body.detector, body.response, _scale(ds))


@router.post("/api/datasets/{dataset_id}/4d/virtual")
def virtual(dataset_id: str, body: VirtualBody, request: Request) -> Response:
    """The virtual image through ``detector``: per scan position, the sum,
    mean or maximum of the pattern pixels inside it (or, ``source``
    "vectors", of the disks found inside it: a digital dark field)."""
    ds = _cube(request, dataset_id)
    image = _image(request, ds, body)
    pixels = int(fourd.detector_mask(ds.signal.data.shape[2:], body.detector).sum())
    return array_response(image, pixels=pixels)


class VirtualsBody(BaseModel):
    items: list[VirtualBody]


@router.post("/api/datasets/{dataset_id}/4d/virtuals")
def virtuals(dataset_id: str, body: VirtualsBody, request: Request) -> Response:
    """The virtual images through several detectors at once (in the order
    given; the pixel ones in one pass over the data), their channels one
    after the other: [channels, y, x], ``channels`` (meta) per item (1; a
    "com" response 2: direction, magnitude)."""
    ds = _cube(request, dataset_id)
    out: list[np.ndarray | None] = [None] * len(body.items)
    pixel = [k for k, i in enumerate(body.items) if i.source == "pixels"]
    if pixel:
        made = fourd.virtual_images(
            ds.signal.data, [(body.items[k].detector, body.items[k].response) for k in pixel], _scale(ds),
        )
        for k, image in zip(pixel, made):
            out[k] = image
    for k, item in enumerate(body.items):
        if item.source == "vectors":
            out[k] = _ddf(request, ds, item)
    parts = [o.reshape(-1, *o.shape[-2:]) for o in out]
    stacked = np.concatenate(parts) if parts else np.zeros((0, *ds.signal.data.shape[:2]), np.float32)
    return array_response(stacked, channels=[len(p) for p in parts])


class SaveBody(VirtualBody):
    title: str | None = None


@router.post("/api/datasets/{dataset_id}/4d/virtual/save")
def save_virtual(dataset_id: str, body: SaveBody, request: Request) -> dict:
    """The virtual image through ``detector`` as an image dataset under the
    4D-STEM data. Returns its ``file`` and ``id``."""
    session = request.app.state.session
    ds = _cube(request, dataset_id)
    axis = ds.signal.axes_manager.signal_axes[0]
    if body.response == "com" and body.source == "pixels":
        # A colour image: a DPC object (direction, magnitude) with its COM x and y under it.
        rx, ry = fourd.com_field(ds.signal.data, body.detector, _scale(ds))
        title = (body.title or "").strip() or fourd.describe(body.detector, "com", axis.scale, axis.units or "px")
        made = session.add_com_colour(ds, rx, ry, title, {"detector": body.detector.model_dump()})
        return {"file": session.files[ds.file_id].summary(session), "id": made.id}
    image = _image(request, ds, body)
    title = (body.title or "").strip() or fourd.describe(body.detector, body.response, axis.scale, axis.units or "px")
    ddf = body.source == "vectors"
    if ddf and not (body.title or "").strip():
        title = f"DDF {title}" + (" (count)" if body.weight == "count" else "")
    how = {"source": "vectors", "weight": body.weight} if ddf else None
    try:
        made = session.add_virtual_image(ds, image, body.detector.model_dump(), body.response, title, how)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"file": session.files[ds.file_id].summary(session), "id": made.id}


@router.get("/api/datasets/{dataset_id}/4d/bf-disk")
def bf_disk(dataset_id: str, request: Request) -> dict:
    """The bright-field disk of the mean pattern: centre and radius (pattern pixels)."""
    from ..processing.frames import get_frame

    ds = _cube(request, dataset_id)
    try:
        return fourd.bright_field_disk(get_frame(ds, projection="mean-pattern"))
    except ValueError as e:
        raise HTTPException(422, str(e))


class BadPixelBody(BaseModel):
    # Hot: brighter than the 8 around by this × the mean pattern's mean intensity (0: off).
    hot: float = Field(5.0, ge=0, le=1e6)
    # Dead: dimmer than this × the median of the 8 around (0: off).
    dead: float = Field(0.2, ge=0, le=1)


def _bad(ds, body: BadPixelBody):
    from ..processing.frames import get_frame

    found = fourd.bad_pixels(get_frame(ds, projection="mean-pattern"), body.hot, body.dead)
    both = found["hot"] | found["dead"]
    return found, fourd.interior(both), int(both.sum() - fourd.interior(both).sum())


@router.post("/api/datasets/{dataset_id}/4d/bad-pixels")
def find_bad_pixels(dataset_id: str, body: BadPixelBody, request: Request) -> dict:
    """The detector's hot and dead pixels in the mean pattern ([y, x] each),
    and how many are on the border (left as they are by the repair)."""
    ds = _cube(request, dataset_id)
    found, _, border = _bad(ds, body)
    as_list = lambda m: [[int(y), int(x)] for y, x in zip(*m.nonzero())]  # noqa: E731
    return {"hot": as_list(found["hot"]), "dead": as_list(found["dead"]), "border": border}


class RepairBody(BadPixelBody):
    in_place: bool = True


@router.post("/api/datasets/{dataset_id}/4d/bad-pixels/repair")
def repair_bad_pixels(dataset_id: str, body: RepairBody, request: Request) -> dict:
    """Replace the hot and dead pixels (not on the border) in every pattern
    with the mean of the 8 around them, in place or as a copy under the data.
    Returns the ``file``, the ``id`` of the data repaired and how many pixels."""
    session = request.app.state.session
    ds = _cube(request, dataset_id)
    _, mask, _ = _bad(ds, body)
    if not mask.any():
        raise HTTPException(422, "No hot or dead pixels to repair")
    try:
        made = session.repair_pixels(ds, mask, body.in_place)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"file": session.file_of(ds).summary(session), "id": made.id, "pixels": int(mask.sum())}


# ---------- background subtraction (pyxem's methods) ----------


class BackgroundPreviewBody(BaseModel):
    options: PatternBackgroundOptions = PatternBackgroundOptions()
    # The pattern: at scan pixel (x, y), the mean over a region, or over the scan.
    x: float | None = None
    y: float | None = None
    region: fourd.Region | None = None
    scan: Literal["mean", "max"] | None = None


def _beam_centres(request: Request, ds):
    """The beam's centre per scan position ([y, x, 2] of x, y px) when the
    Centring is on, else that of the mean pattern's bright-field disk ([2])."""
    from ..processing.frames import get_frame

    origins = request.app.state.session.centring_origins(ds)
    if origins is not None:
        return origins
    try:
        disk = fourd.bright_field_disk(get_frame(ds, projection="mean-pattern"))
        return np.array([disk["cx"], disk["cy"]], float)
    except ValueError:
        h, w = ds.signal.data.shape[2:]
        return np.array([w / 2, h / 2], float)


@router.post("/api/datasets/{dataset_id}/4d/background/preview")
def background_preview(dataset_id: str, body: BackgroundPreviewBody, request: Request) -> Response:
    """One pattern (a scan pixel's, a region's mean, the scan's mean or max)
    with its background subtracted by ``options``, to tune them."""
    from ..processing import pattern_background as background
    from ..processing.frames import get_frame

    ds = _cube(request, dataset_id)
    centres = _beam_centres(request, ds)
    try:
        if body.x is not None and body.y is not None:
            pattern = fourd.pattern_at(ds.signal.data, body.x, body.y)
            centre = centres if centres.ndim == 1 else centres[int(body.y), int(body.x)]
        else:
            if body.region is not None:
                pattern, _ = fourd.pattern_over(ds.signal.data, body.region, "mean")
            else:
                pattern = get_frame(ds, projection="max-pattern" if body.scan == "max" else "mean-pattern")
            centre = centres if centres.ndim == 1 else np.nanmean(centres.reshape(-1, 2), axis=0)
    except IndexError as e:
        raise HTTPException(416, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))
    return array_response(background.subtract(pattern, body.options, (float(centre[0]), float(centre[1]))))


class BackgroundBody(BaseModel):
    options: PatternBackgroundOptions = PatternBackgroundOptions()
    in_place: bool = True


@router.post("/api/datasets/{dataset_id}/4d/background/run")
def background_run(dataset_id: str, body: BackgroundBody, request: Request) -> dict:
    """Subtract the background of every pattern (as a job), in place or as
    a copy under the data. The job's result is ``{file, id}`` (the data changed)."""
    from ..processing import pattern_background as background

    session = request.app.state.session
    ds = _cube(request, dataset_id)
    if ds.lazy and ds.signal.data.nbytes > _session.DIFFRACTION4D_IN_MEMORY_LIMIT:
        raise HTTPException(422, f"{ds.title} is too big to process in memory: bin it first")
    centres = _beam_centres(request, ds)

    def run(job) -> dict:
        job.report(0.0, "Starting")
        data = background.subtract_all(ds.signal.data, body.options, centres, lambda f, m: job.report(0.98 * f, m))
        job.report(0.99, "Keeping the patterns")
        made = session.subtract_background(ds, data, background.describe(body.options), body.in_place)
        return {"file": session.file_of(ds).summary(session), "id": made.id}

    return request.app.state.jobs.submit("background", f"Background of {ds.title}", run).summary()


# ---------- Bragg disk detection ----------


def _template(ds, opts):
    """The template for ``opts`` and what it is ({"kind", "cx", "cy", "r"}:
    the bright-field disk it was measured on, pattern px)."""
    from ..processing import disks
    from ..processing.frames import get_frame

    mean = get_frame(ds, projection="mean-pattern")
    bf = fourd.bright_field_disk(mean)
    if opts.template == "probe":
        return disks.probe_template(mean, bf), {"kind": "probe", **bf}
    r = opts.radius or bf["r"]
    return disks.disc_template(r), {"kind": "disc", "cx": bf["cx"], "cy": bf["cy"], "r": r}


class DiskBody(BaseModel):
    options: DiskOptions = DiskOptions()


@router.post("/api/datasets/{dataset_id}/4d/disks/template")
def disk_template(dataset_id: str, body: DiskBody, request: Request) -> Response:
    """The template disks are found with (the probe from the mean pattern,
    or a disc), with what it is in the headers."""
    ds = _cube(request, dataset_id)
    try:
        template, info = _template(ds, body.options)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return array_response(template, **info)


class PreviewSource(BaseModel):
    """A pattern to try the detection on: at a scan pixel, over a region, or
    the scan's mean or maximum. ``id``: the caller's name for it."""

    id: str
    x: float | None = None
    y: float | None = None
    region: fourd.Region | None = None
    scan: Literal["mean", "max"] | None = None


class PreviewBody(DiskBody):
    sources: list[PreviewSource] = Field(default_factory=list, max_length=64)


@router.post("/api/datasets/{dataset_id}/4d/disks/preview")
def disk_preview(dataset_id: str, body: PreviewBody, request: Request) -> dict:
    """The disks found in a few patterns with ``options``, to tune them before
    finding every pattern's: {"template": …, "disks": {source id: [[ky, kx,
    score], …]}} (pattern px, pixel i spanning [i, i + 1))."""
    from ..processing import disks
    from ..processing.frames import get_frame

    ds = _cube(request, dataset_id)
    try:
        template, info = _template(ds, body.options)
    except ValueError as e:
        raise HTTPException(422, str(e))
    data = ds.signal.data
    out = {}
    for s in body.sources:
        try:
            if s.scan:
                pattern = get_frame(ds, projection=f"{s.scan}-pattern")
            elif s.region is not None:
                pattern, _ = fourd.pattern_over(data, s.region, "mean")
            elif s.x is not None and s.y is not None:
                pattern = fourd.pattern_at(data, s.x, s.y)
            else:
                continue
        except (ValueError, IndexError):
            continue
        out[s.id] = disks.find_disks(pattern, template, body.options).round(3).tolist()
    return {"template": info, "disks": out}


@router.post("/api/datasets/{dataset_id}/4d/disks/run")
def disk_run(dataset_id: str, body: DiskBody, request: Request) -> dict:
    """Find the disks of every pattern (as a job): kept as the data's "Bragg
    vectors" table and "Bragg vector map" (made again if there are). The
    job's result is ``{file, id}`` (the table's id) and ``count``."""
    from ..processing import disks

    session = request.app.state.session
    ds = _cube(request, dataset_id)
    try:
        template, info = _template(ds, body.options)
    except ValueError as e:
        raise HTTPException(422, str(e))
    opts = body.options

    def run(job) -> dict:
        job.report(0.0, "Starting")
        table = disks.find_all(ds.signal.data, template, opts, lambda f, m: job.report(0.97 * f, m))
        job.report(0.98, "Making the diffraction vectors and the Bragg vector map")
        shape = tuple(ds.signal.data.shape[2:])
        bvm = disks.vector_map(table, shape, upsample=2)
        vectors = disks.to_vectors(table, ds.signal)
        ny, nx = ds.signal.data.shape[:2]
        made = session.set_bragg_vectors(
            ds, vectors, len(table), {"options": opts.model_dump(), "template": info, "positions": int(ny * nx)},
            bvm, upsample=2,
        )
        # The map as the data's calibration has it (raw if none).
        session.remake_bragg_map(ds)
        return {"file": session.file_of(ds).summary(session), "id": made.id, "count": int(len(table))}

    return request.app.state.jobs.submit("disks", f"Disk detection of {ds.title}", run).summary()


@router.get("/api/datasets/{dataset_id}/4d/disks")
def disks_at(dataset_id: str, x: float, y: float, request: Request) -> dict:
    """The disks found at scan pixel (x, y): {"disks": [[ky, kx, score], …]};
    404 if the data's disks haven't been found."""
    from ..processing import disks

    session = request.app.state.session
    ds = _cube(request, dataset_id)
    vectors = session.bragg_vectors_of(ds)
    if vectors is None:
        raise HTTPException(404, f"The disks of {ds.title} haven't been found")
    try:
        rows = disks.vectors_at(vectors.signal, x, y)
    except IndexError as e:
        raise HTTPException(416, str(e))
    return {"disks": rows.round(3).tolist()}


def _vectors(request: Request, ds):
    vectors = request.app.state.session.bragg_vectors_of(ds)
    if vectors is None:
        raise HTTPException(409, f"Find the disks of {ds.title} first (Disk detection)")
    return vectors.signal


class MagnitudesBody(BaseModel):
    bins: int = Field(100, ge=10, le=1000)
    structures: list[str] = []  # Structures whose reflections are marked


@router.post("/api/datasets/{dataset_id}/4d/disks/magnitudes")
def disk_magnitudes(dataset_id: str, body: MagnitudesBody, request: Request) -> dict:
    """pyxem's histogram of the Bragg vectors' magnitudes (from the pattern
    axes' 0), with the Structures' reflections (diffsims'
    ``calculate_diffraction1d``) to check the phase and the calibration:
    ``{k, counts, units, reflections: [{structure, hkl, k, intensity}]}``,
    k in the pattern's units."""
    from ..processing import disks, orientation
    from .orientation import PER_ANGSTROM, _structure

    ds = _cube(request, dataset_id)
    k, counts = disks.magnitude_histogram(_vectors(request, ds), body.bins)
    units = str(ds.signal.axes_manager.signal_axes[0].units)
    per = PER_ANGSTROM.get(units)
    reflections = []
    if per is not None and len(k):
        top = float(k[-1]) / per
        for sid in body.structures:
            st, phase = _structure(request, sid)
            for r in orientation.reflections_1d(phase, top):
                reflections.append({"structure": st.title, **r, "k": r["k"] * per})
    return {"k": np.round(k, 6).tolist(), "counts": counts.tolist(), "units": units, "reflections": reflections}


class VectorMaskBody(BaseModel):
    disk_r: float = Field(2.1, gt=0, le=100)  # pattern px
    in_place: bool = False


@router.get("/api/datasets/{dataset_id}/4d/disks/mask")
def vector_mask_at(dataset_id: str, x: float, y: float, request: Request, disk_r: float = 2.1) -> Response:
    """The vector mask at scan pixel (x, y) (uint8, the pattern's shape)."""
    from skimage.draw import disk

    from ..processing import disks

    ds = _cube(request, dataset_id)
    try:
        rows = disks.vectors_at(_vectors(request, ds), x, y)
    except IndexError as e:
        raise HTTPException(416, str(e))
    shape = tuple(ds.signal.data.shape[2:])
    out = np.zeros(shape, np.uint8)
    # As pyxem's to_mask: a disc about each vector's nearest pixel.
    for ky, kx in np.rint(rows[:, :2] - 0.5).astype(int):
        rr, cc = disk((ky, kx), disk_r, shape=shape)
        out[rr, cc] = 1
    return array_response(out.astype(np.float32))


@router.post("/api/datasets/{dataset_id}/4d/disks/mask")
def vector_mask(dataset_id: str, body: VectorMaskBody, request: Request) -> dict:
    """Mask the data by its Bragg vectors (pyxem's ``to_mask``, discs of
    ``disk_r`` px, × the data, as its orientation mapping tutorial does
    before matching), as a copy under it or in place (a job). The job's
    result is ``{file, id}`` (the data masked)."""
    from ..processing import disks

    session = request.app.state.session
    ds = _cube(request, dataset_id)
    vectors = _vectors(request, ds)
    if ds.lazy and ds.signal.data.nbytes > _session.DIFFRACTION4D_IN_MEMORY_LIMIT:
        raise HTTPException(422, f"{ds.title} is too big to process in memory: bin it first")

    def run(job) -> dict:
        job.report(0.05, "Making the mask")
        mask = disks.vector_mask(vectors, ds.signal, body.disk_r)
        job.report(0.8, "Masking the patterns")
        made = session.mask_with_vectors(ds, mask, body.disk_r, body.in_place)
        return {"file": session.file_of(ds).summary(session), "id": made.id}

    return request.app.state.jobs.submit("vector-mask", f"Vector mask of {ds.title}", run).summary()


# ---------- centring ----------


class BeamBody(BaseModel):
    options: BeamOptions = BeamOptions()


@router.post("/api/datasets/{dataset_id}/4d/beam/find")
def beam_find(dataset_id: str, body: BeamBody, request: Request) -> dict:
    """Find the direct beam in every pattern (as a job): kept as the data's
    "Beam shift x" / "Beam shift y" images (made again if there are). The
    job's result is ``{file, ids, range}``."""
    from ..processing import centring
    from ..processing.frames import get_frame

    session = request.app.state.session
    ds = _cube(request, dataset_id)
    seed = body.options.seed
    if seed is not None:
        # The beam marked by an aperture.
        h, w = ds.signal.axes_manager.signal_shape[::-1]
        if not (0 <= seed.cx <= w and 0 <= seed.cy <= h):
            raise HTTPException(422, "The aperture is outside the pattern")
        bf = {"cx": seed.cx, "cy": seed.cy, "r": seed.r}
    else:
        try:
            bf = fourd.bright_field_disk(get_frame(ds, projection="mean-pattern"))
        except ValueError as e:
            raise HTTPException(422, str(e))

    def run(job) -> dict:
        job.report(0.05, "Finding the beam in every pattern")
        shifts = centring.find_beam(ds.signal, body.options, bf)
        stats = centring.shift_stats(shifts)
        # The data's revision: centring (or any change) makes these shifts stale.
        made = session.set_beam_shifts(
            ds, shifts, {"options": body.options.model_dump(), "range": stats, "revision": ds.revision},
        )
        return {"file": session.file_of(ds).summary(session), "ids": [d.id for d in made], "range": stats}

    return request.app.state.jobs.submit("beam", f"Direct beam of {ds.title}", run).summary()


def _shifts(request: Request, ds, plane: PlaneOptions):
    from ..processing import centring

    found = request.app.state.session.beam_shifts_of(ds)
    if found is None:
        raise HTTPException(404, f"The beam of {ds.title} hasn't been found")
    if found.shape[:2] != tuple(ds.signal.data.shape[:2]):
        raise HTTPException(409, "The beam was found on another scan (binned since?): find it again")
    found_on = next((d.beam.get("revision") for d in request.app.state.session.siblings(ds)
                     if d.origin == "beam-shift" and d.parent_id == ds.id and d.beam), None)
    if found_on is not None and found_on != ds.revision:
        raise HTTPException(409, "The data changed since the beam was found (centred?): find it again")
    return found, centring.plane(found, plane)


@router.get("/api/datasets/{dataset_id}/4d/beam")
def beam_at(
    dataset_id: str, x: float, y: float, request: Request,
    mode: Literal["none", "plane", "corners"] = "plane", corners: float = 0.1,
) -> dict:
    """Where the beam was found at scan pixel (x, y), and where the descan
    plane puts it (pattern px, pixel i spanning [i, i + 1)): {"found": [x,
    y], "used": [x, y], "residual": rms of found − used over the scan (px)}."""
    from ..processing import centring

    ds = _cube(request, dataset_id)
    found, used = _shifts(request, ds, PlaneOptions(mode=mode, corners=corners))
    ny, nx = found.shape[:2]
    ix, iy = int(x), int(y)
    if not (0 <= ix < nx and 0 <= iy < ny):
        raise HTTPException(416, f"({ix}, {iy}) is outside the {nx} × {ny} scan")
    shape = tuple(ds.signal.data.shape[2:])
    f = centring.beam_positions(found[iy, ix], shape)
    u = centring.beam_positions(used[iy, ix], shape)
    rms = float(((found - used) ** 2).sum(axis=-1).mean() ** 0.5)
    return {"found": [round(float(v), 3) for v in f], "used": [round(float(v), 3) for v in u], "residual": round(rms, 3)}


class CentreBody(BaseModel):
    plane: PlaneOptions = PlaneOptions()
    in_place: bool = True
    # "patterns": shift every pattern; "axes": only put the pattern axes' 0 on
    # the beam (its median over the scan; for a negligible descan).
    move: Literal["patterns", "axes"] = "patterns"


@router.post("/api/datasets/{dataset_id}/4d/beam/centre")
def beam_centre(dataset_id: str, body: CentreBody, request: Request) -> dict:
    """Shift every pattern so the beam (as found, or the descan plane) is at
    the centre, in place or as a copy under the data (as a job). The job's
    result is ``{file, id}`` (the data centred)."""
    session = request.app.state.session
    ds = _cube(request, dataset_id)
    _, used = _shifts(request, ds, body.plane)

    def run(job) -> dict:
        if body.move == "axes":
            try:
                at = session.centre_axes(ds, used)
            except ValueError as e:
                raise RuntimeError(str(e))
            return {"file": session.file_of(ds).summary(session), "id": ds.id, "at": at}
        try:
            made = session.centre_patterns(ds, used, body.in_place, lambda f, m: job.report(f, m))
        except ValueError as e:
            raise RuntimeError(str(e))
        return {"file": session.file_of(ds).summary(session), "id": made.id}

    return request.app.state.jobs.submit("centre", f"Centring {ds.title}", run).summary()


# ---------- calibration ----------


class OriginBody(BaseModel):
    """Each scan position's origin: none (the ring's centre, else the
    pattern's, for all), the beam found by Centring, or the direct-beam disk
    among its own disks; then as found or a descan plane."""

    origin: Literal["none", "beam", "direct"] = "beam"
    plane: PlaneOptions = PlaneOptions()


class RingBody(BaseModel):
    options: RingOptions
    # What the ring is fitted on: the scan's mean or maximum pattern, or the
    # Bragg vector map (in pattern pixels, each position moved by ``origin``).
    source: Literal["mean", "max", "bvm"] = "mean"
    origin: OriginBody = OriginBody()


@router.post("/api/datasets/{dataset_id}/4d/calibration/ring")
def calibration_ring(dataset_id: str, body: RingBody, request: Request) -> dict:
    """A ring of the mean (or max) pattern, or of the Bragg vector map,
    fitted as an ellipse between ``r_min`` and ``r_max`` (pattern px):
    {"cx", "cy", "a", "b", "angle", "rms", "radius", "ellipticity",
    "affine", "points"} (pattern px, pixel i spanning [i, i + 1); "points":
    where the ring was found, for drawing)."""
    from ..processing import calibration
    from ..processing.frames import get_frame

    ds = _cube(request, dataset_id)
    o = body.options
    try:
        guess = fourd.bright_field_disk(get_frame(ds, projection="mean-pattern"))
        if body.source == "bvm":
            # Fitted on the map (2 × finer than the pattern), in pattern px after.
            up = 2
            image = bvm_pixels(request, ds, body.origin, upsample=up)
            fine = o.model_copy(update={
                "r_min": o.r_min * up, "r_max": o.r_max * up,
                "cx": None if o.cx is None else o.cx * up, "cy": None if o.cy is None else o.cy * up,
            })
            ring = calibration.fit_ring(image, fine, {"cx": guess["cx"] * up, "cy": guess["cy"] * up})
            pts = calibration.ring_points(image, ring["cx"], ring["cy"], fine.r_min, fine.r_max, o.angles) / up
            for k in ("cx", "cy", "a", "b", "rms", "radius"):
                ring[k] /= up
        else:
            image = get_frame(ds, projection=f"{body.source}-pattern")
            ring = calibration.fit_ring(image, o, guess)
            pts = calibration.ring_points(image, ring["cx"], ring["cy"], o.r_min, o.r_max, o.angles)
    except (ValueError, np.linalg.LinAlgError) as e:
        raise HTTPException(422, str(e))
    return {
        **{k: round(v, 4) for k, v in ring.items()},
        "affine": calibration.affine_of(ring),
        "points": np.round(pts, 2).tolist(),
    }


@router.get("/api/datasets/{dataset_id}/4d/calibration/standards")
def calibration_standards(dataset_id: str, request: Request) -> dict:
    """The calibration standards offered: {material: {hkl: d (Å)}}."""
    from ..processing.calibration import STANDARDS

    _cube(request, dataset_id)
    return STANDARDS


class Ring(BaseModel):
    cx: float
    cy: float
    a: float
    b: float
    angle: float
    rms: float = 0.0
    radius: float
    ellipticity: float = 0.0


class EllipseBody(BaseModel):
    ring: Ring


@router.post("/api/datasets/{dataset_id}/4d/calibration/ellipse")
def calibration_ellipse(dataset_id: str, body: EllipseBody, request: Request) -> dict:
    """Keep the ellipse fitted (Calibration step 2) in the data's
    calibration, switched on: the Bragg vectors made circular, the Bragg
    vector map made again. Returns the data's ``file``."""
    session = request.app.state.session
    ds = _cube(request, dataset_id)
    try:
        session.set_ellipse(ds, body.ring.model_dump())
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"file": session.file_of(ds).summary(session)}


class CalibrateBody(BaseModel):
    """How the pixel size is set: from the ring and a reflection (``d``, Å;
    with ``standard`` and ``hkl`` named), from the camera length, detector
    pixel and voltage, or entered (``scale``, in ``units`` per px)."""

    how: Literal["ring", "camera", "entered"]
    ring: Ring | None = None
    d: float | None = Field(None, gt=0)
    standard: str | None = None
    hkl: str | None = None
    camera_length: float | None = Field(None, gt=0)  # mm
    pixel: float | None = Field(None, gt=0)  # detector pixel, µm
    kv: float | None = Field(None, gt=0)
    scale: float | None = Field(None, gt=0)
    units: Literal["1/Å", "1/nm"] = "1/nm"
    # Put 0 at the ring's centre (when there is a ring).
    centre: bool = True


@router.post("/api/datasets/{dataset_id}/4d/calibration")
def calibrate(dataset_id: str, body: CalibrateBody, request: Request) -> dict:
    """Calibrate the pattern axes (the data isn't changed): the scale, its
    units, 0 at the ring's centre; the ring's ellipticity kept (as pyxem's
    affine) for azimuthal integration. Returns the data's ``file``."""
    from ..processing import calibration

    session = request.app.state.session
    ds = _cube(request, dataset_id)
    per = 10.0 if body.units == "1/nm" else 1.0  # 1/Å → the units asked for
    ring = body.ring.model_dump() if body.ring else None
    if body.how == "ring":
        if ring is None or body.d is None:
            raise HTTPException(422, "A ring and the reflection's d-spacing are needed")
        scale = calibration.scale_from_ring(ring["radius"], body.d) * per
        how = {"standard": body.standard, "hkl": body.hkl, "d": body.d}
    elif body.how == "camera":
        if not (body.camera_length and body.pixel and body.kv):
            raise HTTPException(422, "The camera length, detector pixel and voltage are needed")
        scale = calibration.scale_from_camera(body.camera_length, body.pixel, body.kv) * per
        how = {"camera_length": body.camera_length, "pixel": body.pixel, "kv": body.kv}
    else:
        if body.scale is None:
            raise HTTPException(422, "The pixel size is needed")
        scale = body.scale
        how = {"entered": True}
    info = {"how": how}
    try:
        session.calibrate_patterns(ds, scale, body.units, ring if body.centre else None, info)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"file": session.file_of(ds).summary(session), "scale": scale}


# ---------- calibrated Bragg vectors and their map ----------


def _table(request: Request, ds):
    from ..processing import disks

    vectors = request.app.state.session.bragg_vectors_of(ds)
    if vectors is None:
        raise HTTPException(404, f"The disks of {ds.title} haven't been found")
    return disks.table_of(vectors.signal)


def _origins(request: Request, ds, table, body: OriginBody):
    """[y, x, 2] origins (x, y pattern px) for ``body``, or None (origin "none")."""
    from ..processing import centring, disks
    from ..processing.frames import get_frame

    if body.origin == "none":
        return None
    scan = tuple(ds.signal.data.shape[:2])
    if body.origin == "beam":
        found = request.app.state.session.beam_shifts_of(ds)
        if found is None or found.shape[:2] != scan:
            raise HTTPException(409, "Find the beam first (Centring, step 1), or take the origin from the direct-beam disks")
        return centring.beam_positions(centring.plane(found, body.plane), tuple(ds.signal.data.shape[2:]))
    guess = fourd.bright_field_disk(get_frame(ds, projection="mean-pattern"))
    try:
        origins = disks.fill_origins(disks.direct_beam_origins(table, scan, guess))
    except ValueError as e:
        raise HTTPException(422, str(e))
    if body.plane.mode == "none":
        return origins
    # The plane fitted to the origins (as it is to the beam's shifts: linear in each).
    return centring.plane(origins.astype(np.float32), body.plane)


class MeasureOriginBody(BaseModel):
    """Measure each scan position's origin from the direct-beam disk among
    its Bragg vectors ("vectors"), or take the beam found by Centring
    ("beam"); then fit a descan plane (or keep each position's)."""

    source: Literal["vectors", "beam"] = "vectors"
    plane: PlaneOptions = PlaneOptions()


@router.post("/api/datasets/{dataset_id}/4d/calibration/origin")
def calibration_origin(dataset_id: str, body: MeasureOriginBody, request: Request) -> dict:
    """The origin per scan position (py4DSTEM's measure_origin / fit_origin):
    kept as "Origin x / y" (measured) and "Origin x / y (fit)" images under
    the Bragg vector map (from the vectors) or the data (from the beam), and
    in its calibration (switched on); the Bragg vector map is
    made again. Returns ``{file, mean, rms}`` (mean origin, and measured −
    fitted RMS, px)."""
    from ..processing import centring

    session = request.app.state.session
    ds = _cube(request, dataset_id)
    if body.source == "vectors":
        measured = _origins(request, ds, _table(request, ds), OriginBody(origin="direct", plane=PlaneOptions(mode="none")))
    else:
        found = session.beam_shifts_of(ds)
        if found is None or found.shape[:2] != tuple(ds.signal.data.shape[:2]):
            raise HTTPException(409, "Find the beam first (Centring, step 1), or measure from the Bragg vectors")
        measured = centring.beam_positions(found, tuple(ds.signal.data.shape[2:]))
    measured = np.asarray(measured, dtype=np.float32)
    fitted = None if body.plane.mode == "none" else centring.plane(measured, body.plane)
    rms = float(np.sqrt(((measured - fitted) ** 2).sum(axis=-1).mean())) if fitted is not None else 0.0
    mean = [float(v) for v in np.nanmean((fitted if fitted is not None else measured).reshape(-1, 2), axis=0)]
    info = {**body.model_dump(), "mean": mean, "rms": rms}
    # Centring done on the Bragg vectors: its maps go under the Bragg vector map.
    parent = session.bragg_map_of(ds) if body.source == "vectors" else None
    session.set_origin(ds, measured, fitted, info, parent=parent)
    return {"file": session.file_of(ds).summary(session), "mean": mean, "rms": rms}


class SwitchesBody(BaseModel):
    """Parts of the calibration applied to the Bragg vectors, and whether
    the Bragg vector map shows them ("calibrated") or not ("raw")."""

    enabled: dict[Literal["origin", "ellipse", "scale", "rotation"], bool] | None = None
    view: Literal["raw", "calibrated"] | None = None


class QrRotationBody(BaseModel):
    """The QR rotation (degrees, counter-clockwise) and mirror turning the
    pattern's axes into the scan's: found by the least curl of the descan
    step's deflection ("curl", with the curl left) or entered."""

    theta: float = Field(ge=-360, le=360)
    mirror: bool = False
    how: Literal["curl", "entered"] = "entered"
    curl_ratio: float | None = None


@router.post("/api/datasets/{dataset_id}/4d/calibration/rotation/find")
def calibration_rotation_find(dataset_id: str, request: Request) -> dict:
    """The QR rotation leaving the least curl (``processing.com.rotation_of``)
    in the deflection the descan step measured: each position's origin less
    the descan plane fitted to it (a plane over the whole scan when the
    origins were kept as measured). Returns ``{theta, mirror, curl_ratio}``."""
    from ..processing import centring, com

    session = request.app.state.session
    ds = _cube(request, dataset_id)
    maps = session.origin_maps_of(ds)
    if "measured" not in maps:
        raise HTTPException(409, "Measure the origin first (step 1, Descan)")
    measured = np.nan_to_num(np.asarray(maps["measured"], np.float64))
    fitted = maps.get("fitted")
    ref = np.asarray(fitted, np.float64) if fitted is not None else centring.plane(measured.astype(np.float32), PlaneOptions(mode="plane"))
    d = measured - ref
    if min(d.shape[:2]) < 4:
        raise HTTPException(422, "The scan is too small to find a rotation")
    found = com.rotation_of(d[..., 0], d[..., 1])
    return {"theta": float(found["theta"]) % 360.0, "mirror": bool(found["mirror"]), "curl_ratio": float(found["curl_ratio"])}


@router.post("/api/datasets/{dataset_id}/4d/calibration/rotation")
def calibration_rotation(dataset_id: str, body: QrRotationBody, request: Request) -> dict:
    """Keep the QR rotation in the data's calibration (switched on; the
    Bragg vector map made again, in the scan's frame). Returns ``file``."""
    session = request.app.state.session
    ds = _cube(request, dataset_id)
    info = {"how": body.how, **({"curl_ratio": body.curl_ratio} if body.curl_ratio is not None else {})}
    try:
        session.set_qr_rotation(ds, body.theta, body.mirror, info)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"file": session.file_of(ds).summary(session)}


@router.post("/api/datasets/{dataset_id}/4d/calibration/switches")
def calibration_switches(dataset_id: str, body: SwitchesBody, request: Request) -> dict:
    """Switch calibrations on or off (the Bragg vector map made again at
    once). Returns the data's ``file``."""
    session = request.app.state.session
    ds = _cube(request, dataset_id)
    session.set_calibration_switches(ds, body.enabled, body.view)
    return {"file": session.file_of(ds).summary(session)}


def bvm_pixels(request: Request, ds, origin: OriginBody, upsample: int = 2) -> np.ndarray:
    """The Bragg vector map in pattern pixels (``upsample`` × finer), each
    position's disks moved by its origin's offset from the mean origin: the
    rings sharp, where they are on the pattern. For fitting a ring."""
    from ..processing import disks

    table = _table(request, ds)
    origins = _origins(request, ds, table, origin)
    if origins is not None:
        mean = np.nanmean(origins.reshape(-1, 2), axis=0)
        q = disks.calibrated(table, origins, (0, 0)) + mean
        table = np.column_stack([table[:, :2], q[:, 1], q[:, 0], table[:, 4]]).astype(np.float32)
    return disks.vector_map(table, tuple(ds.signal.data.shape[2:]), upsample=upsample)
