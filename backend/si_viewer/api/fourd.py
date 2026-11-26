"""4D-STEM: the pattern at a scan position or over a region, virtual images
through a detector, and virtual images kept as datasets."""

from __future__ import annotations

from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from ..processing import fourd
from ..processing.disks import DiskOptions
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


class VirtualBody(BaseModel):
    detector: fourd.Detector
    response: fourd.Response = "sum"


@router.post("/api/datasets/{dataset_id}/4d/virtual")
def virtual(dataset_id: str, body: VirtualBody, request: Request) -> Response:
    """The virtual image through ``detector``: per scan position, the sum,
    mean or maximum of the pattern pixels inside it."""
    ds = _cube(request, dataset_id)
    image = fourd.virtual_image(ds.signal.data, body.detector, body.response)
    pixels = int(fourd.detector_mask(ds.signal.data.shape[2:], body.detector).sum())
    return array_response(image, pixels=pixels)


class VirtualsBody(BaseModel):
    items: list[VirtualBody]


@router.post("/api/datasets/{dataset_id}/4d/virtuals")
def virtuals(dataset_id: str, body: VirtualsBody, request: Request) -> Response:
    """The virtual images through several detectors at once, [n, y, x] (in
    the order given)."""
    ds = _cube(request, dataset_id)
    return array_response(fourd.virtual_images(ds.signal.data, [(i.detector, i.response) for i in body.items]))


class SaveBody(VirtualBody):
    title: str | None = None


@router.post("/api/datasets/{dataset_id}/4d/virtual/save")
def save_virtual(dataset_id: str, body: SaveBody, request: Request) -> dict:
    """The virtual image through ``detector`` as an image dataset under the
    4D-STEM data. Returns its ``file`` and ``id``."""
    session = request.app.state.session
    ds = _cube(request, dataset_id)
    image = fourd.virtual_image(ds.signal.data, body.detector, body.response)
    axis = ds.signal.axes_manager.signal_axes[0]
    title = (body.title or "").strip() or fourd.describe(body.detector, body.response, axis.scale, axis.units or "px")
    try:
        made = session.add_virtual_image(ds, image, body.detector.model_dump(), body.response, title)
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
        job.report(0.98, "Making the Bragg vector map")
        shape = tuple(ds.signal.data.shape[2:])
        bvm = disks.vector_map(table, shape, upsample=2)
        ny, nx = ds.signal.data.shape[:2]
        made = session.set_bragg_vectors(
            ds, table, {"options": opts.model_dump(), "template": info, "positions": int(ny * nx)}, bvm, upsample=2,
        )
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
    rows = disks.disks_at(np.asarray(vectors.signal.data), x, y)
    return {"disks": rows[:, 2:].round(3).tolist()}
