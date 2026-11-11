"""In-situ series (M12b): show a frame, reduce a frame range (a job), the
simultaneous ADF as navigator, time scans (a job): line scans whose rows
are the frames, and the drift-corrected sum of the frames (a job)."""

from __future__ import annotations

import base64
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field, model_validator

from ..processing.regions import Region

from ..insitu_series import adf_image, adf_stack
from ..jobs import Job
from .datasets import _dataset, array_response
from .eds import _with_file

router = APIRouter(tags=["insitu"])


def _series_dataset(request: Request, dataset_id: str):
    ds = _dataset(request, dataset_id)
    if ds.insitu is None:
        raise HTTPException(400, f"{ds.title} is not an in-situ series")
    return ds


class FrameBody(BaseModel):
    frame: int  # 0-based


@router.put("/api/datasets/{dataset_id}/insitu/frame")
def put_frame(dataset_id: str, body: FrameBody, request: Request) -> dict:
    """Make the dataset hold one frame of its series. Reads that frame (the
    only read this request does) unless it is cached, then reads its
    neighbours ahead in the background. Returns the dataset and its file
    (energy-window maps follow the frame)."""
    ds = _series_dataset(request, dataset_id)
    try:
        request.app.state.session.show_insitu_frame(ds, body.frame)
    except IndexError as e:
        raise HTTPException(416, str(e))
    except OSError as e:
        raise HTTPException(500, f"Could not read frame {body.frame + 1}: {e}")
    return _with_file(request, ds)


class ReduceBody(BaseModel):
    mode: Literal["mean", "sum"]
    start: int  # 0-based, inclusive
    end: int


@router.post("/api/datasets/{dataset_id}/insitu/reduce")
def post_reduce(dataset_id: str, body: ReduceBody, request: Request) -> dict:
    """The mean or sum of a frame range as the dataset's data (as a job: it
    reads every frame of the range, progress "Frame 12 of 40"). The job's
    result is ``{file, dataset_id}``."""
    session = request.app.state.session
    ds = _series_dataset(request, dataset_id)
    series = ds.insitu
    if not 0 <= body.start <= body.end < series.n:
        raise HTTPException(422, f"Choose frames within 1–{series.n}, the first before the last")

    def run(job: Job) -> dict:
        data = series.reduce(body.mode, body.start, body.end, lambda f, m=None: job.report(0.98 * f, m))
        if dataset_id not in session.datasets:
            raise ValueError(f"{ds.title} was closed while its frames were read")
        job.report(0.99)  # last chance to cancel before the result replaces the data
        session.apply_insitu_range(ds, data, body.mode, body.start, body.end)
        try:
            adf_image(ds, session.siblings(ds))  # the ADF over the range is read here, not by the request for it
        except Exception:  # only a courtesy: the navigator asks again
            pass
        return {"file": session.file_of(ds).summary(session), "dataset_id": ds.id}

    label = f"frames {body.start + 1}–{body.end + 1}"
    title = f"{body.mode.capitalize()} of {label} of {ds.title}"
    return request.app.state.jobs.submit("insitu-reduce", title, run).summary()


class TimeScanPixel(BaseModel):
    x: int
    y: int
    size: int = Field(1, ge=1, le=64)


class TimeScanBody(BaseModel):
    """The pixels averaged in every frame: a block at ``pixel`` or a ``region``
    (one of them), in scan pixels."""

    pixel: TimeScanPixel | None = None
    region: Region | None = None
    name: str = Field("Time scan", max_length=100)
    with_linked: bool = True

    @model_validator(mode="after")
    def _one(self) -> "TimeScanBody":
        if (self.pixel is None) == (self.region is None):
            raise ValueError("Give a pixel or a region")
        return self


@router.post("/api/datasets/{dataset_id}/insitu/time-scan")
def post_time_scan(dataset_id: str, body: TimeScanBody, request: Request) -> dict:
    """A line scan over time (as a job: it reads every frame): row k is the
    mean spectrum of the pixel block or region in frame k, on a "Time" axis.
    The series linked to it get one too, linked in turn. The job's result is
    ``{file, dataset_id, partner_files}``."""
    session = request.app.state.session
    ds = _series_dataset(request, dataset_id)
    try:
        sources = session.time_scan_sources(ds, body.with_linked)
        pixel = None if body.pixel is None else (body.pixel.x, body.pixel.y, body.pixel.size)
        session.time_scan_pick(ds, pixel, body.region)  # fail now, not in the job
    except ValueError as e:
        raise HTTPException(422, str(e))
    source = {"pixel": body.pixel.model_dump() if body.pixel else None,
              "region": body.region.model_dump() if body.region else None}

    def run(job: Job) -> dict:
        made = []
        for i, src in enumerate(sources):
            share = 0.98 / len(sources)
            spectra = session.time_scan_spectra(
                src, pixel, body.region, lambda f, m=None, i=i: job.report(share * (i + f), m),
            )
            made.append((src, spectra))
        job.report(0.99)  # last chance to cancel
        new = session.add_time_scans(made, body.name, source)
        files = {d.file_id for d in new}
        return {
            "file": session.file_of(ds).summary(session),
            "dataset_id": new[0].id,
            "partner_files": [session.files[f].summary(session) for f in files if f != ds.file_id],
        }

    title = f"Time scan of {ds.title} · {body.name}"
    return request.app.state.jobs.submit("time-scan", title, run).summary()


@router.get("/api/datasets/{dataset_id}/insitu/adf")
def get_adf(dataset_id: str, request: Request) -> Response:
    """The simultaneous ADF image of the frame (or frame range) the dataset
    holds, at the ADF's own resolution (``factor`` times finer than the scan
    for a sub-scan), for use as navigator."""
    session = request.app.state.session
    ds = _series_dataset(request, dataset_id)
    siblings = session.siblings(ds)
    image = adf_image(ds, siblings)
    if image is None:
        raise HTTPException(404, f"{ds.title} has no ADF series on its scan")
    _, factor = adf_stack(ds, siblings)
    return array_response(image, view=ds.insitu.view.as_dict(), factor=factor)


# ---------- drift-corrected sum ----------


@router.get("/api/datasets/{dataset_id}/insitu/alignment")
def get_alignment_source(dataset_id: str, request: Request) -> dict:
    """What the aligned sum can use: ``frames`` (recorded frames of the
    series) and ``adf`` (the simultaneous ADF stack: ``id``, ``title``, ``k``
    its sub-scan factor, ``aligned`` and how many ``frames`` its alignment
    has; null without one)."""
    return request.app.state.session.alignment_source(_series_dataset(request, dataset_id))


class ShiftsBody(BaseModel):
    """Where the shifts come from: the ADF stack's alignment, or a ``.npy``
    file (base64) in pixels of the ADF or of the scan."""

    source: Literal["adf", "npy"] = "adf"
    npy: str | None = None
    units: Literal["adf", "scan"] = "adf"


def _shifts(request: Request, ds, body: ShiftsBody):
    try:
        raw = base64.b64decode(body.npy) if body.npy is not None else None
    except ValueError:
        raise HTTPException(422, "The .npy file could not be read")
    try:
        return request.app.state.session.aligned_sum_shifts(ds, body.source, raw, body.units)
    except ValueError as e:
        raise HTTPException(422, str(e))


@router.post("/api/datasets/{dataset_id}/insitu/shifts-check")
def post_shifts_check(dataset_id: str, body: ShiftsBody, request: Request) -> dict:
    """Check shifts against the series before summing: ``frames`` summed,
    ``left_out`` (no shift), ``largest`` shift (scan pixels) and the
    ``width`` × ``height`` every frame covers. 422 (with why) if they can't
    be used."""
    from ..processing.insitu_align import covered

    ds = _series_dataset(request, dataset_id)
    frames, shifts, how = _shifts(request, ds, body)
    h, w = ds.insitu.frame_shape[:2]
    try:
        r0, r1, c0, c1 = covered(shifts[frames, 1], shifts[frames, 0], h, w)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {
        "frames": len(frames), "left_out": ds.insitu.n_raw - len(frames),
        "largest": float(np.abs(shifts[frames]).max()), "width": c1 - c0, "height": r1 - r0, "source": how,
    }


@router.post("/api/datasets/{dataset_id}/insitu/aligned-sum")
def post_aligned_sum(dataset_id: str, body: ShiftsBody, request: Request) -> dict:
    """Register every frame of the series with the shifts and sum them (as a
    job: it reads every recorded frame, progress "Frame 12 of 40"): a new
    spectrum image under the series, cropped to the area every frame covers.
    The job's result is ``{file, dataset_id}``."""
    session = request.app.state.session
    ds = _series_dataset(request, dataset_id)
    frames, shifts, how = _shifts(request, ds, body)

    def run(job: Job) -> dict:
        data, crop = session.compute_aligned_sum(ds, frames, shifts, lambda f, m=None: job.report(0.98 * f, m))
        if dataset_id not in session.datasets:
            raise ValueError(f"{ds.title} was closed while its frames were read")
        job.report(0.99)  # last chance to cancel
        new = session.add_aligned_sum(ds, data, crop, frames, how)
        return {"file": session.file_of(ds).summary(session), "dataset_id": new.id}

    return request.app.state.jobs.submit("aligned-sum", f"Aligned sum of {ds.title}", run).summary()
