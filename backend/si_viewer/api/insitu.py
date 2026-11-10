"""In-situ series (M12b): show a frame, reduce a frame range (a job), the
simultaneous ADF as navigator."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

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
