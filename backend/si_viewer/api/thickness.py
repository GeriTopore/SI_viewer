"""Thickness of EELS low-loss spectrum images (M10d): the automatic zero-loss
threshold of the summed spectrum, and the thickness itself (as a job)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..jobs import Job
from ..processing.eels import is_eels_si
from ..processing.thickness import ThicknessOptions, check_options
from .datasets import _dataset

router = APIRouter(tags=["thickness"])


def _low_loss(request: Request, dataset_id: str):
    """The low loss that ``dataset_id`` (an EELS spectrum image) is, or is
    linked to."""
    ds = _dataset(request, dataset_id)
    if not is_eels_si(ds):
        raise HTTPException(400, f"{ds.title} is not an EELS spectrum image")
    ll = request.app.state.session.low_loss_of(ds)
    if ll is None:
        raise HTTPException(400, "No low loss in its group")
    return ll


class ThresholdBody(BaseModel):
    window: float = Field(10.0, gt=0, le=1000)


@router.post("/api/datasets/{dataset_id}/thickness/threshold")
def post_threshold(dataset_id: str, body: ThresholdBody, request: Request) -> dict:
    """The automatic zero-loss threshold of the low loss's summed spectrum:
    ``{low_loss_id, threshold}`` (threshold null when the spectrum has no
    inflexion in the search window). ``dataset_id`` is the low loss or a
    member of its group."""
    ll = _low_loss(request, dataset_id)
    try:
        value = request.app.state.session.estimate_threshold(ll, body.window)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"low_loss_id": ll.id, "threshold": value}


class ThicknessBody(BaseModel):
    options: ThicknessOptions = ThicknessOptions()


@router.post("/api/datasets/{dataset_id}/thickness")
def post_thickness(dataset_id: str, body: ThicknessBody, request: Request) -> dict:
    """Estimate the thickness of the low loss that ``dataset_id`` is, or is
    linked to (as a job). The result is a dataset under the low loss: an image
    for a 2D scan, a profile for a line scan. A thickness the low loss already
    has is replaced (same dataset, new revision). The job's result is
    ``{file, dataset_id}``."""
    session = request.app.state.session
    ll = _low_loss(request, dataset_id)
    options = body.options
    try:
        check_options(ll.signal, options)  # fail now, not in the job
    except ValueError as e:
        raise HTTPException(422, str(e))

    def run(job: Job) -> dict:
        job.report(0.01, "Loading the low loss" if ll.lazy else "Starting")
        data = session.compute_thickness(ll, options, lambda f, m=None: job.report(0.02 + 0.95 * f, m))
        job.report(0.98)  # last chance to cancel before the result is saved
        if ll.id not in session.datasets:
            raise ValueError(f"{ll.title} was deleted while its thickness was estimated")
        ds = session.set_thickness(ll, options, data)
        return {"file": session.file_of(ll).summary(session), "dataset_id": ds.id}

    return request.app.state.jobs.submit("thickness", f"Thickness of {ll.title}", run).summary()
