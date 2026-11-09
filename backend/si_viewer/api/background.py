"""EELS background removal: a live preview, and the removal itself (as a job)."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..jobs import Job
from ..processing.background import BackgroundOptions, EnergyAxis, channel_windows, preview
from .datasets import _dataset

router = APIRouter(tags=["background"])

# Spectra per preview request.
MAX_PREVIEW_SPECTRA = 256


def _eels_spectra(request: Request, dataset_id: str):
    ds = _dataset(request, dataset_id)
    if not request.app.state.session.can_remove_background(ds):
        raise HTTPException(400, "Background removal is for EELS spectra")
    return ds


class PreviewBody(BaseModel):
    options: BackgroundOptions
    # Spectra on the dataset's whole energy axis.
    spectra: list[list[float]]


@router.post("/api/datasets/{dataset_id}/background/preview")
def post_background_preview(dataset_id: str, body: PreviewBody, request: Request) -> dict:
    """The background fitted to each of ``spectra`` (a list of spectra on the
    dataset's whole energy axis), from the first channel of the background
    window (``start``) to the last of the signal window, inclusive; and the
    power-law exponent ``r`` of each (null for other models and failed fits).
    Quick, for use while the windows are dragged."""
    ds = _eels_spectra(request, dataset_id)
    axis = EnergyAxis.of(ds.signal)
    if len(body.spectra) > MAX_PREVIEW_SPECTRA:
        raise HTTPException(422, f"At most {MAX_PREVIEW_SPECTRA} spectra at a time")
    if any(len(s) != axis.size for s in body.spectra):
        raise HTTPException(422, f"Each spectrum needs {axis.size} channels, the dataset's whole energy axis")
    try:
        spectra = np.asarray(body.spectra, dtype=float).reshape(len(body.spectra), axis.size)
        start, background, r = preview(spectra, axis, body.options)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {
        "start": start,
        "background": background.tolist(),
        "r": [float(v) if np.isfinite(v) else None for v in r],
    }


class BackgroundBody(BaseModel):
    options: BackgroundOptions


def _submit(request: Request, src, bg, options: BackgroundOptions) -> dict:
    """Start the job that removes the background of ``src`` (adding a new
    dataset, or replacing ``bg``)."""
    session = request.app.state.session
    try:
        channel_windows(EnergyAxis.of(src.signal), options)  # fail now, not in the job
    except ValueError as e:
        raise HTTPException(422, str(e))

    def run(job: Job) -> dict:
        job.report(0.02, "Fitting the background")
        result = session.compute_background(src, options, lambda f: job.report(0.02 + 0.95 * f))
        job.report(0.98)  # last chance to cancel before the result is saved
        if bg is None:
            ds = session.add_background(src, options, result)
        else:
            if bg.id not in session.datasets:
                raise ValueError(f"{bg.title} was deleted while the background was removed")
            ds = session.replace_background(bg, options, result)
        return {"file": session.file_of(src).summary(session), "dataset_id": ds.id}

    title = f"Remove the background of {src.title}"
    return request.app.state.jobs.submit("background", title, run).summary()


@router.post("/api/datasets/{dataset_id}/background")
def post_background(dataset_id: str, body: BackgroundBody, request: Request) -> dict:
    """Remove the background of EELS spectra (as a job): the result is a new
    dataset under it, holding the signal window with the fitted background
    subtracted (and, for a power law on a 2D spectrum image, a map of the
    exponent r). The job's result is ``{file, dataset_id}``."""
    src = _eels_spectra(request, dataset_id)
    return _submit(request, src, None, body.options)


@router.put("/api/datasets/{dataset_id}/background")
def put_background(dataset_id: str, body: BackgroundBody, request: Request) -> dict:
    """Redo a background-removed dataset with new options (as a job), from its
    source and in place: same id, new revision. The job's result is
    ``{file, dataset_id}``."""
    bg = _dataset(request, dataset_id)
    if bg.origin != "background" or bg.background is None:
        raise HTTPException(400, f"{bg.title} is not a background-removed spectrum")
    session = request.app.state.session
    try:
        src = session.get_dataset(bg.parent_id)
    except KeyError:
        raise HTTPException(404, f"The source of {bg.title} is gone")
    return _submit(request, src, bg, body.options)
