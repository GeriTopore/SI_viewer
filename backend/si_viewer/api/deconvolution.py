"""Plural-scattering deconvolution of EELS spectrum images (M10d): a live
preview, and the deconvolution itself (as a job)."""

from __future__ import annotations

from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..jobs import Job
from ..processing.deconvolution import DeconvolutionOptions, check, preview
from ..processing.regions import Region, pixel_spectrum, region_spectrum
from ..session import si_cube
from .datasets import _dataset

router = APIRouter(tags=["deconvolution"])

# Spectra per preview request.
MAX_PREVIEW_SPECTRA = 64


class Selection(BaseModel):
    """Pixels whose summed spectrum is previewed: all of them, a block around
    a pixel, or a region."""

    kind: Literal["whole", "pixel", "region"]
    x: int = 0
    y: int = 0
    size: int = Field(1, ge=1, le=64)
    region: Region | None = None


def _inputs(request: Request, ds, options: DeconvolutionOptions):
    """The low loss to deconvolve ``ds`` with (400 if the method isn't for it)."""
    try:
        return request.app.state.session.deconvolution_inputs(ds, options)
    except ValueError as e:
        raise HTTPException(400, str(e))


def _summed(data: np.ndarray, sel: Selection) -> tuple[np.ndarray, int]:
    if sel.kind == "whole":
        flat = data.reshape(-1, data.shape[-1])
        return np.nan_to_num(flat.astype(float)).sum(axis=0), len(flat)
    if sel.kind == "pixel":
        return pixel_spectrum(data, sel.x, sel.y, sel.size)
    if sel.region is None:
        raise HTTPException(422, "A region selection needs its region")
    return region_spectrum(data, sel.region)


class PreviewBody(BaseModel):
    options: DeconvolutionOptions
    selections: list[Selection]


@router.post("/api/datasets/{dataset_id}/deconvolution/preview")
def post_deconvolution_preview(dataset_id: str, body: PreviewBody, request: Request) -> dict:
    """The deconvolved summed spectrum of each selection of pixels (null for
    one with no pixels), with ``pixels`` (how many were summed), and the
    zero-loss threshold and FWHM used for the first (null when not used)."""
    ds = _dataset(request, dataset_id)
    ll = _inputs(request, ds, body.options)
    if len(body.selections) > MAX_PREVIEW_SPECTRA:
        raise HTTPException(422, f"At most {MAX_PREVIEW_SPECTRA} spectra at a time")
    ds.materialise()
    ll.materialise()
    src_cube, ll_cube = si_cube(ds), si_cube(ll)
    pairs, counts = [], []
    for sel in body.selections:
        try:
            s, n = _summed(src_cube, sel)
            low, _ = _summed(ll_cube, sel)
        except IndexError as e:
            raise HTTPException(416, str(e))
        counts.append(int(n))
        if n:
            pairs.append((s, low))
    try:
        result = preview(ds.signal, ll.signal, pairs, body.options)
    except ValueError as e:
        raise HTTPException(422, str(e))
    spectra = iter(result["spectra"])
    return {
        "spectra": [next(spectra).tolist() if n else None for n in counts],
        "pixels": counts,
        "threshold": result["threshold"],
        "fwhm": result["fwhm"],
    }


class DeconvolutionBody(BaseModel):
    options: DeconvolutionOptions


def _submit(request: Request, src, dd, options: DeconvolutionOptions) -> dict:
    """Start the job that deconvolves ``src`` (adding a new dataset, or
    replacing ``dd``)."""
    session = request.app.state.session
    ll = _inputs(request, src, options)
    try:
        check(src.signal, ll.signal, options)  # fail now, not in the job
    except ValueError as e:
        raise HTTPException(422, str(e))

    def run(job: Job) -> dict:
        job.report(0.01, "Loading" if src.lazy or ll.lazy else "Starting")
        signal = session.compute_deconvolution(src, options, lambda f, m=None: job.report(0.02 + 0.95 * f, m))
        job.report(0.98)  # last chance to cancel before the result is saved
        if dd is None:
            ds = session.add_deconvolved(src, options, signal)
        else:
            if dd.id not in session.datasets:
                raise ValueError(f"{dd.title} was deleted while it was deconvolved")
            ds = session.replace_deconvolved(dd, options, signal)
        return {"file": session.file_of(src).summary(session), "dataset_id": ds.id}

    return request.app.state.jobs.submit("deconvolution", f"Deconvolve {src.title}", run).summary()


@router.post("/api/datasets/{dataset_id}/deconvolution")
def post_deconvolution(dataset_id: str, body: DeconvolutionBody, request: Request) -> dict:
    """Deconvolve EELS spectra with their low loss (as a job): the result is a
    new spectrum image under them. The job's result is ``{file, dataset_id}``."""
    return _submit(request, _dataset(request, dataset_id), None, body.options)


@router.put("/api/datasets/{dataset_id}/deconvolution")
def put_deconvolution(dataset_id: str, body: DeconvolutionBody, request: Request) -> dict:
    """Redo a deconvolved spectrum image with new options (as a job), from its
    source and in place: same id, new revision. The job's result is
    ``{file, dataset_id}``."""
    dd = _dataset(request, dataset_id)
    if dd.origin != "deconvolved" or dd.deconvolution is None:
        raise HTTPException(400, f"{dd.title} is not a deconvolved spectrum image")
    try:
        src = request.app.state.session.get_dataset(dd.parent_id)
    except KeyError:
        raise HTTPException(404, f"The source of {dd.title} is gone")
    return _submit(request, src, dd, body.options)
