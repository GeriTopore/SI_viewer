"""EELS model fitting (M13): the elements whose edges are fitted, the fit of
one spectrum, and the fit of every pixel (as a job)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..jobs import Job
from ..processing.eels_model import (
    EelsFitOptions, atomic_percent_maps, build_model, fit_maps, fit_spectrum, pixel_fit,
)
from ..processing.fitting import FitSource, source_spectrum
from ..processing.technique import technique_of
from ..session import DatasetKind
from .datasets import _dataset
from .eds import _with_file

router = APIRouter(tags=["eels-fit"])


def _eels(request: Request, dataset_id: str):
    ds = _dataset(request, dataset_id)
    if ds.kind not in (DatasetKind.SPECTRUM, DatasetKind.SPECTRUM_IMAGE) or technique_of(ds.signal) != "EELS":
        raise HTTPException(400, f"{ds.title} is not an EELS spectrum")
    return ds


class ElementsBody(BaseModel):
    elements: list[str]


@router.put("/api/datasets/{dataset_id}/eels-elements")
def put_eels_elements(dataset_id: str, body: ElementsBody, request: Request) -> dict:
    """Set the elements of an EELS spectrum or spectrum image: the ones whose
    edges are marked and fitted. Returns the dataset (its ``edges``) and its file."""
    ds = _eels(request, dataset_id)
    try:
        request.app.state.session.set_eels_elements(ds, body.elements)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return _with_file(request, ds)


class FitBody(BaseModel):
    source: FitSource = FitSource()
    options: EelsFitOptions = EelsFitOptions()


@router.post("/api/datasets/{dataset_id}/eels-fit")
def post_eels_fit(dataset_id: str, body: FitBody, request: Request) -> dict:
    """Fit one EELS spectrum (a spectrum, or all pixels / a pixel block / a
    region of a spectrum image) with exSpy's model: a power-law background and
    the edges of its elements, convolved with the same spectrum of its group's
    low loss when it has one. Returns the curves over the fitted range and
    each element's intensity (``rows``)."""
    ds = _eels(request, dataset_id)
    session = request.app.state.session
    try:
        spec, pixels = source_spectrum(ds, body.source)
        low_loss = None
        ll = session.eels_fit_low_loss(ds) if body.options.convolve else None
        if ll is not None:
            low_loss, _ = source_spectrum(ll, body.source)
        result = fit_spectrum(spec, body.options, low_loss)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {**result, "pixels": pixels, "low_loss_id": ll.id if ll is not None else None}


class FitMapsBody(BaseModel):
    options: EelsFitOptions = EelsFitOptions()
    # Also at% maps from the fitted intensities.
    composition: bool = True


@router.post("/api/datasets/{dataset_id}/eels-fit-maps")
def post_eels_fit_maps(dataset_id: str, body: FitMapsBody, request: Request) -> dict:
    """Fit every pixel of an EELS spectrum image (as a job). Its result holds
    the file, with a fitted intensity map per element and, if asked, at% maps
    (they replace the ones of an earlier fit)."""
    si = _eels(request, dataset_id)
    session = request.app.state.session
    if si.kind is not DatasetKind.SPECTRUM_IMAGE or si.is_line_scan:
        raise HTTPException(400, f"{si.title} is not a 2D spectrum image: fitting every pixel needs one")
    si.materialise()
    ll = session.eels_fit_low_loss(si) if body.options.convolve else None
    low_loss = ll.signal if ll is not None else None
    try:
        # Fail now, not in the job (the model is built again there).
        build_model(si.signal.inav[0, 0], body.options, None if low_loss is None else low_loss.inav[0, 0])
    except ValueError as e:
        raise HTTPException(422, str(e))

    version = si.data_version

    def run(job: Job) -> dict:
        job.report(0.01, "Building the model")
        maps, skipped, fits = fit_maps(
            si.signal, body.options, low_loss, lambda f: job.report(0.02 + 0.95 * f, "Fitting each pixel")
        )
        results = {el: {"counts": (f"{el} (fit)", data)} for el, data in maps.items()}
        if body.composition and len(maps) > 1:
            for el, data in atomic_percent_maps(maps).items():
                results[el]["at%"] = (f"{el} at%", data)
        job.report(0.98)  # last chance to cancel before the results replace old ones
        session.set_fit_maps(si, list(results.items()))
        si.pixel_fits, si.pixel_fits_version = fits, version
        return {"file": session.file_of(si).summary(session), "skipped": skipped}

    return request.app.state.jobs.submit("eels-fit-maps", f"Fit every pixel of {si.title}", run).summary()


@router.get("/api/datasets/{dataset_id}/eels-pixel-fit")
def get_eels_pixel_fit(dataset_id: str, request: Request, x: int, y: int, size: int = 1) -> dict:
    """The per-pixel fit of an EELS spectrum image at the cursor: the pixel
    (x, y), or the ``size`` × ``size`` block there added up (as ``eels-fit``
    returns a fit; 404 when there is no per-pixel fit of the data as it is)."""
    si = _eels(request, dataset_id)
    fits = si.current_pixel_fits
    if fits is None:
        raise HTTPException(404, f"{si.title} has no per-pixel fit")
    h, w = fits.failed.shape
    if not (0 <= x < w and 0 <= y < h) or not 1 <= size <= 64:
        raise HTTPException(422, "The pixel is outside the spectrum image")
    try:
        return pixel_fit(fits, x, y, size)
    except ValueError as e:
        raise HTTPException(422, str(e))
