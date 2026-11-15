"""Curve fitting (M21): HyperSpy's 1D components, their starting values, and
the fit of one spectrum (a spectrum, or all pixels / a pixel block / a region
of a spectrum image), and the fit of every pixel (as a job)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..jobs import Job
from ..processing import curve_fit
from ..processing.fitting import FitSource, source_spectrum
from ..session import DatasetKind
from .datasets import _dataset

router = APIRouter(tags=["curve-fit"])


@router.get("/api/curve-fit/components")
def get_components() -> dict:
    """The components that can be added, grouped, with their parameters."""
    return curve_fit.catalogue()


class CurveBody(BaseModel):
    source: FitSource = FitSource()
    model: curve_fit.ModelSpec = curve_fit.ModelSpec()


def _spectrum(request: Request, dataset_id: str, source: FitSource):
    ds = _dataset(request, dataset_id)
    if ds.kind not in (DatasetKind.SPECTRUM, DatasetKind.SPECTRUM_IMAGE):
        raise HTTPException(400, f"{ds.title} is not a spectrum or spectrum image")
    try:
        return source_spectrum(ds, source)
    except ValueError as e:
        raise HTTPException(422, str(e))


class EstimateBody(CurveBody):
    component: str


@router.post("/api/datasets/{dataset_id}/curve-fit/estimate")
def post_estimate(dataset_id: str, body: EstimateBody, request: Request) -> dict:
    """Starting values of one component from its own ``estimate_parameters``
    over the fit range of the chosen spectrum."""
    spec, pixels = _spectrum(request, dataset_id, body.source)
    try:
        return {**curve_fit.estimate(spec, body.model, body.component), "pixels": pixels}
    except ValueError as e:
        raise HTTPException(422, str(e))


class FitBody(CurveBody):
    # False: draw the model with its current values, without fitting.
    fit: bool = True


@router.post("/api/datasets/{dataset_id}/curve-fit")
def post_fit(dataset_id: str, body: FitBody, request: Request) -> dict:
    """Fit the model to the chosen spectrum (or evaluate it): the curves over
    the fit range, the parameters with their errors, and the goodness of fit."""
    spec, pixels = _spectrum(request, dataset_id, body.source)
    try:
        return {**curve_fit.fit(spec, body.model, run=body.fit), "pixels": pixels}
    except ValueError as e:
        raise HTTPException(422, str(e))


class MapsBody(CurveBody):
    """The model, and the spectrum it was made on (``source``): its linear
    parameters are divided by the pixels that spectrum adds up."""


@router.post("/api/datasets/{dataset_id}/curve-fit/maps")
def post_fit_maps(dataset_id: str, body: MapsBody, request: Request) -> dict:
    """Fit the model at every pixel of a 2D spectrum image (as a job), each
    pixel starting from the model's values. Its result holds the file, with
    the fitted model as a spectrum image under the spectrum image (its
    ``dataset_id``), and under the model a reduced χ² map and a map of each
    parameter that varies (they replace the ones of an earlier fit)."""
    si = _dataset(request, dataset_id)
    session = request.app.state.session
    if si.kind is not DatasetKind.SPECTRUM_IMAGE or si.is_line_scan:
        raise HTTPException(400, f"{si.title} is not a 2D spectrum image: fitting every pixel needs one")
    _, pixels = _spectrum(request, dataset_id, body.source)
    si.materialise()
    try:
        # Fail now, not in the job (the model is built again there).
        curve_fit.check(si.signal.inav[0, 0], body.model)
    except ValueError as e:
        raise HTTPException(422, str(e))
    version = si.data_version

    def run(job: Job) -> dict:
        job.report(0.01, "Building the model")
        maps, fits = curve_fit.fit_pixels(
            si.signal, body.model, pixels, lambda f: job.report(0.02 + 0.95 * f, "Fitting each pixel")
        )
        job.report(0.97, "Making the model's spectra")
        model = curve_fit.model_signal(fits)
        job.report(0.99)  # last chance to cancel before the results replace old ones
        made = session.set_curve_model(si, model, fits.spec.model_dump(), maps)
        si.curve_fits, si.curve_fits_version = fits, version
        failed = int(fits.failed.sum())
        return {"file": session.file_of(si).summary(session), "failed": failed, "dataset_id": made.id}

    return request.app.state.jobs.submit("curve-fit-maps", f"Fit every pixel of {si.title}", run).summary()


@router.get("/api/datasets/{dataset_id}/curve-fit/pixel")
def get_pixel_fit(dataset_id: str, request: Request, x: int, y: int, size: int = 1) -> dict:
    """The curve fit of every pixel at the cursor: the pixel (x, y), or the
    ``size`` × ``size`` block there added up (as ``curve-fit`` returns a fit;
    404 when there is no fit of every pixel of the data as it is)."""
    si = _dataset(request, dataset_id)
    fits = si.current_curve_fits
    if fits is None:
        raise HTTPException(404, f"{si.title} has no curve fit of every pixel")
    h, w = fits.failed.shape
    if not (0 <= x < w and 0 <= y < h) or not 1 <= size <= 64:
        raise HTTPException(422, "The pixel is outside the spectrum image")
    try:
        return curve_fit.pixel_fit(fits, x, y, size)
    except ValueError as e:
        raise HTTPException(422, str(e))


@router.get("/api/datasets/{dataset_id}/curve-model/pixel")
def get_model_pixel(dataset_id: str, request: Request, x: int, y: int, size: int = 1) -> dict:
    """A curve model (origin "curve-model") at the cursor: the pixel (x, y),
    or the ``size`` × ``size`` block there added up, rebuilt from its maps:
    the data of its spectrum image, the model and each component's curve,
    and each parameter (its mean over the block)."""
    ds = _dataset(request, dataset_id)
    session = request.app.state.session
    if ds.origin != "curve-model" or not ds.curve_model:
        raise HTTPException(400, f"{ds.title} is not a curve model")
    si = session.datasets.get(ds.curve_model["source_id"])
    if si is None:
        raise HTTPException(404, "The spectrum image this model was fitted to is gone")
    maps = session.curve_model_maps(ds)
    if "Reduced χ²" not in maps:
        raise HTTPException(404, "The model's maps are gone")
    h, w = maps["Reduced χ²"].shape
    if not (0 <= x < w and 0 <= y < h) or not 1 <= size <= 64:
        raise HTTPException(422, "The pixel is outside the spectrum image")
    si.materialise()
    try:
        spec = curve_fit.ModelSpec.model_validate(ds.curve_model["model"])
        return curve_fit.model_at(si.signal, spec, maps, x, y, size)
    except ValueError as e:
        raise HTTPException(422, str(e))
