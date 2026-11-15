"""Curve fitting (M21): HyperSpy's 1D components, their starting values, and
the fit of one spectrum (a spectrum, or all pixels / a pixel block / a region
of a spectrum image)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

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
