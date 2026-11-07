"""Spectrum-image interaction (point/region spectra, energy maps) and elements."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from ..processing.element_maps import MapOptions
from ..processing.fitting import FitOptions, FitSource, fit_spectrum, source_spectrum
from ..processing.regions import (
    Region, processed_energy_map, processed_pixel_spectrum, processed_region_spectrum,
)
from ..processing.xray import element_table
from ..session import DatasetKind, NotRebinnable
from .datasets import _dataset, array_response

router = APIRouter(tags=["eds"])


def _spectrum_image(request: Request, dataset_id: str):
    ds = _dataset(request, dataset_id)
    if ds.kind is not DatasetKind.SPECTRUM_IMAGE:
        raise HTTPException(400, f"{ds.title} is not a spectrum image")
    ds.materialise()
    return ds


@router.get("/api/datasets/{dataset_id}/pixel-spectrum")
def get_pixel_spectrum(dataset_id: str, request: Request, x: int, y: int, size: int = 1) -> Response:
    """Summed spectrum of the size × size block around pixel (x, y)."""
    ds = _spectrum_image(request, dataset_id)
    try:
        spectrum, n = processed_pixel_spectrum(ds.signal.data, x, y, max(1, min(size, 64)), ds.filter_spec)
    except IndexError as e:
        raise HTTPException(416, str(e))
    return array_response(spectrum, pixels=n)


@router.post("/api/datasets/{dataset_id}/region-spectrum")
def post_region_spectrum(dataset_id: str, region: Region, request: Request) -> Response:
    """Summed spectrum over a rectangle, ellipse or polygon of the scan (of
    the processed spectrum image)."""
    ds = _spectrum_image(request, dataset_id)
    spectrum, n = processed_region_spectrum(ds.signal.data, region, ds.filter_spec)
    return array_response(spectrum, pixels=n)


@router.get("/api/datasets/{dataset_id}/energy-map")
def get_energy_map(dataset_id: str, request: Request, lo: float, hi: float) -> Response:
    """Counts in the energy window [lo, hi] (spectral-axis units) at every pixel."""
    ds = _spectrum_image(request, dataset_id)
    axis = ds.signal.axes_manager.signal_axes[0]
    counts = processed_energy_map(ds.signal.data, axis.offset, axis.scale, lo, hi, ds.filter_spec)
    return array_response(counts, lo=lo, hi=hi)


@router.get("/api/elements")
def get_elements() -> list[dict]:
    """Periodic table data: symbol, Z, name and X-ray lines (keV, weight)."""
    return element_table()


class ElementsBody(BaseModel):
    elements: list[str]


def _with_file(request: Request, ds, **extra) -> dict:
    """The dataset's summary plus its file's (whose maps may have changed)."""
    session = request.app.state.session
    return {"dataset": ds.summary(), "file": session.file_of(ds).summary(session), **extra}


@router.put("/api/datasets/{dataset_id}/elements")
def put_elements(dataset_id: str, body: ElementsBody, request: Request) -> dict:
    """Set the elements of an EDS spectrum or spectrum image.

    For a spectrum image, newly added elements get a computed map and removed
    ones lose theirs. Returns the dataset, its file and any warnings.
    """
    ds = _dataset(request, dataset_id)
    if ds.summary().get("eds") is None:
        raise HTTPException(400, f"{ds.title} is not an EDS dataset")
    try:
        warnings = request.app.state.session.set_elements(ds, body.elements)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return _with_file(request, ds, warnings=warnings)


@router.put("/api/datasets/{dataset_id}/map-options")
def put_map_options(dataset_id: str, options: MapOptions, request: Request) -> dict:
    """Change how a spectrum image's element maps are computed (recomputes them)."""
    ds = _spectrum_image(request, dataset_id)
    request.app.state.session.set_map_options(ds, options.model_dump())
    return _with_file(request, ds)


@router.delete("/api/datasets/{dataset_id}")
def delete_dataset(dataset_id: str, request: Request) -> dict:
    """Delete an element map (also unticks its element). Returns its file."""
    session = request.app.state.session
    ds = _dataset(request, dataset_id)
    entry = session.file_of(ds)
    try:
        session.delete_dataset(ds)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"file": entry.summary(session)}


class RebinBody(BaseModel):
    # One integer factor per axis, in the summary's axis order.
    factors: list[int]


@router.post("/api/datasets/{dataset_id}/rebin")
def rebin_dataset(dataset_id: str, body: RebinBody, request: Request) -> dict:
    """Rebin a spectrum image's loaded data in place (HyperSpy rebin), with
    the images on its scan (same x/y factors); its maps are recomputed.
    Returns the spectrum image and its file."""
    ds = _dataset(request, dataset_id)
    try:
        request.app.state.session.rebin(ds, body.factors)
    except NotRebinnable as e:
        raise HTTPException(400, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))
    return _with_file(request, ds)


class FitBody(BaseModel):
    source: FitSource = FitSource()
    options: FitOptions = FitOptions()


@router.post("/api/datasets/{dataset_id}/fit")
def fit(dataset_id: str, body: FitBody, request: Request) -> dict:
    """Fit one spectrum (a spectrum, or all pixels / a pixel block / a region
    of a spectrum image) with exSpy's EDS model."""
    ds = _dataset(request, dataset_id)
    try:
        spec, pixels = source_spectrum(ds, body.source)
        result = fit_spectrum(spec, body.options)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {**result, "pixels": pixels}
