"""Spectrum-image interaction (point/region spectra, energy maps) and elements."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from ..processing.element_maps import MapOptions
from ..processing.fitting import FitOptions, FitSource, fit_spectrum, source_spectrum
from ..processing.regions import (
    Region, processed_energy_map, processed_pixel_spectrum, processed_region_spectrum,
)
from ..processing.xray import element_table
from ..session import DatasetKind, NotRebinnable, si_cube
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
    """Summed spectrum of the size × size block around pixel (x, y). For a
    line scan x is the position and y is 0."""
    ds = _spectrum_image(request, dataset_id)
    try:
        spectrum, n = processed_pixel_spectrum(si_cube(ds), x, y, max(1, min(size, 64)), ds.filter_spec)
    except IndexError as e:
        raise HTTPException(416, str(e))
    return array_response(spectrum, pixels=n)


@router.post("/api/datasets/{dataset_id}/region-spectrum")
def post_region_spectrum(dataset_id: str, region: Region, request: Request) -> Response:
    """Summed spectrum over a rectangle, ellipse or polygon of the scan (of
    the processed spectrum image)."""
    ds = _spectrum_image(request, dataset_id)
    spectrum, n = processed_region_spectrum(si_cube(ds), region, ds.filter_spec)
    return array_response(spectrum, pixels=n)


@router.get("/api/datasets/{dataset_id}/energy-map")
def get_energy_map(dataset_id: str, request: Request, lo: float, hi: float) -> Response:
    """Counts in the energy window [lo, hi] (spectral-axis units) at every
    pixel (a line scan: 1 × positions)."""
    ds = _spectrum_image(request, dataset_id)
    axis = ds.signal.axes_manager.signal_axes[0]
    counts = processed_energy_map(si_cube(ds), axis.offset, axis.scale, lo, hi, ds.filter_spec)
    return array_response(counts, lo=lo, hi=hi)


class WindowMapBody(BaseModel):
    lo: float
    hi: float
    label: str | None = None


@router.post("/api/datasets/{dataset_id}/window-maps")
def post_window_map(dataset_id: str, body: WindowMapBody, request: Request) -> dict:
    """Keep the energy window [lo, hi] (spectral-axis units) of a 2D spectrum
    image as a map dataset in its file ("Keep as map"), named ``label`` or by
    its window (e.g. "640–660 eV"). Returns the new dataset (origin "window",
    ``parent_id`` the spectrum image, ``window`` {lo, hi, units}) and its file."""
    ds = _spectrum_image(request, dataset_id)
    if ds.is_line_scan:
        raise HTTPException(400, f"{ds.title} is a line scan: it has no maps")
    try:
        window_map = request.app.state.session.add_window_map(ds, body.lo, body.hi, body.label)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return _with_file(request, window_map)


class WindowBody(BaseModel):
    lo: float
    hi: float


@router.patch("/api/datasets/{dataset_id}/window")
def patch_window(dataset_id: str, body: WindowBody, request: Request) -> dict:
    """Move a window map to [lo, hi] (spectral-axis units): the map is
    recomputed in place (same id, new revision), and renamed to the new range
    if it carried the automatic name. Returns the map and its file."""
    session = request.app.state.session
    ds = _dataset(request, dataset_id)
    if ds.origin != "window":
        raise HTTPException(400, f"{ds.title} is not a window map")
    try:
        session.set_window_map(ds, body.lo, body.hi)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return _with_file(request, ds)


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


def _with_partner(request: Request, ds, partner, **extra) -> dict:
    """``_with_file`` plus the EELS partner that was processed along with ``ds``
    (or was unlinked from it): its summary as ``partner`` and its file as
    ``partner_file`` (null when it has none or is in ``ds``'s file)."""
    session = request.app.state.session
    out = _with_file(request, ds, partner=None, partner_file=None, **extra)
    if partner is not None and partner.id in session.datasets:
        out["partner"] = partner.summary()
        if partner.file_id != ds.file_id:
            out["partner_file"] = session.file_of(partner).summary(session)
    return out


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


class TechniqueBody(BaseModel):
    technique: Literal["EDS", "EELS"] | None


@router.put("/api/datasets/{dataset_id}/technique")
def put_technique(dataset_id: str, body: TechniqueBody, request: Request) -> dict:
    """Say whether a spectrum or spectrum image is EDS, EELS or neither (null).
    This sets its signal type; leaving EDS removes the computed element maps.
    Returns the dataset and its file."""
    ds = _dataset(request, dataset_id)
    partner = request.app.state.session.partner_of(ds)  # unlinked if the technique changes
    try:
        request.app.state.session.set_technique(ds, body.technique)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _with_partner(request, ds, partner)


@router.put("/api/datasets/{dataset_id}/map-options")
def put_map_options(dataset_id: str, options: MapOptions, request: Request) -> dict:
    """Change how a spectrum image's element maps are computed (recomputes them)."""
    ds = _spectrum_image(request, dataset_id)
    if ds.is_line_scan:
        raise HTTPException(400, f"{ds.title} is a line scan: it has no element maps")
    request.app.state.session.set_map_options(ds, options.model_dump())
    return _with_file(request, ds)


@router.delete("/api/datasets/{dataset_id}")
def delete_dataset(dataset_id: str, request: Request) -> dict:
    """Delete an element map (also unticks its element) or a window map. Returns its file."""
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
    A linked EELS partner is rebinned by the same navigation factors (with the
    images on its scan). Returns the spectrum image and its file, and the
    partner and its file (``partner``, ``partner_file``; null without one)."""
    ds = _dataset(request, dataset_id)
    try:
        request.app.state.session.rebin(ds, body.factors)
    except NotRebinnable as e:
        raise HTTPException(400, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))
    return _with_partner(request, ds, request.app.state.session.partner_of(ds))


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
