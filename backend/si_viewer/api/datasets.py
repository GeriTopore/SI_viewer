"""Dataset summaries, metadata, display data and calibration."""

from __future__ import annotations

import json

import numpy as np
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from ..processing.frames import NotApplicable, get_frame, get_spectrum, navigator_sources
from ..session import NotAvailable

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


def _dataset(request: Request, dataset_id: str):
    try:
        return request.app.state.session.get_dataset(dataset_id)
    except KeyError:
        raise HTTPException(404, f"Unknown dataset id: {dataset_id}")


def array_response(arr: np.ndarray, **meta) -> Response:
    """Raw little-endian float32 bytes; shape and extra info in headers."""
    arr = np.ascontiguousarray(arr, dtype="<f4")
    return Response(
        arr.tobytes(),
        media_type="application/octet-stream",
        headers={
            "X-Array-Shape": ",".join(map(str, arr.shape)),
            "X-Array-Meta": json.dumps(meta),
            "Cache-Control": "no-store",
        },
    )


@router.get("/{dataset_id}")
def get_dataset(dataset_id: str, request: Request) -> dict:
    return _dataset(request, dataset_id).summary()


@router.post("/{dataset_id}/load")
def load_dataset(dataset_id: str, request: Request) -> dict:
    """Load the dataset into RAM (spectrum images: makes cursor/ROI work fast)."""
    ds = _dataset(request, dataset_id)
    try:
        ds.materialise()
    except NotAvailable as e:
        raise HTTPException(400, str(e))
    return ds.summary()


@router.get("/{dataset_id}/frame")
def frame(
    dataset_id: str, request: Request, index: int = 0, projection: str | None = None,
    bin: int = 1,
) -> Response:
    ds = _dataset(request, dataset_id)
    bin = 1 if ds.is_line_scan else max(1, min(bin, 64))
    try:
        arr = get_frame(ds, index=index, projection=projection, bin=bin)
    except NotApplicable as e:
        raise HTTPException(400, str(e))
    except IndexError as e:
        raise HTTPException(416, str(e))
    return array_response(arr, bin=bin, index=index, projection=projection)


@router.get("/{dataset_id}/spectrum")
def spectrum(dataset_id: str, request: Request) -> Response:
    try:
        arr = get_spectrum(_dataset(request, dataset_id))
    except NotApplicable as e:
        raise HTTPException(400, str(e))
    return array_response(arr)


@router.get("/{dataset_id}/navigators")
def navigators(dataset_id: str, request: Request) -> list[dict]:
    ds = _dataset(request, dataset_id)
    try:
        return navigator_sources(ds, request.app.state.session.siblings(ds))
    except NotApplicable as e:
        raise HTTPException(400, str(e))


class AxisCalibration(BaseModel):
    index: int
    scale: float | None = None
    offset: float | None = None
    units: str | None = None


@router.patch("/{dataset_id}/axes")
def calibrate_axes(dataset_id: str, body: list[AxisCalibration], request: Request) -> dict:
    ds = _dataset(request, dataset_id)
    n_axes = len(ds.signal.axes_manager._axes)
    for cal in body:
        if not 0 <= cal.index < n_axes:
            raise HTTPException(422, f"Axis index {cal.index} out of range")
        try:
            ds.set_axis_calibration(cal.index, cal.scale, cal.offset, cal.units)
        except ValueError as e:
            raise HTTPException(422, str(e))
    return ds.summary()


class TitleBody(BaseModel):
    title: str


@router.put("/{dataset_id}/title")
def rename_dataset(dataset_id: str, body: TitleBody, request: Request) -> dict:
    """Rename a dataset; the name is kept when its data is recomputed."""
    ds = _dataset(request, dataset_id)
    try:
        ds.rename(body.title)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return ds.summary()


class MicroscopeBody(BaseModel):
    beam_energy: float | None = None
    convergence_angle: float | None = None
    collection_angle: float | None = None


@router.patch("/{dataset_id}/microscope")
def set_microscope(dataset_id: str, body: MicroscopeBody, request: Request) -> dict:
    """Set an EELS dataset's beam energy (kV), convergence and collection
    angles (mrad), as its quantification needs."""
    ds = _dataset(request, dataset_id)
    try:
        ds.set_microscope(**body.model_dump())
    except ValueError as e:
        raise HTTPException(422, str(e))
    return ds.summary()


@router.get("/{dataset_id}/thumbnail")
def get_thumbnail(dataset_id: str, request: Request, size: int = 256) -> Response:
    dataset = _dataset(request, dataset_id)
    png = request.app.state.session.thumbnail(dataset, max(32, min(size, 512)))
    # Dataset ids are never reused, so the image can be cached indefinitely.
    return Response(png, media_type="image/png", headers={"Cache-Control": "max-age=31536000, immutable"})


@router.get("/{dataset_id}/metadata")
def get_metadata(dataset_id: str, request: Request) -> dict:
    signal = _dataset(request, dataset_id).signal
    return {
        "metadata": jsonable(signal.metadata.as_dictionary()),
        "original_metadata": jsonable(signal.original_metadata.as_dictionary()),
    }


def jsonable(value):
    """Convert HyperSpy metadata (with NumPy scalars/arrays) to plain JSON types."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist() if value.size <= 1024 else f"<array {value.shape} {value.dtype}>"
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
