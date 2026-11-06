"""Dataset summaries and metadata."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException, Request, Response

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


def _dataset(request: Request, dataset_id: str):
    try:
        return request.app.state.session.get_dataset(dataset_id)
    except KeyError:
        raise HTTPException(404, f"Unknown dataset id: {dataset_id}")


@router.get("/{dataset_id}")
def get_dataset(dataset_id: str, request: Request) -> dict:
    return _dataset(request, dataset_id).summary()


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
