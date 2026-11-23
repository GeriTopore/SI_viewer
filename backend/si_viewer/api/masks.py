"""Masks made from images (processing/masks.py)."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..processing import masks
from ..processing.filters import filter_image
from ..session import DatasetKind

router = APIRouter(tags=["masks"])


def _image(request: Request, dataset_id: str):
    session = request.app.state.session
    try:
        ds = session.get_dataset(dataset_id)
    except KeyError:
        raise HTTPException(404, f"Unknown dataset id: {dataset_id}")
    if ds.kind is not DatasetKind.IMAGE or ds.signal.data.ndim != 2:
        raise HTTPException(400, "A mask is made from an image")
    ds.materialise()
    return ds, filter_image(np.asarray(ds.signal.data, dtype=np.float64), ds.filter_spec)


class ValuesBody(BaseModel):
    smooth: float = 0.0


@router.post("/api/datasets/{dataset_id}/mask/values")
def mask_values(dataset_id: str, body: ValuesBody, request: Request) -> dict:
    """A sample of the image's values (smoothed by ``smooth`` px) for a
    histogram, their ``min``, ``max`` and Otsu threshold."""
    _, image = _image(request, dataset_id)
    if body.smooth < 0:
        raise HTTPException(422, "Smoothing can't be negative")
    try:
        return masks.threshold_values(image, body.smooth)
    except ValueError as e:
        raise HTTPException(422, str(e))


class MaskBody(masks.MaskOptions):
    # A mask of the image to make again (keeping its name); none: a new one.
    replace: str | None = None


@router.post("/api/datasets/{dataset_id}/mask")
def make_mask(dataset_id: str, body: MaskBody, request: Request) -> dict:
    """A mask of the image with these options, a dataset under it (or
    ``replace`` made again). Returns its ``file`` and ``id``."""
    session = request.app.state.session
    ds, image = _image(request, dataset_id)
    replace = None
    if body.replace:
        try:
            replace = session.get_dataset(body.replace)
        except KeyError:
            raise HTTPException(404, f"Unknown dataset id: {body.replace}")
    opts = masks.MaskOptions(**body.model_dump(exclude={"replace"}))
    try:
        made = session.set_mask(ds, masks.make_mask(image, opts), opts.model_dump(), replace)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"file": session.file_of(ds).summary(session), "id": made.id}
