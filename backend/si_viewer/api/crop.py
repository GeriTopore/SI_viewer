"""Cropping a spectrum image to a region and/or an energy window."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..processing.regions import Region
from .datasets import _dataset

router = APIRouter(tags=["crop"])


class CropBody(BaseModel):
    # A rectangle in navigation pixels (as a region's spectrum).
    region: Region | None = None
    # An energy window.
    lo: float | None = None
    hi: float | None = None


@router.post("/api/datasets/{dataset_id}/crop")
def post_crop(dataset_id: str, body: CropBody, request: Request) -> dict:
    """A new spectrum image under this one, cropped to ``region`` and/or the
    energy window lo–hi: its ``id`` and the ``file``."""
    session = request.app.state.session
    ds = _dataset(request, dataset_id)
    try:
        made = session.crop(ds, body.region, body.lo, body.hi)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"id": made.id, "file": session.file_of(made).summary(session)}
