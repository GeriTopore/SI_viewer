"""Pre-processing: non-destructive filters on a dataset (M5)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..processing.filters import Filters
from .datasets import _dataset
from .eds import _with_file

router = APIRouter(tags=["process"])


@router.put("/api/datasets/{dataset_id}/filters")
def put_filters(dataset_id: str, filters: Filters, request: Request) -> dict:
    """Set a dataset's filters (all "none" removes them). The loaded data is
    unchanged; the filters apply whenever it is shown. Returns the dataset,
    its file (a spectrum image's maps follow its spatial filter) and notes."""
    ds = _dataset(request, dataset_id)
    try:
        notes = request.app.state.session.set_filters(ds, filters.model_dump())
    except ValueError as e:
        raise HTTPException(422, str(e))
    return _with_file(request, ds, notes=notes)
