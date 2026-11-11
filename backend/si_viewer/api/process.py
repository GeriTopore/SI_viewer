"""Pre-processing: non-destructive filters on a dataset (M5); line scans cut
from a spectrum image."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..processing.filters import Filters
from ..processing.line_scan import LineSpec
from .datasets import _dataset
from .eds import _with_linked

router = APIRouter(tags=["process"])


@router.put("/api/datasets/{dataset_id}/filters")
def put_filters(dataset_id: str, filters: Filters, request: Request) -> dict:
    """Set a dataset's filters (all "none" removes them). The loaded data is
    unchanged; the filters apply whenever it is shown. Returns the dataset,
    its file (a spectrum image's maps follow its spatial filter) and notes. The
    spatial filter of a linked EELS spectrum image is applied to the ones it is
    linked to too: ``partners`` and ``partner_files`` carry them (empty without
    any)."""
    ds = _dataset(request, dataset_id)
    try:
        notes = request.app.state.session.set_filters(ds, filters.model_dump())
    except ValueError as e:
        raise HTTPException(422, str(e))
    return _with_linked(request, ds, request.app.state.session.linked_of(ds), notes=notes)


class LineScanBody(BaseModel):
    line: LineSpec
    name: str = Field("Line", max_length=100)
    with_linked: bool = True


@router.post("/api/datasets/{dataset_id}/line-scan")
def post_line_scan(dataset_id: str, body: LineScanBody, request: Request) -> dict:
    """Cut a line scan from a 2D spectrum image along a line (and from the
    spectrum images linked to it, linked in turn). Returns the new line scan
    (``dataset``), its file, and the linked ones as ``partners``."""
    ds = _dataset(request, dataset_id)
    try:
        made = request.app.state.session.add_line_scans(ds, body.line, body.name, body.with_linked)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return _with_linked(request, made[0], made[1:])
