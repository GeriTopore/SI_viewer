"""Linked EELS spectrum images: their roles and link groups (M10c)."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..processing.eels import is_eels_si
from .datasets import _dataset

router = APIRouter(tags=["eels"])


def _eels_si(request: Request, dataset_id: str):
    ds = _dataset(request, dataset_id)
    if not is_eels_si(ds):
        raise HTTPException(400, f"{ds.title} is not an EELS spectrum image")
    return ds


def _files_of(request: Request, changed: list) -> dict:
    """``files``: every file with a dataset in ``changed``."""
    session = request.app.state.session
    file_ids = list(dict.fromkeys(d.file_id for d in changed))
    return {"files": [session.files[f].summary(session) for f in file_ids]}


@router.get("/api/datasets/{dataset_id}/eels-partners")
def get_eels_partners(dataset_id: str, request: Request) -> list[dict]:
    """The spectrum images this one could be linked to: every other EELS
    spectrum image with the same scan, whatever its role. ``group`` is the id
    of the link group it is in (null if none)."""
    session = request.app.state.session
    ds = _eels_si(request, dataset_id)
    return [
        {"id": p.id, "title": p.title, "file_id": p.file_id, "role": p.eels_role, "group": p.eels_group}
        for p in session.link_candidates(ds)
    ]


class PartnerBody(BaseModel):
    partner_id: str | None


@router.put("/api/datasets/{dataset_id}/eels-partner")
def put_eels_partner(dataset_id: str, body: PartnerBody, request: Request) -> dict:
    """Put an EELS spectrum image in ``partner_id``'s link group (a new group
    of the two if it has none), leaving the group it was in, or take it out of
    its group (``partner_id`` null). A group has at most one low loss. Returns
    ``files``: every file whose datasets changed."""
    session = request.app.state.session
    ds = _eels_si(request, dataset_id)
    partner = None if body.partner_id is None else _eels_si(request, body.partner_id)
    try:
        changed = session.set_eels_partner(ds, partner)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _files_of(request, changed)


class RoleBody(BaseModel):
    role: Literal["low_loss", "core_loss"]


@router.put("/api/datasets/{dataset_id}/eels-role")
def put_eels_role(dataset_id: str, body: RoleBody, request: Request) -> dict:
    """Say by hand that an EELS spectrum image is a low loss or a core loss
    (a link group has at most one low loss). Returns ``files``: every file
    whose datasets changed (this one's and its group's)."""
    ds = _eels_si(request, dataset_id)
    try:
        changed = request.app.state.session.set_eels_role(ds, body.role)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _files_of(request, changed)
