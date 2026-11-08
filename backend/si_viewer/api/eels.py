"""Low-loss / core-loss pairs of EELS spectrum images (M10c)."""

from __future__ import annotations

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


@router.get("/api/datasets/{dataset_id}/eels-partners")
def get_eels_partners(dataset_id: str, request: Request) -> list[dict]:
    """The spectrum images this one could be linked to: the low loss (for a
    core loss) or core loss (for a low loss) with the same scan, whether
    linked elsewhere or not. ``linked_to`` is the id of the one it is linked to."""
    session = request.app.state.session
    ds = _eels_si(request, dataset_id)
    return [
        {"id": p.id, "title": p.title, "file_id": p.file_id, "linked_to": p.eels_partner_id}
        for p in session.eels_partners(ds)
    ]


class PartnerBody(BaseModel):
    partner_id: str | None


@router.put("/api/datasets/{dataset_id}/eels-partner")
def put_eels_partner(dataset_id: str, body: PartnerBody, request: Request) -> dict:
    """Link an EELS spectrum image to a partner (replacing any link either had),
    or unlink it (``partner_id`` null). Returns ``files``: every file whose
    datasets changed (this one's, the partner's and any old partner's)."""
    session = request.app.state.session
    ds = _eels_si(request, dataset_id)
    partner = None if body.partner_id is None else _eels_si(request, body.partner_id)
    try:
        changed = session.set_eels_partner(ds, partner)
    except ValueError as e:
        raise HTTPException(400, str(e))
    file_ids = list(dict.fromkeys(d.file_id for d in changed))
    return {"files": [session.files[f].summary(session) for f in file_ids]}
