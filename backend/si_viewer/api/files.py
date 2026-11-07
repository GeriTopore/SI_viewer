"""Opening, listing and closing files."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api/files", tags=["files"])


class OpenRequest(BaseModel):
    path: str


def _session(request: Request):
    return request.app.state.session


@router.get("")
def list_files(request: Request) -> list[dict]:
    session = _session(request)
    return [entry.summary(session) for entry in session.files.values()]


@router.post("")
def open_file(body: OpenRequest, request: Request) -> dict:
    session = _session(request)
    try:
        entry = session.open_file(body.path)
    except FileNotFoundError:
        raise HTTPException(404, f"File not found: {body.path}")
    except (OSError, ValueError) as e:
        raise HTTPException(422, f"Could not read {body.path}: {e}")
    return entry.summary(session)


@router.delete("/{file_id}", status_code=204)
def close_file(file_id: str, request: Request) -> None:
    try:
        _session(request).close_file(file_id)
    except KeyError:
        raise HTTPException(404, f"Unknown file id: {file_id}")
    # Registration sessions of its stacks go too (their FFTs can be large).
    alive = _session(request).datasets
    for ds_id in [k for k in request.app.state.registrations if k not in alive]:
        del request.app.state.registrations[ds_id]

