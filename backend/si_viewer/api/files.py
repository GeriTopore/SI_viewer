"""Opening, listing and closing files."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api/files", tags=["files"])


class OpenRequest(BaseModel):
    path: str
    # Open it into this experiment (see open_folder).
    experiment_id: str | None = None


class FolderRequest(BaseModel):
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
    if body.experiment_id is not None and body.experiment_id not in session.experiments:
        raise HTTPException(404, f"Unknown experiment id: {body.experiment_id}")
    from ..project import is_project

    if is_project(body.path):
        raise HTTPException(422, f"{Path(body.path).name} is a saved analysis: open it with Open analysis")
    try:
        entry = session.open_file(body.path, body.experiment_id)
    except FileNotFoundError:
        raise HTTPException(404, f"File not found: {body.path}")
    except (OSError, ValueError) as e:
        raise HTTPException(422, f"Could not read {body.path}: {e}")
    return entry.summary(session)


@router.post("/folder")
def open_folder(body: FolderRequest, request: Request) -> dict:
    """Start an experiment of a folder: ``experiment`` (id, name, path) and
    ``paths``, the files to open into it one by one (POST /api/files with its
    id), in order."""
    try:
        experiment, paths = _session(request).open_folder(body.path)
    except FileNotFoundError:
        raise HTTPException(404, f"Not a folder: {body.path}")
    except OSError as e:
        raise HTTPException(422, f"Could not read {body.path}: {e}")
    return {"experiment": experiment.summary(), "paths": [str(p) for p in paths]}


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

