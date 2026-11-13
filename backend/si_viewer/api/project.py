"""Saving and opening an analysis as a project file (M18)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..jobs import Job
from ..project import ProjectError, estimate, is_project, missing_raw, open_project, save_project

router = APIRouter(tags=["project"])


class SaveBody(BaseModel):
    path: str
    # The frontend's state (views), saved as it is.
    views: dict = {}
    overwrite: bool = False
    # The files to save (with what they link to); all open files if None.
    file_ids: list[str] | None = None
    # "link": raw data referenced; "results": the raw datasets' data stored
    # too; "raw": the raw files embedded (in-situ series: their results stored).
    level: str = "link"


@router.post("/api/project/save")
def post_save(body: SaveBody, request: Request) -> dict:
    """Save the open analysis as a project (``.hspy``) at ``path``: the raw
    files referenced, every result stored, the views. 409 if the file exists
    and ``overwrite`` is not set. Returns where it was saved."""
    from pathlib import Path

    target = Path(body.path).expanduser()
    if not target.suffix:
        target = target.with_suffix(".hspy")
    if target.exists() and not body.overwrite:
        raise HTTPException(409, f"{target.name} already exists")
    if target.exists() and not is_project(target):
        raise HTTPException(409, f"{target.name} is not a saved analysis: choose another name")
    session = request.app.state.session
    if not session.files:
        raise HTTPException(422, "Nothing to save: open some data first")
    try:
        saved = save_project(session, target, body.views, body.level, body.file_ids)
    except (ProjectError, OSError) as e:
        raise HTTPException(422, str(e))
    return {"path": str(saved)}


class EstimateBody(BaseModel):
    file_ids: list[str] | None = None


@router.post("/api/project/estimate")
def post_estimate(body: EstimateBody, request: Request) -> dict:
    """Bytes a save of ``file_ids`` would store at each level (before
    compression), and ``file_ids``: the files it would hold (those and what
    they link to)."""
    try:
        return estimate(request.app.state.session, body.file_ids)
    except ProjectError as e:
        raise HTTPException(404, str(e))


class OpenBody(BaseModel):
    path: str
    # Where raw files that moved are: file id -> path.
    locations: dict[str, str] = {}


@router.post("/api/project/check")
def post_check(body: OpenBody) -> dict:
    """The raw files of a saved analysis that can't be found (with
    ``locations``) and are neither embedded nor stored: ``missing``, {id,
    name, path} each; they have to be located before it opens."""
    from pathlib import Path

    from ..project import read_project

    path = Path(body.path).expanduser()
    try:
        project, _ = read_project(path)
    except ProjectError as e:
        raise HTTPException(422, str(e))
    return {"missing": missing_raw(project, path.resolve().parent, body.locations)}


@router.post("/api/project/open")
def post_open(body: OpenBody, request: Request) -> dict:
    """Open a saved analysis next to what is open: its raw files opened
    again, the operations redone, the results it stored. Returns every open
    file (``files``), the ones of the analysis (``file_ids``), the views
    saved with it and warnings (raw files changed or not found). 422
    (nothing changed) if a raw file can't be found or is open already.
    Drift corrections are redone after, as a job (``job``, or None); its
    result's ``files`` are the files it changed."""
    from pathlib import Path

    from ..project import finish_open

    session = request.app.state.session
    path = Path(body.path).expanduser()
    before = set(session.files)
    deferred = []
    try:
        views, warnings = open_project(session, path, body.locations, defer=deferred)
    except ProjectError as e:
        for file_id in [f for f in session.files if f not in before]:
            session.close_file(file_id)
        raise HTTPException(422, str(e))
    job = None
    if deferred:
        def run(job: Job) -> dict:
            done = finish_open(session, deferred, job.report)
            file_ids = {session.datasets[i].file_id for i in done if i in session.datasets}
            return {"files": [session.files[f].summary(session) for f in file_ids if f in session.files]}

        job = request.app.state.jobs.submit("open-analysis", f"Opening {path.name}", run).summary()
    return {
        "files": [f.summary(session) for f in session.files.values()],
        "file_ids": [f for f in session.files if f not in before],
        "views": views,
        "warnings": warnings,
        "job": job,
    }
