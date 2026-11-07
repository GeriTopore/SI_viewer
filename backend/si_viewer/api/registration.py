"""Drift correction of image stacks: the guided registration session (M9)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request, Response
from pydantic import BaseModel

from ..processing.registration import MaskSettings, MaskType, RegistrationSession
from ..session import Dataset, DatasetKind
from .datasets import _dataset, array_response

router = APIRouter(tags=["registration"])


def _stack(request: Request, dataset_id: str) -> Dataset:
    ds = _dataset(request, dataset_id)
    if ds.kind is not DatasetKind.STACK:
        raise HTTPException(400, f"{ds.title} is not an image stack")
    return ds


def _frames(ds: Dataset):
    """The stack's frames as loaded (alignment comes before any filter)."""
    ds.materialise()
    return ds.signal.data


def _session(request: Request, dataset_id: str) -> RegistrationSession:
    ds = _stack(request, dataset_id)
    s = request.app.state.registrations.get(dataset_id)
    if s is None or s.revision != ds.revision:
        raise HTTPException(404, "No registration in progress for this stack: start one first")
    return s


def _frame_error(fn, *args):
    try:
        return fn(*args)
    except ValueError as e:
        raise HTTPException(422, str(e))


class StartBody(BaseModel):
    first: int = 0
    last: int | None = None


@router.post("/api/datasets/{dataset_id}/registration")
def start_registration(dataset_id: str, body: StartBody, request: Request) -> dict:
    """Start (or restart, e.g. with another frame range) registering a stack:
    loads the frames and takes their FFTs. Returns the session's state."""
    ds = _stack(request, dataset_id)
    old = request.app.state.registrations.get(dataset_id)
    try:
        s = RegistrationSession(_frames(ds), body.first, body.last, revision=ds.revision)
    except ValueError as e:
        raise HTTPException(422, str(e))
    if old is not None and old.revision == ds.revision:
        s.set_settings(bad_frames=old.bad_frames, mask=old.mask)
    request.app.state.registrations[dataset_id] = s
    return s.summary()


@router.get("/api/datasets/{dataset_id}/registration")
def get_registration(dataset_id: str, request: Request) -> dict:
    return _session(request, dataset_id).summary()


@router.delete("/api/datasets/{dataset_id}/registration", status_code=204)
def end_registration(dataset_id: str, request: Request) -> None:
    """Drop the session (and the memory its FFTs use)."""
    request.app.state.registrations.pop(dataset_id, None)


class SettingsBody(BaseModel):
    bad_frames: list[int] | None = None
    mask: MaskSettings | None = None


@router.put("/api/datasets/{dataset_id}/registration")
def put_settings(dataset_id: str, body: SettingsBody, request: Request) -> dict:
    s = _session(request, dataset_id)
    s.set_settings(bad_frames=body.bad_frames, mask=body.mask)
    return s.summary()


@router.get("/api/datasets/{dataset_id}/registration/frame")
def get_frame(dataset_id: str, request: Request, i: int) -> Response:
    """Frame i as loaded (the data the registration uses)."""
    s = _session(request, dataset_id)
    return array_response(_frame_error(s.frame, i), i=i)


@router.get("/api/datasets/{dataset_id}/registration/correlation")
def get_correlation(
    dataset_id: str, request: Request, i: int, j: int, type: MaskType = "hann", n: float = 6.0,
) -> Response:
    """Cross-correlation of frames i and j with a Fourier mask (a preview: the
    session's mask is not changed), zero shift at the centre."""
    s = _session(request, dataset_id)
    try:
        mask = MaskSettings(type=type, n=n)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return array_response(_frame_error(s.correlation, i, j, mask), i=i, j=j)


@router.get("/api/datasets/{dataset_id}/registration/compare-n")
def get_compare_n(
    dataset_id: str, request: Request, i: int, j: int, type: MaskType = "hann",
    n: list[float] = Query(...),
) -> Response:
    """Cross-correlations of frames i, j for several cutoffs n: (len(n), rows, cols)."""
    s = _session(request, dataset_id)
    if not 1 <= len(n) <= 8:
        raise HTTPException(422, "Compare between one and eight values of n")
    try:
        ccs = s.compare_n(i, j, type, n)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return array_response(ccs, i=i, j=j, n=n)
