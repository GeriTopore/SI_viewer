"""Drift correction of image stacks: the guided registration session (M9)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request, Response
from pydantic import BaseModel

from ..jobs import Job
from ..processing.export import safe_name
from ..processing.registration import (
    CorrelationType, MaskEdits, MaskSettings, MaskType, OutlierSettings, PeakSettings, RegistrationSession,
)
from ..processing.registration_export import report_pdf, shifts_csv
from ..session import Dataset, DatasetKind
from .datasets import _dataset, array_response
from .export import download

router = APIRouter(tags=["registration"])


def _stack(request: Request, dataset_id: str) -> Dataset:
    ds = _dataset(request, dataset_id)
    if ds.kind is not DatasetKind.STACK:
        raise HTTPException(400, f"{ds.title} is not an image stack")
    return ds


def _frames(ds: Dataset):
    """The stack's frames as loaded (alignment comes before any filter, and
    an applied alignment is redone from the loaded frames)."""
    ds.materialise()
    return ds.loaded_signal.data


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
    peaks: PeakSettings | None = None
    outliers: OutlierSettings | None = None
    edits: MaskEdits | None = None


@router.put("/api/datasets/{dataset_id}/registration")
def put_settings(dataset_id: str, body: SettingsBody, request: Request) -> dict:
    s = _session(request, dataset_id)
    s.set_settings(bad_frames=body.bad_frames, mask=body.mask, peaks=body.peaks, outliers=body.outliers, edits=body.edits)
    return s.summary()


@router.get("/api/datasets/{dataset_id}/registration/frame")
def get_frame(dataset_id: str, request: Request, i: int) -> Response:
    """Frame i as loaded (the data the registration uses)."""
    s = _session(request, dataset_id)
    return array_response(_frame_error(s.frame, i), i=i)


@router.get("/api/datasets/{dataset_id}/registration/correlation")
def get_correlation(
    dataset_id: str, request: Request, i: int, j: int, type: MaskType = "hann", n: float = 6.0,
    n2: float = 4.0, theta: float = 0.0, corr: CorrelationType = "cc",
) -> Response:
    """Correlation (cross, mutual or phase) of frames i and j with a Fourier
    mask (a preview: the session is not changed), zero shift at the centre."""
    s = _session(request, dataset_id)
    try:
        mask = MaskSettings(type=type, n=n, n2=n2, theta=theta)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return array_response(_frame_error(s.correlation, i, j, mask, corr), i=i, j=j)


class PeakBody(BaseModel):
    i: int
    j: int
    mask: MaskSettings = MaskSettings()
    peaks: PeakSettings = PeakSettings()


@router.post("/api/datasets/{dataset_id}/registration/peak")
def post_peak(dataset_id: str, body: PeakBody, request: Request) -> dict:
    """How the shift of frames i, j is found with these settings (a preview):
    the shift, the peak, and for a Gaussian fit every candidate with its
    cut-out and fitted model."""
    s = _session(request, dataset_id)
    return _frame_error(s.peak, body.i, body.j, body.mask, body.peaks)


@router.post("/api/datasets/{dataset_id}/registration/shifts")
def start_shifts(dataset_id: str, request: Request) -> dict:
    """Compute every pair's shift with the session's settings, as a job. Its
    result is the session's state (with the shift matrices computed)."""
    s = _session(request, dataset_id)
    pairs = s.count * (s.count - 1) // 2

    def run(job: Job) -> dict:
        job.report(0, f"Correlating {pairs} pairs of frames")
        s.compute_shifts(lambda f: job.report(f, f"Correlating pairs of frames ({round(f * pairs)} of {pairs})"))
        return s.summary()

    ds = _stack(request, dataset_id)
    return request.app.state.jobs.submit("registration-shifts", f"Shifts of {ds.title}", run).summary()


class MaskBody(BaseModel):
    outliers: OutlierSettings = OutlierSettings()
    bad_frames: list[int] = []
    edits: MaskEdits = MaskEdits()


@router.post("/api/datasets/{dataset_id}/registration/outlier-mask")
def post_outlier_mask(dataset_id: str, body: MaskBody, request: Request) -> Response:
    """Which pair shifts the outlier settings and the hand edits keep (1) or
    mask (0), as a (frames, frames) array over the frames registered."""
    s = _session(request, dataset_id)
    mask = _frame_error(s.outlier_mask, body.outliers, body.bad_frames, body.edits)
    return array_response(mask, masked=int((mask == 0).sum()), pairs=int(mask.size))


@router.post("/api/datasets/{dataset_id}/registration/correct")
def post_correct(dataset_id: str, body: MaskBody, request: Request) -> dict:
    """Correct the masked shifts (with these outlier settings, hand edits and
    bad frames) and return each frame's measured drift (px, image x right, y
    down)."""
    s = _session(request, dataset_id)
    s.set_settings(bad_frames=body.bad_frames, outliers=body.outliers, edits=body.edits)
    return _frame_error(s.correct, body.outliers, body.bad_frames, body.edits)


@router.get("/api/datasets/{dataset_id}/registration/corrected")
def get_corrected(dataset_id: str, request: Request) -> Response:
    """The corrected shift matrices, (3, frames, frames): Δx, Δy and which
    entries are known (1) after correction."""
    s = _session(request, dataset_id)
    return array_response(_frame_error(s.corrected_matrices), first=s.first)


@router.get("/api/datasets/{dataset_id}/registration/result")
def get_result(dataset_id: str, request: Request) -> dict:
    """Shift the good frames into register and average them (once per
    correction): the frames kept, the area inside every frame (``rect``) and
    the square cropped from it (x, y, width, height in aligned pixels), and
    each frame's drift."""
    s = _session(request, dataset_id)
    return _frame_error(s.result_summary)


@router.get("/api/datasets/{dataset_id}/registration/aligned")
def get_aligned(dataset_id: str, request: Request, i: int, crop: bool = True) -> Response:
    """Frame i shifted into register, cropped to the square (or whole)."""
    s = _session(request, dataset_id)
    return array_response(_frame_error(s.aligned_frame, i, crop), i=i)


@router.get("/api/datasets/{dataset_id}/registration/average")
def get_average(dataset_id: str, request: Request, crop: bool = True) -> Response:
    """The average of the aligned frames, cropped to the square (or whole)."""
    s = _session(request, dataset_id)
    return array_response(_frame_error(s.average, crop))


@router.post("/api/datasets/{dataset_id}/registration/apply")
def post_apply(dataset_id: str, request: Request) -> dict:
    """Store the result with the stack (shown aligned from now on, the
    loaded frames kept) and add the average to the file as an image."""
    ds = _stack(request, dataset_id)
    s = _session(request, dataset_id)
    session = request.app.state.session
    avg = session.apply_alignment(ds, _frame_error(s.applied))
    return {"dataset": ds.summary(), "average": avg.summary(), "file": session.file_of(ds).summary(session)}


class AlignmentView(BaseModel):
    active: bool


@router.put("/api/datasets/{dataset_id}/alignment")
def put_alignment(dataset_id: str, body: AlignmentView, request: Request) -> dict:
    """Show the stack's frames aligned (active) or as loaded."""
    ds = _stack(request, dataset_id)
    session = request.app.state.session
    _frame_error(session.show_alignment, ds, body.active)
    return {"dataset": ds.summary(), "file": session.file_of(ds).summary(session)}


@router.delete("/api/datasets/{dataset_id}/alignment")
def delete_alignment(dataset_id: str, request: Request) -> dict:
    """Remove the stack's alignment and its average: the frames as loaded."""
    ds = _stack(request, dataset_id)
    session = request.app.state.session
    session.remove_alignment(ds)
    return {"dataset": ds.summary(), "file": session.file_of(ds).summary(session)}


@router.get("/api/datasets/{dataset_id}/registration/matrices")
def get_matrices(dataset_id: str, request: Request) -> Response:
    """The shift matrices, (2, frames, frames): Δx then Δy (px) of frame j
    relative to frame i, for the frames registered (first … last - 1)."""
    s = _session(request, dataset_id)
    m = _frame_error(s.matrices)
    return array_response(m, first=s.first)


@router.get("/api/datasets/{dataset_id}/registration/compare-n")
def get_compare_n(
    dataset_id: str, request: Request, i: int, j: int, type: MaskType = "hann",
    n: list[float] = Query(...), n1: float = 6.0, n2: float = 4.0, theta: float = 0.0,
) -> Response:
    """Cross-correlations of frames i, j for several cutoffs n: (len(n), rows,
    cols). An elliptical mask (n1, n2, theta) keeps its shape as n varies."""
    s = _session(request, dataset_id)
    if not 1 <= len(n) <= 8:
        raise HTTPException(422, "Compare between one and eight values of n")
    try:
        ccs = s.compare_n(i, j, MaskSettings(type=type, n=n1, n2=n2, theta=theta), n)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return array_response(ccs, i=i, j=j, n=n)


def _image_scale(ds: Dataset) -> tuple[float | None, str]:
    """Scale and units of the stack's image x axis, as loaded (the drift is in
    pixels of the loaded frames)."""
    axis = ds.loaded_signal.axes_manager.signal_axes[0]
    return float(axis.scale), axis.units if isinstance(axis.units, str) else ""


@router.get("/api/datasets/{dataset_id}/registration/shifts.csv")
def get_shifts_csv(dataset_id: str, request: Request) -> Response:
    """Each kept frame's shift and drift (px, and Å when calibrated) of the
    current correction, as a CSV with the settings in comment lines."""
    ds = _stack(request, dataset_id)
    s = _session(request, dataset_id)
    info = _frame_error(s.applied)["info"]
    text = shifts_csv(info, ds.title, *_image_scale(ds))
    return download(text.encode("utf-8"), f"{safe_name(ds.title)}_shifts.csv")


@router.get("/api/datasets/{dataset_id}/registration/report.pdf")
def get_report_pdf(dataset_id: str, request: Request) -> Response:
    """A two-page PDF report of the current correction: settings, drift,
    aligned average and its FFT, and the shift matrices."""
    ds = _stack(request, dataset_id)
    s = _session(request, dataset_id)
    pdf = _frame_error(report_pdf, s, ds.title, *_image_scale(ds))
    return download(pdf, f"{safe_name(ds.title)}_alignment_report.pdf")


@router.get("/api/datasets/{dataset_id}/alignment/shifts.csv")
def get_alignment_csv(dataset_id: str, request: Request) -> Response:
    """The shifts and drift of the alignment applied to the stack."""
    ds = _stack(request, dataset_id)
    if ds.alignment is None:
        raise HTTPException(404, f"{ds.title} has no applied alignment")
    text = shifts_csv(ds.alignment, ds.title, *_image_scale(ds))
    return download(text.encode("utf-8"), f"{safe_name(ds.title)}_shifts.csv")
