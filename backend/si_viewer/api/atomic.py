"""Atomic-resolution images with kemstem (M22): FFT peaks."""

from __future__ import annotations

from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..processing import atomic
from ..processing.filters import filter_image
from ..session import DatasetKind

router = APIRouter(tags=["atomic"])


def image_of(request: Request, dataset_id: str) -> np.ndarray:
    """The image as shown (its filters applied); 404/400 for anything else."""
    session = request.app.state.session
    try:
        ds = session.get_dataset(dataset_id)
    except KeyError:
        raise HTTPException(404, f"Unknown dataset id: {dataset_id}")
    if ds.kind is not DatasetKind.IMAGE:
        raise HTTPException(400, "Atomic analysis needs an image")
    ds.materialise()
    return filter_image(np.asarray(ds.signal.data, dtype=np.float64), ds.filter_spec)


class PeaksBody(BaseModel):
    # Spatial frequencies (fy, fx) in cycles per image pixel.
    peaks: list[tuple[float, float]] = Field(max_length=500)
    method: Literal["gauss", "com"] = "gauss"
    window: int = 9
    blur: float = 1.0
    edge: Literal["none", "hann", "pplus"] = "none"
    pattern: Literal["log", "abs"] = "log"


@router.post("/api/datasets/{dataset_id}/atomic/fft-peaks")
def refine_fft_peaks(dataset_id: str, body: PeaksBody, request: Request) -> dict:
    """Refine FFT peaks of the image: ``peaks`` in order, each ``fy``, ``fx``,
    ``failed`` and, for Gaussian fits, ``amplitude``, ``background`` and
    ``sigma``; ``shape``, the FFT's (rows, columns)."""
    image = image_of(request, dataset_id)
    opts = atomic.PeakOptions(body.method, body.window, body.blur, body.edge, body.pattern)
    try:
        opts.check()
    except ValueError as e:
        raise HTTPException(422, str(e))
    peaks = atomic.refine_peaks(image, np.array(body.peaks, dtype=np.float64).reshape(-1, 2), opts)
    return {"peaks": peaks, "shape": list(image.shape)}


class FourierBody(BaseModel):
    # Every peak used: its number as listed (1-based) → frequency (fy, fx).
    peaks: dict[int, tuple[float, float]] = Field(max_length=50)
    # Per peak (those in ``per_peak``): amplitude, grating, phase, strain (ε∥, ε⊥).
    per_peak: list[int] = []
    outputs: list[Literal["amplitude", "grating", "phase", "strain"]] = []
    # Two peaks: the strain tensor from them.
    tensor: tuple[int, int] | None = None
    sigma: float = 20.0
    phase: Literal["lockin", "gpa"] = "lockin"
    lock_sigma: float | None = None
    edge: Literal["none", "hann", "pplus"] = "none"
    mask_threshold: float | None = None


@router.post("/api/datasets/{dataset_id}/atomic/fourier")
def fourier(dataset_id: str, body: FourierBody, request: Request) -> dict:
    """Fourier filtering round FFT peaks and what follows (kemstem), as a
    job: the maps (amplitude, grating, phase, ε∥ and ε⊥ per peak; the strain
    tensor from two) under the image, replacing ones made before from the
    same peaks. The job's result: ``file`` (the image's file) and ``ids``."""
    session = request.app.state.session
    image = image_of(request, dataset_id)
    ds = session.get_dataset(dataset_id)
    opts = atomic.FourierOptions(body.sigma, body.phase, body.lock_sigma, body.edge, body.mask_threshold)
    used = set(body.per_peak) | set(body.tensor or ())
    try:
        opts.check()
        if not ((body.per_peak and body.outputs) or body.tensor):
            raise ValueError("Choose what to make")
        missing = used - set(body.peaks)
        if missing:
            raise ValueError(f"Unknown peak {min(missing)}")
        if body.tensor:
            a, b = body.tensor
            if a == b or abs(np.linalg.det(np.array([body.peaks[a], body.peaks[b]]))) < 1e-12:
                raise ValueError("The strain tensor needs two peaks in different directions")
    except ValueError as e:
        raise HTTPException(422, str(e))
    peaks = {n: body.peaks[n] for n in used}
    settings = {
        "sigma": opts.sigma, "phase": opts.phase, "lock_sigma": opts.lock_sigma or 2 * opts.sigma,
        "edge": opts.edge, "mask_threshold": opts.mask_threshold,
        "coarsening": atomic.coarsening(image.shape, opts.sigma),
        "freqs": {str(n): list(f) for n, f in peaks.items()},
    }

    def run(job) -> dict:
        made = atomic.fourier_maps(image, peaks, set(body.outputs), opts, body.per_peak, body.tensor,
                                   report=lambda f, msg: job.report(0.95 * f, msg))
        job.report(0.97, "Saving the maps")
        out = session.add_atomic_maps(ds, made, settings)
        return {"file": session.file_of(ds).summary(session), "ids": [d.id for d in out]}

    return request.app.state.jobs.submit("atomic-fourier", f"Fourier filtering of {ds.title}", run).summary()


# ---------- M22d: atomic columns ----------

class ColumnsBody(BaseModel):
    blur: float = 0.5
    distance: int = 4
    threshold: float = 0.1
    window: int = 7

    def options(self) -> atomic.ColumnOptions:
        try:
            return atomic.ColumnOptions(self.blur, self.distance, self.threshold, self.window).check()
        except ValueError as e:
            raise HTTPException(422, str(e))


@router.post("/api/datasets/{dataset_id}/atomic/columns/find")
def find_columns(dataset_id: str, body: ColumnsBody, request: Request) -> dict:
    """The image's atomic columns found by kemstem (``find_columns``), as
    ``columns``: [y, x] pixels (centres at integers)."""
    image = image_of(request, dataset_id)
    found = atomic.find_columns(image, body.options())
    return {"columns": found.tolist()}


class TestBody(ColumnsBody):
    at: tuple[float, float]


@router.post("/api/datasets/{dataset_id}/atomic/columns/test")
def test_column(dataset_id: str, body: TestBody, request: Request) -> dict:
    """One column fitted (at [y, x]): ``data`` and ``fit`` patches and its
    ``row`` of the table (``fields``)."""
    image = image_of(request, dataset_id)
    return {**atomic.test_column(image, body.at, body.options()), "fields": list(atomic.COLUMN_FIELDS)}


class RefineBody(ColumnsBody):
    columns: list[tuple[float, float]] = Field(max_length=500_000)


@router.post("/api/datasets/{dataset_id}/atomic/columns/refine")
def refine_columns(dataset_id: str, body: RefineBody, request: Request) -> dict:
    """Fit every column with a 2D Gaussian (kemstem's ``refine_columns``), as
    a job; the table goes into the image's Columns dataset (made again in
    place). The job's result: ``file`` and ``id`` (the Columns dataset)."""
    session = request.app.state.session
    image = image_of(request, dataset_id)
    ds = session.get_dataset(dataset_id)
    opts = body.options()
    if not body.columns:
        raise HTTPException(422, "Find the columns first")
    columns = np.array(body.columns, dtype=np.float64)
    settings = {"blur": opts.blur, "distance": opts.distance, "threshold": opts.threshold, "window": opts.window}

    def run(job) -> dict:
        table = atomic.refine_columns(image, columns, opts,
                                      report=lambda f: job.report(0.97 * f, f"Fitting {len(columns)} columns"))
        job.report(0.98, "Saving the columns")
        made = session.set_columns(ds, table, settings)
        return {"file": session.file_of(ds).summary(session), "id": made.id}

    return request.app.state.jobs.submit("atomic-columns", f"Fit the columns of {ds.title}", run).summary()


@router.get("/api/datasets/{dataset_id}/atomic/columns")
def get_columns(dataset_id: str, request: Request) -> dict:
    """A Columns dataset's table: ``fields`` and ``rows`` (NaN as null)."""
    session = request.app.state.session
    try:
        ds = session.get_dataset(dataset_id)
    except KeyError:
        raise HTTPException(404, f"Unknown dataset id: {dataset_id}")
    if not (ds.atomic and ds.atomic.get("output") == "columns"):
        raise HTTPException(400, f"{ds.title} is not a table of columns")
    table = np.asarray(ds.signal.data, dtype=np.float64)
    fields = list(atomic.COLUMN_FIELDS)
    # The sublattice of each column, from the sublattices under it.
    labels = session.column_labels(ds)
    if labels is not None:
        table = np.column_stack([table, labels])
        fields.append("sublattice")
    rows = np.where(np.isfinite(table), np.round(table, 4), None).tolist()
    return {"fields": fields, "rows": rows}


def _columns_of(request: Request, dataset_id: str):
    session = request.app.state.session
    try:
        ds = session.get_dataset(dataset_id)
    except KeyError:
        raise HTTPException(404, f"Unknown dataset id: {dataset_id}")
    if not (ds.atomic and ds.atomic.get("output") == "columns"):
        raise HTTPException(400, f"{ds.title} is not a table of columns")
    return ds


class SublatticesBody(BaseModel):
    labels: list[int | None] = Field(max_length=500_000)
    names: list[str] = Field(max_length=len(atomic.SUBLATTICE_NAMES))


@router.put("/api/datasets/{dataset_id}/atomic/columns/sublattices")
def set_sublattices(dataset_id: str, body: SublatticesBody, request: Request) -> dict:
    """Split the columns of a Columns dataset into sublattices (each column's
    index into ``names``; null: none), datasets under it; no names removes
    them. Returns its ``file`` and the sublattices' ``ids``."""
    session = request.app.state.session
    ds = _columns_of(request, dataset_id)
    try:
        made = session.set_sublattices(ds, body.labels, body.names)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"file": session.file_of(ds).summary(session), "ids": [d.id for d in made]}


class KMeansBody(BaseModel):
    values: list[float | None] = Field(max_length=500_000)
    k: int = 2


@router.post("/api/atomic/kmeans")
def kmeans(body: KMeansBody) -> dict:
    """``centres`` of ``k`` clusters of a column value, largest first."""
    values = np.array([np.nan if v is None else v for v in body.values], dtype=np.float64)
    try:
        return {"centres": atomic.kmeans_centres(values, body.k)}
    except ValueError as e:
        raise HTTPException(422, str(e))
