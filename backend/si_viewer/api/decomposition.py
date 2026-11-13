"""Decomposition of spectrum images with HyperSpy's algorithms (M19a)."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..jobs import Job
from .datasets import _dataset, array_response

router = APIRouter(tags=["decomposition"])


class PcaBody(BaseModel):
    # The energy window (the whole axis where not given).
    lo: float | None = None
    hi: float | None = None
    components: int = Field(32, ge=1, le=256)
    # Weight by the counting noise (Keenan & Kotula): for counts.
    poisson: bool = True
    algorithm: str = "SVD"
    # Each spectrum normalised first: {mode: "range" | "value" | "mean", lo, hi}.
    normalise: dict | None = None


@router.post("/api/datasets/{dataset_id}/pca")
def post_pca(dataset_id: str, body: PcaBody, request: Request) -> dict:
    """Decompose a spectrum image (a job). Its result: ``pca_id``, the PCA
    stack made (or remade) under it, and the ``file``."""
    session = request.app.state.session
    ds = _dataset(request, dataset_id)
    if not session.can_decompose(ds):
        raise HTTPException(400, f"{ds.title} is not an EELS or EDS spectrum image")
    from ..session import DECOMPOSITIONS

    if body.algorithm not in DECOMPOSITIONS:
        raise HTTPException(422, f"Unknown decomposition algorithm {body.algorithm!r}")
    n = body.normalise
    if n is not None and (n.get("mode") not in ("range", "value", "mean") or (n["mode"] != "range" and n.get("lo") is None)):
        raise HTTPException(422, "Normalise to 0–1, a value at an energy or a mean over a window")
    try:
        session.pca_channels(ds, body.lo, body.hi)
    except ValueError as e:
        raise HTTPException(422, str(e))

    def run(job: Job) -> dict:
        pca = session.make_pca(
            ds, body.lo, body.hi, body.components, body.poisson, report=job.report, algorithm=body.algorithm,
            normalise=body.normalise,
        )
        return {"pca_id": pca.id, "file": session.file_of(ds).summary(session)}

    name = DECOMPOSITIONS[body.algorithm][0]
    return request.app.state.jobs.submit("pca", f"{name} of {ds.title}", run).summary()


class DenoiseBody(BaseModel):
    n: int = Field(..., ge=1)


@router.post("/api/datasets/{dataset_id}/pca/denoise")
def post_denoise(dataset_id: str, body: DenoiseBody, request: Request) -> dict:
    """The decomposed spectrum image rebuilt from the PCA's first ``n``
    components: a new spectrum image under it (``id``)."""
    session = request.app.state.session
    pca = _dataset(request, dataset_id)
    try:
        made = session.denoise_pca(pca, body.n)
    except (KeyError, ValueError) as e:
        raise HTTPException(422, str(e).strip("'"))
    return {"id": made.id, "file": session.file_of(made).summary(session)}


@router.get("/api/datasets/{dataset_id}/pca/factor")
def get_factor(dataset_id: str, request: Request, index: int = 0):
    """The spectrum of component ``index`` (0-based) of a PCA stack, with its
    energy axis (offset, scale, units) in the meta."""
    session = request.app.state.session
    pca = _dataset(request, dataset_id)
    try:
        factors = session.pca_factors(pca)
    except ValueError as e:
        raise HTTPException(422, str(e))
    data = np.asarray(factors.signal.data)
    if not 0 <= index < len(data):
        raise HTTPException(422, f"Component {index + 1} out of range 1–{len(data)}")
    ax = factors.signal.axes_manager.signal_axes[0]
    units = ax.units if isinstance(ax.units, str) and ax.units != "None" else ""
    return array_response(data[index], offset=float(ax.offset), scale=float(ax.scale), units=units)
