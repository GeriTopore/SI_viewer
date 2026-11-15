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
