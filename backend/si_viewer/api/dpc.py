"""DPC and iDPC images from the four quadrants of a segmented detector."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..session import _detector_info
from .datasets import array_response

router = APIRouter(tags=["dpc"])


def _session(request: Request):
    return request.app.state.session


@router.get("/api/datasets/{dataset_id}/dpc")
def get_dpc(dataset_id: str, request: Request) -> dict:
    """What a DPC of the image ``dataset_id`` (a segment image or a DPC
    image) can be made from: ``candidates``, the images of the same size in
    its file or experiment with detector segments recorded (id, title,
    segments, quadrant, origin), and ``choice``, a letter -> id for each
    quadrant found (the images made the same way as this one first), and
    ``settings`` of the DPC images made from that choice, if any."""
    from ..processing.dpc import LETTERS, quadrant

    session = _session(request)
    try:
        ds = session.get_dataset(dataset_id)
    except KeyError:
        raise HTTPException(404, f"Unknown dataset id: {dataset_id}")
    if ds.origin == "dpc" and ds.dpc and ds.dpc.get("source") == "4d":
        raise HTTPException(409, f"{ds.title} was made from 4D-STEM data (its DPC / iCOM process)")
    if ds.origin == "dpc" and ds.dpc and ds.dpc["inputs"].get("A") in session.datasets:
        ds = session.datasets[ds.dpc["inputs"]["A"]]
    entry = session.file_of(ds)
    files = [entry] if entry.experiment_id is None else [
        f for f in session.files.values() if f.experiment_id == entry.experiment_id
    ]
    shape = tuple(ds.signal.data.shape)
    candidates = []
    for f in files:
        for i in f.dataset_ids:
            d = session.datasets[i]
            # Images for an image, stacks (with the same frames) for a stack.
            if d.kind is not ds.kind or tuple(d.signal.data.shape) != shape:
                continue
            det = _detector_info(d.loaded_signal)
            if det is None:
                continue
            try:
                q = quadrant(det["segments"])
            except ValueError:
                continue
            candidates.append({
                "id": d.id, "title": d.title, "segments": det["segments"], "quadrant": LETTERS[q], "origin": d.origin,
            })
    choice = {}
    for letter in LETTERS:
        options = [c for c in candidates if c["quadrant"] == letter]
        if not options:
            continue
        own = [c for c in options if c["id"] == ds.id]
        same = [c for c in options if c["origin"] == ds.origin]
        choice[letter] = (own or same or options)[0]["id"]
    settings = None
    if len(choice) == 4:
        for d in session.datasets.values():
            if d.origin == "dpc" and d.dpc and d.dpc.get("inputs") == choice:
                settings = {k: v for k, v in d.dpc.items() if k not in ("output",)}
                break
    return {"candidates": candidates, "choice": choice, "settings": settings, "recorded": _recorded_rotation(ds)}


def _recorded_rotation(ds) -> float | None:
    """The detector's rotation recorded in a Velox file (degrees, 0–360):
    the rotationalOffset Velox makes its iDPC with. None if not recorded."""
    import math

    value = ds.loaded_signal.original_metadata.get_item("CustomProperties.DetectorRotation.value")
    try:
        return round(math.degrees(float(value)) % 360.0, 2)
    except (TypeError, ValueError):
        return None


class Inputs(BaseModel):
    # The image of each quadrant: {"A": id, "B": id, "C": id, "D": id}.
    inputs: dict[str, str]
    normalise: bool = True


@router.post("/api/dpc/rotation")
def post_rotation(body: Inputs, request: Request) -> dict:
    """The detector's rotation (degrees) and mirror found from the four
    images: the ones leaving the least curl (``curl_ratio``: the curl left,
    relative to the divergence)."""
    try:
        return _session(request).dpc_rotation(body.inputs, body.normalise)
    except (KeyError, ValueError) as e:
        raise HTTPException(422, str(e).strip("'"))


class MakeBody(Inputs):
    theta: float = 0.0
    mirror: bool = False


@router.post("/api/dpc")
def post_dpc(body: MakeBody, request: Request) -> dict:
    """Make DPC x, DPC y and iDPC images from the four images (again, in
    place, if made from them before). Returns their ids and the files
    changed."""
    session = _session(request)
    try:
        made = session.make_dpc(body.inputs, body.normalise, body.theta, body.mirror)
    except (KeyError, ValueError) as e:
        raise HTTPException(422, str(e).strip("'"))
    files = dict.fromkeys(d.file_id for d in made)
    return {
        "ids": {d.dpc["output"]: d.id for d in made},  # colour, x, y, idpc
        "files": [session.files[f].summary(session) for f in files],
    }


@router.get("/api/datasets/{dataset_id}/dpc/frame")
def get_dpc_frame(
    dataset_id: str, request: Request, index: int = 0, projection: str | None = None, bin: int = 1,
):
    """A DPC object as shown: its direction (degrees) and magnitude, shape
    (2, h, w); of frame ``index`` of a DPC stack, or its ``projection``."""
    session = _session(request)
    try:
        ds = session.get_dataset(dataset_id)
        data = session.dpc_frame(ds, index, projection, max(1, bin))
    except KeyError:
        raise HTTPException(404, f"Unknown dataset id: {dataset_id}")
    except (ValueError, IndexError) as e:
        raise HTTPException(422, str(e))
    return array_response(data)


@router.get("/api/datasets/{dataset_id}/dpc/magnitude")
def get_magnitude(dataset_id: str, request: Request):
    """The magnitude of a DPC object's deflection (its brightness)."""
    session = _session(request)
    try:
        ds = session.get_dataset(dataset_id)
        data = session.dpc_magnitude(ds)
    except KeyError:
        raise HTTPException(404, f"Unknown dataset id: {dataset_id}")
    except ValueError as e:
        raise HTTPException(422, str(e))
    return array_response(data)
