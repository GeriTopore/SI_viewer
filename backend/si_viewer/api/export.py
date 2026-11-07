"""Saving datasets (and all of a file's maps) as downloads."""

from __future__ import annotations

from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Request, Response

from ..processing.export import FORMATS, export_bytes, maps_zip, safe_name
from ..session import DatasetKind
from .datasets import _dataset

router = APIRouter(tags=["export"])

MEDIA_TYPES = {
    "tif": "image/tiff",
    "csv": "text/csv",
    "msa": "text/plain",
    "hspy": "application/x-hdf5",
    "zip": "application/zip",
}


def download(data: bytes, name: str) -> Response:
    """``data`` as a file download called ``name`` (any characters)."""
    ext = name.rsplit(".", 1)[-1]
    ascii_name = name.encode("ascii", "replace").decode().replace("?", "_")
    return Response(
        data,
        media_type=MEDIA_TYPES.get(ext, "application/octet-stream"),
        headers={
            "Content-Disposition": f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(name)}",
            "Cache-Control": "no-store",
        },
    )


@router.get("/api/export-formats")
def export_formats() -> dict[str, list[dict]]:
    """Formats each kind of dataset can be saved in."""
    return {
        kind.value: [{"format": f, "extension": ext, "label": label} for f, ext, label in fmts]
        for kind, fmts in FORMATS.items()
    }


@router.get("/api/datasets/{dataset_id}/export")
def export_dataset(dataset_id: str, request: Request, format: str) -> Response:
    """The dataset as a file: the data as shown (filters applied); a
    spectrum image as loaded, with its processing in the metadata."""
    ds = _dataset(request, dataset_id)
    try:
        data, name = export_bytes(ds, format)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return download(data, name)


@router.get("/api/files/{file_id}/maps")
def export_maps(file_id: str, request: Request, format: str = "tiff") -> Response:
    """Every element map of a file, as a zip."""
    session = request.app.state.session
    if file_id not in session.files:
        raise HTTPException(404, f"Unknown file id: {file_id}")
    entry = session.files[file_id]
    maps = [d for d in session.siblings_in_file(file_id) if d.is_element_map and d.kind is DatasetKind.IMAGE]
    if not maps:
        raise HTTPException(404, "This file has no element maps")
    try:
        data = maps_zip(maps, format)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return download(data, f"{safe_name(entry.label or entry.path.stem)}_maps.zip")
