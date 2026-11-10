"""Saving datasets (and all of a file's maps) as downloads."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

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
    "pdf": "application/pdf",
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


class SaveBody(BaseModel):
    format: str
    # The file to write (a full path; "~" allowed).
    path: str
    overwrite: bool = False


class BatchBody(BaseModel):
    ids: list[str]
    format: str
    # The folder to write into; each dataset is named after its title.
    folder: str
    overwrite: bool = False


def _target_folder(folder: str) -> Path:
    path = Path(folder).expanduser()
    if not path.is_dir():
        raise HTTPException(404, f"Not a folder: {path}")
    return path


def _write(path: Path, data: bytes) -> None:
    try:
        path.write_bytes(data)
    except OSError as e:
        raise HTTPException(422, f"Could not write {path}: {e.strerror or e}")


@router.post("/api/datasets/{dataset_id}/save")
def save_dataset(dataset_id: str, body: SaveBody, request: Request) -> dict:
    """Write the dataset as ``format`` to ``path`` (409 if the file exists
    and ``overwrite`` is not set). Returns the path written."""
    ds = _dataset(request, dataset_id)
    path = Path(body.path).expanduser()
    _target_folder(str(path.parent))
    if path.is_dir():
        raise HTTPException(422, f"{path} is a folder")
    if body.format not in {f for f, _, _ in FORMATS.get(ds.kind, [])}:
        raise HTTPException(422, f"{ds.title} can't be saved as {body.format}")
    if path.exists() and not body.overwrite:
        raise HTTPException(409, f"{path.name} already exists")
    try:
        data, _ = export_bytes(ds, body.format)
    except ValueError as e:
        raise HTTPException(422, str(e))
    _write(path, data)
    return {"path": str(path)}


@router.post("/api/export/batch")
def batch_export(body: BatchBody, request: Request) -> dict:
    """Write each dataset as ``format`` into ``folder``, named after its
    title (made unique). 409 lists the files that exist unless ``overwrite``."""
    session = request.app.state.session
    folder = _target_folder(body.folder)
    try:
        datasets = [session.get_dataset(i) for i in body.ids]
    except KeyError as e:
        raise HTTPException(404, f"Unknown dataset id: {e.args[0]}")
    if not datasets:
        raise HTTPException(422, "Nothing to export")
    names: list[str] = []
    for ds in datasets:
        known = {f: ext for f, ext, _ in FORMATS.get(ds.kind, [])}
        if body.format not in known:
            raise HTTPException(422, f"{ds.title} can't be saved as {body.format}")
        stem, k = safe_name(ds.title), 2
        name = f"{stem}.{known[body.format]}"
        while name in names:
            name, k = f"{stem}_{k}.{known[body.format]}", k + 1
        names.append(name)
    existing = [n for n in names if (folder / n).exists()]
    if existing and not body.overwrite:
        raise HTTPException(409, f"Already in the folder: {', '.join(existing)}")
    written = []
    for ds, name in zip(datasets, names):
        try:
            data, _ = export_bytes(ds, body.format)
        except ValueError as e:
            raise HTTPException(422, str(e))
        _write(folder / name, data)
        written.append(str(folder / name))
    return {"paths": written}


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
