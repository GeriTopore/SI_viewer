"""Server-side file browser, so the frontend can pick files by path.

The backend binds to localhost and runs as the user, so this exposes nothing
the user can't already read.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..io import SUPPORTED_SUFFIXES

router = APIRouter(prefix="/api/fs", tags=["filesystem"])


@router.get("/list")
def list_directory(path: str | None = None, ext: str | None = None) -> dict:
    """Folders and the files that can be opened, or with ``ext`` (e.g.
    "tif", for saving) the files with that extension instead."""
    suffixes = {f".{ext.lower().lstrip('.')}"} if ext else SUPPORTED_SUFFIXES
    directory = Path(path).expanduser() if path else Path.home()
    try:
        directory = directory.resolve()
    except OSError as e:
        raise HTTPException(422, str(e))
    if not directory.is_dir():
        raise HTTPException(404, f"Not a directory: {directory}")

    entries = []
    try:
        children = list(directory.iterdir())
    except PermissionError:
        raise HTTPException(403, f"Permission denied: {directory}")
    for child in children:
        if child.name.startswith("."):
            continue
        try:
            is_dir = child.is_dir()
            stat = child.stat()
        except OSError:
            continue
        supported = not is_dir and child.suffix.lower() in suffixes
        if not (is_dir or supported):
            continue
        entries.append(
            {
                "name": child.name,
                "path": str(child),
                "is_dir": is_dir,
                "size": None if is_dir else stat.st_size,
                "modified": stat.st_mtime,
            }
        )
    entries.sort(key=lambda e: (not e["is_dir"], e["name"].lower()))
    parent = directory.parent
    return {
        "path": str(directory),
        "parent": str(parent) if parent != directory else None,
        "entries": entries,
    }
