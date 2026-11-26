"""Server-side file browser, so the frontend can pick files by path.

The backend runs as the user, so this reads whatever they can. On a shared
machine any user can reach a localhost port, so the server needs its token
(``auth.py``) unless started with ``--no-token``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

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


# ---------- dropped files and folders ----------

# A browser gives a dropped item's name, size and modification time, never its
# path: the item is looked for on disk instead, so nothing is copied.

# Seconds a search may take after the quick places have been tried.
LOCATE_BUDGET = 4.0
# How deep folders are walked under the home folder and mounted volumes.
LOCATE_DEPTH = 6
_SKIPPED_DIRS = {"node_modules", "Library", "__pycache__", "site-packages"}


class DroppedChild(BaseModel):
    name: str
    size: int


class DroppedItem(BaseModel):
    name: str
    is_dir: bool = False
    # Files: bytes and modification time (ms since the epoch).
    size: int | None = None
    modified: float | None = None
    # Folders: some of the files directly inside.
    children: list[DroppedChild] = []


class LocateRequest(BaseModel):
    items: list[DroppedItem]
    # Folders to look in first (e.g. those of the open files).
    hints: list[str] = []


def _matches(path: Path, item: DroppedItem) -> bool:
    try:
        if item.is_dir:
            if not path.is_dir():
                return False
            return all((path / c.name).is_file() and (path / c.name).stat().st_size == c.size for c in item.children)
        if not path.is_file():
            return False
        stat = path.stat()
        if item.size is not None and stat.st_size != item.size:
            return False
        # exFAT keeps times to 2 s.
        return item.modified is None or abs(stat.st_mtime - item.modified / 1000) <= 2.5
    except OSError:
        return False


def _spotlight(name: str) -> list[Path]:
    """Paths of that name from macOS' Spotlight index (none elsewhere)."""
    if sys.platform != "darwin" or '"' in name or "\\" in name:
        return []
    try:
        out = subprocess.run(
            ["mdfind", f'kMDItemFSName == "{name}"'], capture_output=True, text=True, timeout=LOCATE_BUDGET,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [Path(line) for line in out.splitlines() if line]


def _walk(roots: list[Path], names: set[str], deadline: float):
    """(name, path) of entries called one of ``names`` under ``roots``, breadth
    first, until the deadline."""
    queue = [(root, 0) for root in roots]
    while queue and time.monotonic() < deadline:
        folder, depth = queue.pop(0)
        try:
            with os.scandir(folder) as entries:
                for entry in entries:
                    if entry.name in names:
                        yield entry.name, Path(entry.path)
                    if depth < LOCATE_DEPTH and not entry.name.startswith(".") and entry.name not in _SKIPPED_DIRS:
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                queue.append((Path(entry.path), depth + 1))
                        except OSError:
                            pass
        except OSError:
            continue


@router.post("/locate")
def locate(body: LocateRequest) -> dict:
    """Where the dropped files and folders are on disk: ``paths``, one per item
    (null when not found). Looked for in ``hints`` and the usual folders, then
    in Spotlight's index, then by walking the home folder and mounted volumes
    for a few seconds. A file must have the dropped one's size and
    modification time; a folder must hold its files."""
    home = Path.home()
    quick = [Path(h).expanduser() for h in body.hints]
    quick += [p.parent for p in quick]
    quick += [home / "Desktop", home / "Downloads", home / "Documents", home]
    found: dict[int, Path] = {}
    for i, item in enumerate(body.items):
        for folder in quick:
            if _matches(folder / item.name, item):
                found[i] = folder / item.name
                break
    for i, item in enumerate(body.items):
        if i not in found:
            match = next((p for p in _spotlight(item.name)[:50] if _matches(p, item)), None)
            if match is not None:
                found[i] = match
    missing = {i for i in range(len(body.items)) if i not in found}
    if missing:
        volumes = Path("/Volumes")
        roots = [home] + (sorted(p for p in volumes.iterdir() if not p.is_symlink()) if volumes.is_dir() else [])
        names = {body.items[i].name for i in missing}
        for name, path in _walk(roots, names, time.monotonic() + LOCATE_BUDGET):
            for i in sorted(missing):
                if body.items[i].name == name and _matches(path, body.items[i]):
                    found[i] = path
                    missing.discard(i)
            if not missing:
                break
    return {"paths": [str(found[i].resolve()) if i in found else None for i in range(len(body.items))]}
