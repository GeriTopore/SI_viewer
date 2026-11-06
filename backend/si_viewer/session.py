"""In-memory registry of open files and the datasets (signals) they contain."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .io import load_file


class DatasetKind(str, Enum):
    IMAGE = "image"
    STACK = "stack"
    SPECTRUM = "spectrum"
    SPECTRUM_IMAGE = "spectrum_image"
    OTHER = "other"


def classify(signal) -> DatasetKind:
    am = signal.axes_manager
    nav, sig = am.navigation_dimension, am.signal_dimension
    if sig == 2 and nav == 0:
        return DatasetKind.IMAGE
    if sig == 2 and nav == 1:
        return DatasetKind.STACK
    if sig == 1 and nav == 0:
        return DatasetKind.SPECTRUM
    if sig == 1 and nav == 2:
        return DatasetKind.SPECTRUM_IMAGE
    return DatasetKind.OTHER


def _is_element_map(signal, kind: DatasetKind) -> bool:
    """Velox element maps are images titled with an element symbol."""
    if kind is not DatasetKind.IMAGE:
        return False
    from exspy.material import elements

    return signal.metadata.General.get_item("title", "") in elements.keys()


def _new_id() -> str:
    return uuid.uuid4().hex[:10]


def _defined(value) -> str:
    """Axis names/units are traits ``Undefined`` (or the string "None") when unset."""
    return value if isinstance(value, str) and value != "None" else ""


@dataclass
class Dataset:
    id: str
    file_id: str
    signal: object
    kind: DatasetKind
    is_element_map: bool = False
    thumbnails: dict[int, bytes] = field(default_factory=dict, repr=False)

    @property
    def title(self) -> str:
        return self.signal.metadata.General.get_item("title", "") or "untitled"

    @property
    def lazy(self) -> bool:
        return bool(getattr(self.signal, "_lazy", False))

    def materialise(self) -> None:
        """Load the data into RAM (no-op if already in memory)."""
        if self.lazy:
            self.signal.compute(close_file=False, show_progressbar=False)

    def summary(self) -> dict:
        s = self.signal
        am = s.axes_manager
        return {
            "id": self.id,
            "file_id": self.file_id,
            "title": self.title,
            "kind": self.kind.value,
            "signal_type": s.metadata.Signal.get_item("signal_type", "") or "",
            "is_element_map": self.is_element_map,
            "navigation_shape": list(am.navigation_shape),
            "signal_shape": list(am.signal_shape),
            "dtype": str(s.data.dtype),
            "nbytes": int(s.data.nbytes),
            "lazy": self.lazy,
            "axes": [
                {
                    "name": _defined(ax.name),
                    "size": int(ax.size),
                    "scale": float(ax.scale),
                    "offset": float(ax.offset),
                    "units": _defined(ax.units),
                    "navigate": bool(ax.navigate),
                }
                for ax in am._axes
            ],
        }


@dataclass
class FileEntry:
    id: str
    path: Path
    dataset_ids: list[str] = field(default_factory=list)

    def summary(self, session: "Session") -> dict:
        return {
            "id": self.id,
            "path": str(self.path),
            "name": self.path.name,
            "datasets": [session.datasets[d].summary() for d in self.dataset_ids],
        }


class Session:
    def __init__(self) -> None:
        self.files: dict[str, FileEntry] = {}
        self.datasets: dict[str, Dataset] = {}
        self._lock = threading.RLock()

    def open_file(self, path: str | Path) -> FileEntry:
        path = Path(path).expanduser().resolve()
        with self._lock:
            for entry in self.files.values():
                if entry.path == path:
                    return entry
        signals = load_file(path, lazy=True)
        entry = FileEntry(id=_new_id(), path=path)
        datasets = []
        for i, signal in enumerate(signals):
            # e.g. TIFF files carry no title: name them after the file.
            if not signal.metadata.General.get_item("title", ""):
                signal.metadata.General.title = (
                    path.stem if len(signals) == 1 else f"{path.stem} [{i}]"
                )
            kind = classify(signal)
            datasets.append(
                Dataset(
                    id=_new_id(),
                    file_id=entry.id,
                    signal=signal,
                    kind=kind,
                    is_element_map=_is_element_map(signal, kind),
                )
            )
        with self._lock:
            self.files[entry.id] = entry
            for ds in datasets:
                self.datasets[ds.id] = ds
                entry.dataset_ids.append(ds.id)
        return entry

    def close_file(self, file_id: str) -> None:
        with self._lock:
            entry = self.files.pop(file_id)
            for ds_id in entry.dataset_ids:
                self.datasets.pop(ds_id, None)

    def get_dataset(self, dataset_id: str) -> Dataset:
        with self._lock:
            return self.datasets[dataset_id]

    def siblings(self, dataset: Dataset) -> list[Dataset]:
        """All datasets from the same file (including ``dataset``)."""
        with self._lock:
            return [self.datasets[d] for d in self.files[dataset.file_id].dataset_ids]

    def thumbnail(self, dataset: Dataset, size: int) -> bytes:
        from .thumbnails import make_thumbnail

        key = size
        if key not in dataset.thumbnails:
            dataset.thumbnails[key] = make_thumbnail(dataset, self.siblings(dataset), size)
        return dataset.thumbnails[key]
