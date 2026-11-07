"""In-memory registry of open files and the datasets (signals) they contain."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

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


# Spectrum images up to this size are loaded into RAM when their file opens
# (raw maps are computed from them straight away); larger ones stay lazy.
SI_IN_MEMORY_LIMIT = 4 << 30


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
    # "file" (loaded) or "computed" (an element map made from a spectrum image).
    origin: str = "file"
    # Element symbol of an element map (Velox maps: their title).
    element: str | None = None
    # Spectrum image a computed map was made from.
    parent_id: str | None = None
    # Bumped whenever the data is recomputed, so clients refresh caches.
    revision: int = 0
    # Spectrum images: how their computed element maps are made.
    map_options: dict = field(default_factory=dict)
    # Rebinning applied to the loaded data, e.g. ["x×2, y×2"] (oldest first).
    rebinned: list[str] = field(default_factory=list)
    thumbnails: dict[int, bytes] = field(default_factory=dict, repr=False)
    # Derived arrays (projections, sum spectra, …) keyed by name.
    cache: dict[str, object] = field(default_factory=dict, repr=False)
    _load_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def title(self) -> str:
        return self.signal.metadata.General.get_item("title", "") or "untitled"

    @property
    def lazy(self) -> bool:
        return bool(getattr(self.signal, "_lazy", False))

    def materialise(self) -> None:
        """Load the data into RAM (no-op if already in memory)."""
        with self._load_lock:
            if self.lazy:
                self.signal.compute(close_file=False, show_progressbar=False)

    def set_axis_calibration(
        self, index: int, scale: float | None = None, offset: float | None = None,
        units: str | None = None,
    ) -> None:
        """Change the calibration of axis ``index`` (array order, as in ``summary``)."""
        axis = self.signal.axes_manager._axes[index]
        if scale is not None:
            if not scale > 0:
                raise ValueError("Scale must be positive")
            axis.scale = scale
        if offset is not None:
            axis.offset = offset
        if units is not None:
            axis.units = units

    def replace_signal(self, signal) -> None:
        """Swap in recomputed data (new revision, caches cleared)."""
        self.signal = signal
        self.revision += 1
        self.thumbnails.clear()
        self.cache.clear()

    def summary(self) -> dict:
        from .processing.element_maps import MapOptions
        from .processing.xray import eds_info

        s = self.signal
        am = s.axes_manager
        eds = eds_info(s)
        if eds is not None and self.kind is DatasetKind.SPECTRUM_IMAGE:
            eds["map_options"] = MapOptions(**self.map_options).model_dump()
        return {
            "eds": eds,
            "origin": self.origin,
            "element": self.element,
            "parent_id": self.parent_id,
            "revision": self.revision,
            "rebinned": list(self.rebinned),
            "xray_line": s.metadata.get_item("Sample.xray_line_map.line"),
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
    # Shown instead of the file name (e.g. an example's title).
    label: str | None = None

    def summary(self, session: "Session") -> dict:
        return {
            "id": self.id,
            "path": str(self.path),
            "name": self.label or self.path.name,
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
        return self.open_signals(path, load_file(path, lazy=True))

    def open_example(self, example_id: str) -> FileEntry:
        """Open one of the bundled example datasets (see ``examples.py``)."""
        from .examples import get_example

        example = get_example(example_id)  # KeyError if unknown
        path = Path(f"example:{example_id}")
        with self._lock:
            for entry in self.files.values():
                if entry.path == path:
                    return entry
        return self.open_signals(path, example.load(), label=example.title)

    def open_signals(self, path: Path, signals: list, label: str | None = None) -> FileEntry:
        """Register loaded signals as one file entry (with raw element maps
        for EDS spectrum images)."""
        entry = FileEntry(id=_new_id(), path=path, label=label)
        datasets = []
        for i, signal in enumerate(signals):
            # e.g. TIFF files carry no title: name them after the file.
            if not signal.metadata.General.get_item("title", ""):
                signal.metadata.General.title = (
                    path.stem if len(signals) == 1 else f"{path.stem} [{i}]"
                )
            kind = classify(signal)
            is_map = _is_element_map(signal, kind)
            datasets.append(
                Dataset(
                    id=_new_id(),
                    file_id=entry.id,
                    signal=signal,
                    kind=kind,
                    is_element_map=is_map,
                    element=signal.metadata.General.title if is_map else None,
                )
            )
        # Element maps stored by Velox are processed (quantified, often
        # filtered). When the file has an EDS spectrum image, raw net-count maps
        # are computed from its spectrum stream instead.
        eds_sis = [
            d for d in datasets
            if d.kind is DatasetKind.SPECTRUM_IMAGE
            and (d.signal.metadata.Signal.get_item("signal_type", "") or "").startswith("EDS")
        ]
        if eds_sis:
            datasets = [d for d in datasets if not d.is_element_map]
        with self._lock:
            self.files[entry.id] = entry
            for ds in datasets:
                self.datasets[ds.id] = ds
                entry.dataset_ids.append(ds.id)
        for si in eds_sis:
            self._compute_initial_maps(si)
        return entry

    def _compute_initial_maps(self, si: Dataset) -> None:
        """Raw maps for the elements recorded with a spectrum image."""
        from .processing.xray import eds_info

        if si.signal.data.nbytes <= SI_IN_MEMORY_LIMIT:
            si.materialise()  # all maps then come from one in-memory copy
        for element in eds_info(si.signal)["elements"]:
            self._add_computed_map(si, element)

    def close_file(self, file_id: str) -> None:
        with self._lock:
            entry = self.files.pop(file_id)
            for ds_id in entry.dataset_ids:
                self.datasets.pop(ds_id, None)

    def get_dataset(self, dataset_id: str) -> Dataset:
        with self._lock:
            return self.datasets[dataset_id]

    def file_of(self, dataset: Dataset) -> FileEntry:
        return self.files[dataset.file_id]

    # ---------- element maps computed from spectrum images ----------

    def computed_maps(self, si: Dataset, element: str | None = None) -> list[Dataset]:
        with self._lock:
            return [
                d for d in self.siblings(si)
                if d.origin == "computed" and d.parent_id == si.id and (element is None or d.element == element)
            ]

    def _remove(self, dataset: Dataset) -> None:
        with self._lock:
            self.datasets.pop(dataset.id, None)
            ids = self.files[dataset.file_id].dataset_ids
            if dataset.id in ids:
                ids.remove(dataset.id)

    def _make_map(self, si: Dataset, element: str):
        """``element``'s map from ``si`` (None if it has no usable line)."""
        from .processing.element_maps import MapOptions, compute_map

        return compute_map(si.signal, element, MapOptions(**si.map_options))

    def _add_computed_map(self, si: Dataset, element: str) -> bool:
        signal = self._make_map(si, element)
        if signal is None:
            return False
        ds = Dataset(
            id=_new_id(), file_id=si.file_id, signal=signal, kind=DatasetKind.IMAGE,
            is_element_map=True, origin="computed", element=element, parent_id=si.id,
        )
        with self._lock:
            self.datasets[ds.id] = ds
            self.files[si.file_id].dataset_ids.append(ds.id)
        return True

    def set_elements(self, ds: Dataset, elements: list[str]) -> list[str]:
        """Set an EDS dataset's elements. For a spectrum image, newly added
        elements get a computed map and removed ones lose theirs (maps loaded
        from the file are kept). Returns warnings, e.g. elements without a line
        in the spectrum's range."""
        from .processing.xray import eds_info, set_elements

        old = set(eds_info(ds.signal)["elements"])
        set_elements(ds.signal, elements)
        new = set(elements)
        warnings = []
        if ds.kind is DatasetKind.SPECTRUM_IMAGE:
            ds.materialise()
            for element in old - new:
                for m in self.computed_maps(ds, element):
                    self._remove(m)
            for element in sorted(new - old):
                if not self.computed_maps(ds, element) and not self._add_computed_map(ds, element):
                    warnings.append(f"{element}: no X-ray line in the spectrum's energy range, so no map")
        return warnings

    def delete_dataset(self, dataset: Dataset) -> None:
        """Delete an element map.

        This also unticks its element in the map's spectrum image and in the
        file's other (non-image) EDS spectra, which removes that element's
        computed maps too. Other data can't be deleted.
        """
        from .processing.xray import eds_info

        if not dataset.is_element_map:
            raise ValueError(f"{dataset.title} is not an element map and can't be deleted")
        self._remove(dataset)
        element = dataset.element
        for d in self.siblings_in_file(dataset.file_id):
            if d.kind is DatasetKind.SPECTRUM_IMAGE and dataset.parent_id not in (None, d.id):
                continue  # another spectrum image in the same file
            info = eds_info(d.signal)
            if info is not None and element in info["elements"]:
                self.set_elements(d, [e for e in info["elements"] if e != element])

    def rebin(self, dataset: Dataset, factors: list[int]) -> Dataset:
        """Rebin the loaded data in place by an integer factor along each axis.

        The rebinned data replaces the loaded data (the file on disk is not
        changed). ``factors`` follow the summary's axis order (array order,
        e.g. y, x, energy). Sizes that don't divide are cropped (HyperSpy's
        rebin). A spectrum image's computed element maps are recomputed at
        the new size.
        """
        am = dataset.signal.axes_manager
        axes = list(am._axes)
        if len(factors) != len(axes):
            raise ValueError(f"Expected {len(axes)} factors, got {len(factors)}")
        for f, ax in zip(factors, axes):
            if not 1 <= f <= ax.size:
                raise ValueError(f"Factor for {ax.name or 'axis'} must be between 1 and {ax.size}")
        if all(f == 1 for f in factors):
            raise ValueError("Choose a factor above 1 for at least one axis")

        # HyperSpy's scale is in natural order (x, y | energy).
        natural = am._get_axes_in_natural_order()
        scale = [factors[next(i for i, a in enumerate(axes) if a is ax)] for ax in natural]
        if dataset.kind is DatasetKind.SPECTRUM_IMAGE:
            dataset.materialise()
        signal = dataset.signal.rebin(scale=scale)
        # Integer sums come back as uint64: keep uint32 when the counts fit.
        if signal.data.dtype == np.uint64 and not getattr(signal, "_lazy", False) and signal.data.max() < 2**32:
            signal.change_dtype("uint32")

        # e.g. "x×2, y×2": the binned axes, by name.
        label = ", ".join(
            f"{_defined(ax.name) or f'axis {i}'}×{f}" for i, (ax, f) in enumerate(zip(natural, scale)) if f > 1
        )
        dataset.replace_signal(signal)  # frees the unbinned data
        dataset.kind = classify(signal)
        dataset.rebinned.append(label)

        if dataset.kind is DatasetKind.SPECTRUM_IMAGE:
            for m in self.computed_maps(dataset):
                new_map = self._make_map(dataset, m.element)
                if new_map is None:  # its line left the (rebinned) energy range
                    self._remove(m)
                else:
                    m.replace_signal(new_map)
        return dataset

    def set_map_options(self, si: Dataset, options: dict) -> None:
        """Change how ``si``'s element maps are computed and recompute them."""
        from .processing.element_maps import MapOptions

        si.map_options = MapOptions(**options).model_dump()
        si.materialise()
        for m in self.computed_maps(si):
            signal = self._make_map(si, m.element)
            if signal is not None:
                m.replace_signal(signal)

    def siblings_in_file(self, file_id: str) -> list[Dataset]:
        with self._lock:
            return [self.datasets[d] for d in self.files[file_id].dataset_ids]

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
