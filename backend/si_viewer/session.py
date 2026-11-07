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

# Element maps made by fitting every pixel (net counts, and at% composition).
FIT_ORIGINS = ("fitted", "composition")


class NotRebinnable(ValueError):
    """Rebinning was asked of a dataset that isn't a spectrum image."""


def _rebin_in_place(dataset: Dataset, scale: list[int]) -> None:
    """HyperSpy rebin of ``dataset`` by ``scale`` (natural axis order),
    replacing its loaded data and recording what was binned."""
    if all(f == 1 for f in scale):
        return
    signal = dataset.signal.rebin(scale=scale)
    # Integer sums come back as uint64: keep uint32 when the counts fit.
    if signal.data.dtype == np.uint64 and not getattr(signal, "_lazy", False) and signal.data.max() < 2**32:
        signal.change_dtype("uint32")
    natural = dataset.signal.axes_manager._get_axes_in_natural_order()
    # e.g. "x×2, y×2": the binned axes, by name.
    label = ", ".join(
        f"{_defined(ax.name) or f'axis {i}'}×{f}" for i, (ax, f) in enumerate(zip(natural, scale)) if f > 1
    )
    dataset.replace_signal(signal)  # frees the unbinned data
    dataset.kind = classify(signal)
    dataset.rebinned.append(label)


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
    # Non-destructive filters applied when the data is read (see
    # processing/filters.py); filter_rev is bumped whenever they change.
    filters: dict = field(default_factory=dict)
    filter_rev: int = 0
    # Stacks: an applied drift correction (see Session.apply_alignment). While
    # it is shown, ``signal`` is the aligned stack and ``loaded_signal`` the
    # stack as loaded; ``alignment`` says how it was made.
    alignment: dict | None = None
    aligned_signal: object | None = field(default=None, repr=False)
    original_signal: object | None = field(default=None, repr=False)
    # A dataset made from another (e.g. a stack's aligned average).
    derived_from: str | None = None
    thumbnails: dict[int, bytes] = field(default_factory=dict, repr=False)
    # Derived arrays (projections, sum spectra, …) keyed by name.
    cache: dict[str, object] = field(default_factory=dict, repr=False)
    _load_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def title(self) -> str:
        return self.signal.metadata.General.get_item("title", "") or "untitled"

    @property
    def filter_spec(self):
        from .processing.filters import Filters

        return Filters(**self.filters)

    @property
    def loaded_signal(self):
        """The data as loaded (a stack's frames before any alignment)."""
        return self.original_signal if self.original_signal is not None else self.signal

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
            "filters": self.filter_spec.model_dump(),
            "filter_rev": self.filter_rev,
            "alignment": None if self.alignment is None else {
                **self.alignment,
                "active": self.aligned_signal is not None and self.signal is self.aligned_signal,
                "loaded_frames": int(self.loaded_signal.axes_manager.navigation_size),
            },
            "derived_from": self.derived_from,
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
        from .processing.filters import filter_image

        # The map of the processed spectrum image: smoothed spectra, then the
        # spatial filter on the map (they commute).
        signal = compute_map(si.signal, element, MapOptions(**si.map_options), si.filter_spec)
        if signal is not None and si.filter_spec.has_spatial:
            signal.data = filter_image(signal.data, si.filter_spec)
        return signal

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
        if dataset.origin in FIT_ORIGINS:
            return  # results of a per-pixel fit: nothing else to undo
        element = dataset.element
        for d in self.siblings_in_file(dataset.file_id):
            if d.kind is DatasetKind.SPECTRUM_IMAGE and dataset.parent_id not in (None, d.id):
                continue  # another spectrum image in the same file
            info = eds_info(d.signal)
            if info is not None and element in info["elements"]:
                self.set_elements(d, [e for e in info["elements"] if e != element])

    def _fit_maps(self, si: Dataset) -> list[Dataset]:
        return [d for d in self.siblings_in_file(si.file_id) if d.parent_id == si.id and d.origin in FIT_ORIGINS]

    def set_fit_maps(self, si: Dataset, maps: list[tuple[str, str, str, np.ndarray]]) -> None:
        """Replace ``si``'s per-pixel fit results with ``maps``: tuples of
        (origin "fitted" or "composition", element, title, data)."""
        from .processing.element_maps import scan_image

        for d in self._fit_maps(si):
            self._remove(d)
        for origin, element, title, data in maps:
            ds = Dataset(
                id=_new_id(), file_id=si.file_id, signal=scan_image(si.signal, data, title),
                kind=DatasetKind.IMAGE, is_element_map=True, origin=origin, element=element, parent_id=si.id,
            )
            with self._lock:
                self.datasets[ds.id] = ds
                self.files[si.file_id].dataset_ids.append(ds.id)

    def rebin(self, si: Dataset, factors: list[int]) -> list[Dataset]:
        """Rebin a spectrum image and everything recorded on its scan, in place.

        Only spectrum images can be rebinned, and always as a whole: the
        spectrum image by an integer factor along each axis, and the file's
        images and stacks on the same scan grid (e.g. the HAADF) by the same
        x and y factors, so they still line up. Computed element maps are
        recomputed at the new size and per-pixel fit results are removed. The
        rebinned data replaces the loaded data (the file on disk is not
        changed). ``factors`` follow the summary's axis order (array order: y,
        x, energy). Sizes that don't divide are cropped (HyperSpy's rebin).
        Returns the datasets rebinned, the spectrum image first.
        """
        if si.kind is not DatasetKind.SPECTRUM_IMAGE:
            raise NotRebinnable(f"{si.title} can't be rebinned: only spectrum images can (with their whole scan)")
        am = si.signal.axes_manager
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
        fx, fy = scale[0], scale[1]
        scan = tuple(am.navigation_shape)
        on_scan = [
            d for d in self.siblings_in_file(si.file_id)
            if d.origin == "file" and d.kind in (DatasetKind.IMAGE, DatasetKind.STACK)
            and tuple(d.loaded_signal.axes_manager.signal_shape) == scan
        ]

        si.materialise()
        for d in on_scan:
            self.remove_alignment(d)  # its shifts are in pixels of the old size
        _rebin_in_place(si, scale)
        for d in on_scan:
            # Natural order: (frames |) x, y.
            _rebin_in_place(d, [1] * d.signal.axes_manager.navigation_dimension + [fx, fy])

        for d in self._fit_maps(si):
            self._remove(d)  # made at the old size: refit to update
        for m in self.computed_maps(si):
            new_map = self._make_map(si, m.element)
            if new_map is None:  # its line left the (rebinned) energy range
                self._remove(m)
            else:
                m.replace_signal(new_map)
        return [si, *on_scan]

    def set_map_options(self, si: Dataset, options: dict) -> None:
        """Change how ``si``'s element maps are computed and recompute them."""
        from .processing.element_maps import MapOptions

        si.map_options = MapOptions(**options).model_dump()
        si.materialise()
        for m in self.computed_maps(si):
            signal = self._make_map(si, m.element)
            if signal is not None:
                m.replace_signal(signal)

    def set_filters(self, dataset: Dataset, filters: dict) -> list[str]:
        """Set a dataset's filters (applied whenever its data is read; the
        loaded data is unchanged).

        A spectrum image's filters are the processing of all its data: its
        spectra, fits and every map made from it. Computed maps are remade;
        per-pixel fit results, made from the old processing, are removed.
        Those maps can't be filtered on their own. Returns notes for the user.
        """
        from .processing.filters import Filters, check_applicable

        if dataset.parent_id is not None and dataset.parent_id in self.datasets:
            parent = self.datasets[dataset.parent_id]
            raise ValueError(
                f"{dataset.title} follows the processing of {parent.title}: filter the spectrum image instead"
            )
        new = Filters(**filters)
        check_applicable(dataset.kind.value, new)
        old = dataset.filter_spec
        dataset.filters = new.model_dump()
        dataset.filter_rev += 1
        dataset.thumbnails.clear()
        for key in [k for k in dataset.cache if "|" in k]:
            del dataset.cache[key]  # filtered results (raw ones are kept)

        notes = []
        if dataset.kind is DatasetKind.SPECTRUM_IMAGE and new != old:
            dataset.materialise()
            for m in self.computed_maps(dataset):
                signal = self._make_map(dataset, m.element)
                if signal is not None:
                    m.replace_signal(signal)
            fits = self._fit_maps(dataset)
            for d in fits:
                self._remove(d)
            if fits:
                notes.append("Fitted maps were removed: fit every pixel again to fit the processed data")
        return notes

    # ---------- drift correction of stacks ----------

    def _averages(self, stack: Dataset) -> list[Dataset]:
        return [d for d in self.siblings(stack) if d.derived_from == stack.id and d.origin == "aligned-average"]

    @staticmethod
    def _view_changed(ds: Dataset) -> None:
        """The data read from ``ds`` changed: clients refetch (filter_rev),
        derived arrays are remade."""
        ds.filter_rev += 1
        ds.thumbnails.clear()
        ds.cache.clear()

    def apply_alignment(self, stack: Dataset, applied: dict) -> Dataset:
        """Store a drift correction with ``stack`` and show it: its frames
        are then the aligned ones (bad frames left out, cropped to the square
        inside every frame); the loaded frames are kept, to switch back.
        ``applied`` is ``RegistrationSession.applied()``. The average is added
        to the file as an image (replacing an earlier one) and returned."""
        from .processing.registration import aligned_signals

        original = stack.loaded_signal
        aligned, mean = aligned_signals(original, applied["stack"], applied["average"], applied["info"]["square"])
        stack.original_signal = original
        stack.aligned_signal = aligned
        stack.alignment = applied["info"]
        stack.signal = aligned
        self._view_changed(stack)
        for d in self._averages(stack):
            self._remove(d)
        avg = Dataset(
            id=_new_id(), file_id=stack.file_id, signal=mean, kind=DatasetKind.IMAGE,
            origin="aligned-average", derived_from=stack.id,
        )
        with self._lock:
            self.datasets[avg.id] = avg
            ids = self.files[stack.file_id].dataset_ids
            ids.insert(ids.index(stack.id) + 1, avg.id)
        return avg

    def show_alignment(self, stack: Dataset, active: bool) -> None:
        """Show a stack's frames aligned or as loaded."""
        if stack.alignment is None:
            raise ValueError(f"{stack.title} has no alignment")
        target = stack.aligned_signal if active else stack.original_signal
        if stack.signal is not target:
            stack.signal = target
            self._view_changed(stack)

    def remove_alignment(self, stack: Dataset) -> None:
        """Drop a stack's alignment (and its average): back to the frames as loaded."""
        if stack.alignment is None:
            return
        stack.signal = stack.original_signal
        stack.alignment = stack.aligned_signal = stack.original_signal = None
        self._view_changed(stack)
        for d in self._averages(stack):
            self._remove(d)

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
