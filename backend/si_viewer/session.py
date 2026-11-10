"""In-memory registry of open files and the datasets (signals) they contain."""

from __future__ import annotations

import math
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from .io import load_file
from .io.dm_experiment import is_survey, survey_regions
from .insitu_series import InsituSeries, ViewState
from .io.insitu import insitu_summary, is_insitu_series
from .processing.technique import apply_technique, detect_technique, technique_of


class DatasetKind(str, Enum):
    IMAGE = "image"
    STACK = "stack"
    SPECTRUM = "spectrum"
    SPECTRUM_IMAGE = "spectrum_image"
    OTHER = "other"


class NotAvailable(ValueError):
    """The operation isn't available for this dataset (yet)."""


def classify(signal) -> DatasetKind:
    am = signal.axes_manager
    nav, sig = am.navigation_dimension, am.signal_dimension
    if sig == 2 and nav == 0:
        return DatasetKind.IMAGE
    if sig == 2 and nav == 1:
        return DatasetKind.STACK
    if sig == 1 and nav == 0:
        return DatasetKind.SPECTRUM
    # A line scan (one navigation axis) is a spectrum image of a single row.
    if sig == 1 and nav in (1, 2):
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
    """Rebinning was asked of a dataset that isn't a (2D) spectrum image."""


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


def si_cube(ds: "Dataset"):
    """A spectrum image's data as (height, width, channels). A line scan has
    one row: (1, positions, channels). A view of the loaded data, not a copy."""
    data = ds.signal.data
    return data[np.newaxis] if ds.is_line_scan else data


@dataclass
class Dataset:
    id: str
    file_id: str
    signal: object
    kind: DatasetKind
    is_element_map: bool = False
    # "file" (loaded), "computed" (an element map made from a spectrum image),
    # "fitted" / "composition" (per-pixel fits), "window" (an energy-window map),
    # "background" (EELS spectra with the background removed), "background-r"
    # (the power-law exponent map of one), "thickness" (the thickness of an
    # EELS low loss: a map, or a profile for a line scan), "zlp-shift" (the
    # energy shift zero-loss alignment applied at every pixel of a low loss)
    # or "deconvolved" (EELS spectra with plural scattering removed).
    origin: str = "file"
    # Element symbol of an element map (Velox maps: their title).
    element: str | None = None
    # Spectrum image a computed map was made from.
    parent_id: str | None = None
    # Window maps: the energy window {lo, hi, units} they sum (lo <= hi).
    window: dict | None = None
    # Background-removed EELS spectra ("background"): the options they were made
    # with (see processing/background.py), kept to re-run them.
    background: dict | None = None
    # Thickness of a low loss ("thickness"): the options it was made with (see
    # processing/thickness.py) and the energy units of the threshold.
    thickness: dict | None = None
    # Deconvolved EELS spectra ("deconvolved"): the options they were made
    # with (see processing/deconvolution.py) and the low loss used.
    deconvolution: dict | None = None
    # Bumped whenever the data is recomputed, so clients refresh caches.
    revision: int = 0
    # Bumped when the data is replaced by other data of the same kind (the next
    # frame of an in-situ series): clients refetch but keep their display
    # settings (contrast, zoom, cursor), unlike after a new revision.
    data_rev: int = 0
    # In-situ series: the series this dataset is a view of. The dataset holds
    # one frame (or a reduction of a frame range) as an ordinary 2D spectrum
    # image (see insitu_series.py).
    insitu: InsituSeries | None = field(default=None, repr=False)
    # Background-removed spectra made from an in-situ frame: which one ("frame 37").
    source_frame: str | None = None
    # Spectrum images: how their computed element maps are made.
    map_options: dict = field(default_factory=dict)
    # Rebinning applied to the loaded data, e.g. ["x×2, y×2"] (oldest first).
    rebinned: list[str] = field(default_factory=list)
    # Zero-loss alignment applied to the loaded data, e.g. ["sub-pixel, cropped"]
    # (oldest first).
    zlp_aligned: list[str] = field(default_factory=list)
    # Non-destructive filters applied when the data is read (see
    # processing/filters.py); filter_rev is bumped whenever they change.
    filters: dict = field(default_factory=dict)
    filter_rev: int = 0
    # Stacks: an applied drift correction (see Session.apply_alignment). While
    # it is shown, ``signal`` is the aligned stack and ``loaded_signal`` the
    # stack as loaded; ``alignment`` says how it was made.
    alignment: dict | None = None
    # Spectra: where the technique (EDS or EELS) came from, see
    # processing/technique.py; None when unknown.
    technique_source: str | None = None
    # EELS spectrum images: "low_loss" or "core_loss" and where that came from
    # ("metadata", "user" or "default", see processing/eels.py); the link group
    # (see Session.eels_groups) and, mirroring it, the others' ids (low loss
    # first, then core losses by energy offset).
    eels_role: str | None = None
    eels_role_source: str | None = None
    eels_group: str | None = None
    eels_linked: list[str] = field(default_factory=list)
    aligned_signal: object | None = field(default=None, repr=False)
    original_signal: object | None = field(default=None, repr=False)
    # A dataset made from another (e.g. a stack's aligned average).
    derived_from: str | None = None
    # A name the user gave (see ``rename``): kept when the data is recomputed.
    custom_title: str | None = None
    # The survey image of a spectrum imaging run: {"regions": the rectangles
    # GMS drew on it (see io/dm_experiment.py)}.
    survey: dict | None = None
    thumbnails: dict[int, bytes] = field(default_factory=dict, repr=False)
    # Derived arrays (projections, sum spectra, …) keyed by name.
    cache: dict[str, object] = field(default_factory=dict, repr=False)
    _load_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def title(self) -> str:
        return self.custom_title or self.signal.metadata.General.get_item("title", "") or "untitled"

    def rename(self, title: str) -> None:
        """Give the dataset a name of the user's (also written into its
        metadata, so saved files carry it)."""
        title = " ".join(title.split())
        if not title:
            raise ValueError("Enter a name")
        if len(title) > 200:
            raise ValueError("The name is too long")
        self.custom_title = title
        for s in (self.signal, self.original_signal, self.aligned_signal):
            if s is not None:
                s.metadata.General.title = title

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

    @property
    def is_line_scan(self) -> bool:
        """A spectrum image with one navigation axis (data: positions × channels)."""
        return self.kind is DatasetKind.SPECTRUM_IMAGE and self.signal.axes_manager.navigation_dimension == 1

    def materialise(self) -> None:
        """Load the data into RAM (no-op if already in memory)."""
        with self._load_lock:
            if self.insitu is not None:
                self.insitu.load_view(self)  # one frame, never the series
            elif self.lazy:
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
        if self.custom_title:
            signal.metadata.General.title = self.custom_title
        self.signal = signal
        self.revision += 1
        self.thumbnails.clear()
        self.cache = {}  # a new dict: a result still being computed from the old data lands in the old one

    def refresh_data(self, signal) -> None:
        """Swap in other data of the same kind (the next frame): the same
        revision, so clients keep their display settings, but a new ``data_rev``
        so they refetch."""
        if self.custom_title:
            signal.metadata.General.title = self.custom_title
        self.signal = signal
        self.data_rev += 1
        self.thumbnails = {}
        self.cache = {}

    def summary(self) -> dict:
        from .processing.element_maps import MapOptions
        from .processing.eels import eels_info
        from .processing.technique import technique_of
        from .processing.xray import eds_info

        s = self.signal
        am = s.axes_manager
        eds = eds_info(s)
        is_spectrum = self.kind in (DatasetKind.SPECTRUM, DatasetKind.SPECTRUM_IMAGE)
        technique = technique_of(s) if is_spectrum else None
        if eds is not None and self.kind is DatasetKind.SPECTRUM_IMAGE:
            eds["map_options"] = MapOptions(**self.map_options).model_dump()
        return {
            "eds": eds,
            "technique": technique,
            "technique_source": self.technique_source if is_spectrum else None,
            "eels": eels_info(self),
            "origin": self.origin,
            "element": self.element,
            "parent_id": self.parent_id,
            "window": None if self.window is None else dict(self.window),
            "background": None if self.background is None else {
                "source_id": self.parent_id,
                **{k: list(v) if isinstance(v, (list, tuple)) else v for k, v in self.background.items()},
                "units": _defined(am.signal_axes[0].units),
            },
            "thickness": None if self.thickness is None else {"source_id": self.parent_id, **self.thickness},
            "deconvolution": None if self.deconvolution is None else {
                "source_id": self.parent_id,
                **self.deconvolution,
                "units": _defined(am.signal_axes[0].units),
            },
            "revision": self.revision,
            "data_rev": self.data_rev,
            "source_frame": self.source_frame,
            "rebinned": list(self.rebinned),
            "zlp_aligned": list(self.zlp_aligned),
            "filters": self.filter_spec.model_dump(),
            "filter_rev": self.filter_rev,
            "alignment": None if self.alignment is None else {
                **self.alignment,
                "active": self.aligned_signal is not None and self.signal is self.aligned_signal,
                "loaded_frames": int(self.loaded_signal.axes_manager.navigation_size),
            },
            "derived_from": self.derived_from,
            "survey": self.survey,
            "xray_line": s.metadata.get_item("Sample.xray_line_map.line"),
            "insitu": self.insitu.summary() if self.insitu is not None else insitu_summary(s),
            "id": self.id,
            "file_id": self.file_id,
            "title": self.title,
            "kind": self.kind.value,
            "line_scan": self.is_line_scan,
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
class Experiment:
    """The files of one folder, opened together (``Session.open_folder``)."""

    id: str
    path: Path
    file_ids: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {"id": self.id, "name": self.path.name, "path": str(self.path)}


@dataclass
class FileEntry:
    id: str
    path: Path
    dataset_ids: list[str] = field(default_factory=list)
    # Shown instead of the file name (e.g. an example's title).
    label: str | None = None
    # The experiment (folder) it was opened in.
    experiment_id: str | None = None

    def role(self, session: "Session") -> str | None:
        """What the file is in its experiment: "Survey", "ADF", "EELS LL",
        "EELS HL", "EDS", ... (None if it can't tell)."""
        from .io.dm_experiment import scan_signal_name
        from .processing.eels import is_eels_si

        loaded = [session.datasets[d] for d in self.dataset_ids if session.datasets[d].origin == "file"]
        for d in loaded:
            if d.survey is not None:
                return "Survey"
        for d in loaded:
            if is_eels_si(d):
                if d.eels_role_source == "default" and not d.eels_linked:
                    return "EELS"
                return "EELS LL" if d.eels_role == "low_loss" else "EELS HL"
        for d in loaded:
            if d.kind is DatasetKind.SPECTRUM_IMAGE and technique_of(d.signal):
                return technique_of(d.signal)
        for d in loaded:
            name = scan_signal_name(d.signal)
            if name:
                return name
        return None

    def summary(self, session: "Session") -> dict:
        experiment = session.experiments.get(self.experiment_id) if self.experiment_id else None
        return {
            "id": self.id,
            "path": str(self.path),
            "name": self.label or self.path.name,
            "experiment": experiment.summary() if experiment else None,
            "role": self.role(session) if experiment else None,
            "datasets": [session.datasets[d].summary() for d in self.dataset_ids],
        }


class Session:
    def __init__(self) -> None:
        self.files: dict[str, FileEntry] = {}
        self.datasets: dict[str, Dataset] = {}
        # Link groups of EELS spectrum images (two or more, at most one low
        # loss): group id -> member dataset ids.
        self.eels_groups: dict[str, list[str]] = {}
        self.experiments: dict[str, Experiment] = {}
        self._lock = threading.RLock()

    def open_file(self, path: str | Path, experiment_id: str | None = None) -> FileEntry:
        """Open ``path`` (an open file is not opened again). With
        ``experiment_id``, the file joins that experiment and its EELS
        spectrum images are linked with the experiment's."""
        path = Path(path).expanduser().resolve()
        experiment = None
        if experiment_id is not None:
            experiment = self.experiments.get(experiment_id)
            if experiment is None:
                raise KeyError(experiment_id)
        with self._lock:
            entry = next((e for e in self.files.values() if e.path == path), None)
        if entry is None:
            entry = self.open_signals(path, load_file(path, lazy=True))
        if experiment is not None and entry.experiment_id != experiment.id:
            with self._lock:
                self._leave_experiment(entry)
                entry.experiment_id = experiment.id
                experiment.file_ids.append(entry.id)
            self._link_experiment(entry, experiment)
        return entry

    # ---------- experiments (the files of one folder) ----------

    def open_folder(self, folder: str | Path) -> tuple[Experiment, list[Path]]:
        """Start an experiment of ``folder`` (the same one again if it is
        open). Returns it and the files to open into it, in order (an in-situ
        spectrum image's image series open with it, so they are left out)."""
        from .io import SUPPORTED_SUFFIXES
        from .io.insitu import insitu_companions

        folder = Path(folder).expanduser().resolve()
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        paths = sorted(
            p for p in folder.iterdir()
            if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in SUPPORTED_SUFFIXES
        )
        companions = {c for p in paths if p.suffix.lower() == ".dm4" for c in insitu_companions(p)}
        with self._lock:
            experiment = next((x for x in self.experiments.values() if x.path == folder), None)
            if experiment is None:
                experiment = Experiment(id=_new_id(), path=folder)
                self.experiments[experiment.id] = experiment
        return experiment, [p for p in paths if p not in companions]

    def _leave_experiment(self, entry: FileEntry) -> None:
        experiment = self.experiments.get(entry.experiment_id) if entry.experiment_id else None
        entry.experiment_id = None
        if experiment is None:
            return
        experiment.file_ids = [f for f in experiment.file_ids if f != entry.id]
        if not experiment.file_ids:
            self.experiments.pop(experiment.id, None)

    def _link_experiment(self, entry: FileEntry, experiment: Experiment) -> None:
        """Link the EELS spectrum images of ``entry`` with those of the other
        files of its experiment: a dual EELS pair (each names the other), or
        else ones of the same scan (and DM spectrum imaging run when both say)."""
        from .io.dm_experiment import dual_sibling_id, experiment_ids, unique_id
        from .processing.eels import compatible, is_eels_si

        def eels(file_id: str) -> list[Dataset]:
            f = self.files.get(file_id)
            ds = [self.datasets[d] for d in f.dataset_ids] if f else []
            return [d for d in ds if d.origin == "file" and is_eels_si(d)]

        def pairs(a: Dataset, b: Dataset) -> bool:
            if not compatible(a, b):
                return False
            ia, ib = unique_id(a.signal), unique_id(b.signal)
            if ia and ib and (dual_sibling_id(a.signal) == ib or dual_sibling_id(b.signal) == ia):
                return True
            ea, eb = set(experiment_ids(a.signal)), set(experiment_ids(b.signal))
            return not (ea and eb) or bool(ea & eb)

        mine = eels(entry.id)
        others = [d for f in experiment.file_ids if f != entry.id for d in eels(f)]
        for d in mine:
            for o in others:
                if d.eels_group is not None and d.eels_group == o.eels_group:
                    continue
                if pairs(d, o):
                    try:
                        self.set_eels_partner(d, o)
                    except ValueError:
                        pass  # e.g. two low losses: left for the user

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
            # An in-situ series (far too big to load) is shown one frame at a
            # time: its dataset holds the frame, as an ordinary spectrum image.
            series = None
            if is_insitu_series(signal):
                series = InsituSeries(signal)
                signal = series.placeholder()
            kind = classify(signal)
            is_map = _is_element_map(signal, kind)
            technique, source = detect_technique(signal)
            if source in ("dm_tag", "energy_axis"):
                apply_technique(signal, technique)  # exSpy's class follows
            datasets.append(
                Dataset(
                    id=_new_id(),
                    file_id=entry.id,
                    signal=signal,
                    kind=kind,
                    is_element_map=is_map,
                    element=signal.metadata.General.title if is_map else None,
                    technique_source=source,
                    insitu=series,
                    survey={"regions": survey_regions(signal)}
                    if kind is DatasetKind.IMAGE and is_survey(signal) else None,
                )
            )
        # Element maps stored by Velox are processed (quantified, often
        # filtered). When the file has an EDS spectrum image, raw net-count maps
        # are computed from its spectrum stream instead.
        eds_sis = [
            d for d in datasets
            if d.kind is DatasetKind.SPECTRUM_IMAGE and not d.is_line_scan  # no maps of a line scan
            and technique_of(d.signal) == "EDS"
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
        self._set_new_roles(datasets)
        return entry

    # ---------- linked EELS spectrum images (low loss / core losses) ----------

    def linked_of(self, ds: Dataset) -> list[Dataset]:
        """The EELS spectrum images linked to ``ds`` (low loss first, then core
        losses by energy offset); empty when it is not in a group."""
        with self._lock:
            return [self.datasets[i] for i in ds.eels_linked if i in self.datasets]

    def _set_role(self, ds: Dataset) -> None:
        """Read the role from the metadata (default: core loss). A dataset made
        from another (e.g. background removed) has no markers of its own."""
        from .processing.eels import role_of

        ds.eels_role, ds.eels_role_source = role_of(ds)
        if ds.eels_role is not None and ds.origin != "file":
            ds.eels_role, ds.eels_role_source = "core_loss", "default"

    def _refresh_group(self, group_id: str) -> None:
        """Order a group's members (low loss first, then by energy offset) and
        tell each one who the others are."""
        from .processing.eels import energy_offset

        members = [self.datasets[i] for i in self.eels_groups[group_id]]
        members.sort(key=lambda d: (d.eels_role != "low_loss", energy_offset(d.signal), d.id))
        self.eels_groups[group_id] = [d.id for d in members]
        for d in members:
            d.eels_group = group_id
            d.eels_linked = [o.id for o in members if o is not d]

    def _leave(self, ds: Dataset) -> list[Dataset]:
        """Take ``ds`` out of its group (a group of one dissolves). Returns the
        datasets that were linked to it."""
        with self._lock:
            others = self.linked_of(ds)
            gid, ds.eels_group, ds.eels_linked = ds.eels_group, None, []
            if gid is None:
                return others
            remaining = [i for i in self.eels_groups.get(gid, []) if i != ds.id and i in self.datasets]
            if len(remaining) < 2:
                self.eels_groups.pop(gid, None)
                for i in remaining:
                    self.datasets[i].eels_group, self.datasets[i].eels_linked = None, []
            else:
                self.eels_groups[gid] = remaining
                self._refresh_group(gid)
            return others

    @staticmethod
    def _check_group(members: list[Dataset]) -> None:
        """Raise ValueError unless ``members`` can be one group: all pairwise
        compatible and at most one low loss."""
        from itertools import combinations

        from .processing.eels import compatible

        low = [d for d in members if d.eels_role == "low_loss"]
        if len(low) > 1:
            raise ValueError(f"{low[0].title} and {low[1].title} are both low loss: a group has at most one")
        for a, b in combinations(members, 2):
            if not compatible(a, b):
                raise ValueError(
                    f"{a.title} and {b.title} can't be linked: they need the same scan size and "
                    "calibration (and beam energy)"
                )

    def _join(self, datasets: list[Dataset]) -> list[Dataset]:
        """Make ``datasets`` and everything linked to them one group. Raises
        ValueError (changing nothing) if that can't be a group. Returns the
        datasets in it."""
        with self._lock:
            members: list[Dataset] = []
            for d in datasets:
                for m in [d, *self.linked_of(d)]:
                    if m not in members:
                        members.append(m)
            self._check_group(members)
            for gid in {m.eels_group for m in members if m.eels_group}:
                self.eels_groups.pop(gid, None)
            gid = _new_id()
            self.eels_groups[gid] = [m.id for m in members]
            self._refresh_group(gid)
            return members

    def _set_new_roles(self, new: list[Dataset]) -> None:
        """Give the EELS spectrum images of a file just opened their roles.
        Nothing is linked: only the user links spectrum images."""
        from .processing.eels import is_eels_si

        for d in new:
            if is_eels_si(d):
                self._set_role(d)

    def link_candidates(self, ds: Dataset) -> list[Dataset]:
        """The spectrum images ``ds`` could be linked to (whether they are in
        a group or not)."""
        from .processing.eels import compatible

        with self._lock:
            return [o for o in self.datasets.values() if compatible(ds, o)]

    def set_eels_partner(self, ds: Dataset, partner: Dataset | None) -> list[Dataset]:
        """Put ``ds`` in ``partner``'s group (or a new group of the two),
        leaving the one it was in; with None, just leave its group. Raises
        ValueError (changing nothing) if they are not compatible or the group
        would get a second low loss. Returns the datasets whose links changed."""
        from .processing.eels import compatible

        with self._lock:
            if partner is None:
                return [ds, *self._leave(ds)]
            if ds.eels_group is not None and ds.eels_group == partner.eels_group:
                return [ds]
            if not compatible(ds, partner):
                raise ValueError(
                    f"{ds.title} and {partner.title} can't be linked: they need the same scan size and "
                    "calibration (and beam energy)"
                )
            target = [partner, *self.linked_of(partner)]
            self._check_group([*target, ds])
            changed = [ds, *self._leave(ds), *target]
            self._join([ds, partner])
            return list({d.id: d for d in changed}.values())

    def set_eels_role(self, ds: Dataset, role: str) -> list[Dataset]:
        """Say by hand that ``ds`` is a low loss or a core loss. Raises
        ValueError if its group already has another low loss. Returns the
        datasets whose summaries changed (its group)."""
        from .processing.eels import ROLES, is_eels_si

        if not is_eels_si(ds):
            raise ValueError(f"{ds.title} is not an EELS spectrum image")
        if role not in ROLES:
            raise ValueError(f"Unknown role: {role}")
        with self._lock:
            others = self.linked_of(ds)
            low = next((o for o in others if o.eels_role == "low_loss"), None)
            if role == "low_loss" and low is not None:
                raise ValueError(f"{low.title} is already the low loss of this group")
            ds.eels_role, ds.eels_role_source = role, "user"
            if ds.eels_group is not None:
                self._refresh_group(ds.eels_group)
            if role != "low_loss":
                for t in [*self.thickness_maps(ds), *self.zlp_shift_maps(ds)]:
                    self._remove(t)  # a thickness and a ZLP shift are of a low loss
            return [ds, *others]

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
            self._leave_experiment(entry)
            for ds_id in entry.dataset_ids:
                ds = self.datasets.get(ds_id)
                if ds is not None:
                    self._leave(ds)
                    if ds.insitu is not None:
                        ds.insitu.close()
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
            self._leave(dataset)
            self.datasets.pop(dataset.id, None)
            ids = self.files[dataset.file_id].dataset_ids
            if dataset.id in ids:
                ids.remove(dataset.id)

    def _remove_tree(self, dataset: Dataset) -> None:
        """Remove ``dataset`` and everything made from it (its maps, and so on)."""
        for child in [d for d in self.siblings(dataset) if d.parent_id == dataset.id]:
            self._remove_tree(child)
        self._remove(dataset)

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

    # ---------- energy-window maps ("Keep as map") ----------

    def window_maps(self, si: Dataset) -> list[Dataset]:
        with self._lock:
            return [d for d in self.siblings(si) if d.origin == "window" and d.parent_id == si.id]

    @staticmethod
    def _make_window_map(si: Dataset, lo: float, hi: float, title: str):
        """The map of the counts in [lo, hi] of ``si`` as processed (spectral and
        spatial filters applied, as for the viewer's energy map); None if the
        window is outside the spectrum's energy range."""
        from .processing.element_maps import scan_image
        from .processing.regions import processed_energy_map

        axis = si.signal.axes_manager.signal_axes[0]
        lo, hi = sorted((lo, hi))
        start, end = sorted((axis.offset, axis.offset + axis.scale * axis.size))
        if hi < start or lo > end:
            return None
        counts = processed_energy_map(si_cube(si), axis.offset, axis.scale, lo, hi, si.filter_spec)
        return scan_image(si.signal, counts, title)

    def add_window_map(self, si: Dataset, lo: float, hi: float, label: str | None = None) -> Dataset:
        """Keep the energy window [lo, hi] (the spectral axis' units) of a 2D
        spectrum image as an image dataset under it, named ``label`` or by its
        window, e.g. "640–660 eV". It follows the spectrum image's processing
        and rebinning (see ``rebin``, ``set_filters``) and is deleted with it.
        """
        from .processing.regions import window_label

        if si.kind is not DatasetKind.SPECTRUM_IMAGE or si.is_line_scan:
            raise ValueError(f"{si.title} is not a 2D spectrum image: only those have window maps")
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise ValueError("The window's limits must be numbers")
        lo, hi = sorted((float(lo), float(hi)))
        si.materialise()
        energy = si.signal.axes_manager.signal_axes[0]
        units = _defined(energy.units)
        title = (label or "").strip() or window_label(lo, hi, units, energy.scale)
        signal = self._make_window_map(si, lo, hi, title)
        if signal is None:
            raise ValueError(f"The window {window_label(lo, hi, units)} is outside the spectrum's energy range")
        ds = Dataset(
            id=_new_id(), file_id=si.file_id, signal=signal, kind=DatasetKind.IMAGE,
            origin="window", parent_id=si.id, window={"lo": lo, "hi": hi, "units": units},
        )
        with self._lock:
            self.datasets[ds.id] = ds
            self.files[si.file_id].dataset_ids.append(ds.id)
        return ds

    def set_window_map(self, m: Dataset, lo: float, hi: float) -> Dataset:
        """Move a window map to [lo, hi] (the spectral axis' units): same
        dataset, recomputed, new revision. A title that was the automatic name
        of the old window follows the new one; a label the user gave is kept."""
        from .processing.regions import window_label

        if m.origin != "window" or m.window is None:
            raise ValueError(f"{m.title} is not a window map")
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise ValueError("The window's limits must be numbers")
        si = self.get_dataset(m.parent_id)
        lo, hi = sorted((float(lo), float(hi)))
        si.materialise()
        energy = si.signal.axes_manager.signal_axes[0]
        units = _defined(energy.units)
        old = (m.window["lo"], m.window["hi"])
        automatic = {window_label(*old, units, energy.scale), window_label(*old, units)}
        title = window_label(lo, hi, units, energy.scale) if m.title in automatic else m.title
        signal = self._make_window_map(si, lo, hi, title)
        if signal is None:
            raise ValueError(f"The window {window_label(lo, hi, units)} is outside the spectrum's energy range")
        m.replace_signal(signal)
        m.window = {"lo": lo, "hi": hi, "units": units}
        return m

    def _remake_window_maps(self, si: Dataset, keep_display: bool = False) -> None:
        """Recompute ``si``'s window maps after its data or processing changed
        (a map whose window left the energy range is removed). ``keep_display``:
        the maps keep their revision (and so their contrast), see
        ``Dataset.refresh_data``."""
        for m in self.window_maps(si):
            signal = self._make_window_map(si, m.window["lo"], m.window["hi"], m.title)
            if signal is None:
                self._remove(m)
            elif keep_display:
                m.refresh_data(signal)
            else:
                m.replace_signal(signal)

    # ---------- background removal (EELS) ----------

    def background_sis(self, src: Dataset) -> list[Dataset]:
        """The background-removed spectra made from ``src``."""
        with self._lock:
            return [d for d in self.siblings(src) if d.origin == "background" and d.parent_id == src.id]

    def _background_r_maps(self, bg: Dataset) -> list[Dataset]:
        with self._lock:
            return [d for d in self.siblings(bg) if d.origin == "background-r" and d.parent_id == bg.id]

    @staticmethod
    def can_remove_background(ds: Dataset) -> bool:
        from .processing.technique import technique_of

        return ds.kind in (DatasetKind.SPECTRUM, DatasetKind.SPECTRUM_IMAGE) and technique_of(ds.signal) == "EELS"

    def compute_background(self, src: Dataset, options, report=lambda f: None):
        """(signal, r) of ``src``'s spectra cut to the signal window with the
        background subtracted (see ``processing.background.remove_background``).
        Nothing is added to the session. Raises ValueError for windows that
        don't fit the spectra."""
        from .processing.background import remove_background

        if not self.can_remove_background(src):
            raise ValueError("Background removal is for EELS spectra")
        src.materialise()
        return remove_background(src.signal, options, report)

    def _set_background_r(self, bg: Dataset, r: np.ndarray | None) -> None:
        """Give a background-removed 2D spectrum image the power-law exponent
        map ``r`` (None: none), keeping the map's dataset when it has one."""
        from .processing.element_maps import scan_image

        maps = self._background_r_maps(bg)
        if r is None or bg.kind is not DatasetKind.SPECTRUM_IMAGE or bg.is_line_scan:
            for m in maps:
                self._remove(m)
            return
        signal = scan_image(bg.signal, r, "r (power-law exponent)")
        if maps:
            maps[0].replace_signal(signal)
            return
        ds = Dataset(
            id=_new_id(), file_id=bg.file_id, signal=signal, kind=DatasetKind.IMAGE,
            origin="background-r", parent_id=bg.id,
        )
        with self._lock:
            self.datasets[ds.id] = ds
            self.files[bg.file_id].dataset_ids.append(ds.id)

    def add_background(self, src: Dataset, options, result) -> Dataset:
        """Add ``result`` (from ``compute_background``) as a spectra dataset
        under ``src``, titled by its signal window. It follows ``src``'s
        rebinning and is deleted with it, but is a spectrum image (or spectrum)
        like any other. A power-law exponent map comes with it."""
        from .processing.background import EnergyAxis, automatic_title

        signal, r = result
        # Made from one frame (or range) of an in-situ series: the title says which.
        frame = None if src.insitu is None else src.insitu.view.label()
        signal.metadata.General.title = automatic_title(options, EnergyAxis.of(signal)) + (f" ({frame})" if frame else "")
        ds = Dataset(
            id=_new_id(), file_id=src.file_id, signal=signal, kind=classify(signal),
            origin="background", parent_id=src.id, background=options.model_dump(),
            technique_source="metadata", source_frame=frame,
        )
        with self._lock:
            self.datasets[ds.id] = ds
            self.files[src.file_id].dataset_ids.append(ds.id)
        self._set_role(ds)
        self._set_background_r(ds, r)
        return ds

    def replace_background(self, bg: Dataset, options, result) -> Dataset:
        """Replace a background-removed dataset's data with ``result`` made with
        new ``options``: same dataset, new revision. A title that was the
        automatic name of the old signal window follows the new one."""
        from .processing.background import BackgroundOptions, EnergyAxis, automatic_title
        from .processing.regions import window_label

        if bg.origin != "background" or bg.background is None:
            raise ValueError(f"{bg.title} is not a background-removed spectrum")
        signal, r = result
        axis = EnergyAxis.of(signal)
        old = BackgroundOptions(**bg.background)
        automatic = {automatic_title(old, axis), f"{window_label(*old.signal, axis.units)}, bg removed"}
        if bg.source_frame:
            automatic |= {f"{t} ({bg.source_frame})" for t in automatic}
        src = self.datasets.get(bg.parent_id)
        frame = None if src is None or src.insitu is None else src.insitu.view.label()
        suffix = f" ({frame})" if frame else ""
        signal.metadata.General.title = automatic_title(options, axis) + suffix if bg.title in automatic else bg.title
        bg.source_frame = frame
        bg.replace_signal(signal)
        bg.background = options.model_dump()
        bg.rebinned = []
        self._set_background_r(bg, r)
        self._remake_window_maps(bg)
        return bg

    def rerun_background(self, bg: Dataset, options=None, report=lambda f: None) -> Dataset:
        """Remake a background-removed dataset from its source, with ``options``
        or (None) the ones it has."""
        from .processing.background import BackgroundOptions

        src = self.get_dataset(bg.parent_id)
        options = options or BackgroundOptions(**bg.background)
        return self.replace_background(bg, options, self.compute_background(src, options, report))

    def _remake_backgrounds(self, src: Dataset) -> None:
        """Recompute ``src``'s background-removed spectra after its data changed
        (one whose windows no longer fit is removed)."""
        for bg in self.background_sis(src):
            try:
                self.rerun_background(bg)
            except ValueError:
                self._remove_tree(bg)

    # ---------- plural-scattering deconvolution (EELS) ----------

    def deconvolution_low_loss(self, ds: Dataset) -> Dataset | None:
        """The low loss ``ds`` is deconvolved with: its group's (itself if it
        is one); for spectra made from another spectrum image (background
        removed, deconvolved), the low loss of that one."""
        seen = set()
        with self._lock:
            while ds.origin in ("background", "deconvolved") and ds.parent_id in self.datasets and ds.id not in seen:
                seen.add(ds.id)
                ds = self.datasets[ds.parent_id]
        return self.low_loss_of(ds)

    def deconvolution_inputs(self, ds: Dataset, options) -> Dataset:
        """The low loss to deconvolve ``ds`` with by ``options``' method.
        Raises ValueError (in plain English) when the method isn't for ``ds``."""
        from .processing.eels import is_eels_si

        if not is_eels_si(ds):
            raise ValueError(f"{ds.title} is not an EELS spectrum image")
        ll = self.deconvolution_low_loss(ds)
        if ll is None:
            raise ValueError("No low loss in its group")
        if options.method == "fourier_log" and ds is not ll:
            raise ValueError("Fourier-log is for the low loss")
        if options.method == "fourier_ratio" and ds.origin != "background":
            raise ValueError("Remove the background first")
        if options.method == "richardson_lucy":
            if ds.origin == "background":
                raise ValueError("Richardson–Lucy is for spectra before background removal")
            if options.psf == "low_loss" and ds is ll:
                raise ValueError("The low loss can't be its own point-spread function: use its zero-loss peak")
        return ll

    def compute_deconvolution(self, ds: Dataset, options, report=lambda f, m=None: None):
        """``ds``'s spectra deconvolved (see ``processing.deconvolution``).
        Nothing is added to the session. Raises ValueError if they can't be."""
        from .processing.deconvolution import deconvolve

        ll = self.deconvolution_inputs(ds, options)
        ds.materialise()
        ll.materialise()
        return deconvolve(ds.signal, ll.signal, options, report)

    def deconvolved_sis(self, src: Dataset) -> list[Dataset]:
        """The deconvolved spectra made from ``src``."""
        with self._lock:
            return [d for d in self.siblings(src) if d.origin == "deconvolved" and d.parent_id == src.id]

    @staticmethod
    def _deconvolved_title(src_title: str, options, frame: str | None) -> str:
        from .processing.deconvolution import METHOD_NAMES

        return f"{src_title} ({METHOD_NAMES[options.method]})" + (f" ({frame})" if frame else "")

    def add_deconvolved(self, src: Dataset, options, signal) -> Dataset:
        """Add ``signal`` (from ``compute_deconvolution``) as a spectrum image
        under ``src``. It is made once: later changes to ``src`` leave it as
        it is (re-run it to update)."""
        frame = None if src.insitu is None else src.insitu.view.label()
        signal.metadata.General.title = self._deconvolved_title(src.title, options, frame)
        ll = self.deconvolution_low_loss(src)
        ds = Dataset(
            id=_new_id(), file_id=src.file_id, signal=signal, kind=classify(signal),
            origin="deconvolved", parent_id=src.id,
            deconvolution={**options.model_dump(), "low_loss_id": ll.id if ll else None},
            technique_source="metadata", source_frame=frame,
        )
        with self._lock:
            self.datasets[ds.id] = ds
            self.files[src.file_id].dataset_ids.append(ds.id)
        self._set_role(ds)
        return ds

    def replace_deconvolved(self, dd: Dataset, options, signal) -> Dataset:
        """Replace a deconvolved dataset's data with ``signal`` made with new
        ``options``: same dataset, new revision. An automatic title follows
        the method."""
        from .processing.deconvolution import DeconvolutionOptions

        if dd.origin != "deconvolved" or dd.deconvolution is None:
            raise ValueError(f"{dd.title} is not a deconvolved spectrum image")
        src = self.datasets.get(dd.parent_id)
        old = DeconvolutionOptions(**{k: v for k, v in dd.deconvolution.items() if k != "low_loss_id"})
        frame = None if src is None or src.insitu is None else src.insitu.view.label()
        automatic = src is not None and dd.title == self._deconvolved_title(src.title, old, dd.source_frame)
        signal.metadata.General.title = (
            self._deconvolved_title(src.title, options, frame) if automatic else dd.title
        )
        ll = None if src is None else self.deconvolution_low_loss(src)
        dd.source_frame = frame
        dd.replace_signal(signal)
        dd.deconvolution = {**options.model_dump(), "low_loss_id": ll.id if ll else None}
        dd.rebinned = []
        self._remake_window_maps(dd)
        return dd

    # ---------- thickness of a low loss (EELS) ----------

    def low_loss_of(self, ds: Dataset) -> Dataset | None:
        """The low loss thickness is estimated from: ``ds`` if it is one, else
        the low loss of its link group (None if it has none, or is not an EELS
        spectrum image)."""
        from .processing.eels import is_eels_si

        if not is_eels_si(ds):
            return None
        if ds.eels_role == "low_loss":
            return ds
        return next((d for d in self.linked_of(ds) if d.eels_role == "low_loss"), None)

    def thickness_maps(self, ll: Dataset) -> list[Dataset]:
        """The thickness made from the low loss ``ll`` (at most one)."""
        with self._lock:
            return [d for d in self.siblings(ll) if d.origin == "thickness" and d.parent_id == ll.id]

    def estimate_threshold(self, ll: Dataset, window: float) -> float | None:
        """The automatic zero-loss threshold of the low loss ``ll``'s summed
        spectrum (as shown: spectrally smoothed), or None if there is none."""
        from .processing.frames import get_spectrum
        from .processing.thickness import estimate_threshold

        ll.materialise()
        return estimate_threshold(get_spectrum(ll), ll.signal, window)

    def compute_thickness(self, ll: Dataset, options, report=lambda f, m=None: None) -> np.ndarray:
        """The thickness of every pixel of the low loss ``ll`` (see
        ``processing.thickness.thickness_map``). Nothing is added to the
        session. Raises ValueError for options that don't fit the spectra."""
        from .processing.eels import is_eels_si
        from .processing.thickness import thickness_map

        if not is_eels_si(ll) or ll.eels_role != "low_loss":
            raise ValueError(f"{ll.title} is not an EELS low loss")
        ll.materialise()
        return thickness_map(ll.signal, options, ll.filter_spec, report)

    def set_thickness(self, ll: Dataset, options, data: np.ndarray) -> Dataset:
        """Save ``data`` (from ``compute_thickness``) as the thickness of the
        low loss ``ll``: a new dataset under it, or (it has one) that dataset
        with its data replaced: same id, new revision. It follows ``ll``'s
        rebinning and processing and is deleted with it."""
        from .processing.thickness import TITLES, result_signal

        signal = result_signal(ll.signal, data, options)
        units = _defined(ll.signal.axes_manager.signal_axes[0].units)
        info = {**options.model_dump(), "units": units}
        existing = self.thickness_maps(ll)
        for extra in existing[1:]:
            self._remove(extra)
        if existing:
            ds = existing[0]
            if ds.title not in TITLES:
                signal.metadata.General.title = ds.title  # a name the user gave is kept
            ds.replace_signal(signal)
            ds.kind = classify(signal)
            ds.thickness = info
            return ds
        ds = Dataset(
            id=_new_id(), file_id=ll.file_id, signal=signal, kind=classify(signal),
            origin="thickness", parent_id=ll.id, thickness=info,
        )
        with self._lock:
            self.datasets[ds.id] = ds
            self.files[ll.file_id].dataset_ids.append(ds.id)
        return ds

    def _remake_thickness(self, ll: Dataset, shift: float = 0.0) -> None:
        """Recompute the thickness of ``ll`` with the options it was made with,
        after its data or processing changed (one that no longer fits is
        removed). ``shift``: how far its energy axis was moved, which a fixed
        threshold follows."""
        from .processing.thickness import ThicknessOptions
        from .processing.zlp import round_energy

        for t in self.thickness_maps(ll):
            try:
                options = ThicknessOptions(**{k: v for k, v in t.thickness.items() if k != "units"})
                if not options.auto:
                    options.threshold = round_energy(options.threshold + shift)
                self.set_thickness(ll, options, self.compute_thickness(ll, options))
            except ValueError:
                self._remove(t)

    # ---------- zero-loss peak alignment (EELS) ----------

    def zlp_shift_maps(self, ll: Dataset) -> list[Dataset]:
        """The ZLP shift saved with the low loss ``ll`` (at most one)."""
        with self._lock:
            return [d for d in self.siblings(ll) if d.origin == "zlp-shift" and d.parent_id == ll.id]

    def zlp_followers(self, ll: Dataset) -> list[Dataset]:
        """The spectrum images aligned along with the low loss ``ll``: the EELS
        spectrum images linked to it, on its scan (a background-removed one
        whose source is aligned too is remade from it instead)."""
        from .processing.eels import is_eels_si

        linked = [
            d for d in self.linked_of(ll)
            if is_eels_si(d) and tuple(d.signal.axes_manager.navigation_shape) == tuple(ll.signal.axes_manager.navigation_shape)
        ]
        ids = {ll.id, *(d.id for d in linked)}
        return [d for d in linked if not (d.origin == "background" and d.parent_id in ids)]

    def compute_zlp_alignment(self, ll: Dataset, options, report=lambda f, m=None: None):
        """Align the zero-loss peak of the low loss ``ll`` and of its followers
        (see ``processing.zlp.align_zero_loss_peak``). Nothing in the session
        changes. Returns ``(datasets, originals, result)``: the datasets (the
        low loss first), the signals they had, and the aligned ones with the
        shifts. Raises ValueError if the low loss can't be aligned."""
        from .processing.eels import is_eels_si
        from .processing.zlp import align_zero_loss_peak

        if not is_eels_si(ll) or ll.eels_role != "low_loss":
            raise ValueError(f"{ll.title} is not an EELS low loss")
        datasets = [ll, *self.zlp_followers(ll)]
        for d in datasets:
            if d.insitu is not None:
                raise ValueError(f"{d.title} is a frame of an in-situ series and can't be aligned: the next frame would undo it")
        for d in datasets:
            d.materialise()
        originals = [d.signal for d in datasets]
        result = align_zero_loss_peak(originals[0], originals[1:], options, report)
        return datasets, originals, result

    def apply_zlp_alignment(self, datasets: list[Dataset], originals: list, result, options) -> Dataset:
        """Replace the loaded data of ``datasets`` (from ``compute_zlp_alignment``)
        with the aligned ``result``, like a rebin: the file on disk is not
        changed. What was made from them is remade: energy windows, thickness
        and background-removed spectra follow the shift of the axis (a fixed
        threshold, a window), maps made at the old data are removed. The shifts
        are saved as the "ZLP shift" of the low loss (replacing an earlier
        one), which is returned. Raises ValueError, changing nothing, if a
        dataset was deleted or changed since."""
        ll = datasets[0]
        for d, signal in zip(datasets, originals):
            if self.datasets.get(d.id) is not d or d.signal is not signal:
                raise ValueError(f"{d.title} changed while its zero-loss peak was aligned")
        shift = result.mean_shift
        for d, signal in zip(datasets, result.signals):
            d.replace_signal(signal)
            d.kind = classify(signal)
            d.zlp_aligned.append(options.label)
        for d in datasets:
            self._remake_after_zlp(d, shift)
        if ll.eels_group is not None:
            self._refresh_group(ll.eels_group)  # linked ones are ordered by their energy offset
        return self.set_zlp_shift(ll, result.shifts)

    def _remake_after_zlp(self, d: Dataset, shift: float) -> None:
        """Recompute what was made from ``d`` after its data was aligned, with
        energies ``shift`` further along the axis than before."""
        from .processing.background import BackgroundOptions
        from .processing.zlp import round_energy

        for f in self._fit_maps(d):
            self._remove(f)  # made from the old data: fit again to update
        for m in self.computed_maps(d):
            new_map = self._make_map(d, m.element)
            if new_map is None:
                self._remove(m)
            else:
                m.replace_signal(new_map)
        for m in self.window_maps(d):
            try:
                self.set_window_map(m, round_energy(m.window["lo"] + shift), round_energy(m.window["hi"] + shift))
            except ValueError:  # the window left the energy range
                self._remove(m)
        for bg in self.background_sis(d):
            try:
                old = BackgroundOptions(**bg.background)
                moved = {k: [round_energy(v + shift) for v in getattr(old, k)] for k in ("background", "signal")}
                new = BackgroundOptions(**{**old.model_dump(), **moved})
                self.replace_background(bg, new, self.compute_background(d, new))
            except ValueError:  # its windows no longer fit
                self._remove_tree(bg)
        self._remake_thickness(d, shift)

    def set_zlp_shift(self, ll: Dataset, shifts: np.ndarray) -> Dataset:
        """Save ``shifts`` (eV, the shape of the scan) as the ZLP shift of the
        low loss ``ll``: a new dataset under it, or (it has one) that dataset
        with its data replaced: same id, new revision."""
        from .processing.zlp import shift_map_signal

        signal = shift_map_signal(ll.signal, shifts)
        existing = self.zlp_shift_maps(ll)
        for extra in existing[1:]:
            self._remove(extra)
        if existing:
            ds = existing[0]
            if ds.title != "ZLP shift":
                signal.metadata.General.title = ds.title  # a name the user gave is kept
            ds.replace_signal(signal)
            ds.kind = classify(signal)
            return ds
        ds = Dataset(
            id=_new_id(), file_id=ll.file_id, signal=signal, kind=classify(signal),
            origin="zlp-shift", parent_id=ll.id,
        )
        with self._lock:
            self.datasets[ds.id] = ds
            self.files[ll.file_id].dataset_ids.append(ds.id)
        return ds

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
        if ds.kind is DatasetKind.SPECTRUM_IMAGE and not ds.is_line_scan:
            ds.materialise()
            for element in old - new:
                for m in self.computed_maps(ds, element):
                    self._remove(m)
            for element in sorted(new - old):
                if not self.computed_maps(ds, element) and not self._add_computed_map(ds, element):
                    warnings.append(f"{element}: no X-ray line in the spectrum's energy range, so no map")
        return warnings

    def set_technique(self, ds: Dataset, technique: str | None) -> None:
        """Say by hand whether a spectrum or spectrum image is EDS or EELS
        (None: neither), which sets its signal type so that exSpy's class
        follows. Leaving EDS removes the spectrum image's computed element maps
        and fitted maps; entering it computes nothing (the user picks the
        elements). Maps loaded from the file are kept."""
        if ds.kind not in (DatasetKind.SPECTRUM, DatasetKind.SPECTRUM_IMAGE):
            raise ValueError(f"{ds.title} is not a spectrum or spectrum image")
        if technique not in ("EDS", "EELS", None):
            raise ValueError(f"Unknown technique: {technique}")
        was = technique_of(ds.signal)
        if was != technique:
            apply_technique(ds.signal, technique)  # in place: ds.signal stays the same object
            self._leave(ds)
            ds.eels_role = ds.eels_role_source = None
            if was == "EDS":
                for d in [*self.computed_maps(ds), *self._fit_maps(ds)]:
                    self._remove(d)
            if was == "EELS":
                for d in [*self.background_sis(ds), *self.thickness_maps(ds), *self.zlp_shift_maps(ds)]:
                    self._remove_tree(d)  # background removal, thickness and ZLP shifts are for EELS only
            if technique == "EELS":
                self._set_role(ds)
            self._view_changed(ds)
        ds.technique_source = "user"

    def delete_dataset(self, dataset: Dataset) -> None:
        """Delete an element map, a window map, a thickness or a
        background-removed or deconvolved spectrum image (with everything made
        from it).

        Deleting an element map also unticks its element in the map's spectrum image and in the
        file's other (non-image) EDS spectra, which removes that element's
        computed maps too. Other data can't be deleted.
        """
        from .processing.xray import eds_info

        if dataset.origin in ("window", "thickness", "zlp-shift"):
            self._remove(dataset)  # no element to untick
            return
        if dataset.origin in ("background", "background-r", "deconvolved"):
            self._remove_tree(dataset)
            return
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

    def rebin(self, si: Dataset, factors: list[int], with_linked: bool = True) -> list[Dataset]:
        """Rebin a spectrum image and everything recorded on its scan, in place.

        Only spectrum images can be rebinned, and always as a whole: the
        spectrum image by an integer factor along each axis, and the file's
        images and stacks on the same scan grid (e.g. the HAADF) by the same
        x and y factors, so they still line up. Computed element maps are
        recomputed at the new size and per-pixel fit results are removed. The
        rebinned data replaces the loaded data (the file on disk is not
        changed). ``factors`` follow the summary's axis order (array order: y,
        x, energy). Sizes that don't divide are cropped (HyperSpy's rebin).
        The EELS spectrum images linked to it (its group's low loss and core
        losses) are rebinned by the same factors, energy included, with the
        images on their scans; the factors are checked against every one of
        them before anything is rebinned. Returns the datasets
        rebinned, the spectrum image first, then its scan's images, then the
        linked ones and their images.
        """
        if si.kind is not DatasetKind.SPECTRUM_IMAGE:
            raise NotRebinnable(f"{si.title} can't be rebinned: only spectrum images can (with their whole scan)")
        if si.is_line_scan:
            raise NotRebinnable(f"{si.title} is a line scan and can't be rebinned: rebinning needs a 2D scan")
        for d in (si, *(self.linked_of(si) if with_linked else [])):
            if d.insitu is not None:
                raise NotRebinnable(f"{d.title} is a frame of an in-situ series and can't be rebinned: the next frame would undo it")
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
        linked = [
            d for d in (self.linked_of(si) if with_linked else [])
            if d.kind is DatasetKind.SPECTRUM_IMAGE and not d.is_line_scan
            and tuple(d.signal.axes_manager.navigation_shape) == scan  # else no longer the same scan
        ]
        for d in linked:
            d_axes = list(d.signal.axes_manager._axes)
            if len(d_axes) != len(factors):
                raise ValueError(f"{d.title} (linked) has {len(d_axes)} axes, so it can't get the same binning")
            for f, ax in zip(factors, d_axes):
                if f > ax.size:
                    raise ValueError(
                        f"Factor for {ax.name or 'axis'} must be at most {ax.size} for {d.title} (linked)"
                    )
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

        for d in [*self._fit_maps(si), *self.zlp_shift_maps(si)]:
            self._remove(d)  # made at the old size: refit or align again to update
        for m in self.computed_maps(si):
            new_map = self._make_map(si, m.element)
            if new_map is None:  # its line left the (rebinned) energy range
                self._remove(m)
            else:
                m.replace_signal(new_map)
        self._remake_window_maps(si)
        self._remake_backgrounds(si)
        self._remake_thickness(si)
        if si.origin == "background":
            for r in self._background_r_maps(si):
                self._remove(r)  # made at the old size: re-run from the source to update
        rebinned = [si, *on_scan]
        for d in linked:
            rebinned += self.rebin(d, factors, with_linked=False)
        return rebinned

    def set_map_options(self, si: Dataset, options: dict) -> None:
        """Change how ``si``'s element maps are computed and recompute them."""
        from .processing.element_maps import MapOptions

        si.map_options = MapOptions(**options).model_dump()
        si.materialise()
        for m in self.computed_maps(si):
            signal = self._make_map(si, m.element)
            if signal is not None:
                m.replace_signal(signal)

    def set_filters(self, dataset: Dataset, filters: dict, with_linked: bool = True) -> list[str]:
        """Set a dataset's filters (applied whenever its data is read; the
        loaded data is unchanged).

        A spectrum image's filters are the processing of all its data: its
        spectra, fits and every map made from it. Computed maps are remade;
        per-pixel fit results, made from the old processing, are removed.
        Those maps can't be filtered on their own. The EELS spectrum images
        linked to it get the same spatial filter (their spectral smoothing is
        their own), so their pixels keep matching. Returns notes for the user.
        """
        from .processing.filters import Filters, check_applicable

        if (
            dataset.origin not in ("background", "deconvolved")
            and dataset.parent_id is not None
            and dataset.parent_id in self.datasets
        ):
            parent = self.datasets[dataset.parent_id]
            raise ValueError(
                f"{dataset.title} follows the processing of {parent.title}: filter the spectrum image instead"
            )
        new = Filters(**filters)
        check_applicable(dataset.kind.value, new, line_scan=dataset.is_line_scan)
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
            self._remake_window_maps(dataset)
            self._remake_thickness(dataset)
            fits = self._fit_maps(dataset)
            for d in fits:
                self._remove(d)
            if fits:
                notes.append("Fitted maps were removed: fit every pixel again to fit the processed data")
        for other in self.linked_of(dataset) if with_linked else []:
            theirs = other.filter_spec
            if theirs.spatial_params() != new.spatial_params():
                same_spatial = {**theirs.model_dump(), **new.spatial_params()}
                notes += self.set_filters(other, same_spatial, with_linked=False)
        return notes

    # ---------- in-situ series: the frame (or frame range) a dataset holds ----------

    @staticmethod
    def insitu_series(ds: Dataset) -> InsituSeries:
        if ds.insitu is None:
            raise ValueError(f"{ds.title} is not an in-situ series")
        return ds.insitu

    def show_insitu_frame(self, ds: Dataset, k: int) -> Dataset:
        """Make ``ds`` hold frame ``k`` (0-based; one frame is read, or taken
        from the cache) and read its neighbours ahead. Raises IndexError for a
        frame outside the series."""
        series = self.insitu_series(ds)
        data = series.read(k)
        self._set_insitu_view(ds, data, ViewState("frame", k, k))
        series.schedule_ahead(k)
        return ds

    def apply_insitu_range(self, ds: Dataset, data: np.ndarray, mode: str, start: int, end: int) -> Dataset:
        """Make ``ds`` hold ``data``, the mean or sum of frames ``start``…``end``
        (from ``InsituSeries.reduce``)."""
        series = self.insitu_series(ds)
        self._set_insitu_view(ds, data, ViewState(mode if start != end else "frame", start, end))
        if start == end:
            series.schedule_ahead(start)
        else:
            series.cancel_ahead()
        return ds

    def _set_insitu_view(self, ds: Dataset, data: np.ndarray, state: ViewState) -> None:
        """Swap ``data`` into ``ds`` (same dataset, same revision: the filters,
        windows, cursor and contrast stay) and remake what follows the data:
        energy-window maps and computed element maps. Results that are
        snapshots of a frame (background removal, thickness, ZLP shift) are
        kept as they are made; fits are removed."""
        series = self.insitu_series(ds)
        with ds._load_lock:
            ds.refresh_data(series.fresh_signal(ds.signal, data))
            series.view = state
        if ds.kind is DatasetKind.SPECTRUM_IMAGE and not ds.is_line_scan:
            for m in self.computed_maps(ds):
                new_map = self._make_map(ds, m.element)
                if new_map is None:
                    self._remove(m)
                else:
                    m.refresh_data(new_map)
            self._remake_window_maps(ds, keep_display=True)
            for f in self._fit_maps(ds):
                self._remove(f)  # made from another frame: fit again to update

    # ---------- drift correction of stacks ----------

    def _averages(self, stack: Dataset) -> list[Dataset]:
        return [d for d in self.siblings(stack) if d.derived_from == stack.id and d.origin == "aligned-average"]

    @staticmethod
    def _view_changed(ds: Dataset) -> None:
        """The data read from ``ds`` changed: clients refetch (filter_rev),
        derived arrays are remade."""
        ds.filter_rev += 1
        ds.thumbnails.clear()
        ds.cache = {}

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
