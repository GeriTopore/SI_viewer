"""In-memory registry of open files and the datasets (signals) they contain."""

from __future__ import annotations

import itertools
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
    # A diffraction pattern at every scan position (two scan, two pattern axes).
    DIFFRACTION4D = "4dstem"
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
    if sig == 2 and nav == 2:
        return DatasetKind.DIFFRACTION4D
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
# 4D-STEM data up to this size is loaded into RAM when its file opens (every
# view of it reads the whole cube); larger data stays lazy.
DIFFRACTION4D_IN_MEMORY_LIMIT = 4 << 30

# What a map made by fitting every pixel can show: the fitted intensity, and
# the at% composition made from the maps of the fit.
FIT_QUANTITIES = ("counts", "at%")


class NotRebinnable(ValueError):
    """Rebinning was asked of a dataset that isn't a (2D) spectrum image."""


# Order of the operations recorded on datasets, across datasets: a saved
# project replays them in this order (see ``Dataset.ops``).
_OP_SEQ = itertools.count(1)


def _op(record: dict, seq: int | None = None) -> dict:
    """``record`` with its place in the order of operations."""
    return {**record, "seq": next(_OP_SEQ) if seq is None else seq}


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
    dataset.ops.append(_op({"op": "rebin", "scale": [int(f) for f in scale]}))


def _new_id() -> str:
    return uuid.uuid4().hex[:10]


def _defined(value) -> str:
    """Axis names/units are traits ``Undefined`` (or the string "None") when unset."""
    return value if isinstance(value, str) and value != "None" else ""


def _set_microscope(signal, params: dict) -> None:
    """exSpy's ``set_microscope_parameters`` (or the same metadata keys for a
    signal that isn't an EELS spectrum yet)."""
    if hasattr(signal, "set_microscope_parameters"):
        signal.set_microscope_parameters(**params)
        return
    paths = {
        "beam_energy": "Acquisition_instrument.TEM.beam_energy",
        "convergence_angle": "Acquisition_instrument.TEM.convergence_angle",
        "collection_angle": "Acquisition_instrument.TEM.Detector.EELS.collection_angle",
    }
    for k, v in params.items():
        signal.metadata.set_item(paths[k], v)


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
    # "fitted" (per-pixel fits), "window" (an energy-window map),
    # "background" (EELS spectra with the background removed), "background-r"
    # (the power-law exponent map of one), "thickness" (the thickness of an
    # EELS low loss: a map, or a profile for a line scan), "zlp-shift" (the
    # energy shift zero-loss alignment applied at every pixel of a low loss)
    # "deconvolved" (EELS spectra with plural scattering removed),
    # "line-scan" (spectra along a line of a spectrum image), "aligned-sum"
    # (the frames of an in-situ series registered and summed), or the result
    # of an operation run as a new dataset: "rebinned", "zlp-aligned" (with
    # its linked ones) or "aligned" (a stack's frames registered).
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
    # Fitted maps: the image of each quantity it has (see FIT_QUANTITIES) and
    # the one shown (``signal`` is that image).
    quantities: dict[str, object] = field(default_factory=dict, repr=False)
    quantity: str | None = None
    # EELS spectrum images: the fit of every pixel (see
    # processing/eels_model.PixelFits) and the data it was made from
    # (``data_version``); shown at the cursor while that is current.
    pixel_fits: object | None = field(default=None, repr=False)
    pixel_fits_version: tuple = ()
    # Curve fitting of every pixel (processing/curve_fit.PixelFits), likewise.
    curve_fits: object | None = field(default=None, repr=False)
    curve_fits_version: tuple = ()
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
    # The operations that changed the loaded data, oldest first, as records
    # ({"op": "rebin", "scale": [...]}, {"op": "zlp-align", "options": {...}},
    # {"op": "align", ...}): what a saved project replays on the raw data.
    # ``rebinned`` / ``zlp_aligned`` are their labels.
    ops: list[dict] = field(default_factory=list)
    # A copy made by an operation run as a new dataset: how many of its
    # source's ``ops`` the source had then (its own ``ops`` follow).
    copied_at: int | None = None
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
    # DPC images ("dpc"): how they were made: {"inputs": {"A": id, ...},
    # "normalise", "theta" (degrees), "mirror", "output": "x" |
    # "y" | "idpc"}.
    dpc: dict | None = None
    # PCA (M19a): "pca" (the components' maps, a stack), "pca-factors" (their
    # spectra) and "pca-denoised" (an SI rebuilt from the first n): {"source_id",
    # "window": [lo, hi] (energies), "channels": [c0, c1), "poisson",
    # "components", "explained": [...], and "n" for a denoised SI}.
    pca: dict | None = None
    # A curve model of every pixel (origin "curve-model", M21c): the fitted
    # model as a spectrum image, its maps under it. {"source_id", "model": the
    # model per pixel (processing/curve_fit.ModelSpec, dumped)}.
    curve_model: dict | None = None
    # Atomic-resolution maps (kemstem, M22; origin "atomic"): {"output"
    # (amplitude, grating, phase, strain-par, strain-perp, exx, eyy, exy,
    # oxy), "peaks": the FFT peaks' numbers, "freqs": their frequencies
    # (cycles/px), "sigma", "phase", "lock_sigma", "edge", "mask_threshold",
    # "coarsening" (image px)}.
    atomic: dict | None = None
    # A mask made from an image (origin "mask", processing/masks.py): 0/1
    # data; {"source_id", the MaskOptions it was made with, "fraction" of
    # the pixels in it}.
    mask: dict | None = None
    # A virtual image of 4D-STEM data (origin "virtual", processing/fourd.py):
    # {"source_id", "detector" (Detector fields, pattern pixels), "response"}.
    virtual: dict | None = None
    # Bragg disks found in 4D-STEM data (processing/disks.py): on the "Bragg
    # vectors" table (origin "disks": one row per disk, VECTOR_FIELDS) and its
    # "Bragg vector map" image (origin "bragg-map"): {"source_id", "options"
    # (DiskOptions), "template" ({"kind", "cx", "cy", "r"} pattern px),
    # "count", "positions" (scan positions)}.
    disks: dict | None = None
    # The direct beam found in 4D-STEM data (processing/centring.py): on its
    # "Beam shift x" / "Beam shift y" images (origin "beam-shift": pyxem's
    # shifts that centre each pattern, px): {"source_id", "axis" ("x"/"y"),
    # "options" (BeamOptions), "range" (each shift's min, max)}.
    beam: dict | None = None
    # 4D-STEM patterns calibrated (processing/calibration.py): {"scale" (per
    # px), "units", "ring" (the ellipse fitted: cx, cy, a, b, angle, rms,
    # radius, ellipticity; pattern px) or None, "affine" (pyxem's, making the
    # ring circular) or None, "how" ({"standard", "hkl", "d"} / {"camera_length",
    # "pixel", "kv"} / {"entered"})}. The pattern axes carry the scale, and 0
    # at the ring's centre when there is one; the data isn't resampled.
    calibration: dict | None = None
    # Each scan position's origin ("Origin x / y", "Origin x / y (fit)" of
    # 4D-STEM data; origin "origin-map", pattern px): {"axis", "kind"
    # ("measured" / "fitted"), "source_id"}.
    origin_map: dict | None = None
    # Microscope parameters the user set (EELS: "beam_energy" kV,
    # "convergence_angle" and "collection_angle" mrad), kept over recomputes.
    microscope: dict | None = None
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
    def data_version(self) -> tuple:
        """Changes whenever the data is recomputed, replaced or filtered."""
        return (self.revision, self.data_rev, self.filter_rev)

    @property
    def current_pixel_fits(self):
        """The fit of every pixel, if made from the data as it is now."""
        return self.pixel_fits if self.pixel_fits_version == self.data_version else None

    @property
    def current_curve_fits(self):
        """The curve fit of every pixel, if made from the data as it is now."""
        return self.curve_fits if self.curve_fits_version == self.data_version else None

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

    def _keep_eels_elements(self, signal) -> None:
        """Give ``signal`` (about to replace this dataset's) the elements chosen
        for an EELS model fit, when it comes without any (another frame)."""
        from .processing.eels_model import set_elements

        elements = self.signal.metadata.get_item("Sample.elements", None)
        if not elements or technique_of(self.signal) != "EELS" or technique_of(signal) != "EELS":
            return
        if not signal.metadata.get_item("Sample.elements", None):
            set_elements(signal, list(elements))

    def set_microscope(self, **params: float | None) -> None:
        """Set EELS microscope parameters (beam_energy kV, convergence_angle
        and collection_angle mrad; None leaves one as it is) with exSpy's
        ``set_microscope_parameters``, on every version of the data."""
        params = {k: float(v) for k, v in params.items() if v is not None}
        for k, v in params.items():
            if k not in ("beam_energy", "convergence_angle", "collection_angle"):
                raise ValueError(f"Unknown microscope parameter {k}")
            if not (v > 0 and v < 1e4):
                raise ValueError(f"The {k.replace('_', ' ')} must be a positive number")
        self.microscope = {**(self.microscope or {}), **params}
        for s in (self.signal, self.original_signal, self.aligned_signal):
            if s is not None:
                _set_microscope(s, self.microscope)

    def replace_signal(self, signal) -> None:
        """Swap in recomputed data (new revision, caches cleared)."""
        if self.custom_title:
            signal.metadata.General.title = self.custom_title
        if self.microscope:
            _set_microscope(signal, self.microscope)
        self._keep_eels_elements(signal)
        self.signal = signal
        self.revision += 1
        self.pixel_fits = self.curve_fits = None  # made from the old data
        self.thumbnails.clear()
        self.cache = {}  # a new dict: a result still being computed from the old data lands in the old one

    def refresh_data(self, signal) -> None:
        """Swap in other data of the same kind (the next frame): the same
        revision, so clients keep their display settings, but a new ``data_rev``
        so they refetch."""
        if self.custom_title:
            signal.metadata.General.title = self.custom_title
        self._keep_eels_elements(signal)
        self.signal = signal
        self.data_rev += 1
        self.pixel_fits = self.curve_fits = None
        self.thumbnails = {}
        self.cache = {}

    def summary(self) -> dict:
        from .processing.element_maps import MapOptions
        from .processing.eels import eels_info, microscope_info
        from .processing.eels_model import edges_info
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
            "edges": edges_info(s) if technique == "EELS" else None,
            "microscope": microscope_info(s) if technique == "EELS" else None,
            # Images of a segmented detector: the segments summed (from the
            # acquisition record), e.g. {"name": "DFS", "segments": [0, 4], ...}.
            "detector": _detector_info(self.loaded_signal),
            "origin": self.origin,
            "element": self.element,
            "pixel_fit": self.current_pixel_fits is not None,
            # The curve fit of every pixel: the model it was made with.
            "curve_pixel_fit": (
                self.current_curve_fits.spec.model_dump() if self.current_curve_fits is not None else None
            ),
            "quantities": list(self.quantities),
            "quantity": self.quantity,
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
            "dpc": self.dpc,
            "pca": self.pca,
            "curve_model": self.curve_model,
            "atomic": self.atomic,
            "mask": self.mask,
            "virtual": self.virtual,
            "disks": self.disks,
            "beam": self.beam,
            "calibration": self.calibration,
            "origin_map": self.origin_map,
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
    # Opened from a saved analysis without its raw file: its datasets are the
    # data stored in the analysis.
    raw_missing: bool = False
    # Its raw file's fingerprint when the analysis was saved (raw_missing).
    fingerprint: dict | None = None
    # Opened from a copy of the raw file embedded in a saved analysis (``path``
    # is that copy): where the raw file was ({"path", "fingerprint"}).
    saved_ref: dict | None = None

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
            "raw_missing": self.raw_missing,
            "embedded": self.saved_ref is not None,
            "experiment": experiment.summary() if experiment else None,
            "role": self.role(session) if experiment else None,
            "datasets": [session.datasets[d].summary() for d in self.dataset_ids],
        }


def _detector_info(signal) -> dict | None:
    """The detector segments an image was summed from, if recorded."""
    det = signal.metadata.get_item("Acquisition_instrument.TEM.Detector")
    if det is None or not det.has_item("segments"):
        return None
    segments = [int(x) for x in det.segments]
    polarities = list(det.get_item("segment_polarities", [""] * len(segments)))
    return {
        "name": str(det.get_item("name", "")),
        "segments": segments,
        "polarities": [str(x) for x in polarities],
        "label": signal.metadata.General.get_item("original_title"),
    }


def spectrum_normalisation(signal, norm: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per spectrum (navigation shape) offset and scale normalising it as the
    spectrum plot does: ``mode`` "range" to 0–1 (minimum to maximum), "value"
    divided by its value at ``lo``, "mean" by its mean over ``lo``–``hi``. A
    spectrum whose reference is zero or less (or flat) is left as it is."""
    from .processing.background import EnergyAxis

    data = np.asarray(signal.data, np.float32)
    mode = norm.get("mode", "range")
    with np.errstate(all="ignore"):
        if mode == "range":
            lo, hi = np.nanmin(data, axis=-1), np.nanmax(data, axis=-1)
            ok = hi > lo
            return np.where(ok, lo, 0).astype(np.float32), np.where(ok, hi - lo, 1).astype(np.float32)
        axis = EnergyAxis.of(signal)
        n = data.shape[-1]
        at = lambda e: int(min(n - 1, max(0, round((float(e) - axis.offset) / axis.scale))))  # noqa: E731
        e0 = norm.get("lo")
        if e0 is None:
            raise ValueError("Normalisation needs an energy")
        e1 = e0 if mode == "value" else norm.get("hi", e0)
        c0, c1 = sorted((at(e0), at(e1)))
        ref = np.nanmean(data[..., c0:c1 + 1], axis=-1)
        ok = ref > 0
        return np.zeros(ref.shape, np.float32), np.where(ok, ref, 1).astype(np.float32)


# HyperSpy's decomposition algorithms: their name here, and whether they
# centre the data (scikit-learn's).
DECOMPOSITIONS: dict[str, tuple[str, bool]] = {
    "SVD": ("PCA", False),
    "MLPCA": ("MLPCA", False),
    "sklearn_pca": ("PCA", True),
    "NMF": ("NMF", False),
    "sparse_pca": ("Sparse PCA", True),
    "mini_batch_sparse_pca": ("Sparse PCA", True),
    "RPCA": ("RPCA", False),
    "ORPCA": ("ORPCA", False),
    "ORNMF": ("ORNMF", False),
}


class Session:
    def __init__(self) -> None:
        self.files: dict[str, FileEntry] = {}
        self.datasets: dict[str, Dataset] = {}
        # Link groups of EELS spectrum images (two or more, at most one low
        # loss): group id -> member dataset ids.
        self.eels_groups: dict[str, list[str]] = {}
        self.experiments: dict[str, Experiment] = {}
        # Deletions that can be undone, oldest first (see ``delete_dataset``).
        self.undoable: dict[str, dict] = {}
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
        for d in datasets:
            if d.kind is DatasetKind.DIFFRACTION4D and d.lazy and d.signal.data.nbytes <= DIFFRACTION4D_IN_MEMORY_LIMIT:
                d.materialise()
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
            for token in [t for t, u in self.undoable.items() if u["file_id"] == file_id]:
                del self.undoable[token]
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

    def add_line_scans(self, si: Dataset, line, name: str, with_linked: bool = True) -> list[Dataset]:
        """Cut a line scan from the 2D spectrum image ``si`` along ``line`` (a
        ``LineSpec`` in its scan's pixels; see processing/line_scan.py), titled
        "<si> · <name>" and listed under it. With ``with_linked``, the spectrum
        images linked to it are cut along the same line too and their line
        scans linked as they are. Made once from the loaded data (filters are
        not applied). Returns the new datasets, ``si``'s first. ValueError for
        a line that can't be cut."""
        from .processing.line_scan import line_scan_signal

        if si.kind is not DatasetKind.SPECTRUM_IMAGE or si.is_line_scan:
            raise ValueError(f"{si.title} is not a 2D spectrum image")
        sources = [si, *(self.linked_of(si) if with_linked else [])]
        signals = []
        for src in sources:
            src.materialise()
            signals.append(line_scan_signal(src.signal, line, f"{src.title} · {name}"))
        made = []
        for src, signal in zip(sources, signals):
            frame = None if src.insitu is None else src.insitu.view.label()
            ds = Dataset(
                id=_new_id(), file_id=src.file_id, signal=signal, kind=classify(signal),
                origin="line-scan", parent_id=src.id, technique_source=src.technique_source,
                eels_role=src.eels_role, eels_role_source=src.eels_role_source, source_frame=frame,
            )
            with self._lock:
                self.datasets[ds.id] = ds
                self.files[src.file_id].dataset_ids.append(ds.id)
            made.append(ds)
        if len(made) > 1:
            self._join(made)
        return made

    # ---------- operations run as a new dataset ----------

    def copy_as(self, src: Dataset, origin: str, title: str, signal=None) -> Dataset:
        """A new dataset under ``src`` holding a copy of its loaded data (or
        ``signal``), as the starting point of an operation run as a new dataset
        (``origin``). It keeps ``src``'s calibration, technique, EELS role and
        filters; it is not linked to anything."""
        import copy

        if signal is None:
            src.materialise()
            signal = src.loaded_signal.deepcopy()
        signal.metadata.General.title = title
        ds = Dataset(
            id=_new_id(), file_id=src.file_id, signal=signal, kind=classify(signal), origin=origin,
            parent_id=src.id, technique_source=src.technique_source, eels_role=src.eels_role,
            eels_role_source=src.eels_role_source, filters=copy.deepcopy(src.filters),
            map_options=copy.deepcopy(src.map_options), rebinned=list(src.rebinned),
            zlp_aligned=list(src.zlp_aligned), copied_at=len(src.ops),
        )
        with self._lock:
            self.datasets[ds.id] = ds
            ids = self.files[src.file_id].dataset_ids
            ids.insert(ids.index(src.id) + 1, ds.id)
        return ds

    # ---------- drift-corrected sum of an in-situ series ----------

    def alignment_source(self, ds: Dataset) -> dict:
        """What the aligned sum of the series ``ds`` can use: its frames, and
        its simultaneous ADF (sub-scan factor k: ADF pixels per scan pixel)
        with its applied alignment, if any."""
        from .insitu_series import adf_stack

        series = self.insitu_series(ds)
        found = adf_stack(ds, self.siblings(ds))
        adf = None
        if found is not None:
            stack, k = found
            a = stack.alignment
            adf = {
                "id": stack.id, "title": stack.title, "k": k, "aligned": a is not None,
                "frames": len(a["drift"]["frames"]) if a else 0,
            }
        return {"frames": series.n_raw, "adf": adf}

    def aligned_sum_shifts(
        self, ds: Dataset, source: str, raw: bytes | None = None, units: str = "adf",
    ) -> tuple[list[int], np.ndarray, str]:
        """(frames to sum, shifts (frames, 2): the (x, y) shift that aligns each
        recorded frame in pixels of the scan, a description). ``source``
        "adf": the alignment applied to the simultaneous ADF stack (its drift,
        divided by the sub-scan factor). "npy": the ``.npy`` file ``raw``,
        shifts in pixels of the ADF (``units`` "adf") or of the scan ("scan");
        frames whose shift is NaN are left out. ValueError (plain English)
        when they can't be used."""
        from .insitu_series import adf_stack
        from .processing.insitu_align import parse_shifts

        series = self.insitu_series(ds)
        n = series.n_raw
        found = adf_stack(ds, self.siblings(ds))
        if source == "adf":
            if found is None:
                raise ValueError(f"{ds.title} has no simultaneous ADF stack on its scan")
            stack, k = found
            a = stack.alignment
            if a is None:
                raise ValueError(f"Align {stack.title} first (Process tab, Align frames)")
            shifts = np.full((n, 2), np.nan)
            for f, x, y in zip(a["drift"]["frames"], a["drift"]["x"], a["drift"]["y"]):
                if 0 <= f < n:
                    shifts[f] = (-x / k, -y / k)  # the shift that aligns = minus the drift
            how = f"alignment of {stack.title} (÷{k}, its sub-scan)" if k > 1 else f"alignment of {stack.title}"
        elif source == "npy":
            if raw is None:
                raise ValueError("Choose a .npy file of shifts")
            shifts = parse_shifts(raw, n)
            k = 1
            if units == "adf":
                if found is None:
                    raise ValueError("There is no ADF stack to take the pixel size from: give the shifts in scan pixels")
                k = found[1]
                shifts = shifts / k
            how = f"imported shifts ({'ADF' if units == 'adf' else 'scan'} pixels{f', ÷{k}' if k > 1 else ''})"
        else:
            raise ValueError(f"Unknown source of shifts: {source}")
        frames = [int(f) for f in np.flatnonzero(~np.isnan(shifts).any(axis=1))]
        if len(frames) < 2:
            raise ValueError("At least two frames need a shift")
        h, w = series.frame_shape[:2]
        if np.abs(shifts[frames]).max() >= min(h, w):
            raise ValueError("A shift is as large as the scan: the shifts don't fit this series")
        return frames, shifts, how

    def compute_aligned_sum(self, ds: Dataset, frames: list[int], shifts: np.ndarray, report=lambda f, m=None: None):
        """(sum, crop): see ``processing.insitu_align.aligned_sum``; the
        recorded frames are read one at a time, binned in y, x and E as the
        series is."""
        from .processing.insitu_align import aligned_sum

        series = self.insitu_series(ds)
        return aligned_sum(series.read_recorded, frames, shifts, report)

    def add_aligned_sum(self, ds: Dataset, data: np.ndarray, crop, frames: list[int], how: str) -> Dataset:
        """Add the aligned sum ``data`` (from ``compute_aligned_sum``) of the
        series ``ds`` as a spectrum image under it: the scan cropped to
        ``crop`` (r0, r1, c0, c1), calibrated as the series."""
        r0, r1, c0, c1 = crop
        ds.materialise()
        signal = ds.signal.inav[c0:c1, r0:r1].deepcopy()  # natural order: x, y
        signal.data = data
        title = f"{ds.title} (aligned sum, {len(frames)} frames)"
        signal.metadata.General.title = title
        signal.metadata.set_item("Processing.si_viewer.aligned_sum", {
            "frames": [f + 1 for f in frames], "shifts": how, "crop_rows": [r0, r1], "crop_columns": [c0, c1],
        })
        out = Dataset(
            id=_new_id(), file_id=ds.file_id, signal=signal, kind=classify(signal), origin="aligned-sum",
            parent_id=ds.id, technique_source=ds.technique_source, eels_role=ds.eels_role,
            eels_role_source=ds.eels_role_source,
        )
        with self._lock:
            self.datasets[out.id] = out
            self.files[ds.file_id].dataset_ids.append(out.id)
        return out

    # ---------- time scans of an in-situ series ----------

    def time_scan_sources(self, ds: Dataset, with_linked: bool = True) -> list[Dataset]:
        """The in-situ series a time scan of ``ds`` is cut from: ``ds``, and the
        series linked to it on the same scan with the same frames."""
        series = self.insitu_series(ds)
        if ds.kind is not DatasetKind.SPECTRUM_IMAGE or ds.is_line_scan:
            raise ValueError(f"{ds.title} is not a series of 2D spectrum images")
        scan = series.frame_shape[:2]
        others = [
            d for d in (self.linked_of(ds) if with_linked else [])
            if d.insitu is not None and d.insitu.n == series.n and d.insitu.frame_shape[:2] == scan
        ]
        for d in (ds, *others):
            d.materialise()  # one frame: the time scan takes its axes and metadata
        return [ds, *others]

    @staticmethod
    def time_scan_pick(src: Dataset, pixel: tuple[int, int, int] | None, region):
        """The function taking a frame of ``src``'s series to the mean spectrum
        of the ``pixel`` block (x, y, size) or the ``region`` (a ``Region``),
        in scan pixels. ValueError if they hold no pixel of the scan."""
        from .processing.regions import pixel_spectrum, region_mask, region_spectrum

        h, w = src.insitu.frame_shape[:2]
        if pixel is not None:
            x, y, size = pixel
            if not (0 <= x < w and 0 <= y < h):
                raise ValueError(f"Pixel ({x}, {y}) is outside the {w}×{h} scan")

            def pick(frame):
                total, n = pixel_spectrum(frame, x, y, size)
                return total / n
        else:
            if not region_mask((h, w), region)[2].any():
                raise ValueError("The region holds no pixel of the scan")

            def pick(frame):
                total, n = region_spectrum(frame, region)
                return total / n
        return pick

    def time_scan_spectra(self, src: Dataset, pixel, region, report=lambda f, m=None: None) -> np.ndarray:
        """The mean spectrum of the same pixels in every frame of ``src``'s
        series (frames, channels); see ``time_scan_pick``."""
        return src.insitu.spectra_over_time(self.time_scan_pick(src, pixel, region), report)

    def add_time_scans(self, made: list[tuple[Dataset, np.ndarray]], name: str, source: dict) -> list[Dataset]:
        """Add a line scan over time under each series of ``made`` ((series,
        spectra) from ``time_scan_spectra``), titled "<series> · <name>"; more
        than one are linked as their series are. Returns the new datasets."""
        from .processing.line_scan import time_scan_signal

        out = []
        for src, spectra in made:
            if src.id not in self.datasets:
                raise ValueError(f"{src.title} was closed while its frames were read")
            signal = time_scan_signal(src.signal, spectra, src.insitu.times, f"{src.title} · {name}", source)
            ds = Dataset(
                id=_new_id(), file_id=src.file_id, signal=signal, kind=classify(signal),
                origin="line-scan", parent_id=src.id, technique_source=src.technique_source,
                eels_role=src.eels_role, eels_role_source=src.eels_role_source,
            )
            with self._lock:
                self.datasets[ds.id] = ds
                self.files[src.file_id].dataset_ids.append(ds.id)
            out.append(ds)
        if len(out) > 1:
            self._join(out)
        return out

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

    def apply_zlp_alignment(self, datasets: list[Dataset], originals: list, result, options, in_place: bool = True) -> Dataset:
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
        seq = next(_OP_SEQ)  # one operation on the whole group
        if not in_place:
            # The aligned spectra as new datasets under their sources, linked
            # as the sources are; the sources are left as they are.
            made = []
            for d, signal in zip(datasets, result.signals):
                new = self.copy_as(d, "zlp-aligned", f"{d.title} (ZLP aligned)", signal=signal)
                new.zlp_aligned.append(options.label)
                new.ops.append(_op({"op": "zlp-align", "options": options.model_dump()}, seq))
                made.append(new)
            if len(made) > 1:
                self._join(made)
            return self.set_zlp_shift(made[0], result.shifts)
        shift = result.mean_shift
        for d, signal in zip(datasets, result.signals):
            d.replace_signal(signal)
            d.kind = classify(signal)
            d.zlp_aligned.append(options.label)
            d.ops.append(_op({"op": "zlp-align", "options": options.model_dump()}, seq))
        for d in datasets:
            self._remake_after_zlp(d, shift)
        if ll.eels_group is not None:
            self._refresh_group(ll.eels_group)  # linked ones are ordered by their energy offset
        return self.set_zlp_shift(ll, result.shifts)

    def shift_energy(self, ds: Dataset, shift: float, in_place: bool = True, record: bool = True) -> Dataset:
        """ZLP alignment by hand: move the energy axis of the EELS spectrum
        image (or LineScan) ``ds`` by ``shift`` (its units; positive: the
        spectra move up), with what was made from it. Not ``in_place``: a
        shifted copy, a new dataset under ``ds`` (origin "zlp-aligned").
        Returns the dataset shifted. ``record``: add it to ``ops``."""
        import math

        from .processing.zlp import round_energy

        if ds.kind is not DatasetKind.SPECTRUM_IMAGE or technique_of(ds.loaded_signal) != "EELS":
            raise ValueError(f"{ds.title} is not an EELS spectrum image")
        if ds.insitu is not None:
            raise ValueError(f"{ds.title} is a frame of an in-situ series: the next frame would undo the shift")
        if not math.isfinite(shift) or shift == 0:
            raise ValueError("Enter a shift other than 0")
        label = f"shifted {round_energy(shift):+g} {ds.signal.axes_manager.signal_axes[0].units or ''}".strip()
        if not in_place:
            ds.materialise()
            target = self.copy_as(ds, "zlp-aligned", f"{ds.title} (shifted)")
        else:
            target = ds
        target.materialise()
        signal = target.signal
        signal.axes_manager.signal_axes[0].offset += shift
        target.replace_signal(signal)
        target.zlp_aligned.append(label)
        if record:
            target.ops.append(_op({"op": "energy-shift", "shift": shift}))
        if in_place:
            self._remake_after_zlp(target, shift)
        return target

    # ---------- DPC and iDPC from the quadrants of a segmented detector ----------

    def dpc_quadrants(self, inputs: dict[str, str]) -> dict[str, Dataset]:
        """The four images ``inputs`` (letter -> dataset id) checked: 2D
        images of one size, each summed from segments of the quadrant its
        letter stands for (A: 0, B: 1, C: 2, D: 3, from the segment numbers
        recorded at acquisition). ValueError otherwise."""
        from .processing.dpc import LETTERS, quadrant

        if sorted(inputs) != list(LETTERS):
            raise ValueError("Choose an image for each of A, B, C and D")
        out = {}
        for q, letter in enumerate(LETTERS):
            ds = self.get_dataset(inputs[letter])
            if ds.kind not in (DatasetKind.IMAGE, DatasetKind.STACK):
                raise ValueError(f"{ds.title} is not an image or a stack")
            det = _detector_info(ds.loaded_signal)
            if det is None:
                raise ValueError(f"{ds.title} has no detector segments recorded")
            found = quadrant(det["segments"])
            if found != q:
                raise ValueError(
                    f"{ds.title} is segments {', '.join(map(str, det['segments']))} (quadrant {LETTERS[found]}), not {letter}"
                )
            out[letter] = ds
        if len({d.kind for d in out.values()}) != 1:
            raise ValueError("The four must be all images or all stacks")
        shapes = {tuple(d.signal.data.shape) for d in out.values()}
        if len(shapes) != 1:
            raise ValueError("The four must be of one size (stacks: the same frames, all aligned or none)")
        return out

    @staticmethod
    def _dpc_arrays(q: dict[str, Dataset]) -> list[np.ndarray]:
        """The data of the four quadrant images (or stacks), as shown."""
        for d in q.values():
            d.materialise()
        return [np.asarray(q[k].signal.data, np.float64) for k in "ABCD"]

    def dpc_rotation(self, inputs: dict[str, str], normalise: bool = True) -> dict:
        """The rotation (and mirror) of the detector found from the data."""
        from .processing.dpc import differences, find_rotation

        dx, dy = differences(*self._dpc_arrays(self.dpc_quadrants(inputs)), normalise=normalise)
        if dx.ndim == 3:  # stacks: from their mean (the rotation is the same in every frame)
            dx, dy = dx.mean(axis=0), dy.mean(axis=0)
        return find_rotation(dx, dy)

    def add_atomic_maps(self, image: Dataset, made: list[dict], settings: dict) -> list[Dataset]:
        """Maps of kemstem's analysis (``processing.atomic.fourier_maps``)
        under ``image``, each with its calibration. A map of the same output
        and peaks made before is replaced in place. Returns them."""
        with self._lock:
            earlier = {
                (d.atomic["output"], tuple(d.atomic["peaks"])): d for d in self.datasets.values()
                if d.origin == "atomic" and d.parent_id == image.id and d.atomic and "peaks" in d.atomic
            }
        out: list[Dataset] = []
        for m in made:
            signal = image.signal._deepcopy_with_new_data(m["data"])
            signal.metadata.General.title = m["title"]
            info = {**settings, "output": m["output"], "peaks": m["peaks"],
                    "freqs": [settings["freqs"][str(n)] for n in m["peaks"]]}
            info.pop("all_freqs", None)
            ds = earlier.get((m["output"], tuple(m["peaks"])))
            if ds is not None:
                ds.replace_signal(signal)
                ds.atomic = info
            else:
                ds = Dataset(id=_new_id(), file_id=image.file_id, signal=signal, kind=DatasetKind.IMAGE,
                             origin="atomic", parent_id=image.id, atomic=info)
                with self._lock:
                    self.datasets[ds.id] = ds
                    ids = self.files[image.file_id].dataset_ids
                    prev = [o.id for o in out if o.id in ids]
                    at = ids.index(prev[-1]) if prev else ids.index(image.id)
                    ids.insert(at + 1, ds.id)
            out.append(ds)
        return out

    def set_columns(self, image: Dataset, table: np.ndarray, settings: dict) -> Dataset:
        """The image's atomic columns (``processing.atomic.refine_columns``'s
        table) as a "Columns" dataset under it (kind "other": a table, shown
        over its image). Made again in place. Returns it."""
        import hyperspy.api as hs

        from .processing.atomic import COLUMN_FIELDS

        signal = hs.signals.BaseSignal(np.ascontiguousarray(table, dtype=np.float32))
        signal.metadata.General.title = "Columns"
        signal.metadata.set_item("Atomic.fields", list(COLUMN_FIELDS))
        info = {**settings, "output": "columns", "count": int(len(table)),
                "failed": int(np.count_nonzero(table[:, -1])) if len(table) else 0}
        with self._lock:
            earlier = next((d for d in self.datasets.values() if d.origin == "atomic" and d.parent_id == image.id
                            and d.atomic and d.atomic.get("output") == "columns"), None)
        if earlier is not None:
            # What was made from the old columns (sublattices, neighbourhoods) goes.
            made = [d for d in self.siblings(earlier) if d.parent_id == earlier.id and d.origin == "atomic"]
            gone = {d.id for d in made}
            for d in made:
                self._remove_tree(d)
            self._drop_neighbourhoods_of(gone | {earlier.id})
            earlier.replace_signal(signal)
            earlier.atomic = info
            return earlier
        ds = Dataset(id=_new_id(), file_id=image.file_id, signal=signal, kind=DatasetKind.OTHER,
                     origin="atomic", parent_id=image.id, atomic=info)
        with self._lock:
            self.datasets[ds.id] = ds
            ids = self.files[image.file_id].dataset_ids
            ids.insert(ids.index(image.id) + 1, ds.id)
        return ds

    def sublattices_of(self, columns: Dataset) -> list[Dataset]:
        """The sublattices of a Columns dataset (datasets under it), in order."""
        with self._lock:
            subs = [d for d in self.datasets.values() if d.parent_id == columns.id
                    and d.atomic and d.atomic.get("output") == "sublattice"]
        return sorted(subs, key=lambda d: d.atomic.get("index", 0))

    def column_labels(self, columns: Dataset) -> np.ndarray | None:
        """Each column's sublattice (its place among ``sublattices_of``; NaN:
        none), or None when there are none."""
        subs = self.sublattices_of(columns)
        if not subs:
            return None
        n = len(columns.signal.data)
        labels = np.full(n, np.nan)
        for k, sub in enumerate(subs):
            rows = np.asarray(sub.signal.data, dtype=np.int64).ravel()
            labels[rows[(rows >= 0) & (rows < n)]] = k
        return labels

    def set_sublattices(self, columns: Dataset, labels: list[int | None], names: list[str]) -> list[Dataset]:
        """Split a Columns dataset's columns into sublattices: ``labels`` gives
        each column's (an index into ``names``; None: none). Each sublattice
        is a dataset under the Columns (kind "other", origin "atomic"),
        holding its columns' rows; earlier ones are replaced. No names (and
        no labels) removes them. Returns the new sublattices."""
        import hyperspy.api as hs

        if not (columns.atomic and columns.atomic.get("output") == "columns"):
            raise ValueError(f"{columns.title} is not a table of columns")
        n = len(columns.signal.data)
        if len(labels) != n:
            raise ValueError(f"{len(labels)} labels for {n} columns")
        if any(x is not None and not 0 <= x < len(names) for x in labels):
            raise ValueError("A label names no sublattice")
        names = [x.strip() for x in names]
        if len(set(names)) != len(names) or not all(names):
            raise ValueError("The sublattices need distinct names")
        old_subs = self.sublattices_of(columns)
        for old in old_subs:
            self._remove_tree(old)
        self._drop_neighbourhoods_of({d.id for d in old_subs})
        lab = np.array([-1 if x is None else x for x in labels], dtype=np.int64)
        made = []
        with self._lock:
            ids = self.files[columns.file_id].dataset_ids
            at = ids.index(columns.id) + 1
            for k, name in enumerate(names):
                rows = np.flatnonzero(lab == k).astype(np.int32)
                signal = hs.signals.BaseSignal(rows)
                signal.metadata.General.title = f"Sublattice {name}"
                ds = Dataset(id=_new_id(), file_id=columns.file_id, signal=signal, kind=DatasetKind.OTHER,
                             origin="atomic", parent_id=columns.id,
                             atomic={"output": "sublattice", "index": k, "count": int(len(rows))})
                self.datasets[ds.id] = ds
                ids.insert(at + k, ds.id)
                made.append(ds)
        return made

    def add_sublattice(self, columns: Dataset, rows: np.ndarray, info: dict | None = None) -> Dataset:
        """One more sublattice of a Columns dataset (its columns' ``rows``),
        next to those it has, named by the first free letter. Returns it."""
        import hyperspy.api as hs

        from .processing.atomic import SUBLATTICE_NAMES

        if not (columns.atomic and columns.atomic.get("output") == "columns"):
            raise ValueError(f"{columns.title} is not a table of columns")
        subs = self.sublattices_of(columns)
        taken = {d.title for d in subs}
        name = next((x for x in SUBLATTICE_NAMES if f"Sublattice {x}" not in taken), None)
        if name is None:
            raise ValueError(f"{columns.title} has {len(subs)} sublattices, the most it can have")
        signal = hs.signals.BaseSignal(np.asarray(rows, dtype=np.int32))
        signal.metadata.General.title = f"Sublattice {name}"
        index = max((d.atomic.get("index", 0) for d in subs), default=-1) + 1
        ds = Dataset(id=_new_id(), file_id=columns.file_id, signal=signal, kind=DatasetKind.OTHER,
                     origin="atomic", parent_id=columns.id,
                     atomic={**(info or {}), "output": "sublattice", "index": index, "count": int(len(rows))})
        with self._lock:
            self.datasets[ds.id] = ds
            ids = self.files[columns.file_id].dataset_ids
            ids.insert(ids.index(subs[-1].id if subs else columns.id) + 1, ds.id)
        return ds

    def cage_centres(self, nb: Dataset, guesses, threshold: float, nearest: bool = False) -> dict:
        """New column guesses at the centre of each origin's cage (the
        neighbours nearest the ``guesses``, picked clusters) of neighbourhood
        ``nb``, to fit and add to its Columns (``append_columns``):
        ``columns`` ([y, x] px), ``added``, the Columns' ``id`` and the
        ``name`` of the sublattice they will make."""
        from .processing.atomic import SUBLATTICE_NAMES

        columns = self._columns_of(nb)
        _, o, c = self._cage(nb, guesses, threshold, nearest)
        centres = o + c
        centres = centres[np.isfinite(centres).all(axis=1)]
        if not len(centres):
            raise ValueError("No origin has every picked neighbour within the threshold")
        subs = self.sublattices_of(columns)
        taken = {d.title for d in subs} | ({f"Sublattice {SUBLATTICE_NAMES[0]}"} if not subs else set())
        name = next((x for x in SUBLATTICE_NAMES if f"Sublattice {x}" not in taken), None)
        if name is None:
            raise ValueError("The columns have as many sublattices as they can")
        return {"columns": np.round(centres, 4).tolist(), "added": int(len(centres)), "id": columns.id, "name": name}

    def append_columns(self, columns: Dataset, table: np.ndarray, info: dict | None = None) -> Dataset:
        """Fitted columns (a table of COLUMN_FIELDS) added after a Columns
        dataset's own, which stay as they are with what was made from them.
        The new ones are a sublattice of their own; when the Columns had no
        sublattices, the old ones become one first (A), and neighbourhoods of
        all the old columns are then of that sublattice. Tables with a row
        per column (displacements) get empty rows. Returns the new sublattice."""
        import hyperspy.api as hs

        from .processing.atomic import COLUMN_FIELDS

        if not (columns.atomic and columns.atomic.get("output") == "columns"):
            raise ValueError(f"{columns.title} is not a table of columns")
        old = np.asarray(columns.signal.data, dtype=np.float32)
        n_old, n_new = len(old), len(table)
        subs = self.sublattices_of(columns)
        if not subs:
            import re

            base = self.add_sublattice(columns, np.arange(n_old))
            name = self._short_name(base)
            with self._lock:
                for d in self.datasets.values():
                    if not (d.atomic and d.atomic.get("output") == "neighbourhood"):
                        continue
                    if (d.atomic.get("origins") or d.parent_id) == columns.id:
                        d.atomic = {**d.atomic, "origins": base.id}
                    if d.atomic.get("neighbours") == columns.id:
                        d.atomic = {**d.atomic, "neighbours": base.id}
                    # Its default title follows ("Neighbourhood all → all" → "… A → A").
                    m = re.fullmatch(r"Neighbourhood (\S+) → (\S+)( \d+)?", d.title)
                    if m and not d.custom_title:
                        o, n = (name if x == "all" else x for x in m.group(1, 2))
                        d.signal.metadata.General.title = f"Neighbourhood {o} → {n}{m.group(3) or ''}"
        merged = np.concatenate([old, np.asarray(table, dtype=np.float32)])
        signal = hs.signals.BaseSignal(np.ascontiguousarray(merged))
        signal.metadata.General.title = columns.signal.metadata.General.title
        signal.metadata.set_item("Atomic.fields", list(COLUMN_FIELDS))
        columns.replace_signal(signal)
        columns.atomic = {**columns.atomic, "count": int(len(merged)),
                          "failed": int(np.count_nonzero(merged[:, -1]))}
        # Tables with a row per column stay aligned.
        under = {columns.id}
        with self._lock:
            grew = True
            while grew:
                more = {d.id for d in self.datasets.values() if d.parent_id in under} - under
                grew = bool(more)
                under |= more
            per_column = [self.datasets[i] for i in under if self.datasets[i].atomic
                          and self.datasets[i].atomic.get("output") == "displacements"]
        for d in per_column:
            t = np.asarray(d.signal.data, dtype=np.float32)
            if len(t) == n_old:
                pad = np.full((n_new, t.shape[1]), np.nan, dtype=np.float32)
                s2 = hs.signals.BaseSignal(np.concatenate([t, pad]))
                s2.metadata.General.title = d.signal.metadata.General.title
                d.replace_signal(s2)
        return self.add_sublattice(columns, np.arange(n_old, n_old + n_new), info)

    def set_cage_displacement(self, nb: Dataset, guesses, threshold: float, nearest: bool = False) -> Dataset:
        """Each origin's displacement from the centre of its cage (the mean
        of the neighbours nearest the ``guesses``), as a "Displacement"
        dataset under neighbourhood ``nb``: a table of DISPLACEMENT_FIELDS
        with a row per column of the Columns (NaN for the others, and where
        a neighbour is missing). Made again in place."""
        import hyperspy.api as hs

        columns = self._columns_of(nb)
        rows, _, c = self._cage(nb, guesses, threshold, nearest)
        d = -c
        table = np.full((len(columns.signal.data), 4), np.nan)
        table[rows, :2] = d
        table[rows, 2] = np.hypot(d[:, 0], d[:, 1])
        table[rows, 3] = np.degrees(np.arctan2(d[:, 0], d[:, 1]))
        mag = table[rows, 2]
        ok = np.isfinite(mag)
        signal = hs.signals.BaseSignal(np.ascontiguousarray(table, dtype=np.float32))
        signal.metadata.General.title = "Displacement"
        info = {"output": "displacements", "kind": "cage", "nearest": bool(nearest),
                "cage": [] if nearest else [[float(a), float(b)] for a, b in np.asarray(guesses, dtype=np.float64).reshape(-1, 2)],
                "threshold": float(threshold), "count": int(len(rows)), "missing": int((~ok).sum()),
                "median": float(np.median(mag[ok])) if ok.any() else None,
                "max": float(mag[ok].max()) if ok.any() else None}
        with self._lock:
            earlier = next((x for x in self.datasets.values() if x.parent_id == nb.id and x.atomic
                            and x.atomic.get("output") == "displacements"), None)
        if earlier is not None:
            earlier.replace_signal(signal)
            earlier.atomic = info
            return earlier
        ds = Dataset(id=_new_id(), file_id=nb.file_id, signal=signal, kind=DatasetKind.OTHER,
                     origin="atomic", parent_id=nb.id, atomic=info)
        with self._lock:
            self.datasets[ds.id] = ds
            ids = self.files[nb.file_id].dataset_ids
            under = [i for i in ids if self.datasets[i].parent_id == nb.id]
            ids.insert(ids.index(under[-1] if under else nb.id) + 1, ds.id)
        return ds

    def _cage(self, nb: Dataset, guesses, threshold: float, nearest: bool):
        """Neighbourhood ``nb``'s origins (rows, positions) and the vector to
        each one's cage centre: the neighbours nearest the picked ``guesses``,
        or (``nearest``) all its k nearest neighbours."""
        from .processing.atomic import cage_vectors, nearest_cage

        origins, neighbours = self._nb_sets(nb)
        rows, o = self.column_positions(origins)
        if nearest:
            k = int(nb.atomic.get("k") or 0)
            v = np.asarray(nb.signal.data, dtype=np.float64)
            if not k or len(v) != k * len(o):
                raise ValueError(f"{nb.title} is out of date: make it again")
            return rows, o, nearest_cage(v, k, threshold)
        _, n = self.column_positions(neighbours)
        return rows, o, cage_vectors(n, o, guesses, threshold)

    def _nb_sets(self, nb: Dataset) -> tuple[Dataset, Dataset]:
        try:
            return self.get_dataset(nb.atomic.get("origins") or nb.parent_id), self.get_dataset(nb.atomic["neighbours"])
        except KeyError:
            raise ValueError("The neighbourhood's columns are gone")

    # ---------- masks ----------

    def set_mask(self, image: Dataset, data: np.ndarray, settings: dict, replace: Dataset | None = None) -> Dataset:
        """A mask (``data``, bool) made from ``image``: an image of 0 and 1
        with its calibration under it, titled "Mask" ("Mask 2", …); or
        ``replace`` (a mask of ``image``) made again, keeping its title.
        Returns it."""
        if replace is not None and not (replace.origin == "mask" and replace.parent_id == image.id):
            raise ValueError(f"{replace.title} is not a mask of {image.title}")
        if data.shape != tuple(image.signal.data.shape):
            raise ValueError("The mask's shape is not the image's")
        signal = image.signal._deepcopy_with_new_data(np.asarray(data, dtype=np.float32))
        info = {"source_id": image.id, **settings, "fraction": round(float(np.mean(data)), 6)}
        if replace is not None:
            signal.metadata.General.title = replace.title
            replace.replace_signal(signal)
            replace.mask = info
            return replace
        with self._lock:
            taken = {d.title for d in self.datasets.values() if d.parent_id == image.id}
        title = "Mask" if "Mask" not in taken else next(f"Mask {i}" for i in range(2, 1000) if f"Mask {i}" not in taken)
        signal.metadata.General.title = title
        ds = Dataset(id=_new_id(), file_id=image.file_id, signal=signal, kind=DatasetKind.IMAGE,
                     origin="mask", parent_id=image.id, mask=info)
        with self._lock:
            self.datasets[ds.id] = ds
            ids = self.files[image.file_id].dataset_ids
            # After the image's other masks, else right after the image.
            after = [i for i in ids if self.datasets[i].parent_id == image.id and self.datasets[i].origin == "mask"]
            ids.insert(ids.index(after[-1] if after else image.id) + 1, ds.id)
        return ds

    # ---------- 4D-STEM ----------

    def add_virtual_image(self, cube: Dataset, data: np.ndarray, detector: dict, response: str, title: str,
                          made: dict | None = None) -> Dataset:
        """A virtual image of 4D-STEM data (``data``: [y, x]) under it, with
        the scan's calibration and the detector it was made with (and
        ``made``: e.g. a digital dark field's source and weight); titled
        ``title`` ("… 2", … if taken). Returns it."""
        import hyperspy.api as hs

        if cube.kind is not DatasetKind.DIFFRACTION4D:
            raise ValueError(f"{cube.title} is not 4D-STEM data")
        if data.shape != tuple(cube.signal.data.shape[:2]):
            raise ValueError("The image's shape is not the scan's")
        signal = hs.signals.Signal2D(np.asarray(data, dtype=np.float32))
        for axis, src in zip(signal.axes_manager.signal_axes, cube.signal.axes_manager.navigation_axes):
            axis.name, axis.scale, axis.offset, axis.units = src.name, src.scale, src.offset, src.units
        signal.metadata.set_item("Signal.quantity", "Intensity")
        with self._lock:
            taken = {d.title for d in self.datasets.values() if d.parent_id == cube.id}
        if title in taken:
            title = next(f"{title} {i}" for i in range(2, 10000) if f"{title} {i}" not in taken)
        signal.metadata.General.title = title
        ds = Dataset(
            id=_new_id(), file_id=cube.file_id, signal=signal, kind=DatasetKind.IMAGE, origin="virtual",
            parent_id=cube.id, virtual={"source_id": cube.id, "detector": detector, "response": response, **(made or {})},
        )
        with self._lock:
            self.datasets[ds.id] = ds
            ids = self.files[cube.file_id].dataset_ids
            after = [i for i in ids if self.datasets[i].parent_id == cube.id]
            ids.insert(ids.index(after[-1] if after else cube.id) + 1, ds.id)
        return ds

    def set_bragg_vectors(self, cube: Dataset, vectors, count: int, info: dict, bvm: np.ndarray, upsample: int) -> Dataset:
        """The Bragg disks found in 4D-STEM data: "Bragg vectors" under it
        (pyxem ``DiffractionVectors``, ``processing.disks.to_vectors``; kind
        "other", origin "disks"; ``count`` disks) and its "Bragg vector map"
        image under that (origin "bragg-map", ``upsample`` × finer than the
        pattern). Made again in place. Returns the vectors."""
        import hyperspy.api as hs

        if cube.kind is not DatasetKind.DIFFRACTION4D:
            raise ValueError(f"{cube.title} is not 4D-STEM data")
        info = {**info, "source_id": cube.id, "count": int(count)}
        table_signal = vectors
        table_signal.metadata.General.title = "Bragg vectors"
        map_signal = hs.signals.Signal2D(np.asarray(bvm, dtype=np.float32))
        map_signal.metadata.General.title = "Bragg vector map"
        map_signal.metadata.set_item("Signal.quantity", "Disks")
        for axis, src in zip(map_signal.axes_manager.signal_axes, cube.signal.axes_manager.signal_axes):
            axis.name, axis.units = src.name, src.units
            axis.scale = src.scale / upsample
            axis.offset = src.offset - src.scale / 2 + src.scale / (2 * upsample)
        with self._lock:
            vectors = next((d for d in self.datasets.values() if d.origin == "disks" and d.parent_id == cube.id), None)
            bvm_ds = vectors and next(
                (d for d in self.datasets.values() if d.origin == "bragg-map" and d.parent_id == vectors.id), None
            )
        if vectors is not None:
            vectors.replace_signal(table_signal)
            vectors.disks = info
        else:
            vectors = Dataset(id=_new_id(), file_id=cube.file_id, signal=table_signal, kind=DatasetKind.OTHER,
                              origin="disks", parent_id=cube.id, disks=info)
            with self._lock:
                self.datasets[vectors.id] = vectors
                ids = self.files[cube.file_id].dataset_ids
                after = [i for i in ids if self.datasets[i].parent_id == cube.id]
                ids.insert(ids.index(after[-1] if after else cube.id) + 1, vectors.id)
        if bvm_ds is not None:
            bvm_ds.replace_signal(map_signal)
            bvm_ds.disks = info
        else:
            bvm_ds = Dataset(id=_new_id(), file_id=cube.file_id, signal=map_signal, kind=DatasetKind.IMAGE,
                             origin="bragg-map", parent_id=vectors.id, disks=info)
            with self._lock:
                self.datasets[bvm_ds.id] = bvm_ds
                ids = self.files[cube.file_id].dataset_ids
                ids.insert(ids.index(vectors.id) + 1, bvm_ds.id)
        return vectors

    def set_beam_shifts(self, cube: Dataset, shifts: np.ndarray, info: dict) -> list[Dataset]:
        """The direct beam found in 4D-STEM data: pyxem's shifts ([y, x, 2])
        as "Beam shift x" and "Beam shift y" images under it (origin
        "beam-shift", px, on the scan). Made again in place. Returns them."""
        import hyperspy.api as hs

        if cube.kind is not DatasetKind.DIFFRACTION4D:
            raise ValueError(f"{cube.title} is not 4D-STEM data")
        out = []
        for k, axis in enumerate(("x", "y")):
            signal = hs.signals.Signal2D(np.ascontiguousarray(shifts[..., k], dtype=np.float32))
            for a, src in zip(signal.axes_manager.signal_axes, cube.signal.axes_manager.navigation_axes):
                a.name, a.scale, a.offset, a.units = src.name, src.scale, src.offset, src.units
            signal.metadata.General.title = f"Beam shift {axis}"
            signal.metadata.set_item("Signal.quantity", "Shift (px)")
            beam = {**info, "source_id": cube.id, "axis": axis}
            with self._lock:
                earlier = next((d for d in self.datasets.values() if d.origin == "beam-shift"
                                and d.parent_id == cube.id and d.beam and d.beam.get("axis") == axis), None)
            if earlier is not None:
                earlier.replace_signal(signal)
                earlier.beam = beam
                out.append(earlier)
                continue
            ds = Dataset(id=_new_id(), file_id=cube.file_id, signal=signal, kind=DatasetKind.IMAGE,
                         origin="beam-shift", parent_id=cube.id, beam=beam)
            with self._lock:
                self.datasets[ds.id] = ds
                ids = self.files[cube.file_id].dataset_ids
                after = [i for i in ids if self.datasets[i].parent_id == cube.id]
                ids.insert(ids.index(after[-1] if after else cube.id) + 1, ds.id)
            out.append(ds)
        return out

    def beam_shifts_of(self, cube: Dataset) -> np.ndarray | None:
        """The shifts found for 4D-STEM data ([y, x, 2]), if its beam was found."""
        with self._lock:
            found = {d.beam["axis"]: d for d in self.datasets.values()
                     if d.origin == "beam-shift" and d.parent_id == cube.id and d.beam}
        if set(found) != {"x", "y"}:
            return None
        return np.stack([np.asarray(found["x"].signal.data), np.asarray(found["y"].signal.data)], axis=-1)

    def centre_patterns(self, cube: Dataset, shifts: np.ndarray, in_place: bool = True, report=None) -> Dataset:
        """Shift every pattern of 4D-STEM data by ``shifts`` so the beam is at
        the centre (``processing.centring.centre``): in place (a new revision;
        the file is unchanged), or as a copy under it (origin "centred").
        Returns the dataset centred. ValueError for data too big to hold."""
        from .processing.centring import centre

        if cube.kind is not DatasetKind.DIFFRACTION4D:
            raise ValueError(f"{cube.title} is not 4D-STEM data")
        if cube.lazy:
            if cube.signal.data.nbytes * 4 > DIFFRACTION4D_IN_MEMORY_LIMIT:
                raise ValueError(f"{cube.title} is too big to centre in memory: bin it first")
            cube.materialise()
        if shifts.shape[:2] != tuple(cube.signal.data.shape[:2]):
            raise ValueError("The shifts are not on this scan (was it binned since?)")
        centred = centre(cube.signal, shifts, report)
        if not in_place:
            return self.copy_as(cube, "centred", f"{cube.title} (centred)", signal=centred)
        centred.metadata.General.title = cube.signal.metadata.General.title
        cube.replace_signal(centred)
        return cube

    def calibrate_patterns(self, cube: Dataset, scale: float, units: str, ring: dict | None, info: dict) -> Dataset:
        """Calibrate 4D-STEM data's pattern axes: ``scale`` ``units`` per
        pixel, and 0 at ``ring``'s centre if given (pattern px, pixel i
        spanning [i, i + 1)), else the axes' offsets kept in the new units.
        The data isn't changed; the calibration is kept in ``calibration``."""
        if cube.kind is not DatasetKind.DIFFRACTION4D:
            raise ValueError(f"{cube.title} is not 4D-STEM data")
        if not (np.isfinite(scale) and scale > 0):
            raise ValueError("The pixel size must be a positive number")
        axes = cube.signal.axes_manager.signal_axes  # [kx, ky]
        centre = (ring["cx"], ring["cy"]) if ring else None
        zeros = [
            (centre[k] - 0.5) if centre else (-axis.offset / axis.scale if axis.scale else 0.0)
            for k, axis in enumerate(axes)
        ]
        # What was calibrated before (the origin, the switches) is kept.
        old = cube.calibration or {}
        enabled = {**old.get("enabled", {}), "scale": True, **({"ellipse": True} if ring else {})}
        cube.calibration = {**old, **info, "scale": float(scale), "units": units, "ring": ring, "zero": zeros,
                            "enabled": enabled}
        self.apply_calibration_axes(cube)
        self.remake_bragg_map(cube)
        return cube

    # ---------- the calibration of 4D-STEM data (py4DSTEM-style) ----------

    def origin_maps_of(self, cube: Dataset) -> dict[str, np.ndarray]:
        """The origin maps measured for 4D-STEM data: {"measured": [y, x, 2],
        "fitted": [y, x, 2]} (x, y pattern px; "fitted" when fitted)."""
        with self._lock:
            maps = [d for d in self.datasets.values()
                    if d.origin == "origin-map" and d.origin_map and d.origin_map.get("source_id") == cube.id]
        out = {}
        for kind in ("measured", "fitted"):
            parts = {d.origin_map["axis"]: d for d in maps if d.origin_map["kind"] == kind}
            if set(parts) == {"x", "y"}:
                out[kind] = np.stack([np.asarray(parts["x"].signal.data), np.asarray(parts["y"].signal.data)], axis=-1)
        return out

    def set_origin(self, cube: Dataset, measured: np.ndarray, fitted: np.ndarray | None, info: dict,
                   parent: Dataset | None = None) -> list[Dataset]:
        """The centring of 4D-STEM data: each scan position's origin (the
        direct beam: [y, x, 2] of x, y pattern px), as measured and as fitted
        (a descan plane; None: as measured), kept as "Origin x / y" and
        "Origin x / y (fit)" images (origin "origin-map") under ``parent``
        (the Bragg vector map when measured from the disks; else the data)
        and in its calibration (switched on). Made again in place (moved if
        the parent changed); the Bragg vector map is made again."""
        import hyperspy.api as hs

        if cube.kind is not DatasetKind.DIFFRACTION4D:
            raise ValueError(f"{cube.title} is not 4D-STEM data")
        parent = parent or cube
        wanted = [("measured", measured)] + ([("fitted", fitted)] if fitted is not None else [])
        with self._lock:
            old = [d for d in self.datasets.values()
                   if d.origin == "origin-map" and d.origin_map and d.origin_map.get("source_id") == cube.id]
        # Maps under another parent (measured from elsewhere before) go: made again under this one.
        for d in [d for d in old if d.parent_id != parent.id]:
            self._remove_tree(d)
        old = [d for d in old if d.parent_id == parent.id]
        out = []
        for kind, data in wanted:
            for k, axis in enumerate(("x", "y")):
                title = f"Origin {axis}" + (" (fit)" if kind == "fitted" else "")
                signal = hs.signals.Signal2D(np.ascontiguousarray(data[..., k], dtype=np.float32))
                for a, src in zip(signal.axes_manager.signal_axes, cube.signal.axes_manager.navigation_axes):
                    a.name, a.scale, a.offset, a.units = src.name, src.scale, src.offset, src.units
                signal.metadata.General.title = title
                signal.metadata.set_item("Signal.quantity", "Origin (px)")
                meta = {"axis": axis, "kind": kind, "source_id": cube.id}
                earlier = next((d for d in old if d.origin_map and d.origin_map.get("axis") == axis
                                and d.origin_map.get("kind") == kind), None)
                if earlier is not None:
                    earlier.replace_signal(signal)
                    earlier.origin_map = meta
                    out.append(earlier)
                    continue
                ds = Dataset(id=_new_id(), file_id=cube.file_id, signal=signal, kind=DatasetKind.IMAGE,
                             origin="origin-map", parent_id=parent.id, origin_map=meta)
                with self._lock:
                    self.datasets[ds.id] = ds
                    ids = self.files[cube.file_id].dataset_ids
                    after = [i for i in ids if self.datasets[i].parent_id == parent.id]
                    ids.insert(ids.index(after[-1] if after else parent.id) + 1, ds.id)
                out.append(ds)
        # Fitted maps no longer wanted (fitted "none" this time) go.
        if fitted is None:
            for d in old:
                if d.origin_map and d.origin_map.get("kind") == "fitted":
                    self._remove_tree(d)
        cal = cube.calibration or {}
        cube.calibration = {**cal, "origin": info, "enabled": {**cal.get("enabled", {}), "origin": True}}
        self.remake_bragg_map(cube)
        return out

    def set_calibration_switches(self, cube: Dataset, enabled: dict | None = None, view: str | None = None) -> None:
        """Switch parts of 4D-STEM data's calibration on or off for its
        Bragg vectors ("origin", "ellipse", "scale"), and whether its Bragg
        vector map is shown "raw" or "calibrated"; the map is made again."""
        cal = dict(cube.calibration or {})
        if enabled:
            cal["enabled"] = {**cal.get("enabled", {}), **{k: bool(v) for k, v in enabled.items()}}
        if view:
            cal["bvm"] = view
        cube.calibration = cal
        self.remake_bragg_map(cube)

    def vectors_table(self, cube: Dataset) -> np.ndarray | None:
        """4D-STEM data's Bragg vectors as a table (``processing.disks``'
        rows of scan y, x, ky, kx, score in pixels), kept until they change;
        None if its disks haven't been found."""
        from .processing import disks

        vectors = self.bragg_vectors_of(cube)
        if vectors is None:
            return None
        key = ("table", vectors.revision)
        table = vectors.cache.get(key)
        if table is None:
            table = disks.table_of(vectors.signal)
            vectors.cache.clear()
            vectors.cache[key] = table
        return table

    def centring_origins(self, cube: Dataset) -> np.ndarray | None:
        """Each scan position's origin ([y, x, 2] of x, y pattern px; fitted
        if it was) when the centring is switched on (Info tab), else None."""
        if not (cube.calibration or {}).get("enabled", {}).get("origin"):
            return None
        maps = self.origin_maps_of(cube)
        origins = maps.get("fitted", maps.get("measured"))
        if origins is None or origins.shape[:2] != tuple(cube.signal.data.shape[:2]):
            return None
        return origins

    def remake_bragg_map(self, cube: Dataset) -> Dataset | None:
        """Make 4D-STEM data's Bragg vector map again from its disks and its
        calibration (the parts switched on, or raw when its view is "raw").
        None if its disks haven't been found."""
        from .processing import disks

        table = self.vectors_table(cube)
        if table is None:
            return None
        cal = cube.calibration or {}
        on = cal.get("enabled", {})
        calibrated = cal.get("bvm", "calibrated") == "calibrated"
        origins = self.centring_origins(cube) if calibrated else None
        ring = cal.get("ring") if calibrated and on.get("ellipse") else None
        scale = cal.get("scale") if calibrated and on.get("scale") else None
        shape = tuple(cube.signal.data.shape[2:])
        centre = (ring["cx"], ring["cy"]) if ring else (cal["ring"]["cx"], cal["ring"]["cy"]) if cal.get("ring") else None
        data, step, offset = disks.map_of(table, shape, origins, centre, ring, scale)
        raw = origins is None and ring is None and scale is None
        units = cal.get("units", "px") if scale else "px"
        info = {"view": "raw" if raw else "calibrated", "origin": origins is not None, "ellipse": ring is not None,
                "scale": scale is not None, "units": units}
        return self.set_bragg_map(cube, data, step, offset, units, info)

    @staticmethod
    def apply_calibration_axes(ds: Dataset) -> None:
        """Set 4D-STEM data's pattern axes from its ``calibration`` (after
        calibrating, or when an analysis is opened from the raw file)."""
        cal = ds.calibration
        if not cal or not cal.get("scale") or ds.kind is not DatasetKind.DIFFRACTION4D:
            return
        for axis, zero in zip(ds.signal.axes_manager.signal_axes, cal.get("zero") or [0.0, 0.0]):
            axis.scale, axis.units = cal["scale"], cal["units"]
            axis.offset = -cal["scale"] * zero

    def set_bragg_map(self, cube: Dataset, data: np.ndarray, scale: float, offset: float, units: str, info: dict) -> Dataset:
        """Make the Bragg vector map of 4D-STEM data again (its disks must
        have been found): ``data`` [ky, kx] with both axes at ``scale``
        ``units`` per pixel from ``offset``; ``info`` (how it was calibrated)
        kept as its ``disks["map"]``. Returns it."""
        import hyperspy.api as hs

        vectors = self.bragg_vectors_of(cube)
        if vectors is None:
            raise ValueError(f"The disks of {cube.title} haven't been found")
        with self._lock:
            bvm = next((d for d in self.datasets.values() if d.origin == "bragg-map" and d.parent_id == vectors.id), None)
        signal = hs.signals.Signal2D(np.asarray(data, dtype=np.float32))
        signal.metadata.General.title = "Bragg vector map"
        signal.metadata.set_item("Signal.quantity", "Disks")
        for axis, src in zip(signal.axes_manager.signal_axes, cube.signal.axes_manager.signal_axes):
            axis.name, axis.units, axis.scale, axis.offset = src.name, units, scale, offset
        disks = {**(vectors.disks or {}), "map": info}
        if bvm is None:
            bvm = Dataset(id=_new_id(), file_id=cube.file_id, signal=signal, kind=DatasetKind.IMAGE,
                          origin="bragg-map", parent_id=vectors.id, disks=disks)
            with self._lock:
                self.datasets[bvm.id] = bvm
                ids = self.files[cube.file_id].dataset_ids
                ids.insert(ids.index(vectors.id) + 1, bvm.id)
        else:
            bvm.replace_signal(signal)
            bvm.disks = disks
        return bvm

    def bragg_vectors_of(self, cube: Dataset) -> Dataset | None:
        """The Bragg vectors table of 4D-STEM data, if its disks were found."""
        with self._lock:
            return next((d for d in self.datasets.values() if d.origin == "disks" and d.parent_id == cube.id), None)

    def bragg_map_of(self, cube: Dataset) -> Dataset | None:
        """The Bragg vector map of 4D-STEM data, if its disks were found."""
        vectors = self.bragg_vectors_of(cube)
        with self._lock:
            return vectors and next(
                (d for d in self.datasets.values() if d.origin == "bragg-map" and d.parent_id == vectors.id), None
            )

    def subtract_background(self, cube: Dataset, data: np.ndarray, how: str, in_place: bool = True) -> Dataset:
        """4D-STEM data with its patterns' background subtracted (``data``:
        float32, the data's shape; ``how``: e.g. "median kernel 19 px"): in
        place (a new revision; the file is unchanged), or as a copy under it
        (origin "pattern-background"; "background" is EELS'). Returns the dataset changed."""
        if cube.kind is not DatasetKind.DIFFRACTION4D:
            raise ValueError(f"{cube.title} is not 4D-STEM data")
        if data.shape != tuple(cube.signal.data.shape):
            raise ValueError("The patterns are not the data's (was it binned since?)")
        signal = cube.signal._deepcopy_with_new_data(data)
        signal.metadata.set_item("Signal.background", how)
        if not in_place:
            return self.copy_as(cube, "pattern-background", f"{cube.title} (background subtracted)", signal=signal)
        signal.metadata.General.title = cube.signal.metadata.General.title
        cube.replace_signal(signal)
        return cube

    def repair_pixels(self, cube: Dataset, mask: np.ndarray, in_place: bool = True) -> Dataset:
        """Replace the detector pixels of ``mask`` in every pattern of 4D-STEM
        data by the mean of the 8 around them: in place (a new revision; the
        file is unchanged), or as a copy under it (origin "repaired"). Returns
        the dataset repaired. ValueError for data too big to hold in memory."""
        from .processing.fourd import repair

        if cube.kind is not DatasetKind.DIFFRACTION4D:
            raise ValueError(f"{cube.title} is not 4D-STEM data")
        if cube.lazy:
            if cube.signal.data.nbytes > DIFFRACTION4D_IN_MEMORY_LIMIT:
                raise ValueError(f"{cube.title} is too big to repair in memory: bin it first")
            cube.materialise()
        target = cube if in_place else self.copy_as(cube, "repaired", f"{cube.title} (pixels repaired)")
        target.replace_signal(repair(target.signal, mask))
        return target

    # ---------- neighbourhoods and bonds (M22e) ----------

    def column_positions(self, ds: Dataset) -> tuple[np.ndarray, np.ndarray]:
        """The fitted positions ([y, x] px) of a Columns dataset's columns or a
        sublattice's, with their rows in the Columns table; columns without a
        position are left out."""
        out = ds.atomic.get("output") if ds.atomic else None
        if out == "columns":
            table = np.asarray(ds.signal.data, dtype=np.float64)
            rows = np.arange(len(table))
        elif out == "sublattice":
            table = np.asarray(self.get_dataset(ds.parent_id).signal.data, dtype=np.float64)
            rows = np.asarray(ds.signal.data, dtype=np.int64).ravel()
            rows = rows[(rows >= 0) & (rows < len(table))]
        else:
            raise ValueError(f"{ds.title} is not a set of columns")
        yx = table[rows, :2]
        ok = np.isfinite(yx).all(axis=1)
        return rows[ok], yx[ok]

    def _short_name(self, ds: Dataset) -> str:
        if ds.atomic and ds.atomic.get("output") == "columns":
            return "all"
        return ds.title.removeprefix("Sublattice ").strip() or ds.title

    def _drop_neighbourhoods_of(self, gone: set[str]) -> None:
        """Neighbourhoods whose origins or neighbours were in ``gone`` go too."""
        with self._lock:
            stale = [d for d in self.datasets.values() if d.atomic and d.atomic.get("output") == "neighbourhood"
                     and (d.atomic.get("neighbours") in gone or d.atomic.get("origins") in gone)]
        for d in stale:
            if d.id in self.datasets:
                self._remove_tree(d)

    def _columns_of(self, ds: Dataset) -> Dataset:
        """The Columns dataset of Columns or a sublattice."""
        return ds if ds.atomic and ds.atomic.get("output") == "columns" else self.get_dataset(ds.parent_id)

    def make_neighbourhood(self, origins: Dataset, neighbours: Dataset, k: int, replace: Dataset | None = None) -> Dataset:
        """The vectors from each column of ``origins`` (Columns or a sublattice)
        to its ``k`` nearest in ``neighbours`` (kemstem's ``get_neighborhood``),
        as a "Neighbourhood" dataset under the Columns (it may join two
        sublattices; ``atomic["origins"]`` and ``["neighbours"]`` name them):
        (k × m, 2) [y, x] px. A new one each time (several can be kept, with
        any settings), or ``replace`` made again with these settings: in
        place (its name kept; its clusters and bonds go)."""
        import hyperspy.api as hs

        from .processing.atomic import neighbourhood

        if replace is not None and not (replace.atomic and replace.atomic.get("output") == "neighbourhood"):
            raise ValueError(f"{replace.title} is not a neighbourhood")
        columns = self._columns_of(origins)
        if self._columns_of(neighbours).id != columns.id:
            raise ValueError("The origins and neighbours must be columns of one image")
        _, o = self.column_positions(origins)
        _, n = self.column_positions(neighbours)
        vectors = neighbourhood(n, o, k)
        signal = hs.signals.BaseSignal(np.ascontiguousarray(vectors, dtype=np.float32))
        info = {"output": "neighbourhood", "origins": origins.id, "neighbours": neighbours.id, "k": int(k),
                "count": int(len(o))}
        if replace is not None:
            for child in [d for d in self.siblings(replace) if d.parent_id == replace.id]:
                self._remove_tree(child)
            signal.metadata.General.title = replace.title
            replace.replace_signal(signal)
            replace.atomic = info
            return replace
        # A name of its own: "Neighbourhood A → all", "… 2" …
        base = f"Neighbourhood {self._short_name(origins)} → {self._short_name(neighbours)}"
        with self._lock:
            taken = {d.title for d in self.datasets.values() if d.parent_id == columns.id}
        title = base if base not in taken else next(f"{base} {i}" for i in range(2, 1000) if f"{base} {i}" not in taken)
        signal.metadata.General.title = title
        ds = Dataset(id=_new_id(), file_id=columns.file_id, signal=signal, kind=DatasetKind.OTHER,
                     origin="atomic", parent_id=columns.id, atomic=info)
        with self._lock:
            self.datasets[ds.id] = ds
            ids = self.files[columns.file_id].dataset_ids
            # After the Columns' other results (their sublattices and earlier neighbourhoods).
            tree, todo = {columns.id}, [columns.id]
            while todo:
                cur = todo.pop()
                for x in ids:
                    if x in self.datasets and self.datasets[x].parent_id == cur and x not in tree:
                        tree.add(x)
                        todo.append(x)
            ids.insert(max(i for i, x in enumerate(ids) if x in tree) + 1, ds.id)
        return ds

    def cluster_neighbourhood(self, nb: Dataset, n: int) -> Dataset:
        """Cluster a neighbourhood's vectors (kemstem's ``cluster_neighbors``):
        the centres go into its ``atomic["clusters"]`` ([y, x] px)."""
        from .processing.atomic import cluster_neighbourhood

        centres = cluster_neighbourhood(np.asarray(nb.signal.data, dtype=np.float64), n)
        nb.atomic = {**nb.atomic, "clusters": [[round(float(a), 4), round(float(b), 4)] for a, b in centres]}
        return nb

    def make_bonds(self, nb: Dataset, name: str, guess: tuple[float, float], threshold: float) -> Dataset:
        """From each origin of neighbourhood ``nb``, the vector to the
        neighbour nearest ``origin + guess`` (kemstem's
        ``get_vector_to_neighbors``), with its length and angle, as a
        "Bonds" dataset under it (a table of BOND_FIELDS). Made again in
        place for the same name."""
        import hyperspy.api as hs

        from .processing.atomic import BOND_FIELDS, bonds

        name = name.strip()
        if not name:
            raise ValueError("The bonds need a name")
        try:
            origins = self.get_dataset(nb.atomic.get("origins") or nb.parent_id)
            neighbours = self.get_dataset(nb.atomic["neighbours"])
        except KeyError:
            raise ValueError("The neighbourhood's columns are gone")
        rows, o = self.column_positions(origins)
        _, n = self.column_positions(neighbours)
        table = bonds(n, o, rows, guess, threshold)
        signal = hs.signals.BaseSignal(np.ascontiguousarray(table, dtype=np.float32))
        signal.metadata.General.title = f"Bonds {name}"
        missing = int(np.count_nonzero(~np.isfinite(table[:, 3])))
        length = table[:, 3]
        info = {"output": "bonds", "name": name, "vector": [float(guess[0]), float(guess[1])],
                "threshold": float(threshold), "count": int(len(table)), "missing": missing,
                "mean_length": float(np.nanmean(length)) if missing < len(table) else None,
                "fields": list(BOND_FIELDS)}
        with self._lock:
            earlier = next((d for d in self.datasets.values() if d.parent_id == nb.id and d.atomic
                            and d.atomic.get("output") == "bonds" and d.atomic.get("name") == name), None)
        if earlier is not None:
            earlier.replace_signal(signal)
            earlier.atomic = info
            return earlier
        ds = Dataset(id=_new_id(), file_id=nb.file_id, signal=signal, kind=DatasetKind.OTHER,
                     origin="atomic", parent_id=nb.id, atomic=info)
        with self._lock:
            self.datasets[ds.id] = ds
            ids = self.files[nb.file_id].dataset_ids
            later = [i for i, x in enumerate(ids) if x in self.datasets and self.datasets[x].parent_id == nb.id]
            ids.insert((max(later) if later else ids.index(nb.id)) + 1, ds.id)
        return ds

    # ---------- periodic lattice displacements (M22f) ----------

    def set_pld_images(self, image: Dataset, made: dict, settings: dict) -> list[Dataset]:
        """The PLD reference (the image with its superlattice peaks damped) and
        difference (reference − image) under ``image``: images with its
        calibration, made again in place. Returns them."""
        out = []
        for output, title, data in (("pld-reference", "PLD reference", made["reference"]),
                                    ("pld-difference", "PLD difference", made["difference"])):
            signal = image.signal._deepcopy_with_new_data(np.asarray(data, dtype=np.float32))
            signal.metadata.General.title = title
            info = {**settings, "output": output}
            with self._lock:
                earlier = next((d for d in self.datasets.values() if d.parent_id == image.id and d.atomic
                                and d.atomic.get("output") == output), None)
            if earlier is not None:
                earlier.replace_signal(signal)
                earlier.atomic = info
                out.append(earlier)
                continue
            ds = Dataset(id=_new_id(), file_id=image.file_id, signal=signal, kind=DatasetKind.IMAGE,
                         origin="atomic", parent_id=image.id, atomic=info)
            with self._lock:
                self.datasets[ds.id] = ds
                ids = self.files[image.file_id].dataset_ids
                at = ids.index(out[-1].id) if out else ids.index(image.id)
                ids.insert(at + 1, ds.id)
            out.append(ds)
        return out

    def pld_reference_of(self, image: Dataset) -> Dataset | None:
        with self._lock:
            return next((d for d in self.datasets.values() if d.parent_id == image.id and d.atomic
                         and d.atomic.get("output") == "pld-reference"), None)

    def set_displacements(self, columns: Dataset, table: np.ndarray, settings: dict) -> Dataset:
        """Each column's periodic lattice displacement (a table of
        DISPLACEMENT_FIELDS, a row per column) as "Displacements" under the
        Columns. Made again in place. Returns it."""
        import hyperspy.api as hs

        signal = hs.signals.BaseSignal(np.ascontiguousarray(table, dtype=np.float32))
        signal.metadata.General.title = "Displacements"
        mag = table[:, 2]
        ok = np.isfinite(mag)
        info = {**settings, "output": "displacements", "count": int(len(table)), "missing": int((~ok).sum()),
                "median": float(np.median(mag[ok])) if ok.any() else None,
                "max": float(mag[ok].max()) if ok.any() else None}
        with self._lock:
            earlier = next((d for d in self.datasets.values() if d.parent_id == columns.id and d.atomic
                            and d.atomic.get("output") == "displacements"), None)
        if earlier is not None:
            earlier.replace_signal(signal)
            earlier.atomic = info
            return earlier
        ds = Dataset(id=_new_id(), file_id=columns.file_id, signal=signal, kind=DatasetKind.OTHER,
                     origin="atomic", parent_id=columns.id, atomic=info)
        with self._lock:
            self.datasets[ds.id] = ds
            ids = self.files[columns.file_id].dataset_ids
            ids.insert(ids.index(columns.id) + 1, ds.id)
        return ds

    def make_dpc(
        self, inputs: dict[str, str], normalise: bool = True, theta: float = 0.0, mirror: bool = False,
    ) -> list[Dataset]:
        """Two objects next to the inputs: "DPC", the direction of the
        deflection (degrees, from A − C and B − D rotated into the image's
        frame, their means removed; shown in a circular colour map,
        brightness its magnitude) with
        "DPC x" and "DPC y" under it, and "iDPC", their integral. Made again
        in place if made from these inputs before. Returns them (DPC, x, y,
        iDPC). From four stacks: stacks, each frame from the same frames."""
        from .processing.dpc import differences, integrate, rotate

        q = self.dpc_quadrants(inputs)
        a = q["A"]
        gx, gy = rotate(*differences(*self._dpc_arrays(q), normalise=normalise), theta, mirror)
        # The mean deflection (segment gains, descan) is not the specimen's.
        gx = gx - gx.mean(axis=(-2, -1), keepdims=True)
        gy = gy - gy.mean(axis=(-2, -1), keepdims=True)
        phase = integrate(gx, gy)
        angle = np.degrees(np.arctan2(gy, gx))
        settings = {
            "inputs": {k: q[k].id for k in "ABCD"}, "normalise": bool(normalise), "theta": float(theta),
            "mirror": bool(mirror),
        }
        with self._lock:
            earlier = {
                d.dpc["output"]: d for d in self.datasets.values()
                if d.origin == "dpc" and d.dpc and d.dpc.get("inputs") == settings["inputs"]
            }
        made: list[Dataset] = []
        colour_id = None
        for output, title, data in (
            ("colour", "DPC", angle), ("x", "DPC x", gx), ("y", "DPC y", gy), ("idpc", "iDPC", phase),
        ):
            signal = a.signal._deepcopy_with_new_data(np.ascontiguousarray(data, dtype=np.float32))
            signal.metadata.General.title = title
            for item in ("Acquisition_instrument.TEM.Detector", "General.original_title"):
                if signal.metadata.has_item(item):  # not a segment image
                    parent, name = item.rsplit(".", 1)
                    delattr(signal.metadata.get_item(parent), name)
            # x and y under the DPC object; DPC and iDPC objects of their own.
            parent = colour_id if output in ("x", "y") else None
            ds = earlier.get(output)
            if ds is not None:
                ds.replace_signal(signal)
                ds.dpc = {**settings, "output": output}
                ds.parent_id, ds.derived_from = parent, None
            else:
                ds = Dataset(
                    id=_new_id(), file_id=a.file_id, signal=signal, kind=a.kind, origin="dpc",
                    parent_id=parent, dpc={**settings, "output": output},
                )
                with self._lock:
                    self.datasets[ds.id] = ds
                    ids = self.files[a.file_id].dataset_ids
                    # After the inputs in this file, then each after the one before.
                    prev = [m.id for m in made if m.id in ids]
                    at = ids.index(prev[-1]) if prev else max(ids.index(x.id) for x in q.values() if x.id in ids)
                    ids.insert(at + 1, ds.id)
            if output == "colour":
                colour_id = ds.id
            made.append(ds)
        return made

    def make_com_dpc(
        self, cube: Dataset, gx: np.ndarray, gy: np.ndarray, theta: float, mirror: bool, settings: dict,
    ) -> list[Dataset]:
        """DPC from 4D-STEM data's centre of mass (``processing.com``): the
        deflection ``gx``, ``gy`` ([y, x], pattern px) rotated by ``theta``
        (degrees; ``mirror``: y flipped first) into the scan's frame, as a
        "DPC (COM)" object under the data (its direction, shown in a circular
        colour map) with "COM x" and "COM y" under it, and "iCOM" (their
        integral) under the data, all on the scan's axes and in the pattern's
        units. Made again in place. ``settings``: how (kept in ``dpc``, with
        "source": "4d"). Returns them (DPC, x, y, iCOM)."""
        import hyperspy.api as hs

        from .processing.dpc import integrate, rotate

        if cube.kind is not DatasetKind.DIFFRACTION4D:
            raise ValueError(f"{cube.title} is not 4D-STEM data")
        k_axis = cube.signal.axes_manager.signal_axes[0]
        scale, units = float(k_axis.scale or 1.0), str(k_axis.units or "px")
        rx, ry = rotate(gx * scale, gy * scale, theta, mirror)
        rx, ry = rx - rx.mean(), ry - ry.mean()
        phase = integrate(rx, ry, pad=True)
        dpc = {**settings, "source": "4d", "cube": cube.id, "theta": float(theta), "mirror": bool(mirror)}
        with self._lock:
            earlier = {
                d.dpc["output"]: d for d in self.datasets.values()
                if d.origin == "dpc" and d.dpc and d.dpc.get("cube") == cube.id
            }
        made: list[Dataset] = []
        colour_id = None
        for output, title, data, quantity in (
            ("colour", "DPC (COM)", np.degrees(np.arctan2(ry, rx)), "Direction (°)"),
            # The centre of mass's shift (not segment differences): COM x / y.
            ("x", "COM x", rx, f"Centre-of-mass shift ({units})"),
            ("y", "COM y", ry, f"Centre-of-mass shift ({units})"),
            ("idpc", "iCOM", phase, f"Integrated deflection ({units} px)"),
        ):
            signal = hs.signals.Signal2D(np.ascontiguousarray(data, dtype=np.float32))
            for axis, src in zip(signal.axes_manager.signal_axes, cube.signal.axes_manager.navigation_axes):
                axis.name, axis.scale, axis.offset, axis.units = src.name, src.scale, src.offset, src.units
            signal.metadata.General.title = title
            signal.metadata.set_item("Signal.quantity", quantity)
            parent = colour_id if output in ("x", "y") else cube.id
            ds = earlier.get(output)
            if ds is not None:
                ds.replace_signal(signal)
                ds.dpc = {**dpc, "output": output}
                ds.parent_id = parent
            else:
                ds = Dataset(
                    id=_new_id(), file_id=cube.file_id, signal=signal, kind=DatasetKind.IMAGE, origin="dpc",
                    parent_id=parent, dpc={**dpc, "output": output},
                )
                with self._lock:
                    self.datasets[ds.id] = ds
                    ids = self.files[cube.file_id].dataset_ids
                    # Under the data (after what is there), each after the one before.
                    prev = [m.id for m in made if m.id in ids]
                    after = [i for i in ids if self.datasets[i].parent_id == cube.id]
                    at = ids.index(prev[-1]) if prev else ids.index(after[-1] if after else cube.id)
                    ids.insert(at + 1, ds.id)
            if output == "colour":
                colour_id = ds.id
            made.append(ds)
        return made

    # ---------- cropping ----------

    def crop(self, ds: Dataset, region=None, lo: float | None = None, hi: float | None = None) -> Dataset:
        """A new spectrum image under ``ds``: its pixels in the rectangle
        ``region`` (a ``processing.regions.Region``; the pixels whose centres
        are inside, as for its spectrum) and/or its channels in the energy
        window lo–hi (HyperSpy's ``inav`` / ``isig``). Returns it."""
        from .processing.background import EnergyAxis
        from .processing.regions import region_mask

        if region is None and lo is None and hi is None:
            raise ValueError("Give a region or an energy window to crop to")
        if ds.insitu is not None:
            return self._crop_series(ds, region, lo, hi)
        if ds.kind in (DatasetKind.IMAGE, DatasetKind.STACK):
            return self._crop_image(ds, region, lo, hi)
        if ds.kind not in (DatasetKind.SPECTRUM_IMAGE, DatasetKind.SPECTRUM):
            raise ValueError(f"{ds.title} can't be cropped")
        if ds.kind is DatasetKind.SPECTRUM and region is not None:
            raise ValueError(f"{ds.title} is a spectrum: crop it to an energy window")
        ds.materialise()
        signal = ds.signal
        parts = []
        if region is not None:
            h, w = si_cube(ds).shape[:2]
            rows, cols = self._crop_box(region, h, w)
            if ds.is_line_scan:
                signal = signal.inav[cols.start:cols.stop]
                parts.append(f"{cols.start}–{cols.stop - 1}")
            else:
                signal = signal.inav[cols.start:cols.stop, rows.start:rows.stop]
                parts.append(f"x {cols.start}–{cols.stop - 1}, y {rows.start}–{rows.stop - 1}")
        if lo is not None or hi is not None:
            c0, c1 = self.energy_channels(ds, lo, hi)
            signal = signal.isig[c0:c1]
            axis = EnergyAxis.of(ds.signal)
            parts.append(
                f"{axis.offset + axis.scale * c0:g}–{axis.offset + axis.scale * (c1 - 1):g}"
                f"{' ' + axis.units if axis.units else ''}"
            )
        signal = signal.deepcopy()
        return self.copy_as(ds, "cropped", f"{ds.title} ({'; '.join(parts)})", signal=signal)

    @staticmethod
    def _crop_box(region, h: int, w: int) -> tuple[slice, slice]:
        """The rows and columns of a rectangle ``region`` (the pixels whose
        centres are inside, as for its spectrum) in an h × w image."""
        from .processing.regions import region_mask

        if region.kind != "rect":
            raise ValueError("Only a rectangle can be cropped to")
        rows, cols, _ = region_mask((h, w), region)
        if cols.stop - cols.start < 1 or rows.stop - rows.start < 1:
            raise ValueError("The region holds no pixels")
        return rows, cols

    def _crop_image(self, ds: Dataset, region, lo, hi) -> Dataset:
        """An image or stack cropped to a rectangle (a stack: every frame, as
        shown, i.e. aligned if it is), as a new dataset under it."""
        if region is None or lo is not None or hi is not None:
            raise ValueError(f"{ds.title} has no energy axis: crop it to a rectangle")
        h, w = ds.signal.data.shape[-2:]
        rows, cols = self._crop_box(region, h, w)
        signal = ds.signal.isig[cols.start:cols.stop, rows.start:rows.stop].deepcopy()
        title = f"{ds.title} (x {cols.start}–{cols.stop - 1}, y {rows.start}–{rows.stop - 1})"
        return self.copy_as(ds, "cropped", title, signal=signal)

    def _crop_series(self, ds: Dataset, region, lo, hi) -> Dataset:
        """An in-situ series cropped to a rectangle and/or an energy window,
        every frame (its time kept): a new series of the same file, read and
        cropped a frame at a time, under it."""
        from .processing.technique import technique_of

        series = ds.insitu
        _, fy, fx, fe = series.binning
        h, w, n = series.frame_shape
        parts = []
        rows, cols, chans = slice(0, h), slice(0, w), slice(0, n)
        if region is not None:
            rows, cols = self._crop_box(region, h, w)
            parts.append(f"x {cols.start}–{cols.stop - 1}, y {rows.start}–{rows.stop - 1}")
        if lo is not None or hi is not None:
            c0, c1 = self.energy_channels(ds, lo, hi)
            chans = slice(c0, c1)
            axis = ds.signal.axes_manager.signal_axes[0]
            units = axis.units if isinstance(axis.units, str) and axis.units != "None" else ""
            parts.append(f"{axis.offset + axis.scale * c0:g}–{axis.offset + axis.scale * (c1 - 1):g}{' ' + units if units else ''}")
        # In the recorded frames (binning undone), within any crop it already has.
        base = series.crop or ((0, series.raw_frame_shape[0]), (0, series.raw_frame_shape[1]), (0, series.raw_frame_shape[2]))
        crop = tuple(
            (b[0] + s.start * f, b[0] + s.stop * f) for b, s, f in zip(base, (rows, cols, chans), (fy, fx, fe))
        )
        made = self.crop_series_raw(ds, crop, f"{ds.title} ({'; '.join(parts)})")
        if series.binning != (1, 1, 1, 1):
            made.insitu.set_binning(series.binning)
            title = made.title
            made.signal = made.insitu.placeholder()
            apply_technique(made.signal, technique_of(ds.signal))
            made.signal.metadata.General.title = title
        self.show_insitu_frame(made, min(series.view.start, made.insitu.n - 1))
        return made

    def crop_series_raw(self, src: Dataset, crop: tuple, title: str, ds_id: str | None = None) -> Dataset:
        """A series of ``src``'s file cropped to ``crop`` (rows, columns,
        channels of the recorded frames, as (start, stop) each), under ``src``
        (``ds_id``: the id it gets)."""
        from .insitu_series import InsituSeries
        from .processing.technique import technique_of

        series = InsituSeries(src.insitu.signal, crop=crop)
        signal = series.placeholder()
        apply_technique(signal, technique_of(src.signal))
        signal.metadata.General.title = title
        made = Dataset(
            id=ds_id or _new_id(), file_id=src.file_id, signal=signal, kind=DatasetKind.SPECTRUM_IMAGE,
            origin="cropped", parent_id=src.id, technique_source=src.technique_source, insitu=series,
            eels_role=src.eels_role, eels_role_source=src.eels_role_source,
        )
        with self._lock:
            self.datasets[made.id] = made
            ids = self.files[src.file_id].dataset_ids
            if made.id not in ids:
                ids.insert(ids.index(src.id) + 1, made.id)
        return made

    # ---------- PCA (M19a) ----------

    @staticmethod
    def can_decompose(ds: Dataset) -> bool:
        """An EELS or EDS spectrum image (or line scan) to decompose."""
        from .processing.technique import technique_of

        return ds.kind is DatasetKind.SPECTRUM_IMAGE and ds.insitu is None and technique_of(ds.signal) in ("EELS", "EDS")

    def pca_channels(self, ds: Dataset, lo: float | None, hi: float | None) -> tuple[int, int]:
        """The channels [c0, c1) of the energy window lo–hi of ``ds`` (the
        whole axis where not given). ValueError for fewer than 2 channels."""
        return self.energy_channels(ds, lo, hi)

    def energy_channels(self, ds: Dataset, lo: float | None, hi: float | None) -> tuple[int, int]:
        """The channels [c0, c1) of the energy window lo–hi of ``ds`` (the
        whole axis where not given). ValueError for fewer than 2 channels."""
        from .processing.background import EnergyAxis

        axis = EnergyAxis.of(ds.signal)
        c0 = 0 if lo is None else int(round((lo - axis.offset) / axis.scale))
        c1 = axis.size if hi is None else int(round((hi - axis.offset) / axis.scale)) + 1
        c0, c1 = max(0, min(c0, axis.size)), max(0, min(c1, axis.size))
        if c1 - c0 < 2:
            raise ValueError("The energy window must hold at least 2 channels")
        return c0, c1

    def make_pca(
        self, ds: Dataset, lo: float | None = None, hi: float | None = None, components: int = 32,
        poisson: bool = True, report=None, algorithm: str = "SVD", normalise: dict | None = None,
    ) -> Dataset:
        """Decompose the spectrum image ``ds`` over the energy window lo–hi
        with HyperSpy's ``decomposition`` (``algorithm``, one of
        ``DECOMPOSITIONS``; ``normalize_poissonian_noise`` if ``poisson``): a
        stack of the components' maps (loadings) under it, titled by the
        algorithm, with their spectra (factors) under that. Made again in place
        if ``ds`` was decomposed with the same algorithm before. Returns the
        stack."""
        import hyperspy.api as hs

        from .processing.background import EnergyAxis

        if not self.can_decompose(ds):
            raise ValueError(f"{ds.title} is not an EELS or EDS spectrum image")
        if algorithm not in DECOMPOSITIONS:
            raise ValueError(f"Unknown decomposition algorithm {algorithm!r}")
        name, centred = DECOMPOSITIONS[algorithm]
        # scikit-learn's centre the data: the mean is put back when denoising,
        # which can't be done after Poisson weighting. Normalised spectra are
        # not counts: no Poisson weighting either.
        poisson = bool(poisson) and not centred and not normalise
        c0, c1 = self.pca_channels(ds, lo, hi)
        ds.materialise()
        if report:
            report(0.05, "Decomposing")
        window = ds.signal.isig[c0:c1].deepcopy()
        window.change_dtype("float32")
        # Each spectrum normalised as on the plot (over its whole energy
        # range), undone when denoising: v' = (v - offset) / scale.
        norm_offset = norm_scale = None
        if normalise:
            norm_offset, norm_scale = spectrum_normalisation(ds.signal, normalise)
            window.data = (window.data - norm_offset[..., None]) / norm_scale[..., None]
        k = max(1, min(int(components), window.axes_manager.navigation_size, c1 - c0))
        extra = {
            "SVD": {"svd_solver": "randomized"},
            # Maximum likelihood PCA needs the noise variance: the counts'.
            "MLPCA": {"var_func": lambda x: np.maximum(x, 1)},
        }.get(algorithm, {})
        window.decomposition(
            normalize_poissonian_noise=poisson, algorithm=algorithm, output_dimension=k, print_info=False, **extra,
        )
        lr = window.learning_results
        h, w = si_cube(ds).shape[:2]
        loadings = np.ascontiguousarray(lr.loadings[:, :k].T.reshape(k, h, w), dtype=np.float32)
        factors = np.ascontiguousarray(lr.factors[:, :k].T, dtype=np.float32)
        # NMF and the online robust ones rank nothing by variance.
        ev = lr.explained_variance_ratio
        explained = [] if ev is None else np.asarray(ev)[:k]
        mean = None if lr.mean is None else np.asarray(lr.mean, np.float32).ravel()
        if mean is not None and mean.size != c1 - c0:
            mean = None
        del window, lr
        if report:
            report(0.95, "Making the components")
        axis = EnergyAxis.of(ds.signal)
        settings = {
            "source_id": ds.id, "window": [axis.offset + axis.scale * c0, axis.offset + axis.scale * (c1 - 1)],
            "channels": [c0, c1], "poisson": bool(poisson), "components": k,
            "explained": [float(v) for v in explained], "algorithm": algorithm, "name": name,
            "normalise": dict(normalise) if normalise else None,
        }
        comp_axis = {"name": "Component", "size": k, "offset": 1, "scale": 1, "units": "", "navigate": True}
        nav = [ax.get_axis_dictionary() for ax in ds.signal.axes_manager.navigation_axes[::-1]]  # array order
        for d in nav:
            d["navigate"] = False
        if ds.is_line_scan:
            nav = [{"name": "y", "size": 1, "offset": 0, "scale": 1, "units": nav[0].get("units", ""), "navigate": False}, *nav]
        maps = hs.signals.Signal2D(loadings, axes=[comp_axis, *nav])  # (k, 1, w) for a line scan
        maps.metadata.General.title = name
        energy = ds.signal.axes_manager.signal_axes[0].get_axis_dictionary()
        energy.update(offset=axis.offset + axis.scale * c0, size=c1 - c0, navigate=False)
        spectra = hs.signals.Signal1D(factors, axes=[comp_axis, energy])
        spectra.metadata.General.title = f"{name} factors"
        if mean is not None:
            spectra.metadata.set_item("Decomposition.mean", mean)
        if norm_offset is not None:
            spectra.metadata.set_item("Decomposition.norm_offset", norm_offset.astype(np.float32))
            spectra.metadata.set_item("Decomposition.norm_scale", norm_scale.astype(np.float32))

        with self._lock:
            earlier = {
                d.origin: d for d in self.datasets.values()
                if d.origin in ("pca", "pca-factors") and d.pca and d.pca.get("source_id") == ds.id
                and d.pca.get("algorithm", "SVD") == algorithm
            }
        made = {}
        for origin, signal, parent in (("pca", maps, ds.id), ("pca-factors", spectra, None)):
            parent = parent or made["pca"].id
            old = earlier.get(origin)
            if old is not None:
                old.replace_signal(signal)
                old.pca = dict(settings)
                made[origin] = old
                continue
            new = Dataset(
                id=_new_id(), file_id=ds.file_id, signal=signal, kind=classify(signal), origin=origin,
                parent_id=parent, pca=dict(settings),
            )
            with self._lock:
                self.datasets[new.id] = new
                ids = self.files[ds.file_id].dataset_ids
                ids.insert(ids.index(parent) + 1, new.id)
            made[origin] = new
        return made["pca"]

    def pca_factors(self, pca: Dataset) -> Dataset:
        """The "PCA factors" under a PCA stack."""
        if pca.origin != "pca" or not pca.pca:
            raise ValueError(f"{pca.title} is not a PCA")
        for d in self.datasets.values():
            if d.origin == "pca-factors" and d.parent_id == pca.id:
                return d
        raise ValueError(f"{pca.title} has lost its PCA factors")

    def denoise_pca(self, pca: Dataset, n: int) -> Dataset:
        """The spectrum image the PCA ``pca`` was made from, rebuilt from its
        first ``n`` components over its energy window (HyperSpy's
        ``get_decomposition_model`` of the stored loadings and factors): a new
        spectrum image under the source (or the one made with the same ``n``,
        remade)."""
        from hyperspy.learn import LearningResults

        factors = self.pca_factors(pca)
        source = self.get_dataset(pca.pca["source_id"])
        k = int(pca.pca["components"])
        n = max(1, min(int(n), k))
        c0, c1 = pca.pca["channels"]
        source.materialise()
        window = source.signal.isig[c0:c1]
        loadings = np.asarray(pca.signal.data, np.float32)
        if loadings.reshape(k, -1).shape[1] != window.axes_manager.navigation_size or factors.signal.data.shape[1] != c1 - c0:
            raise ValueError(f"{source.title} has changed since it was decomposed: decompose it again")
        window.learning_results.loadings = loadings.reshape(k, -1).T
        window.learning_results.factors = np.asarray(factors.signal.data, np.float32).T
        signal = window.get_decomposition_model(n)
        signal.change_dtype("float32")
        # The mean taken out by scikit-learn's algorithms, put back (HyperSpy
        # can't: it expects one per pixel).
        mean = factors.signal.metadata.get_item("Decomposition.mean")
        if mean is not None:
            signal.data += np.asarray(mean, np.float32)
        # Decomposed normalised: back to the spectra's own scale.
        offset = factors.signal.metadata.get_item("Decomposition.norm_offset")
        scale = factors.signal.metadata.get_item("Decomposition.norm_scale")
        if offset is not None and scale is not None:
            signal.data = signal.data * np.asarray(scale, np.float32)[..., None] + np.asarray(offset, np.float32)[..., None]
        signal.learning_results = LearningResults()  # the PCA keeps them, not the denoised SI
        title = f"{source.title} ({pca.pca.get('name', 'PCA')}, {n})"
        settings = {**pca.pca, "n": n}
        with self._lock:
            earlier = next((
                d for d in self.datasets.values()
                if d.origin == "pca-denoised" and d.pca and d.pca.get("source_id") == source.id and d.pca.get("n") == n
                and d.pca.get("algorithm", "SVD") == pca.pca.get("algorithm", "SVD")
            ), None)
        if earlier is not None:
            signal.metadata.General.title = title
            earlier.replace_signal(signal)
            earlier.pca = settings
            return earlier
        new = self.copy_as(source, "pca-denoised", title, signal=signal)
        new.pca = settings
        return new

    def dpc_frame(self, ds: Dataset, index: int = 0, projection: str | None = None, bin: int = 1) -> np.ndarray:
        """A DPC object's direction (degrees) and magnitude as shown, (2, h,
        w): of the image, or of frame ``index`` of a stack (``projection``
        "mean"/"sum": of the mean deflection), mean-binned by ``bin``."""
        from .processing.frames import bin2d

        x, y = self._dpc_parts(ds)
        if x.ndim == 3:
            if projection in ("mean", "sum"):
                x, y = x.mean(axis=0), y.mean(axis=0)
            else:
                if not 0 <= index < len(x):
                    raise IndexError(f"Frame {index} out of range 0–{len(x) - 1}")
                x, y = x[index], y[index]
        x, y = bin2d(x, bin), bin2d(y, bin)
        return np.stack([np.degrees(np.arctan2(y, x)), np.sqrt(x * x + y * y)]).astype(np.float32)

    def _dpc_parts(self, ds: Dataset) -> tuple[np.ndarray, np.ndarray]:
        """The DPC x and DPC y under a DPC object."""
        if ds.origin != "dpc" or not ds.dpc or ds.dpc.get("output") != "colour":
            raise ValueError(f"{ds.title} is not a DPC image")
        parts = self._dpc_children(ds)
        if "x" not in parts or "y" not in parts:
            raise ValueError(f"{ds.title} has lost its x or y component")
        return np.asarray(parts["x"].signal.data, np.float32), np.asarray(parts["y"].signal.data, np.float32)

    def dpc_magnitude(self, ds: Dataset) -> np.ndarray:
        """The magnitude of the deflection of a DPC object, √(x² + y²), from
        the DPC x and DPC y under it."""
        x, y = self._dpc_parts(ds)
        return np.sqrt(x * x + y * y)

    def _remake_after_zlp(self, d: Dataset, shift: float) -> None:
        """Recompute what was made from ``d`` after its data was aligned, with
        energies ``shift`` further along the axis than before."""
        from .processing.background import BackgroundOptions
        from .processing.zlp import round_energy

        self._drop_fits(d)  # made from the old data: fit again to update
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

    def set_eels_elements(self, ds: Dataset, elements: list[str]) -> None:
        """Set the elements of an EELS spectrum or spectrum image: the ones
        whose edges a model fit gets (see processing/eels_model.py)."""
        from .processing.eels_model import set_elements

        if ds.kind not in (DatasetKind.SPECTRUM, DatasetKind.SPECTRUM_IMAGE) or technique_of(ds.signal) != "EELS":
            raise ValueError(f"{ds.title} is not an EELS spectrum")
        for signal in dict.fromkeys(s for s in (ds.signal, ds.original_signal, ds.aligned_signal) if s is not None):
            set_elements(signal, elements)

    def eels_fit_low_loss(self, ds: Dataset) -> Dataset | None:
        """The low loss an EELS model of ``ds`` is convolved with: its group's
        (for spectra made from another spectrum image, that one's), on the
        same scan; None without one, or for the low loss itself."""
        ll = self.deconvolution_low_loss(ds)
        if ll is None or ll is ds:
            return None
        ll.materialise()
        same = tuple(ll.signal.axes_manager.navigation_shape) == tuple(ds.signal.axes_manager.navigation_shape)
        return ll if same else None

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

    def delete_dataset(self, dataset: Dataset) -> str:
        """Delete an element map, a window map, a thickness or a
        background-removed or deconvolved spectrum image (with everything made
        from it). Returns a token that ``undo`` takes to bring it back.

        Deleting an element map also unticks its element in the map's spectrum image and in the
        file's other (non-image) EDS spectra, which removes that element's
        computed maps too. Other data can't be deleted.
        """
        from .processing.xray import eds_info

        entry: dict = {"file_id": dataset.file_id, "restore": [], "elements": []}
        if dataset.origin in ("window", "thickness", "zlp-shift"):
            entry["restore"] = self._tree_places(dataset)
            self._remove(dataset)  # no element to untick
            return self._keep_undoable(entry)
        if dataset.origin in ("background", "background-r", "deconvolved", "line-scan", "aligned-sum", "rebinned", "zlp-aligned", "aligned", "dpc", "pca", "pca-factors", "pca-denoised", "cropped", "curve-fit", "curve-model", "atomic", "mask"):
            if dataset.origin == "curve-model" and dataset.parent_id in self.datasets:
                self.datasets[dataset.parent_id].curve_fits = None  # its fit at the cursor goes with it
            places = self._tree_places(dataset)
            # Neighbourhoods of a deleted sublattice (as origins or neighbours) go with it.
            gone = {d.id for d, _ in places}
            with self._lock:
                stale = [d for d in self.datasets.values() if d.atomic and d.atomic.get("output") == "neighbourhood"
                         and d.id not in gone and (d.atomic.get("origins") in gone or d.atomic.get("neighbours") in gone)]
            for nb in stale:
                places += [p for p in self._tree_places(nb) if p[0].id not in gone]
            entry["restore"] = sorted(places, key=lambda p: p[1])
            self._remove_tree(dataset)
            for nb in stale:
                if nb.id in self.datasets:
                    self._remove_tree(nb)
            return self._keep_undoable(entry)
        if not dataset.is_element_map:
            raise ValueError(f"{dataset.title} is not an element map and can't be deleted")
        # A computed map comes back with its element; any other is kept.
        if dataset.origin != "computed":
            entry["restore"] = self._tree_places(dataset)
        self._remove(dataset)
        if dataset.origin == "fitted":
            return self._keep_undoable(entry)  # results of a per-pixel fit: nothing else to undo
        element = dataset.element
        for d in self.siblings_in_file(dataset.file_id):
            if d.kind is DatasetKind.SPECTRUM_IMAGE and dataset.parent_id not in (None, d.id):
                continue  # another spectrum image in the same file
            info = eds_info(d.signal)
            if info is not None and element in info["elements"]:
                entry["elements"].append((d.id, list(info["elements"])))
                self.set_elements(d, [e for e in info["elements"] if e != element])
        return self._keep_undoable(entry)

    # ---------- undoing a deletion ----------

    UNDO_LIMIT = 20

    def _tree_places(self, dataset: Dataset) -> list[tuple[Dataset, int]]:
        """``dataset`` and everything made from it, each with its place in its
        file's list, in list order."""
        with self._lock:
            ids = self.files[dataset.file_id].dataset_ids
            tree, todo = [], [dataset]
            while todo:
                d = todo.pop()
                tree.append(d)
                todo += [c for c in self.siblings(d) if c.parent_id == d.id]
            return sorted(((d, ids.index(d.id)) for d in tree if d.id in ids), key=lambda p: p[1])

    def _keep_undoable(self, entry: dict) -> str:
        token = _new_id()
        with self._lock:
            self.undoable[token] = entry
            while len(self.undoable) > self.UNDO_LIMIT:
                self.undoable.pop(next(iter(self.undoable)))
        return token

    def undo(self, token: str) -> list[str]:
        """Bring back what the deletion ``token`` removed: the datasets in
        their places, and the elements it unticked (which remakes their maps).
        Returns the ids of the files that changed. KeyError if the token is
        unknown (or too old); ValueError if the file was closed."""
        with self._lock:
            entry = self.undoable.pop(token)
            entry_file = self.files.get(entry["file_id"])
            if entry_file is None:
                raise ValueError("Its file was closed")
            for d, index in entry["restore"]:
                if d.parent_id is not None and d.parent_id not in self.datasets:
                    continue  # made from something that is gone too
                self.datasets[d.id] = d
                if d.id not in entry_file.dataset_ids:
                    entry_file.dataset_ids.insert(min(index, len(entry_file.dataset_ids)), d.id)
        for ds_id, elements in entry["elements"]:
            ds = self.datasets.get(ds_id)
            if ds is not None:
                self.set_elements(ds, elements)
        return [entry["file_id"]]

    def _curve_fit_maps(self, si: Dataset) -> list[Dataset]:
        """The results of ``si``'s curve fit of every pixel: its model (with
        the maps under it)."""
        return [
            d for d in self.siblings_in_file(si.file_id)
            if d.parent_id == si.id and d.origin in ("curve-model", "curve-fit")
        ]

    def _drop_fits(self, si: Dataset) -> bool:
        """Remove the results of fitting every pixel of ``si`` (made from data
        that has changed). Returns whether there were any."""
        old = [*self._fit_maps(si), *self._curve_fit_maps(si)]
        for d in old:
            if d.id in self.datasets:
                self._remove_tree(d)
        return bool(old)

    def set_curve_model(self, si: Dataset, model, spec: dict, maps: dict[str, np.ndarray]) -> Dataset:
        """Replace ``si``'s curve model: ``model`` (the fitted model at every
        pixel, a spectrum image) under ``si``, with ``maps`` (title → data:
        the reduced χ² and each parameter's) under it. ``spec``: the model per
        pixel. Returns the model's dataset."""
        from .processing.element_maps import scan_image

        for d in self._curve_fit_maps(si):
            if d.id in self.datasets:
                self._remove_tree(d)
        model.metadata.General.title = "Curve model"
        made = Dataset(
            id=_new_id(), file_id=si.file_id, signal=model, kind=DatasetKind.SPECTRUM_IMAGE,
            origin="curve-model", parent_id=si.id, curve_model={"source_id": si.id, "model": spec},
        )
        items = list(maps.items())
        chi, params = items[-1], items[:-1]
        children = [
            Dataset(
                id=_new_id(), file_id=si.file_id, signal=scan_image(si.signal, data, title), kind=DatasetKind.IMAGE,
                origin="curve-fit", parent_id=made.id,
            )
            for title, data in [chi, *params]
        ]
        with self._lock:
            for d in [made, *children]:
                self.datasets[d.id] = d
                self.files[si.file_id].dataset_ids.append(d.id)
        return made

    def curve_model_maps(self, model: Dataset) -> dict[str, np.ndarray]:
        """A curve model's maps by title."""
        return {
            d.title: np.asarray(d.signal.data)
            for d in self.siblings_in_file(model.file_id)
            if d.parent_id == model.id and d.origin == "curve-fit"
        }

    def _fit_maps(self, si: Dataset) -> list[Dataset]:
        return [d for d in self.siblings_in_file(si.file_id) if d.parent_id == si.id and d.origin == "fitted"]

    def set_fit_maps(self, si: Dataset, maps: list[tuple[str, dict[str, tuple[str, np.ndarray]]]]) -> None:
        """Replace ``si``'s per-pixel fit results with ``maps``: one per
        element, (element, {quantity: (title, data)}) with "counts" first. The
        quantity shown stays the one the old maps showed, where a map has it."""
        from .processing.element_maps import scan_image

        old = self._fit_maps(si)
        shown = "at%" if any(d.quantity == "at%" for d in old) else "counts"
        for d in old:
            self._remove(d)
        for element, by_quantity in maps:
            quantities = {q: scan_image(si.signal, data, title) for q, (title, data) in by_quantity.items()}
            quantity = shown if shown in quantities else "counts"
            ds = Dataset(
                id=_new_id(), file_id=si.file_id, signal=quantities[quantity], kind=DatasetKind.IMAGE,
                is_element_map=True, origin="fitted", element=element, parent_id=si.id,
                quantities=quantities, quantity=quantity,
            )
            with self._lock:
                self.datasets[ds.id] = ds
                self.files[si.file_id].dataset_ids.append(ds.id)

    def set_fit_quantity(self, si: Dataset, quantity: str) -> None:
        """Show ``quantity`` on every map of ``si``'s per-pixel fit that has it
        (a new revision, so contrast is redone). ValueError if none has it."""
        maps = [d for d in self._fit_maps(si) if quantity in d.quantities]
        if not maps:
            raise ValueError(f"{si.title} has no fitted maps in {quantity}")
        for d in maps:
            if d.quantity != quantity:
                d.quantity = quantity
                d.replace_signal(d.quantities[quantity])

    def rebin(
        self, si: Dataset, factors: list[int], with_linked: bool = True, time: int = 1, in_place: bool = True,
        with_scan: bool = True,
    ) -> list[Dataset]:
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

        An in-situ series is rebinned as a series (see ``_rebin_series``), and
        can also be binned along ``time`` (frames per binned frame). A line
        scan (positions or times × energy) is rebinned along both its axes,
        with the line scans linked to it. A stack is rebinned on its own (see
        ``_rebin_stack``).

        Not ``in_place``: ``si`` is left as it is and a copy of it is rebinned
        instead, a new dataset under it (``origin`` "rebinned"), without the
        images on its scan; ``with_linked``: the linked EELS spectrum images
        are copied and rebinned too, the copies linked to each other as the
        sources are (else it is on its own). In-situ series are rebinned in
        place only. ``with_scan``: the images on the scan are binned too (in
        place only).
        """
        if not in_place:
            if si.insitu is not None:
                raise ValueError(f"{si.title} is an in-situ series: it can only be rebinned in place")
            if si.kind not in (DatasetKind.SPECTRUM_IMAGE, DatasetKind.STACK, DatasetKind.DIFFRACTION4D):
                raise NotRebinnable(f"{si.title} can't be rebinned: only spectrum images, stacks and 4D-STEM data can")
            sources = [si] + (self._same_scan_linked(si) if with_linked and si.kind is DatasetKind.SPECTRUM_IMAGE else [])
            for d in sources:
                self._check_factors(d, factors)
            copies = []
            try:
                for d in sources:
                    copy = self.copy_as(d, "rebinned", f"{d.title} (rebinned)")
                    copies.append(copy)
                    self.rebin(copy, factors, with_linked=False, time=time, with_scan=False)
            except Exception:
                for c in copies:
                    self._remove(c)
                raise
            if len(copies) > 1:
                self._join(copies)
            return copies
        if si.kind is DatasetKind.STACK:
            if time != 1:
                raise ValueError(f"{si.title} has no time axis to rebin: only in-situ series do")
            return self._rebin_stack(si, factors)
        if si.kind is DatasetKind.DIFFRACTION4D:
            if time != 1:
                raise ValueError(f"{si.title} has no time axis to rebin: only in-situ series do")
            return self._rebin_4d(si, factors)
        if si.kind is not DatasetKind.SPECTRUM_IMAGE:
            raise NotRebinnable(f"{si.title} can't be rebinned: only spectrum images and stacks can")
        linked_all = self.linked_of(si) if with_linked else []
        if si.insitu is None:
            if time != 1:
                raise ValueError(f"{si.title} has no time axis to rebin: only in-situ series do")
            for d in linked_all:
                if d.insitu is not None:
                    raise NotRebinnable(f"{d.title} (linked) is an in-situ series: rebin it from the series instead")
        elif any(d.insitu is None for d in linked_all):
            raise NotRebinnable(f"{si.title} is linked to spectrum images that are not in-situ series: unlink them first")
        am = si.signal.axes_manager
        axes = list(am._axes)
        if len(factors) != len(axes):
            raise ValueError(f"Expected {len(axes)} factors, got {len(factors)}")
        for f, ax in zip(factors, axes):
            if not 1 <= f <= ax.size:
                raise ValueError(f"Factor for {ax.name or 'axis'} must be between 1 and {ax.size}")
        if si.insitu is not None and not 1 <= time <= si.insitu.n:
            raise ValueError(f"Factor for Time must be between 1 and {si.insitu.n}")
        if all(f == 1 for f in factors) and time == 1:
            raise ValueError("Choose a factor above 1 for at least one axis")

        # HyperSpy's scale is in natural order (x, y | energy).
        natural = am._get_axes_in_natural_order()
        scale = [factors[next(i for i, a in enumerate(axes) if a is ax)] for ax in natural]
        scan = tuple(am.navigation_shape)
        linked = self._same_scan_linked(si) if with_linked else []
        for d in linked:
            d_axes = list(d.signal.axes_manager._axes)
            if len(d_axes) != len(factors):
                raise ValueError(f"{d.title} (linked) has {len(d_axes)} axes, so it can't get the same binning")
            for f, ax in zip(factors, d_axes):
                if f > ax.size:
                    raise ValueError(
                        f"Factor for {ax.name or 'axis'} must be at most {ax.size} for {d.title} (linked)"
                    )
            if d.insitu is not None and time > d.insitu.n:
                raise ValueError(f"Factor for Time must be at most {d.insitu.n} for {d.title} (linked)")
        on_scan = [] if si.is_line_scan or not with_scan else [
            d for d in self.siblings_in_file(si.file_id)
            if d.origin == "file" and d.kind in (DatasetKind.IMAGE, DatasetKind.STACK)
            and tuple(d.loaded_signal.axes_manager.signal_shape) == scan
        ]

        si.materialise()
        for d in on_scan:
            self.remove_alignment(d)  # its shifts are in pixels of the old size
        if si.insitu is not None:
            self._rebin_series(si, scale, time)
        else:
            _rebin_in_place(si, scale)
        for d in on_scan:
            # Natural order: (frames |) x, y.
            _rebin_in_place(d, [1] * d.signal.axes_manager.navigation_dimension + scale[:2])

        self._drop_fits(si)
        for d in self.zlp_shift_maps(si):
            self._remove(d)  # made at the old size: refit or align again to update
        for m in self.computed_maps(si):
            new_map = self._make_map(si, m.element)
            if new_map is None:  # its line left the (rebinned) energy range
                self._remove(m)
            else:
                m.replace_signal(new_map)
        self._remake_window_maps(si)
        if si.insitu is None:  # of an in-situ series they are snapshots of a frame, as they were made
            self._remake_backgrounds(si)
            self._remake_thickness(si)
        if si.origin == "background":
            for r in self._background_r_maps(si):
                self._remove(r)  # made at the old size: re-run from the source to update
        rebinned = [si, *on_scan]
        for d in linked:
            rebinned += self.rebin(d, factors, with_linked=False, time=time)
        return rebinned

    def _same_scan_linked(self, si: Dataset) -> list[Dataset]:
        """The EELS spectrum images linked to ``si`` still on its scan (the
        ones rebinned with it)."""
        scan = tuple(si.signal.axes_manager.navigation_shape)
        return [
            d for d in self.linked_of(si)
            if d.kind is DatasetKind.SPECTRUM_IMAGE and d.is_line_scan == si.is_line_scan
            and tuple(d.signal.axes_manager.navigation_shape) == scan  # else no longer the same scan
        ]

    @staticmethod
    def _check_factors(ds: Dataset, factors: list[int]) -> None:
        """ValueError unless ``factors`` has one factor per axis of ``ds``, each
        1…its size, one above 1."""
        axes = list(ds.loaded_signal.axes_manager._axes)
        if len(factors) != len(axes):
            raise ValueError(f"Expected {len(axes)} factors, got {len(factors)}")
        for f, ax in zip(factors, axes):
            if not 1 <= f <= ax.size:
                raise ValueError(f"Factor for {ax.name or 'axis'} must be between 1 and {ax.size}")
        if all(f == 1 for f in factors):
            raise ValueError("Choose a factor above 1 for at least one axis")

    def _rebin_stack(self, stack: Dataset, factors: list[int]) -> list[Dataset]:
        """Rebin a stack along its frames, y and x (``factors`` in the
        summary's axis order: frames, y, x) in place; sizes that don't divide
        are cropped. A drift correction (in pixels of the old size) is dropped
        with its average. The simultaneous ADF of an in-situ series keeps its
        frames, one per frame of the series."""
        am = stack.loaded_signal.axes_manager
        axes = list(am._axes)
        if len(factors) != len(axes):
            raise ValueError(f"Expected {len(axes)} factors, got {len(factors)}")
        for f, ax in zip(factors, axes):
            if not 1 <= f <= ax.size:
                raise ValueError(f"Factor for {ax.name or 'axis'} must be between 1 and {ax.size}")
        if all(f == 1 for f in factors):
            raise ValueError("Choose a factor above 1 for at least one axis")
        natural = am._get_axes_in_natural_order()
        scale = [factors[next(i for i, a in enumerate(axes) if a is ax)] for ax in natural]
        insitu_adf = stack.loaded_signal.metadata.get_item("Insitu.frames") is not None
        if insitu_adf and any(f > 1 for f, ax in zip(scale, natural) if ax.navigate):
            raise ValueError(
                f"{stack.title} is the simultaneous ADF of an in-situ series: its frames follow the series "
                "(rebin the series in time instead); x and y can be rebinned"
            )
        self.remove_alignment(stack)
        stack.materialise()
        _rebin_in_place(stack, scale)
        return [stack]

    def _rebin_4d(self, ds: Dataset, factors: list[int]) -> list[Dataset]:
        """Rebin 4D-STEM data in place: its scan (Bin R) and its patterns
        (Bin Q) by integer factors (``factors`` in the summary's axis order:
        y, x, ky, kx), summing; sizes that don't divide are cropped. Integer
        data becomes the smallest integer type that holds the sums (HyperSpy
        alone would make it 64-bit)."""
        self._check_factors(ds, factors)
        am = ds.loaded_signal.axes_manager
        axes = list(am._axes)
        natural = am._get_axes_in_natural_order()
        scale = [factors[next(i for i, a in enumerate(axes) if a is ax)] for ax in natural]
        dtype = ds.signal.data.dtype
        target = None
        if dtype.kind in "iu":
            info = np.iinfo(dtype)
            most = int(np.prod(factors)) * max(abs(int(info.max)), abs(int(info.min)))
            target = next(
                t for t in ((np.uint8, np.uint16, np.uint32, np.uint64) if dtype.kind == "u"
                            else (np.int8, np.int16, np.int32, np.int64))
                if np.iinfo(t).max >= most or t in (np.uint64, np.int64)
            )
            target = np.dtype(target) if np.dtype(target).itemsize >= dtype.itemsize else dtype
        # Cropped to whole bins first: HyperSpy interpolates otherwise.
        signal = ds.signal
        sizes = [ax.size // f * f for ax, f in zip(natural, scale)]
        if any(n != ax.size for n, ax in zip(sizes, natural)):
            nav = len(am.navigation_axes)
            signal = signal.inav[tuple(slice(0, n) for n in sizes[:nav])].isig[tuple(slice(0, n) for n in sizes[nav:])]
        rebinned = signal.rebin(scale=scale, dtype=target)
        if not getattr(rebinned, "_lazy", False) or rebinned.data.nbytes <= DIFFRACTION4D_IN_MEMORY_LIMIT:
            if getattr(rebinned, "_lazy", False):
                rebinned.compute(show_progressbar=False)
        ds.replace_signal(rebinned)
        return [ds]

    @staticmethod
    def _rebin_series(ds: Dataset, scale: list[int], time: int) -> None:
        """Rebin an in-situ series by ``scale`` (natural order: x, y, E) and
        ``time``: every frame read from now on is binned (on top of any
        earlier binning), and ``ds`` holds the binned frame where its view
        started (a frame range becomes that one frame), with the axes HyperSpy's
        rebin gives the frame it held."""
        series = ds.insitu
        fx, fy, fe = scale
        ft0, fy0, fx0, fe0 = series.binning
        raw_start = series.view.start * ft0
        template = ds.signal.rebin(scale=scale) if any(f > 1 for f in scale) else ds.signal
        series.set_binning((ft0 * time, fy0 * fy, fx0 * fx, fe0 * fe))
        k = min(raw_start // series.binning[0], series.n - 1)
        data = series.read(k)
        with ds._load_lock:
            ds.replace_signal(series.fresh_signal(template, data))
            series.view = ViewState("frame", k, k)
        natural = ds.signal.axes_manager._get_axes_in_natural_order()
        label = ", ".join(
            [f"Time×{time}"] * (time > 1)
            + [f"{_defined(ax.name) or f'axis {i}'}×{f}" for i, (ax, f) in enumerate(zip(natural, scale)) if f > 1]
        )
        ds.rebinned.append(label)
        ds.ops.append(_op({"op": "rebin-series", "scale": [int(f) for f in scale], "time": int(time)}))
        series.schedule_ahead(k)

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
            if self._drop_fits(dataset):
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
            self._drop_fits(ds)  # made from another frame: fit again to update

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

    def apply_alignment(self, stack: Dataset, applied: dict, in_place: bool = True) -> Dataset:
        """Store a drift correction with ``stack`` and show it: its frames
        are then the aligned ones (bad frames left out, cropped to the square
        inside every frame); the loaded frames are kept, to switch back.
        ``applied`` is ``RegistrationSession.applied()``. The average is added
        to the file as an image (replacing an earlier one) and returned.
        Not ``in_place``: the aligned frames become a new stack under
        ``stack`` (``origin`` "aligned"), with the average under it; ``stack``
        is left as it is."""
        from .processing.registration import aligned_signals

        if not in_place:
            aligned, mean = aligned_signals(
                stack.loaded_signal, applied["stack"], applied["average"], applied["info"]["square"],
            )
            new = self.copy_as(stack, "aligned", f"{stack.title} (aligned)", signal=aligned)
            new.ops.append(_op({"op": "align", "info": applied["info"]}))
            avg = Dataset(
                id=_new_id(), file_id=stack.file_id, signal=mean, kind=DatasetKind.IMAGE,
                origin="aligned-average", derived_from=new.id, parent_id=new.id,
            )
            with self._lock:
                self.datasets[avg.id] = avg
                ids = self.files[stack.file_id].dataset_ids
                ids.insert(ids.index(new.id) + 1, avg.id)
            return avg

        mean = self.restore_alignment(stack, applied)
        stack.ops.append(_op({"op": "align", "info": applied["info"]}))
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

    def restore_alignment(self, stack: Dataset, applied: dict):
        """Show ``stack``'s frames aligned as ``applied`` (no op recorded, its
        average left as it is). Returns the average as a signal."""
        from .processing.registration import aligned_signals

        original = stack.loaded_signal
        aligned, mean = aligned_signals(original, applied["stack"], applied["average"], applied["info"]["square"])
        stack.original_signal = original
        stack.aligned_signal = aligned
        stack.alignment = applied["info"]
        stack.signal = aligned
        self._view_changed(stack)
        return mean

    def measured_shifts(self, stack: Dataset) -> list[Dataset]:
        """The drift-corrected stacks whose measured shifts ``stack`` can be
        aligned with: the same frames and frame size as loaded, in the same
        file or experiment (e.g. the HAADF recorded with the segments, for a
        DPC or iDPC stack made from them); a HAADF first."""
        if stack.kind is not DatasetKind.STACK:
            return []
        entry = self.file_of(stack)
        files = [entry] if entry.experiment_id is None else [
            f for f in self.files.values() if f.experiment_id == entry.experiment_id
        ]
        shape = tuple(stack.loaded_signal.data.shape)
        out = [
            d for f in files for d in (self.datasets[i] for i in f.dataset_ids)
            if d is not stack and d.kind is DatasetKind.STACK and d.alignment is not None
            and tuple(d.loaded_signal.data.shape) == shape
        ]
        return sorted(out, key=lambda d: "HAADF" not in d.title.upper())

    def align_from(self, stack: Dataset, source: Dataset) -> None:
        """Align ``stack`` with the shifts measured on ``source`` (see
        ``measured_shifts``): its frames shifted and cropped as ``source``'s.
        A DPC object: its DPC x and y aligned, its direction remade from them
        (a direction is not shifted itself); a DPC stack's data of either sign
        is shifted as such."""
        from .project import realigned

        if source not in self.measured_shifts(stack):
            raise ValueError(f"{source.title}'s shifts don't apply to {stack.title}")
        info = source.alignment
        signed = stack.origin == "dpc"
        if stack.origin == "dpc" and stack.dpc and stack.dpc.get("output") == "colour":
            parts = self._dpc_children(stack)
            for p in parts.values():
                if p.alignment is not None:
                    self.remove_alignment(p)
                self.restore_alignment(p, realigned(p, info, signed=True))
                p.ops.append(_op({"op": "align", "info": info}))
            x = np.asarray(parts["x"].signal.data, np.float64)
            y = np.asarray(parts["y"].signal.data, np.float64)
            angle = np.degrees(np.arctan2(y, x)).astype(np.float32)
            mean = np.degrees(np.arctan2(y.mean(axis=0), x.mean(axis=0))).astype(np.float32)
            if stack.alignment is not None:
                self.remove_alignment(stack)
            self.restore_alignment(stack, {"stack": angle, "average": mean, "info": info})
            stack.ops.append(_op({"op": "align", "info": info}))
            return
        if stack.alignment is not None:
            self.remove_alignment(stack)
        self.apply_alignment(stack, realigned(stack, info, signed=signed))

    def _dpc_children(self, ds: Dataset) -> dict[str, Dataset]:
        """The DPC x and DPC y under a DPC object."""
        return {
            d.dpc["output"]: d for d in self.datasets.values()
            if d.parent_id == ds.id and d.origin == "dpc" and d.dpc and d.dpc.get("output") in ("x", "y")
        }

    def show_alignment(self, stack: Dataset, active: bool) -> None:
        """Show a stack's frames aligned or as loaded (a DPC object's DPC x
        and y with it)."""
        if stack.alignment is None:
            raise ValueError(f"{stack.title} has no alignment")
        target = stack.aligned_signal if active else stack.original_signal
        if stack.signal is not target:
            stack.signal = target
            self._view_changed(stack)
        if stack.origin == "dpc" and stack.dpc and stack.dpc.get("output") == "colour":
            for p in self._dpc_children(stack).values():
                if p.alignment is not None:
                    self.show_alignment(p, active)

    def remove_alignment(self, stack: Dataset) -> None:
        """Drop a stack's alignment (and its average): back to the frames as loaded."""
        if stack.alignment is None:
            return
        stack.signal = stack.original_signal
        stack.alignment = stack.aligned_signal = stack.original_signal = None
        stack.ops.append(_op({"op": "unalign"}))
        self._view_changed(stack)
        for d in self._averages(stack):
            self._remove(d)
        if stack.origin == "dpc" and stack.dpc and stack.dpc.get("output") == "colour":
            for p in self._dpc_children(stack).values():
                self.remove_alignment(p)

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
