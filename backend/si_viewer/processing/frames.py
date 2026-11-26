"""2D frames and 1D spectra extracted from datasets for display."""

from __future__ import annotations

import numpy as np

from ..insitu_series import ADF_SOURCE_ID, adf_stack
from ..session import Dataset, DatasetKind, si_cube
from .filters import filter_image, filter_spectrum

# Stacks smaller than this are loaded into RAM on first access so scrubbing is
# instant; larger ones are read frame by frame.
STACK_IN_MEMORY_LIMIT = 1 << 30


class NotApplicable(ValueError):
    """The requested view doesn't exist for this kind of dataset."""


def bin2d(arr: np.ndarray, factor: int) -> np.ndarray:
    """Mean-bin the last two axes by ``factor``, dropping any remainder rows/cols."""
    if factor <= 1:
        return arr
    h, w = arr.shape[-2:]
    h2, w2 = h // factor * factor, w // factor * factor
    arr = arr[..., :h2, :w2]
    return arr.reshape(*arr.shape[:-2], h2 // factor, factor, w2 // factor, factor).mean(
        axis=(-3, -1)
    )


def _numpy(data) -> np.ndarray:
    if hasattr(data, "compute"):
        data = data.compute()
    return np.asarray(data, dtype=np.float32)


def get_frame(
    ds: Dataset, index: int = 0, projection: str | None = None, bin: int = 1
) -> np.ndarray:
    """The 2D array to display for ``ds``, with its spatial filter applied.

    - image: the image
    - stack: frame ``index``, or the ``"mean"``/``"sum"`` projection
    - spectrum image: the total-counts map
    - line scan: the line-scan image itself, positions (rows) × energy
      channels (columns), spectrally smoothed if asked; ``bin`` is ignored
    - 4D-STEM: ``"scan"`` (default) the total intensity at every scan
      position; ``"mean-pattern"`` and ``"max-pattern"`` the mean and
      maximum diffraction pattern over the scan
    """
    if ds.kind is DatasetKind.DIFFRACTION4D:
        return bin2d(_frame_4d(ds, projection or "scan"), bin).astype(np.float32, copy=False)
    f = ds.filter_spec
    if ds.kind is DatasetKind.IMAGE:
        frame = _cached(ds, "image", lambda: _numpy(ds.signal.data))
        if f.has_spatial:
            frame = _cached(ds, "image|filtered", lambda: filter_image(frame, f))
    elif ds.kind is DatasetKind.STACK:
        if ds.lazy and ds.signal.data.nbytes <= STACK_IN_MEMORY_LIMIT:
            ds.materialise()
        if projection in ("mean", "sum"):
            frame = _cached(
                ds, f"stack-{projection}",
                lambda: _numpy(getattr(ds.signal.data, projection)(axis=0)),
            )
            if f.has_spatial:
                frame = _cached(ds, f"stack-{projection}|filtered", lambda: filter_image(frame, f))
        else:
            n = ds.signal.axes_manager.navigation_size
            if not 0 <= index < n:
                raise IndexError(f"Frame {index} out of range 0–{n - 1}")
            frame = filter_image(_numpy(ds.signal.data[index]), f)
    elif ds.kind is DatasetKind.SPECTRUM_IMAGE:
        ds.materialise()
        if ds.is_line_scan:
            raw = _cached(ds, "line-scan", lambda: _numpy(ds.signal.data))
            if not f.has_spectral:
                return raw
            return _cached(ds, "line-scan|smoothed", lambda: filter_spectrum(raw, f).astype(np.float32))
        frame = _cached(ds, "total-counts", lambda: _numpy(ds.signal.data.sum(axis=-1)))
        if f.has_spatial:
            frame = _cached(ds, "total-counts|filtered", lambda: filter_image(frame, f))
    else:
        raise NotApplicable(f"{ds.kind.value} datasets have no 2D frame")
    return bin2d(frame, bin).astype(np.float32, copy=False)


FOUR_D_FRAMES = {
    # Array axes summed or reduced: the data is [y, x, ky, kx].
    "scan": lambda data: data.sum(axis=(-2, -1), dtype=np.float64),
    "mean-pattern": lambda data: data.mean(axis=(0, 1), dtype=np.float64),
    "max-pattern": lambda data: data.max(axis=(0, 1)),
}


def _frame_4d(ds: Dataset, projection: str) -> np.ndarray:
    """A 4D-STEM image: the scan's total intensity or the mean or maximum
    pattern (lazy data is read in full, once: the results are cached)."""
    reduce = FOUR_D_FRAMES.get(projection)
    if reduce is None:
        raise NotApplicable(f"4D-STEM data has no {projection!r} view (one of {', '.join(FOUR_D_FRAMES)})")
    return _cached(ds, f"4d-{projection}", lambda: _numpy(reduce(ds.signal.data)))


def get_spectrum(ds: Dataset) -> np.ndarray:
    """The spectrum of a spectrum dataset, or the sum spectrum of a spectrum
    image, with its spectral smoothing applied."""
    if ds.kind is DatasetKind.SPECTRUM:
        raw = _cached(ds, "spectrum", lambda: _numpy(ds.signal.data).ravel())
    elif ds.kind is DatasetKind.SPECTRUM_IMAGE:
        ds.materialise()
        raw = _cached(
            ds, "sum-spectrum",
            lambda: si_cube(ds).sum(axis=(0, 1), dtype=np.float64).astype(np.float32),
        )
    else:
        raise NotApplicable(f"{ds.kind.value} datasets have no spectrum")
    f = ds.filter_spec
    if not f.has_spectral:
        return raw
    return _cached(ds, f"{ds.kind.value}|smoothed", lambda: filter_spectrum(raw, f).astype(np.float32))


def navigator_sources(si: Dataset, siblings: list[Dataset]) -> list[dict]:
    """Images that can be shown as the navigator of a spectrum image.

    The companion ADF image comes first, then the total-counts map, then other
    images of the same scan shape (e.g. Velox element maps). An image sampling
    the same field of view k times more finely (a Gatan "sub-scan" ADF) is
    offered at its own resolution.

    Each source is ``{"id", "label", "factor", "dataset_id"}``: ``factor`` is
    k (1: the scan's own grid) and ``dataset_id`` the image dataset (its
    calibration is the navigator's); the series' simultaneous ADF is
    ``insitu-adf`` with the ADF stack as ``dataset_id``.
    """
    from ..subscan import scan_factor_of
    from ..thumbnails import si_companion_image

    if si.kind is not DatasetKind.SPECTRUM_IMAGE:
        raise NotApplicable("Only spectrum images have navigators")
    if si.is_line_scan:
        return [{"id": si.id, "label": "Line scan", "factor": 1, "dataset_id": si.id}]
    companion = si_companion_image(si, siblings)
    others = []
    for d in siblings:
        if d.kind is DatasetKind.IMAGE and d is not companion:
            k = scan_factor_of(si, d)
            if k is not None:
                others.append((d, k))
    sources = []
    if si.insitu is not None:
        # The simultaneous ADF of the frame (or range) the dataset holds.
        found = adf_stack(si, siblings)
        if found is not None:
            adf, k = found
            sources.append({"id": ADF_SOURCE_ID, "label": adf.title, "factor": k, "dataset_id": adf.id})
    if companion is not None:
        k = scan_factor_of(si, companion)
        sources.append({"id": companion.id, "label": companion.title, "factor": k, "dataset_id": companion.id})
    sources.append({"id": si.id, "label": "Total counts", "factor": 1, "dataset_id": si.id})
    sources.extend({"id": d.id, "label": d.title, "factor": k, "dataset_id": d.id} for d, k in others)
    return sources


def _cached(ds: Dataset, key: str, compute):
    # ``ds.cache`` is replaced (not cleared) when the data changes: a result
    # computed from the old data lands in the old dict and is never seen again.
    cache = ds.cache
    if key not in cache:
        cache[key] = compute()
    return cache[key]
