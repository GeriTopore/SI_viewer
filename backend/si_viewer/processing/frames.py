"""2D frames and 1D spectra extracted from datasets for display."""

from __future__ import annotations

import numpy as np

from ..session import Dataset, DatasetKind
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
    """
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
        frame = _cached(ds, "total-counts", lambda: _numpy(ds.signal.data.sum(axis=-1)))
        if f.has_spatial:
            frame = _cached(ds, "total-counts|filtered", lambda: filter_image(frame, f))
    else:
        raise NotApplicable(f"{ds.kind.value} datasets have no 2D frame")
    return bin2d(frame, bin).astype(np.float32, copy=False)


def get_spectrum(ds: Dataset) -> np.ndarray:
    """The spectrum of a spectrum dataset, or the sum spectrum of a spectrum
    image, with its spectral smoothing applied."""
    if ds.kind is DatasetKind.SPECTRUM:
        raw = _cached(ds, "spectrum", lambda: _numpy(ds.signal.data).ravel())
    elif ds.kind is DatasetKind.SPECTRUM_IMAGE:
        ds.materialise()
        raw = _cached(
            ds, "sum-spectrum",
            lambda: ds.signal.data.sum(axis=(0, 1), dtype=np.float64).astype(np.float32),
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
    images of the same scan shape (e.g. Velox element maps).
    """
    from ..thumbnails import si_companion_image

    if si.kind is not DatasetKind.SPECTRUM_IMAGE:
        raise NotApplicable("Only spectrum images have navigators")
    shape = tuple(si.signal.axes_manager.navigation_shape)
    companion = si_companion_image(si, siblings)
    others = [
        d
        for d in siblings
        if d.kind is DatasetKind.IMAGE
        and d is not companion
        and tuple(d.signal.axes_manager.signal_shape) == shape
    ]
    sources = []
    if companion is not None:
        sources.append({"id": companion.id, "label": companion.title})
    sources.append({"id": si.id, "label": "Total counts"})
    sources.extend({"id": d.id, "label": d.title} for d in others)
    return sources


def _cached(ds: Dataset, key: str, compute):
    if key not in ds.cache:
        ds.cache[key] = compute()
    return ds.cache[key]
