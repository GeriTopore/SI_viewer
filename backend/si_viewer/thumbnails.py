"""PNG thumbnails for the dataset list's icon view."""

from __future__ import annotations

import io
import re

import numpy as np
from PIL import Image, ImageDraw

from .session import Dataset, DatasetKind

# Images shown for a spectrum image, in order of preference (matched on title).
_SI_PREVIEW_TITLES = [r"^HAADF", r"^ADF", r"^DF", r"^BF", r"^ABF"]

_SPECTRUM_FILL = (76, 141, 255, 90)
_SPECTRUM_LINE = (76, 141, 255, 255)


def make_thumbnail(dataset: Dataset, siblings: list[Dataset], size: int = 256) -> bytes:
    """Render ``dataset`` as a PNG no larger than ``size`` × ``size``."""
    kind = dataset.kind
    if kind is DatasetKind.SPECTRUM:
        img = _spectrum_image(_to_numpy(dataset.signal.data), size)
    else:
        img = _gray_image(_preview_array(dataset, siblings), size)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _preview_array(dataset: Dataset, siblings: list[Dataset]) -> np.ndarray:
    from .processing.filters import filter_image

    data = dataset.signal.data
    if dataset.kind is DatasetKind.IMAGE:
        return filter_image(_to_numpy(data), dataset.filter_spec)
    if dataset.kind is DatasetKind.STACK:
        return filter_image(_to_numpy(data.mean(axis=0)), dataset.filter_spec)
    if dataset.insitu is not None:
        # The frame held (the first one until another is shown): one frame
        # read, never the series.
        dataset.materialise()
        data = dataset.signal.data
        return filter_image(np.asarray(data, dtype=np.float32).sum(axis=-1), dataset.filter_spec)
    if dataset.is_line_scan:
        from .processing.frames import get_frame

        dataset.materialise()
        return get_frame(dataset)  # the line-scan image: positions × energy
    if dataset.kind is DatasetKind.SPECTRUM_IMAGE:
        companion = si_companion_image(dataset, siblings)
        if companion is not None:
            return filter_image(_to_numpy(companion.signal.data), companion.filter_spec)
        return filter_image(_to_numpy(data.sum(axis=-1)), dataset.filter_spec)  # total counts map
    # Fallback for other dimensionalities: collapse down to 2D.
    arr = _to_numpy(data)
    while arr.ndim > 2:
        arr = arr.sum(axis=0)
    return np.atleast_2d(arr)


def si_companion_image(si: Dataset, siblings: list[Dataset]) -> Dataset | None:
    """The ADF-type image acquired with a spectrum image, if the file has one
    (on the scan's grid, or a whole multiple finer: a DM "sub-scan")."""
    from .subscan import scan_factor_of

    if si.is_line_scan:
        return None
    candidates = [
        d
        for d in siblings
        if d.kind is DatasetKind.IMAGE and not d.is_element_map and scan_factor_of(si, d) is not None
    ]
    for pattern in _SI_PREVIEW_TITLES:
        for d in candidates:
            if re.match(pattern, d.title, re.IGNORECASE):
                return d
    return None


def _to_numpy(data) -> np.ndarray:
    if hasattr(data, "compute"):
        data = data.compute()
    return np.asarray(data, dtype=np.float32)


def _gray_image(arr: np.ndarray, size: int) -> Image.Image:
    finite = arr[np.isfinite(arr)]
    lo, hi = np.percentile(finite, (0.5, 99.5)) if finite.size else (0.0, 1.0)
    if hi <= lo:
        hi = lo + 1
    scaled = np.clip((arr - lo) / (hi - lo), 0, 1)
    img = Image.fromarray(np.nan_to_num(scaled * 255).astype(np.uint8), mode="L")
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    return img


def _spectrum_image(spectrum: np.ndarray, size: int) -> Image.Image:
    """A filled sparkline of the spectrum, trimmed to the range with counts.

    Drawn on a log scale so minor peaks stay visible next to a dominant one.
    """
    spectrum = np.clip(np.nan_to_num(spectrum.ravel()), 0, None)
    if spectrum.max() > 0:
        above = np.nonzero(spectrum > 1e-3 * spectrum.max())[0]
        spectrum = spectrum[: above[-1] + 1]
    spectrum = np.log1p(spectrum)
    width, height = size, size // 2
    # Max-pool into one value per pixel column so narrow peaks survive.
    edges = np.linspace(0, len(spectrum), width + 1).astype(int)
    cols = np.array([spectrum[a:max(b, a + 1)].max() for a, b in zip(edges[:-1], edges[1:])])
    peak = cols.max() or 1.0
    ys = height - 1 - (cols / peak) * (height - 4)
    points = list(zip(range(width), ys.tolist()))

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.polygon([(0, height - 1), *points, (width - 1, height - 1)], fill=_SPECTRUM_FILL)
    draw.line(points, fill=_SPECTRUM_LINE, width=2)
    return img
