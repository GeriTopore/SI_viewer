"""Saving datasets to files (M7).

Data is saved as shown: an image, map, stack or spectrum with its filters
applied (a spectrum image's maps already follow its processing). A spectrum
image is saved as loaded (rebinned, if it was), with its processing settings
recorded in the metadata: applying them would mean a processed copy of the
whole cube.
"""

from __future__ import annotations

import io
import re
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

from ..session import Dataset, DatasetKind

# Formats per kind of dataset: (format, extension, description).
FORMATS: dict[DatasetKind, list[tuple[str, str, str]]] = {
    DatasetKind.IMAGE: [
        ("tiff", "tif", "TIFF (32-bit float, calibrated)"),
        ("hspy", "hspy", "HyperSpy (.hspy)"),
        ("csv", "csv", "CSV (pixel values)"),
    ],
    DatasetKind.STACK: [
        ("tiff", "tif", "TIFF stack (32-bit float, calibrated)"),
        ("hspy", "hspy", "HyperSpy (.hspy)"),
    ],
    DatasetKind.SPECTRUM: [
        ("msa", "msa", "EMSA/MSA spectrum"),
        ("csv", "csv", "CSV (energy, counts)"),
        ("hspy", "hspy", "HyperSpy (.hspy)"),
    ],
    DatasetKind.SPECTRUM_IMAGE: [("hspy", "hspy", "HyperSpy (.hspy)")],
}


def safe_name(title: str) -> str:
    """A file name from a title, e.g. "Fe Kα (fit)" → "Fe_Kα_fit"."""
    return re.sub(r"[^\w.-]+", "_", title).strip("_") or "dataset"


def _processed_signal(ds: Dataset):
    """A copy of the dataset's signal holding the data as shown."""
    from .frames import get_frame, get_spectrum
    from .filters import filter_image

    ds.materialise()
    signal = ds.signal.deepcopy()
    f = ds.filter_spec
    if ds.kind is DatasetKind.IMAGE:
        signal.data = get_frame(ds)
    elif ds.kind is DatasetKind.STACK and f.has_spatial:
        signal.data = np.stack([filter_image(np.asarray(frame, np.float32), f) for frame in signal.data])
    elif ds.kind is DatasetKind.SPECTRUM and f.has_spectral:
        signal.data = get_spectrum(ds).astype(np.float64)
    description = f.describe()
    if description:
        key = "applied" if ds.kind is not DatasetKind.SPECTRUM_IMAGE else "not_applied"
        signal.metadata.set_item(f"Processing.si_viewer.filters_{key}", description)
    if ds.rebinned:
        signal.metadata.set_item("Processing.si_viewer.rebinned", "; then ".join(ds.rebinned))
    if ds.insitu is not None:
        view = ds.insitu.view
        signal.metadata.set_item("Processing.si_viewer.insitu_view", view.label())
        times = ds.insitu.times
        signal.metadata.set_item("Processing.si_viewer.insitu_time_s", [float(times[view.start]), float(times[view.end])])
    return signal


def _iso_date(signal) -> None:
    """EMSA needs an ISO date (Velox and exSpy data may have e.g. 14.10.2014)."""
    date = signal.metadata.get_item("General.date")
    if not date:
        return
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            signal.metadata.General.date = datetime.strptime(str(date), fmt).strftime("%Y-%m-%d")
            return
        except ValueError:
            continue
    del signal.metadata.General.date


def export_bytes(ds: Dataset, fmt: str) -> tuple[bytes, str]:
    """The dataset saved as ``fmt``: (file contents, file name)."""
    known = {f: ext for f, ext, _ in FORMATS.get(ds.kind, [])}
    if fmt not in known:
        raise ValueError(f"{ds.title} can't be saved as {fmt} (choose from {', '.join(known) or 'nothing'})")
    frame = f" ({ds.insitu.view.label()})" if ds.insitu is not None else ""  # which frame of an in-situ series
    name = f"{safe_name(ds.title + frame)}.{known[fmt]}"
    signal = _processed_signal(ds)

    if fmt == "csv":
        buf = io.StringIO()
        if ds.kind is DatasetKind.SPECTRUM:
            axis = signal.axes_manager.signal_axes[0]
            buf.write(f"{axis.name or 'Energy'} ({axis.units}),counts\n")
            np.savetxt(buf, np.column_stack([axis.axis, np.ravel(signal.data)]), delimiter=",", fmt="%.10g")
        else:
            np.savetxt(buf, np.asarray(signal.data), delimiter=",", fmt="%.8g")
        return buf.getvalue().encode(), name

    if fmt == "tiff":
        signal.change_dtype("float32")
    if fmt == "msa":
        _iso_date(signal)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        signal.save(path, overwrite=True)
        return path.read_bytes(), name


def maps_zip(maps: list[Dataset], fmt: str = "tiff") -> bytes:
    """Every map saved as ``fmt`` in one zip file (names made unique)."""
    buf = io.BytesIO()
    used: set[str] = set()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for ds in maps:
            data, name = export_bytes(ds, fmt)
            stem, ext = name.rsplit(".", 1)
            unique, k = name, 2
            while unique in used:
                unique, k = f"{stem}_{k}.{ext}", k + 1
            used.add(unique)
            z.writestr(unique, data)
    return buf.getvalue()
