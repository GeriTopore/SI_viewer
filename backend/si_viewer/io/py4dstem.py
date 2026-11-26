"""py4DSTEM's files (emdfile ``.h5``): its data cubes, and its 2D arrays
(virtual images, diffraction slices, probes), as HyperSpy signals. Read with
h5py, without py4DSTEM.

An emdfile holds one or more root groups (``emd_group_type`` "root"), each
with a ``metadatabundle/calibration`` (py4DSTEM's ``Calibration``: pixel sizes
and units of real and reciprocal space) and arrays (``emd_group_type``
"array": a ``data`` dataset with ``dim0``… vectors). A ``DataCube`` is
indexed ``[Rx, Ry, Qx, Qy]``; it becomes a signal with Rx as its scan's rows
(y) and Ry as its columns (x), Qx as the pattern's rows (ky) and Qy as its
columns (kx), so images look as py4DSTEM and its browser draw them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Data cubes up to this size are read into RAM; larger ones stay in the file (lazy).
IN_MEMORY_LIMIT = 4 << 30
# py4DSTEM's unit names, as the app writes them.
UNITS = {"A": "Å", "A^-1": "1/Å", "nm": "nm", "nm^-1": "1/nm", "mrad": "mrad", "pixels": "px", "pixel": "px"}
# Classes of 2D arrays in real space (the rest are in reciprocal space).
REAL_SPACE = {"RealSlice", "VirtualImage"}


def is_emdfile(path: str | Path) -> bool:
    """Whether ``path`` is an HDF5 file written by emdfile (py4DSTEM)."""
    import h5py

    try:
        with h5py.File(path, "r") as f:
            if f.attrs.get("authoring_program") == "emdfile" or f.attrs.get("emd_group_type") == "file":
                return True
            return any(g.attrs.get("emd_group_type") == "root" for g in f.values() if isinstance(g, h5py.Group))
    except OSError:
        return False


def _text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _text(value[()])
    return str(value)


def _calibration(root) -> dict:
    """The root's py4DSTEM calibration: every number and text in it."""
    import h5py

    group = root.get("metadatabundle/calibration")
    out = {}
    if group is None:
        return out
    for name, item in group.items():
        if not isinstance(item, h5py.Dataset) or name.startswith("_"):
            continue
        value = item[()]
        if isinstance(value, (bytes, str)):
            out[name] = _text(value)
        elif np.ndim(value) == 0:
            out[name] = value.item() if hasattr(value, "item") else value
        else:
            out[name] = np.asarray(value).tolist()
    return out


def _space(cal: dict, which: str) -> tuple[float, str]:
    """Pixel size and units of real ("R") or reciprocal ("Q") space."""
    size = cal.get(f"{which}_pixel_size", 1)
    units = UNITS.get(cal.get(f"{which}_pixel_units", "pixels"), cal.get(f"{which}_pixel_units", "px"))
    try:
        size = float(size)
    except (TypeError, ValueError):
        size = 1.0
    return (size if size > 0 else 1.0), units


def _array(dataset, lazy: bool, keep_open: list):
    """The array of an h5py dataset: in RAM, or (large and ``lazy``) a dask
    array reading from the file, which is then kept open."""
    if not lazy or dataset.nbytes <= IN_MEMORY_LIMIT:
        return dataset[()]
    import dask.array as da

    keep_open.append(dataset.file)
    chunks = (1, *dataset.shape[1:]) if dataset.ndim == 4 else "auto"
    return da.from_array(dataset, chunks=chunks)


def _set_axes(signal, names, sizes_units) -> None:
    """Name and calibrate the signal's axes, in array order."""
    for axis, name, (scale, units) in zip(signal.axes_manager._axes, names, sizes_units):
        axis.name, axis.scale, axis.offset, axis.units = name, scale, 0.0, units


def load_py4dstem(path: str | Path, lazy: bool = True) -> list:
    """The data cubes and 2D arrays of a py4DSTEM file, as HyperSpy signals
    (each titled with its name in the file). ValueError if it holds none."""
    import h5py
    import hyperspy.api as hs

    path = Path(path)
    f = h5py.File(path, "r")
    keep_open: list = []
    signals = []
    try:
        roots = [g for g in f.values() if isinstance(g, h5py.Group) and g.attrs.get("emd_group_type") == "root"]
        for root in roots:
            cal = _calibration(root)
            r_space, q_space = _space(cal, "R"), _space(cal, "Q")
            arrays = []
            root.visititems(lambda name, g: arrays.append((name, g)) if isinstance(g, h5py.Group)
                            and g.attrs.get("emd_group_type") == "array" and "data" in g else None)
            for name, group in arrays:
                data = group["data"]
                kind = _text(group.attrs.get("python_class", ""))
                title = name.split("/")[-1]
                if data.ndim == 4:
                    signal = hs.signals.Signal2D(_array(data, lazy, keep_open))
                    if not isinstance(signal.data, np.ndarray):
                        signal = signal.as_lazy()
                    # Array order: Rx (y), Ry (x), Qx (ky), Qy (kx).
                    _set_axes(signal, ("y", "x", "ky", "kx"), (r_space, r_space, q_space, q_space))
                elif data.ndim == 2 and data.dtype.kind in "biuf":
                    signal = hs.signals.Signal2D(data[()])
                    space = r_space if kind in REAL_SPACE else q_space
                    names = ("y", "x") if kind in REAL_SPACE else ("ky", "kx")
                    _set_axes(signal, names, (space, space))
                else:
                    continue  # Bragg vectors, point lists, stacks of slices: not yet
                signal.metadata.General.title = title
                signal.metadata.General.original_filename = path.name
                signal.metadata.set_item("Signal.quantity", "Intensity")
                signal.metadata.set_item("py4DSTEM.python_class", kind)
                signal.metadata.set_item("py4DSTEM.path", f"{root.name}/{name}")
                if cal:
                    signal.metadata.set_item("py4DSTEM.calibration", cal)
                signals.append(signal)
    finally:
        if not keep_open:
            f.close()
    if not signals:
        raise ValueError(f"{path.name} has no data cube or 2D array that can be read")
    return signals
