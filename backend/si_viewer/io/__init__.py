"""File loading. :func:`load_file` dispatches on the file type."""

from __future__ import annotations

from pathlib import Path

from .emd import load_emd

SUPPORTED_SUFFIXES = (
    ".emd", ".hspy", ".dm3", ".dm4", ".bcf", ".tif", ".tiff",
    # 4D-STEM: py4DSTEM (emdfile) .h5, Quantum Detectors Merlin, blockfiles.
    ".h5", ".hdf5", ".mib", ".blo",
)


def load_file(path: str | Path, lazy: bool = True) -> list:
    """Load all signals in ``path`` as a list of HyperSpy signals."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".emd":
        return load_emd(path, lazy=lazy)

    if suffix in (".h5", ".hdf5"):
        from .py4dstem import is_emdfile, load_py4dstem

        if is_emdfile(path):
            return [name_4d_axes(s) for s in load_py4dstem(path, lazy=lazy)]

    if suffix == ".dm4":
        from .insitu import load_insitu

        # An in-situ series (.dm4 + .raw): always lazy, one chunk per frame.
        series = load_insitu(path)
        if series is not None:
            return series

    import hyperspy.api as hs

    try:
        result = hs.load(str(path), lazy=lazy)
    except ValueError as e:
        if "multiple file readers" in str(e):
            raise ValueError(f"{path.name}: this kind of HDF5 file can't be read yet") from None
        raise
    return [name_4d_axes(s) for s in (result if isinstance(result, list) else [result])]


def name_4d_axes(signal):
    """A 4D-STEM signal (two scan and two diffraction axes) with its axes named
    x, y (scan) and kx, ky (pattern): readers name them inconsistently (a DM
    4D-STEM file's scan axes come as "<undefined>" and "z")."""
    am = signal.axes_manager
    if am.navigation_dimension == 2 and am.signal_dimension == 2:
        for axis, name in zip((*am.navigation_axes, *am.signal_axes), ("x", "y", "kx", "ky")):
            axis.name = name
    return signal
