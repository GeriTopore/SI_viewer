"""File loading. :func:`load_file` dispatches on the file type."""

from __future__ import annotations

from pathlib import Path

from .emd import load_emd

SUPPORTED_SUFFIXES = (".emd", ".hspy", ".dm3", ".dm4", ".bcf", ".tif", ".tiff")


def load_file(path: str | Path, lazy: bool = True) -> list:
    """Load all signals in ``path`` as a list of HyperSpy signals."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".emd":
        return load_emd(path, lazy=lazy)

    if path.suffix.lower() == ".dm4":
        from .insitu import load_insitu

        # An in-situ series (.dm4 + .raw): always lazy, one chunk per frame.
        series = load_insitu(path)
        if series is not None:
            return series

    import hyperspy.api as hs

    result = hs.load(str(path), lazy=lazy)
    return result if isinstance(result, list) else [result]
