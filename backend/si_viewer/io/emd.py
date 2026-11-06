"""Loading of ThermoFisher Velox ``.emd`` files.

RosettaSciIO only recognises Velox files whose ``Version`` dataset reports the
format ``"Velox"`` or ``"DevelopersKit"``. Recent Velox releases (file version
13) write ``"VeloxLite"``, which the reader then rejects with
"The file is not a supported EMD file." although its existing Velox reader
handles the content fine. :func:`ensure_velox_lite_support` widens the check.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import h5py

_logger = logging.getLogger(__name__)

VELOX_FORMATS = ("Velox", "DevelopersKit", "VeloxLite")

_patched = False


def velox_format(file: str | Path | h5py.File) -> str | None:
    """Return the format string of a Velox EMD file, or None if not Velox-like."""
    if isinstance(file, (str, Path)):
        with h5py.File(file, "r") as f:
            return velox_format(f)
    if "Version" not in file:
        return None
    try:
        info = json.loads(file["Version"][0].decode("utf-8"))
    except (ValueError, KeyError, AttributeError):
        return None
    fmt = info.get("format")
    return fmt if fmt in VELOX_FORMATS else None


def ensure_velox_lite_support() -> None:
    """Make RosettaSciIO's EMD reader accept VeloxLite files (idempotent)."""
    global _patched
    if _patched:
        return
    import rsciio.emd._api as emd_api

    original = emd_api.is_EMD_Velox

    def is_EMD_Velox(file):
        return original(file) or velox_format(file) is not None

    emd_api.is_EMD_Velox = is_EMD_Velox
    _patched = True
    _logger.debug("Patched rsciio is_EMD_Velox to accept %s", VELOX_FORMATS)


def load_emd(path: str | Path, lazy: bool = True) -> list:
    """Load every signal in a Velox EMD file.

    Returns the signals from a standard load, plus any multi-frame image
    stacks (e.g. the HAADF frames acquired alongside an EDS spectrum image),
    which RosettaSciIO only returns when ``load_SI_image_stack=True``.
    """
    import hyperspy.api as hs

    ensure_velox_lite_support()
    path = str(path)
    signals = _as_list(hs.load(path, lazy=lazy))

    if velox_format(path) is not None and _has_multiframe_images(path):
        images = _as_list(
            hs.load(path, lazy=lazy, select_type="images", load_SI_image_stack=True)
        )
        signals.extend(s for s in images if s.axes_manager.navigation_dimension == 1)
    return signals


def _has_multiframe_images(path: str) -> bool:
    with h5py.File(path, "r") as f:
        group = f.get("Data/Image")
        if group is None:
            return False
        return any(
            "Data" in group[key] and group[key]["Data"].ndim == 3
            and group[key]["Data"].shape[-1] > 1
            for key in group
        )


def _as_list(result) -> list:
    return result if isinstance(result, list) else [result]
