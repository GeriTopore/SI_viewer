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
        # Only the stacks the standard load gave as a single frame (those of a
        # spectrum image): a file of image stacks alone has them all already.
        have = {(s.metadata.General.title, tuple(s.data.shape)) for s in signals}
        images = _as_list(
            hs.load(path, lazy=lazy, select_type="images", load_SI_image_stack=True)
        )
        signals.extend(
            s for s in images
            if s.axes_manager.navigation_dimension == 1
            and (s.metadata.General.title, tuple(s.data.shape)) not in have
        )
    if velox_format(path) is not None:
        _add_image_info(path, signals)
    return signals


def _display_labels(f: h5py.File) -> dict[str, str]:
    """The names Velox shows its images by (e.g. "DF-A" for the segments
    DFS(0,4)), from its saved display layout: image group name -> label."""
    out = {}
    group = f.get("Presentation/Displays/ImageDisplay")
    if group is None:
        return out
    for key in group:
        try:
            d = json.loads(group[key][0].decode("utf-8", "ignore").rstrip("\x00"))
        except (ValueError, KeyError, AttributeError, IndexError):
            continue
        label = (d.get("display") or {}).get("label") if isinstance(d.get("display"), dict) else None
        data_path = d.get("dataPath") or ""
        if label and data_path.startswith("/Data/Image/"):
            out[data_path.rsplit("/", 1)[1]] = str(label)
    return out


def _stem_inputs(f: h5py.File) -> dict[str, dict]:
    """What each image was acquired with, from Velox's acquisition record
    (``Operations/StemInputOperation``): image group name -> {"name": the
    detector, "segments": hardware segment numbers summed, "polarities"}."""
    out = {}
    group = f.get("Operations/StemInputOperation")
    if group is None:
        return out
    for key in group:
        try:
            d = json.loads(group[key][0].decode("utf-8", "ignore").rstrip("\x00"))
        except (ValueError, KeyError, AttributeError, IndexError):
            continue
        data_path = d.get("dataPath") or ""
        info = d.get("detectorInfo") or {}
        if not data_path.startswith("/Data/Image/") or not isinstance(info, dict):
            continue
        segments = info.get("segments") or []
        if not isinstance(segments, list):
            segments = []
        try:
            numbers = [int(s["index"]) for s in segments]
        except (KeyError, TypeError, ValueError):
            continue
        out[data_path.rsplit("/", 1)[1]] = {
            "name": str(info.get("name") or d.get("detector") or ""),
            "segments": numbers,
            "polarities": [str(s.get("polarity", "")) for s in segments],
        }
    return out


def _add_image_info(path: str, signals: list) -> None:
    """For the images of a Velox file: the title Velox shows (e.g. "DF-A"),
    and the detector segments each was summed from, as recorded at
    acquisition (``metadata.Acquisition_instrument.TEM.Detector``).
    RosettaSciIO keeps no link to the image group a signal came from, so each
    is matched by the top-left pixels of its first frame."""
    import numpy as np

    with h5py.File(path, "r") as f:
        labels = _display_labels(f)
        inputs = _stem_inputs(f)
        if not labels and not inputs:
            return
        corners = {}
        for key in set(labels) | set(inputs):
            data = f.get(f"Data/Image/{key}/Data")
            if data is not None and data.ndim == 3:
                corners[key] = _plain(np.asarray(data[:4, :4, 0]))
    for s in signals:
        if s.axes_manager.signal_dimension != 2 or s.axes_manager.navigation_dimension > 1:
            continue
        try:
            corner = np.asarray(s.data[0, :4, :4] if s.axes_manager.navigation_dimension == 1 else s.data[:4, :4])
        except Exception:  # noqa: BLE001 - not worth failing the load
            continue
        corner = _plain(corner)
        found = [k for k, c in corners.items() if c.shape == corner.shape and _same(c, corner)]
        if len(found) != 1:
            continue
        key = found[0]
        acq = inputs.get(key)
        if acq and acq["segments"]:
            s.metadata.set_item("Acquisition_instrument.TEM.Detector.name", acq["name"])
            s.metadata.set_item("Acquisition_instrument.TEM.Detector.segments", acq["segments"])
            s.metadata.set_item("Acquisition_instrument.TEM.Detector.segment_polarities", acq["polarities"])
        label = labels.get(key)
        if label and label != s.metadata.General.title:
            s.metadata.set_item("General.original_title", s.metadata.General.title)
            s.metadata.General.title = label


def _plain(a):
    """``a`` as a plain array: a structured one of a real and an imaginary
    field (Velox's complex images, e.g. a DPC) as complex."""
    import numpy as np

    names = a.dtype.names
    if names and len(names) == 2:
        return a[names[0]] + 1j * a[names[1]]
    return a


def _same(a, b) -> bool:
    import numpy as np

    try:
        return bool(np.array_equal(a, b))
    except TypeError:  # kinds that can't be compared: not the same image
        return False


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
