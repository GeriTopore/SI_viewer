"""Low-loss / core-loss pairs of EELS spectrum images (M10c): which role a
spectrum image has, whether two can be linked, and the signs that they belong
together. The links themselves live in ``Session.eels_links``."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

ROLES = ("low_loss", "core_loss")

# The zero-loss peak counts when the sum spectrum peaks within this many eV
# (or three channels) of 0.
ZLP_TOLERANCE_EV = 3.0

# Navigation scales that differ by less than this (relative) are the same.
SCALE_TOLERANCE = 0.01

_TO_EV = {"ev": 1.0, "kev": 1e3, "mev": 1e6}


def is_eels_si(ds) -> bool:
    """An EELS spectrum image (a line scan too)."""
    from ..session import DatasetKind
    from .technique import technique_of

    return ds.kind is DatasetKind.SPECTRUM_IMAGE and technique_of(ds.signal) == "EELS"


def _energy_axis_ev(signal) -> tuple[float, float, int]:
    """(offset, scale, size) of the energy axis in eV."""
    axis = signal.axes_manager.signal_axes[0]
    factor = _TO_EV.get(str(axis.units).strip().lower(), 1.0)
    return float(axis.offset) * factor, float(axis.scale) * factor, int(axis.size)


def role_of(ds) -> str | None:
    """"low_loss" when the energy range contains 0 eV and the sum spectrum
    peaks near it (the zero-loss peak), "core_loss" for any other EELS spectrum
    image, None for everything else. The sum spectrum is only read when 0 eV
    is in range."""
    from .frames import get_spectrum

    if not is_eels_si(ds):
        return None
    offset, scale, size = _energy_axis_ev(ds.signal)
    if scale > 0 and offset <= 0 <= offset + scale * (size - 1):
        spectrum = get_spectrum(ds)
        peak_ev = offset + scale * int(np.argmax(spectrum))
        if abs(peak_ev) <= max(ZLP_TOLERANCE_EV, 3 * scale):
            return "low_loss"
    return "core_loss"


def _number(signal, path: str) -> float | None:
    value = signal.metadata.get_item(path, None)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def eels_info(ds, partner_id: str | None) -> dict | None:
    """The ``eels`` part of a dataset's summary (None unless an EELS spectrum image)."""
    if not is_eels_si(ds):
        return None
    s = ds.signal
    return {
        "role": ds.eels_role,
        "partner_id": partner_id,
        "beam_energy": _number(s, "Acquisition_instrument.TEM.beam_energy"),
        "convergence_angle": _number(s, "Acquisition_instrument.TEM.convergence_angle"),
        "collection_angle": _number(s, "Acquisition_instrument.TEM.Detector.EELS.collection_angle"),
    }


# ---------- compatibility ----------


def _defined(value) -> str:
    return value if isinstance(value, str) and value != "None" else ""


def compatible(a, b) -> bool:
    """One low loss and one core loss, same navigation shape, navigation
    scales within 1 % (same units), and the same beam energy where both give one."""
    if not (is_eels_si(a) and is_eels_si(b)) or a is b:
        return False
    if {a.eels_role, b.eels_role} != set(ROLES):
        return False
    am, bm = a.signal.axes_manager, b.signal.axes_manager
    if tuple(am.navigation_shape) != tuple(bm.navigation_shape):
        return False
    for x, y in zip(am.navigation_axes, bm.navigation_axes):
        if _defined(x.units) != _defined(y.units):
            return False
        if not math.isclose(float(x.scale), float(y.scale), rel_tol=SCALE_TOLERANCE):
            return False
    ea = _number(a.signal, "Acquisition_instrument.TEM.beam_energy")
    eb = _number(b.signal, "Acquisition_instrument.TEM.beam_energy")
    if ea is not None and eb is not None and not math.isclose(ea, eb, rel_tol=1e-3):
        return False
    return True


# ---------- signs that two spectrum images belong together ----------


def _dm_values(signal) -> tuple[set, set]:
    """(set of (date, start time), set of survey image IDs) in a DM file's
    ``ImageList`` tags (empty sets for other files)."""
    acquisitions, surveys = set(), set()
    om = signal.original_metadata
    try:
        images = om.get_item("ImageList")
        keys = list(images.keys()) if hasattr(images, "keys") else []
    except Exception:
        return acquisitions, surveys
    for key in keys:
        base = f"ImageList.{key}.ImageTags.SI.Acquisition"
        try:
            date = om.get_item(f"{base}.Date", None)
            start = om.get_item(f"{base}.Start time", None)
            survey = om.get_item(f"{base}.Survey Image.Unique Image ID", None)
        except Exception:
            continue
        if isinstance(date, str) and isinstance(start, str) and date.strip() and start.strip():
            acquisitions.add((date.strip(), start.strip()))
        if survey is not None and str(survey).strip() not in ("", "0"):
            surveys.add(str(survey).strip())
    return acquisitions, surveys


_LOSS_TOKENS = re.compile(
    r"(?:low|high|core)[\s_\-]*loss"
    r"|(?<![a-z0-9])(?:ll|hl|cl)(?![a-z0-9])"
)


def normalised_name(name: str) -> str:
    """``name`` lowercased, without low-/high-/core-loss (and standalone LL,
    HL, CL) tokens and without anything but letters and digits."""
    name = _LOSS_TOKENS.sub("", name.lower())
    return re.sub(r"[^a-z0-9]", "", name)


def link_sign(a, path_a: Path, b, path_b: Path) -> str | None:
    """Why ``a`` and ``b`` are probably one experiment: "acquisition" (same DM
    acquisition date and start time), "survey" (same survey image ID), "name"
    (titles or file names equal but for low-/high-loss tokens); None if there
    is no such sign."""
    acq_a, survey_a = _dm_values(a.signal)
    acq_b, survey_b = _dm_values(b.signal)
    if acq_a & acq_b:
        return "acquisition"
    if survey_a & survey_b:
        return "survey"
    pairs = ((a.title, b.title), (Path(path_a).stem, Path(path_b).stem))
    for name_a, name_b in pairs:
        n = normalised_name(name_a)
        if n and n == normalised_name(name_b):
            return "name"
    return None
