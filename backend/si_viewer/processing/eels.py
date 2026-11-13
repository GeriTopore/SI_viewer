"""Linked EELS spectrum images (M10c): which role (low loss / core loss) a
spectrum image has and whether two can be linked. Only the user links them;
the link groups live in ``Session.eels_groups``."""

from __future__ import annotations

import math

ROLES = ("low_loss", "core_loss")

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


def _dm_tag(signal, tail: str) -> list:
    """The value of ``tail`` (a path below ``ImageList.<key>``) for every image
    of a DM file's tags that has it (nothing for other files)."""
    om = signal.original_metadata
    try:
        images = om.get_item("ImageList")
        keys = list(images.keys()) if hasattr(images, "keys") else []
    except Exception:
        return []
    values = []
    for key in keys:
        try:
            value = om.get_item(f"ImageList.{key}.{tail}", None)
        except Exception:
            continue
        if value is not None:
            values.append(value)
    return values


def energy_offset(signal) -> float:
    """The first energy channel in eV."""
    return _energy_axis_ev(signal)[0]


def role_of(ds) -> tuple[str | None, str | None]:
    """(role, source) of an EELS spectrum image: "low_loss" / "core_loss" from
    the file's metadata ("metadata": DM's ``Is dual acquire low-loss`` tag), or
    "core_loss" by default ("default") when there is no marker. (None, None)
    for everything but an EELS spectrum image. Only tags are read."""
    if not is_eels_si(ds):
        return None, None
    for value in _dm_tag(ds.signal, "ImageTags.EELS.Acquisition.Is dual acquire low-loss"):
        try:
            return ("low_loss" if int(value) else "core_loss"), "metadata"
        except (TypeError, ValueError):
            continue
    return "core_loss", "default"


def _number(signal, path: str) -> float | None:
    value = signal.metadata.get_item(path, None)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def eels_info(ds) -> dict | None:
    """The ``eels`` part of a dataset's summary (None unless an EELS spectrum
    image): its role and where it came from, and the ids of the others in its
    link group (low loss first, then core losses by energy offset)."""
    if not is_eels_si(ds):
        return None
    s = ds.signal
    return {
        "role": ds.eels_role,
        "role_source": ds.eels_role_source,
        "linked": list(ds.eels_linked),
        "beam_energy": _number(s, "Acquisition_instrument.TEM.beam_energy"),
        "convergence_angle": _number(s, "Acquisition_instrument.TEM.convergence_angle"),
        "collection_angle": _number(s, "Acquisition_instrument.TEM.Detector.EELS.collection_angle"),
    }


def microscope_info(signal) -> dict:
    """The microscope parameters an EELS quantification needs (None where
    the metadata has none): beam energy (kV), convergence and collection
    angles (mrad)."""
    return {
        "beam_energy": _number(signal, "Acquisition_instrument.TEM.beam_energy"),
        "convergence_angle": _number(signal, "Acquisition_instrument.TEM.convergence_angle"),
        "collection_angle": _number(signal, "Acquisition_instrument.TEM.Detector.EELS.collection_angle"),
    }


# ---------- compatibility ----------


def _defined(value) -> str:
    return value if isinstance(value, str) and value != "None" else ""


def compatible(a, b) -> bool:
    """Whether two EELS spectrum images can be linked, whatever their roles:
    same navigation shape, navigation scales within 1 % (same units), and the
    same beam energy where both give one."""
    if not (is_eels_si(a) and is_eels_si(b)) or a is b:
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
