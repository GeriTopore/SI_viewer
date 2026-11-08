"""Which technique a spectrum or spectrum image comes from: EDS or EELS."""

from __future__ import annotations

# Where the technique was decided: the file's signal type, the Gatan DM tag
# ``Meta Data.Signal``, the look of the energy axis, or the user ("user" is set
# by Session.set_technique, not here).
SOURCES = ("metadata", "dm_tag", "energy_axis", "user")

# DM's own names for the signal of a spectrum, as RosettaSciIO reads them.
_DM_TAGS = {"EELS": "EELS", "X-ray": "EDS", "EDS": "EDS", "EDX": "EDS"}


def technique_of(signal) -> str | None:
    """The technique a signal's ``signal_type`` stands for (EDS_TEM, EDS_SEM,
    EDS → "EDS"; EELS → "EELS"; anything else None)."""
    signal_type = signal.metadata.Signal.get_item("signal_type", "") or ""
    if signal_type.startswith("EDS"):
        return "EDS"
    if signal_type == "EELS":
        return "EELS"
    return None


def signal_type_for(signal, technique: str | None) -> str:
    """The ``signal_type`` to give ``signal`` so that exSpy's class matches
    ``technique`` (EDS_TEM when the metadata has a TEM instrument, else EDS_SEM)."""
    if technique == "EELS":
        return "EELS"
    if technique == "EDS":
        return "EDS_TEM" if signal.metadata.get_item("Acquisition_instrument.TEM") is not None else "EDS_SEM"
    return ""


def _from_dm_tag(signal) -> str | None:
    """``ImageList.<n>.ImageTags.Meta Data.Signal`` of a DM file's tags."""
    images = signal.original_metadata.get_item("ImageList")
    if images is None or not hasattr(images, "keys"):
        return None
    for key in images.keys():
        value = signal.original_metadata.get_item(f"ImageList.{key}.ImageTags.Meta Data.Signal")
        if isinstance(value, str) and value.strip() in _DM_TAGS:
            return _DM_TAGS[value.strip()]
    return None


def _from_energy_axis(signal) -> str | None:
    """EELS for an "energy loss" axis; EDS for a keV axis starting near 0.
    An eV axis named just "Energy" could be either: not guessed."""
    axis = signal.axes_manager.signal_axes[0]
    name = str(getattr(axis, "name", "") or "")
    units = str(getattr(axis, "units", "") or "").strip()
    if "loss" in name.lower():
        return "EELS"
    if units.lower() == "kev":
        start = float(axis.offset)
        end = start + float(axis.scale) * (int(axis.size) - 1)
        if start >= -0.5 and end <= 40:
            return "EDS"
    return None


def detect_technique(signal) -> tuple[str | None, str | None]:
    """(technique, source) of a spectrum or spectrum image; other signals,
    and spectra that can't be told apart, give (None, None).

    In order: the metadata's ``signal_type``; the DM tag ``Meta Data.Signal``;
    the energy axis. A signal that already has some other signal type (e.g.
    "CL") is left alone.
    """
    from ..session import DatasetKind, classify

    if classify(signal) not in (DatasetKind.SPECTRUM, DatasetKind.SPECTRUM_IMAGE):
        return None, None
    signal_type = signal.metadata.Signal.get_item("signal_type", "") or ""
    found = technique_of(signal)
    if found is not None:
        return found, "metadata"
    if signal_type:
        return None, None
    for source, detect in (("dm_tag", _from_dm_tag), ("energy_axis", _from_energy_axis)):
        found = detect(signal)
        if found is not None:
            return found, source
    return None, None


def apply_technique(signal, technique: str | None) -> None:
    """Give ``signal`` the signal type of ``technique`` (in place: HyperSpy swaps
    the object's class, so exSpy's methods follow)."""
    signal.set_signal_type(signal_type_for(signal, technique))
