"""X-ray line data from exSpy's element database, and elements on EDS signals."""

from __future__ import annotations

from functools import cache


@cache
def element_table() -> list[dict]:
    """Every element in the database with its X-ray lines, ordered by Z."""
    from exspy.material import elements as db

    table = []
    for symbol in db.keys():
        props = db[symbol]
        general = props.General_properties
        lines = []
        if "Xray_lines" in props.Atomic_properties:
            for name, line in props.Atomic_properties.Xray_lines.as_dictionary().items():
                lines.append({"line": name, "energy": line["energy (keV)"], "weight": line["weight"]})
        lines.sort(key=lambda l: l["energy"])
        table.append({
            "symbol": symbol, "Z": int(general.Z), "name": general.name,
            "atomic_weight": float(general.atomic_weight), "lines": lines,
        })
    table.sort(key=lambda e: e["Z"])
    return table


def eds_info(signal) -> dict | None:
    """Elements, lines and acquisition parameters of an EDS signal (else None)."""
    signal_type = signal.metadata.Signal.get_item("signal_type", "") or ""
    if not signal_type.startswith("EDS"):
        return None
    md = signal.metadata
    instrument = "TEM" if signal_type == "EDS_TEM" else "SEM"
    beam = md.get_item(f"Acquisition_instrument.{instrument}.beam_energy")
    resolution = md.get_item(f"Acquisition_instrument.{instrument}.Detector.EDS.energy_resolution_MnKa")
    return {
        "elements": list(md.get_item("Sample.elements", [])),
        "xray_lines": list(md.get_item("Sample.xray_lines", [])),
        "beam_energy": float(beam) if beam is not None else None,
        "energy_resolution_MnKa": float(resolution) if resolution is not None else None,
    }


def set_elements(signal, elements: list[str]) -> None:
    """Replace the signal's elements; X-ray lines are picked automatically."""
    from exspy.material import elements as db

    unknown = [e for e in elements if e not in db.keys()]
    if unknown:
        raise ValueError(f"Unknown element(s): {', '.join(unknown)}")
    sample = signal.metadata.get_item("Sample")
    if sample is not None:
        for key in ("elements", "xray_lines"):
            if key in sample:
                sample.__delattr__(key)
    if elements:
        signal.set_elements(sorted(set(elements)))
        if eds_info(signal)["beam_energy"] is not None:  # line choice needs it
            signal.add_lines()
