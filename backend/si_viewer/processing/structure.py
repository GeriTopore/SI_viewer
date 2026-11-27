"""Crystal structures (CIFs) as orix phases, the way pyxem takes them: a
diffpy.structure read by its CIF parser, with the space group, for diffsims'
template simulation and orix's orientation and phase maps."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

# Where a structure dataset keeps its CIF (the placeholder signal's metadata).
CIF_ITEM = "Sample.structure.cif"
NAME_ITEM = "Sample.structure.name"


class StructureError(ValueError):
    """A CIF that can't be read, in plain English."""


def phase_from_cif(text: str, name: str):
    """The orix ``Phase`` of the CIF ``text`` (as ``Phase.from_cif``)."""
    from diffpy.structure.parsers import p_cif
    from orix.crystal_map import Phase

    parser = p_cif.P_cif()
    try:
        structure = parser.parse(text)
    except Exception as e:  # diffpy raises several types for a bad file
        raise StructureError(f"{name} can't be read as a CIF: {e}") from None
    if structure is None or len(structure) == 0:
        raise StructureError(f"{name} has no atoms")
    space_group = getattr(getattr(parser, "spacegroup", None), "number", None)
    return Phase(name, space_group, structure=structure)


def structure_signal(path: str | Path):
    """A placeholder signal holding the CIF at ``path`` (read once to check it)."""
    import hyperspy.api as hs

    path = Path(path)
    text = path.read_text(errors="replace")
    phase_from_cif(text, path.stem)
    signal = hs.signals.BaseSignal(np.zeros(1, dtype=np.uint8))
    signal.metadata.General.title = path.stem
    signal.metadata.set_item(CIF_ITEM, text)
    signal.metadata.set_item(NAME_ITEM, path.stem)
    return signal


def is_structure(signal) -> bool:
    return signal.metadata.has_item(CIF_ITEM)


def phase_of(signal):
    """The orix ``Phase`` of a structure dataset's signal."""
    return phase_from_cif(signal.metadata.get_item(CIF_ITEM), signal.metadata.get_item(NAME_ITEM) or "phase")


def _formula(composition: dict[str, float]) -> str:
    """Reduced formula, elements in the CIF's order (counts divided by their
    greatest common divisor when whole)."""
    counts = list(composition.values())
    if all(abs(c - round(c)) < 1e-6 for c in counts):
        g = np.gcd.reduce([int(round(c)) for c in counts]) or 1
        counts = [round(c) / g for c in counts]
    fmt = lambda n: "" if abs(n - 1) < 1e-6 else (f"{n:g}")
    return "".join(f"{e}{fmt(n)}" for e, n in zip(composition, counts))


def structure_info(signal) -> dict:
    """What the Info tab and the structure view show: ``name``, ``formula``,
    ``spacegroup`` (number, symbol), ``point_group``, ``crystal_system``,
    ``cell`` (a, b, c in Å; α, β, γ in degrees), ``composition`` (atoms per
    cell, by occupancy) and ``sites`` (element, fractional xyz, occupancy)."""
    phase = phase_of(signal)
    s = phase.structure
    composition: Counter = Counter()
    for atom in s:
        composition[atom.element] += float(atom.occupancy)
    group = phase.space_group
    return {
        "name": phase.name,
        "formula": _formula(dict(composition)),
        "spacegroup": {"number": int(group.number), "symbol": str(group.short_name)} if group is not None else None,
        "point_group": phase.point_group.name if phase.point_group is not None else None,
        "crystal_system": getattr(group, "crystal_system", None) if group is not None else None,
        "cell": [round(float(x), 5) for x in s.lattice.abcABG()],
        "composition": {k: round(v, 4) for k, v in composition.items()},
        "sites": [
            {"element": a.element, "xyz": [round(float(x), 5) for x in a.xyz], "occupancy": round(float(a.occupancy), 4)}
            for a in s
        ],
    }


# ---------- the atoms to draw (the abTEM plugin's reader, with ASE) ----------

# Atoms are drawn at this fraction of their covalent radius.
DRAWN_RADIUS = 0.45
# Fractional coordinates this close to 0 are on a cell boundary.
BOUNDARY = 1e-4


def structure_atoms(signal) -> dict:
    """The unit cell to draw from a Structure's CIF: ``cell`` (a, b, c as
    Cartesian vectors, Å); ``atoms`` with ``symbol``, ``frac``, ``xyz`` and
    ``site`` per drawn atom, atoms on a cell boundary repeated on the opposite
    side; ``sites``, the occupancy of each site; ``elements``, the colour,
    drawn radius and covalent radius (Å) of each element."""
    import io
    from itertools import product

    import ase.io
    from ase.data import atomic_numbers, covalent_radii
    from ase.data.colors import jmol_colors

    name = signal.metadata.get_item(NAME_ITEM) or "structure"
    try:
        atoms = ase.io.read(io.StringIO(signal.metadata.get_item(CIF_ITEM)), format="cif", index=0)
    except Exception as e:  # ASE raises many types for a bad file
        raise StructureError(f"{name} can't be read as a CIF: {e}") from None
    cell = atoms.cell.array
    frac = np.mod(np.round(atoms.get_scaled_positions(wrap=True), 8), 1.0)
    symbols = atoms.get_chemical_symbols()
    occupancy = atoms.info.get("occupancy") or {}
    kinds = atoms.arrays.get("spacegroup_kinds", atoms.get_tags())

    drawn_symbols, drawn_frac, drawn_site = [], [], []
    for i, f in enumerate(frac):
        shifts = [(0, 1) if x < BOUNDARY else (0,) for x in f]
        for shift in product(*shifts):
            drawn_symbols.append(symbols[i])
            drawn_frac.append(f + shift)
            drawn_site.append(i)
    drawn = np.array(drawn_frac)
    sites = [
        {k: float(v) for k, v in (occupancy.get(str(int(kind))) or {symbols[i]: 1.0}).items()}
        for i, kind in enumerate(kinds)
    ]
    names = sorted({e for site in sites for e in site} | set(symbols))
    return {
        "cell": cell.round(6).tolist(),
        "atoms": {
            "symbol": drawn_symbols,
            "frac": drawn.round(6).tolist(),
            "xyz": (drawn @ cell).round(6).tolist(),
            "site": drawn_site,
        },
        "sites": sites,
        "elements": {
            e: {
                "color": [round(float(c), 4) for c in jmol_colors[atomic_numbers[e]]],
                "radius": round(float(covalent_radii[atomic_numbers[e]]) * DRAWN_RADIUS, 4),
                "covalent": round(float(covalent_radii[atomic_numbers[e]]), 4),
            }
            for e in names
        },
    }
