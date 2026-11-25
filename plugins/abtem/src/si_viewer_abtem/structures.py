"""Reading crystal structures with ASE."""

from __future__ import annotations

from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np

# Atoms are drawn at this fraction of their covalent radius.
DRAWN_RADIUS = 0.45
# Fractional coordinates this close to 0 are on a cell boundary.
BOUNDARY = 1e-4


class StructureError(ValueError):
    """A structure that can't be read, in plain English."""


def structure_atoms(path: str | Path) -> dict:
    """The unit cell to draw from the CIF at ``path``: ``cell`` (a, b, c as
    Cartesian vectors, Å); ``atoms`` with ``symbol``, ``frac``, ``xyz`` and
    ``site`` per drawn atom, atoms on a cell boundary repeated on the opposite
    side; ``sites``, the occupancy of each site; ``elements``, the colour,
    drawn radius and covalent radius (Å) of each element."""
    import ase.io
    from ase.data import atomic_numbers, covalent_radii
    from ase.data.colors import jmol_colors

    path = Path(path)
    try:
        atoms = ase.io.read(path, format="cif", index=0)
    except Exception as e:
        raise StructureError(f"{path.name} can't be read as a CIF: {e}") from None
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


def read_structure(path: str | Path) -> dict:
    """What the tree and the Structure tab show of the CIF at ``path``:
    ``formula`` (None with mixed sites), ``atoms`` (sites in the cell),
    ``composition`` (element -> atoms per cell, by occupancy), ``cell``
    (a, b, c in Å; α, β, γ in degrees), ``spacegroup`` and ``mixed_sites``."""
    import ase.io

    path = Path(path)
    try:
        atoms = ase.io.read(path, format="cif", index=0)
    except Exception as e:  # ASE raises many types for a bad file
        raise StructureError(f"{path.name} can't be read as a CIF: {e}") from None
    if len(atoms) == 0:
        raise StructureError(f"{path.name} has no atoms")

    occupancy = atoms.info.get("occupancy") or {}
    composition: Counter = Counter()
    mixed = set()
    # ASE keys the occupancies by the site's kind in the asymmetric unit.
    kinds = atoms.arrays.get("spacegroup_kinds", atoms.get_tags())
    for symbol, kind in zip(atoms.get_chemical_symbols(), kinds):
        site = occupancy.get(str(int(kind)))
        if not site:
            composition[symbol] += 1.0
            continue
        for element, fraction in site.items():
            composition[element] += float(fraction)
        if len(site) > 1 or any(abs(float(f) - 1) > 1e-6 for f in site.values()):
            mixed.add(int(kind))

    group = atoms.info.get("spacegroup")
    return {
        "formula": None if mixed else atoms.get_chemical_formula("metal"),
        "atoms": len(atoms),
        "composition": {k: round(v, 4) for k, v in sorted(composition.items())},
        "cell": [round(float(x), 5) for x in atoms.cell.cellpar()],
        "spacegroup": {"number": int(group.no), "symbol": str(group.symbol)} if group is not None else None,
        "mixed_sites": len(mixed),
    }
