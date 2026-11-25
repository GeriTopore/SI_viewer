"""Oriented, orthogonal supercells of a structure, for multislice.

The beam runs along z, so a model's z axis is the zone axis [uvw]. x and y are
the shortest lattice directions perpendicular to it and to each other; y is
the one closest to the structure viewer's "up". When no such lattice vectors
are exactly perpendicular, the nearest ones are used and the remaining
difference is taken up by strain, which is reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .structures import StructureError

# Lattice directions searched for x and y, up to this index.
MAX_INDEX = 12
# Largest angle (degrees) from perpendicular tried, smallest first.
ANGLE_STEPS = (1e-4, 0.05, 0.2, 0.5)
MAX_STRAIN_PERCENT = 1.0
MAX_MODEL_ATOMS = 50000
MAX_REPEAT = 200
# A model starts as one orthogonal cell.
DEFAULT_REPEATS = (1, 1, 1)
# A cell axis this aligned with the zone axis (|cos|) can't set "up".
UP_LIMIT = 0.98


@dataclass(frozen=True)
class Orientation:
    """The cell along a zone axis: ``matrix`` has the lattice directions of x,
    y and z as rows, ``lengths`` their lengths (Å)."""

    matrix: np.ndarray
    lengths: np.ndarray
    strain_percent: float

    @property
    def cells(self) -> int:
        return abs(round(float(np.linalg.det(self.matrix))))


def _reduced(uvw) -> np.ndarray:
    v = np.asarray(uvw, int)
    if v.shape != (3,) or not v.any():
        raise StructureError("A zone axis needs three indices, not all zero")
    return v // np.gcd.reduce(v)


def _strain(h: np.ndarray, lengths: np.ndarray) -> float:
    """The largest principal strain (%) taking the rows of ``h`` to the axes."""
    f = np.linalg.solve(h, np.diag(lengths))
    e = (f @ f.T - np.eye(3)) / 2
    return float(np.abs(np.linalg.eigvalsh((e + e.T) / 2)).max() * 100)


def orient(cell, uvw) -> Orientation:
    """The orthogonal supercell of ``cell`` (a, b, c as rows) with z along
    [uvw]. StructureError if none is found within the index and strain limits."""
    cell = np.asarray(cell, float)
    z = _reduced(uvw)
    if np.abs(z).max() > MAX_INDEX:
        raise StructureError(f"Zone axis indices can go up to {MAX_INDEX}")
    t = z @ cell
    t_hat = t / np.linalg.norm(t)

    axis = np.arange(-MAX_INDEX, MAX_INDEX + 1)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    grid = grid[np.gcd.reduce(grid, axis=1) == 1]
    vectors = grid @ cell
    lengths = np.linalg.norm(vectors, axis=1)
    units = vectors / lengths[:, None]

    reference = next(v for v in cell[::-1] if abs(v @ t_hat) / np.linalg.norm(v) < UP_LIMIT)
    up = reference - (reference @ t_hat) * t_hat
    up /= np.linalg.norm(up)

    best = None
    for degrees in ANGLE_STEPS:
        limit = np.sin(np.radians(degrees))
        across = np.abs(units @ t_hat) < limit
        ys = np.flatnonzero(across & (units @ up > 1e-6))
        if not ys.size:
            continue
        for iy in ys[np.lexsort((-np.round(units[ys] @ up, 6), np.round(lengths[ys], 6)))]:
            y_hat = units[iy]
            # x makes (x, y, z) right-handed.
            xs = np.flatnonzero(across & (np.abs(units @ y_hat) < limit) & (units @ np.cross(y_hat, t_hat) > 0))
            if not xs.size:
                continue
            ix = xs[np.lexsort((-grid[xs].sum(axis=1), np.round(lengths[xs], 6)))[0]]
            matrix = np.array([grid[ix], grid[iy], z])
            h = matrix @ cell
            axes = np.linalg.norm(h, axis=1)
            strain = _strain(h, axes)
            if strain <= MAX_STRAIN_PERCENT and (best is None or np.prod(axes) < np.prod(best.lengths) - 1e-6):
                best = Orientation(matrix, axes, 0.0 if strain < 1e-6 else strain)
            break
        if best is not None:
            break
    if best is None:
        raise StructureError(f"No orthogonal cell along [{z[0]} {z[1]} {z[2]}] within {MAX_STRAIN_PERCENT:g}% strain")
    return best


def _read_unit(path: str | Path):
    import ase.io

    path = Path(path)
    try:
        return ase.io.read(path, format="cif", index=0)
    except Exception as e:
        raise StructureError(f"{path.name} can't be read as a CIF: {e}") from None


def preview(path: str | Path, uvw, repeats=None) -> dict:
    """What a model along [uvw] would be, without building it."""
    unit = _read_unit(path)
    o = orient(unit.cell.array, uvw)
    repeats = list(repeats) if repeats else list(DEFAULT_REPEATS)
    per_cell = len(unit) * o.cells
    occupancy = unit.info.get("occupancy") or {}
    return {
        "uvw": _reduced(uvw).tolist(),
        "axes": o.matrix.tolist(),
        "cell": [round(float(x), 5) for x in o.lengths],
        "strain_percent": round(o.strain_percent, 4),
        "repeats": repeats,
        "size": [round(float(x) * n, 3) for x, n in zip(o.lengths, repeats)],
        "atoms": per_cell * int(np.prod(repeats)),
        "max_atoms": MAX_MODEL_ATOMS,
        "mixed": any(len(s) > 1 or any(abs(float(x) - 1) > 1e-6 for x in s.values()) for s in occupancy.values()),
    }


def _assign_sites(atoms, occupancy: dict, seed: int):
    """Give each atom one element, by its site's occupancy: each site gets the
    nearest whole numbers of each element, placed at random; vacancies are
    removed."""
    from ase.data import atomic_numbers

    kinds = atoms.arrays["spacegroup_kinds"]
    numbers = atoms.numbers.copy()
    keep = np.ones(len(atoms), bool)
    rng = np.random.default_rng(seed)
    for kind in np.unique(kinds):
        site = occupancy.get(str(int(kind)))
        if not site:
            continue
        members = rng.permutation(np.flatnonzero(kinds == kind))
        names = sorted(site)
        fractions = np.array([float(site[n]) for n in names] + [max(0.0, 1 - sum(float(x) for x in site.values()))])
        fractions /= fractions.sum()
        exact = fractions * len(members)
        counts = np.floor(exact).astype(int)
        for i in np.argsort(counts - exact)[: len(members) - counts.sum()]:  # largest remainders
            counts[i] += 1
        start = 0
        for name, count in zip(names, counts[:-1]):
            numbers[members[start : start + count]] = atomic_numbers[name]
            start += count
        keep[members[start:]] = False
    atoms.numbers = numbers
    return atoms[keep]


def build_model(path: str | Path, uvw, repeats, seed: int = 0):
    """The model along [uvw], ``repeats`` (x, y, z) of the orthogonal cell:
    (ase.Atoms with a diagonal, periodic cell and one element per atom,
    its ``preview``). Mixed sites are filled at random from ``seed``."""
    from ase import Atoms
    from ase.build import make_supercell

    if len(repeats) != 3 or any(not 1 <= int(n) <= MAX_REPEAT for n in repeats):
        raise StructureError(f"Repeats go from 1 to {MAX_REPEAT}")
    info = preview(path, uvw, repeats)
    if info["atoms"] > MAX_MODEL_ATOMS:
        raise StructureError(f"{info['atoms']:,} atoms: the limit is {MAX_MODEL_ATOMS:,}")

    unit = _read_unit(path)
    o = orient(unit.cell.array, uvw)
    kinds = unit.arrays.get("spacegroup_kinds", np.arange(len(unit)))
    occupancy = unit.info.get("occupancy") or {}
    plain = Atoms(numbers=unit.numbers, positions=unit.positions, cell=unit.cell, pbc=True)
    plain.set_array("spacegroup_kinds", np.asarray(kinds, int))
    cell = make_supercell(plain, o.matrix)
    if len(cell) != len(unit) * o.cells:
        raise StructureError("The supercell has the wrong number of atoms")

    # Fractional coordinates are kept: this applies the rotation (and any strain).
    scaled = np.mod(np.round(cell.get_scaled_positions(wrap=True), 9), 1.0)
    model = Atoms(numbers=cell.numbers, scaled_positions=scaled, cell=np.diag(o.lengths), pbc=True)
    model.set_array("spacegroup_kinds", cell.arrays["spacegroup_kinds"])
    model = model.repeat(tuple(int(n) for n in repeats))
    if occupancy:
        model = _assign_sites(model, occupancy, seed)
    del model.arrays["spacegroup_kinds"]
    info["atoms"] = len(model)
    return model, info


def write_model(model, path: str | Path) -> None:
    import ase.io

    ase.io.write(path, model, format="extxyz")


def read_model(path: str | Path):
    import ase.io

    return ase.io.read(path, format="extxyz", index=0)


def model_summary(model) -> dict:
    """The composition and formula of a built model."""
    symbols, counts = np.unique(model.get_chemical_symbols(), return_counts=True)
    return {
        "formula": model.get_chemical_formula("metal", empirical=True),
        "atoms": len(model),
        "composition": {str(s): int(n) for s, n in zip(symbols, counts)},
        "cell": [round(float(x), 5) for x in model.cell.cellpar()],
        "spacegroup": None,
        "mixed_sites": 0,
    }


def model_atoms(path: str | Path) -> dict:
    """A model's atoms to draw, in the form of ``structure_atoms``: each atom
    once, since a model is periodic."""
    from ase.data import atomic_numbers, covalent_radii
    from ase.data.colors import jmol_colors

    from .structures import DRAWN_RADIUS

    model = read_model(path)
    symbols = model.get_chemical_symbols()
    names = sorted(set(symbols))
    site_of = {name: i for i, name in enumerate(names)}
    frac = model.get_scaled_positions(wrap=False)
    return {
        "cell": model.cell.array.round(6).tolist(),
        "atoms": {
            "symbol": symbols,
            "frac": frac.round(6).tolist(),
            "xyz": model.positions.round(6).tolist(),
            "site": [site_of[s] for s in symbols],
        },
        "sites": [{name: 1.0} for name in names],
        "elements": {
            e: {
                "color": [round(float(c), 4) for c in jmol_colors[atomic_numbers[e]]],
                "radius": round(float(covalent_radii[atomic_numbers[e]]) * DRAWN_RADIUS, 4),
                "covalent": round(float(covalent_radii[atomic_numbers[e]]), 4),
            }
            for e in names
        },
    }
