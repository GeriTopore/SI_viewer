"""Example datasets for trying out features (and for a future tutorial).

Everything here works offline: exSpy's and HyperSpy's bundled/synthetic data,
plus a synthetic EDS spectrum image built from exSpy's FePt spectrum.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Example:
    id: str
    title: str
    description: str
    load: Callable[[], list]  # → HyperSpy signals


def _fept_spectrum():
    import exspy.data

    return exspy.data.EDS_TEM_FePt_nanoparticles()


def _tm002_spectrum():
    import exspy.data

    return exspy.data.EDS_SEM_TM002()


def _atomic_resolution_image():
    import hyperspy.data

    s = hyperspy.data.atomic_resolution_image()
    s.metadata.General.title = "Atomic resolution (synthetic)"
    return s


def _two_gaussians():
    import hyperspy.data

    return hyperspy.data.two_gaussians()


@cache
def _fept_shapes() -> dict[str, np.ndarray]:
    """Fe, Pt, Cu and background spectra (counts per channel, full energy
    axis) from a fit of exSpy's FePt spectrum."""
    from .processing.fitting import FitOptions, fit_spectrum

    spec = _fept_spectrum()
    spec.add_elements(["Cu"])  # from the grid
    spec.change_dtype("float64")
    fit = fit_spectrum(spec, FitOptions())
    axis = spec.axes_manager.signal_axes[0]
    i0 = int(round((fit["x"][0] - axis.offset) / axis.scale))
    shapes = {}
    for el in ("Fe", "Pt", "Cu"):
        full = np.zeros(axis.size)
        full[i0:i0 + len(fit["x"])] = np.clip(fit["elements"][el], 0, None)
        shapes[el] = full
    # Background: whatever the elements don't explain (bremsstrahlung, noise peak).
    shapes["background"] = np.clip(spec.data - sum(shapes[el] for el in ("Fe", "Pt", "Cu")), 0, None)
    return shapes


@cache
def _synthetic_fept_si_data() -> tuple[np.ndarray, np.ndarray, dict]:
    """Core-shell FePt particles on a carbon film: counts (y, x, E) and a
    matching HAADF image. Deterministic (fixed seed)."""
    size = 64
    rng = np.random.default_rng(20261107)
    yy, xx = np.mgrid[0:size, 0:size] + 0.5
    fe = np.zeros((size, size))
    pt = np.zeros((size, size))
    thickness = np.full((size, size), 0.3)  # carbon film everywhere
    # (x, y, radius): Pt-rich cores inside Fe-rich shells.
    for cx, cy, r in [(18, 20, 11), (46, 17, 8), (40, 45, 13), (14, 50, 6)]:
        d = np.hypot(xx - cx, yy - cy) / r
        inside = np.clip(1 - d**2, 0, None)  # projected thickness of a sphere
        core = np.clip(1 - (d / 0.55) ** 2, 0, None)
        pt += 0.9 * core + 0.15 * inside
        fe += 0.7 * np.clip(inside - core, 0, None) + 0.1 * inside
        thickness += inside
    shapes = _fept_shapes()
    # Scale so a particle centre collects a few hundred counts.
    scale = 300 / (shapes["Pt"].sum() + shapes["background"].sum())
    expected = (
        thickness[..., None] * shapes["background"]
        + fe[..., None] * shapes["Fe"] * 2.5
        + pt[..., None] * shapes["Pt"]
        + 0.05 * shapes["Cu"]
    ) * scale
    counts = rng.poisson(expected).astype(np.uint16)
    haadf = 1000 * (0.2 * thickness + 3.0 * pt + 1.5 * fe) + rng.normal(0, 40, (size, size))
    return counts, haadf.astype(np.float32), {"pixel_nm": 0.5}


def _synthetic_fept_si() -> list:
    import hyperspy.api as hs

    counts, haadf, info = _synthetic_fept_si_data()
    template = _fept_spectrum()
    e_axis = template.axes_manager.signal_axes[0].get_axis_dictionary()
    nav = [
        {"name": "y", "size": counts.shape[0], "scale": info["pixel_nm"], "offset": 0, "units": "nm", "navigate": True},
        {"name": "x", "size": counts.shape[1], "scale": info["pixel_nm"], "offset": 0, "units": "nm", "navigate": True},
    ]
    si = hs.signals.Signal1D(counts, axes=[*nav, e_axis])
    si.set_signal_type("EDS_TEM")
    si.metadata.Acquisition_instrument = template.metadata.Acquisition_instrument.deepcopy()
    si.metadata.General.title = "EDS (synthetic FePt)"
    si.set_elements(["Cu", "Fe", "Pt"])
    image_axes = [{**a, "navigate": False} for a in nav]
    image = hs.signals.Signal2D(haadf, axes=image_axes)
    image.metadata.General.title = "HAADF"
    return [si, image]


EXAMPLES = [
    Example(
        "fept-si",
        "FePt nanoparticles: EDS spectrum image (synthetic)",
        "64×64 core-shell FePt particles (Pt-rich cores, Fe-rich shells) on carbon, with HAADF. "
        "Built from exSpy's FePt spectrum with Poisson noise; good for maps, regions and fitting.",
        _synthetic_fept_si,
    ),
    Example(
        "fept-spectrum",
        "FePt nanoparticles: EDS TEM spectrum",
        "exSpy example: 200 kV Super-X spectrum of FePt particles (plus Cu from the grid). "
        "Good counts, for fitting and quantification.",
        lambda: [_fept_spectrum()],
    ),
    Example(
        "tm002-spectrum",
        "TM002 alloy: EDS SEM spectrum",
        "exSpy example: SEM spectrum of a multi-element reference sample.",
        lambda: [_tm002_spectrum()],
    ),
    Example(
        "atomic-image",
        "Atomic-resolution image (synthetic)",
        "HyperSpy example: 512×512 simulated atomic-column image, for image display, overlays and rebinning.",
        lambda: [_atomic_resolution_image()],
    ),
    Example(
        "two-gaussians",
        "Two Gaussians: spectrum image (synthetic)",
        "HyperSpy example: 32×32 spectrum image of two Gaussian peaks varying across the scan "
        "(not EDS), for the cursor and regions.",
        lambda: [_two_gaussians()],
    ),
]

_BY_ID = {e.id: e for e in EXAMPLES}


def list_examples() -> list[dict]:
    return [{"id": e.id, "title": e.title, "description": e.description} for e in EXAMPLES]


def get_example(example_id: str) -> Example:
    return _BY_ID[example_id]
