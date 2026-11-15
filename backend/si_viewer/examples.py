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


def _fit_two_gaussians() -> list:
    """HyperSpy's "Two Gaussians" simulation (the docs' example), as a fitting
    test: a 32 × 32 spectrum image of two Gaussians (σ 10 and 20) whose
    centres and areas vary at random from pixel to pixel, with Poisson noise.
    The seed is fixed, and the true centre and area maps come with it."""
    import hyperspy.api as hs

    rng = np.random.default_rng(20261114)
    n, channels = 32, 1024
    s = hs.signals.Signal1D(np.zeros((n, n, channels)))
    for ax, name in zip(s.axes_manager.navigation_axes, ("x", "y")):
        ax.name, ax.units = name, "nm"
    e = s.axes_manager.signal_axes[0]
    e.name, e.units, e.scale, e.offset = "Energy", "eV", 1.0, 0.0
    m = s.create_model()
    truth = {}
    for name, sigma, centre, spread, area in (("Gaussian 1", 10, 256, 10, 10000), ("Gaussian 2", 20, 768, 20, 20000)):
        g = hs.model.components1D.Gaussian()
        g.name = name
        m.append(g)
        g.sigma.map["values"][:] = sigma
        g.centre.map["values"][:] = centre + (rng.random((n, n)) - 0.5) * spread
        g.A.map["values"][:] = area * rng.random((n, n))
        for p in g.parameters:
            p.map["is_set"][:] = True
        truth[name] = {"centre": g.centre.map["values"].copy(), "A": g.A.map["values"].copy(), "sigma": sigma}
    si = m.as_signal(show_progressbar=False)
    si.data = rng.poisson(np.clip(si.data, 0, None)).astype(np.float32)
    si.set_signal_origin("simulation")
    si.metadata.General.title = "Two Gaussians (fit test)"
    si.metadata.set_item("Simulation.description", "Gaussian 1: σ 10, centre 256 ± 5, A 0–10000; "
                         "Gaussian 2: σ 20, centre 768 ± 10, A 0–20000; Poisson noise")
    image_axes = [{**ax.get_axis_dictionary(), "navigate": False} for ax in s.axes_manager.navigation_axes[::-1]]
    out = [si]
    for name, t in truth.items():
        for param in ("centre", "A"):
            img = hs.signals.Signal2D(t[param].astype(np.float32), axes=image_axes)
            img.metadata.General.title = f"Truth: {name} {param}"
            out.append(img)
    return out


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


def kemstem_data() -> "Path | None":
    """kemstem's example images (``example_notebooks/data`` of its clone,
    installed with ``pip install -e``), or None without them."""
    from pathlib import Path

    try:
        import kemstem
    except ImportError:
        return None
    data = Path(kemstem.__file__).resolve().parents[2] / "example_notebooks" / "data"
    return data if data.is_dir() else None


def _kemstem_image(name: str, title: str, nm_per_px: float | None):
    """One of kemstem's example HAADF images, calibrated as in its notebook."""
    def load() -> list:
        import hyperspy.api as hs

        s = hs.load(kemstem_data() / name)
        s.metadata.General.title = title
        for ax, axis_name in zip(s.axes_manager.signal_axes, ("x", "y")):
            ax.name = axis_name
            if nm_per_px:
                ax.scale, ax.units = nm_per_px, "nm"
        return [s]

    return load


# (id, file, title, description, nm per pixel from the notebook)
_KEMSTEM = [
    ("kemstem-strain", "rI_1341 2.6 Mx B_C270_0_HAADF.tif", "Relaxed film with misfit dislocations (kemstem)",
     "kemstem example (notebook 1): 2031×2025 HAADF of a partially relaxed perovskite film. "
     "For FFT peaks, Fourier filtering, strain and wave fitting. 0.0184 nm/px.", 0.0184),
    ("kemstem-sites", "rI_2138 B1_RT 5.3 Mx_0_HAADF.tif", "Double perovskite on DyScO₃ (kemstem)",
     "kemstem example (notebook 2): 1017×1017 HAADF of a double perovskite film on its substrate. "
     "For columns, sublattices, bond lengths and angles. Uncalibrated (px).", None),
    ("kemstem-pld", "FigS6_0716__Cryo_NSMO110_5-1Mx_0p5us_1024px_REGISTERED.tif",
     "Charge-ordered Nd₀.₅Sr₀.₅MnO₃ (kemstem)",
     "kemstem example (notebook 3): 963×898 registered cryo HAADF of a charge-ordered manganite. "
     "For superlattice peaks and periodic lattice displacements. 0.0184 nm/px.", 0.0184),
]


def _kemstem_examples() -> list[Example]:
    data = kemstem_data()
    if data is None:
        return []
    return [
        Example(i, title, desc, _kemstem_image(name, title, px))
        for i, name, title, desc, px in _KEMSTEM
        if (data / name).is_file()
    ]


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
    Example(
        "fit-two-gaussians",
        "Fitting test: two Gaussians (synthetic)",
        "HyperSpy's two-Gaussians simulation with a fixed seed: 32×32 pixels, Gaussians of σ 10 and 20 at "
        "256 ± 5 and 768 ± 10 with random areas, Poisson noise; the true centre and area maps come with it.",
        _fit_two_gaussians,
    ),
    *_kemstem_examples(),
]

_BY_ID = {e.id: e for e in EXAMPLES}


def list_examples() -> list[dict]:
    return [{"id": e.id, "title": e.title, "description": e.description} for e in EXAMPLES]


def get_example(example_id: str) -> Example:
    return _BY_ID[example_id]
