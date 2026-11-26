"""Example datasets for trying out features (and for a future tutorial),
grouped by what they are for.

exSpy's and HyperSpy's bundled/synthetic data, a synthetic EDS spectrum image
built from exSpy's FePt spectrum, kemstem's images, and pyxem's 4D-STEM data:
its simulations (offline) and its recorded datasets (downloaded from Zenodo
by pyxem on first use, then cached).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Callable

import numpy as np

# The groups, in the menu's order.
GROUPS = [
    "EDS",
    "Spectrum fitting",
    "Atomic-resolution images",
    "4D-STEM: calibration",
    "4D-STEM: strain",
    "4D-STEM: orientation and phase",
    "4D-STEM: DPC and beam shift",
    "4D-STEM: amorphous (FEM)",
]


@dataclass(frozen=True)
class Example:
    id: str
    title: str
    description: str
    load: Callable[[], list]  # → HyperSpy signals
    group: str = "EDS"
    # pyxem's file of a recorded dataset (downloaded on first use) and its size.
    download: tuple[str, float] | None = None


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
        Example(i, title, desc, _kemstem_image(name, title, px), "Atomic-resolution images")
        for i, name, title, desc, px in _KEMSTEM
        if (data / name).is_file()
    ]


def pyxem_cached(filename: str) -> bool:
    """Whether pyxem's file of a recorded dataset is in its cache already."""
    from pyxem.data._data import Dataset

    return Dataset(filename).is_in_cache


def _pyxem(name: str, title: str, recorded: bool = False, **kwargs):
    """One of pyxem's datasets (``pyxem.data.<name>``): a simulation, or a
    recorded dataset (downloaded by pyxem on first use)."""
    def load() -> list:
        import pyxem.data

        f = getattr(pyxem.data, name)
        s = f(allow_download=True, **kwargs) if recorded else f(**kwargs)
        s = s[0] if isinstance(s, (list, tuple)) else s
        if not recorded and s.data.dtype in (np.float64, np.int64):
            s.change_dtype("float32")  # half the memory, plenty for counts
        s.metadata.General.title = title
        for ax in s.axes_manager.signal_axes:
            ax.units = _PYXEM_UNITS.get(ax.units, ax.units)
        return [s]

    return load


# pyxem's reciprocal units (LaTeX and others) as the app writes them.
_PYXEM_UNITS = {"$\\AA^{-1}$": "1/Å", "$A^{-1}$": "1/Å", "k_A^-1": "1/Å", "A^-1": "1/Å", "nm^-1": "1/nm", "k_nm^-1": "1/nm"}


# (id, pyxem function, title, group, description, keyword arguments)
_PYXEM_SIMULATED = [
    ("pyxem-strain", "simulated_strain", "Strained disks (pyxem, simulated)", "4D-STEM: strain",
     "32×32 scan of 256×256 patterns: a disk lattice under a known strain gradient. "
     "For disk detection, lattice vectors and strain.", {"signal_shape": (256, 256), "disk_radius": 10}),
    ("pyxem-si-grains-simple", "si_grains_simple", "Si grains at zone axes (pyxem, simulated)",
     "4D-STEM: orientation and phase",
     "6×6 scan of 128×128 patterns: Si grains at low-index zone axes. For template matching.", {}),
    ("pyxem-si-grains", "si_grains", "Si grains, random orientations (pyxem, simulated)",
     "4D-STEM: orientation and phase",
     "6×6 scan of 128×128 patterns: 4 Si grains in random orientations. For orientation mapping.", {}),
    ("pyxem-fe-grains", "fe_multi_phase_grains", "Fe bcc and fcc grains (pyxem, simulated)",
     "4D-STEM: orientation and phase",
     "20×20 scan of 128×128 patterns: grains of bcc and fcc Fe. For phase mapping.", {}),
    ("pyxem-overlap", "simulated_overlap", "Overlapping nanocrystals (pyxem, simulated)",
     "4D-STEM: orientation and phase",
     "64×64 scan of 64×64 patterns: 50 nanocrystals overlapping in projection, with noise. "
     "For virtual dark field, disk vectors and clustering.", {"real_space_pixels": 64}),
    ("pyxem-stripes", "simulated_stripes", "Magnetic stripes (pyxem, simulated)", "4D-STEM: DPC and beam shift",
     "60×60 scan of 64×64 patterns: the direct beam shifted one way or the other in alternating stripes. "
     "For centre of mass and DPC.", {}),
    ("pyxem-pn-junction", "simulated_pn_junction", "p–n junction (pyxem, simulated)", "4D-STEM: DPC and beam shift",
     "30×30 scan of 256×256 patterns: the beam shifted by the junction's field. For centre of mass and DPC.", {}),
    ("pyxem-constant-shift", "simulated_constant_shift_magnitude", "Constant beam shift (pyxem, simulated)",
     "4D-STEM: DPC and beam shift",
     "20×20 scan of 128×128 patterns: a field everywhere, shifting the beam by a constant amount. "
     "For descan correction when a plane fit fails.", {}),
    ("pyxem-tilt-boundary", "tilt_boundary_data", "Tilt boundary (pyxem, simulated)", "4D-STEM: DPC and beam shift",
     "10×10 scan of 256×256 patterns: the beam tilted across a boundary. For beam shift and its correction.", {}),
]

# (id, pyxem function, file, MB, title, group, description)
_PYXEM_RECORDED = [
    ("pyxem-au-grating", "au_grating", "au_xgrating_100kX.hspy", 0.1, "Au cross-grating (pyxem)",
     "4D-STEM: calibration", "254×254 pattern of a gold cross-grating. For the ring fit and the pixel size."),
    ("pyxem-au-grating-20cm", "au_grating_20cm", "au_xgrating_20cm.tif", 0.3, "Au cross-grating, 20 cm (pyxem)",
     "4D-STEM: calibration", "256×256 PACBED of a gold cross-grating at 20 cm camera length. For calibration."),
    ("pyxem-zrnb", "zrnb_precipitate", "ZrNbPercipitate.zspy", 394.2, "ZrNb precipitate (pyxem)",
     "4D-STEM: strain", "60×40 scan of 256×256 patterns across a precipitate. For strain mapping."),
    ("pyxem-sped-ag", "sped_ag", "SPED-Ag.zspy", 214.0, "Ag, precession (pyxem)", "4D-STEM: orientation and phase",
     "208×64 scan of 112×112 precession patterns of Ag grains. For template matching."),
    ("pyxem-twinned-nanowire", "twinned_nanowire", "twinned_nanowire.hdf5", 34.5, "Twinned nanowire (pyxem)",
     "4D-STEM: orientation and phase", "30×100 scan of 144×144 patterns of a twinned GaAs nanowire. "
     "For orientation mapping."),
    ("pyxem-sample-with-g", "sample_with_g", "sample_with_g.zspy", 51.7, "Overlapping grains (pyxem)",
     "4D-STEM: orientation and phase", "25×26 scan of 256×256 patterns of overlapping grains. "
     "For orientation mapping."),
    ("pyxem-cuag", "cuag_orientations", "cuzipProcessed.zspy", 32.9, "CuAg orientations (pyxem)",
     "4D-STEM: orientation and phase", "CuAg with grains of Cu (fcc) in several orientations. For orientation mapping."),
    ("pyxem-organic", "organic_semiconductor", "data_processed.zspy", 58.3, "Organic semiconductor (pyxem)",
     "4D-STEM: orientation and phase", "75×75 scan of 128×128 patterns of a beam-sensitive organic semiconductor. "
     "For orientation mapping."),
    ("pyxem-mgo", "mgo_nanocrystals", "mgo_nano.zspy", 104.3, "MgO nanocrystals (pyxem)",
     "4D-STEM: orientation and phase", "105×105 scan of 144×144 patterns of overlapping MgO nanocrystals. "
     "For virtual imaging and disk vectors."),
    ("pyxem-feal", "feal_stripes", "FeAl_stripes.zspy", 382.1, "FeAl magnetic stripes (pyxem)",
     "4D-STEM: DPC and beam shift", "Magnetic stripes in FeAl. For centre of mass and DPC."),
    ("pyxem-ptychography", "small_ptychography", "smallPtychography.hspy", 12.5, "Bilayer WS₂ (pyxem)",
     "4D-STEM: DPC and beam shift", "Bilayer WS₂ with 8×8-pixel patterns. For centre of mass and ptychography."),
    ("pyxem-pdnip", "pdnip_glass", "PdNiP.zspy", 304.6, "PdNiP metallic glass (pyxem)", "4D-STEM: amorphous (FEM)",
     "128×128 scan of 128×128 patterns of a metallic glass. For radial profiles and FEM."),
    *[(f"pyxem-zrcual-{i}", f"zrcual_{i}", f"pos{p}-1.zspy", 18.3, f"ZrCuAl metallic glass {i} (pyxem)",
       "4D-STEM: amorphous (FEM)", "A ZrCuAl metallic glass. For FEM.") for i, p in ((1, 1), (2, 5), (3, 6))],
]


def _pyxem_examples() -> list[Example]:
    try:
        import pyxem.data  # noqa: F401
    except ImportError:
        return []
    return [
        *(Example(i, title, desc, _pyxem(f, title, **kw), group) for i, f, title, group, desc, kw in _PYXEM_SIMULATED),
        *(Example(i, title, desc, _pyxem(f, title, recorded=True), group, (file, mb))
          for i, f, file, mb, title, group, desc in _PYXEM_RECORDED),
    ]


EXAMPLES = [
    Example(
        "fept-si",
        "FePt nanoparticles: EDS spectrum image (synthetic)",
        "64×64 core-shell FePt particles (Pt-rich cores, Fe-rich shells) on carbon, with HAADF. "
        "Built from exSpy's FePt spectrum with Poisson noise; good for maps, regions and fitting.",
        _synthetic_fept_si,
        "EDS",
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
        "Atomic-resolution images",
    ),
    Example(
        "two-gaussians",
        "Two Gaussians: spectrum image (synthetic)",
        "HyperSpy example: 32×32 spectrum image of two Gaussian peaks varying across the scan "
        "(not EDS), for the cursor and regions.",
        lambda: [_two_gaussians()],
        "Spectrum fitting",
    ),
    Example(
        "fit-two-gaussians",
        "Fitting test: two Gaussians (synthetic)",
        "HyperSpy's two-Gaussians simulation with a fixed seed: 32×32 pixels, Gaussians of σ 10 and 20 at "
        "256 ± 5 and 768 ± 10 with random areas, Poisson noise; the true centre and area maps come with it.",
        _fit_two_gaussians,
        "Spectrum fitting",
    ),
    *_kemstem_examples(),
    *_pyxem_examples(),
]

_BY_ID = {e.id: e for e in EXAMPLES}


def list_examples() -> list[dict]:
    """Every example, by group (in ``GROUPS`` order); a recorded pyxem
    dataset not in its cache yet says how much it downloads."""
    order = {g: i for i, g in enumerate(GROUPS)}
    out = []
    for e in sorted(EXAMPLES, key=lambda e: order.get(e.group, len(order))):
        info = {"id": e.id, "title": e.title, "description": e.description, "group": e.group}
        if e.download and not pyxem_cached(e.download[0]):
            info["download_mb"] = e.download[1]
        out.append(info)
    return out


def get_example(example_id: str) -> Example:
    return _BY_ID[example_id]
