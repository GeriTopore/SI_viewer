"""STEM imaging in the abTEM worker (runs in abTEM's environment, never
imports si_viewer): a probe scanned over a potential with a flexible annular
detector, and annuli integrated from its polar data afterwards.

``main.py`` loads this file by path and calls ``stem`` (a job), then
``integrate`` and ``post`` (answered by the long-running worker).
"""

from __future__ import annotations

import base64

# Probes per task at most, and the largest share of memory (bytes) a batch of
# waves may hold, so that progress is fine-grained but batches stay small.
TARGET_TASKS = 60
BATCH_BYTES = 256 * 1024**2


def _batch(n_positions: int, gpts: tuple[int, int]) -> int:
    """Probe positions per batch (``max_batch``)."""
    by_memory = max(1, BATCH_BYTES // (gpts[0] * gpts[1] * 8 * 4))
    return int(max(1, min(by_memory, n_positions // TARGET_TASKS)))


def _images(polar, step: float, sampling, limits):
    """Images (list of [x, y] float32 arrays) of ``polar`` ([x, y, radial],
    bins of ``step`` mrad) integrated over each of ``limits`` (inner, outer),
    with abTEM's ``integrate_radial``."""
    import numpy as np
    from abtem.core.axes import ScanAxis
    from abtem.measurements import PolarMeasurements

    array = np.asarray(polar, np.float32)[..., None]
    measurement = PolarMeasurements(
        array, radial_sampling=step, azimuthal_sampling=2 * np.pi,
        ensemble_axes_metadata=[
            ScanAxis(label=name, units="Å", sampling=float(s), endpoint=False) for name, s in zip("xy", sampling)
        ],
    )
    out = []
    for inner, outer in limits:
        image = measurement.integrate_radial(float(inner), float(outer))
        out.append(np.asarray(image.array, np.float32))
    return out


def _encode(images) -> dict:
    return {
        "images": [base64.b64encode(image.tobytes()).decode("ascii") for image in images],
        "shape": list(images[0].shape),
    }


def stem(request: dict, main) -> dict:
    """A probe scanned over the model's potential (tiled laterally, the scan
    staying over the original cell). Saves ``polar.npy`` in ``folder``
    (float32, [x, y, radial bin]: the share of the beam in each bin of
    ``step`` mrad from 0, averaged over frozen-phonon configurations) and
    ``images.npy`` ([detector, x, y], the requested annuli, for the caller to
    read and remove). ``main``: the worker's module, for its potential."""
    from pathlib import Path

    import abtem
    import numpy as np

    folder = Path(request["folder"])
    tx, ty = request["tile"]
    crystal = main._potential(request, (tx, ty))
    if getattr(crystal, "num_configurations", 1) == 1:
        # Built once, instead of again for every batch of probes.
        crystal = main._compute(crystal.build(), request["workers"], 0.05, 0.1, "Building the potential slices")

    probe = abtem.Probe(
        energy=request["energy"], semiangle_cutoff=request["semiangle"], defocus=request["defocus"], Cs=request["cs"],
    )
    probe.grid.match(crystal)
    (x0, y0), (x1, y1) = request["start"], request["end"]
    scan = abtem.GridScan(
        start=(x0 / tx, y0 / ty), end=(x1 / tx, y1 / ty), sampling=request["scan_sampling"], fractional=True, potential=crystal,
    )
    detector = abtem.FlexibleAnnularDetector(step_size=request["step"])
    n_positions = int(np.prod(scan.shape))
    gpts = tuple(int(n) for n in crystal.gpts)
    measurement = probe.scan(crystal, scan=scan, detectors=detector, max_batch=_batch(n_positions, gpts))
    measurement = main._compute(measurement, request["workers"], 0.15, 0.8, "Scanning the probe")

    # [..., x, y, radial, azimuthal (one bin)]. Frozen phonons are averaged by
    # abTEM; any other leading axes are averaged here.
    array = np.asarray(measurement.array, np.float32)
    array = array.reshape((-1,) + array.shape[-4:])[..., 0].mean(axis=0)
    scan_sampling = [float(s) for s in scan.sampling]
    np.save(folder / "polar.npy", array)
    step = float(measurement.radial_sampling)
    max_angle = step * array.shape[-1]
    limits = [(d["inner"], min(d["outer"], max_angle)) for d in request["detectors"]]
    images = _images(array, step, scan_sampling, limits)
    np.save(folder / "images.npy", np.stack(images))
    return {
        "shape": list(array.shape),
        "scan_sampling": scan_sampling,
        "scan_extent": [float(array.shape[0] * scan_sampling[0]), float(array.shape[1] * scan_sampling[1])],
        "step": step,
        "max_angle": float(max_angle),
        "positions": n_positions,
        "gpts": list(gpts),
        "abtem": abtem.__version__,
    }


def integrate(request: dict) -> dict:
    """Annuli of a run's polar data (``polar``: .npy, [x, y, radial]):
    ``limits`` is a list of [inner, outer] (mrad). Returns the images
    (float32 [x, y], base64) and their shape."""
    import numpy as np

    polar = np.load(request["polar"], mmap_mode="r")
    return _encode(_images(polar, request["step"], request["sampling"], request["limits"]))


def post(request: dict) -> dict:
    """An image (``image``: .npy, float32 [x, y], ``sampling`` Å) as the STEM
    quickstart processes it: interpolated to a finer ``interpolate`` Å, blurred
    by the source size (``source_size``: Gaussian σ, Å), then Poisson noise at
    ``dose`` e⁻/Å² (``seed``). Unused steps are None. Returns the image."""
    import numpy as np
    from abtem.measurements import Images

    array = np.load(request["image"]).astype(np.float32)
    images = Images(array, sampling=tuple(request["sampling"]))
    if request["interpolate"]:
        images = images.interpolate(request["interpolate"])
    if request["source_size"]:
        images = images.gaussian_filter(request["source_size"])
    if request["dose"]:
        images = images.poisson_noise(dose_per_area=request["dose"], seed=request["seed"])
    result = np.asarray(images.array, np.float32)
    return {**_encode([result]), "sampling": [float(s) for s in images.sampling]}
