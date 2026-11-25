"""The abTEM worker: runs in abTEM's own environment, never imports si_viewer.

Run as ``python -I main.py <command> [request.json]``. It writes JSON lines to stdout:
``{"progress": 0.4, "message": "…"}`` as it goes, ``{"result": {…}}`` at the end.
"""

from __future__ import annotations

import json
import platform
import sys


def emit(**message) -> None:
    print(json.dumps(message), flush=True)


def ping() -> dict:
    """The versions of what simulations will run with."""
    from importlib.metadata import version

    import abtem

    return {
        "abtem": abtem.__version__,
        "python": platform.python_version(),
        **{name: version(name) for name in ("ase", "dask", "numpy", "zarr")},
    }


def _compute(lazy, workers: int, start: float, span: float, message: str):
    """Compute ``lazy`` on threads, reporting progress from ``start`` over ``span``."""
    from dask.callbacks import Callback

    class Progress(Callback):
        def _start_state(self, dsk, state):
            self.total = max(1, sum(len(state[k]) for k in ("ready", "waiting", "running", "finished")))
            self.done = 0

        def _posttask(self, key, result, dsk, state, worker_id):
            self.done += 1
            emit(progress=start + span * self.done / self.total, message=message)

    emit(progress=start, message=message)
    with Progress():
        return lazy.compute(scheduler="threads", num_workers=workers)


def _model(settings: dict):
    """The model's atoms."""
    import abtem
    import ase.io

    abtem.config.set({"diagnostics.progress_bar": False})
    emit(progress=0.05, message="Reading the model")
    return ase.io.read(settings["model"], format="extxyz")


def _configs(settings: dict) -> int:
    """The frozen phonon configurations ``settings`` ask for (0: none)."""
    n = int(settings.get("phonon_configs") or 0)
    return n if n >= 2 else 0


def _phonons(settings: dict, atoms):
    """abTEM's FrozenPhonons of ``atoms`` for ``settings`` (``phonon_configs``,
    ``phonon_sigma`` Å, ``phonon_seed``), or None if they are off. The waves
    of a multislice through them keep every configuration (a leading axis):
    averaging them is up to the caller, of intensities."""
    import warnings

    import abtem

    if not _configs(settings):
        return None
    warnings.filterwarnings("ignore", message="ensemble_mean=False")
    return abtem.FrozenPhonons(
        atoms, _configs(settings), float(settings.get("phonon_sigma", 0.1)),
        seed=int(settings.get("phonon_seed", 0)), ensemble_mean=False,
    )


def _potential_kwargs(settings: dict) -> dict:
    return {
        "slice_thickness": settings["slice_thickness"],
        "parametrization": settings["parametrization"],
        "projection": settings["projection"],
    }


def _potential(settings: dict, tile: tuple[int, int] = (1, 1), exit_planes: int | None = None):
    """The model's abTEM potential, with frozen phonons if ``settings`` ask
    (then multislice through it gives waves with a leading configuration
    axis). ``tile``: the model repeated along x and y (the phonons of every
    repeat are independent); ``exit_planes``: as abTEM's."""
    import math

    import abtem

    atoms = _model(settings)
    extra = {} if exit_planes is None else {"exit_planes": exit_planes}
    if tuple(tile) == (1, 1):
        grid = {"sampling": settings["sampling"]}
    else:
        # The grid of the unit, repeated: the same pixels as tiling its potential.
        x, y = (math.ceil(float(length) / settings["sampling"]) * t for length, t in zip(atoms.cell.lengths(), tile))
        atoms = atoms * (tile[0], tile[1], 1)
        grid = {"gpts": (x, y)}
    return abtem.Potential(_phonons(settings, atoms) or atoms, **grid, **_potential_kwargs(settings), **extra)


def hrtem(request: dict) -> dict:
    """A plane wave through the model (the incident beam being 1). Saves its
    exit wave in ``folder`` as ``exit_wave.npy`` (complex): [x, y], or with
    frozen phonons [config, x, y], one wave per configuration (never
    averaged: average intensities, as ``image`` does). Images are formed from
    it with a CTF (``image``). ``configs`` is 0 without frozen phonons."""
    from pathlib import Path

    import abtem
    import numpy as np

    potential = _potential(request)
    folder = Path(request["folder"])
    waves = abtem.PlaneWave(energy=request["energy"]).multislice(potential)
    waves = _compute(waves, request["workers"], 0.1, 0.85, "Running multislice")
    exit_wave = np.asarray(waves.array, np.complex64)
    configs = _configs(request)
    assert exit_wave.ndim == (3 if configs else 2), exit_wave.shape
    np.save(folder / "exit_wave.npy", exit_wave)
    return {
        "shape": list(exit_wave.shape),
        "configs": configs,
        "sampling": [float(x) for x in waves.sampling],
        "slices": len(potential),
        # The mean over configurations (and pixels) of |ψ|².
        "exit_intensity": float((np.abs(exit_wave) ** 2).mean()),
        "abtem": abtem.__version__,
    }


def wavefunction(request: dict) -> dict:
    """A plane wave or probe through the (tiled) potential, kept at every
    depth. Saves ``wave.npy`` in ``folder``: complex, indexed [plane, x, y],
    plane 0 being the incident wave and plane n the wave after n slices; with
    frozen phonons [config, plane, x, y], one wave per configuration (never
    averaged). Their tiles move independently: the displaced model is tiled,
    not its potential. ``configs`` is 0 without frozen phonons."""
    from pathlib import Path

    import abtem
    import numpy as np

    configs = _configs(request)
    if configs:
        tiled = _potential(request, tile=tuple(request["tile"]), exit_planes=1)
        thickness = float(tiled.slice_thickness[0])
    else:
        built = _potential(request)
        unit = _compute(built.build(), request["workers"], 0.1, 0.2, "Building the potential slices")
        tiled = abtem.CrystalPotential(unit, repetitions=(*request["tile"], 1), exit_planes=1)
        thickness = float(unit.slice_thickness[0])
    position = None
    if request["beam"] == "probe":
        position = request["position"] or [tiled.extent[0] / 2, tiled.extent[1] / 2]
        probe = abtem.Probe(
            energy=request["energy"], semiangle_cutoff=request["semiangle"],
            defocus=request["defocus"], Cs=request["cs"],
        )
        waves = probe.multislice(tiled, scan=abtem.CustomScan([position]), max_batch=1)
    else:
        waves = abtem.PlaneWave(energy=request["energy"]).multislice(tiled)
    waves = _compute(waves, request["workers"], 0.3, 0.65, "Propagating through the slices")
    array = np.asarray(waves.array, np.complex64)
    if request["beam"] == "probe":
        array = array.squeeze(axis=-3)  # the single probe position
    assert array.ndim == (4 if configs else 3), array.shape
    np.save(Path(request["folder"]) / "wave.npy", array)
    first = array[0] if configs else array  # the incident wave is the same in every configuration
    incident = float((np.abs(first[0]) ** 2).sum())
    return {
        "shape": list(array.shape),
        "configs": configs,
        "sampling": [float(x) for x in waves.sampling],
        "extent": [float(x) for x in tiled.extent],
        "slice_thickness": thickness,
        "position": [float(x) for x in position] if position else None,
        # Scales |ψ|²: a plane wave's incident pixels are 1, a probe's total is 1.
        "intensity_scale": 1.0 if request["beam"] == "plane" else 1 / incident,
        "abtem": abtem.__version__,
    }


def potential(request: dict) -> dict:
    """Build a model's potential slices and save them as ``output`` (.npy,
    indexed [slice, x, y]; each slice is the potential integrated through it,
    in V·Å, so their sum is the projected potential). With frozen phonons it
    is the mean over the configurations (built one at a time), the thermally
    smeared potential. ``configs`` is 0 without frozen phonons."""
    import abtem
    import numpy as np

    atoms = _model(request)
    phonons = _phonons(request, atoms)
    kwargs = {"sampling": request["sampling"], **_potential_kwargs(request)}
    if phonons is None:
        built = abtem.Potential(atoms, **kwargs)
        slices = _compute(built.build(), request["workers"], 0.1, 0.85, "Building the potential slices")
        array = np.asarray(slices.array, np.float32)
    else:
        n = phonons.num_configs
        array = None
        for i, displaced in enumerate(phonons):
            built = abtem.Potential(displaced, **kwargs)
            message = f"Building the potential slices, configuration {i + 1} of {n}"
            slices = _compute(built.build(), request["workers"], 0.1 + 0.85 * i / n, 0.85 / n, message)
            part = np.asarray(slices.array, np.float32)
            array = part if array is None else array + part
        array = array / np.float32(n)
    np.save(request["output"], array)
    return {
        "path": request["output"],
        "shape": list(array.shape),
        "configs": 0 if phonons is None else phonons.num_configs,
        "sampling": [float(x) for x in built.sampling],
        "slice_thickness": [float(x) for x in slices.slice_thickness],
        "abtem": abtem.__version__,
    }


def _transfer(request: dict, focal_spread: float | None = None):
    """abTEM's CTF for ``request`` (``focal_spread``: instead of the request's)."""
    import abtem
    import numpy as np

    coefficients = {
        "C10": -request["defocus"], "C12": request["c12"], "phi12": request["phi12"],
        "C30": request["cs"], "C50": request["c5"],
    }
    return abtem.CTF(
        energy=request["energy"],
        semiangle_cutoff=request["semiangle"] or np.inf,
        focal_spread=request["focal_spread"] if focal_spread is None else focal_spread,
        angular_spread=request["angular_spread"],
        aberration_coefficients=coefficients,
    )


# The partial coherence tutorial's incoherent focal series: Gaussian
# weights on defocus values within ±1.5 σ, σ = focal spread / √2.
FOCAL_SAMPLES = 7
FOCAL_LIMIT = 1.5


def _image_of(array, request: dict):
    """The intensity image (float32, [x, y]) of one complex wave ``array``
    ([x, y]) through the CTF of ``request`` (see ``image``)."""
    import abtem
    import numpy as np

    waves = abtem.Waves(array, energy=request["energy"], sampling=tuple(request["sampling"]))
    if request["coherence"] == "focal-series" and request["focal_spread"] > 0:
        transfer = _transfer(request, focal_spread=0.0)
        transfer.defocus = abtem.distributions.gaussian(
            center=request["defocus"], standard_deviation=request["focal_spread"] / np.sqrt(2),
            num_samples=FOCAL_SAMPLES, sampling_limit=FOCAL_LIMIT,
        )
        images = waves.apply_ctf(transfer).intensity()
        images = images.compute() if hasattr(images, "compute") else images
        result = images.reduce_ensemble()
    else:
        result = waves.apply_ctf(_transfer(request)).intensity()
    result = result.compute() if hasattr(result, "compute") else result
    return np.asarray(result.array, np.float32)


def image(request: dict) -> dict:
    """An exit wave (``wave``: .npy, complex, [x, y]) through a CTF, as the
    HRTEM walkthroughs form an image: ``apply_ctf`` then the intensity.
    A wave with a leading configuration axis ([config, x, y]: frozen phonons,
    or any ensemble) gives an image per configuration, averaged: intensities
    are averaged, never complex waves. ``coherence``: "envelope" keeps the
    focal spread as abTEM's quasi-coherent envelope; "focal-series" averages
    images over a Gaussian spread of defocus instead (reduce_ensemble, within
    each configuration). The angular spread stays an envelope.
    Returns the image (float32, [x, y], base64) and the ``configs`` averaged
    (0 for a plain [x, y] wave)."""
    import base64

    import numpy as np

    array = np.load(request["wave"], mmap_mode="r")
    if array.ndim not in (2, 3):
        raise ValueError(f"A wave is [x, y] or [config, x, y], not {list(array.shape)}")
    configs = array.shape[0] if array.ndim == 3 else 0
    if configs:
        total = None
        for i in range(configs):
            part = _image_of(np.array(array[i]), request).astype(np.float64)
            total = part if total is None else total + part
        intensity = (total / configs).astype(np.float32)
    else:
        intensity = _image_of(np.array(array), request)
    return {
        "image": base64.b64encode(intensity.tobytes()).decode("ascii"),
        "shape": list(intensity.shape),
        "configs": configs,
        "mean": float(intensity.mean()),
    }


def ctf(request: dict) -> dict:
    """A contrast transfer function: its radial profiles (abTEM's: the CTF,
    including its envelopes and aperture, then each of those), its 2D form
    (indexed [x, y], zero frequency at the centre) and its landmarks."""
    import base64

    import numpy as np
    from abtem.transfer import point_resolution

    transfer = _transfer(request)
    max_angle = request["max_angle"]
    profiles = transfer.profiles(gpts=request["profile_points"], max_angle=max_angle, phi=request["phi"])
    values = np.asarray(profiles.array, np.float64)
    names = list(profiles.ensemble_axes_metadata[-1].values) if values.ndim > 1 else ["ctf"]
    values = values.reshape(len(names), -1)
    curves = {name: values[i].tolist() for i, name in enumerate(names)}
    k = np.arange(values.shape[1]) * profiles.sampling

    # Information limit: where the envelopes together fall to exp(-2).
    envelope = np.ones_like(k)
    for name in ("temporal envelope", "spatial envelope"):
        if name in curves:
            envelope = envelope * np.asarray(curves[name])
    below = np.nonzero(envelope < np.exp(-2))[0]
    information_limit = float(1 / k[below[0]]) if below.size and k[below[0]] > 0 else None

    out = {}
    n = request["image_points"]
    if n:
        # abTEM's to_diffraction_patterns ignores the gpts it is given (1.0.11): set the grid first.
        transfer.gpts = (n, n)
        image = transfer.to_diffraction_patterns(max_angle=max_angle)
        array = np.asarray(image.array).imag.astype(np.float32)
        out = {
            "image": base64.b64encode(array.tobytes()).decode("ascii"),
            "image_shape": list(array.shape),
            "image_step": [float(x) for x in image.sampling],
        }
    return {
        **out,
        "k_step": float(profiles.sampling),
        "curves": curves,
        "wavelength": float(transfer.wavelength),
        "scherzer_defocus": float(transfer.scherzer_defocus) if request["cs"] else None,
        "point_resolution": float(point_resolution(request["cs"], request["energy"])) if request["cs"] else None,
        "crossover_angle": float(transfer.crossover_angle) if request["cs"] else None,
        "information_limit": information_limit,
    }


def _series():
    """``series.py``, beside this file, loaded by path (``-I`` keeps this folder off the import path)."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("abtem_worker_series", Path(__file__).with_name("series.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def thickness_series(request: dict) -> dict:
    """The exit wave through the model at several thicknesses (see ``series.py``)."""
    return _series().thickness_series(request, sys.modules[__name__])


def defocus_series(request: dict) -> dict:
    """An exit wave's images through one CTF at several defocus values (see ``series.py``)."""
    return _series().defocus_series(request, sys.modules[__name__])


# Commands a long-running worker answers (``serve``): quick, one at a time.
SERVED = {"ctf": ctf, "image": image, "defocus_series": defocus_series}


def serve() -> None:
    """Answer requests until stdin closes: each line ``{"command": …,
    "request": {…}}`` gets one line back, ``{"result": …}`` or ``{"error": …}``."""
    import abtem

    abtem.config.set({"diagnostics.progress_bar": False})
    emit(ready=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
            emit(result=SERVED[message["command"]](message["request"]))
        except Exception as e:  # noqa: BLE001 - reported back, the worker keeps serving
            emit(error=f"{type(e).__name__}: {e}")


# Each command, and whether it takes a request file.
COMMANDS = {
    "ping": (ping, False), "potential": (potential, True), "hrtem": (hrtem, True), "wavefunction": (wavefunction, True),
    "thickness_series": (thickness_series, True),
}


def _stem():
    """``stem.py``, beside this file, loaded by path (as ``series.py`` is)."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("abtem_worker_stem", Path(__file__).with_name("stem.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# STEM imaging (see ``stem.py``): a job, and two quick commands.
COMMANDS["stem"] = (lambda request: _stem().stem(request, sys.modules[__name__]), True)
SERVED["stem-integrate"] = lambda request: _stem().integrate(request)
SERVED["stem-post"] = lambda request: _stem().post(request)


def main(argv: list[str]) -> int:
    if argv == ["serve"]:
        serve()
        return 0
    command = COMMANDS.get(argv[0]) if argv else None
    if command is None or len(argv) != (2 if command[1] else 1):
        print(f"Unknown worker command: {' '.join(argv)}", file=sys.stderr)
        return 2
    run, takes_request = command
    if takes_request:
        with open(argv[1], encoding="utf-8") as f:
            emit(result=run(json.load(f)))
    else:
        emit(result=run())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
