"""Thickness and defocus series: the worker's part (see ``main.py``).

Loaded by ``main.py`` by path (the worker runs as ``python -I main.py``, which
keeps this folder off the import path); each command is given the worker's own
module for its helpers (``_potential``, ``_compute``, ``_transfer``, ``emit``).
"""

from __future__ import annotations

# Complex pixels of a defocus series' chunk (bytes): the CTF is applied to
# many defocus values at once, a chunk of them at a time.
CHUNK_BYTES = 256 * 1024**2


def _focal_samples(worker, request: dict):
    """The defocus offsets (Å) an image is averaged over, with their weights
    (summing to 1): one zero for an envelope, abTEM's Gaussian focal series
    (as ``image``) when ``coherence`` asks for it and there is a focal spread."""
    import abtem
    import numpy as np

    if request["coherence"] != "focal-series" or not request["focal_spread"] > 0:
        return np.zeros(1), np.ones(1)
    spread = abtem.distributions.gaussian(
        center=0.0, standard_deviation=request["focal_spread"] / np.sqrt(2),
        num_samples=worker.FOCAL_SAMPLES, sampling_limit=worker.FOCAL_LIMIT,
    )
    weights = np.ravel(spread.weights).astype(np.float64)
    return np.ravel(spread.values).astype(np.float64), weights / weights.sum()


def thickness_series(request: dict, worker) -> dict:
    """A plane wave through the model, its exit wave kept at the potential's
    ``exit_planes`` (slice indices; -1 is the incident wave). Saves
    ``exit_waves.npy`` in ``folder``: complex, indexed [plane, x, y], or with
    frozen phonons [config, plane, x, y], one wave per configuration (never
    averaged). With ``ctf`` (a CTF request and its ``coherence``), the image
    that CTF forms of every plane, averaged over configurations as intensities,
    is saved as ``images.npy`` (float32, [plane, x, y])."""
    from pathlib import Path

    import abtem
    import numpy as np

    potential = worker._potential(request, exit_planes=tuple(request["exit_planes"]))
    waves = abtem.PlaneWave(energy=request["energy"]).multislice(potential)
    waves = worker._compute(waves, request["workers"], 0.1, 0.7, "Running multislice")
    exit_waves = np.asarray(waves.array, np.complex64)
    configs = worker._configs(request)
    assert exit_waves.ndim == (4 if configs else 3), exit_waves.shape
    folder = Path(request["folder"])
    np.save(folder / "exit_waves.npy", exit_waves)

    ctf = request.get("ctf")
    if ctf:
        worker.emit(progress=0.8, message="Forming the images")
        offsets, weights = _focal_samples(worker, ctf)
        transfer = worker._transfer(ctf, focal_spread=0.0 if len(offsets) > 1 else None)
        total = 0.0
        for i, (offset, weight) in enumerate(zip(offsets, weights)):
            transfer.defocus = ctf["defocus"] + float(offset)
            images = waves.apply_ctf(transfer).intensity()
            images = images.compute() if hasattr(images, "compute") else images
            data = np.asarray(images.array, np.float64)
            # Intensities are averaged over the configurations, never the waves.
            total = total + weight * (data.mean(axis=0) if configs else data)
            worker.emit(progress=0.8 + 0.15 * (i + 1) / len(offsets), message="Forming the images")
        np.save(folder / "images.npy", np.asarray(total, np.float32))

    last = exit_waves[..., -1, :, :]
    return {
        "shape": list(exit_waves.shape[-3:]),
        "configs": configs,
        "sampling": [float(x) for x in waves.sampling],
        "slice_thickness": float(potential.slice_thickness[0]),
        "thicknesses": [float(x) for x in potential.exit_thicknesses],
        "exit_intensity": float((np.abs(last) ** 2).mean()),
        "abtem": abtem.__version__,
    }


def defocus_series(request: dict, worker) -> dict:
    """An exit wave (``wave``: .npy, complex, [x, y] or [config, x, y]) through
    one CTF at each of the ``defoci`` (Å), as ``image`` forms an image:
    ``apply_ctf``, then the intensity, with the focal spread as an envelope or,
    for ``coherence`` "focal-series", a Gaussian average. The defocus values are
    one ensemble of the CTF, applied a chunk at a time; configurations are
    averaged as intensities. Saves the images as ``output`` (.npy, float32,
    [defocus, x, y]). Served (``serve``), so it reports no progress."""
    import abtem
    import numpy as np

    array = np.load(request["wave"], mmap_mode="r")
    if array.ndim not in (2, 3):
        raise ValueError(f"A wave is [x, y] or [config, x, y], not {list(array.shape)}")
    configs = array.shape[0] if array.ndim == 3 else 1
    offsets, weights = _focal_samples(worker, request)
    transfer = worker._transfer(request, focal_spread=0.0 if len(offsets) > 1 else None)
    defoci = np.asarray(request["defoci"], np.float64)
    chunk = max(1, int(CHUNK_BYTES // (len(offsets) * array.shape[-2] * array.shape[-1] * 8)))
    total = np.zeros((len(defoci), *array.shape[-2:]), np.float64)
    for k in range(configs):
        waves = abtem.Waves(
            np.array(array[k] if array.ndim == 3 else array), energy=request["energy"], sampling=tuple(request["sampling"]),
        )
        for start in range(0, len(defoci), chunk):
            part = defoci[start:start + chunk]
            values = (part[:, None] + offsets[None, :]).ravel()
            if len(values) == 1:
                transfer.defocus = float(values[0])
            else:
                transfer.defocus = abtem.distributions.from_values(values, ensemble_mean=False)
            result = waves.apply_ctf(transfer).intensity()
            result = result.compute() if hasattr(result, "compute") else result
            data = np.asarray(result.array, np.float64).reshape(len(part), len(offsets), *array.shape[-2:])
            total[start:start + len(part)] += np.einsum("j,ijxy->ixy", weights, data)
    images = (total / configs).astype(np.float32)
    np.save(request["output"], images)
    return {"path": request["output"], "shape": list(images.shape)}
