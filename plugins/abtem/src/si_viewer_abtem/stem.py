"""STEM imaging of a Potential: a probe scanned over it, annular detectors
integrated from its polar data afterwards (abTEM's FlexibleAnnularDetector),
and the post-processing the STEM quickstart applies to the images.

``register`` adds the routes; ``plugin.py`` calls it with what it shares.
"""

from __future__ import annotations

import base64
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel, Field, model_validator

from . import engine
from .potentials import MAX_GPTS, MAX_STACK_BYTES, to_signal
from .runs import ANGSTROM_PER_UM, clear_unfinished, finish_run, max_semiangle, new_run, wavelength

STEM_KIND = "stem"
ITEM_STEM = "stem"
LABEL_STEM = "STEM"
RESULT_IMAGE = "stem-image"
MAX_TILE = 64
QUANTITY = "Intensity (share of the beam)"
COUNTS = "Counts per pixel"
MIN_SAMPLING = 0.01  # Å
# Names that have a badge colour of their own; other detectors are "Detector".
KNOWN_LABELS = ("BF", "ABF", "ADF", "MAADF", "HAADF")
LABEL_DETECTOR = "Detector"
MAX_DETECTORS = 12


class Detector(BaseModel):
    """An annulus (mrad) to integrate the scattered beam over."""

    name: str = Field(min_length=1, max_length=24)
    inner: float = Field(ge=0, le=2000)
    outer: float = Field(gt=0, le=2000)

    @model_validator(mode="after")
    def _ordered(self):
        if self.inner >= self.outer:
            raise ValueError("The inner angle must be below the outer angle")
        return self


class StemSettings(BaseModel):
    """A probe scanned over a potential. ``start`` and ``end`` are fractions of
    the model's cell; the scan stays over that cell however much the potential
    is tiled. ``sampling`` (Å): None is the Nyquist sampling of the probe."""

    energy_kv: float = Field(300.0, ge=20, le=1000)
    semiangle: float = Field(20.0, gt=0, le=1000)  # mrad
    # abTEM's convention: positive defocus focuses into the specimen.
    defocus: float = Field(0.0, ge=-100000, le=100000)  # Å
    cs_um: float = Field(0.0, ge=-100000, le=100000)  # µm
    tile: list[int] = Field(default=[1, 1], min_length=2, max_length=2)
    start: list[float] = Field(default=[0.0, 0.0], min_length=2, max_length=2)
    end: list[float] = Field(default=[1.0, 1.0], min_length=2, max_length=2)
    sampling: float | None = Field(None, ge=MIN_SAMPLING, le=10)
    step: float = Field(1.0, ge=0.1, le=10)  # mrad per radial bin of the saved data
    detectors: list[Detector] = Field(min_length=1, max_length=MAX_DETECTORS)


class AddDetector(BaseModel):
    name: str = Field("Detector", min_length=1, max_length=24)
    inner: float = Field(ge=0, le=2000)
    outer: float = Field(gt=0, le=2000)

    @model_validator(mode="after")
    def _ordered(self):
        if self.inner >= self.outer:
            raise ValueError("The inner angle must be below the outer angle")
        return self


class PostSettings(BaseModel):
    """What the STEM quickstart does to an image, in its order: interpolate to
    ``interpolate`` Å, blur by the source size (Gaussian σ, Å), then Poisson
    noise at ``dose`` e⁻/Å². Each is off when None or 0."""

    interpolate: float | None = Field(None, ge=0, le=10)
    source_size: float = Field(0.0, ge=0, le=10)
    dose: float | None = Field(None, ge=0, le=1e12)
    seed: int | None = Field(None, ge=0, le=2**31 - 1)

    @model_validator(mode="after")
    def _something(self):
        if not (self.interpolate or self.source_size or self.dose):
            raise ValueError("Choose interpolation, a source size or a dose")
        return self


def nyquist_sampling(energy_kv: float, semiangle: float) -> float:
    """abTEM's ``probe.aperture.nyquist_sampling`` (Å)."""
    return wavelength(energy_kv) * 1000 / (4 * semiangle)


def scan_gpts(extent: float, sampling: float) -> int:
    """Positions along a scan of ``extent`` Å at ``sampling`` Å (abTEM's GridScan)."""
    return max(1, math.ceil(round(extent / sampling, 9)))


def label_of(name: str) -> str:
    return name if name in KNOWN_LABELS else LABEL_DETECTOR


def title_of(d: Detector | AddDetector) -> str:
    return f"{d.name} {d.inner:g}–{d.outer:g} mrad"


def _decode(result: dict) -> list[np.ndarray]:
    shape = result["shape"]
    return [np.frombuffer(base64.b64decode(x), np.float32).reshape(shape) for x in result["images"]]


def register(ctx, router, service, potential_object, item, item_file, run_under_way) -> None:
    """The STEM routes. ``potential_object(id)``, ``item(id, *kinds)`` and
    ``item_file(ds)`` are the plugin's own lookups (404 when it is not one)."""
    manifest = ctx.manifest

    def stem_job(potential=None):
        """The STEM run under way on ``potential`` (or on any)."""
        kind = f"{manifest.id}:{STEM_KIND}:"
        if potential is not None:
            kind += potential.id
        return next(
            (j for j in ctx.app.state.jobs.jobs.values() if j.active and (j.kind == kind or (potential is None and j.kind.startswith(kind)))),
            None,
        )

    def stem_object(dataset_id: str):
        ds = ctx.session.datasets.get(dataset_id)
        if ds is None or not ds.plugin or ds.plugin.get("id") != manifest.id or ds.plugin.get("item") != ITEM_STEM:
            raise HTTPException(404, f"Not a STEM run: {dataset_id}")
        return ds

    def plan(source, body: StemSettings) -> dict:
        """Check ``body`` against the potential it scans, before any run
        folder is made. HTTP 422 if it can't run. Returns the numbers the
        run needs: the cell, the scan's sampling and shape, the bins."""
        potential_settings = source.plugin["settings"]["potential"]
        if not all(1 <= n <= MAX_TILE for n in body.tile):
            raise HTTPException(422, f"Tiling: 1 to {MAX_TILE} per side")
        limit = max_semiangle(body.energy_kv, potential_settings["sampling"])
        if body.semiangle > limit:
            raise HTTPException(422, f"Convergence: the sampling allows up to {limit:.0f} mrad")
        for d in body.detectors:
            if d.outer > limit:
                raise HTTPException(422, f"{d.name}: outer {d.outer:g} mrad; the sampling allows up to {limit:.0f} mrad")
        for s, e in zip(body.start, body.end):
            if not (0 <= s < e <= 1):
                raise HTTPException(422, "Scan region: 0 ≤ start < end ≤ 1 of the cell")
        gpts = source.plugin["settings"]["gpts"]
        pixel = source.plugin["settings"]["sampling_xy"]
        cell = [n * s for n, s in zip(gpts, pixel)]
        sampling = body.sampling or nyquist_sampling(body.energy_kv, body.semiangle)
        shape = [scan_gpts((e - s) * c, sampling) for s, e, c in zip(body.start, body.end, cell)]
        tiled = [n * t for n, t in zip(gpts, body.tile)]
        if max(tiled) > MAX_GPTS:
            raise HTTPException(422, f"{tiled[0]} × {tiled[1]} pixels: the limit is {MAX_GPTS} per side")
        bins = int(limit // body.step)
        polar = shape[0] * shape[1] * bins * 4
        if polar > MAX_STACK_BYTES:
            raise HTTPException(
                422, f"{shape[0]} × {shape[1]} positions × {bins} bins is {polar / 1024**3:.1f} GB: the limit is {MAX_STACK_BYTES / 1024**3:.0f} GB",
            )
        return {"potential": potential_settings, "cell": cell, "sampling": sampling, "shape": shape, "limit": limit}

    def add_images(run, images, detectors, sampling) -> list:
        """The images of ``detectors`` (arrays [x, y] from abTEM) under
        ``run``, listed in the given order. Returns them in that order."""
        made = []
        # Each is added directly under the run, so the last added is listed first.
        for array, d in reversed(list(zip(images, detectors))):
            title = title_of(d)
            made.append(ctx.add_dataset(
                run, to_signal(array, sampling, title, QUANTITY), title,
                {"result": RESULT_IMAGE, "run": run.id, "detector": d.model_dump()},
                label=label_of(d.name),
            ))
        return made[::-1]

    @router.get("/potentials/{dataset_id}/stem")
    def get_stem(dataset_id: str) -> dict:
        """The STEM run under way on the potential, if any."""
        job = stem_job(potential_object(dataset_id))
        return {"job": job.summary(with_result=False) if job else None}

    @router.post("/potentials/{dataset_id}/stem")
    def post_stem(dataset_id: str, body: StemSettings) -> dict:
        """Scan a probe over the potential, as a job. Its result is a STEM run
        under the potential with an image for each detector; the polar data
        every annulus can be integrated from stays in the run."""
        source = potential_object(dataset_id)
        model = item(source.plugin["settings"]["model"], "model")
        entry = ctx.session.file_of(source)
        if not engine.status()["installed"]:
            raise HTTPException(409, "Install abTEM first")
        if stem_job(source) is not None:
            raise HTTPException(409, "This potential already has a STEM run under way")
        numbers = plan(source, body)
        potential_settings = numbers["potential"]

        if not run_under_way():
            clear_unfinished(entry.path)
        run_id, folder = new_run(entry.path)
        wanted = {**body.model_dump(), "sampling": numbers["sampling"], "potential": potential_settings}
        request = {
            **potential_settings,
            "model": str(item_file(model)),
            "folder": str(folder),
            "energy": body.energy_kv * 1e3,
            "semiangle": body.semiangle,
            "defocus": body.defocus,
            "cs": body.cs_um * ANGSTROM_PER_UM,
            "tile": body.tile,
            "start": body.start,
            "end": body.end,
            "scan_sampling": numbers["sampling"],
            "step": body.step,
            "detectors": [d.model_dump() for d in body.detectors],
            "workers": max(1, (os.cpu_count() or 2) - 2),
        }
        (folder / "request.json").write_text(json.dumps(request), encoding="utf-8")
        title = f"STEM {body.energy_kv:g} kV, {body.semiangle:g} mrad, Δf {body.defocus:g} Å"

        def on_result(result: dict) -> dict:
            current = ctx.session.datasets.get(source.id)
            if current is None:
                shutil.rmtree(folder, ignore_errors=True)
                raise ValueError("The potential was deleted during the run")
            settings = {
                **wanted,
                "result": STEM_KIND,
                "model": model.id,
                "potential_id": source.id,
                "run": f"runs/{run_id}",
                "gpts": result["shape"][:2],
                "bins": result["shape"][2],
                "tiled_gpts": result["gpts"],
                "scan_sampling": result["scan_sampling"],
                "scan_extent": result["scan_extent"],
                "step": result["step"],
                "max_angle": result["max_angle"],
                "positions": result["positions"],
                "abtem": result["abtem"],
            }
            images = np.load(folder / "images.npy")
            (folder / "images.npy").unlink()
            final = finish_run(folder, run_id, wanted, {
                "mode": STEM_KIND,
                "model": model.plugin["settings"]["file"],
                "outputs": {"polar": "polar.npy"},
                "axes": {
                    "order": ["x", "y", "radial"], "sampling": result["scan_sampling"], "extent": result["scan_extent"],
                    "radial_step": result["step"], "radial_units": "mrad", "units": "Å",
                },
                **{k: settings[k] for k in ("gpts", "bins", "max_angle", "positions", "abtem")},
            })
            run = ctx.add_item(entry, ITEM_STEM, LABEL_STEM, title, settings, f"{body.energy_kv:g} kV", parent=current)
            try:
                made = add_images(run, images, body.detectors, result["scan_sampling"])[-1]
                ctx.save_project(entry)
            except Exception:
                ctx.session._remove_tree(run)
                shutil.rmtree(final, ignore_errors=True)
                raise
            return {"file": entry.summary(ctx.session), "dataset_id": made.id}

        job = ctx.submit_process(
            f"{STEM_KIND}:{source.id}",
            f"STEM of {model.title}",
            [str(engine.python_path()), "-I", str(engine.WORKER), "stem", str(folder / "request.json")],
            on_result,
        )
        return job.summary()

    @router.post("/stems/{dataset_id}/detectors")
    def post_detector(dataset_id: str, body: AddDetector) -> dict:
        """An annulus integrated from the run's polar data: a new image under
        the run, saved. Returns the file and the image."""
        run = stem_object(dataset_id)
        settings = run.plugin["settings"]
        if body.outer > settings["max_angle"]:
            raise HTTPException(422, f"Outer angle: the run holds up to {settings['max_angle']:.0f} mrad")
        entry = ctx.session.file_of(run)
        polar = entry.path / settings["run"] / "polar.npy"
        if not polar.is_file():
            raise HTTPException(404, f"{settings['run']}/polar.npy is missing from the project")
        if not engine.status()["installed"]:
            raise HTTPException(409, "Install abTEM first")
        try:
            result = service.call("stem-integrate", {
                "polar": str(polar), "step": settings["step"], "sampling": settings["scan_sampling"],
                "limits": [[body.inner, body.outer]],
            })
        except engine.ServiceError as e:
            raise HTTPException(502, str(e))
        made = add_images(run, _decode(result), [body], settings["scan_sampling"])[0]
        try:
            ctx.save_project(entry)
        except Exception:
            ctx.session._remove(made)
            raise
        return {"file": entry.summary(ctx.session), "dataset_id": made.id}

    @router.post("/stem-images/{dataset_id}/post")
    def post_image_processing(dataset_id: str, body: PostSettings) -> dict:
        """A STEM image interpolated, blurred by the source size and given
        Poisson noise (the STEM quickstart's post-processing): a new image
        under the same run, saved. Returns the file and the image."""
        ds = ctx.session.datasets.get(dataset_id)
        if ds is None or not ds.plugin or ds.plugin.get("id") != manifest.id or ds.plugin.get("settings", {}).get("result") != RESULT_IMAGE:
            raise HTTPException(404, f"Not a STEM image: {dataset_id}")
        run = ctx.session.datasets.get(ds.parent_id)
        if not engine.status()["installed"]:
            raise HTTPException(409, "Install abTEM first")
        axes = ds.signal.axes_manager.signal_axes
        sampling = [float(a.scale) for a in axes]
        # The stored image is [y (down), x]; abTEM's is [x, y].
        array = np.ascontiguousarray(np.flipud(np.asarray(ds.signal.data, np.float32)).T)
        entry = ctx.session.file_of(ds)
        with tempfile.TemporaryDirectory(prefix="si-viewer-stem-") as folder:
            path = Path(folder) / "image.npy"
            np.save(path, array)
            try:
                result = service.call("stem-post", {
                    "image": str(path), "sampling": sampling, "interpolate": body.interpolate or None,
                    "source_size": body.source_size or None, "dose": body.dose or None, "seed": body.seed,
                })
            except engine.ServiceError as e:
                raise HTTPException(502, str(e))
        out = _decode(result)[0]
        parts = [
            f"{body.interpolate:g} Å" if body.interpolate else None,
            f"σ {body.source_size:g} Å" if body.source_size else None,
            f"{body.dose:g} e⁻/Å²" if body.dose else None,
        ]
        title = f"{ds.title} · {', '.join(p for p in parts if p)}"
        made = ctx.add_dataset(
            run, to_signal(out, result["sampling"], title, COUNTS if body.dose else ds.signal.metadata.get_item("Signal.quantity", QUANTITY)),
            title, {"result": RESULT_IMAGE, "run": ds.plugin["settings"].get("run"), "from": ds.id, "post": body.model_dump()},
            label=ds.plugin.get("label"),
        )
        try:
            ctx.save_project(entry)
        except Exception:
            ctx.session._remove(made)
            raise
        return {"file": entry.summary(ctx.session), "dataset_id": made.id}
