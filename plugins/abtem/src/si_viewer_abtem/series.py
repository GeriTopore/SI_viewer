"""Thickness series and defocus series of HRTEM.

A *thickness series* sends a plane wave through a Potential keeping the exit
wave after every n slices (abTEM's ``exit_planes``); a CTF forms the image of
each. A *defocus series* applies one CTF at a range of defocus values to an
HRTEM run's exit wave, without repeating multislice.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel, Field

from . import engine
from .ctfs import worker_request as ctf_request
from .potentials import HIDDEN_PROCESSES, MAX_GPTS, MAX_STACK_BYTES, configs_of, normalized, stack_bytes, to_stack
from .runs import Coherence, clear_unfinished, finish_run, new_run

PLUGIN_ID = "abtem"
THICKNESS_KIND = "thickness-series"
ITEM_SERIES = "series"
LABEL_SERIES = "Thickness series"
LABEL_DEFOCUS = "Defocus series"
# The objects these series are made from (as the plugin's own).
ITEM_MODEL = "model"
ITEM_RUN = "run"
HRTEM_KIND = "hrtem"
MAX_STEP = 1000
MAX_DEFOCI = 100
# A defocus series is one request to the long-running worker.
DEFOCUS_TIMEOUT = 600
# The planes through thickness open on the last one; a sum of phases means nothing.
THICKNESS_STACK = {"frame": "Thickness", "frames": "thicknesses", "sum": "Sum", "start": "last", "planes": True}
THICKNESS_PHASE_STACK = {**THICKNESS_STACK, "projections": False}
DEFOCUS_STACK = {"frame": "Defocus", "frames": "defoci", "sum": "Sum", "planes": True}
IMAGE_QUANTITY = "Intensity (incident beam = 1)"


class ThicknessSettings(BaseModel):
    """A plane wave through a potential, kept every ``step`` slices. With
    ``ctf`` (a CTF of the project), its energy, and the image it forms of
    every plane."""

    energy_kv: float = Field(300.0, ge=20, le=1000)  # without a CTF
    ctf: str | None = None
    coherence: Coherence = "envelope"
    step: int = Field(1, ge=1, le=MAX_STEP)  # slices


class DefocusSettings(BaseModel):
    """One CTF at the defocus values ``start``, ``start + step``, … up to ``stop``."""

    ctf: str
    coherence: Coherence = "envelope"
    start: float = Field(ge=-1e5, le=1e5)  # Å
    stop: float = Field(ge=-1e5, le=1e5)  # Å
    step: float = Field(gt=0, le=1e5)  # Å


def defocus_values(start: float, stop: float, step: float) -> list[float]:
    """``start``, ``start + step``, … up to ``stop`` (Å). ValueError if there
    are fewer than two or more than ``MAX_DEFOCI``."""
    n = math.floor((stop - start) / step + 1e-9) + 1
    if n < 2:
        raise ValueError("Defocus: at least two values (from, to and step)")
    if n > MAX_DEFOCI:
        raise ValueError(f"Defocus: {n} values, the limit is {MAX_DEFOCI}")
    return [round(start + i * step, 6) for i in range(n)]


def exit_planes_of(slices: int, step: int) -> list[int]:
    """The slice indices a thickness series keeps: the incident wave (-1), then
    every ``step`` slices (those that fit)."""
    return [-1, *range(step - 1, slices, step)]


def to_series_stack(array, sampling, spacing: float, title: str, quantity: str, axis: str, units: str = "Å", offset: float = 0.0):
    """A stack from abTEM's [plane, x, y] array, its planes along ``axis``."""
    signal = to_stack(array, sampling, spacing, title, quantity)
    a = signal.axes_manager.navigation_axes[0]
    a.name, a.units, a.scale, a.offset = axis, units, float(spacing), float(offset)
    return signal


def register_series(ctx, router, helpers: dict) -> None:
    """The series routes. ``helpers`` is the plugin's ``register`` scope: the
    functions ``item``, ``item_file``, ``potential_object``, ``ctf_object``,
    ``ctf_settings_of``, ``check_ctf_for`` and ``run_under_way``, and ``service``."""
    item = helpers["item"]
    item_file = helpers["item_file"]
    potential_object = helpers["potential_object"]
    ctf_object = helpers["ctf_object"]
    ctf_settings_of = helpers["ctf_settings_of"]
    check_ctf_for = helpers["check_ctf_for"]
    run_under_way = helpers["run_under_way"]
    service = helpers["service"]

    def series_job(potential):
        """The thickness series under way on ``potential``, if any."""
        kind = f"{PLUGIN_ID}:{THICKNESS_KIND}:{potential.id}"
        return next((j for j in ctx.app.state.jobs.jobs.values() if j.kind == kind and j.active), None)

    @router.get("/potentials/{dataset_id}/thickness-series")
    def get_thickness_series(dataset_id: str) -> dict:
        """The thickness series under way on the potential, if any."""
        job = series_job(potential_object(dataset_id))
        return {"job": job.summary(with_result=False) if job else None}

    @router.post("/potentials/{dataset_id}/thickness-series")
    def post_thickness_series(dataset_id: str, body: ThicknessSettings) -> dict:
        """Run a plane wave through the potential keeping its exit wave every
        ``step`` slices, as a job. Its result is a Thickness series under the
        potential: the exit wave's Amplitude and Phase and, with a CTF, the
        Image it forms, as stacks through thickness."""
        source = potential_object(dataset_id)
        model = item(source.plugin["settings"]["model"], ITEM_MODEL)
        entry = ctx.session.file_of(source)
        if not engine.status()["installed"]:
            raise HTTPException(409, "Install abTEM first")
        if series_job(source) is not None:
            raise HTTPException(409, "This potential already has a thickness series under way")
        potential_settings = normalized(source.plugin["settings"]["potential"])
        configs = configs_of(potential_settings)
        slices = source.plugin["settings"]["slices"]
        if body.step > slices:
            raise HTTPException(422, f"Step: 1 to {slices} slices")
        exit_planes = exit_planes_of(slices, body.step)
        ctf = ctf_object(body.ctf) if body.ctf else None
        energy_kv = ctf_settings_of(ctf).energy_kv if ctf else body.energy_kv
        s = check_ctf_for(ctf, energy_kv, potential_settings["sampling"]) if ctf else None
        nx, ny = source.plugin["settings"]["gpts"]
        planes = len(exit_planes)
        if max(nx, ny) > MAX_GPTS:
            raise HTTPException(422, f"{nx} × {ny} pixels: the limit is {MAX_GPTS} per side")
        # Complex values, two floats per pixel of each plane (of each frozen phonon configuration).
        wave_bytes = 2 * stack_bytes(nx, ny, planes) * (configs or 1)
        if wave_bytes > MAX_STACK_BYTES:
            shape = f"{nx} × {ny} × {planes} planes" + (f" × {configs} configs" if configs else "")
            raise HTTPException(422, f"{shape} is {wave_bytes / 1024**3:.1f} GB: the limit is {MAX_STACK_BYTES / 1024**3:.0f} GB")
        coherence = "envelope" if s is None or s.focal_spread == 0 else body.coherence

        if not run_under_way():
            clear_unfinished(entry.path)
        run_id, folder = new_run(entry.path)
        wanted = {
            "energy_kv": energy_kv, "potential": potential_settings, "step": body.step, "ctf": body.ctf, "coherence": coherence,
        }
        request = {
            **potential_settings,
            "model": str(item_file(model)),
            "folder": str(folder),
            "energy": energy_kv * 1e3,
            "exit_planes": exit_planes,
            "ctf": {**ctf_request(s), "coherence": coherence} if s else None,
            "workers": max(1, (os.cpu_count() or 2) - 2),
        }
        (folder / "request.json").write_text(json.dumps(request), encoding="utf-8")
        title = f"Thickness series {energy_kv:g} kV" + (f", every {body.step} slices" if body.step > 1 else "")

        def on_result(result: dict) -> dict:
            current = ctx.session.datasets.get(source.id)
            if current is None:
                shutil.rmtree(folder, ignore_errors=True)
                raise ValueError("The potential was deleted during the run")
            result_configs = result.get("configs", 0)
            spacing = body.step * result["slice_thickness"]
            settings = {
                **wanted,
                "result": THICKNESS_KIND,
                "model": model.id,
                "potential_id": source.id,
                "run": f"runs/{run_id}",
                # exit_waves.npy is [plane, x, y], or [config, plane, x, y] with frozen phonons.
                "gpts": result["shape"][1:],
                "planes": result["shape"][0],
                "configs": result_configs,
                "sampling_xy": result["sampling"],
                "slices": slices,
                "slice_thickness": result["slice_thickness"],
                "thicknesses": result["thicknesses"],
                "exit_intensity": result["exit_intensity"],
                "ctf_settings": s.model_dump() if s else None,
                "abtem": result["abtem"],
            }
            images_file = folder / "images.npy"
            images = np.load(images_file) if images_file.is_file() else None
            images_file.unlink(missing_ok=True)
            final = finish_run(folder, run_id, wanted, {
                "mode": THICKNESS_KIND,
                "model": model.plugin["settings"]["file"],
                "outputs": {"exit_waves": "exit_waves.npy"},
                "axes": {
                    "order": (["config"] if result_configs else []) + ["plane", "x", "y"],
                    "sampling": result["sampling"], "slice_thickness": result["slice_thickness"], "step": body.step,
                    "thicknesses": result["thicknesses"], "units": "Å",
                },
                **{k: settings[k] for k in ("gpts", "planes", "configs", "slices", "exit_intensity", "abtem")},
            })
            waves = np.load(final / "exit_waves.npy", mmap_mode="r")
            run = ctx.add_item(entry, ITEM_SERIES, LABEL_SERIES, title, settings, f"{settings['planes']} planes", parent=current)
            sampling = result["sampling"]
            # With frozen phonons, the amplitude and phase shown are of one wave, the first configuration's.
            suffix = " (config 1)" if result_configs else ""
            wave = np.array(waves[0] if result_configs else waves)
            outputs = [
                (f"Exit wave phase{suffix}", np.angle(wave), "Phase (rad)", "phase", "Phase", THICKNESS_PHASE_STACK),
                (f"Exit wave amplitude{suffix}", np.abs(wave), "Amplitude", "amplitude", "Amplitude", THICKNESS_STACK),
            ]
            if images is not None:
                image_title = (
                    f"Image · {ctf.title}, Δf {s.defocus:g} Å" + (", focal series" if coherence == "focal-series" else "")
                    + (f", {result_configs} phonon configs" if result_configs else "")
                )
                outputs.append((image_title, images, IMAGE_QUANTITY, "image", "Image", THICKNESS_STACK))
            made = None
            try:
                # Each is added directly under the series, so the last added is listed first.
                for name, data, quantity, output, label, names in outputs:
                    made = ctx.add_dataset(
                        run, to_series_stack(data, sampling, spacing, name, quantity, "t"), name,
                        {"result": f"{THICKNESS_KIND}-{output}", "run": run.id}, label=label,
                        stack=names, hidden_processes=HIDDEN_PROCESSES,
                    )
                ctx.save_project(entry)
            except Exception:
                ctx.session._remove_tree(run)
                shutil.rmtree(final, ignore_errors=True)
                raise
            return {"file": entry.summary(ctx.session), "dataset_id": made.id}

        job = ctx.submit_process(
            f"{THICKNESS_KIND}:{source.id}",
            f"Thickness series of {model.title}",
            [str(engine.python_path()), "-I", str(engine.WORKER), "thickness_series", str(folder / "request.json")],
            on_result,
        )
        return job.summary()

    @router.post("/runs/{dataset_id}/defocus-series")
    def post_defocus_series(dataset_id: str, body: DefocusSettings) -> dict:
        """Apply a CTF at a range of defocus values to an HRTEM run's exit wave:
        a stack of images through defocus, added under the run and saved.
        Returns the file and the stack."""
        run = item(dataset_id, ITEM_RUN)
        settings = run.plugin["settings"]
        if settings.get("result") != HRTEM_KIND:
            raise HTTPException(404, f"Not an HRTEM run: {dataset_id}")
        if not engine.status()["installed"]:
            raise HTTPException(409, "Install abTEM first")
        entry = ctx.session.file_of(run)
        ctf = ctf_object(body.ctf)
        s = check_ctf_for(ctf, settings["energy_kv"], settings["potential"]["sampling"])
        try:
            defoci = defocus_values(body.start, body.stop, body.step)
        except ValueError as e:
            raise HTTPException(422, str(e))
        nx, ny = settings["gpts"][-2:]
        if stack_bytes(nx, ny, len(defoci)) > MAX_STACK_BYTES:
            raise HTTPException(
                422, f"{nx} × {ny} × {len(defoci)} images is {stack_bytes(nx, ny, len(defoci)) / 1024**3:.1f} GB: "
                f"the limit is {MAX_STACK_BYTES / 1024**3:.0f} GB",
            )
        coherence = "envelope" if s.focal_spread == 0 else body.coherence
        wave = entry.path / settings["run"] / "exit_wave.npy"
        if not wave.is_file():
            raise HTTPException(404, f"{settings['run']}/exit_wave.npy is missing from the project")
        folder = Path(tempfile.gettempdir()) / "si-viewer-abtem"
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / f"defocus-series-{uuid.uuid4().hex[:10]}.npy"
        try:
            try:
                result = service.call("defocus_series", {
                    **ctf_request(s), "wave": str(wave), "sampling": settings["sampling_xy"], "coherence": coherence,
                    "defoci": defoci, "output": str(output),
                }, timeout=DEFOCUS_TIMEOUT)
            except engine.ServiceError as e:
                raise HTTPException(502, str(e))
            images = np.load(result["path"])
        finally:
            output.unlink(missing_ok=True)
        title = (
            f"Defocus series · {ctf.title}, Δf {defoci[0]:g} to {defoci[-1]:g} Å"
            + (", focal series" if coherence == "focal-series" else "")
            + (f", {settings['configs']} phonon configs" if settings.get("configs") else "")
        )
        made = ctx.add_dataset(
            run,
            to_series_stack(images, settings["sampling_xy"], body.step, title, IMAGE_QUANTITY, "Δf", offset=defoci[0]),
            title,
            {"result": f"{HRTEM_KIND}-defocus-series", "run": run.id, "ctf": ctf.id, "ctf_settings": s.model_dump(),
             "coherence": coherence, "start": defoci[0], "stop": defoci[-1], "step": body.step, "defoci": len(defoci)},
            label=LABEL_DEFOCUS, stack=DEFOCUS_STACK, hidden_processes=HIDDEN_PROCESSES,
        )
        try:
            ctx.save_project(entry)
        except Exception:
            ctx.session._remove(made)
            raise
        return {"file": entry.summary(ctx.session), "dataset_id": made.id}
