"""The abTEM plugin: simulation projects and their structures.

A project is a folder::

    <name>/
      <name>.hspy        the analysis file (tree, views, results)
      structures/        structures as loaded
      models/            oriented supercells made from them
      runs/              simulation runs
"""

from __future__ import annotations

import atexit
import base64
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from si_viewer.plugins import PluginContext, PluginManifest

from . import __version__, engine
from .models import build_model, model_atoms, model_summary, preview, write_model
from .potentials import (
    HIDDEN_PROCESSES, MAX_GPTS, MAX_STACK_BYTES, STACK_NAMES, UNITS, PotentialSettings, configs_of, grid, normalized,
    stack_bytes, title_of, to_signal, to_stack,
)
from .runs import (
    ANGSTROM_PER_UM, HrtemSettings, ImageSettings, WaveSettings, clear_unfinished, finish_run, max_semiangle, new_run,
)
from .ctfs import CtfSettings, title_of as ctf_title, to_signal as ctf_signal, worker_request as ctf_request
from .sections import SectionLine, section
from .series import THICKNESS_KIND, register_series
from .structures import StructureError, read_structure, structure_atoms

manifest = PluginManifest(
    id="abtem", name="abTEM", version=__version__, frontend=Path(__file__).parent / "frontend",
)
PROJECT_TYPE = "simulation"
ITEM_STRUCTURE = "structure"
ITEM_MODEL = "model"
RESULT_POTENTIAL = "potential"
POTENTIAL_KIND = "potential"
LABEL_POTENTIAL = "Potential"
ITEM_RUN = "run"
HRTEM_KIND = "hrtem"
ITEM_CTF = "ctf"
LABEL_CTF = "CTF"
LABEL_IMAGE = "Image"
CTF_COLORMAP = "rdbu"
# Image processes that mean nothing for a transfer function.
CTF_HIDDEN_PROCESSES = [
    "rebin", "filters", "mask", "fourier", "columns", "sublattices", "neighbourhoods", "pld", "pca", "curvefit",
]
ITEM_WAVE = "wavefunction"
WAVE_KIND = "wavefunction"
MAX_TILE = 64
# A wave's planes through depth; they open on the exit plane. A sum of phases means nothing.
WAVE_INTENSITY_STACK = {"frame": "Plane", "frames": "planes", "sum": "Sum", "start": "last", "planes": True}
WAVE_PHASE_STACK = {**WAVE_INTENSITY_STACK, "projections": False}


class PathBody(BaseModel):
    path: str


class ModelBody(BaseModel):
    uvw: list[int] = Field(min_length=3, max_length=3)
    repeats: list[int] | None = Field(default=None, min_length=3, max_length=3)
    seed: int = Field(default=0, ge=0)


def _zone_name(uvw) -> str:
    return "[" + " ".join(str(x) for x in uvw) + "]" if any(abs(x) > 9 or x < 0 for x in uvw) else "[" + "".join(str(x) for x in uvw) + "]"


def add_model(ctx: PluginContext, entry, structure, uvw, repeats, seed: int = 0):
    """Build the model of ``structure`` along [uvw], write it to the project's
    ``models/`` folder and add it under the structure, saved."""
    source = entry.path / structure.plugin["settings"]["file"]
    model, info = build_model(source, uvw, repeats, seed)
    folder = entry.path / "models"
    folder.mkdir(exist_ok=True)
    nx, ny, nz = info["repeats"]
    zone = _zone_name(info["uvw"])
    target = _free_name(folder, f"{structure.title} {zone} {nx}x{ny}x{nz}.extxyz")
    write_model(model, target)
    settings = {
        **model_summary(model),
        "file": target.relative_to(entry.path).as_posix(),
        "structure": structure.id,
        "uvw": info["uvw"],
        "axes": info["axes"],
        "repeats": info["repeats"],
        "strain_percent": info["strain_percent"],
        "seed": seed if info["mixed"] else None,
    }
    ds = ctx.add_item(
        entry, ITEM_MODEL, "Model", f"{zone} {nx}×{ny}×{nz}", settings, f"{len(model)} atoms", parent=structure,
    )
    try:
        ctx.save_project(entry)
    except Exception:
        ctx.session._remove(ds)
        target.unlink(missing_ok=True)
        raise
    return ds


def create_project(ctx: PluginContext, path: str | Path):
    """A new, empty project in the folder ``path`` (made here). ValueError if
    it can't be: the folder is in use or its parent doesn't exist."""
    folder = Path(path).expanduser().resolve()
    if not folder.parent.is_dir():
        raise ValueError(f"The folder {folder.parent} does not exist")
    existed = folder.exists()
    if existed and (not folder.is_dir() or any(folder.iterdir())):
        raise ValueError(f"{folder.name} already exists: choose another name")
    (folder / "structures").mkdir(parents=True)
    (folder / "runs").mkdir()
    entry = ctx.add_project(folder, PROJECT_TYPE, folder.name)
    try:
        ctx.save_project(entry)
    except Exception:
        ctx.session.close_file(entry.id)
        shutil.rmtree(folder if not existed else folder / "structures", ignore_errors=True)
        if existed:
            shutil.rmtree(folder / "runs", ignore_errors=True)
        raise
    return entry


def _free_name(folder: Path, name: str) -> Path:
    """``folder/name``, numbered if that is taken."""
    target = folder / name
    stem, suffix = target.stem, target.suffix
    n = 2
    while target.exists():
        target = folder / f"{stem} {n}{suffix}"
        n += 1
    return target


def add_structure(ctx: PluginContext, entry, path: str | Path):
    """Read the CIF at ``path``, copy it into the project and add it to the
    tree, saved. StructureError (nothing changed) if it can't be read."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise StructureError(f"File not found: {source}")
    info = read_structure(source)
    folder = entry.path / "structures"
    folder.mkdir(exist_ok=True)
    copy = _free_name(folder, source.name)
    shutil.copy2(source, copy)
    detail = f"{info['atoms']} atoms"
    settings = {**info, "file": copy.relative_to(entry.path).as_posix(), "source": str(source)}
    ds = ctx.add_item(entry, ITEM_STRUCTURE, "Structure", source.stem, settings, detail)
    try:
        ctx.save_project(entry)
    except Exception:
        ctx.session._remove(ds)
        copy.unlink(missing_ok=True)
        raise
    return ds


def register(ctx: PluginContext) -> None:
    router = APIRouter()

    def project(file_id: str):
        entry = ctx.session.files.get(file_id)
        if entry is None or not entry.plugin or entry.plugin.get("id") != manifest.id:
            raise HTTPException(404, f"Not a simulation project: {file_id}")
        return entry

    def item(dataset_id: str, *kinds: str):
        ds = ctx.session.datasets.get(dataset_id)
        if ds is None or not ds.plugin or ds.plugin.get("id") != manifest.id or ds.plugin.get("item") not in kinds:
            raise HTTPException(404, f"Not a {' or '.join(kinds)}: {dataset_id}")
        return ds

    def item_file(ds) -> Path:
        path = ctx.session.file_of(ds).path / ds.plugin["settings"]["file"]
        if not path.is_file():
            raise HTTPException(404, f"{ds.plugin['settings']['file']} is missing from the project")
        return path

    def install_job():
        """The install of abTEM under way, if any."""
        kind = f"{manifest.id}:{engine.INSTALL_KIND}"
        return next((j for j in ctx.app.state.jobs.jobs.values() if j.kind == kind and j.active), None)

    @router.get("/engine")
    def get_engine() -> dict:
        """Whether abTEM is installed, and the install under way."""
        job = install_job()
        return {**engine.status(), "job": job.summary(with_result=False) if job else None}

    @router.post("/engine/install")
    def post_engine_install() -> dict:
        """Install abTEM into its own environment, as a job."""
        if install_job() is not None:
            raise HTTPException(409, "abTEM is already being installed")
        if engine.status()["installed"]:
            raise HTTPException(409, "abTEM is already installed")
        job = ctx.submit_process(engine.INSTALL_KIND, f"Install abTEM {engine.ABTEM_VERSION}", engine.setup_command())
        return job.summary()

    @router.post("/projects")
    def post_project(body: PathBody) -> dict:
        """Create a project in the new folder ``path``. Returns its file."""
        try:
            entry = create_project(ctx, body.path)
        except (ValueError, OSError) as e:
            raise HTTPException(422, str(e))
        return {"file": entry.summary(ctx.session)}

    @router.post("/projects/{file_id}/structures")
    def post_structure(file_id: str, body: PathBody) -> dict:
        """Add the CIF at ``path`` to the project. Returns its file and the new item."""
        entry = project(file_id)
        try:
            ds = add_structure(ctx, entry, body.path)
        except (StructureError, OSError) as e:
            raise HTTPException(422, str(e))
        return {"file": entry.summary(ctx.session), "dataset_id": ds.id}

    @router.get("/structures/{dataset_id}/atoms")
    def get_atoms(dataset_id: str) -> dict:
        """The atoms of a structure item, to draw (see ``structure_atoms``)."""
        ds = item(dataset_id, ITEM_STRUCTURE, ITEM_MODEL)
        path = item_file(ds)
        try:
            return model_atoms(path) if ds.plugin["item"] == ITEM_MODEL else structure_atoms(path)
        except StructureError as e:
            raise HTTPException(422, str(e))

    def potentials_of(model) -> list:
        return [
            d for d in ctx.session.siblings_in_file(model.file_id)
            if d.parent_id == model.id and d.plugin and d.plugin.get("label") == LABEL_POTENTIAL
        ]

    def potential_object(dataset_id: str):
        """A Potential (a stack of slices under its model)."""
        ds = ctx.session.datasets.get(dataset_id)
        if ds is None or not ds.plugin or ds.plugin.get("id") != manifest.id or ds.plugin.get("label") != LABEL_POTENTIAL:
            raise HTTPException(404, f"Not a potential: {dataset_id}")
        return ds

    def potential_job(model):
        """The potential being computed for ``model``, if any."""
        kind = f"{manifest.id}:{POTENTIAL_KIND}:{model.id}"
        return next((j for j in ctx.app.state.jobs.jobs.values() if j.kind == kind and j.active), None)

    @router.get("/models/{dataset_id}/potential")
    def get_potential(dataset_id: str) -> dict:
        """The model's potentials, and the one being built."""
        model = item(dataset_id, ITEM_MODEL)
        job = potential_job(model)
        return {
            "results": [{"dataset_id": d.id, "settings": d.plugin["settings"]["potential"]} for d in potentials_of(model)],
            "job": job.summary(with_result=False) if job else None,
            "max_gpts": MAX_GPTS,
        }

    @router.post("/models/{dataset_id}/potential")
    def post_potential(dataset_id: str, body: PotentialSettings) -> dict:
        """Build the potential's slices as a job. Its result is a Potential (a
        stack, whose Sum is the projected potential) under the model. The same
        settings again just give that Potential."""
        model = item(dataset_id, ITEM_MODEL)
        entry = ctx.session.file_of(model)
        wanted = body.model_dump()
        # Potentials made before frozen phonons have none of their settings.
        same = next((d for d in potentials_of(model) if normalized(d.plugin["settings"]["potential"]) == wanted), None)
        if same is not None:
            result = {"file": entry.summary(ctx.session), "dataset_id": same.id}
            return {"id": None, "status": "done", "progress": 1.0, "message": "", "error": None, "result": result}
        if not engine.status()["installed"]:
            raise HTTPException(409, "Install abTEM first")
        if potential_job(model) is not None:
            raise HTTPException(409, "This model's potential is already being computed")
        nx, ny, slices = grid(model.plugin["settings"]["cell"], body)
        if max(nx, ny) > MAX_GPTS:
            raise HTTPException(422, f"{nx} × {ny} pixels: the limit is {MAX_GPTS} per side")
        if stack_bytes(nx, ny, slices) > MAX_STACK_BYTES:
            size = stack_bytes(nx, ny, slices) / 1024**3
            raise HTTPException(422, f"{nx} × {ny} × {slices} slices is {size:.1f} GB: the limit is {MAX_STACK_BYTES / 1024**3:.0f} GB")

        folder = Path(tempfile.gettempdir()) / "si-viewer-abtem" / model.id
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        request = {
            **wanted,
            "model": str(item_file(model)),
            "output": str(folder / "slices.npy"),
            "workers": max(1, (os.cpu_count() or 2) - 2),
        }
        (folder / "request.json").write_text(json.dumps(request), encoding="utf-8")

        def on_result(result: dict) -> dict:
            try:
                current = ctx.session.datasets.get(model.id)
                if current is None:
                    raise ValueError("The model was deleted while its potential was computed")
                array = np.load(result["path"])
                title = title_of(body)
                thickness = result["slice_thickness"][0]
                settings = {
                    "result": RESULT_POTENTIAL,
                    "potential": wanted,
                    "model": model.id,
                    "gpts": result["shape"][1:],
                    "sampling_xy": result["sampling"],
                    "slices": result["shape"][0],
                    "slice_thickness": thickness,
                    "units": UNITS,
                    "abtem": result["abtem"],
                }
                made = ctx.add_dataset(
                    current, to_stack(array, result["sampling"], thickness, title), title, settings,
                    label=LABEL_POTENTIAL, stack=STACK_NAMES, hidden_processes=HIDDEN_PROCESSES,
                )
                ctx.save_project(entry)
                return {"file": entry.summary(ctx.session), "dataset_id": made.id}
            finally:
                shutil.rmtree(folder, ignore_errors=True)

        job = ctx.submit_process(
            f"{POTENTIAL_KIND}:{model.id}",
            f"Potential of {model.title}",
            [str(engine.python_path()), "-I", str(engine.WORKER), "potential", str(folder / "request.json")],
            on_result,
        )
        return job.summary()

    def hrtem_job(potential=None):
        """The HRTEM run under way on ``potential`` (or on any)."""
        kind = f"{manifest.id}:{HRTEM_KIND}:"
        if potential is not None:
            kind += potential.id
        return next(
            (j for j in ctx.app.state.jobs.jobs.values() if j.active and (j.kind == kind or (potential is None and j.kind.startswith(kind)))),
            None,
        )

    service = engine.Service()
    atexit.register(service.close)

    def evaluate_ctf(body: CtfSettings, image: bool = True) -> dict:
        """``body``'s CTF from abTEM: its profiles, landmarks and (if
        ``image``) 2D form."""
        if not engine.status()["installed"]:
            raise HTTPException(409, "Install abTEM first")
        try:
            return service.call("ctf", {**ctf_request(body), **({} if image else {"image_points": 0})})
        except engine.ServiceError as e:
            raise HTTPException(502, str(e))

    def ctf_object(dataset_id: str):
        """A CTF (an image of the 2D CTF)."""
        ds = ctx.session.datasets.get(dataset_id)
        if ds is None or not ds.plugin or ds.plugin.get("id") != manifest.id or ds.plugin.get("label") != LABEL_CTF:
            raise HTTPException(404, f"Not a CTF: {dataset_id}")
        return ds

    @router.post("/projects/{file_id}/ctfs")
    def post_ctf(file_id: str, body: CtfSettings) -> dict:
        """A new CTF in the project, with ``body``'s settings: an image of the
        2D CTF. Returns its file and the new dataset."""
        entry = project(file_id)
        result = evaluate_ctf(body)
        title = ctf_title(body)
        ds = ctx.add_dataset(
            None, ctf_signal(result, title), title, {"result": ITEM_CTF, **body.model_dump()},
            label=LABEL_CTF, hidden_processes=CTF_HIDDEN_PROCESSES, entry=entry, colormap=CTF_COLORMAP,
        )
        try:
            ctx.save_project(entry)
        except Exception:
            ctx.session._remove(ds)
            raise
        return {"file": entry.summary(ctx.session), "dataset_id": ds.id}

    @router.put("/ctfs/{dataset_id}")
    def put_ctf(dataset_id: str, body: CtfSettings, save: bool = True) -> dict:
        """Change a CTF's settings and redraw its image (saved with the
        project unless ``save`` is false). Returns its file."""
        ds = ctf_object(dataset_id)
        result = evaluate_ctf(body)
        ds.refresh_data(ctf_signal(result, ds.title))
        ds.plugin = {**ds.plugin, "settings": {"result": ITEM_CTF, **body.model_dump()}}
        entry = ctx.session.file_of(ds)
        if save:
            ctx.save_project(entry)
        return {"file": entry.summary(ctx.session)}

    @router.post("/ctfs/evaluate")
    def post_ctf_evaluate(body: CtfSettings, image: bool = True) -> dict:
        """``body``'s CTF from abTEM: its profiles, landmarks and (if ``image``) 2D form."""
        return evaluate_ctf(body, image)

    @router.post("/stacks/{dataset_id}/section")
    def post_section(dataset_id: str, body: SectionLine) -> dict:
        """A stack of the plugin (a Potential, a wave's planes) cut along a
        line through all its depth, with its spatial filter: ``data``
        (float32, base64) as [plane, position], and the steps in Å."""
        import base64

        from si_viewer.processing.filters import filter_image
        from si_viewer.session import DatasetKind

        ds = ctx.session.datasets.get(dataset_id)
        if ds is None or not ds.plugin or ds.plugin.get("id") != manifest.id or ds.kind is not DatasetKind.STACK:
            raise HTTPException(404, f"Not a simulated stack: {dataset_id}")
        data = np.asarray(ds.signal.data)
        if ds.filter_spec.has_spatial:
            data = np.stack([filter_image(plane, ds.filter_spec) for plane in data])
        try:
            values, n = section(data, body)
        except ValueError as e:
            raise HTTPException(422, str(e))
        z = ds.signal.axes_manager.navigation_axes[0]
        sx, sy = (a.scale for a in ds.signal.axes_manager.signal_axes)
        step = float(np.hypot((body.x1 - body.x0) * sx, (body.y1 - body.y0) * sy) / n)
        return {
            "shape": list(values.shape),
            "data": base64.b64encode(values.tobytes()).decode("ascii"),
            "step": step,
            "z_step": float(z.scale),
            "units": "Å",
        }

    def run_under_way() -> bool:
        """Whether any run is writing to a temporary run folder."""
        kinds = tuple(f"{manifest.id}:{kind}:" for kind in (HRTEM_KIND, WAVE_KIND, "stem", THICKNESS_KIND))
        return any(j.active and j.kind.startswith(kinds) for j in ctx.app.state.jobs.jobs.values())

    def wave_job(potential):
        """The wavefunction being propagated through ``potential``, if any."""
        kind = f"{manifest.id}:{WAVE_KIND}:{potential.id}"
        return next((j for j in ctx.app.state.jobs.jobs.values() if j.kind == kind and j.active), None)

    @router.get("/potentials/{dataset_id}/wavefunction")
    def get_wavefunction(dataset_id: str) -> dict:
        """The wavefunction under way on the potential, if any."""
        job = wave_job(potential_object(dataset_id))
        return {"job": job.summary(with_result=False) if job else None}

    @router.post("/potentials/{dataset_id}/wavefunction")
    def post_wavefunction(dataset_id: str, body: WaveSettings) -> dict:
        """Send a plane wave or probe through the potential, as a job. Its
        result is a Wavefunction under the potential, with its intensity and
        phase as stacks through depth; the complex wave stays in the run."""
        source = potential_object(dataset_id)
        model = item(source.plugin["settings"]["model"], ITEM_MODEL)
        entry = ctx.session.file_of(source)
        if not engine.status()["installed"]:
            raise HTTPException(409, "Install abTEM first")
        if wave_job(source) is not None:
            raise HTTPException(409, "This potential already has a wavefunction under way")
        potential_settings = normalized(source.plugin["settings"]["potential"])
        configs = configs_of(potential_settings)
        if not all(1 <= n <= MAX_TILE for n in body.tile):
            raise HTTPException(422, f"Tiling: 1 to {MAX_TILE} per side")
        probe = body.beam == "probe"
        limit = max_semiangle(body.energy_kv, potential_settings["sampling"])
        if probe and body.semiangle > limit:
            raise HTTPException(422, f"Convergence: the sampling allows up to {limit:.0f} mrad")
        nx, ny = (n * t for n, t in zip(source.plugin["settings"]["gpts"], body.tile))
        planes = source.plugin["settings"]["slices"] + 1
        if max(nx, ny) > MAX_GPTS:
            raise HTTPException(422, f"{nx} × {ny} pixels: the limit is {MAX_GPTS} per side")
        # Complex values, two floats per pixel of each plane (of each frozen phonon configuration).
        wave_bytes = 2 * stack_bytes(nx, ny, planes) * (configs or 1)
        if wave_bytes > MAX_STACK_BYTES:
            size = wave_bytes / 1024**3
            shape = f"{nx} × {ny} × {planes} planes" + (f" × {configs} configs" if configs else "")
            raise HTTPException(422, f"{shape} is {size:.1f} GB: the limit is {MAX_STACK_BYTES / 1024**3:.0f} GB")

        if not run_under_way():
            clear_unfinished(entry.path)
        run_id, folder = new_run(entry.path)
        wanted = {**body.model_dump(), "potential": potential_settings}
        request = {
            **potential_settings,
            "model": str(item_file(model)),
            "folder": str(folder),
            "beam": body.beam,
            "energy": body.energy_kv * 1e3,
            "semiangle": body.semiangle,
            "defocus": body.defocus,
            "cs": body.cs_um * ANGSTROM_PER_UM,
            "tile": body.tile,
            "position": body.position,
            "workers": max(1, (os.cpu_count() or 2) - 2),
        }
        (folder / "request.json").write_text(json.dumps(request), encoding="utf-8")
        title = (
            f"Probe {body.energy_kv:g} kV, {body.semiangle:g} mrad, Δf {body.defocus:g} Å"
            if probe else f"Plane wave {body.energy_kv:g} kV"
        )

        def on_result(result: dict) -> dict:
            current = ctx.session.datasets.get(source.id)
            if current is None:
                shutil.rmtree(folder, ignore_errors=True)
                raise ValueError("The potential was deleted during the run")
            settings = {
                **wanted,
                "result": WAVE_KIND,
                "model": model.id,
                "potential_id": source.id,
                "run": f"runs/{run_id}",
                # wave.npy is [plane, x, y], or [config, plane, x, y] with frozen phonons.
                "gpts": result["shape"][-2:],
                "planes": result["shape"][-3],
                "configs": result.get("configs", 0),
                "sampling_xy": result["sampling"],
                "extent": result["extent"],
                "slice_thickness": result["slice_thickness"],
                "position": result["position"],
                "intensity_scale": result["intensity_scale"],
                "abtem": result["abtem"],
            }
            final = finish_run(folder, run_id, wanted, {
                "mode": WAVE_KIND,
                "model": model.plugin["settings"]["file"],
                "outputs": {"wave": "wave.npy"},
                "axes": {
                    "order": (["config"] if settings["configs"] else []) + ["plane", "x", "y"],
                    "sampling": result["sampling"], "slice_thickness": result["slice_thickness"], "units": "Å",
                },
                **{k: settings[k] for k in ("gpts", "planes", "configs", "extent", "position", "intensity_scale", "abtem")},
            })
            wave = np.load(final / "wave.npy", mmap_mode="r")
            run = ctx.add_item(entry, ITEM_WAVE, "Wavefunction", title, settings, f"{settings['planes']} planes", parent=current)
            sampling, thickness = result["sampling"], result["slice_thickness"]
            quantity = "Intensity (share of the beam)" if probe else "Intensity (incident beam = 1)"
            if settings["configs"]:
                # The phase is of the first configuration; the intensity is the mean over all of them.
                phase = np.angle(wave[0])
                intensity = sum(np.abs(wave[i]) ** 2 for i in range(settings["configs"])) / settings["configs"]
                phase_name, intensity_name = "Wave phase (config 1)", f"Wave intensity (mean of {settings['configs']} configs)"
            else:
                phase, intensity = np.angle(wave), np.abs(wave) ** 2
                phase_name, intensity_name = "Wave phase", "Wave intensity"
            outputs = (
                (phase_name, phase, "Phase (rad)", "phase", "Phase", WAVE_PHASE_STACK),
                (intensity_name, intensity * result["intensity_scale"], quantity, "intensity", "Intensity", WAVE_INTENSITY_STACK),
            )
            made = None
            try:
                # Each is added directly under the run, so the last added is listed first.
                for name, data, label_quantity, output, label, names in outputs:
                    made = ctx.add_dataset(
                        run, to_stack(data, sampling, thickness, name, label_quantity), name,
                        {"result": f"{WAVE_KIND}-{output}", "run": run.id},
                        label=label, stack=names, hidden_processes=HIDDEN_PROCESSES,
                    )
                ctx.save_project(entry)
            except Exception:
                ctx.session._remove_tree(run)
                shutil.rmtree(final, ignore_errors=True)
                raise
            return {"file": entry.summary(ctx.session), "dataset_id": made.id}

        job = ctx.submit_process(
            f"{WAVE_KIND}:{source.id}",
            f"Wavefunction of {model.title}",
            [str(engine.python_path()), "-I", str(engine.WORKER), "wavefunction", str(folder / "request.json")],
            on_result,
        )
        return job.summary()

    def ctf_settings_of(ctf) -> CtfSettings:
        return CtfSettings(**{k: v for k, v in ctf.plugin["settings"].items() if k in CtfSettings.model_fields})

    def check_ctf_for(ctf, energy_kv: float, sampling: float) -> CtfSettings:
        """The CTF's settings, if it can form an image of a wave at ``energy_kv``
        sampled at ``sampling`` Å (else 422)."""
        s = ctf_settings_of(ctf)
        if abs(s.energy_kv - energy_kv) > 1e-6:
            raise HTTPException(422, f"{ctf.title} is for {s.energy_kv:g} kV; the run is {energy_kv:g} kV")
        limit = max_semiangle(energy_kv, sampling)
        if s.semiangle is not None and s.semiangle > limit:
            raise HTTPException(422, f"{ctf.title}'s aperture is {s.semiangle:g} mrad; the sampling allows up to {limit:.0f} mrad")
        return s

    def add_image(run, ctf, coherence: str):
        """The image ``ctf`` forms of ``run``'s exit wave, added under the run
        (as the HRTEM walkthrough: apply_ctf, then the intensity)."""
        settings = run.plugin["settings"]
        entry = ctx.session.file_of(run)
        s = check_ctf_for(ctf, settings["energy_kv"], settings["potential"]["sampling"])
        if coherence == "focal-series" and s.focal_spread == 0:
            coherence = "envelope"  # nothing to average over
        wave = entry.path / settings["run"] / "exit_wave.npy"
        if not wave.is_file():
            raise HTTPException(404, f"{settings['run']}/exit_wave.npy is missing from the project")
        try:
            result = service.call("image", {
                **ctf_request(s), "wave": str(wave), "sampling": settings["sampling_xy"], "coherence": coherence,
            })
        except engine.ServiceError as e:
            raise HTTPException(502, str(e))
        array = np.frombuffer(base64.b64decode(result["image"]), np.float32).reshape(result["shape"])
        # With frozen phonons the exit wave has a wave per configuration; the image is their mean.
        configs = result.get("configs", 0)
        title = (
            f"Image · {ctf.title}, Δf {s.defocus:g} Å" + (", focal series" if coherence == "focal-series" else "")
            + (f", {configs} phonon configs" if configs else "")
        )
        made = ctx.add_dataset(
            run, to_signal(array, settings["sampling_xy"], title, "Intensity (incident beam = 1)"), title,
            {"result": f"{HRTEM_KIND}-image", "run": run.id, "ctf": ctf.id, "ctf_settings": s.model_dump(),
             "coherence": coherence, "mean": result["mean"], "configs": configs},
            label=LABEL_IMAGE,
        )
        return made

    @router.get("/projects/{file_id}/ctfs")
    def get_ctfs(file_id: str) -> list[dict]:
        """The project's CTFs: id, title and settings."""
        entry = project(file_id)
        return [
            {"id": d.id, "title": d.title, "settings": ctf_settings_of(d).model_dump()}
            for d in ctx.session.siblings_in_file(entry.id)
            if d.plugin and d.plugin.get("id") == manifest.id and d.plugin.get("label") == LABEL_CTF
        ]

    @router.post("/runs/{dataset_id}/images")
    def post_run_image(dataset_id: str, body: ImageSettings) -> dict:
        """Apply a CTF to an HRTEM run's exit wave: its image, added under the
        run and saved. Returns the file and the image."""
        run = item(dataset_id, ITEM_RUN)
        if run.plugin["settings"].get("result") != HRTEM_KIND:
            raise HTTPException(404, f"Not an HRTEM run: {dataset_id}")
        if not engine.status()["installed"]:
            raise HTTPException(409, "Install abTEM first")
        made = add_image(run, ctf_object(body.ctf), body.coherence)
        entry = ctx.session.file_of(run)
        try:
            ctx.save_project(entry)
        except Exception:
            ctx.session._remove(made)
            raise
        return {"file": entry.summary(ctx.session), "dataset_id": made.id}

    @router.get("/potentials/{dataset_id}/hrtem")
    def get_hrtem(dataset_id: str) -> dict:
        """The HRTEM run under way on the potential, if any."""
        job = hrtem_job(potential_object(dataset_id))
        return {"job": job.summary(with_result=False) if job else None}

    @router.post("/potentials/{dataset_id}/hrtem")
    def post_hrtem(dataset_id: str, body: HrtemSettings) -> dict:
        """Run a plane wave through the potential, as a job. Its result is a
        run under the potential holding the exit wave and, with a CTF, the
        image it forms."""
        source = potential_object(dataset_id)
        model = item(source.plugin["settings"]["model"], ITEM_MODEL)
        entry = ctx.session.file_of(source)
        if not engine.status()["installed"]:
            raise HTTPException(409, "Install abTEM first")
        if hrtem_job(source) is not None:
            raise HTTPException(409, "This potential already has an HRTEM run under way")
        potential_settings = normalized(source.plugin["settings"]["potential"])
        configs = configs_of(potential_settings)
        ctf = ctf_object(body.ctf) if body.ctf else None
        energy_kv = ctf_settings_of(ctf).energy_kv if ctf else body.energy_kv
        if ctf:
            check_ctf_for(ctf, energy_kv, potential_settings["sampling"])
        # The exit wave of every frozen phonon configuration is kept (complex).
        nx, ny = source.plugin["settings"]["gpts"]
        wave_bytes = 2 * stack_bytes(nx, ny, 1) * (configs or 1)
        if wave_bytes > MAX_STACK_BYTES:
            shape = f"{nx} × {ny}" + (f" × {configs} configs" if configs else "")
            raise HTTPException(422, f"{shape} is {wave_bytes / 1024**3:.1f} GB: the limit is {MAX_STACK_BYTES / 1024**3:.0f} GB")

        if not run_under_way():
            clear_unfinished(entry.path)
        run_id, folder = new_run(entry.path)
        wanted = {"energy_kv": energy_kv, "potential": potential_settings}
        request = {
            **potential_settings,
            "model": str(item_file(model)),
            "folder": str(folder),
            "energy": energy_kv * 1e3,
            "workers": max(1, (os.cpu_count() or 2) - 2),
        }
        (folder / "request.json").write_text(json.dumps(request), encoding="utf-8")
        title = f"HRTEM {energy_kv:g} kV"

        def on_result(result: dict) -> dict:
            current = ctx.session.datasets.get(source.id)
            if current is None:
                shutil.rmtree(folder, ignore_errors=True)
                raise ValueError("The potential was deleted during the run")
            settings = {
                **wanted,
                "result": HRTEM_KIND,
                "model": model.id,
                "potential_id": source.id,
                "run": f"runs/{run_id}",
                # exit_wave.npy is [x, y], or [config, x, y] with frozen phonons.
                "gpts": result["shape"][-2:],
                "configs": result.get("configs", 0),
                "sampling_xy": result["sampling"],
                "slices": result["slices"],
                "exit_intensity": result["exit_intensity"],
                "abtem": result["abtem"],
            }
            final = finish_run(folder, run_id, wanted, {
                "mode": HRTEM_KIND,
                "model": model.plugin["settings"]["file"],
                "outputs": {"exit_wave": "exit_wave.npy"},
                "axes": {"order": (["config"] if settings["configs"] else []) + ["x", "y"], "sampling": result["sampling"], "units": "Å"},
                **{k: settings[k] for k in ("gpts", "configs", "slices", "exit_intensity", "abtem")},
            })
            wave = np.load(final / "exit_wave.npy", mmap_mode="r")
            run = ctx.add_item(entry, ITEM_RUN, "HRTEM", title, settings, f"{energy_kv:g} kV", parent=current)
            sampling = result["sampling"]
            # With frozen phonons, the amplitude and phase shown are of one wave, the first configuration's.
            suffix = " (config 1)" if settings["configs"] else ""
            wave = np.array(wave[0] if settings["configs"] else wave)
            outputs = (
                (f"Exit wave phase{suffix}", np.angle(wave), "Phase (rad)", "phase", "Phase"),
                (f"Exit wave amplitude{suffix}", np.abs(wave), "Amplitude", "amplitude", "Amplitude"),
            )
            made = None
            try:
                # Each is added directly under the run, so the last added is listed first.
                for name, data, quantity, output, label in outputs:
                    made = ctx.add_dataset(
                        run, to_signal(data, sampling, name, quantity), name,
                        {"result": f"{HRTEM_KIND}-{output}", "run": run.id}, label=label,
                    )
                current_ctf = ctx.session.datasets.get(ctf.id) if ctf else None
                if current_ctf is not None:
                    made = add_image(run, current_ctf, body.coherence)
                ctx.save_project(entry)
            except Exception:
                ctx.session._remove_tree(run)
                shutil.rmtree(final, ignore_errors=True)
                raise
            return {"file": entry.summary(ctx.session), "dataset_id": made.id}

        job = ctx.submit_process(
            f"{HRTEM_KIND}:{source.id}",
            f"HRTEM of {model.title}",
            [str(engine.python_path()), "-I", str(engine.WORKER), "hrtem", str(folder / "request.json")],
            on_result,
        )
        return job.summary()

    @router.post("/structures/{dataset_id}/model/preview")
    def post_model_preview(dataset_id: str, body: ModelBody) -> dict:
        """What a model along ``uvw`` would be, without making it."""
        ds = item(dataset_id, ITEM_STRUCTURE)
        try:
            return preview(item_file(ds), body.uvw, body.repeats)
        except StructureError as e:
            raise HTTPException(422, str(e))

    @router.post("/structures/{dataset_id}/models")
    def post_model(dataset_id: str, body: ModelBody) -> dict:
        """Make a model of the structure. Returns its file and the new item."""
        ds = item(dataset_id, ITEM_STRUCTURE)
        path = item_file(ds)
        entry = ctx.session.file_of(ds)
        try:
            repeats = body.repeats or preview(path, body.uvw)["repeats"]
            made = add_model(ctx, entry, ds, body.uvw, repeats, body.seed)
        except (StructureError, OSError) as e:
            raise HTTPException(422, str(e))
        return {"file": entry.summary(ctx.session), "dataset_id": made.id}

    from .stem import register as register_stem; register_stem(ctx, router, service, potential_object, item, item_file, run_under_way)  # noqa: E702
    register_series(ctx, router, locals())  # thickness and defocus series (series.py)
    ctx.add_router(router)
