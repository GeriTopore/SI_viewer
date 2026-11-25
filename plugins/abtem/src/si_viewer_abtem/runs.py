"""Simulation runs: their settings, and their folders in a project.

A run is written to ``runs/.tmp-<id>`` while it computes, and becomes
``runs/<id>`` (with ``settings.json`` and ``manifest.json``) only when it has
finished, so a cancelled or failed run leaves no result behind.
"""

from __future__ import annotations

import json
import math
import shutil
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

# Share of the Nyquist frequency abTEM keeps, to avoid aliasing.
ANTIALIAS = 2 / 3
ANGSTROM_PER_UM = 1e4
TMP_PREFIX = ".tmp-"


Coherence = Literal["envelope", "focal-series"]


class HrtemSettings(BaseModel):
    """A plane wave through the potential the run is made from. With ``ctf``
    (a CTF of the project), its energy, and the image it forms is added."""

    energy_kv: float = Field(300.0, ge=20, le=1000)  # without a CTF
    ctf: str | None = None
    coherence: Coherence = "envelope"


class ImageSettings(BaseModel):
    """A CTF applied to a run's exit wave."""

    ctf: str
    coherence: Coherence = "envelope"


class WaveSettings(BaseModel):
    """The beam sent through a potential, kept at every depth."""

    beam: Literal["plane", "probe"] = "plane"
    energy_kv: float = Field(300.0, ge=20, le=1000)
    # The probe's; a plane wave has no lens before the specimen.
    semiangle: float = Field(20.0, gt=0, le=1000)  # mrad
    # abTEM's convention: positive defocus focuses into the specimen.
    defocus: float = Field(0.0, ge=-100000, le=100000)  # Å
    cs_um: float = Field(0.0, ge=-100000, le=100000)  # µm
    tile: list[int] = Field(default=[1, 1], min_length=2, max_length=2)
    position: list[float] | None = Field(None, min_length=2, max_length=2)  # Å; None: the centre


def wavelength(energy_kv: float) -> float:
    """The relativistic electron wavelength (Å)."""
    return 12.398420 / math.sqrt(energy_kv * (2 * 510.99895 + energy_kv))


def max_semiangle(energy_kv: float, sampling: float) -> float:
    """The largest angle (mrad) the simulated wave holds."""
    return wavelength(energy_kv) / (2 * sampling) * ANTIALIAS * 1000


def new_run(project: Path) -> tuple[str, Path]:
    """A new run's id and its temporary folder."""
    run_id = uuid.uuid4().hex[:10]
    folder = project / "runs" / f"{TMP_PREFIX}{run_id}"
    folder.mkdir(parents=True)
    return run_id, folder


def clear_unfinished(project: Path) -> None:
    """Remove what cancelled or failed runs left."""
    runs = project / "runs"
    if runs.is_dir():
        for folder in runs.glob(f"{TMP_PREFIX}*"):
            shutil.rmtree(folder, ignore_errors=True)


def finish_run(folder: Path, run_id: str, settings: dict, manifest: dict) -> Path:
    """Write the run's records and give it its final folder."""
    (folder / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (folder / "request.json").unlink(missing_ok=True)
    final = folder.parent / run_id
    folder.rename(final)
    return final
