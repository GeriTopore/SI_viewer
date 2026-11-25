"""Potential settings of a model, and its projected potential as an image."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

MAX_GPTS = 4096
MAX_PHONON_CONFIGS = 64
DEFAULT_PHONON_SIGMA = 0.1  # Å, the frozen phonons walkthrough's
# The slices are kept in memory and in the analysis file.
MAX_STACK_BYTES = 2 * 1024**3
UNITS = "V·Å"
# Stack processes that don't apply to a potential's slices.
HIDDEN_PROCESSES = ["align", "rebin"]
STACK_NAMES = {"frame": "Slice", "frames": "slices", "sum": "Projected", "initial": "sum"}


class PotentialSettings(BaseModel):
    sampling: float = Field(0.05, ge=0.01, le=0.5)  # Å per pixel
    slice_thickness: float = Field(1.0, ge=0.1, le=10.0)  # Å
    parametrization: Literal["lobato", "kirkland"] = "lobato"
    projection: Literal["infinite", "finite"] = "infinite"
    # Frozen phonons (abTEM's FrozenPhonons): 0 or 1 configurations is off.
    phonon_configs: int = Field(0, ge=0, le=MAX_PHONON_CONFIGS)
    phonon_sigma: float = Field(DEFAULT_PHONON_SIGMA, gt=0, le=1.0)  # Å, per direction
    phonon_seed: int = Field(0, ge=0, le=2**31 - 1)

    @model_validator(mode="after")
    def _phonons_off(self):
        # So that settings which differ only in unused values are the same settings.
        if self.phonon_configs < 2:
            self.phonon_configs = 0
            self.phonon_sigma = DEFAULT_PHONON_SIGMA
            self.phonon_seed = 0
        return self


def normalized(stored: dict) -> dict:
    """Potential settings as stored (maybe before frozen phonons existed) with
    every setting present."""
    return PotentialSettings(**{k: v for k, v in stored.items() if k in PotentialSettings.model_fields}).model_dump()


def configs_of(potential: dict) -> int:
    """The frozen phonon configurations of stored potential settings (0: none)."""
    return int(potential.get("phonon_configs") or 0)


def grid(lengths, settings: PotentialSettings) -> tuple[int, int, int]:
    """Pixels along x and y, and slices along z, for a cell of ``lengths`` (Å)."""
    x, y, z = (float(v) for v in lengths[:3])
    return (
        math.ceil(x / settings.sampling),
        math.ceil(y / settings.sampling),
        math.ceil(z / settings.slice_thickness),
    )


def title_of(settings: PotentialSettings) -> str:
    title = f"Potential {settings.sampling:g} Å, {settings.slice_thickness:g} Å slices"
    if settings.parametrization != "lobato":
        title += f", {settings.parametrization.capitalize()}"
    if settings.projection != "infinite":
        title += f", {settings.projection}"
    if settings.phonon_configs:
        title += f", {settings.phonon_configs} phonon configs"
        if settings.phonon_sigma != DEFAULT_PHONON_SIGMA:
            title += f", σ {settings.phonon_sigma:g} Å"
        if settings.phonon_seed:
            title += f", seed {settings.phonon_seed}"
    return title


def stack_bytes(nx: int, ny: int, slices: int) -> int:
    return nx * ny * slices * 4


def to_stack(array: np.ndarray, sampling, thickness: float, title: str, quantity: str = f"Potential per slice ({UNITS})"):
    """A stack from abTEM's [slice, x, y] array: slices along z, each as the
    model's beam view shows it (x to the right, y up)."""
    import hyperspy.api as hs

    data = np.flip(np.asarray(array, np.float32).transpose(0, 2, 1), axis=1)
    signal = hs.signals.Signal2D(data)
    signal.metadata.General.title = title
    signal.metadata.set_item("Signal.quantity", quantity)
    z = signal.axes_manager.navigation_axes[0]
    z.name, z.units, z.scale, z.offset = "z", "Å", float(thickness), 0.0
    for axis, name, scale in zip(signal.axes_manager.signal_axes, "xy", sampling):
        axis.name, axis.units, axis.scale, axis.offset = name, "Å", float(scale), 0.0
    return signal


def to_signal(array: np.ndarray, sampling, title: str, quantity: str = f"Projected potential ({UNITS})"):
    """An image from abTEM's [x, y] array, as the model's beam view shows it:
    x to the right and y up. The y axis counts from the top of the image."""
    import hyperspy.api as hs

    data = np.flipud(np.asarray(array, np.float32).T)
    signal = hs.signals.Signal2D(data)
    signal.metadata.General.title = title
    signal.metadata.set_item("Signal.quantity", quantity)
    for axis, name, scale in zip(signal.axes_manager.signal_axes, "xy", sampling):
        axis.name, axis.units, axis.scale, axis.offset = name, "Å", float(scale), 0.0
    return signal
