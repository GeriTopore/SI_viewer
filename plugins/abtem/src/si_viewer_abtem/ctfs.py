"""Contrast transfer functions: their settings, and the request abTEM evaluates."""

from __future__ import annotations

import math

from pydantic import BaseModel, Field

ANGSTROM_PER_UM = 1e4
ANGSTROM_PER_MM = 1e7
PROFILE_POINTS = 600
IMAGE_POINTS = 256


class CtfSettings(BaseModel):
    """The objective lens, in microscope units. abTEM's conventions: positive
    defocus is underfocus (C10 = -defocus); C12 and its angle are the twofold
    astigmatism."""

    energy_kv: float = Field(300.0, ge=20, le=1000)
    defocus: float = Field(54.3, ge=-1e5, le=1e5)  # Å
    cs_um: float = Field(10.0, ge=-1e5, le=1e5)  # µm
    c5_mm: float = Field(0.0, ge=-1e3, le=1e3)  # mm
    astigmatism: float = Field(0.0, ge=0, le=1e4)  # Å (C12)
    astigmatism_angle: float = Field(0.0, ge=-360, le=360)  # degrees
    semiangle: float | None = Field(None, gt=0, le=1000)  # mrad; None: no aperture
    focal_spread: float = Field(0.0, ge=0, le=1e4)  # Å, 1/e width
    angular_spread: float = Field(0.0, ge=0, le=100)  # mrad, 1/e width
    # Kept to fill in the focal spread: δ = √2 Cc ΔE / E (the CTF walkthrough).
    cc_mm: float = Field(1.2, ge=0, le=100)  # mm
    energy_spread: float = Field(0.35, ge=0, le=10)  # eV, standard deviation
    max_angle: float = Field(40.0, gt=0, le=500)  # mrad shown
    profile_angle: float = Field(0.0, ge=-360, le=360)  # degrees: the profiles' direction


def worker_request(s: CtfSettings) -> dict:
    """The settings in abTEM's units (eV, Å, rad, mrad)."""
    return {
        "energy": s.energy_kv * 1e3,
        "defocus": s.defocus,
        "cs": s.cs_um * ANGSTROM_PER_UM,
        "c5": s.c5_mm * ANGSTROM_PER_MM,
        "c12": s.astigmatism,
        "phi12": math.radians(s.astigmatism_angle),
        "semiangle": s.semiangle,
        "focal_spread": s.focal_spread,
        "angular_spread": s.angular_spread,
        "max_angle": s.max_angle,
        "phi": math.radians(s.profile_angle),
        "profile_points": PROFILE_POINTS,
        "image_points": IMAGE_POINTS,
    }


def to_signal(result: dict, title: str):
    """The 2D CTF the worker returned ([kx, ky], zero frequency at index n/2)
    as an image in 1/Å from the centre: kx to the right and ky down, as a
    diffraction pattern, so every pixel reads its own (kx, ky)."""
    import base64

    import hyperspy.api as hs
    import numpy as np

    array = np.frombuffer(base64.b64decode(result["image"]), np.float32).reshape(result["image_shape"])
    signal = hs.signals.Signal2D(np.ascontiguousarray(array.T))
    signal.metadata.General.title = title
    signal.metadata.set_item("Signal.quantity", "CTF")
    for axis, name, n, step in zip(signal.axes_manager.signal_axes, ("kx", "ky"), result["image_shape"], result["image_step"]):
        axis.name, axis.units, axis.scale, axis.offset = name, "1/Å", float(step), -float(step) * (n // 2)
    return signal


def title_of(s: CtfSettings) -> str:
    return f"CTF {s.energy_kv:g} kV"
