"""Differential phase contrast from 4D-STEM data: each pattern's centre of
mass (pyxem, as the Centring finds the beam), less the descan (a plane
fitted over the scan), is the beam's deflection by the specimen. Rotated
into the scan's frame (the rotation leaving the least curl, as for a
segmented detector: ``processing.dpc``), it is shown as DPC (direction and
magnitude) and integrated into iCOM (the projected potential, up to scale).
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from .centring import BeamOptions, PlaneOptions


class Rotation(BaseModel):
    # Found from the data (the least curl, its 180° ambiguity resolved by
    # the integrated image's sharp extremes being maxima), or as given.
    auto: bool = True
    theta: float = Field(0.0, ge=-360, le=360)
    mirror: bool = False


class ComOptions(BaseModel):
    beam: BeamOptions = BeamOptions(method="center_of_mass")
    plane: PlaneOptions = PlaneOptions()
    rotation: Rotation = Rotation()


class ComLive(BaseModel):
    """A virtual detector's centre-of-mass responses (``fourd.COM_RESPONSES``):
    the descan and rotation, as the DPC / iCOM process."""

    plane: PlaneOptions = PlaneOptions()
    rotation: Rotation = Rotation()

    def deflection(self, cx: np.ndarray, cy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The centre of mass ([y, x] each, pattern px) less the descan: the
        plane fitted to it (mode "none": its mean)."""
        from .centring import plane as descan

        c = np.stack([np.asarray(cx, np.float64), np.asarray(cy, np.float64)], axis=-1)
        ref = c.mean(axis=(0, 1), keepdims=True) if self.plane.mode == "none" else descan(c, self.plane)
        d = c - ref
        return d[..., 0], d[..., 1]


def deflection(shifts: np.ndarray, plane: PlaneOptions) -> tuple[np.ndarray, np.ndarray]:
    """The beam's deflection by the specimen ([y, x] each, pattern px) from
    pyxem's centring shifts (``centring.find_beam``: what moves the beam to
    the centre, so the beam is the centre less them): less the descan (the
    plane fitted to them; mode "none": their mean)."""
    from .centring import plane as descan

    s = np.asarray(shifts, dtype=np.float64)
    ref = s.mean(axis=(0, 1), keepdims=True) if plane.mode == "none" else descan(s, plane)
    d = -(s - ref)
    return d[..., 0], d[..., 1]


def rotation_of(gx: np.ndarray, gy: np.ndarray) -> dict:
    """The rotation (degrees) and mirror leaving the least curl, and the
    curl left relative to the divergence (``processing.dpc.find_rotation``),
    band-passed for the scan's size."""
    from .dpc import find_rotation

    n = min(gx.shape)
    fine = 1.0 if n < 64 else 2.0
    coarse = float(max(4.0, min(50.0, max(gx.shape) / 2)))
    return find_rotation(gx, gy, fine=fine, coarse=coarse)
