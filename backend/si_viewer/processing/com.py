"""The rotation between 4D-STEM data's pattern axes and its scan's: a beam
deflection measured per scan position (the Calibration's descan step: each
position's origin less the descan plane), rotated into the scan's frame, is
a gradient field (curl-free) only at the right rotation and mirror, as for
a segmented detector (``processing.dpc``)."""

from __future__ import annotations

import numpy as np


def rotation_of(gx: np.ndarray, gy: np.ndarray) -> dict:
    """The rotation (degrees) and mirror leaving the least curl, and the
    curl left relative to the divergence (``processing.dpc.find_rotation``),
    band-passed for the scan's size."""
    from .dpc import find_rotation

    n = min(gx.shape)
    fine = 1.0 if n < 64 else 2.0
    coarse = float(max(4.0, min(50.0, max(gx.shape) / 2)))
    return find_rotation(gx, gy, fine=fine, coarse=coarse)
