"""Drift-corrected sum of an in-situ spectrum-image series (M12e).

The shifts that register the frames are measured on an image stack (the
simultaneous ADF, aligned with rigidregistration; see ``registration.py``) or
imported. rigidregistration's own ``generateShiftedImage`` shifts a 2D image
only (and keeps the modulus of the result), so the same Fourier shift is done
here for a whole frame of spectra at once: every energy channel of a frame
moves by the frame's shift, and the real part is kept (spectra may be
negative after processing). Shifts are in pixels of the spectrum image's scan
(an ADF ``k`` times finer than the scan, as DM's sub-scan, has its shifts
divided by ``k``).

Frames are read one at a time; the shifted frames are summed (not averaged:
the point is the counts) and cropped to the area every frame covers, as the
package's ``crop_image`` crops its average.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def shift_frame(frame: np.ndarray, row: float, col: float) -> np.ndarray:
    """``frame`` (height, width[, channels]) moved by ``row`` and ``col``
    pixels (content at p lands at p + shift, wrapping round), by the Fourier
    shift theorem, as rigidregistration's ``generateShiftedImage(im, row,
    col)`` but for any number of channels and keeping the real part. float64."""
    h, w = frame.shape[:2]
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    ramp = np.exp(-2j * np.pi * (fy * row + fx * col))
    if frame.ndim == 3:
        ramp = ramp[:, :, None]
    spectrum = np.fft.fft2(np.asarray(frame, dtype=np.float64), axes=(0, 1))
    return np.fft.ifft2(spectrum * ramp, axes=(0, 1)).real


def covered(rows: np.ndarray, cols: np.ndarray, height: int, width: int) -> tuple[int, int, int, int]:
    """(r0, r1, c0, c1): the pixels every frame covers once moved by its
    ``rows``/``cols`` shift (the package's ``crop_image``: content lands at
    p + shift). ValueError if none are."""
    r0 = int(np.ceil(max(rows.max(), 0)))
    r1 = int(np.floor(min(height + rows.min(), height)))
    c0 = int(np.ceil(max(cols.max(), 0)))
    c1 = int(np.floor(min(width + cols.min(), width)))
    if r1 <= r0 or c1 <= c0:
        raise ValueError("No area is inside every frame: the drift is larger than the scan")
    return r0, r1, c0, c1


def aligned_sum(
    read: Callable[[int], np.ndarray],
    frames: list[int],
    shifts: np.ndarray,
    report: Callable[..., None] = lambda f, m=None: None,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """The sum of ``read(k)`` (height, width, channels) over ``frames``, each
    moved by its shift (``shifts[k]`` = (x, y): columns then rows, in scan
    pixels), cropped to the area every frame covers. float32, and the crop
    (r0, r1, c0, c1). ``report(fraction, "Frame 3 of 40")`` may raise to cancel."""
    if not frames:
        raise ValueError("No frames to sum")
    cols = np.array([shifts[k][0] for k in frames], dtype=float)
    rows = np.array([shifts[k][1] for k in frames], dtype=float)
    total = None
    crop = None
    for i, k in enumerate(frames):
        report(i / len(frames), f"Frame {i + 1} of {len(frames)}")
        frame = read(k)
        if crop is None:
            crop = covered(rows, cols, *frame.shape[:2])
        r0, r1, c0, c1 = crop
        moved = shift_frame(frame, rows[i], cols[i])[r0:r1, c0:c1]
        total = moved if total is None else total + moved
    return total.astype(np.float32), crop


def parse_shifts(raw: bytes, n_frames: int) -> np.ndarray:
    """Shifts read from a ``.npy`` file: (n_frames, 2) or (2, n_frames), the
    (x, y) shift that aligns each frame; NaN leaves a frame out. Returned as
    (n_frames, 2) float64. ValueError (plain English) when it can't be used."""
    import io

    try:
        a = np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception as e:  # not an .npy file, or pickled objects
        raise ValueError(f"Not a NumPy .npy array: {e}") from None
    if not np.issubdtype(a.dtype, np.number):
        raise ValueError(f"The shifts must be numbers, not {a.dtype}")
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 2 and a.shape == (2, n_frames) and n_frames != 2:
        a = a.T  # one row of x, one of y
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError(f"Expected shifts shaped ({n_frames}, 2): one (x, y) per frame; got {a.shape}")
    if a.shape[0] != n_frames:
        raise ValueError(f"The series has {n_frames} frames, but there are {a.shape[0]} shifts")
    if np.isinf(a).any():
        raise ValueError("The shifts must be finite (NaN leaves a frame out)")
    if np.isnan(a).any(axis=1).sum() == n_frames:
        raise ValueError("Every frame is left out (all NaN)")
    return a
