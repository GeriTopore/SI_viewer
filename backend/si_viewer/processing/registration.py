"""Drift correction of image stacks with rigidregistration (M9).

The registration is a guided, step-by-step session: the user sets each
step's parameters while looking at its diagnostics (see docs/PLAN.md 4.10).
A ``RegistrationSession`` holds the package's ``imstack`` for one stack
(frames, their FFTs, later the shift matrices) between those steps.

The package is written for notebooks, so it is made headless here:
matplotlib uses the Agg backend, progress bars report to a callback (which
can cancel), and FFTW's wisdom is cached in the user's cache folder instead
of the package's source folder.

Axes: the package takes frames last, (rows, columns, frames). Its "x" is the
row axis, i.e. the image's y.
"""

from __future__ import annotations

import functools
import os
import threading
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from pydantic import BaseModel, Field

MaskType = Literal["bandpass", "hann", "gaussian", "hamming", "blackman", "none"]

# Above this, the FFTs of all frames (complex128) need a warning.
FFT_MEMORY_WARNING = 1 << 30


class MaskSettings(BaseModel):
    # Fourier mask: features smaller than ~n pixels are ignored.
    type: MaskType = "hann"
    n: float = Field(6.0, gt=0.5, le=200)


def _wisdom_file() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "si-viewer"
    base.mkdir(parents=True, exist_ok=True)
    return base / "fftw_wisdom.pkl"


@functools.cache
def _package():
    """rigidregistration, made headless (imported on first use)."""
    import matplotlib

    matplotlib.use("Agg")
    import rigidregistration
    from rigidregistration import FFTW, stackregistration

    # Progress goes to the job (or nowhere), not to a notebook progress bar.
    stackregistration.tqdm = _Progress
    # Keep FFTW's plans in the user's cache, not the package's source folder.
    if FFTW.hasfftw:
        stackregistration.FFTW.WrapFFTW = functools.partial(FFTW.WrapFFTW, wisdomfile=str(_wisdom_file()))
    return rigidregistration


class _Progress:
    """Stands in for tqdm inside the package: reports to a callback, which
    may raise to cancel."""

    report: Callable[[float], None] | None = None

    def __init__(self, iterable=None, total=None, **_):
        self.iterable = iterable
        self.total = total if total is not None else (len(iterable) if iterable is not None else 0)
        self.n = 0

    def __iter__(self):
        for item in self.iterable:
            yield item
            self.update(1)

    def update(self, k=1):
        self.n += k
        if _Progress.report and self.total:
            _Progress.report(self.n / self.total)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class RegistrationSession:
    """One stack's registration in progress.

    ``frames`` is the stack (frames, rows, cols); only ``first … last - 1``
    are registered. Frame numbers in the API are the stack's own.
    """

    def __init__(self, frames: np.ndarray, first: int = 0, last: int | None = None, revision: int = 0):
        n = frames.shape[0]
        last = n if last is None else last
        if not 0 <= first < last <= n or last - first < 2:
            raise ValueError(f"Choose at least two frames within 0–{n - 1}")
        self.revision = revision
        self.n_total = n
        self.first, self.last = first, last
        self.bad_frames: list[int] = []
        self.mask = MaskSettings()
        self.lock = threading.Lock()
        sub = np.asarray(frames[first:last], dtype=np.float64)
        self.frames = sub
        rr = _package()
        self.imstack = rr.stackregistration.imstack(np.moveaxis(sub, 0, -1))
        self.imstack.getFFTs()
        self._apply_mask(self.mask)

    @property
    def count(self) -> int:
        return self.last - self.first

    def summary(self) -> dict:
        rows, cols = self.frames.shape[1:]
        return {
            "first": self.first,
            "last": self.last,
            "n_total": self.n_total,
            "shape": [rows, cols],
            "bad_frames": self.bad_frames,
            "mask": self.mask.model_dump(),
            "fft_bytes": int(self.imstack.fftstack.nbytes),
            "fftw": bool(_package().FFTW.hasfftw),
        }

    def _index(self, frame: int) -> int:
        if not self.first <= frame < self.last:
            raise ValueError(f"Frame {frame} is outside the frames being registered ({self.first}–{self.last - 1})")
        return frame - self.first

    def _apply_mask(self, mask: MaskSettings) -> None:
        self.imstack.makeFourierMask(mask=mask.type, n=mask.n)

    def set_settings(self, bad_frames: list[int] | None = None, mask: MaskSettings | None = None) -> None:
        with self.lock:
            if bad_frames is not None:
                self.bad_frames = sorted({f for f in bad_frames if self.first <= f < self.last})
            if mask is not None:
                self.mask = mask
                self._apply_mask(mask)

    # ---------- previews ----------

    def frame(self, frame: int) -> np.ndarray:
        return self.frames[self._index(frame)].astype(np.float32)

    def correlation(self, i: int, j: int, mask: MaskSettings | None = None) -> np.ndarray:
        """Cross-correlation of frames i and j with ``mask`` (default: the
        session's), shifted so zero shift is at the centre (rows // 2, cols // 2)."""
        with self.lock:
            changed = mask is not None and mask != self.mask
            if changed:
                self._apply_mask(mask)
            try:
                s = self.imstack
                cc = s.getSingleCrossCorrelation(s.fftstack[:, :, self._index(i)], s.fftstack[:, :, self._index(j)])
            finally:
                if changed:
                    self._apply_mask(self.mask)
        return np.fft.fftshift(cc).astype(np.float32)

    def compare_n(self, i: int, j: int, mask_type: MaskType, ns: list[float]) -> np.ndarray:
        """Cross-correlations of frames i, j for each cutoff n: (len(ns), rows, cols)."""
        return np.stack([self.correlation(i, j, MaskSettings(type=mask_type, n=n)) for n in ns])
