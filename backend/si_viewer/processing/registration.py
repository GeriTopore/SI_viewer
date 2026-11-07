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

MaskType = Literal["bandpass", "hann", "gaussian", "hamming", "blackman", "elliptical", "none"]

# Above this, the FFTs of all frames (complex128) need a warning.
FFT_MEMORY_WARNING = 1 << 30


class MaskSettings(BaseModel):
    # Fourier mask: features smaller than ~n pixels are ignored.
    type: MaskType = "hann"
    n: float = Field(6.0, gt=0.5, le=200)
    # Elliptical Gaussian (``makeFourierMask_eg``): n is the cutoff along the
    # first axis, n2 along the second, the first axis at theta degrees.
    n2: float = Field(4.0, gt=0.5, le=200)
    theta: float = Field(0.0, ge=-360, le=360)


def make_mask(target, mask: MaskSettings) -> None:
    """Set ``mask`` on an ``imstack`` (or a stand-in with its attributes)."""
    cls = _package().stackregistration.imstack
    if mask.type == "elliptical":
        cls.makeFourierMask_eg(target, n1=mask.n, n2=mask.n2, theta=np.radians(mask.theta))
    else:
        cls.makeFourierMask(target, mask=mask.type, n=mask.n)


CorrelationType = Literal["cc", "mc", "pc"]
PeakMethod = Literal["pixel", "gf", "com"]


class PeakSettings(BaseModel):
    """How each pair's shift is read from its correlation (the package's
    ``findImageShifts`` options and Gaussian-fit parameters)."""

    # Cross, mutual or phase correlation.
    correlation: CorrelationType = "cc"
    # Brightest pixel, Gaussian fit (subpixel) or centre of mass.
    method: PeakMethod = "pixel"
    # Gaussian fit: candidate maxima fitted, starting σ, cut-out radius (px).
    num_peaks: int = Field(5, ge=1, le=20)
    sigma_guess: float = Field(2.0, gt=0, le=100)
    window_radius: int = Field(6, ge=2, le=64)


class OutlierSettings(BaseModel):
    """Which pair shifts are left out as outliers (the package's
    ``get_outliers`` / ``get_outliers_NN`` and ``mask_off_diagonal``)."""

    # Transitivity: shifts must add up along paths of pairs (i→k→j = i→j).
    # Neighbours: a shift must agree with the neighbouring matrix entries.
    method: Literal["transitivity", "neighbours", "none"] = "transitivity"
    # Transitivity: allowed mismatch (px), averaged over this many paths.
    threshold: float = Field(10.0, gt=0, le=1000)
    max_paths: int = Field(5, ge=1, le=50)
    # Neighbours: allowed difference from neighbouring entries (px).
    max_shift: float = Field(5.0, gt=0, le=1000)
    # Use only pairs at most this many frames apart (None: all pairs).
    off_diagonal: int | None = Field(None, ge=1)


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


def normalised(frames: np.ndarray) -> np.ndarray:
    """Frames scaled into 0–1, as the package's notebooks prepare them ("for
    best performance, data should be normalized between 0 and 1"): integer
    data by its type's range (uint16 / 2¹⁶, as in the notebooks), float data
    by its largest magnitude. The Gaussian peak fit depends on this scale."""
    if np.issubdtype(frames.dtype, np.integer):
        return frames / float(2 ** (8 * frames.dtype.itemsize))
    peak = float(np.max(np.abs(frames))) or 1.0
    return np.asarray(frames, np.float64) / peak


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

    The settings live here; they are put on the package's ``imstack`` only
    when the shifts are computed (a job). Previews never change the
    ``imstack``: they use a stand-in (``_ops``) with the package's own methods
    and their own FFT object (FFTW plans aren't shared between threads), so
    they can run while a job is computing and give the same numbers.
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
        self.peaks = PeakSettings()
        self.outliers = OutlierSettings()
        # Settings the shift matrices were computed with (None: not yet).
        self.shifts_settings: dict | None = None
        # Settings of the last correction (None: not corrected).
        self.correction: dict | None = None
        # The aligned frames, their average and the crop, for the last
        # correction (None: not made yet).
        self._result: dict | None = None
        self.lock = threading.Lock()
        # Previews take turns on their own FFT object.
        self.preview_lock = threading.Lock()
        self._preview_fft = None
        sub = np.asarray(frames[first:last])
        self.frames = sub
        rr = _package()
        self.imstack = rr.stackregistration.imstack(np.moveaxis(normalised(sub), 0, -1))
        self.imstack.getFFTs()

    @property
    def count(self) -> int:
        return self.last - self.first

    def _settings(self) -> dict:
        return {"mask": self.mask.model_dump(), "peaks": self.peaks.model_dump()}

    def summary(self) -> dict:
        rows, cols = self.frames.shape[1:]
        return {
            "first": self.first,
            "last": self.last,
            "n_total": self.n_total,
            "shape": [rows, cols],
            "bad_frames": self.bad_frames,
            "mask": self.mask.model_dump(),
            "peaks": self.peaks.model_dump(),
            "outliers": self.outliers.model_dump(),
            "shifts": None if self.shifts_settings is None else {
                **self.shifts_settings,
                # The mask or peak finding changed since: recompute to use them.
                "stale": self.shifts_settings != self._settings(),
            },
            "fft_bytes": int(self.imstack.fftstack.nbytes),
            "fftw": bool(_package().FFTW.hasfftw),
        }

    def _index(self, frame: int) -> int:
        if not self.first <= frame < self.last:
            raise ValueError(f"Frame {frame} is outside the frames being registered ({self.first}–{self.last - 1})")
        return frame - self.first

    def set_settings(
        self, bad_frames: list[int] | None = None, mask: MaskSettings | None = None, peaks: PeakSettings | None = None,
        outliers: OutlierSettings | None = None,
    ) -> None:
        if bad_frames is not None:
            self.bad_frames = sorted({f for f in bad_frames if self.first <= f < self.last})
        if mask is not None:
            self.mask = mask
        if peaks is not None:
            self.peaks = peaks
        if outliers is not None:
            self.outliers = outliers

    # ---------- previews ----------

    def _ops(self, mask: MaskSettings, peaks: PeakSettings):
        """A stand-in for the ``imstack`` with ``mask`` and ``peaks`` applied:
        the package's methods run on it unchanged, with NumPy FFTs."""
        from types import SimpleNamespace

        rr = _package()
        s = self.imstack
        if self._preview_fft is None:
            # The same FFT the registration uses (FFTW when available), so
            # previews give the shift matrices' numbers exactly.
            self._preview_fft = rr.stackregistration.FFTW.WrapFFTW((s.nx, s.ny))
        ops = SimpleNamespace(
            nx=s.nx, ny=s.ny, kx=s.kx, ky=s.ky, kr=s.kr, rx=s.rx, ry=s.ry,
            fftw=self._preview_fft,
            num_peaks=peaks.num_peaks, sigma_guess=peaks.sigma_guess, window_radius=peaks.window_radius,
            num_iter=s.num_iter, min_window_frac=s.min_window_frac,
        )
        make_mask(ops, mask)
        return ops

    def _correlation(self, ops, i: int, j: int, kind: CorrelationType) -> np.ndarray:
        """Correlation of frames i, j as the package computes it (unshifted)."""
        cls = type(self.imstack)
        fn = {
            "cc": cls.getSingleCrossCorrelation,
            "mc": cls.getSingleMutualCorrelation,
            "pc": cls.getSinglePhaseCorrelation,
        }[kind]
        fft = self.imstack.fftstack
        return fn(ops, fft[:, :, self._index(i)], fft[:, :, self._index(j)])

    def frame(self, frame: int) -> np.ndarray:
        return self.frames[self._index(frame)].astype(np.float32)

    def correlation(
        self, i: int, j: int, mask: MaskSettings | None = None, kind: CorrelationType | None = None,
    ) -> np.ndarray:
        """Correlation of frames i and j with ``mask`` (default: the
        session's), shifted so zero shift is at the centre (rows // 2, cols // 2)."""
        with self.preview_lock:
            ops = self._ops(mask or self.mask, self.peaks)
            cc = self._correlation(ops, i, j, kind or self.peaks.correlation)
        return np.fft.fftshift(cc).astype(np.float32)

    def compare_n(self, i: int, j: int, mask: MaskSettings, ns: list[float]) -> np.ndarray:
        """Cross-correlations of frames i, j for each cutoff n (an elliptical
        mask keeps its shape: n2 scales with n): (len(ns), rows, cols)."""
        ratio = mask.n2 / mask.n
        return np.stack([
            self.correlation(i, j, mask.model_copy(update={"n": n, "n2": n * ratio}), "cc") for n in ns
        ])

    def peak(self, i: int, j: int, mask: MaskSettings | None = None, peaks: PeakSettings | None = None) -> dict:
        """How the shift of frames i, j is found, as the registration will
        find it: the shift (dx, dy in image pixels), where it is in the
        centred correlation, and for a Gaussian fit every candidate maximum
        with its cut-out, fitted model and parameters (``show_Gaussian_fit``).
        Positions are (x, y) = (column, row) in the centred correlation."""
        from rigidregistration.utils import fit_gaussian, gauss2d, get_cutout, on_edge

        if i > j:
            # The registration finds each pair once, for i < j, and takes
            # shift(i, j) = -shift(j, i); do the same. Correlation (i, j) is
            # correlation (j, i) mirrored through the centre.
            p = self.peak(j, i, mask, peaks)
            rows, cols = self.frames.shape[1:]
            mx = lambda x: 2 * (cols // 2) - x  # noqa: E731
            my = lambda y: 2 * (rows // 2) - y  # noqa: E731
            for c in p["candidates"]:
                c["x"], c["y"] = mx(c["x"]), my(c["y"])
                if c["fitted"]:
                    c["fit_x"], c["fit_y"] = mx(c["fit_x"]), my(c["fit_y"])
                    c["data"] = [row[::-1] for row in c["data"][::-1]]
                    c["model"] = [row[::-1] for row in c["model"][::-1]]
            return {**p, "dx": -p["dx"], "dy": -p["dy"], "x": mx(p["x"]), "y": my(p["y"])}

        peaks = peaks or self.peaks
        cls = type(self.imstack)
        with self.preview_lock:
            ops = self._ops(mask or self.mask, peaks)
            cc = self._correlation(ops, i, j, peaks.correlation)
        rows, cols = cc.shape
        candidates = []
        if peaks.method == "pixel":
            r, c = cls.getSingleShift_pixel(ops, cc)
            # Wrap as findImageShifts does.
            dr = r if r < rows / 2 else r - rows
            dc = c if c < cols / 2 else c - cols
        elif peaks.method == "com":
            dr, dc = cls.getSingleShift_com(ops, cc)
        else:
            centred = np.fft.fftshift(cc)
            maxima = cls.get_n_cross_correlation_maxima(ops, cc, peaks.num_peaks)
            rad = peaks.window_radius
            best, best_score = None, -np.inf
            for r0, c0 in maxima:
                cand = {"x": float(c0), "y": float(r0), "fitted": False}
                if not on_edge(centred, r0, c0, rad):
                    cutout = get_cutout(centred, r0, c0, rad)
                    popt, _, ok = fit_gaussian(
                        rad, rad, peaks.sigma_guess, peaks.sigma_guess, 0, 0, cutout, plot=False, verbose=False,
                    )
                    # The fitted model on the cut-out, as getGaussianFitResult draws it.
                    grid = np.meshgrid(range(cutout.shape[0]), range(cutout.shape[1]))
                    model = np.reshape(gauss2d(grid, *popt), cutout.shape)
                    # fit_peaks: (row, col) = popt[1:3] reversed + the window's corner.
                    fr, fc = popt[2] + r0 - rad, popt[1] + c0 - rad
                    score = popt[6] + popt[0]  # what getSingleShift_gaussianFit maximises
                    cand.update({
                        "fitted": True, "success": bool(ok), "fit_x": float(fc), "fit_y": float(fr),
                        "amplitude": float(popt[0]), "sigma": [float(abs(popt[3])), float(abs(popt[4]))],
                        "offset": float(popt[6]),
                        "data": cutout.astype(np.float32).tolist(), "model": model.astype(np.float32).tolist(),
                    })
                    if score > best_score:
                        best, best_score = len(candidates), score
                candidates.append(cand)
            if best is None:
                raise ValueError("Every candidate peak is too close to the edge to fit: lower the window radius")
            dr = candidates[best]["fit_y"] - rows / 2
            dc = candidates[best]["fit_x"] - cols / 2
            for k, cand in enumerate(candidates):
                cand["chosen"] = k == best
        return {
            "dx": float(dc), "dy": float(dr),
            "x": float(dc + cols // 2), "y": float(dr + rows // 2),
            "candidates": candidates,
        }

    # ---------- shift matrices ----------

    def compute_shifts(self, report: Callable[[float], None]) -> None:
        """Every pair's shift (the package's ``findImageShifts``) with the
        current settings. ``report`` gets the progress and may raise to cancel."""
        with self.lock:
            settings = self._settings()
            mask, peaks = self.mask, self.peaks
            s = self.imstack
            make_mask(s, mask)
            s.setGaussianFitParams(num_peaks=peaks.num_peaks, sigma_guess=peaks.sigma_guess, window_radius=peaks.window_radius)
            _Progress.report = report
            try:
                s.findImageShifts(correlationType=peaks.correlation, findMaxima=peaks.method)
            finally:
                _Progress.report = None
            self.shifts_settings = settings
            self.correction = None
            self._result = None

    def _set_mask(self, outliers: OutlierSettings, bad: list[int]) -> None:
        """Mark the package's ``Rij_mask`` as the user set it (call with the lock)."""
        s = self.imstack
        s.outlier_mask = np.ones((s.nz, s.nz), dtype=bool)
        if outliers.method == "transitivity":
            s.get_outliers(threshold=outliers.threshold, maxpaths=outliers.max_paths)
        elif outliers.method == "neighbours":
            s.get_outliers_NN(outliers.max_shift)
        s.mask_off_diagonal(outliers.off_diagonal or False)
        s.set_bad_images([f - self.first for f in bad if self.first <= f < self.last])
        s.refine_mask()

    def outlier_mask(self, outliers: OutlierSettings | None = None, bad_frames: list[int] | None = None) -> np.ndarray:
        """Which pair shifts are kept (1) or masked (0), (frames, frames), as
        the package marks them before correcting: outliers, pairs too far
        off the diagonal and bad frames' rows and columns; the diagonal of
        good frames is kept (``refine_mask``)."""
        if self.shifts_settings is None:
            raise ValueError("Compute the shifts first")
        with self.lock:
            self._set_mask(outliers or self.outliers, self.bad_frames if bad_frames is None else bad_frames)
            return self.imstack.Rij_mask.astype(np.float32)

    def correct(self, outliers: OutlierSettings | None = None, bad_frames: list[int] | None = None) -> dict:
        """Fill the masked shifts from paths of good ones (the package's
        ``make_corrected_Rij``), then each frame's shift as the mean of its
        row (``get_imshifts``). Returns each frame's measured drift.

        The package's loop runs until every good frame's pairs are filled.
        A round that fills nothing changes nothing for the next, so where the
        package repeats such rounds until its 1000th (when it doubles the
        paths tried), this doubles them at once, with the same result. Its
        path generator runs past the ends of the stack (frame -1, nz, …);
        NumPy wraps the negative ones to the last frames, which is still a
        valid chain of pairs, and the package relies on it. Indices past the
        end (an IndexError in the package) wrap the same way. With no
        progress at 640 paths per pair it stops with an error, where the
        package would loop forever."""
        from rigidregistration.utils import getpaths

        if self.shifts_settings is None:
            raise ValueError("Compute the shifts first")
        o = outliers or self.outliers
        bad = self.bad_frames if bad_frames is None else bad_frames
        with self.lock:
            self._set_mask(o, bad)
            s = self.imstack
            nz = s.nz
            mask = s.Rij_mask
            good = np.nonzero(np.any(mask, axis=1))[0]
            known = np.copy(mask)
            xc = np.where(mask, s.X_ij, np.nan)
            yc = np.where(mask, s.Y_ij, np.nan)
            maxpaths = 5

            def paths(i: int, j: int):
                for p in getpaths(i, j, maxpaths, nz):
                    yield [(a % nz, b % nz) for a, b in p]

            while not np.all(known[np.ix_(good, good)]):
                for i in range(nz - 1):
                    for j in range(i + 1, nz):
                        if known[i, j]:
                            continue
                        n, x, y = 0, 0.0, 0.0
                        for p in paths(i, j):
                            if np.all([known[ip] for ip in p]):
                                x += np.array([xc[ip] for ip in p]).sum()
                                y += np.array([yc[ip] for ip in p]).sum()
                                n += 1
                        if n:
                            xc[i, j], xc[j, i] = -x / n, x / n
                            yc[i, j], yc[j, i] = -y / n, y / n
                filled = ~np.isnan(xc) & ~np.isnan(yc)
                if np.array_equal(filled, known):
                    if maxpaths >= 640:
                        stuck = sorted({int(self.first + k) for k in good if not np.all(filled[k, good])})
                        raise ValueError(
                            "Some frames have too few good pairs to correct ("
                            + ", ".join(map(str, stuck[:12])) + ("…" if len(stuck) > 12 else "")
                            + "): loosen the outlier mask or mark those frames bad"
                        )
                    maxpaths *= 2
                known = filled
            s.Rij_mask_c = known
            s.X_ij_c = np.nan_to_num(xc)
            s.Y_ij_c = np.nan_to_num(yc)
            s.get_imshifts()
            self.correction = {"outliers": o.model_dump(), "bad_frames": list(bad), "shifts": self.shifts_settings}
            self._result = None
            return self.drift()

    def drift(self) -> dict:
        """Each registered frame's measured position relative to the mean
        (px, image x right and y down): minus the shift that aligns it. Bad
        frames (no shift) are left out."""
        s = self.imstack
        rows = np.flatnonzero(s.Rij_mask_c.sum(axis=1) > 0)
        return {
            "frames": [int(self.first + k) for k in rows],
            "x": [float(-s.shifts_y[k]) for k in rows],
            "y": [float(-s.shifts_x[k]) for k in rows],
        }

    # ---------- result ----------

    def result(self) -> dict:
        """Every good frame shifted into register and their average, as the
        package's ``get_averaged_image`` makes them (a Fourier shift of each
        frame by its shift; the loaded data, not normalised), with the area
        every frame covers: the package's ``crop_image`` rectangle and the
        largest square centred in it. Rectangles are (x, y, width, height)
        in pixels of the aligned frames; a frame's own pixel p lands at
        p - drift. Made once per correction."""
        from rigidregistration.utils import generateShiftedImage

        if self.correction is None:
            raise ValueError("Correct the shift matrices first")
        with self.lock:
            if self._result is not None:
                return self._result
            s = self.imstack
            rows = np.flatnonzero(s.Rij_mask_c.sum(axis=1) > 0)
            aligned = np.stack([
                generateShiftedImage(self.frames[k].astype(np.float64), s.shifts_x[k], s.shifts_y[k]) for k in rows
            ]).astype(np.float32)
            average = aligned.mean(axis=0)
            # crop_image: content lands at p + shift, so the rows every frame
            # covers run from the largest shift to the height plus the smallest.
            sx, sy = s.shifts_x[rows], s.shifts_y[rows]
            r0, r1 = int(np.ceil(max(sx.max(), 0))), int(np.floor(min(s.nx + sx.min(), s.nx)))
            c0, c1 = int(np.ceil(max(sy.max(), 0))), int(np.floor(min(s.ny + sy.min(), s.ny)))
            if r1 <= r0 or c1 <= c0:
                raise ValueError("No area is inside every frame: the drift is larger than the frame")
            side = min(r1 - r0, c1 - c0)
            sq_r, sq_c = r0 + (r1 - r0 - side) // 2, c0 + (c1 - c0 - side) // 2
            self._result = {
                "frames": [int(self.first + k) for k in rows],
                "aligned": aligned,
                "average": average,
                "rect": [c0, r0, c1 - c0, r1 - r0],
                "square": [sq_c, sq_r, side, side],
            }
            return self._result

    def result_summary(self) -> dict:
        r = self.result()
        return {"frames": r["frames"], "rect": r["rect"], "square": r["square"], "drift": self.drift()}

    def _cropped(self, image: np.ndarray, crop: bool) -> np.ndarray:
        if not crop:
            return image
        x, y, w, h = self.result()["square"]
        return np.ascontiguousarray(image[..., y:y + h, x:x + w])

    def aligned_frame(self, frame: int, crop: bool = True) -> np.ndarray:
        """Frame ``frame`` shifted into register (cropped to the square)."""
        r = self.result()
        if frame not in r["frames"]:
            raise ValueError(f"Frame {frame} is not in the aligned stack (bad frames are left out)")
        return self._cropped(r["aligned"][r["frames"].index(frame)], crop)

    def average(self, crop: bool = True) -> np.ndarray:
        """The average of the aligned frames (cropped to the square)."""
        return self._cropped(self.result()["average"], crop)

    def corrected_matrices(self) -> np.ndarray:
        """Corrected shift matrices (2, frames, frames), Δx then Δy, and which
        entries are known (the third plane: 1 known, 0 not)."""
        if getattr(self, "correction", None) is None:
            raise ValueError("Correct the shift matrices first")
        s = self.imstack
        return np.stack([s.Y_ij_c, s.X_ij_c, s.Rij_mask_c]).astype(np.float32)

    def matrices(self) -> np.ndarray:
        """The shift matrices as (2, frames, frames): Δx (columns) then Δy
        (rows) of frame j relative to frame i, in pixels. (The package's X is
        the row axis.)"""
        if self.shifts_settings is None:
            raise ValueError("Compute the shifts first")
        return np.stack([self.imstack.Y_ij, self.imstack.X_ij]).astype(np.float32)
