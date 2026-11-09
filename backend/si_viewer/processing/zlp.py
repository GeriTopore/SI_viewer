"""Zero-loss peak alignment of EELS low-loss spectrum images (M10d).

exSpy's ``align_zero_loss_peak`` (``calibrate=True``), taken apart so that the
shifts it applies can be kept and a job can report progress and be cancelled:

1. the peak of every spectrum is found (the channel of its maximum) and every
   spectrum is shifted to the mean position;
2. the spectral axis' offset is changed so that the mean peak is at 0;
3. with ``subpixel``: the shifts that line every spectrum up with the first
   pixel's are estimated by cross-correlation within +-3 (energy units) of the
   peak, and applied (a peak shifted to the mean position gets shifted a few
   fractions of a channel more);
4. the offset is corrected again.

Every spectrum image passed as ``others`` (the low loss's core losses) gets
the same shifts, in energy units, and the same changes of offset, and is
cropped the way its own axis needs. Works on copies: the signals passed in
are left as they are.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from pydantic import BaseModel, ConfigDict

from .eels import _TO_EV
from .thickness import quiet

Report = Callable[..., None]

# exSpy's range, around the peak, for the sub-pixel step (eV).
SUBPIXEL_RANGE = 3.0
# Fewest channels the sub-pixel step is run on.
MIN_RANGE_CHANNELS = 4
MIN_CHANNELS = 16


class ZlpOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Refine the alignment by cross-correlation (else the peak maxima are lined up).
    subpixel: bool = True
    # Crop the channels the shifts leave empty (else they are filled with 0).
    crop: bool = True

    @property
    def label(self) -> str:
        """e.g. "sub-pixel, cropped": what the history of a dataset says."""
        return f"{'sub-pixel' if self.subpixel else 'peak maximum'}, {'cropped' if self.crop else 'not cropped'}"


@dataclass
class ZlpResult:
    # Aligned copies, in the order of the signals passed (low loss first).
    signals: list = field(default_factory=list)
    # The energy shift applied to every pixel, in eV: the shape of the scan.
    shifts: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    # Their mean, in eV: how far the axis was recalibrated.
    mean_shift: float = 0.0


def _units_in_ev(signal) -> float:
    axis = signal.axes_manager.signal_axes[0]
    return _TO_EV.get(str(axis.units).strip().lower(), 1.0)


def check_signal(signal) -> None:
    """Raise ValueError (in plain English) if the zero-loss peak of the low
    loss ``signal`` can't be aligned."""
    am = signal.axes_manager
    if am.signal_dimension != 1 or am.navigation_size < 2:
        raise ValueError("Zero-loss alignment needs a spectrum image with more than one spectrum")
    axis = am.signal_axes[0]
    if not axis.is_uniform or not axis.scale > 0:
        raise ValueError("Zero-loss alignment needs an energy axis that increases evenly")
    if axis.size < MIN_CHANNELS:
        raise ValueError(f"Zero-loss alignment needs at least {MIN_CHANNELS} channels")


def _title(signal) -> str:
    return signal.metadata.General.get_item("title", "") or "spectrum image"


def _empty_outside(signal, first: float, last: float, coarse, fine, calibration: float) -> None:
    """Set to 0, in every spectrum of ``signal`` (which was ``first`` ... ``last``
    on its energy axis before it was shifted by ``coarse`` and then ``fine`` and
    its offset lowered by ``calibration``), the channels that nothing was moved
    into from the spectrum."""
    axis = signal.axes_manager.signal_axes[0]
    lo = first + fine + np.maximum(coarse, 0) - calibration
    hi = last + fine + np.minimum(coarse, 0) - calibration
    energy = axis.axis
    tol = 1e-6 * axis.scale
    empty = (energy < lo[..., None] - tol) | (energy > hi[..., None] + tol)
    signal.data[empty] = 0


def align_zero_loss_peak(low, others, options: ZlpOptions, report: Report = lambda f, m=None: None) -> ZlpResult:
    """Align the zero-loss peak of the in-memory EELS signal ``low`` (one or
    two navigation axes) and apply the same shifts to ``others`` (EELS signals
    on the same scan). Returns the aligned copies, ``low`` first, and the
    shifts. ``report(fraction, message)`` is called between the steps."""
    check_signal(low)
    nav = tuple(low.axes_manager._navigation_shape_in_array)
    for o in others:
        if tuple(o.axes_manager._navigation_shape_in_array) != nav:
            raise ValueError(f"{_title(o)} is not on the same scan as {_title(low)}")
        check_signal(o)

    signals = [low.deepcopy(), *[o.deepcopy() for o in others]]
    # Shifting interpolates: counts become floating point (kept as they were
    # if they were; the work is done in double precision).
    dtypes = [s.data.dtype if s.data.dtype.kind == "f" else np.dtype("float32") for s in signals]
    for s in signals:
        if s.data.dtype.kind != "f":
            s.change_dtype("float32")
    ll, rest = signals[0], signals[1:]
    # Energy shifts are in the low loss's units; the others' may differ.
    ratio = [_units_in_ev(low) / _units_in_ev(s) for s in signals]
    # (HyperSpy extrapolates what a shift moves in from outside the spectrum, whatever
    # its ``fill_value``: without cropping, those channels are emptied afterwards.)
    ends = [(float(s.axes_manager.signal_axes[0].axis[0]), float(s.axes_manager.signal_axes[0].axis[-1])) for s in signals]

    # Work, in channels summed over the spectra (the weights of the progress).
    weights = [s.data.size for s in signals]
    total = 0.05 * weights[0] + sum(weights)
    if options.subpixel:
        total += 0.5 * weights[0] + sum(weights)
    done = 0.0

    def step(message: str, work: float) -> None:
        nonlocal done
        report(done / total, message)
        done += work

    def shift_all(shifts: np.ndarray, message_for: Callable[[object], str]) -> None:
        # The others first, as exSpy does: the low loss's data is the reference.
        for s, r in zip([*rest, ll], [*ratio[1:], ratio[0]]):
            step(message_for(s), s.data.size)
            s.shift1D(shifts * r, crop=options.crop, show_progressbar=False)

    def recalibrate(centre: float) -> None:
        for s, r in zip(signals, ratio):
            s.axes_manager[-1].offset -= centre * r

    with quiet():
        step("Finding the zero-loss peaks", 0.05 * weights[0])
        peaks = ll.estimate_zero_loss_peak_centre().data
        mean = float(np.nanmean(peaks))
        first = -peaks + mean
        total_shift = first.astype(float)
        fine = np.zeros(nav)
        shift_all(first, lambda s: f"Shifting {_title(s)}")
        centre = float(np.nanmean(ll.estimate_zero_loss_peak_centre().data))
        recalibrate(centre)
        calibration = centre

        if options.subpixel:
            axis = ll.axes_manager.signal_axes[0]
            lo, hi = -SUBPIXEL_RANGE / _units_in_ev(low), SUBPIXEL_RANGE / _units_in_ev(low)
            lo, hi = max(lo, float(axis.axis[0])), min(hi, float(axis.axis[-1]))
            if hi - lo < MIN_RANGE_CHANNELS * axis.scale:
                raise ValueError("The zero-loss peak is too close to the edge of the spectrum to be refined")
            step("Refining the alignment", 0.5 * weights[0])
            reference = tuple(0 for _ in ll.axes_manager.navigation_axes)
            fine = np.asarray(
                ll.estimate_shift1D(lo, hi, reference_indices=reference, show_progressbar=False), dtype=float
            ).reshape(nav)
            shift_all(fine, lambda s: f"Shifting {_title(s)} again")
            total_shift = total_shift + fine
            centre = float(np.nanmean(ll.estimate_zero_loss_peak_centre().data))
            recalibrate(centre)
            calibration += centre
        report(1.0)

    if not options.crop:
        for s, r, (first_energy, last_energy) in zip(signals, ratio, ends):
            _empty_outside(s, first_energy, last_energy, first * r, fine * r, calibration * r)
    for s, dtype in zip(signals, dtypes):
        if s.data.dtype != dtype:
            s.data = s.data.astype(dtype)

    # What each spectrum was moved by, in eV: the shifts and the changes of offset.
    applied = (total_shift - calibration) * _units_in_ev(low)
    return ZlpResult(
        signals=signals,
        shifts=applied.astype(np.float32),
        mean_shift=float(np.nanmean(applied)) if np.isfinite(applied).any() else 0.0,
    )


def shift_map_signal(si_signal, data: np.ndarray):
    """The shifts ``data`` (eV) as a dataset on the spectrum image's scan: an
    image for a 2D scan, a profile along the scan for a line scan."""
    import hyperspy.api as hs

    from .element_maps import scan_image

    title = "ZLP shift"
    nav = si_signal.axes_manager.navigation_axes
    if len(nav) == 2:
        out = scan_image(si_signal, data, title)
    else:
        axis = nav[0].get_axis_dictionary()
        axis["navigate"] = False
        out = hs.signals.Signal1D(np.asarray(data, np.float32), axes=[axis])
        out.metadata.General.title = title
    out.metadata.set_item("Signal.quantity", "Shift (eV)")
    return out


def round_energy(value: float) -> float:
    """``value`` without floating-point noise (a window moved by a shift)."""
    return round(value, 6) if math.isfinite(value) else value
