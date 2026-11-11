"""Browsing a Gatan in-situ series (see ``io/insitu.py``) one frame at a time.

A series is far too big to load (tens of GB), so its dataset holds ONE frame,
or a reduction of a frame range, in memory as an ordinary 2D spectrum image:
every spectrum-image tool works on it unchanged. The dataset's ``insitu``
attribute is an :class:`InsituSeries`, which owns the lazy series signal, the
frame cache and the read-ahead, and says what the dataset currently holds.

The unit of reading is the whole frame (a pixel's spectrum costs as much as a
frame: the raw layout is (E, y, x) per frame). Frames read are kept in an LRU
cache bounded by bytes; after a frame is shown its neighbours are read ahead
on a background thread. Nothing here reads more than one frame at a time
except :meth:`InsituSeries.reduce`, which is run as a job.

A series can be rebinned (:meth:`InsituSeries.set_binning`): every frame read
from then on is binned as it is read, along y, x and energy (summed blocks,
the remainder cropped, as HyperSpy's rebin) and along time (a frame is then
the sum of that many recorded frames, at their mean time). Frame indices are
then in binned frames.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

import numpy as np

# Frames kept in memory besides the one shown (the cache is bounded by bytes).
FRAME_CACHE_BYTES = 1_500_000_000

# Frames read ahead after one is shown, nearest first.
READ_AHEAD = (1, -1)

MODES = ("frame", "mean", "sum")


class FrameCache:
    """Frames by index, least recently used out first once the total size
    exceeds ``max_bytes``. The newest frame always stays, however big."""

    def __init__(self, max_bytes: int = FRAME_CACHE_BYTES) -> None:
        self.max_bytes = int(max_bytes)
        self._frames: OrderedDict[int, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def nbytes(self) -> int:
        with self._lock:
            return sum(f.nbytes for f in self._frames.values())

    def keys(self) -> list[int]:
        """Cached frame indices, least recently used first."""
        with self._lock:
            return list(self._frames)

    def __contains__(self, k: int) -> bool:
        with self._lock:
            return k in self._frames

    def get(self, k: int) -> np.ndarray | None:
        with self._lock:
            frame = self._frames.get(k)
            if frame is not None:
                self._frames.move_to_end(k)
            return frame

    def put(self, k: int, frame: np.ndarray) -> None:
        with self._lock:
            self._frames[k] = frame
            self._frames.move_to_end(k)
            total = sum(f.nbytes for f in self._frames.values())
            while total > self.max_bytes and len(self._frames) > 1:
                _, old = self._frames.popitem(last=False)
                total -= old.nbytes

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()


@dataclass(frozen=True)
class ViewState:
    """What the dataset holds: one frame, or the mean/sum of frames
    ``start``…``end`` (0-based, inclusive)."""

    mode: str = "frame"
    start: int = 0
    end: int = 0

    def label(self) -> str:
        """e.g. "frame 37", "mean of frames 5–20" (frames counted from 1)."""
        if self.mode == "frame" or self.start == self.end:
            return f"frame {self.start + 1}"
        return f"{self.mode} of frames {self.start + 1}–{self.end + 1}"

    def as_dict(self) -> dict:
        return {"mode": self.mode, "start": self.start, "end": self.end}


class InsituSeries:
    """The series behind a dataset: lazy signal, frame cache, read-ahead and
    the current :class:`ViewState`."""

    def __init__(self, signal, cache_bytes: int | None = None) -> None:
        self.signal = signal  # lazy, (frames, y, x, E)
        md = signal.metadata.Insitu.as_dictionary()
        self.meta = md
        self.n_raw = int(md["frames"])
        self.times_raw = np.asarray(md.get("frame_times_s") or np.arange(self.n_raw), dtype=np.float64)
        self.raw_frame_shape = tuple(int(s) for s in signal.data.shape[1:])  # (y, x, E)
        # Binning factors along (time, y, x, E); see set_binning.
        self.binning: tuple[int, int, int, int] = (1, 1, 1, 1)
        self.cache = FrameCache(FRAME_CACHE_BYTES if cache_bytes is None else cache_bytes)
        self.view = ViewState()
        # ADF images (at the ADF's resolution) of the views held lately, by view.
        self.adf_images: OrderedDict[ViewState, np.ndarray] = OrderedDict()
        self.closed = False
        self._guard = threading.Lock()
        self._inflight: dict[int, threading.Event] = {}
        self._ahead_gen = 0
        self._ahead = ThreadPoolExecutor(max_workers=1, thread_name_prefix="readahead")

    # ---------- binning ----------

    @property
    def n(self) -> int:
        """Frames of the series as browsed (binned along time)."""
        return self.n_raw // self.binning[0]

    @property
    def times(self) -> np.ndarray:
        """Time of each (binned) frame: the mean of the frames binned in it."""
        ft = self.binning[0]
        return self.times_raw[: self.n * ft].reshape(self.n, ft).mean(axis=1)

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        """(y, x, E) of a (binned) frame."""
        return tuple(s // f for s, f in zip(self.raw_frame_shape, self.binning[1:]))

    def set_binning(self, binning: tuple[int, int, int, int]) -> None:
        """Bin every frame read from now on by ``binning`` (time, y, x, E),
        counted from the recorded frames. The cache and the ADF images (of
        the old binning) are dropped. Raises ValueError for a factor below 1
        or one that leaves nothing."""
        binning = tuple(int(f) for f in binning)
        sizes = (self.n_raw, *self.raw_frame_shape)
        if len(binning) != 4 or any(not 1 <= f <= n for f, n in zip(binning, sizes)):
            raise ValueError(f"Binning factors must be between 1 and {list(sizes)}")
        self.cancel_ahead()
        self.binning = binning  # type: ignore[assignment]
        self.cache.clear()
        self.adf_images.clear()

    def _binned(self, raw: list[np.ndarray]) -> np.ndarray:
        """The frame binned from the recorded frames ``raw`` (one per frame of
        the time bin), each (y, x, E): summed along time and in blocks along
        y, x and E. Integers stay integers (uint32 when the counts fit)."""
        fy, fx, fe = self.binning[1:]
        y, x, e = self.frame_shape
        acc = np.zeros((y, x, e), dtype=np.float64 if raw[0].dtype.kind == "f" else np.int64)
        for f in raw:
            block = f[: y * fy, : x * fx, : e * fe].reshape(y, fy, x, fx, e, fe)
            acc += block.sum(axis=(1, 3, 5), dtype=acc.dtype)
        if acc.dtype.kind == "f":
            return acc.astype(raw[0].dtype if raw[0].dtype.itemsize >= 4 else np.float32)
        if acc.min(initial=0) >= 0 and acc.max(initial=0) < 2**32:
            return acc.astype(np.uint32)
        return acc

    # ---------- the dataset's signal ----------

    def placeholder(self):
        """A lazy signal for frame 0 with the series' axes and metadata but no
        ``Insitu`` tags (it is one frame, an ordinary spectrum image). Nothing
        is read."""
        s = self.signal.inav[:, :, 0]  # Time is the last navigation axis
        if s.metadata.get_item("Insitu") is not None:
            del s.metadata.Insitu
        return s

    @staticmethod
    def fresh_signal(current, data: np.ndarray):
        """``current`` (the dataset's signal, lazy or not) with ``data`` in
        place of its data: same axes (calibrations the user set), metadata and
        class. ``data`` is not copied."""
        if getattr(current, "_lazy", False):
            s = current.deepcopy()
            s.data = data
            s._lazy = False
            s._assign_subclass()
            return s
        return current._deepcopy_with_new_data(data)

    # ---------- reading ----------

    def _read_raw(self, k: int) -> np.ndarray:
        return np.asarray(self.signal.data[k].compute(scheduler="synchronous"))

    def _read_now(self, k: int) -> np.ndarray:
        """(Binned) frame ``k``, read: the recorded frames of its time bin."""
        ft = self.binning[0]
        if self.binning == (1, 1, 1, 1):
            frame = self._read_raw(k)
        else:
            frame = self._binned([self._read_raw(j) for j in range(k * ft, (k + 1) * ft)])
        frame.flags.writeable = False  # shared with the cache: tools must not change it in place
        return frame

    def read(self, k: int, keep: bool = True) -> np.ndarray:
        """Frame ``k`` as (y, x, E): from the cache, else read (one sequential
        read; a frame being read by another thread is waited for, not read
        twice). ``keep``: put it in the cache."""
        if not 0 <= k < self.n:
            raise IndexError(f"Frame {k + 1} out of range 1–{self.n}")
        hit = self.cache.get(k)
        if hit is not None:
            return hit
        with self._guard:
            event = self._inflight.get(k)
            owner = event is None
            if owner:
                event = self._inflight[k] = threading.Event()
        if not owner:
            event.wait()
            return self.read(k, keep)  # the owner's result is cached now (or it failed: we read it)
        try:
            binning = self.binning
            frame = self._read_now(k)
            if keep and self.binning == binning:  # not a frame of a binning since replaced
                self.cache.put(k, frame)
            return frame
        finally:
            with self._guard:
                self._inflight.pop(k, None)
            event.set()

    def cancel_ahead(self) -> None:
        """Drop the read-ahead tasks that have not started yet."""
        with self._guard:
            self._ahead_gen += 1

    def schedule_ahead(self, k: int) -> None:
        """Read the neighbours of frame ``k`` in the background. Tasks of an
        earlier call that have not started yet are dropped."""
        self.cancel_ahead()
        gen = self._ahead_gen
        if self.closed:
            return
        for step in READ_AHEAD:
            j = k + step
            if 0 <= j < self.n and j not in self.cache:
                try:
                    self._ahead.submit(self._ahead_task, j, gen)
                except RuntimeError:  # shut down
                    return

    def _ahead_task(self, k: int, gen: int) -> None:
        if gen != self._ahead_gen or self.closed:
            return
        try:
            self.read(k)
        except Exception:  # read-ahead is only a courtesy
            pass

    def load_view(self, ds) -> None:
        """Read the frame the dataset is to hold if it only has the lazy
        placeholder (first use)."""
        if not getattr(ds.signal, "_lazy", False):
            return
        k = self.view.start
        ds.signal = self.fresh_signal(ds.signal, self.read(k))
        self.schedule_ahead(k)

    def reduce(
        self, mode: str, start: int, end: int, report: Callable[..., None] = lambda f, m=None: None
    ) -> np.ndarray:
        """The mean or sum of frames ``start``…``end`` (inclusive) as float32
        (y, x, E). Reads every frame of the range (those in the cache are not
        read again; the others are not added to it) and reports progress
        (``report(fraction, "Frame 12 of 40")``, which may raise to cancel)."""
        if mode not in ("mean", "sum"):
            raise ValueError(f"Unknown reduction: {mode}")
        if not 0 <= start <= end < self.n:
            raise IndexError(f"Frames {start + 1}–{end + 1} out of range 1–{self.n}")
        count = end - start + 1
        total = np.zeros(self.frame_shape, dtype=np.float64)
        for i, k in enumerate(range(start, end + 1)):
            report(i / count, f"Frame {i + 1} of {count}")
            total += self.read(k, keep=False)
        if mode == "mean":
            total /= count
        return total.astype(np.float32)

    # ---------- summaries ----------

    def summary(self) -> dict:
        md = self.meta
        times = self.times
        return {
            "frames": self.n,
            "recorded_frames": md.get("recorded_frames"),
            "time_binning": self.binning[0],
            "frame_shape": [int(s) for s in self.frame_shape],
            "first_time_s": float(times[0]),
            "last_time_s": float(times[-1]),
            "time_source": md.get("time_source"),
            "duration_s": md.get("duration_s"),
            "note": md.get("note"),
            "frame_times_s": [float(t) for t in times],
            "frame_step_s": float(np.median(np.diff(times))) if self.n > 1 else None,
            "start_time": md.get("start_time"),
            "frame_bytes": md.get("frame_bytes"),
            "raw_bytes": md.get("raw_bytes"),
            "raw_files": list(md.get("raw_files") or []),
            "view": self.view.as_dict(),
        }

    def close(self) -> None:
        self.closed = True
        self._ahead.shutdown(wait=False, cancel_futures=True)
        self.cache.clear()


# ---------- the simultaneous ADF as navigator ----------

# Id of the ADF among a series' navigator sources (the other sources are
# dataset ids).
ADF_SOURCE_ID = "insitu-adf"


def adf_stack(ds, siblings: list):
    """(ADF stack, k) for the series dataset ``ds``: a stack of its file with
    one frame per frame of the series whose frames are the series' scan
    (k = 1), or the same field of view sampled k times more finely in both
    directions (the sub-scan of Gatan DM; see ``subscan.py``). None without
    such a stack."""
    if ds.insitu is None:
        return None
    from .session import DatasetKind
    from .subscan import scan_factor_of

    for d in siblings:
        if d is ds or d.kind is not DatasetKind.STACK:
            continue
        s = d.loaded_signal
        if s.metadata.get_item("Insitu.frames") is None or s.axes_manager.navigation_size != ds.insitu.n_raw:
            continue
        k = scan_factor_of(ds, d)
        if k is not None:
            return d, k
    return None


def adf_image(ds, siblings: list) -> np.ndarray | None:
    """The ADF image matching what ``ds`` holds, at the ADF's own resolution
    (k times finer than the scan for a sub-scan, see :func:`adf_stack`): the
    ADF frame of the same index, or the mean/sum of the same range of frames.
    A series binned along time takes the ADF frames binned the same way (each
    the sum of the recorded frames of its bin). None if the file has no
    matching ADF."""
    found = adf_stack(ds, siblings)
    if found is None:
        return None
    adf, _ = found
    series = ds.insitu
    view = series.view
    hit = series.adf_images.get(view)
    if hit is not None:
        return hit
    ft = series.binning[0]
    data = adf.loaded_signal.data
    chunk = data[view.start * ft : (view.end + 1) * ft]
    chunk = np.asarray(chunk.compute() if hasattr(chunk, "compute") else chunk, dtype=np.float64)
    if ft > 1:
        chunk = chunk.reshape(-1, ft, *chunk.shape[1:]).sum(axis=1)
    if view.mode == "frame" or view.start == view.end:
        image = chunk[0]
    elif view.mode == "mean":
        image = chunk.mean(axis=0)
    else:
        image = chunk.sum(axis=0)
    image = image.astype(np.float32, copy=False)
    series.adf_images[view] = image
    while len(series.adf_images) > 16:
        series.adf_images.popitem(last=False)
    return image
