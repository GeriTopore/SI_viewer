"""Gatan in-situ series: a ``.dm4`` plus its frames in ``.raw`` chunk files.

Per detector, GMS writes ``<name>.dm4`` (one frame and all the tags),
``<name>.raw``, ``<name>.raw_1``, ``<name>.raw_2``, … The chunk files are one
byte stream (a frame can straddle two of them) holding the frames back to
back, each in DM order, first dimension (``Data0``, x) fastest: a NumPy
C-order array of shape (Data2, Data1, Data0) = (E, y, x) for a spectrum image
and (Data1, Data0) = (y, x) for an image.

The signal built here is lazy, one dask chunk per frame, each read with ONE
sequential read of the frame's bytes (a pixel's spectrum is strided over the
whole frame, so there is nothing cheaper than the whole frame). Nothing but the
``.dm4`` tags and the raw file sizes is read at open time.

Layout, in HyperSpy order:

- spectrum image: (t, y, x, E), navigation axes Time, y, x
- image series:   (t, y, x), navigation axis Time, signal axes y, x

``metadata.Insitu`` records ``frames`` (the complete frames used),
``recorded_frames`` (``# Frames`` in the tags, or None), ``frame_times_s``
(exact time of each frame, in seconds since the start of the experiment: the
moment the frame was finished), ``time_source`` (``"recorded"``,
``"duration"`` or ``"index"``), ``duration_s``, ``start_time``, ``frame_bytes``,
``raw_files``, ``raw_bytes`` (their total size) and, when the run was cut short or the files are still being
copied, ``note``. The Time axis is uniform (offset = first frame time, scale =
mean spacing); the exact times are in ``frame_times_s``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# DM's image data type codes (RosettaSciIO's digitalmicrograph reader) that are
# plain NumPy types; the raw files are little-endian.
DM_DTYPES = {
    1: "<i2",
    2: "<f4",
    3: "<c8",
    6: "u1",
    7: "<i4",
    9: "i1",
    10: "<u2",
    11: "<u4",
    12: "<f8",
    13: "<c16",
}

_INSITU_KEYS = ("In-situ", "Insitu", "In_situ")
_INDEX_KEY = re.compile(r"^\[(\d+)\]$")


class NotInsitu(ValueError):
    """The ``.dm4`` has no usable in-situ tags."""


@dataclass
class InsituInfo:
    """What the ``Insitu`` tag group says about a series."""

    dims: tuple[int, ...]  # (Data0, Data1[, Data2]) = (x, y[, energy])
    dtype: np.dtype
    frame_bytes: int
    recorded_frames: int | None = None
    duration_s: float | None = None
    start_time: str | None = None
    frame_times_ns: np.ndarray | None = None  # per recorded frame, decoded
    calibrations: list[tuple[float, float, str]] = field(default_factory=list)  # per dim: origin, scale, unit

    @property
    def is_spectrum_image(self) -> bool:
        return len(self.dims) == 3

    @property
    def frame_shape(self) -> tuple[int, ...]:
        """Shape of one frame as stored: (E, y, x) or (y, x)."""
        return tuple(reversed(self.dims))


# ---------- tags ----------


def find_insitu_tags(original_metadata: dict) -> dict | None:
    """The ``Insitu`` tag group inside a DM file's ``original_metadata``
    (a plain dict, ``ImageList.TagGroup0.ImageTags.<group>``), if any."""
    images = original_metadata.get("ImageList") if isinstance(original_metadata, dict) else None
    if not isinstance(images, dict):
        return None
    for group in images.values():
        tags = group.get("ImageTags") if isinstance(group, dict) else None
        if not isinstance(tags, dict):
            continue
        for key in _INSITU_KEYS:
            if isinstance(tags.get(key), dict):
                return tags[key]
    return None


def decode_frame_times(per_frame) -> np.ndarray | None:
    """Per-frame times in ns from ``Sync/Per Frame/Time Since Start (ns)``.

    The value is a list of 32-bit words, two per frame (low, high); some
    readers return it as a dict with keys "[0]", "[1]", … (sorted as text:
    the index in the key is what counts). None if it doesn't decode to
    increasing times.
    """
    if per_frame is None:
        return None
    if isinstance(per_frame, dict):
        items = []
        for key, value in per_frame.items():
            m = _INDEX_KEY.match(str(key).strip())
            if m is None:
                return None
            items.append((int(m.group(1)), value))
        words = [v for _, v in sorted(items)]
    else:
        words = list(np.asarray(per_frame).ravel())
    if len(words) < 2 or len(words) % 2:
        return None
    try:
        w = np.asarray(words, dtype=np.int64)
    except (TypeError, ValueError):
        return None
    if (w < 0).any() or (w > 0xFFFFFFFF).any():
        return None
    times = (w[1::2] << 32) | w[0::2]
    if len(times) > 1 and not (np.diff(times) > 0).all():
        return None
    return times


def _number(d: dict, key: str) -> float | None:
    v = d.get(key) if isinstance(d, dict) else None
    return float(v) if isinstance(v, (int, float, np.number)) and not isinstance(v, bool) else None


def parse_insitu_tags(insitu: dict, fallback_shape: tuple[int, ...] | None = None, fallback_dtype=None) -> InsituInfo:
    """Read an ``Insitu`` tag group (a plain dict). ``fallback_shape`` (the
    ``.dm4`` signal's array shape, (y, x[, E])) and ``fallback_dtype`` fill in
    what older files lack. Raises :class:`NotInsitu` if the frame can't be
    worked out."""
    fmt = insitu.get("Raw File Format Info")
    fmt = fmt if isinstance(fmt, dict) else {}
    dims_tag = fmt.get("Dimensions")
    dims: list[int] = []
    if isinstance(dims_tag, dict):
        i = 0
        while f"Data{i}" in dims_tag:
            dims.append(int(dims_tag[f"Data{i}"]))
            i += 1
    if not dims and fallback_shape is not None:
        dims = list(reversed(fallback_shape))
    if len(dims) not in (2, 3) or any(d < 1 for d in dims):
        raise NotInsitu("The in-situ tags don't give a 2D or 3D frame")

    code = fmt.get("Data Type")
    if code in DM_DTYPES:
        dtype = np.dtype(DM_DTYPES[code])
    elif fallback_dtype is not None:
        dtype = np.dtype(fallback_dtype).newbyteorder("<")
    else:
        raise NotInsitu(f"Unsupported in-situ data type {code!r}")

    frame_bytes = int(np.prod(dims, dtype=np.int64)) * dtype.itemsize
    stated = _number(fmt, "Frame Size (bytes)")
    if stated is not None and int(stated) != frame_bytes:
        # Trust the stated size and the dimensions only if they agree.
        raise NotInsitu(f"Frame size {int(stated)} B doesn't match {dims} × {dtype}")

    calibrations = []
    cal = fmt.get("Calibrations")
    for i in range(len(dims)):
        c = cal.get(f"dim-{i}") if isinstance(cal, dict) else None
        if isinstance(c, dict):
            unit = c.get("Unit")
            calibrations.append(
                (_number(c, "Origin") or 0.0, _number(c, "Scale") or 1.0, unit if isinstance(unit, str) else "")
            )
        else:
            calibrations.append((0.0, 1.0, ""))

    rec = insitu.get("Recorded")
    rec = rec if isinstance(rec, dict) else {}
    frames = _number(rec, "# Frames")
    sync = rec.get("Sync")
    per_frame = sync.get("Per Frame") if isinstance(sync, dict) else None
    times = decode_frame_times(per_frame.get("Time Since Start (ns)") if isinstance(per_frame, dict) else None)
    start = rec.get("Experiment Start Time")
    return InsituInfo(
        dims=tuple(dims),
        dtype=dtype,
        frame_bytes=frame_bytes,
        recorded_frames=int(frames) if frames is not None else None,
        duration_s=_number(rec, "Duration (s)"),
        start_time=start if isinstance(start, str) else None,
        frame_times_ns=times,
        calibrations=calibrations,
    )


# ---------- raw files ----------


def raw_paths(dm4: Path) -> list[Path]:
    """``<stem>.raw``, ``<stem>.raw_1``, … beside ``dm4``, as far as they run
    consecutively (empty if there is no ``<stem>.raw``)."""
    first = dm4.with_suffix(".raw")
    if not first.is_file():
        return []
    paths = [first]
    while True:
        nxt = dm4.with_name(f"{dm4.stem}.raw_{len(paths)}")
        if not nxt.is_file():
            return paths
        paths.append(nxt)


class RawFrames:
    """Equal-size frames laid back to back across chunk files.

    ``n_frames`` counts the COMPLETE frames in the files as they were when this
    was made. A frame is one sequential read (two where it straddles chunks).
    """

    def __init__(self, paths: list[Path], frame_bytes: int, recorded_frames: int | None = None) -> None:
        self.frame_bytes = int(frame_bytes)
        paths = [Path(p) for p in paths]
        sizes = [p.stat().st_size for p in paths]
        covers = recorded_frames is not None and sum(sizes) // self.frame_bytes >= recorded_frames
        if paths and not covers:
            # Still being copied or cut short: a chunk that isn't full (all
            # chunks but the last are as big as the first) ends the stream.
            keep = 1
            while keep < len(paths) and sizes[keep - 1] == sizes[0]:
                keep += 1
            paths, sizes = paths[:keep], sizes[:keep]
        self.paths = paths
        self.sizes = sizes
        self.starts = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        self.n_frames = int(self.starts[-1] // self.frame_bytes)

    def read_bytes(self, k: int) -> np.ndarray:
        """The bytes of frame ``k`` as a uint8 array (one buffer, filled in
        place)."""
        if not 0 <= k < self.n_frames:
            raise IndexError(f"Frame {k} out of range 0–{self.n_frames - 1}")
        buf = np.empty(self.frame_bytes, dtype=np.uint8)
        view = memoryview(buf)
        pos, end = k * self.frame_bytes, (k + 1) * self.frame_bytes
        filled = 0
        for i, path in enumerate(self.paths):
            lo, hi = int(self.starts[i]), int(self.starts[i + 1])
            if hi <= pos or lo >= end:
                continue
            start = max(pos, lo)
            n = min(end, hi) - start
            with open(path, "rb", buffering=0) as f:
                f.seek(start - lo)
                got = 0
                while got < n:
                    r = f.readinto(view[filled + got : filled + n])
                    if not r:
                        raise OSError(f"{path.name} ended early (frame {k})")
                    got += r
            filled += n
        return buf


class FrameReader:
    """Reads frame ``k`` as an array in HyperSpy order, with a leading length-1
    time axis: (1, y, x, E) for a spectrum image, (1, y, x) for an image.
    Picklable (dask may ship it to workers)."""

    def __init__(self, raw: RawFrames, dtype: np.dtype, frame_shape: tuple[int, ...]) -> None:
        self.raw = raw
        self.dtype = np.dtype(dtype)
        self.frame_shape = tuple(frame_shape)  # as stored: (E, y, x) or (y, x)

    @property
    def out_shape(self) -> tuple[int, ...]:
        s = self.frame_shape
        return (1, s[1], s[2], s[0]) if len(s) == 3 else (1, *s)

    def __call__(self, k: int) -> np.ndarray:
        frame = self.raw.read_bytes(k).view(self.dtype).reshape(self.frame_shape)
        if frame.ndim == 3:
            frame = np.ascontiguousarray(frame.transpose(1, 2, 0))  # (E, y, x) → (y, x, E)
        return frame[np.newaxis]


def lazy_series(reader: FrameReader, n_frames: int, token: str):
    """A dask array of shape (n_frames, *frame) with one chunk per frame."""
    import dask
    import dask.array as da

    blocks = [
        da.from_delayed(
            dask.delayed(reader, pure=True)(k, dask_key_name=f"insitu-{token}-{k}"),
            shape=reader.out_shape,
            dtype=reader.dtype,
        )
        for k in range(n_frames)
    ]
    return da.concatenate(blocks, axis=0)


# ---------- time axis ----------


def frame_times(info: InsituInfo, n: int) -> tuple[np.ndarray, str]:
    """(seconds per frame, source) for ``n`` frames: the recorded times, else
    duration ÷ recorded frames, else the frame index."""
    t = info.frame_times_ns
    if t is not None and len(t) >= n:
        return t[:n].astype(np.float64) / 1e9, "recorded"
    recorded = info.recorded_frames or n
    if info.duration_s and info.duration_s > 0:
        step = info.duration_s / recorded
        return step * np.arange(1, n + 1), "duration"
    return np.arange(n, dtype=np.float64), "index"


def _time_axis(times: np.ndarray, source: str) -> dict:
    n = len(times)
    if source == "index":
        return {"name": "Frame", "units": "", "size": n, "scale": 1.0, "offset": 0.0, "navigate": True}
    scale = float(times[-1] - times[0]) / (n - 1) if n > 1 else (float(times[0]) or 1.0)
    return {
        "name": "Time", "units": "s", "size": n,
        "scale": scale if scale > 0 else 1.0, "offset": float(times[0]), "navigate": True,
    }


# ---------- building the signal ----------


def _defined(value) -> str | None:
    return value if isinstance(value, str) and value != "None" else None


def _axes(base, info: InsituInfo) -> list[dict]:
    """Axis dicts (array order: y, x[, E]) from the ``.dm4`` signal, or from the
    in-situ calibrations when the ``.dm4`` signal isn't of the frame's shape."""
    order = [1, 0, 2][: len(info.dims)]  # array order y, x[, E] = Data1, Data0[, Data2]
    names = ["x", "y", "Energy loss"]  # by Data index
    sizes = tuple(info.dims[i] for i in order)
    ams = base.axes_manager._axes if tuple(base.data.shape) == sizes else None
    nav = len(info.dims) == 3  # SI: y, x navigate and E is the signal; image: y, x are the signal
    out = []
    for pos, i in enumerate(order):
        origin, scale, unit = info.calibrations[i]
        d = {"size": info.dims[i], "scale": scale, "offset": origin, "units": unit or None, "name": names[i]}
        if ams is not None:
            ax = ams[pos]
            d.update(
                name=_defined(ax.name) or names[i], units=_defined(ax.units),
                scale=float(ax.scale), offset=float(ax.offset),
            )
        d["navigate"] = nav and pos < 2
        if d["units"] is None:
            del d["units"]
        out.append(d)
    return out


def build_signal(base, info: InsituInfo, raw: RawFrames, token: str, source_name: str = ""):
    """The lazy series signal for ``base`` (the ``.dm4``'s own, one-frame
    signal) with frames from ``raw``."""
    recorded = info.recorded_frames
    n = raw.n_frames if recorded is None else min(raw.n_frames, recorded)
    if n < 1:
        raise NotInsitu("No complete frame in the .raw files")
    note = None
    if recorded is not None and n < recorded:
        note = f"{n} of {recorded} frames are present in the .raw files"

    times, source = frame_times(info, n)
    reader = FrameReader(raw, info.dtype, info.frame_shape)
    data = lazy_series(reader, n, token)
    axes = [_time_axis(times, source), *_axes(base, info)]

    metadata = base.metadata.as_dictionary()
    metadata["Insitu"] = {
        "frames": n,
        "recorded_frames": recorded,
        "frame_times_s": [float(t) for t in times],
        "time_source": source,
        "duration_s": info.duration_s,
        "start_time": info.start_time,
        "frame_bytes": info.frame_bytes,
        "raw_files": [p.name for p in raw.paths],
        "raw_bytes": int(sum(raw.sizes)),
        **({"note": note} if note else {}),
    }
    signal = type(base)(
        data, axes=axes, metadata=metadata, original_metadata=base.original_metadata.as_dictionary()
    )
    return signal


def is_insitu_series(signal) -> bool:
    """Whether ``signal`` is the lazy spectrum image series made by
    :func:`load_insitu` (time, y, x | energy)."""
    am = signal.axes_manager
    return (
        am.navigation_dimension == 3
        and am.signal_dimension == 1
        and signal.metadata.get_item("Insitu.frames") is not None
    )


def insitu_summary(signal) -> dict | None:
    """What a dataset summary says about an in-situ series (None for other
    signals): frames, the frame's shape (as in the array: y, x[, E]), time
    span and any note."""
    node = signal.metadata.get_item("Insitu")
    if node is None or signal.metadata.get_item("Insitu.frames") is None:
        return None
    md = node.as_dictionary()
    if signal.data.ndim < 3 or signal.data.shape[0] != md["frames"]:
        return None  # something made from a series (a mean image, a crop…) keeps the tags
    times = md.get("frame_times_s") or [0.0]
    return {
        "frames": int(md["frames"]),
        "recorded_frames": md.get("recorded_frames"),
        "frame_shape": [int(n) for n in signal.data.shape[1:]],
        "first_time_s": float(times[0]),
        "last_time_s": float(times[-1]),
        "time_source": md.get("time_source"),
        "duration_s": md.get("duration_s"),
        "note": md.get("note"),
    }


# ---------- opening ----------


def _probe(path: Path):
    """(``.dm4`` signal, InsituInfo) for an in-situ ``.dm4``, else None."""
    if path.suffix.lower() != ".dm4" or not path.with_suffix(".raw").is_file():
        return None
    import warnings

    import hyperspy.api as hs

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = hs.load(str(path), lazy=True)
    if isinstance(base, list):
        base = base[0]
    insitu = find_insitu_tags(base.original_metadata.as_dictionary())
    if insitu is None:
        return None
    shape = tuple(int(s) for s in base.data.shape)
    try:
        info = parse_insitu_tags(insitu, fallback_shape=shape, fallback_dtype=base.data.dtype)
    except NotInsitu:
        return None
    return base, info


def is_insitu(path: str | Path) -> bool:
    """Whether ``path`` is a ``.dm4`` with in-situ tags and its ``.raw`` beside it."""
    return _probe(Path(path)) is not None


def _series(path: Path, probed=None):
    """(signal, info) for one in-situ ``.dm4`` (None if it isn't one)."""
    probed = probed or _probe(path)
    if probed is None:
        return None
    base, info = probed
    raw = RawFrames(raw_paths(path), info.frame_bytes, info.recorded_frames)
    token = f"{path}|{path.stat().st_mtime_ns}"
    from dask.base import tokenize

    return build_signal(base, info, raw, tokenize(token), path.name), info


def load_insitu(path: str | Path) -> list | None:
    """The signals to open for an in-situ ``.dm4``, or None if ``path`` isn't
    one. An EELS spectrum image comes with the image series of the same folder
    that have as many frames (the simultaneous ADF); an image series comes alone."""
    path = Path(path)
    main = _series(path)
    if main is None:
        return None
    signal, info = main
    signals = [signal]
    if info.is_spectrum_image:
        n = signal.axes_manager.navigation_shape[-1]  # Time is the last navigation axis
        signals += [series for _, series in _companions(path, n)]
    return signals


def _companions(path: Path, n: int) -> list[tuple[Path, object]]:
    """(path, signal) of the image series in ``path``'s folder with ``n`` frames."""
    out = []
    for other in sorted(path.parent.glob("*.dm4")):
        if other == path or not other.with_suffix(".raw").is_file():
            continue
        probed = _probe(other)
        if probed is None or probed[1].is_spectrum_image:
            continue
        series = _series(other, probed)
        if series is not None and series[0].axes_manager.navigation_shape[-1] == n:
            out.append((other, series[0]))
    return out


def insitu_companions(path: str | Path) -> list[Path]:
    """The image series that open together with an in-situ spectrum image
    (``load_insitu``): none for anything else."""
    path = Path(path)
    main = _series(path)
    if main is None or not main[1].is_spectrum_image:
        return []
    return [p for p, _ in _companions(path, main[0].axes_manager.navigation_shape[-1])]
