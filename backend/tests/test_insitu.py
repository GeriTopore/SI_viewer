"""Gatan in-situ series (.dm4 + .raw chunks): tags, raw reading, the lazy
signal, opening through the session, and the real ``examples/InSitu (3)``."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

import hyperspy.api as hs

from si_viewer.io import insitu
from si_viewer.io.insitu import (
    FrameReader,
    InsituInfo,
    NotInsitu,
    RawFrames,
    decode_frame_times,
    find_insitu_tags,
    parse_insitu_tags,
)
from si_viewer.session import DatasetKind, Session

# ---------- helpers: a synthetic in-situ folder ----------


def words(times_ns):
    """Per-frame times as the .dm4 stores them: (low, high) 32-bit words."""
    out = []
    for t in times_ns:
        out += [int(t) & 0xFFFFFFFF, int(t) >> 32]
    return out


def frame_array(k, dims, dtype):
    """Frame ``k`` as stored: shape (Data2, Data1, Data0) or (Data1, Data0)."""
    n = int(np.prod(dims))
    return ((np.arange(n) + 7 * k) % 251).reshape(tuple(reversed(dims))).astype(dtype)


def tag_group(dims, code, frames, times_ns=None, duration=None, itemsize=4, with_times_as_dict=False):
    fmt = {
        "Dimensions": {f"Data{i}": d for i, d in enumerate(dims)},
        "Data Type": code,
        "Data Size (bytes)": itemsize,
        "Frame Size (bytes)": int(np.prod(dims)) * itemsize,
        "Calibrations": {
            f"dim-{i}": {"Origin": 1.0 + i, "Scale": 0.5 + i, "Unit": "nm" if i < 2 else "eV"}
            for i in range(len(dims))
        },
    }
    rec = {"# Frames": frames, "Experiment Start Time": "2026-01-27 23:43:16.978"}
    if duration is not None:
        rec["Duration (s)"] = duration
    if times_ns is not None:
        w = words(times_ns)
        if with_times_as_dict:
            # Keys sorted as text, as some readers return them.
            w = dict(sorted({f"[{i}]": v for i, v in enumerate(w)}.items()))
        rec["Sync"] = {"Per Frame": {"Time Since Start (ns)": w}}
    return {"Raw File Format Info": fmt, "Recorded": rec}


def make_base(dims, dtype, tags, title="series", scale=(0.25, 0.25)):
    """The one-frame signal hs.load would give for the .dm4 (lazy)."""
    shape = tuple(reversed(dims))  # (y, x[, E])
    data = np.zeros(shape, dtype=dtype)
    if len(dims) == 3:
        s = hs.signals.Signal1D(data).as_lazy()
        s.set_signal_type("EELS")
        names = ["y", "x", "Energy loss"]
        units = ["nm", "nm", "eV"]
    else:
        s = hs.signals.Signal2D(data).as_lazy()
        names, units = ["y", "x"], ["nm", "nm"]
    for ax, name, unit in zip(s.axes_manager._axes, names, units):
        ax.name, ax.units, ax.scale = name, unit, {"x": scale[0], "y": scale[1]}.get(name, 0.25)
    s.metadata.General.title = title
    s.original_metadata.add_dictionary({"ImageList": {"TagGroup0": {"ImageTags": {"In-situ": tags}}}})
    return s


def write_raw(folder, stem, frames_bytes, chunk_sizes=None):
    """Write the frames' bytes as ``stem.raw``, ``stem.raw_1``, … ``chunk_sizes``
    gives each file's size (the last takes the rest)."""
    blob = b"".join(frames_bytes)
    sizes = list(chunk_sizes or [len(blob)])
    pos, paths = 0, []
    for i, size in enumerate(sizes):
        end = len(blob) if i == len(sizes) - 1 else pos + size
        path = folder / (f"{stem}.raw" if i == 0 else f"{stem}.raw_{i}")
        path.write_bytes(blob[pos:end])
        paths.append(path)
        pos = end
    return paths


@pytest.fixture
def fake_dm4(monkeypatch):
    """hs.load answers for the .dm4 files registered here (by file name)."""
    registry = {}
    real_load = hs.load

    def load(path, *a, **kw):
        name = str(path).rsplit("/", 1)[-1]
        if name in registry:
            return registry[name].deepcopy()
        return real_load(path, *a, **kw)

    monkeypatch.setattr(hs, "load", load)
    return registry


def make_series(folder, registry, stem, dims, code, dtype, n_frames, recorded=None, chunk_sizes=None,
                times_ns=None, duration=None, times_as_dict=False, bytes_extra=0, scale=(0.25, 0.25)):
    itemsize = np.dtype(dtype).itemsize
    tags = tag_group(dims, code, recorded or n_frames, times_ns, duration, itemsize, times_as_dict)
    (folder / f"{stem}.dm4").write_bytes(b"")
    registry[f"{stem}.dm4"] = make_base(dims, dtype, tags, stem, scale)
    frames = [frame_array(k, dims, dtype) for k in range(n_frames)]
    write_raw(folder, stem, [f.tobytes() for f in frames] + [b"\0" * bytes_extra], chunk_sizes)
    return folder / f"{stem}.dm4", frames


# ---------- tags ----------


def test_decode_frame_times_list_and_carry():
    ns = [1_259_872_016 + 4 * 2**32, 21_864_351_700, 25_281_477_000]
    t = decode_frame_times(words(ns))
    assert t.tolist() == ns
    assert t[0] / 1e9 == pytest.approx(18.4397412)


def test_decode_frame_times_dict_keys_sorted_as_text():
    ns = [(k + 1) * 3_400_000_000 for k in range(12)]  # more than 10 frames: "[10]" sorts before "[2]"
    w = {f"[{i}]": v for i, v in enumerate(words(ns))}
    assert list(dict(sorted(w.items()))) != list(w)  # the text order differs from the numeric one
    assert decode_frame_times(dict(sorted(w.items()))).tolist() == ns


@pytest.mark.parametrize(
    "bad",
    [None, [], [1], [1, 2, 3], words([5, 5]), words([9, 3]), [-1, 0], [2**33, 0], {"x": 1, "y": 2}, "nope"],
)
def test_decode_frame_times_rejects(bad):
    assert decode_frame_times(bad) is None


def test_find_insitu_tags_variants():
    for key in ("In-situ", "Insitu"):
        om = {"ImageList": {"TagGroup0": {"ImageTags": {key: {"a": 1}}}}}
        assert find_insitu_tags(om) == {"a": 1}
    assert find_insitu_tags({"ImageList": {"TagGroup0": {"ImageTags": {"x": {}}}}}) is None
    assert find_insitu_tags({}) is None


def test_parse_tags_dtypes_and_calibrations():
    for code, dtype in [(2, "<f4"), (6, "u1"), (10, "<u2"), (1, "<i2"), (7, "<i4"), (12, "<f8")]:
        item = np.dtype(dtype).itemsize
        info = parse_insitu_tags(tag_group([5, 3, 4], code, 2, itemsize=item))
        assert info.dtype == np.dtype(dtype)
        assert info.dims == (5, 3, 4) and info.frame_shape == (4, 3, 5)
        assert info.frame_bytes == 60 * item
        assert info.calibrations[0] == (1.0, 0.5, "nm") and info.calibrations[2][2] == "eV"
        assert info.is_spectrum_image


def test_parse_tags_falls_back_to_the_dm4_signal():
    info = parse_insitu_tags({"Recorded": {}}, fallback_shape=(3, 5, 4), fallback_dtype=np.float32)
    assert info.dims == (4, 5, 3) and info.dtype == np.dtype("<f4")
    assert info.recorded_frames is None and info.frame_times_ns is None
    assert info.calibrations == [(0.0, 1.0, "")] * 3


def test_parse_tags_rejects():
    with pytest.raises(NotInsitu):
        parse_insitu_tags({"Recorded": {}})  # no dimensions at all
    bad = tag_group([5, 3], 2, 1)
    bad["Raw File Format Info"]["Frame Size (bytes)"] = 999
    with pytest.raises(NotInsitu):
        parse_insitu_tags(bad)
    with pytest.raises(NotInsitu):
        parse_insitu_tags(tag_group([5, 3], 99, 1))  # unknown type code and no fallback


# ---------- raw frames ----------


def test_raw_frames_straddle_and_truncated_tail(tmp_path):
    fb = 12
    blob = bytes(range(80))  # 6 complete frames and 8 spare bytes
    paths = write_raw(tmp_path, "a", [blob], chunk_sizes=[30, 30, 20])
    raw = RawFrames(paths, fb, recorded_frames=7)
    assert raw.n_frames == 6
    for k in range(6):
        assert raw.read_bytes(k).tobytes() == blob[k * fb : (k + 1) * fb]  # frames 2 and 4 straddle
    with pytest.raises(IndexError):
        raw.read_bytes(6)


def test_raw_frames_unfinished_copy_stops_at_a_short_chunk(tmp_path):
    blob = bytes(range(80))
    # The second chunk is still being copied but a third already exists.
    paths = write_raw(tmp_path, "a", [blob], chunk_sizes=[30, 20, 30])
    assert RawFrames(paths, 12, recorded_frames=7).n_frames == 4  # only the first two chunks count
    # Without a recorded count the same rule applies.
    assert RawFrames(paths, 12).n_frames == 4
    # When the files cover every recorded frame they are all used.
    assert RawFrames(paths, 12, recorded_frames=3).n_frames == 6


def test_raw_paths_are_consecutive(tmp_path):
    (tmp_path / "s.dm4").write_bytes(b"")
    assert insitu.raw_paths(tmp_path / "s.dm4") == []
    for name in ("s.raw", "s.raw_1", "s.raw_3"):
        (tmp_path / name).write_bytes(b"x")
    assert [p.name for p in insitu.raw_paths(tmp_path / "s.dm4")] == ["s.raw", "s.raw_1"]


def test_frame_reader_orders_axes(tmp_path):
    dims = [5, 3, 4]  # x=5, y=3, E=4 (not square)
    frames = [frame_array(k, dims, "<f4") for k in range(2)]
    paths = write_raw(tmp_path, "r", [f.tobytes() for f in frames])
    reader = FrameReader(RawFrames(paths, 240), np.dtype("<f4"), (4, 3, 5))
    out = reader(1)
    assert out.shape == (1, 3, 5, 4) and out.flags.c_contiguous
    np.testing.assert_array_equal(out[0], frames[1].transpose(1, 2, 0))  # (E,y,x) → (y,x,E)
    # Data0 is x, fastest: pixel (y=2, x=4) holds the last pixel of each energy plane.
    np.testing.assert_array_equal(out[0, 2, 4, :], frames[1][:, 2, 4])


# ---------- time axis ----------


def test_frame_times_sources():
    info = InsituInfo(dims=(2, 2), dtype=np.dtype("u1"), frame_bytes=4, recorded_frames=4, duration_s=8.0)
    t, src = insitu.frame_times(info, 3)
    assert src == "duration" and t.tolist() == [2.0, 4.0, 6.0]
    info.duration_s = None
    t, src = insitu.frame_times(info, 3)
    assert src == "index" and t.tolist() == [0.0, 1.0, 2.0]
    info.frame_times_ns = np.array([10, 20, 30, 40]) * 10**9
    t, src = insitu.frame_times(info, 3)
    assert src == "recorded" and t.tolist() == [10.0, 20.0, 30.0]
    info.frame_times_ns = np.array([10, 20]) * 10**9  # too few: fall back
    assert insitu.frame_times(info, 3)[1] == "index"


# ---------- the signal ----------


def test_spectrum_image_series(tmp_path, fake_dm4):
    dims, n = [5, 3, 4], 5
    times = [18_000_000_000 + k * 3_400_000_000 for k in range(n)]
    times[2] += 7 * 10**6  # not quite uniform
    path, frames = make_series(
        tmp_path, fake_dm4, "SI", dims, 2, "<f4", n, chunk_sizes=[700, 800], times_ns=times, duration=times[-1] / 1e9,
        times_as_dict=True,
    )
    assert insitu.is_insitu(path)
    (signal,) = insitu.load_insitu(path)

    assert type(signal).__name__ == "LazyEELSSpectrum"
    assert signal._lazy and signal.data.shape == (5, 3, 5, 4) and signal.data.dtype == np.float32
    assert signal.data.chunks == ((1,) * 5, (3,), (5,), (4,))  # one chunk per frame
    am = signal.axes_manager
    assert [a.name for a in am._axes] == ["Time", "y", "x", "Energy loss"]
    assert [a.navigate for a in am._axes] == [True, True, True, False]
    assert am.navigation_shape == (5, 3, 5) and am.signal_shape == (4,)
    t = am._axes[0]
    assert t.units == "s" and t.offset == pytest.approx(18.0)
    assert t.scale == pytest.approx((times[-1] - times[0]) / 4 / 1e9)
    assert signal.metadata.Signal.signal_type == "EELS"

    md = signal.metadata.Insitu.as_dictionary()
    assert md["frames"] == 5 and md["recorded_frames"] == 5 and md["time_source"] == "recorded"
    assert md["frame_times_s"] == pytest.approx([t_ / 1e9 for t_ in times])
    assert md["raw_files"] == ["SI.raw", "SI.raw_1"] and "note" not in md
    assert "In-situ" in signal.original_metadata.ImageList.TagGroup0.ImageTags.as_dictionary()

    for k in range(n):  # frame 2 straddles the two chunk files
        np.testing.assert_array_equal(signal.data[k].compute(), frames[k].transpose(1, 2, 0))
    assert insitu.insitu_summary(signal)["frame_shape"] == [3, 5, 4]


def test_image_series_uint8_with_duration_times(tmp_path, fake_dm4):
    path, frames = make_series(tmp_path, fake_dm4, "ADF", [6, 4], 6, "u1", 3, duration=9.0)
    (signal,) = insitu.load_insitu(path)
    assert type(signal).__name__ == "LazySignal2D" and signal.data.dtype == np.uint8
    am = signal.axes_manager
    assert am.navigation_shape == (3,) and am.signal_shape == (6, 4)  # x=6, y=4
    assert [a.name for a in am._axes] == ["Time", "y", "x"]
    assert signal.metadata.Insitu.time_source == "duration"
    assert signal.metadata.Insitu.frame_times_s == pytest.approx([3.0, 6.0, 9.0])
    assert am._axes[0].scale == pytest.approx(3.0) and am._axes[0].offset == pytest.approx(3.0)
    for k in range(3):
        np.testing.assert_array_equal(signal.data[k].compute(), frames[k])


def test_incomplete_files_use_complete_frames_with_a_note(tmp_path, fake_dm4):
    # 6 frames recorded; the files hold 3.5 (a run cut short or a copy in progress).
    path, frames = make_series(tmp_path, fake_dm4, "SI", [5, 3, 4], 2, "<f4", 3, recorded=6, bytes_extra=120)
    (signal,) = insitu.load_insitu(path)
    assert signal.data.shape[0] == 3
    assert signal.metadata.Insitu.frames == 3 and signal.metadata.Insitu.recorded_frames == 6
    assert signal.metadata.Insitu.note == "3 of 6 frames are present in the .raw files"
    assert insitu.insitu_summary(signal)["note"] == signal.metadata.Insitu.note
    np.testing.assert_array_equal(signal.data[2].compute(), frames[2].transpose(1, 2, 0))


def test_not_insitu(tmp_path, fake_dm4):
    # A .dm4 without .raw beside it, and one without the tags.
    (tmp_path / "plain.dm4").write_bytes(b"")
    assert not insitu.is_insitu(tmp_path / "plain.dm4")
    assert insitu.load_insitu(tmp_path / "plain.dm4") is None
    (tmp_path / "tagless.dm4").write_bytes(b"")
    (tmp_path / "tagless.raw").write_bytes(b"\0" * 10)
    fake_dm4["tagless.dm4"] = hs.signals.Signal2D(np.zeros((3, 3))).as_lazy()
    assert not insitu.is_insitu(tmp_path / "tagless.dm4")
    assert not insitu.is_insitu(tmp_path / "other.hspy")


# ---------- opening: session and API ----------


def count_reads(monkeypatch):
    calls = []
    real = RawFrames.read_bytes

    def counting(self, k):
        calls.append(k)
        return real(self, k)

    monkeypatch.setattr(RawFrames, "read_bytes", counting)
    return calls


def two_series_folder(tmp_path, fake_dm4):
    n = 4
    times = [10**9 * (k + 1) for k in range(n)]
    si, _ = make_series(tmp_path, fake_dm4, "STEM SI_EELS Spectrum Image", [5, 3, 4], 2, "<f4", n, times_ns=times)
    # The ADF is a sub-scan: 4 × finer than the 5 × 3 scan (0.5 × 1.5 nm pixels, from the tags)
    # over the same field of view.
    adf, adf_frames = make_series(
        tmp_path, fake_dm4, "STEM SI_ADF Image", [20, 12], 2, "<f4", n, times_ns=times, scale=(0.125, 0.375)
    )
    other, _ = make_series(tmp_path, fake_dm4, "Other Image", [20, 12], 2, "<f4", n + 1)  # different frame count
    (tmp_path / "ADF Image (SI Survey).dm4").write_bytes(b"")  # an ordinary image: no .raw
    (tmp_path / "STEM SI_EELS Spectrum Image_summed.dm4").write_bytes(b"")  # user-made: no .raw
    return si, adf, other, adf_frames


def test_opening_the_si_brings_the_matching_adf_stack(tmp_path, fake_dm4, monkeypatch):
    si_path, adf_path, other_path, adf_frames = two_series_folder(tmp_path, fake_dm4)
    reads = count_reads(monkeypatch)
    session = Session()
    entry = session.open_file(si_path)
    assert reads == []  # opening reads no frame
    datasets = [session.datasets[i] for i in entry.dataset_ids]
    # The series is one dataset: an ordinary 2D spectrum image holding a frame.
    assert [d.kind for d in datasets] == [DatasetKind.SPECTRUM_IMAGE, DatasetKind.STACK]
    si, adf = datasets
    assert [d.title for d in datasets] == ["STEM SI_EELS Spectrum Image", "STEM SI_ADF Image"]
    assert si.lazy and adf.lazy and not si.is_line_scan and si.insitu is not None

    s = si.summary()
    assert s["kind"] == "spectrum_image" and s["navigation_shape"] == [5, 3] and s["signal_shape"] == [4]
    assert s["insitu"]["frames"] == 4 and s["insitu"]["frame_shape"] == [3, 5, 4]
    assert s["insitu"]["time_source"] == "recorded" and s["insitu"]["note"] is None
    assert s["insitu"]["view"] == {"mode": "frame", "start": 0, "end": 0}
    assert s["eels"]["role"] == "core_loss" and s["signal_type"] == "EELS"
    assert adf.summary()["insitu"]["frame_shape"] == [12, 20]
    assert reads == []

    # Loading reads one frame (the first), never the series.
    si.materialise()
    assert not si.lazy and reads[0] == 0
    wait_ahead(si)
    assert sorted(reads) == [0, 1]  # one frame, and its neighbour read ahead
    assert session.thumbnail(si, 64)[:4] == b"\x89PNG"
    assert sorted(reads) == [0, 1]

    # The ADF stack works as an ordinary stack.
    from si_viewer.processing.frames import get_frame

    np.testing.assert_array_equal(get_frame(adf, index=2), adf_frames[2])
    assert not adf.lazy  # small: loaded for scrubbing


def test_opening_the_adf_alone(tmp_path, fake_dm4):
    _, adf_path, _, _ = two_series_folder(tmp_path, fake_dm4)
    session = Session()
    entry = session.open_file(adf_path)
    assert [session.datasets[i].kind for i in entry.dataset_ids] == [DatasetKind.STACK]


def wait_ahead(ds):
    """Until the read-ahead tasks queued so far have run."""
    ds.insitu._ahead.submit(lambda: None).result(timeout=10)


def test_api_for_an_insitu_si(tmp_path, fake_dm4):
    from fastapi.testclient import TestClient

    from si_viewer.app import create_app

    si_path, *_ = two_series_folder(tmp_path, fake_dm4)
    client = TestClient(create_app())
    r = client.post("/api/files", json={"path": str(si_path)})
    assert r.status_code == 200
    si = next(d for d in r.json()["datasets"] if d["insitu"] and d["kind"] == "spectrum_image")
    assert si["lazy"] and si["data_rev"] == 0
    loaded = client.post(f"/api/datasets/{si['id']}/load")
    assert loaded.status_code == 200 and not loaded.json()["lazy"]
    assert client.get(f"/api/datasets/{si['id']}/frame").status_code == 200  # total counts of frame 1
    assert client.get(f"/api/datasets/{si['id']}/spectrum").status_code == 200
    assert client.get(f"/api/datasets/{si['id']}/thumbnail").status_code == 200
    meta = client.get(f"/api/datasets/{si['id']}/metadata").json()
    assert "Insitu" not in meta["metadata"]  # it is a frame, an ordinary spectrum image
    assert "In-situ" in meta["original_metadata"]["ImageList"]["TagGroup0"]["ImageTags"]


# ---------- frame browsing (M12b) ----------


def open_pair(tmp_path, fake_dm4, monkeypatch=None):
    """(session, SI dataset, ADF dataset, SI frames as (y, x, E), ADF frames) of the synthetic folder."""
    si_path, _, _, adf_frames = two_series_folder(tmp_path, fake_dm4)
    session = Session()
    entry = session.open_file(si_path)
    si, adf = (session.datasets[i] for i in entry.dataset_ids)
    frames = [frame_array(k, [5, 3, 4], "<f4").transpose(1, 2, 0) for k in range(4)]
    return session, si, adf, frames, adf_frames


def test_setting_the_frame_replaces_the_data_and_keeps_the_dataset(tmp_path, fake_dm4):
    session, si, _, frames, _ = open_pair(tmp_path, fake_dm4)
    entry = session.file_of(si)
    ids = list(entry.dataset_ids)
    si.materialise()
    np.testing.assert_array_equal(si.signal.data, frames[0])
    rev, data_rev = si.revision, si.data_rev
    for k in (2, 1, 3, 0):
        session.show_insitu_frame(si, k)
        np.testing.assert_array_equal(si.signal.data, frames[k])
        assert si.insitu.view.as_dict() == {"mode": "frame", "start": k, "end": k}
        assert si.summary()["insitu"]["view"]["start"] == k
    assert si.revision == rev and si.data_rev == data_rev + 4  # contrast etc. are kept, clients refetch
    assert si.kind is DatasetKind.SPECTRUM_IMAGE and not si.lazy and type(si.signal).__name__ == "EELSSpectrum"
    # No second row for the view: the file lists the series and the ADF, nothing else.
    assert entry.dataset_ids == ids and len(entry.summary(session)["datasets"]) == 2
    with pytest.raises(IndexError):
        session.show_insitu_frame(si, 4)
    with pytest.raises(IndexError):
        session.show_insitu_frame(si, -1)
    # The data is the cached frame itself, so it is read-only (a tool can't change it in place).
    assert not si.signal.data.flags.writeable


def test_showing_a_frame_first_works_without_loading(tmp_path, fake_dm4):
    session, si, _, frames, _ = open_pair(tmp_path, fake_dm4)
    assert si.lazy
    session.show_insitu_frame(si, 3)
    assert not si.lazy
    np.testing.assert_array_equal(si.signal.data, frames[3])


def test_axis_calibration_survives_a_frame_change(tmp_path, fake_dm4):
    session, si, _, _, _ = open_pair(tmp_path, fake_dm4)
    si.materialise()
    si.set_axis_calibration(1, scale=0.7, units="Å")  # x
    session.show_insitu_frame(si, 2)
    x = [a for a in si.summary()["axes"] if a["name"] == "x"][0]
    assert x["scale"] == 0.7 and x["units"] == "Å"


def test_mean_and_sum_of_a_range_match_numpy(tmp_path, fake_dm4):
    session, si, _, frames, _ = open_pair(tmp_path, fake_dm4)
    stack = np.stack(frames).astype(np.float64)
    progress = []
    mean = si.insitu.reduce("mean", 1, 3, lambda f, m=None: progress.append((round(f, 3), m)))
    assert progress == [(0.0, "Frame 1 of 3"), (0.333, "Frame 2 of 3"), (0.667, "Frame 3 of 3")]
    assert mean.dtype == np.float32
    np.testing.assert_allclose(mean, stack[1:4].mean(axis=0), rtol=1e-6)
    total = si.insitu.reduce("sum", 0, 3)
    np.testing.assert_allclose(total, stack.sum(axis=0), rtol=1e-6)

    session.apply_insitu_range(si, total, "sum", 0, 3)
    np.testing.assert_allclose(si.signal.data, stack.sum(axis=0), rtol=1e-6)
    assert si.insitu.view.label() == "sum of frames 1–4"
    session.apply_insitu_range(si, frames[2], "mean", 2, 2)  # a one-frame range is that frame
    assert si.insitu.view.as_dict() == {"mode": "frame", "start": 2, "end": 2}
    with pytest.raises(IndexError):
        si.insitu.reduce("mean", 2, 4)
    with pytest.raises(ValueError):
        si.insitu.reduce("median", 0, 1)


def test_reducing_a_range_reads_it_but_does_not_flood_the_cache(tmp_path, fake_dm4, monkeypatch):
    session, si, _, _, _ = open_pair(tmp_path, fake_dm4)
    reads = count_reads(monkeypatch)
    si.insitu.cache.clear()
    si.insitu.reduce("sum", 0, 3)
    assert sorted(reads) == [0, 1, 2, 3] and si.insitu.cache.keys() == []
    reads.clear()
    session.show_insitu_frame(si, 1)
    wait_ahead(si)
    si.insitu.reduce("sum", 0, 2)  # frame 1 is cached (and 0 and 2 by the read-ahead): not read again
    assert 1 not in reads[1:]


def test_frame_cache_is_bounded_by_bytes_and_evicts_the_least_recently_used():
    from si_viewer.insitu_series import FrameCache

    frame = lambda: np.zeros(100, dtype=np.float32)  # 400 bytes
    cache = FrameCache(max_bytes=1000)  # two frames fit
    cache.put(0, frame())
    cache.put(1, frame())
    assert cache.keys() == [0, 1] and cache.nbytes == 800
    assert cache.get(0) is not None  # 0 is now the most recently used
    cache.put(2, frame())
    assert cache.keys() == [0, 2] and 1 not in cache and cache.nbytes == 800
    big = FrameCache(max_bytes=10)  # the newest frame always stays
    big.put(5, frame())
    big.put(6, frame())
    assert big.keys() == [6]


def test_cache_serves_repeat_visits_and_reads_ahead(tmp_path, fake_dm4, monkeypatch):
    session, si, _, frames, _ = open_pair(tmp_path, fake_dm4)
    reads = count_reads(monkeypatch)
    session.show_insitu_frame(si, 1)
    wait_ahead(si)
    assert sorted(reads) == [0, 1, 2]  # the neighbours were read ahead
    reads.clear()
    session.show_insitu_frame(si, 2)
    np.testing.assert_array_equal(si.signal.data, frames[2])
    wait_ahead(si)
    assert reads == [3]  # 1 and 2 came from the cache; only the new neighbour is read
    reads.clear()
    for k in (3, 2, 1, 0):
        session.show_insitu_frame(si, k)
    wait_ahead(si)
    assert reads == []  # everything was cached


def test_cache_is_bounded_in_a_series(tmp_path, fake_dm4):
    session, si, _, _, _ = open_pair(tmp_path, fake_dm4)
    frame_bytes = 3 * 5 * 4 * 4
    si.insitu.cache.max_bytes = 2 * frame_bytes
    for k in (0, 1, 2, 3):
        session.show_insitu_frame(si, k)
        wait_ahead(si)
    assert si.insitu.cache.nbytes <= 2 * frame_bytes
    assert len(si.insitu.cache.keys()) == 2


def test_stale_read_ahead_is_dropped_when_the_user_jumps(tmp_path, fake_dm4, monkeypatch):
    import threading

    n = 8
    times = [10**9 * (k + 1) for k in range(n)]
    path, frames = make_series(tmp_path, fake_dm4, "S", [5, 3, 4], 2, "<f4", n, times_ns=times)
    session = Session()
    (si,) = (session.datasets[i] for i in session.open_file(path).dataset_ids)
    gate, started = threading.Event(), threading.Event()
    reads = []
    real = RawFrames.read_bytes

    def slow(self, k):
        reads.append(k)
        if k == 1:  # the first read-ahead blocks the (single) read-ahead thread
            started.set()
            assert gate.wait(10)
        return real(self, k)

    monkeypatch.setattr(RawFrames, "read_bytes", slow)
    session.show_insitu_frame(si, 0)  # reads 0; read-ahead of 1 blocks
    assert started.wait(10)
    session.show_insitu_frame(si, 5)  # queues 6 and 4 behind it ...
    session.show_insitu_frame(si, 7)  # ... and the user moves on: only 6 (7's neighbour) is still wanted
    gate.set()
    wait_ahead(si)
    assert sorted(reads) == [0, 1, 5, 6, 7]  # 4 (stale) was never read
    assert 4 not in si.insitu.cache


def test_filters_and_window_maps_persist_across_frames(tmp_path, fake_dm4):
    from si_viewer.processing.filters import Filters, filter_image
    from si_viewer.processing.regions import processed_energy_map

    session, si, _, frames, _ = open_pair(tmp_path, fake_dm4)
    si.materialise()
    session.set_filters(si, {"spatial": "gaussian", "sigma": 0.8})
    wm = session.add_window_map(si, 3.0, 8.0)  # the energy axis starts at 3 eV, 2.5 eV per channel: channels 0–1
    filters, filter_rev, wm_rev = dict(si.filters), si.filter_rev, wm.revision
    before = np.array(wm.signal.data)
    session.show_insitu_frame(si, 2)
    assert si.filters == filters and si.filter_rev == filter_rev  # processing stays
    assert si.summary()["filters"]["spatial"] == "gaussian"
    # The window map (same dataset, same revision: contrast is kept) now maps the new frame.
    assert session.window_maps(si) == [wm] and wm.revision == wm_rev and wm.data_rev == 1
    expected = processed_energy_map(frames[2], 3.0, 2.5, 3.0, 8.0, Filters(**filters))
    np.testing.assert_allclose(wm.signal.data, expected, rtol=1e-5)
    assert not np.allclose(wm.signal.data, before)
    # The total-counts map comes from the new frame, filtered.
    from si_viewer.processing.frames import get_frame

    np.testing.assert_allclose(
        get_frame(si), filter_image(frames[2].sum(axis=-1).astype(np.float32), Filters(**filters)), rtol=1e-5
    )
    assert len(session.file_of(si).dataset_ids) == 3  # series, ADF, window map


def test_spectra_follow_the_frame(tmp_path, fake_dm4):
    from si_viewer.processing.frames import get_spectrum

    session, si, _, frames, _ = open_pair(tmp_path, fake_dm4)
    si.materialise()
    first = get_spectrum(si)
    session.show_insitu_frame(si, 3)
    np.testing.assert_allclose(get_spectrum(si), frames[3].sum(axis=(0, 1)), rtol=1e-6)
    assert not np.allclose(get_spectrum(si), first)


def test_adf_navigator_keeps_the_sub_scan_resolution(tmp_path, fake_dm4):
    from si_viewer.insitu_series import ADF_SOURCE_ID, adf_image, adf_stack
    from si_viewer.processing.frames import navigator_sources

    session, si, adf, _, adf_frames = open_pair(tmp_path, fake_dm4)
    siblings = session.siblings(si)
    assert adf_stack(si, siblings) == (adf, 4)
    sources = navigator_sources(si, siblings)
    assert sources[0] == {"id": ADF_SOURCE_ID, "label": "STEM SI_ADF Image", "factor": 4, "dataset_id": adf.id}
    assert {"id": si.id, "label": "Total counts", "factor": 1, "dataset_id": si.id} in sources
    # The navigator's calibration is the ADF's own (4 × finer than the scan's).
    ax = {a["name"]: a for a in adf.summary()["axes"] if not a["navigate"]}
    assert ax["x"]["scale"] == pytest.approx(0.125) and ax["x"]["size"] == 20 and ax["y"]["size"] == 12
    # Frame k: the ADF frame of the same index, as it is (20 × 12, not binned onto the 5 × 3 scan).
    si.materialise()
    session.show_insitu_frame(si, 2)
    image = adf_image(si, siblings)
    assert image.shape == (12, 20) and image.dtype == np.float32
    np.testing.assert_array_equal(image, adf_frames[2])
    # A range: the mean or sum of the ADF frames over the same range.
    stack = np.stack(adf_frames).astype(np.float64)
    session.apply_insitu_range(si, si.insitu.reduce("mean", 1, 3), "mean", 1, 3)
    np.testing.assert_allclose(adf_image(si, siblings), stack[1:4].mean(axis=0), rtol=1e-5)
    session.apply_insitu_range(si, si.insitu.reduce("sum", 1, 3), "sum", 1, 3)
    np.testing.assert_allclose(adf_image(si, siblings), stack[1:4].sum(axis=0), rtol=1e-5)


def test_adf_with_another_field_of_view_is_not_offered(tmp_path, fake_dm4):
    from si_viewer.insitu_series import adf_image, adf_stack
    from si_viewer.processing.frames import navigator_sources

    session, si, adf, _, _ = open_pair(tmp_path, fake_dm4)
    siblings = session.siblings(si)
    for ax in adf.loaded_signal.axes_manager.signal_axes:
        ax.scale = 0.25  # 20 × 0.25 nm against the scan's 5 × 0.25 nm: not the same field of view
    assert adf_stack(si, siblings) is None and adf_image(si, siblings) is None
    assert [s["label"] for s in navigator_sources(si, siblings)] == ["Total counts"]
    for ax in adf.loaded_signal.axes_manager.signal_axes:
        ax.units = ""  # uncalibrated: the field of view cannot be compared, the multiple counts
    assert adf_stack(si, siblings) == (adf, 4)


def test_no_adf_without_a_matching_stack(tmp_path, fake_dm4):
    from si_viewer.insitu_series import adf_image
    from si_viewer.processing.frames import navigator_sources

    path, _ = make_series(tmp_path, fake_dm4, "Alone", [5, 3, 4], 2, "<f4", 3)
    session = Session()
    (si,) = (session.datasets[i] for i in session.open_file(path).dataset_ids)
    assert [s["label"] for s in navigator_sources(si, session.siblings(si))] == ["Total counts"]
    assert adf_image(si, session.siblings(si)) is None


def test_frame_api(tmp_path, fake_dm4):
    import time

    from fastapi.testclient import TestClient

    from si_viewer.app import create_app

    si_path, _, _, adf_frames = two_series_folder(tmp_path, fake_dm4)
    client = TestClient(create_app())
    file = client.post("/api/files", json={"path": str(si_path)}).json()
    si = next(d for d in file["datasets"] if d["insitu"] and d["kind"] == "spectrum_image")
    url = f"/api/datasets/{si['id']}"

    r = client.put(f"{url}/insitu/frame", json={"frame": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["dataset"]["insitu"]["view"] == {"mode": "frame", "start": 2, "end": 2}
    assert body["dataset"]["data_rev"] == 1 and len(body["file"]["datasets"]) == 2
    assert client.put(f"{url}/insitu/frame", json={"frame": 4}).status_code == 416
    assert client.put(f"/api/datasets/{file['datasets'][1]['id']}/insitu/frame", json={"frame": 0}).status_code == 400

    # The ADF navigator source and its image for frame 3.
    sources = client.get(f"{url}/navigators").json()
    adf_ds = next(d for d in file["datasets"] if d["kind"] == "stack")
    assert sources[0] == {"id": "insitu-adf", "label": adf_ds["title"], "factor": 4, "dataset_id": adf_ds["id"]}
    r = client.get(f"{url}/insitu/adf")
    # The ADF at its own resolution: 20 × 12, four times the 5 × 3 scan.
    assert r.status_code == 200 and r.headers["X-Array-Shape"] == "12,20"
    assert json.loads(r.headers["X-Array-Meta"])["factor"] == 4
    got = np.frombuffer(r.content, dtype="<f4").reshape(12, 20)
    np.testing.assert_array_equal(got, adf_frames[2])

    # Mean of a range: a job with progress that ends with the view holding the mean.
    r = client.post(f"{url}/insitu/reduce", json={"mode": "mean", "start": 0, "end": 3})
    assert r.status_code == 200
    job = r.json()
    for _ in range(100):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] not in ("queued", "running"):
            break
        time.sleep(0.05)
    assert job["status"] == "done", job
    assert job["result"]["dataset_id"] == si["id"]
    view = next(d for d in job["result"]["file"]["datasets"] if d["id"] == si["id"])["insitu"]["view"]
    assert view == {"mode": "mean", "start": 0, "end": 3}
    assert client.post(f"{url}/insitu/reduce", json={"mode": "mean", "start": 2, "end": 9}).status_code == 422
    assert client.post(f"{url}/insitu/reduce", json={"mode": "median", "start": 0, "end": 1}).status_code == 422


def test_reduce_job_can_be_cancelled(tmp_path, fake_dm4, monkeypatch):
    import threading
    import time

    from fastapi.testclient import TestClient

    from si_viewer.app import create_app

    si_path, *_ = two_series_folder(tmp_path, fake_dm4)
    client = TestClient(create_app())
    file = client.post("/api/files", json={"path": str(si_path)}).json()
    si = next(d for d in file["datasets"] if d["insitu"] and d["kind"] == "spectrum_image")
    started, gate = threading.Event(), threading.Event()
    real = RawFrames.read_bytes

    def slow(self, k):
        if k == 1:
            started.set()
            assert gate.wait(10)
        return real(self, k)

    monkeypatch.setattr(RawFrames, "read_bytes", slow)
    job = client.post(f"/api/datasets/{si['id']}/insitu/reduce", json={"mode": "sum", "start": 0, "end": 3}).json()
    assert started.wait(10)
    client.post(f"/api/jobs/{job['id']}/cancel")
    gate.set()
    for _ in range(100):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] not in ("queued", "running"):
            break
        time.sleep(0.05)
    assert job["status"] == "cancelled"
    assert client.get(f"/api/datasets/{si['id']}").json()["insitu"]["view"]["mode"] == "frame"  # unchanged


def test_zlp_alignment_is_refused_for_a_frame(tmp_path, fake_dm4):
    from fastapi.testclient import TestClient

    from si_viewer.app import create_app

    session, si, _, _, _ = open_pair(tmp_path, fake_dm4)
    client = TestClient(create_app(session))
    session.set_eels_role(si, "low_loss")
    r = client.post(f"/api/datasets/{si.id}/zlp-alignment", json={"options": {}})
    assert r.status_code == 422 and "in-situ" in r.json()["detail"]


def test_background_removal_is_titled_with_the_frame(tmp_path, fake_dm4):
    from si_viewer.processing.background import BackgroundOptions

    n = 4
    path, _ = make_series(tmp_path, fake_dm4, "Long", [5, 3, 40], 2, "<f4", n, times_ns=[10**9 * (k + 1) for k in range(n)])
    session = Session()
    (si,) = (session.datasets[i] for i in session.open_file(path).dataset_ids)
    options = BackgroundOptions(background=(10.0, 30.0), signal=(40.0, 80.0), model="offset")
    session.show_insitu_frame(si, 2)
    bg = session.add_background(si, options, session.compute_background(si, options))
    assert bg.title.endswith("bg removed (frame 3)") and bg.source_frame == "frame 3"
    assert bg.parent_id == si.id and bg.id in session.file_of(si).dataset_ids  # attached to the file's entry
    # Moving on leaves it as made; redoing it takes the frame shown now and says so.
    session.show_insitu_frame(si, 0)
    assert bg.title.endswith("(frame 3)")
    session.rerun_background(bg)
    assert bg.title.endswith("bg removed (frame 1)") and bg.source_frame == "frame 1"
    session.apply_insitu_range(si, si.insitu.reduce("mean", 0, 2), "mean", 0, 2)
    session.rerun_background(bg)
    assert bg.title.endswith("bg removed (mean of frames 1–3)")
    # A title the user gave is kept.
    bg.signal.metadata.General.title = "mine"
    session.rerun_background(bg)
    assert bg.title == "mine"


def test_save_names_the_frame(tmp_path, fake_dm4):
    from si_viewer.processing.export import export_bytes

    session, si, _, _, _ = open_pair(tmp_path, fake_dm4)
    session.show_insitu_frame(si, 1)
    data, name = export_bytes(si, "hspy")
    assert name == "STEM_SI_EELS_Spectrum_Image_frame_2.hspy" and data[:4] == b"\x89HDF"


def test_closing_the_file_stops_the_series(tmp_path, fake_dm4):
    session, si, _, _, _ = open_pair(tmp_path, fake_dm4)
    series = si.insitu
    session.close_file(si.file_id)
    assert series.closed and si.id not in session.datasets
    series.schedule_ahead(1)  # harmless after close


def test_info_summary_of_the_series(tmp_path, fake_dm4):
    session, si, _, _, _ = open_pair(tmp_path, fake_dm4)
    info = si.summary()["insitu"]
    assert info["frames"] == 4 and info["frame_times_s"] == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert info["frame_step_s"] == pytest.approx(1.0) and info["first_time_s"] == 1.0 and info["last_time_s"] == 4.0
    assert info["frame_bytes"] == 3 * 5 * 4 * 4 and info["raw_bytes"] == 4 * info["frame_bytes"]
    assert info["raw_files"] == ["STEM SI_EELS Spectrum Image.raw"] and info["start_time"].startswith("2026")


# ---------- the real example (skipped unless fully copied) ----------

REAL = Path(__file__).resolve().parents[2] / "examples" / "InSitu (3)"
SI_NAME, ADF_NAME = "STEM SI_EELS Spectrum Image", "STEM SI_ADF Image"


def _real_complete() -> bool:
    try:
        for name, frame_bytes in ((SI_NAME, 138_240_000), (ADF_NAME, 640_000)):
            dm4 = REAL / f"{name}.dm4"
            paths = insitu.raw_paths(dm4)
            if not paths or sum(p.stat().st_size for p in paths) < 208 * frame_bytes:
                return False
        return True
    except OSError:
        return False


needs_real = pytest.mark.skipif(not _real_complete(), reason="examples/InSitu (3) is missing or not fully copied")


@pytest.fixture(scope="module")
def real_session():
    session = Session()
    calls = []
    real = RawFrames.read_bytes

    def counting(self, k):
        calls.append(k)
        return real(self, k)

    RawFrames.read_bytes = counting
    try:
        entry = session.open_file(REAL / f"{SI_NAME}.dm4")
        opened_reads = list(calls)
    finally:
        RawFrames.read_bytes = real
    return session, entry, opened_reads


@needs_real
def test_real_opening_reads_nothing_and_gives_si_plus_adf(real_session):
    session, entry, opened_reads = real_session
    assert opened_reads == []
    kinds = [session.datasets[i].kind for i in entry.dataset_ids]
    assert kinds == [DatasetKind.SPECTRUM_IMAGE, DatasetKind.STACK]
    assert session.datasets[entry.dataset_ids[0]].insitu is not None


@needs_real
def test_real_shapes_axes_and_times(real_session):
    session, entry, _ = real_session
    si, adf = (session.datasets[i] for i in entry.dataset_ids)
    series = si.insitu.signal
    assert series.data.shape == (208, 100, 100, 3456) and series.data.dtype == np.float32
    assert si.signal.data.shape == (100, 100, 3456)  # the view: one frame
    assert adf.signal.data.shape == (208, 400, 400)
    axes = {a["name"]: a for a in si.summary()["axes"]}
    assert axes["Energy loss"]["units"] == "eV" and axes["Energy loss"]["scale"] == pytest.approx(0.35, rel=1e-5)
    assert axes["Energy loss"]["offset"] == pytest.approx(280.0, rel=1e-5)
    assert axes["x"]["units"] == "nm" and axes["x"]["scale"] == pytest.approx(0.1494662, rel=1e-5)
    time_axis = next(a for a in series.axes_manager._axes if a.name == "Time")
    assert time_axis.units == "s" and time_axis.size == 208
    times = si.insitu.times
    assert len(times) == 208 and (np.diff(times) > 0).all()
    assert times[0] == pytest.approx(18.44, abs=0.01) and times[-1] == pytest.approx(726.27, abs=0.01)
    assert si.summary()["insitu"]["note"] is None
    # The ADF is 4× finer sampling of the same field.
    a, b = adf.summary()["axes"], si.summary()["axes"]
    assert a[1]["size"] == 4 * next(x for x in b if x["name"] == "y")["size"]
    assert adf.signal.metadata.Insitu.frame_times_s == pytest.approx(list(times))


@needs_real
def test_real_frames_match_the_simultaneous_adf(real_session):
    session, entry, _ = real_session
    si, adf = (session.datasets[i] for i in entry.dataset_ids)
    for k in (0, 1):
        counts = si.insitu.signal.data[k].compute().sum(axis=-1)
        image = adf.signal.data[k].compute()
        binned = image.reshape(100, 4, 100, 4).mean(axis=(1, 3))
        assert np.corrcoef(counts.ravel(), binned.ravel())[0, 1] > 0.8


@needs_real
def test_real_frame_straddling_two_chunks(real_session):
    session, entry, _ = real_session
    si = session.datasets[entry.dataset_ids[0]]
    fb = 138_240_000
    paths = insitu.raw_paths(REAL / f"{SI_NAME}.dm4")
    first = paths[0].stat().st_size
    k = first // fb  # the first frame that doesn't fit in chunk 0
    assert k * fb < first < (k + 1) * fb
    head = np.fromfile(paths[0], dtype=np.uint8, count=first - k * fb, offset=k * fb)
    tail = np.fromfile(paths[1], dtype=np.uint8, count=(k + 1) * fb - first)
    expected = np.concatenate([head, tail]).view("<f4").reshape(3456, 100, 100).transpose(1, 2, 0)
    np.testing.assert_array_equal(si.insitu.signal.data[k].compute(), expected)


@needs_real
def test_real_frame_browsing(real_session):
    """Frames are shown one at a time (a few reads), as an ordinary spectrum image."""
    from si_viewer.insitu_series import adf_image
    from si_viewer.processing.frames import get_frame, navigator_sources

    session, entry, _ = real_session
    si, adf = (session.datasets[i] for i in entry.dataset_ids)
    assert len(session.file_of(si).summary(session)["datasets"]) == 2  # no second row for the view
    session.show_insitu_frame(si, 3)
    assert si.signal.data.shape == (100, 100, 3456) and not si.lazy
    np.testing.assert_array_equal(si.signal.data, si.insitu.signal.data[3].compute())
    assert si.summary()["insitu"]["view"] == {"mode": "frame", "start": 3, "end": 3}
    assert si.insitu.times[3] == pytest.approx(si.summary()["insitu"]["frame_times_s"][3])
    # The total-counts navigator and the ADF of the same frame (4 × finer than the 100 × 100 scan, kept at 400 × 400).
    counts = get_frame(si)
    assert counts.shape == (100, 100)
    sources = navigator_sources(si, session.siblings(si))
    assert sources[0]["label"] == ADF_NAME and sources[0]["factor"] == 4
    image = adf_image(si, session.siblings(si))
    assert image.shape == (400, 400)
    np.testing.assert_allclose(image, adf.signal.data[3].compute(), rtol=1e-5)
    from si_viewer.processing.frames import bin2d

    assert np.corrcoef(counts.ravel(), bin2d(image, 4).ravel())[0, 1] > 0.8


@needs_real
def test_real_mean_of_two_frames(real_session):
    session, entry, _ = real_session
    si = session.datasets[entry.dataset_ids[0]]
    mean = si.insitu.reduce("mean", 1, 2)
    expected = (si.insitu.read(1).astype(np.float64) + si.insitu.read(2)) / 2
    np.testing.assert_allclose(mean, expected, rtol=1e-6)
    session.apply_insitu_range(si, mean, "mean", 1, 2)
    assert si.insitu.view.label() == "mean of frames 2–3"


# ---------- rebinning a series ----------


def binnable_pair(tmp_path, fake_dm4, n=6):
    """(session, SI, ADF, SI frames (y, x, E) as float64, ADF frames) of a
    6 × 4 scan with 4 channels and its 4 × finer ADF, ``n`` frames 1 s apart."""
    times = [10**9 * (k + 1) for k in range(n)]
    si_path, _ = make_series(tmp_path, fake_dm4, "STEM SI_EELS Spectrum Image", [6, 4, 4], 2, "<f4", n, times_ns=times)
    _, adf_frames = make_series(
        tmp_path, fake_dm4, "STEM SI_ADF Image", [24, 16], 2, "<f4", n, times_ns=times, scale=(0.125, 0.375)
    )
    session = Session()
    entry = session.open_file(si_path)
    si, adf = (session.datasets[i] for i in entry.dataset_ids)
    frames = [frame_array(k, [6, 4, 4], "<f4").transpose(1, 2, 0).astype(np.float64) for k in range(n)]
    return session, si, adf, frames, adf_frames


def binned(frame, fy, fx, fe):
    y, x, e = (s // f for s, f in zip(frame.shape, (fy, fx, fe)))
    return frame[: y * fy, : x * fx, : e * fe].reshape(y, fy, x, fx, e, fe).sum(axis=(1, 3, 5))


def test_rebinning_a_series_bins_every_frame_and_the_time_axis(tmp_path, fake_dm4):
    from si_viewer.insitu_series import adf_image, adf_stack

    session, si, adf, frames, adf_frames = binnable_pair(tmp_path, fake_dm4)
    si.materialise()
    session.show_insitu_frame(si, 3)
    scales = [ax.scale for ax in si.signal.axes_manager._axes]
    rev = si.revision
    session.rebin(si, [2, 2, 2], time=2)  # y, x, E and frames in pairs

    series = si.insitu
    assert series.n == 3 and series.frame_shape == (2, 3, 2)
    np.testing.assert_allclose(series.times, [1.5, 3.5, 5.5])
    # It holds the binned frame that frame 4 (index 3) went into: frames 3 and 4 summed.
    assert series.view.as_dict() == {"mode": "frame", "start": 1, "end": 1}
    np.testing.assert_allclose(si.signal.data, binned(frames[2] + frames[3], 2, 2, 2))
    assert [ax.scale for ax in si.signal.axes_manager._axes] == pytest.approx([2 * f for f in scales])
    assert si.revision == rev + 1 and si.rebinned == ["Time×2, x×2, y×2, Energy loss×2"]
    s = si.summary()
    assert s["navigation_shape"] == [3, 2] and s["signal_shape"] == [2]
    assert s["insitu"]["frames"] == 3 and s["insitu"]["time_binning"] == 2 and s["insitu"]["frame_shape"] == [2, 3, 2]

    # Every frame read from now on is binned; a range sums binned frames.
    session.show_insitu_frame(si, 2)
    np.testing.assert_allclose(si.signal.data, binned(frames[4] + frames[5], 2, 2, 2))
    np.testing.assert_allclose(series.reduce("sum", 0, 2), binned(sum(frames), 2, 2, 2), rtol=1e-6)
    with pytest.raises(IndexError):
        series.read(3)

    # The ADF follows: the same field of view, now 8 × finer, its frames in the same pairs.
    assert adf_stack(si, session.siblings(si)) == (adf, 8)
    np.testing.assert_allclose(adf_image(si, session.siblings(si)), adf_frames[4] + adf_frames[5])

    # Binning again bins the binned series.
    session.rebin(si, [1, 1, 1], time=3)
    assert series.n == 1 and series.binning == (6, 2, 2, 2) and si.rebinned[-1] == "Time×3"
    np.testing.assert_allclose(si.signal.data, binned(sum(frames), 2, 2, 2), rtol=1e-6)


def test_rebinning_a_series_keeps_integer_counts(tmp_path, fake_dm4):
    times = [10**9 * (k + 1) for k in range(4)]
    path, raw = make_series(tmp_path, fake_dm4, "SI", [4, 2, 4], 10, "<u2", 4, times_ns=times)
    session = Session()
    si = session.datasets[session.open_file(path).dataset_ids[0]]
    si.materialise()
    session.rebin(si, [1, 2, 1], time=2)
    frames = [f.transpose(1, 2, 0).astype(np.int64) for f in raw]
    assert si.signal.data.dtype == np.uint32
    np.testing.assert_array_equal(si.signal.data, binned(frames[0] + frames[1], 1, 2, 1))


def test_rebin_api_for_a_series(tmp_path, fake_dm4):
    from fastapi.testclient import TestClient

    from si_viewer.app import create_app

    session, si, _, _, _ = binnable_pair(tmp_path, fake_dm4)
    client = TestClient(create_app(session))
    url = f"/api/datasets/{si.id}/rebin"
    assert client.post(url, json={"factors": [1, 1, 1], "time": 7}).status_code == 422  # 6 frames
    assert client.post(url, json={"factors": [1, 1, 1], "time": 1}).status_code == 422  # nothing to do
    r = client.post(url, json={"factors": [1, 1, 1], "time": 2})
    assert r.status_code == 200, r.text
    assert r.json()["dataset"]["insitu"]["frames"] == 3
    # A frame of the binned series, then a range job over binned frames.
    assert client.put(f"/api/datasets/{si.id}/insitu/frame", json={"frame": 2}).status_code == 200
    assert client.put(f"/api/datasets/{si.id}/insitu/frame", json={"frame": 3}).status_code == 416


def test_only_series_have_a_time_axis_to_rebin(eels_cuzn):
    session = Session()
    si = next(d for d in (session.datasets[i] for i in session.open_file(eels_cuzn).dataset_ids)
              if d.kind is DatasetKind.SPECTRUM_IMAGE)
    with pytest.raises(ValueError, match="time axis"):
        session.rebin(si, [1, 1, 1], time=2)


# ---------- time scans ----------


def test_time_scan_at_a_pixel_and_over_a_region(tmp_path, fake_dm4):
    from si_viewer.processing.regions import Region

    session, si, _, frames, _ = binnable_pair(tmp_path, fake_dm4)
    si.materialise()
    stack = np.stack(frames)  # (frames, y, x, E)

    # A 3 × 3 block at (2, 1): its mean spectrum in every frame.
    spectra = session.time_scan_spectra(si, (2, 1, 3), None)
    np.testing.assert_allclose(spectra, stack[:, 0:3, 1:4].mean(axis=(1, 2)), rtol=1e-6)
    # A region: the pixels whose centres are inside.
    region = Region(kind="rect", x0=0, y0=0, x1=2, y1=4)
    np.testing.assert_allclose(
        session.time_scan_spectra(si, None, region), stack[:, :, 0:2].mean(axis=(1, 2)), rtol=1e-6,
    )
    with pytest.raises(ValueError, match="outside"):
        session.time_scan_pick(si, (6, 0, 1), None)
    with pytest.raises(ValueError, match="no pixel"):
        session.time_scan_pick(si, None, Region(kind="rect", x0=10, y0=10, x1=12, y1=12))

    (ts,) = session.add_time_scans([(si, spectra)], "cursor (2, 1)", {"pixel": {"x": 2, "y": 1, "size": 3}})
    assert ts.is_line_scan and ts.origin == "line-scan" and ts.parent_id == si.id
    assert ts.title == "STEM SI_EELS Spectrum Image · cursor (2, 1)"
    t_axis, energy = ts.signal.axes_manager.navigation_axes[0], ts.signal.axes_manager.signal_axes[0]
    assert (t_axis.name, t_axis.units, t_axis.size) == ("Time", "s", 6)
    assert t_axis.offset == pytest.approx(1.0) and t_axis.scale == pytest.approx(1.0)
    assert energy.size == 4 and energy.scale == si.signal.axes_manager.signal_axes[0].scale
    md = ts.signal.metadata.Processing.si_viewer.time_scan
    assert md.frame_times_s == pytest.approx([1, 2, 3, 4, 5, 6]) and md.pixel["size"] == 3

    # A time scan is a line scan: it can be rebinned along time and energy.
    session.rebin(ts, [2, 2])
    np.testing.assert_allclose(ts.signal.data, spectra.reshape(3, 2, 2, 2).sum(axis=(1, 3)), rtol=1e-6)
    assert ts.signal.axes_manager.navigation_axes[0].scale == pytest.approx(2.0)


def test_time_scan_of_a_time_binned_series(tmp_path, fake_dm4):
    session, si, _, frames, _ = binnable_pair(tmp_path, fake_dm4)
    si.materialise()
    session.rebin(si, [1, 1, 1], time=3)
    spectra = session.time_scan_spectra(si, (0, 0, 1), None)
    stack = np.stack(frames)
    np.testing.assert_allclose(spectra, stack[:, 0, 0].reshape(2, 3, 4).sum(axis=1), rtol=1e-6)
    (ts,) = session.add_time_scans([(si, spectra)], "cursor", {})
    assert ts.signal.axes_manager.navigation_axes[0].offset == pytest.approx(2.0)  # mean of 1, 2 and 3 s
    assert ts.signal.axes_manager.navigation_axes[0].scale == pytest.approx(3.0)


def test_time_scan_api(tmp_path, fake_dm4):
    from fastapi.testclient import TestClient

    from si_viewer.app import create_app

    session, si, _, frames, _ = binnable_pair(tmp_path, fake_dm4)
    client = TestClient(create_app(session))
    url = f"/api/datasets/{si.id}/insitu/time-scan"
    assert client.post(url, json={"name": "x"}).status_code == 422  # neither a pixel nor a region
    assert client.post(url, json={"pixel": {"x": 9, "y": 0}}).status_code == 422
    r = client.post(url, json={"region": {"kind": "rect", "x0": 0, "y0": 0, "x1": 3, "y1": 2}, "name": "Region 1"})
    assert r.status_code == 200, r.text
    job = r.json()
    for _ in range(500):
        if job["status"] not in ("queued", "running"):
            break
        time.sleep(0.01)
        job = client.get(f"/api/jobs/{job['id']}").json()
    assert job["status"] == "done", job
    ts = next(d for d in job["result"]["file"]["datasets"] if d["id"] == job["result"]["dataset_id"])
    assert ts["line_scan"] and ts["title"].endswith("· Region 1")
    assert ts["navigation_shape"] == [6] and ts["signal_shape"] == [4]
    np.testing.assert_allclose(
        session.get_dataset(ts["id"]).signal.data, np.stack(frames)[:, 0:2, 0:3].mean(axis=(1, 2)), rtol=1e-6,
    )
    # A dataset that is not a series has none.
    adf = next(d for d in job["result"]["file"]["datasets"] if d["kind"] == "stack")
    assert client.post(f"/api/datasets/{adf['id']}/insitu/time-scan", json={"pixel": {"x": 0, "y": 0}}).status_code == 400


def test_the_simultaneous_adf_keeps_its_frames(tmp_path, fake_dm4):
    session, si, adf, _, adf_frames = binnable_pair(tmp_path, fake_dm4)
    with pytest.raises(ValueError, match="in-situ"):
        session.rebin(adf, [2, 1, 1])
    session.rebin(adf, [1, 2, 2])  # x and y only
    np.testing.assert_allclose(adf.signal.data[3], adf_frames[3].reshape(8, 2, 12, 2).sum(axis=(1, 3)))


# ---------- drift-corrected sum ----------


def _align_adf(adf, drift_scan_px, k=4):
    """Give the ADF stack an applied alignment whose drift is ``drift_scan_px``
    ((x, y) per frame, in scan pixels; None: frame left out), in ADF pixels."""
    frames = [f for f, d in enumerate(drift_scan_px) if d is not None]
    adf.alignment = {
        "frames": frames, "square": [0, 0, 8, 8],
        "drift": {"frames": frames, "x": [drift_scan_px[f][0] * k for f in frames], "y": [drift_scan_px[f][1] * k for f in frames]},
    }


def _expected_sum(frames, drift, use):
    """Each frame rolled back by its drift (scan px), summed, cropped to the area every frame covers."""
    total = sum(np.roll(frames[f], (-drift[f][1], -drift[f][0]), axis=(0, 1)) for f in use)
    rows = [-drift[f][1] for f in use]
    cols = [-drift[f][0] for f in use]
    h, w = frames[0].shape[:2]
    r0, r1 = max(max(rows), 0), min(h + min(rows), h)
    c0, c1 = max(max(cols), 0), min(w + min(cols), w)
    return total[r0:r1, c0:c1], (r0, r1, c0, c1)


def test_the_package_shift_and_ours_agree():
    from rigidregistration.utils import generateShiftedImage

    from si_viewer.processing.insitu_align import shift_frame

    im = np.random.default_rng(0).uniform(1, 2, (12, 17))  # not square
    for r, c in [(2, 0), (0, 3), (-3, 5)]:
        np.testing.assert_allclose(shift_frame(im, r, c), generateShiftedImage(im, r, c), atol=1e-12)
    cube = np.random.default_rng(1).uniform(1, 2, (12, 17, 5))
    moved = shift_frame(cube, 1.5, -2.25)
    for e in range(5):  # every channel moves as the image does
        np.testing.assert_allclose(moved[..., e], shift_frame(cube[..., e], 1.5, -2.25), atol=1e-12)


def test_aligned_sum_from_the_adf(tmp_path, fake_dm4):
    session, si, adf, frames, _ = binnable_pair(tmp_path, fake_dm4)
    drift = [(0, 0), (1, 0), (1, 1), None, (0, -1), (-1, 0)]  # frame 4 was a bad frame
    _align_adf(adf, drift)
    src = session.alignment_source(si)
    assert src["frames"] == 6 and src["adf"]["k"] == 4 and src["adf"]["aligned"] and src["adf"]["frames"] == 5

    use, shifts, how = session.aligned_sum_shifts(si, "adf")
    assert use == [0, 1, 2, 4, 5] and np.isnan(shifts[3]).all() and "÷4" in how
    np.testing.assert_allclose(shifts[2], (-1, -1))  # minus the drift, in scan pixels
    data, crop = session.compute_aligned_sum(si, use, shifts)
    expected, crop_expected = _expected_sum(frames, drift, use)
    assert crop == crop_expected
    np.testing.assert_allclose(data, expected, rtol=1e-5, atol=1e-3)  # a sum, not a mean

    out = session.add_aligned_sum(si, data, crop, use, how)
    assert out.origin == "aligned-sum" and out.parent_id == si.id and out.kind is DatasetKind.SPECTRUM_IMAGE
    assert out.insitu is None and out.title.endswith("(aligned sum, 5 frames)")
    r0, r1, c0, c1 = crop
    ax_x, ax_y = out.signal.axes_manager.navigation_axes
    sx, sy = si.signal.axes_manager.navigation_axes
    assert (ax_x.size, ax_y.size) == (c1 - c0, r1 - r0)
    assert ax_x.offset == pytest.approx(sx.offset + c0 * sx.scale) and ax_y.offset == pytest.approx(sy.offset + r0 * sy.scale)
    assert session.delete_dataset(out) is not None and out.id not in session.datasets


def test_aligned_sum_needs_an_aligned_adf(tmp_path, fake_dm4):
    session, si, adf, _, _ = binnable_pair(tmp_path, fake_dm4)
    with pytest.raises(ValueError, match="Align"):
        session.aligned_sum_shifts(si, "adf")


def test_imported_shifts_are_checked(tmp_path, fake_dm4):
    import io

    session, si, _, frames, _ = binnable_pair(tmp_path, fake_dm4)

    def npy(a):
        buf = io.BytesIO()
        np.save(buf, np.asarray(a))
        return buf.getvalue()

    drift = [(0, 0), (1, 0), (1, 1), (0, 0), (0, -1), (-1, 0)]
    shifts = np.array([(-x, -y) for x, y in drift], float)
    # In scan pixels, as (frames, 2) or (2, frames); NaN leaves a frame out.
    shifts[3] = np.nan
    for a in (shifts, shifts.T):
        use, s, _ = session.aligned_sum_shifts(si, "npy", npy(a), "scan")
        assert use == [0, 1, 2, 4, 5]
    data, _ = session.compute_aligned_sum(si, use, s)
    np.testing.assert_allclose(data, _expected_sum(frames, drift, use)[0], rtol=1e-5, atol=1e-3)
    # In ADF pixels: divided by the sub-scan factor (4).
    _, s4, _ = session.aligned_sum_shifts(si, "npy", npy(shifts * 4), "adf")
    np.testing.assert_allclose(s4[~np.isnan(s4)], shifts[~np.isnan(shifts)])
    for bad, why in [
        (np.zeros((5, 2)), "6 frames"),
        (np.zeros((6, 3)), "Expected"),
        (np.full((6, 2), np.nan), "left out"),
        (np.full((6, 2), 50.0), "as large as the scan"),
        (np.array(["a"] * 12).reshape(6, 2), "numbers"),
    ]:
        with pytest.raises(ValueError, match=why):
            session.aligned_sum_shifts(si, "npy", npy(bad), "scan")
    with pytest.raises(ValueError, match="NumPy"):
        session.aligned_sum_shifts(si, "npy", b"not an npy file", "scan")


def test_aligned_sum_api(tmp_path, fake_dm4):
    import base64
    import io

    from fastapi.testclient import TestClient

    from si_viewer.app import create_app

    session, si, adf, frames, _ = binnable_pair(tmp_path, fake_dm4)
    drift = [(0, 0), (1, 0), (1, 1), (0, 1), (0, -1), (-1, 0)]
    _align_adf(adf, drift)
    client = TestClient(create_app(session))
    assert client.get(f"/api/datasets/{si.id}/insitu/alignment").json()["adf"]["aligned"] is True
    check = client.post(f"/api/datasets/{si.id}/insitu/shifts-check", json={"source": "adf"}).json()
    assert check["frames"] == 6 and check["left_out"] == 0 and check["largest"] == 1
    assert (check["width"], check["height"]) == (4, 2)

    # The ADF's shifts as .npy round-trip into the import.
    r = client.get(f"/api/datasets/{adf.id}/alignment/shifts.npy")
    assert r.status_code == 200
    exported = np.load(io.BytesIO(r.content))
    np.testing.assert_allclose(exported, [(-4 * x, -4 * y) for x, y in drift])
    body = {"source": "npy", "units": "adf", "npy": base64.b64encode(r.content).decode()}
    assert client.post(f"/api/datasets/{si.id}/insitu/shifts-check", json=body).json()["frames"] == 6

    job = client.post(f"/api/datasets/{si.id}/insitu/aligned-sum", json=body).json()
    for _ in range(500):
        if job["status"] not in ("queued", "running"):
            break
        time.sleep(0.01)
        job = client.get(f"/api/jobs/{job['id']}").json()
    assert job["status"] == "done", job
    new = next(d for d in job["result"]["file"]["datasets"] if d["id"] == job["result"]["dataset_id"])
    assert new["origin"] == "aligned-sum" and new["kind"] == "spectrum_image" and new["navigation_shape"] == [4, 2]
    np.testing.assert_allclose(
        session.get_dataset(new["id"]).signal.data, _expected_sum(frames, drift, range(6))[0], rtol=1e-5, atol=1e-3,
    )
    bad = {"source": "npy", "units": "scan", "npy": base64.b64encode(b"nope").decode()}
    assert client.post(f"/api/datasets/{si.id}/insitu/aligned-sum", json=bad).status_code == 422
