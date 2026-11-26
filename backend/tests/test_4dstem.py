"""4D-STEM data: py4DSTEM (emdfile) files, the 4D-STEM kind, its frames and binning."""

import h5py
import numpy as np
import pytest

from si_viewer.io import load_file, name_4d_axes
from si_viewer.io import py4dstem
from si_viewer.processing.frames import NotApplicable, get_frame
from si_viewer.session import DatasetKind, Session, classify


def _cube(rx=4, ry=3, qx=6, qy=5, dtype=np.uint16):
    # Each value tells where it is: 1000 Rx + 100 Ry + 10 Qx + Qy.
    rx_, ry_, qx_, qy_ = np.meshgrid(np.arange(rx), np.arange(ry), np.arange(qx), np.arange(qy), indexing="ij")
    return (1000 * rx_ + 100 * ry_ + 10 * qx_ + qy_).astype(dtype)


def _write_emdfile(path, cube, q=(0.02, "A^-1"), r=(1.5, "A"), extra_2d=True):
    """A file laid out as py4DSTEM 0.14 / emdfile 1.0 save one."""
    with h5py.File(path, "w") as f:
        f.attrs.update({"authoring_program": "emdfile", "emd_group_type": "file", "version_major": 1})
        root = f.create_group("experiment")
        root.attrs.update({"emd_group_type": "root", "python_class": "Root"})
        dc = root.create_group("datacube")
        dc.attrs.update({"emd_group_type": "array", "python_class": "DataCube"})
        dc.create_dataset("data", data=cube)
        for i, name in enumerate(("Rx", "Ry", "Qx", "Qy")):
            dim = dc.create_dataset(f"dim{i}", data=np.array([0, 1]))
            dim.attrs.update({"name": name, "units": "pixels"})
        cal = root.create_group("metadatabundle/calibration")
        cal.attrs.update({"emd_group_type": "metadata", "python_class": "Calibration"})
        cal.create_dataset("Q_pixel_size", data=q[0])
        cal.create_dataset("Q_pixel_units", data=q[1].encode())
        cal.create_dataset("R_pixel_size", data=r[0])
        cal.create_dataset("R_pixel_units", data=r[1].encode())
        cal.create_dataset("_root_treepath", data=b"")
        if extra_2d:
            vi = root.create_group("datacube/virtual_bf")
            vi.attrs.update({"emd_group_type": "array", "python_class": "VirtualImage"})
            vi.create_dataset("data", data=np.ones(cube.shape[:2], np.float32))
            bv = root.create_group("braggvectors")
            bv.attrs.update({"emd_group_type": "custom", "python_class": "BraggVectors"})


def test_reads_a_py4dstem_file_in_its_orientation(tmp_path):
    path = tmp_path / "cube.h5"
    cube = _cube()
    _write_emdfile(path, cube)
    assert py4dstem.is_emdfile(path)
    signals = load_file(path)
    assert [s.metadata.General.title for s in signals] == ["datacube", "virtual_bf"]
    s = signals[0]
    # [Rx, Ry, Qx, Qy] stays the array: Rx the scan's rows, Qx the pattern's rows.
    np.testing.assert_array_equal(s.data, cube)
    am = s.axes_manager
    assert [(a.name, a.size) for a in am.navigation_axes] == [("x", 3), ("y", 4)]
    assert [(a.name, a.size) for a in am.signal_axes] == [("kx", 5), ("ky", 6)]
    assert {(a.scale, a.units) for a in am.navigation_axes} == {(1.5, "Å")}
    assert {(a.scale, a.units) for a in am.signal_axes} == {(0.02, "1/Å")}
    assert s.metadata.py4DSTEM.calibration["R_pixel_units"] == "A"
    image = signals[1]
    assert image.data.shape == (4, 3) and {a.units for a in image.axes_manager.signal_axes} == {"Å"}


def test_uncalibrated_axes_are_pixels(tmp_path):
    path = tmp_path / "cube.h5"
    _write_emdfile(path, _cube(), q=(1, "pixels"), r=(1, "pixels"), extra_2d=False)
    s = load_file(path)[0]
    assert {(a.scale, a.units) for a in s.axes_manager._axes} == {(1.0, "px")}


def test_large_cubes_stay_in_the_file(tmp_path, monkeypatch):
    path = tmp_path / "cube.h5"
    cube = _cube()
    _write_emdfile(path, cube, extra_2d=False)
    monkeypatch.setattr(py4dstem, "IN_MEMORY_LIMIT", 10)
    s = load_file(path, lazy=True)[0]
    assert s._lazy
    np.testing.assert_array_equal(np.asarray(s.data.compute()), cube)


def test_other_hdf5_files_are_refused(tmp_path):
    path = tmp_path / "other.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=np.zeros(3))
    assert not py4dstem.is_emdfile(path)
    with pytest.raises(Exception):
        load_file(path)


def test_kind_and_axis_names():
    import hyperspy.api as hs

    s = hs.signals.Signal2D(np.zeros((2, 3, 4, 5)))
    s.axes_manager.navigation_axes[1].name = "z"
    name_4d_axes(s)
    assert classify(s) is DatasetKind.DIFFRACTION4D
    assert [a.name for a in s.axes_manager._axes] == ["y", "x", "ky", "kx"]
    assert classify(hs.signals.Signal2D(np.zeros((3, 4, 5)))) is DatasetKind.STACK


@pytest.fixture
def opened(tmp_path):
    path = tmp_path / "cube.h5"
    _write_emdfile(path, _cube(), extra_2d=False)
    session = Session()
    entry = session.open_file(path)
    return session, session.datasets[entry.dataset_ids[0]]


def test_opened_in_memory_with_its_frames(opened):
    session, ds = opened
    cube = _cube().astype(np.float64)
    assert ds.kind is DatasetKind.DIFFRACTION4D and not ds.lazy
    assert ds.summary()["kind"] == "4dstem"
    np.testing.assert_allclose(get_frame(ds, projection="scan"), cube.sum(axis=(2, 3)))
    np.testing.assert_allclose(get_frame(ds), cube.sum(axis=(2, 3)))
    np.testing.assert_allclose(get_frame(ds, projection="mean-pattern"), cube.mean(axis=(0, 1)), rtol=1e-6)
    np.testing.assert_allclose(get_frame(ds, projection="max-pattern"), cube.max(axis=(0, 1)))
    with pytest.raises(NotApplicable):
        get_frame(ds, projection="nope")
    assert session.thumbnail(ds, 64).startswith(b"\x89PNG")


def test_bin_q_and_r(opened):
    session, ds = opened
    cube = _cube()
    q = session.rebin(ds, [1, 1, 2, 2], in_place=False)[0]
    # 6 × 5 patterns: cropped to whole bins, then summed; the sums fit uint32.
    assert q.signal.data.shape == (4, 3, 3, 2) and q.signal.data.dtype == np.uint32
    np.testing.assert_array_equal(q.signal.data[1, 2, 0, 1], cube[1, 2, 0:2, 2:4].sum())
    assert [round(a.scale, 3) for a in q.signal.axes_manager.signal_axes] == [0.04, 0.04]
    r = session.rebin(ds, [2, 1, 1, 1])[0]
    assert r is ds and ds.signal.data.shape == (2, 3, 6, 5) and ds.revision == 1
    np.testing.assert_array_equal(ds.signal.data[1, 0], cube[2:4, 0].sum(axis=0))
    with pytest.raises(ValueError):
        session.rebin(ds, [1, 1, 1, 1])


def test_opened_through_the_api(tmp_path):
    from fastapi.testclient import TestClient

    from si_viewer.app import create_app

    path = tmp_path / "cube.h5"
    _write_emdfile(path, _cube(), extra_2d=False)
    client = TestClient(create_app(Session()))
    r = client.post("/api/files", json={"path": str(path)})
    assert r.status_code == 200, r.text
    d = r.json()["datasets"][0]
    assert d["kind"] == "4dstem" and d["navigation_shape"] and d["signal_shape"]
    frame = client.get(f"/api/datasets/{d['id']}/frame?projection=mean-pattern")
    assert frame.status_code == 200
    assert client.get(f"/api/datasets/{d['id']}/frame?projection=nope").status_code == 400


# ---------- the 4D viewer: patterns and virtual images ----------

from si_viewer.processing import fourd


def test_detector_masks():
    circle = fourd.detector_mask((6, 6), fourd.Detector(shape="circle", cx=3, cy=3, r=1))
    assert circle.sum() == 4 and circle[2:4, 2:4].all()  # the 4 pixels whose centres are 0.71 away
    ring = fourd.detector_mask((9, 9), fourd.Detector(shape="annulus", cx=4.5, cy=4.5, r=2, r_inner=1))
    assert not ring[4, 4] and ring[4, 6] and not ring[4, 8]
    rect = fourd.detector_mask((5, 5), fourd.Detector(shape="rectangle", x0=3, y0=1, x1=1, y1=3))
    assert rect.sum() == 4 and rect[1:3, 1:3].all()
    point = fourd.detector_mask((5, 5), fourd.Detector(shape="point", cx=2.7, cy=0.2))
    assert point.sum() == 1 and point[0, 2]
    with pytest.raises(ValueError):
        fourd.Detector(shape="annulus", r=1, r_inner=2)


def test_virtual_image_and_patterns():
    data = _cube().astype(np.uint16)  # [Rx, Ry, Qx, Qy] = [y, x, ky, kx]
    d = fourd.Detector(shape="rectangle", x0=1, y0=2, x1=3, y1=4)  # kx 1:3, ky 2:4
    block = data[:, :, 2:4, 1:3].astype(np.float64)
    np.testing.assert_allclose(fourd.virtual_image(data, d, "sum"), block.sum(axis=(2, 3)))
    np.testing.assert_allclose(fourd.virtual_image(data, d, "mean"), block.mean(axis=(2, 3)), rtol=1e-6)
    np.testing.assert_allclose(fourd.virtual_image(data, d, "max"), block.max(axis=(2, 3)))
    none = fourd.Detector(shape="circle", cx=-50, cy=-50, r=1)
    assert not fourd.virtual_image(data, none).any()
    np.testing.assert_array_equal(fourd.pattern_at(data, x=2.5, y=1.2), data[1, 2])
    with pytest.raises(IndexError):
        fourd.pattern_at(data, x=3, y=0)
    region = fourd.Region(kind="rect", x0=0, y0=1, x1=2, y1=3)  # columns 0:2, rows 1:3
    mean, n = fourd.pattern_over(data, region)
    assert n == 4
    np.testing.assert_allclose(mean, data[1:3, 0:2].mean(axis=(0, 1)), rtol=1e-6)
    np.testing.assert_allclose(fourd.pattern_over(data, region, "max")[0], data[1:3, 0:2].max(axis=(0, 1)))
    with pytest.raises(ValueError):
        fourd.pattern_over(data, fourd.Region(kind="rect", x0=0.6, y0=0, x1=1.4, y1=1))
    # A triangle over positions (0, 0), (1, 0), (0, 1) (centres inside), on lazy data too.
    import dask.array as da

    tri = fourd.Region(kind="polygon", points=[(0, 0), (2.2, 0), (0, 2.2)])
    want = np.stack([data[0, 0], data[0, 1], data[1, 0]])
    for cube in (data, da.from_array(data, chunks=(2, 2, 6, 5))):
        got, n = fourd.pattern_over(cube, tri, "max")
        assert n == 3
        np.testing.assert_allclose(got, want.max(axis=0))


def test_lazy_data_gives_the_same_virtual_image():
    import dask.array as da

    data = _cube()
    d = fourd.Detector(shape="annulus", cx=2.5, cy=3, r=3, r_inner=1)
    np.testing.assert_allclose(fourd.virtual_image(da.from_array(data, chunks=(1, 3, 6, 5)), d), fourd.virtual_image(data, d))


def test_several_virtual_images_at_once():
    import dask.array as da

    data = _cube()
    items = [
        (fourd.Detector(shape="circle", cx=2.5, cy=3, r=1.6), "sum"),
        (fourd.Detector(shape="annulus", cx=2.5, cy=3, r=3, r_inner=1), "mean"),
        (fourd.Detector(shape="circle", cx=-50, cy=-50, r=1), "max"),  # covers nothing
    ]
    want = np.stack([fourd.virtual_image(data, d, r) for d, r in items])
    for cube in (data, da.from_array(data, chunks=(1, 3, 6, 5))):
        got = fourd.virtual_images(cube, items)
        assert got.shape == (3, *data.shape[:2]) and got.dtype == np.float32
        np.testing.assert_allclose(got, want, rtol=1e-6)
    assert fourd.virtual_images(data, []).shape == (0, *data.shape[:2])


def test_virtual_images_through_the_api_and_kept(tmp_path):
    from fastapi.testclient import TestClient

    from si_viewer.app import create_app
    from si_viewer.project import open_project, save_project

    path = tmp_path / "cube.h5"
    _write_emdfile(path, _cube(), extra_2d=False)
    session = Session()
    client = TestClient(create_app(session))
    cube_id = client.post("/api/files", json={"path": str(path)}).json()["datasets"][0]["id"]
    detector = {"shape": "circle", "cx": 2.5, "cy": 3, "r": 1.6}
    r = client.post(f"/api/datasets/{cube_id}/4d/virtual", json={"detector": detector, "response": "mean"})
    assert r.status_code == 200 and r.headers["X-Array-Shape"] == "4,3"
    assert client.get(f"/api/datasets/{cube_id}/4d/pattern?x=1&y=3").headers["X-Array-Shape"] == "6,5"
    assert client.get(f"/api/datasets/{cube_id}/4d/pattern?x=9&y=0").status_code == 416
    region = client.post(f"/api/datasets/{cube_id}/4d/pattern", json={"region": {"kind": "rect", "x0": 0, "y0": 0, "x1": 3, "y1": 2}})
    assert region.status_code == 200 and '"positions": 6' in region.headers["X-Array-Meta"]
    assert client.post(f"/api/datasets/{cube_id}/4d/virtual", json={"detector": {"shape": "nope"}}).status_code == 422
    two = client.post(f"/api/datasets/{cube_id}/4d/virtuals", json={"items": [{"detector": detector}, {"detector": detector, "response": "max"}]})
    assert two.status_code == 200 and two.headers["X-Array-Shape"] == "2,4,3"

    saved = client.post(f"/api/datasets/{cube_id}/4d/virtual/save", json={"detector": detector, "response": "mean"}).json()
    image = next(d for d in saved["file"]["datasets"] if d["id"] == saved["id"])
    assert image["kind"] == "image" and image["parent_id"] == cube_id and image["origin"] == "virtual"
    assert image["title"] == "Virtual circle r 0.032 1/Å, mean"
    assert image["virtual"]["detector"]["r"] == 1.6 and image["virtual"]["response"] == "mean"
    assert [a["units"] for a in image["axes"]] == ["Å", "Å"]
    second = client.post(f"/api/datasets/{cube_id}/4d/virtual/save", json={"detector": detector, "response": "mean"}).json()
    assert next(d for d in second["file"]["datasets"] if d["id"] == second["id"])["title"].endswith(", mean 2")

    out = save_project(session, tmp_path / "a.hspy", level="results")
    again = Session()
    open_project(again, out)
    kept = again.datasets[saved["id"]]
    assert kept.virtual["detector"]["shape"] == "circle" and kept.origin == "virtual"
    np.testing.assert_allclose(kept.signal.data, session.datasets[saved["id"]].signal.data)


# ---------- phase 2: apertures, the bright-field disk, bad pixels ----------


def test_aperture_sets_are_the_union_of_discs():
    d = fourd.Detector(shape="apertures", r=1, apertures=[{"x": 2, "y": 2}, {"x": 6.5, "y": 2.5}, {"x": 2.2, "y": 2.1}])
    mask = fourd.detector_mask((8, 9), d)
    assert mask[1:3, 1:3].all() and mask[2, 6] and mask.sum() == 4 + 5
    assert not fourd.detector_mask((8, 9), fourd.Detector(shape="apertures", r=2)).any()
    data = _cube()
    expected = data[:, :, 1:3, 1:3].astype(float).sum(axis=(2, 3))
    np.testing.assert_allclose(
        fourd.virtual_image(data, fourd.Detector(shape="apertures", r=1, apertures=[{"x": 2, "y": 2}])), expected,
    )
    assert fourd.describe(d, "sum", 0.5, "mrad") == "Virtual 3 apertures r 0.5 mrad"


def test_bright_field_disk():
    p = np.full((40, 40), 2.0)
    yy, xx = np.mgrid[0:40, 0:40] + 0.5
    p[(xx - 20.5) ** 2 + (yy - 15.5) ** 2 <= 36] = 100
    p[3, 30] = 99  # a hot pixel isn't the disk
    disk = fourd.bright_field_disk(p)
    assert disk["cx"] == pytest.approx(20.5) and disk["cy"] == pytest.approx(15.5)
    assert disk["r"] == pytest.approx(6, abs=0.1)
    with pytest.raises(ValueError):
        fourd.bright_field_disk(np.ones((5, 5)))


def test_bad_pixels_found_outside_the_disk_and_repaired():
    import hyperspy.api as hs

    pattern = np.full((20, 20), 10.0)
    yy, xx = np.mgrid[0:20, 0:20] + 0.5
    pattern[(xx - 10) ** 2 + (yy - 10) ** 2 <= 9] = 500  # the bright-field disk: its edge isn't hot
    pattern[3, 4] = 300  # hot
    pattern[15, 16] = 0.5  # dead
    pattern[0, 9] = 400  # hot, on the border
    found = fourd.bad_pixels(pattern, hot=5, dead=0.2)
    assert sorted(map(tuple, np.argwhere(found["hot"]))) == [(0, 9), (3, 4)]
    assert list(map(tuple, np.argwhere(found["dead"]))) == [(15, 16)]
    mask = fourd.interior(found["hot"] | found["dead"])
    assert mask.sum() == 2

    data = np.broadcast_to(pattern, (2, 3, 20, 20)).astype(np.uint16).copy()
    out = fourd.repair(hs.signals.Signal2D(data), mask)
    assert out.data.dtype == np.float32
    assert (out.data[..., 3, 4] == 10).all() and (out.data[..., 15, 16] == 10).all()
    assert (out.data[..., 0, 9] == 400).all()  # the border is left
    np.testing.assert_array_equal(np.delete(out.data.reshape(6, -1), [3 * 20 + 4, 15 * 20 + 16], axis=1),
                                  np.delete(data.reshape(6, -1), [3 * 20 + 4, 15 * 20 + 16], axis=1))


def test_bad_pixels_through_the_api(tmp_path):
    from fastapi.testclient import TestClient

    from si_viewer.app import create_app

    pattern = np.full((12, 12), 20, np.uint16)
    pattern[4:8, 4:8] = 900
    pattern[2, 3] = 2000
    path = tmp_path / "cube.h5"
    _write_emdfile(path, np.broadcast_to(pattern, (3, 2, 12, 12)).copy(), extra_2d=False)
    session = Session()
    client = TestClient(create_app(session))
    cube = client.post("/api/files", json={"path": str(path)}).json()["datasets"][0]["id"]
    disk = client.get(f"/api/datasets/{cube}/4d/bf-disk").json()
    assert disk["cx"] == pytest.approx(6) and disk["pixels"] == 16
    found = client.post(f"/api/datasets/{cube}/4d/bad-pixels", json={"hot": 5, "dead": 0.2}).json()
    assert found == {"hot": [[2, 3]], "dead": [], "border": 0}
    copy = client.post(f"/api/datasets/{cube}/4d/bad-pixels/repair", json={"in_place": False}).json()
    assert copy["id"] != cube and copy["pixels"] == 1
    assert (session.datasets[copy["id"]].signal.data[..., 2, 3] == 20).all()
    assert (session.datasets[cube].signal.data[..., 2, 3] == 2000).all()
    done = client.post(f"/api/datasets/{cube}/4d/bad-pixels/repair", json={}).json()
    assert done["id"] == cube and (session.datasets[cube].signal.data[..., 2, 3] == 20).all()
    assert client.post(f"/api/datasets/{cube}/4d/bad-pixels/repair", json={}).status_code == 422
