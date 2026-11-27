"""Bragg disk detection (processing/disks.py, the 4D-STEM disk endpoints)."""

import time

import numpy as np

from si_viewer.processing import disks, fourd

R = 4.0  # disk radius (px)
SPOTS = [(0, 0, 100.0), (-14, 0, 30.0), (14, 0, 30.0), (0, -14, 30.0), (0, 14, 30.0)]  # dy, dx, intensity


def _pattern(cy, cx, n=64, spots=SPOTS, ss=4):
    """Discs at (cy + dy, cx + dx) (pixel-centre positions, pixel i spanning
    [i, i + 1)), drawn supersampled ``ss`` × so their centres are sub-pixel."""
    yy, xx = (np.mgrid[0 : n * ss, 0 : n * ss] + 0.5) / ss
    fine = np.zeros((n * ss, n * ss))
    for dy, dx, a in spots:
        fine += a * (np.hypot(yy - (cy + dy), xx - (cx + dx)) <= R)
    return fine.reshape(n, ss, n, ss).mean(axis=(1, 3)).astype(np.float32) + 1.0


def _cube(ny=3, nx=4):
    # The pattern drifts by 0.25 px per scan step (descan), so positions differ.
    return np.stack([np.stack([_pattern(32.3 + 0.25 * iy, 31.8 + 0.25 * ix) for ix in range(nx)]) for iy in range(ny)])


def _template(cube):
    mean = cube.mean(axis=(0, 1))
    return disks.probe_template(mean, fourd.bright_field_disk(mean))


def test_finds_the_disks_of_a_pattern_to_a_fraction_of_a_pixel():
    pattern = _pattern(32.3, 31.8)
    found = disks.find_disks(pattern, _template(_cube()), disks.DiskOptions(threshold=0.3))
    assert len(found) == 5
    want = np.array([(32.3 + dy, 31.8 + dx) for dy, dx, _ in SPOTS])
    for ky, kx, score in found:
        assert np.min(np.hypot(want[:, 0] - ky, want[:, 1] - kx)) < 0.35
        assert 0.3 <= score <= 1.0
    # The strongest first; a higher threshold or a lower cap keeps fewer.
    assert found[0, 2] == found[:, 2].max()
    assert len(disks.find_disks(pattern, _template(_cube()), disks.DiskOptions(threshold=0.3, max_disks=2))) == 2


def test_a_disc_template_and_no_disks():
    flat = np.ones((64, 64), np.float32)
    assert disks.find_disks(flat, disks.disc_template(R), disks.DiskOptions()).shape == (0, 3)
    found = disks.find_disks(_pattern(32.3, 31.8), disks.disc_template(R), disks.DiskOptions(threshold=0.3))
    assert len(found) == 5


def test_every_pattern_and_the_vector_map():
    import dask.array as da

    cube = _cube()
    opts = disks.DiskOptions(threshold=0.3)
    reports = []
    table = disks.find_all(cube, _template(cube), opts, lambda f, m: reports.append(f))
    assert table.shape == (3 * 4 * 5, 5) and reports[-1] == 1.0
    # Rows by scan position (pixel centres), each position's 5 disks.
    assert sorted({(y, x) for y, x in table[:, :2]}) == [(y + 0.5, x + 0.5) for y in range(3) for x in range(4)]
    at = disks.disks_at(table, x=2, y=1)
    assert len(at) == 5
    assert np.min(np.hypot(at[:, 2] - (32.3 + 0.25), at[:, 3] - (31.8 + 0.5))) < 0.35
    # Lazy data: the same.
    lazy = disks.find_all(da.from_array(cube, chunks=(1, 2, 64, 64)), _template(cube), opts)
    np.testing.assert_allclose(lazy, table)
    bvm = disks.vector_map(table, (64, 64), upsample=2)
    assert bvm.shape == (128, 128)
    assert abs(bvm.sum() - len(table)) < 1e-3
    # Each spot gathers the 12 disks found there across the scan (map pixels
    # are half a pattern pixel; the scan drifts by up to 0.75 px).
    for dy, dx, _ in SPOTS:
        cy, cx = 2 * (32.55 + dy), 2 * (32.175 + dx)
        assert abs(bvm[int(cy) - 3 : int(cy) + 4, int(cx) - 3 : int(cx) + 4].sum() - 12) < 1e-3


def test_bragg_vectors_as_pyxem_diffraction_vectors():
    import hyperspy.api as hs
    from pyxem.signals import DiffractionVectors

    cube = _cube()
    s = hs.signals.Signal2D(cube)
    for ax, (name, scale, offset, units) in zip(
        s.axes_manager.navigation_axes + s.axes_manager.signal_axes,
        [("x", 1.5, 0, "nm"), ("y", 1.5, 0, "nm"), ("kx", 0.02, -0.64, "1/nm"), ("ky", 0.03, -0.96, "1/nm")],
    ):
        ax.name, ax.scale, ax.offset, ax.units = name, scale, offset, units
    table = disks.find_all(cube, _template(cube), disks.DiskOptions(threshold=0.3))
    v = disks.to_vectors(table, s)
    assert isinstance(v, DiffractionVectors)
    assert v.axes_manager.navigation_shape == (4, 3)
    assert [a.scale for a in v.axes_manager.navigation_axes] == [1.5, 1.5]
    assert v.column_names == ["ky", "kx", "intensity"] and v.units[:2] == ["1/nm", "1/nm"]
    # In the pattern's calibration: value = offset + scale × index (pixel i's centre at index i).
    row = table[(table[:, 0] == 1.5) & (table[:, 1] == 2.5)]
    got = np.asarray(v.data[1, 2])
    want = np.column_stack([-0.96 + 0.03 * (row[:, 2] - 0.5), -0.64 + 0.02 * (row[:, 3] - 0.5), row[:, 4]])
    np.testing.assert_allclose(np.sort(got, axis=0), np.sort(want, axis=0), rtol=1e-5, atol=1e-6)
    # And back to pattern pixels.
    np.testing.assert_allclose(np.sort(disks.vectors_at(v, 2, 1), axis=0), np.sort(row[:, 2:], axis=0), atol=1e-4)
    np.testing.assert_allclose(disks.table_of(v)[np.lexsort(disks.table_of(v).T[::-1])], table[np.lexsort(table.T[::-1])], atol=1e-4)
    # pyxem's own tools work on it.
    assert v.get_magnitudes().data.shape == (3, 4)


def _wait(client, job):
    for _ in range(600):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_disks_through_the_api_and_kept_with_the_analysis(tmp_path):
    from fastapi.testclient import TestClient

    from si_viewer.app import create_app
    from si_viewer.project import open_project, save_project
    from si_viewer.session import Session

    from test_4dstem import _write_emdfile

    path = tmp_path / "disks.h5"
    _write_emdfile(path, _cube(), extra_2d=False)
    session = Session()
    client = TestClient(create_app(session))
    cube_id = client.post("/api/files", json={"path": str(path)}).json()["datasets"][0]["id"]
    opts = {"threshold": 0.3}

    t = client.post(f"/api/datasets/{cube_id}/4d/disks/template", json={"options": opts})
    assert t.status_code == 200 and '"kind": "probe"' in t.headers["X-Array-Meta"]
    preview = client.post(
        f"/api/datasets/{cube_id}/4d/disks/preview",
        json={"options": opts, "sources": [{"id": "c1", "x": 1, "y": 2}, {"id": "mean", "scan": "mean"}, {"id": "off", "x": 99, "y": 0}]},
    ).json()
    assert len(preview["disks"]["c1"]) == 5 and "mean" in preview["disks"] and "off" not in preview["disks"]
    assert client.get(f"/api/datasets/{cube_id}/4d/disks?x=0&y=0").status_code == 404

    job = _wait(client, client.post(f"/api/datasets/{cube_id}/4d/disks/run", json={"options": opts}).json())
    assert job["status"] == "done", job["error"]
    vectors_id = job["result"]["id"]
    assert job["result"]["count"] == 60
    datasets = {d["id"]: d for d in job["result"]["file"]["datasets"]}
    vectors = datasets[vectors_id]
    assert vectors["kind"] == "other" and vectors["origin"] == "disks" and vectors["parent_id"] == cube_id
    assert vectors["disks"]["count"] == 60 and vectors["disks"]["options"]["threshold"] == 0.3
    bvm = next(d for d in datasets.values() if d["origin"] == "bragg-map")
    assert bvm["parent_id"] == vectors_id and bvm["kind"] == "image"
    assert len(client.get(f"/api/datasets/{cube_id}/4d/disks?x=1&y=2").json()["disks"]) == 5

    # Found again: the same datasets, made again.
    job = _wait(client, client.post(f"/api/datasets/{cube_id}/4d/disks/run", json={"options": {"threshold": 0.3, "max_disks": 1}}).json())
    assert job["result"]["id"] == vectors_id and job["result"]["count"] == 12

    from pyxem.signals import DiffractionVectors

    assert isinstance(session.get_dataset(vectors_id).signal, DiffractionVectors)
    saved = save_project(session, tmp_path / "p.hspy")
    other = Session()
    open_project(other, saved)
    restored = other.get_dataset(vectors_id)
    assert isinstance(restored.signal, DiffractionVectors)
    assert restored.disks["count"] == 12 and restored.signal.data.shape == (3, 4)
    assert other.bragg_vectors_of(other.get_dataset(cube_id)).id == vectors_id
    assert len(disks.vectors_at(restored.signal, 1, 2)) == 1


def test_the_vector_mask_and_magnitudes_through_the_api(tmp_path):
    """pyxem's ``to_mask`` × the data (its orientation mapping tutorial's
    pre-processing) and its magnitude histogram."""
    from fastapi.testclient import TestClient

    from si_viewer.app import create_app
    from si_viewer.session import Session

    from test_4dstem import _write_emdfile

    path = tmp_path / "mask.h5"
    cube = _cube()
    _write_emdfile(path, cube, extra_2d=False)
    session = Session()
    client = TestClient(create_app(session))
    cube_id = client.post("/api/files", json={"path": str(path)}).json()["datasets"][0]["id"]
    assert client.post(f"/api/datasets/{cube_id}/4d/disks/mask", json={}).status_code == 409
    job = _wait(client, client.post(f"/api/datasets/{cube_id}/4d/disks/run", json={"options": {"threshold": 0.3}}).json())
    assert job["status"] == "done", job["error"]

    # The histogram holds every vector.
    m = client.post(f"/api/datasets/{cube_id}/4d/disks/magnitudes", json={"bins": 50}).json()
    assert len(m["k"]) == 50 and sum(m["counts"]) == 60 and m["reflections"] == []

    # The mask at a position: discs about its disks.
    one = client.get(f"/api/datasets/{cube_id}/4d/disks/mask?x=2&y=1&disk_r=3")
    assert one.status_code == 200

    job = _wait(client, client.post(f"/api/datasets/{cube_id}/4d/disks/mask", json={"disk_r": 3}).json())
    assert job["status"] == "done", job["error"]
    masked = session.get_dataset(job["result"]["id"])
    assert masked.origin == "vector-masked" and masked.parent_id == cube_id
    data = np.asarray(masked.signal.data)
    # Each disk's centre kept, the space between them zeroed.
    for iy in range(3):
        for ix in range(4):
            cy, cx = 32.3 + 0.25 * iy, 31.8 + 0.25 * ix
            for dy, dx, _ in SPOTS:
                y, x = int(cy + dy), int(cx + dx)
                assert data[iy, ix, y, x] == cube[iy, ix, y, x]
            assert data[iy, ix, int(cy) + 7, int(cx) + 7] == 0
    # As pyxem makes it.
    vectors = session.bragg_vectors_of(session.get_dataset(cube_id)).signal
    mask = disks.vector_mask(vectors, session.get_dataset(cube_id).signal, 3)
    np.testing.assert_array_equal(data, np.where(mask, cube, 0))
    assert not vectors.metadata.has_item("Peaks.signal_axes")

    # Anything made in the app can be deleted (with what was made from it); the file's data can't.
    vectors_id = session.bragg_vectors_of(session.get_dataset(cube_id)).id
    bvm_id = session.bragg_map_of(session.get_dataset(cube_id)).id
    assert client.delete(f"/api/datasets/{masked.id}").status_code == 200
    r = client.delete(f"/api/datasets/{vectors_id}")
    assert r.status_code == 200
    left = {d["id"] for d in r.json()["file"]["datasets"]}
    assert vectors_id not in left and bvm_id not in left and masked.id not in left
    assert client.post(f"/api/datasets/{cube_id}/4d/disks/magnitudes", json={}).status_code == 409
    client.post(f"/api/undo/{r.json()['undo']}")
    assert session.bragg_map_of(session.get_dataset(cube_id)).id == bvm_id
    assert client.delete(f"/api/datasets/{cube_id}").status_code == 400


def test_powder_reflections_of_a_structure():
    from orix.crystal_map import Phase
    from diffpy.structure import Atom, Lattice, Structure

    from si_viewer.processing.orientation import reflections_1d

    a = 4.078
    au = Structure([Atom("Au", p) for p in ((0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5))],
                   Lattice(a, a, a, 90, 90, 90))
    got = reflections_1d(Phase(space_group=225, structure=au), 0.8)
    assert [r["hkl"] for r in got][:3] == [[1, 1, 1], [2, 0, 0], [2, 2, 0]]
    assert abs(got[0]["k"] - np.sqrt(3) / a) < 1e-6 and all(r["k"] <= 0.8 for r in got)
