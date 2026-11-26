"""Centring 4D-STEM data: the direct beam found, the descan plane, the
patterns shifted (processing/centring.py and the /4d/beam endpoints)."""

import time

import numpy as np

from si_viewer.processing import centring, fourd

N = 64


def _cube(ny=6, nx=8, ss=4):
    """A disk (r 4) whose centre drifts linearly over the scan (descan):
    returns the data and the true centres ([y, x, 2]: x, y, pixel i spanning
    [i, i + 1))."""
    yy, xx = (np.mgrid[0 : N * ss, 0 : N * ss] + 0.5) / ss
    data = np.zeros((ny, nx, N, N), np.float32)
    true = np.zeros((ny, nx, 2))
    for iy in range(ny):
        for ix in range(nx):
            cx, cy = 30.3 + 0.4 * ix, 33.1 - 0.3 * iy
            true[iy, ix] = cx, cy
            fine = (np.hypot(xx - cx, yy - cy) <= 4).astype(float)
            data[iy, ix] = fine.reshape(N, ss, N, ss).mean(axis=(1, 3)) * 100 + 1
    return data, true


def _signal(data):
    import hyperspy.api as hs

    return hs.signals.Signal2D(data)


def _bf(data):
    return fourd.bright_field_disk(data.mean(axis=(0, 1)))


def test_finds_the_beam_and_the_descan_plane():
    data, true = _cube()
    shifts = centring.find_beam(_signal(data), centring.BeamOptions(), _bf(data))
    assert shifts.shape == (6, 8, 2)
    np.testing.assert_allclose(centring.beam_positions(shifts, (N, N)), true, atol=0.05)
    # Blur: to the pixel.
    blur = centring.find_beam(_signal(data), centring.BeamOptions(method="blur"), _bf(data))
    np.testing.assert_allclose(centring.beam_positions(blur, (N, N)), true, atol=0.75)
    # The drift is linear: a plane fits it (all of it, or its corners).
    for mode in ("plane", "corners"):
        used = centring.plane(shifts, centring.PlaneOptions(mode=mode, corners=0.25))
        np.testing.assert_allclose(centring.beam_positions(used, (N, N)), true, atol=0.05)
    assert centring.plane(shifts, centring.PlaneOptions(mode="none")) is shifts
    stats = centring.shift_stats(shifts)
    assert stats["x"][1] - stats["x"][0] > 2.5 and stats["y"][1] - stats["y"][0] > 1.2


def test_centring_puts_the_beam_at_the_centre():
    data, _ = _cube()
    shifts = centring.find_beam(_signal(data), centring.BeamOptions(), _bf(data))
    out = centring.centre(_signal(data), centring.plane(shifts, centring.PlaneOptions()))
    assert out.data.dtype == np.float32 and out.data.shape == data.shape
    # Every pattern's beam at the centre (index N/2: pixel-edge position N/2 + 0.5).
    for iy, ix in [(0, 0), (5, 7), (2, 3)]:
        bf = fourd.bright_field_disk(out.data[iy, ix])
        assert abs(bf["cx"] - (N / 2 + 0.5)) < 0.15 and abs(bf["cy"] - (N / 2 + 0.5)) < 0.15
    # The disks' intensity is kept (float; the background of 1 shifted out at
    # the edges is replaced by 0, so it's left out).
    assert abs((out.data - 1).clip(0).sum() / (data - 1).sum() - 1) < 0.01
    # The pattern axes: 0 at the centre.
    assert [a.offset for a in out.axes_manager.signal_axes] == [-N / 2, -N / 2]


def _wait(client, job):
    for _ in range(600):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_centring_through_the_api(tmp_path):
    from fastapi.testclient import TestClient
    from test_4dstem import _write_emdfile

    from si_viewer.app import create_app
    from si_viewer.project import open_project, save_project
    from si_viewer.session import Session

    data, true = _cube()
    path = tmp_path / "drift.h5"
    _write_emdfile(path, data, extra_2d=False)
    session = Session()
    client = TestClient(create_app(session))
    cube_id = client.post("/api/files", json={"path": str(path)}).json()["datasets"][0]["id"]
    assert client.get(f"/api/datasets/{cube_id}/4d/beam?x=0&y=0").status_code == 404

    job = _wait(client, client.post(f"/api/datasets/{cube_id}/4d/beam/find", json={}).json())
    assert job["status"] == "done", job["error"]
    made = {d["id"]: d for d in job["result"]["file"]["datasets"] if d["origin"] == "beam-shift"}
    assert sorted(d["beam"]["axis"] for d in made.values()) == ["x", "y"]
    assert all(d["parent_id"] == cube_id and d["kind"] == "image" for d in made.values())

    at = client.get(f"/api/datasets/{cube_id}/4d/beam?x=7&y=5").json()
    np.testing.assert_allclose(at["found"], true[5, 7], atol=0.05)
    np.testing.assert_allclose(at["used"], true[5, 7], atol=0.05)
    assert at["residual"] < 0.05
    assert client.get(f"/api/datasets/{cube_id}/4d/beam?x=99&y=0").status_code == 416

    # As a copy: the data is left as it is.
    job = _wait(client, client.post(f"/api/datasets/{cube_id}/4d/beam/centre", json={"in_place": False}).json())
    assert job["status"] == "done", job["error"]
    copy_id = job["result"]["id"]
    assert copy_id != cube_id
    copy = session.get_dataset(copy_id)
    assert copy.origin == "centred" and copy.kind.value == "4dstem"
    bf = fourd.bright_field_disk(np.asarray(copy.signal.data[5, 7]))
    assert abs(bf["cx"] - (N / 2 + 0.5)) < 0.15
    # In place: a new revision of the data.
    rev = session.get_dataset(cube_id).revision
    job = _wait(client, client.post(f"/api/datasets/{cube_id}/4d/beam/centre", json={"in_place": True}).json())
    assert job["result"]["id"] == cube_id and session.get_dataset(cube_id).revision > rev
    # The shifts were for the data before: not used again (that would shift it twice).
    assert client.post(f"/api/datasets/{cube_id}/4d/beam/centre", json={}).status_code == 409
    assert client.get(f"/api/datasets/{cube_id}/4d/beam?x=0&y=0").status_code == 409
    # Found again on the centred data: the beam is at the centre everywhere.
    job = _wait(client, client.post(f"/api/datasets/{cube_id}/4d/beam/find", json={}).json())
    r = job["result"]["range"]
    assert max(abs(v) for v in r["x"] + r["y"]) < 0.2

    saved = save_project(session, tmp_path / "p.hspy")
    other = Session()
    open_project(other, saved)
    assert other.beam_shifts_of(other.get_dataset(cube_id)).shape == (6, 8, 2)
