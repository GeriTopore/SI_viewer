"""Background subtraction of diffraction patterns (pyxem's methods)."""

import time

import numpy as np
import pytest

from si_viewer.processing.pattern_background import PatternBackgroundOptions, subtract, subtract_all

N = 64
CX, CY = 30.5, 33.5  # the beam, off the middle


def _pattern():
    """Disks (r 2.5) on a diffuse background falling off from the beam."""
    yy, xx = np.mgrid[0:N, 0:N] + 0.5
    r = np.hypot(xx - CX, yy - CY)
    z = 200 * np.exp(-r / 12) + 5
    disks = [(CX, CY), (CX + 15, CY), (CX, CY - 15), (CX - 11, CY + 11)]
    for x, y in disks:
        z = z + 300 * (np.hypot(xx - x, yy - y) <= 2.5)
    return z.astype(np.float32), disks


@pytest.mark.parametrize("opts", [
    PatternBackgroundOptions(method="median kernel", footprint=11),
    PatternBackgroundOptions(method="difference of gaussians", min_sigma=1, max_sigma=8),
    PatternBackgroundOptions(method="radial median"),
    PatternBackgroundOptions(method="h-dome", h=0.3),
])
def test_each_method_keeps_the_disks_and_drops_the_background(opts):
    z, disks = _pattern()
    out = subtract(z, opts, (CX, CY))
    assert out.shape == z.shape and out.dtype == np.float32
    # Between the disks, near the beam where the background is strong: mostly gone.
    yy, xx = np.mgrid[0:N, 0:N] + 0.5
    off = np.ones_like(z, bool)
    for x, y in disks:
        off &= np.hypot(xx - x, yy - y) > 5
    near = off & (np.hypot(xx - CX, yy - CY) < 14)
    ring_disk = (int(CY), int(CX + 15))
    if opts.method == "h-dome":  # normalised peaks: the disks stand out of near zero
        assert out[ring_disk] > 0.5 and np.median(np.abs(out[near])) < 0.1
    else:
        assert np.median(np.abs(out[near])) < 0.2 * np.median(z[near])
        assert out[ring_disk] > 0.5 * 300


def test_radial_median_needs_the_beams_centre():
    z, _ = _pattern()
    yy, xx = np.mgrid[0:N, 0:N] + 0.5
    ring = np.abs(np.hypot(xx - CX, yy - CY) - 8) < 1
    radial = PatternBackgroundOptions(method="radial median")
    # About the beam: the ring's background goes; about the middle (pyxem's guess), it doesn't.
    assert np.abs(subtract(z, radial, (CX, CY))[ring]).mean() < np.abs(subtract(z, radial, None)[ring]).mean() / 3


def test_subtract_all_per_position_centres():
    z, _ = _pattern()
    data = np.broadcast_to(z, (2, 3, N, N)).copy()
    opts = PatternBackgroundOptions(method="radial median")
    centres = np.broadcast_to(np.array([CX, CY]), (2, 3, 2))
    out = subtract_all(data, opts, centres)
    assert out.shape == data.shape
    np.testing.assert_allclose(out[1, 2], subtract(z, opts, (CX, CY)))
    np.testing.assert_allclose(subtract_all(data, opts, np.array([CX, CY]))[0, 0], out[0, 0])


def _wait(client, job):
    for _ in range(600):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_background_through_the_api(tmp_path):
    from fastapi.testclient import TestClient
    from test_4dstem import _write_emdfile

    from si_viewer.app import create_app
    from si_viewer.session import Session

    z, _ = _pattern()
    cube = np.broadcast_to(z, (3, 4, N, N)).copy()
    path = tmp_path / "bg.h5"
    _write_emdfile(path, cube, extra_2d=False)
    session = Session()
    client = TestClient(create_app(session))
    cube_id = client.post("/api/files", json={"path": str(path)}).json()["datasets"][0]["id"]
    url = f"/api/datasets/{cube_id}/4d/background"
    opts = {"method": "median kernel", "footprint": 11}

    r = client.post(f"{url}/preview", json={"options": opts, "x": 1, "y": 2})
    assert r.status_code == 200 and r.headers["X-Array-Shape"] == f"{N},{N}"
    one = np.frombuffer(r.content, np.float32).reshape(N, N)
    np.testing.assert_allclose(one, subtract(z, PatternBackgroundOptions(**opts)), atol=1e-4)
    assert client.post(f"{url}/preview", json={"options": opts, "scan": "mean"}).status_code == 200
    assert client.post(f"{url}/preview", json={"options": {"method": "nope"}}).status_code == 422

    # As a copy: the data is kept, the copy under it is subtracted.
    job = _wait(client, client.post(f"{url}/run", json={"options": opts, "in_place": False}).json())
    assert job["status"] == "done", job["error"]
    copy = session.get_dataset(job["result"]["id"])
    assert copy.id != cube_id and copy.parent_id == cube_id and copy.origin == "pattern-background"
    np.testing.assert_allclose(np.asarray(copy.signal.data)[2, 3], one, atol=1e-4)
    assert copy.signal.metadata.Signal.background == "median kernel 11 px"
    np.testing.assert_array_equal(np.asarray(session.get_dataset(cube_id).signal.data), cube)
    # In place: the data itself.
    job = _wait(client, client.post(f"{url}/run", json={"options": opts}).json())
    assert job["status"] == "done" and job["result"]["id"] == cube_id
    np.testing.assert_allclose(np.asarray(session.get_dataset(cube_id).signal.data)[0, 0], one, atol=1e-4)
