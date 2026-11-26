"""Calibrated Bragg vectors and their map (py4DSTEM-style): each position's
origin (the beam found, or its direct-beam disk; a descan plane), the
ellipse made circular, the pixel size; the ring fitted on the map."""

import time

import numpy as np
import pytest

from si_viewer.processing import disks, fourd

N = 64
A, B, ANGLE = 20.0, 17.0, 0.3  # the ring: semi-axes (px), a's angle from x
NY, NX = 4, 5


def _origin(iy, ix):
    return 31.3 + 0.3 * ix, 32.4 - 0.25 * iy  # the beam drifts (descan)


def _ring_point(t):
    c, s = np.cos(ANGLE), np.sin(ANGLE)
    return A * np.cos(t) * c - B * np.sin(t) * s, A * np.cos(t) * s + B * np.sin(t) * c


def _cube(ss=4):
    """Every pattern: the direct beam at its origin and 6 disks on the
    elliptical ring around it, at angles that differ from position to
    position (so the map fills the ring)."""
    yy, xx = (np.mgrid[0 : N * ss, 0 : N * ss] + 0.5) / ss
    out = np.zeros((NY, NX, N, N), np.float32)
    for iy in range(NY):
        for ix in range(NX):
            ox, oy = _origin(iy, ix)
            fine = 100.0 * (np.hypot(xx - ox, yy - oy) <= 3)
            for k in range(6):
                dx, dy = _ring_point(k * np.pi / 3 + 0.37 * (iy * NX + ix))
                fine += 40.0 * (np.hypot(xx - ox - dx, yy - oy - dy) <= 2.5)
            out[iy, ix] = fine.reshape(N, ss, N, ss).mean(axis=(1, 3)) + 1
    return out


@pytest.fixture(scope="module")
def found():
    cube = _cube()
    mean = cube.mean(axis=(0, 1))
    bf = fourd.bright_field_disk(mean)
    table = disks.find_all(cube, disks.probe_template(mean, bf), disks.DiskOptions(threshold=0.3, min_spacing=3))
    return cube, bf, table


def test_direct_beam_origins(found):
    _, bf, table = found
    origins = disks.direct_beam_origins(table, (NY, NX), bf)
    want = np.array([[_origin(iy, ix) for ix in range(NX)] for iy in range(NY)])
    np.testing.assert_allclose(origins, want, atol=0.3)
    gappy = origins.copy()
    gappy[0, 0] = np.nan
    filled = disks.fill_origins(gappy)
    np.testing.assert_allclose(filled[0, 0], np.nanmedian(gappy.reshape(-1, 2), axis=0))
    with pytest.raises(ValueError):
        disks.fill_origins(np.full((2, 2, 2), np.nan))


def test_calibrated_vectors_make_a_circle(found):
    _, bf, table = found
    origins = np.array([[_origin(iy, ix) for ix in range(NX)] for iy in range(NY)])
    ring = {"a": A, "b": B, "angle": ANGLE}
    q = disks.calibrated(table, origins, (0, 0), ring, scale=0.1)
    r = np.hypot(q[:, 0], q[:, 1])
    on_ring = r > 0.5  # the direct beam at 0
    # Every ring disk at √(a b) × 0.1, whatever its angle or position.
    np.testing.assert_allclose(r[on_ring], np.sqrt(A * B) * 0.1, atol=0.04)
    np.testing.assert_allclose(r[~on_ring], 0, atol=0.04)
    # Without the origins (one centre for all): the ring is smeared by the drift.
    raw = np.hypot(*disks.calibrated(table, None, _origin(0, 0), ring, scale=0.1).T)
    assert np.std(raw[raw > 0.5]) > 3 * np.std(r[on_ring])
    # The map: its total is the disks', its ring within one bin's width.
    m = disks.vector_map_of(q, np.ones(len(q)), half=3.2, bins=128)
    assert abs(m.sum() - len(q)) < 1e-3


def test_qr_rotation_turns_the_vectors(found):
    _, _, table = found
    centre = _origin(0, 0)
    q = disks.calibrated(table, None, centre)
    # As processing.dpc.rotate: y flipped if mirrored, then counter-clockwise.
    turned = disks.calibrated(table, None, centre, rotation={"theta": 90, "mirror": False})
    np.testing.assert_allclose(turned, np.column_stack([-q[:, 1], q[:, 0]]), atol=1e-9)
    mirrored = disks.calibrated(table, None, centre, rotation={"theta": 0, "mirror": True})
    np.testing.assert_allclose(mirrored, np.column_stack([q[:, 0], -q[:, 1]]), atol=1e-9)
    # After the pixel size (py4DSTEM's order): lengths scale, angles don't.
    both = disks.calibrated(table, None, centre, scale=0.1, rotation={"theta": 30, "mirror": True})
    np.testing.assert_allclose(np.hypot(*both.T), 0.1 * np.hypot(*q.T), atol=1e-9)


def _wait(client, job):
    for _ in range(600):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_calibrated_map_through_the_api(tmp_path, found):
    from fastapi.testclient import TestClient
    from test_4dstem import _write_emdfile

    from si_viewer.app import create_app
    from si_viewer.session import Session

    cube, _, _ = found
    path = tmp_path / "ring.h5"
    _write_emdfile(path, cube, extra_2d=False)
    session = Session()
    client = TestClient(create_app(session))
    cube_id = client.post("/api/files", json={"path": str(path)}).json()["datasets"][0]["id"]
    url = f"/api/datasets/{cube_id}"

    # No disks yet: no origin from them.
    assert client.post(f"{url}/4d/calibration/origin", json={"source": "vectors"}).status_code == 404
    job = _wait(client, client.post(f"{url}/4d/disks/run", json={"options": {"threshold": 0.3, "min_spacing": 3}}).json())
    assert job["status"] == "done", job["error"]
    bvm_of = lambda: next(d for d in session.datasets.values() if d.origin == "bragg-map")  # noqa: E731
    # Nothing calibrated yet: the map is raw (pattern px).
    assert bvm_of().disks["map"]["view"] == "raw" and bvm_of().signal.axes_manager.signal_axes[0].units == "px"
    # The beam's origin needs the beam found (Centring).
    assert client.post(f"{url}/4d/calibration/origin", json={"source": "beam"}).status_code == 409

    # The origin measured from the vectors and fitted: maps of each, on the scan.
    r = client.post(f"{url}/4d/calibration/origin", json={"source": "vectors", "plane": {"mode": "plane"}}).json()
    maps = {d["title"]: d for d in r["file"]["datasets"] if d["origin"] == "origin-map"}
    assert sorted(maps) == ["Origin x", "Origin x (fit)", "Origin y", "Origin y (fit)"]
    # Centring done on the vectors: its maps are under the Bragg vector map.
    assert {d["parent_id"] for d in maps.values()} == {bvm_of().id}
    assert r["rms"] < 0.3 and r["mean"][0] == pytest.approx(np.mean([_origin(0, i)[0] for i in range(NX)]), abs=0.3)
    cube_ds = session.get_dataset(cube_id)
    assert cube_ds.calibration["enabled"]["origin"] is True
    fitted = session.origin_maps_of(cube_ds)["fitted"]
    np.testing.assert_allclose(fitted, [[_origin(iy, ix) for ix in range(NX)] for iy in range(NY)], atol=0.3)
    # The map is now origin-corrected (px, 0 at the centre).
    assert bvm_of().disks["map"]["origin"] is True

    # The map on the pattern's pixels (the DP pane's "BVM"): every disk, the
    # pattern's size, each position moved to the mean origin while centring is on.
    import json as _json

    r = client.get(f"{url}/4d/bvm-pattern")
    assert r.status_code == 200
    shape = tuple(int(v) for v in r.headers["X-Array-Shape"].split(","))
    meta = _json.loads(r.headers["X-Array-Meta"])
    assert shape == (N, N) and meta == {"disks": job["result"]["count"], "centred": True}
    data = np.frombuffer(r.content, "<f4").reshape(shape)
    assert abs(data.sum() - job["result"]["count"]) < 1e-2
    # The direct beam lands in one spot at the mean origin (not smeared by the drift).
    mx, my = np.nanmean(fitted.reshape(-1, 2), axis=0)
    assert data[int(my) - 1 : int(my) + 2, int(mx) - 1 : int(mx) + 2].sum() >= 0.99 * NY * NX

    # The ring fitted on the map (origin-corrected): sharp, the right ellipse.
    ring = client.post(f"{url}/4d/calibration/ring", json={
        "options": {"r_min": 12, "r_max": 26}, "source": "bvm", "origin": {"origin": "direct", "plane": {"mode": "plane"}},
    }).json()
    assert ring["a"] == pytest.approx(A, abs=0.4) and ring["b"] == pytest.approx(B, abs=0.4)
    assert ring["ellipticity"] == pytest.approx(A / B - 1, abs=0.03)

    # Step 2 keeps the ellipse; step 4 the pixel size (the ellipse left as kept).
    assert client.post(f"{url}/4d/calibration/ellipse", json={"ring": ring}).status_code == 200
    r = client.post(f"{url}/4d/calibration", json={"how": "ring", "ring": ring, "d": 2.355, "units": "1/nm"}).json()
    scale = r["scale"]
    # The ellipse and the pixel size switched on; the origin stays.
    assert session.get_dataset(cube_id).calibration["enabled"] == {"origin": True, "ellipse": True, "scale": True}
    bvm = bvm_of()
    kx, ky = bvm.signal.axes_manager.signal_axes
    # Twice as fine as the pattern, in 1/nm, 0 at the centre.
    assert kx.units == "1/nm" and kx.scale == pytest.approx(scale / 2, rel=1e-6)
    assert kx.offset + kx.scale * (kx.size - 1) / 2 == pytest.approx(0, abs=1e-9)
    data = np.asarray(bvm.signal.data)
    assert abs(data.sum() - job["result"]["count"]) < 1e-2
    # The ring at 1/d (Au 111) in the map's units: most disks within a bin or two of it.
    yy, xx = np.indices(data.shape)
    k = np.hypot(kx.offset + kx.scale * xx, ky.offset + ky.scale * yy)
    on = np.abs(k - 10 / 2.355) < 2 * kx.scale
    assert data[on].sum() > 0.7 * data[k > 1].sum()

    # Switches: the ellipse off; then the raw map; then calibrated again.
    client.post(f"{url}/4d/calibration/switches", json={"enabled": {"ellipse": False}})
    assert bvm_of().disks["map"]["ellipse"] is False and bvm_of().disks["map"]["origin"] is True
    client.post(f"{url}/4d/calibration/switches", json={"view": "raw"})
    assert bvm_of().disks["map"]["view"] == "raw" and bvm_of().signal.axes_manager.signal_axes[0].units == "px"
    client.post(f"{url}/4d/calibration/switches", json={"view": "calibrated", "enabled": {"ellipse": True}})
    assert bvm_of().disks["map"] == {"view": "calibrated", "origin": True, "ellipse": True, "scale": True,
                                     "rotation": False, "units": "1/nm"}

    # The QR rotation (as Calibration step 2 finds it): kept, switched on,
    # the map turned into the scan's frame (its total unchanged; the ellipse
    # made circular, so the ring stays at 1/d).
    before = np.asarray(bvm_of().signal.data).copy()
    r = client.post(f"{url}/4d/calibration/rotation", json={"theta": 90, "mirror": False, "how": "curl", "curl_ratio": 0.05})
    assert r.status_code == 200, r.text
    cal = session.get_dataset(cube_id).calibration
    assert cal["rotation"] == {"how": "curl", "curl_ratio": 0.05, "theta": 90.0, "mirror": False}
    assert cal["enabled"]["rotation"] is True and bvm_of().disks["map"]["rotation"] is True
    turned = np.asarray(bvm_of().signal.data)
    assert abs(turned.sum() - before.sum()) < 1e-2
    # 90° counter-clockwise (x right, y down the rows): (x, y) → (−y, x), the map turned as an array.
    np.testing.assert_allclose(turned, np.rot90(before, k=-1), atol=0.05 * before.max())
    # Still circular with the rotation on (origin, ellipse, rotation, then the pixel size).
    table = session.vectors_table(session.get_dataset(cube_id))
    from si_viewer.processing import disks as _disks

    cal = session.get_dataset(cube_id).calibration
    q = _disks.calibrated(table, session.centring_origins(session.get_dataset(cube_id)), (0, 0), cal["ring"],
                          cal["scale"], cal["rotation"])
    rr = np.hypot(q[:, 0], q[:, 1])
    band = np.abs(rr - 10 / 2.355) < 0.15 * 10 / 2.355
    t = np.arctan2(q[band, 1], q[band, 0])
    # Radius against angle: no cos 2θ / sin 2θ left (the ellipse made circular).
    m = np.column_stack([np.ones(band.sum()), np.cos(2 * t), np.sin(2 * t)])
    c = np.linalg.lstsq(m, rr[band], rcond=None)[0]
    assert np.hypot(c[1], c[2]) / c[0] < 0.005
    assert client.post(f"{url}/4d/calibration/rotation", json={"theta": 400}).status_code == 422
    # Switched off: the map as before.
    client.post(f"{url}/4d/calibration/switches", json={"enabled": {"rotation": False}})
    np.testing.assert_allclose(np.asarray(bvm_of().signal.data), before, atol=1e-5)
    client.post(f"{url}/4d/calibration/switches", json={"enabled": {"rotation": True}})

    # Kept with the analysis: the calibration, the origin maps.
    from si_viewer.project import open_project, save_project

    saved = save_project(session, tmp_path / "p.hspy")
    other = Session()
    open_project(other, saved)
    c2 = other.get_dataset(cube_id)
    assert c2.calibration["enabled"]["origin"] is True and c2.calibration["origin"]["source"] == "vectors"
    assert c2.calibration["rotation"]["theta"] == 90.0 and c2.calibration["enabled"]["rotation"] is True
    np.testing.assert_allclose(other.origin_maps_of(c2)["fitted"], fitted, atol=1e-5)
