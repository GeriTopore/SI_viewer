"""DPC and iCOM from 4D-STEM data's centre of mass."""

import math
import time

import numpy as np

from si_viewer.processing import com
from si_viewer.processing.centring import PlaneOptions

NY, NX, K = 32, 32, 32
THETA = 30.0  # the detector's rotation from the scan (degrees)


def _potential():
    yy, xx = np.mgrid[0:NY, 0:NX].astype(float)
    phi = np.zeros((NY, NX))
    for cy, cx in [(8, 8), (8, 24), (24, 8), (24, 24), (16, 16)]:
        phi += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 2.5**2))
    return phi


def _corr(a, b):
    """Correlation of two scan images less their best-fit quadratic surfaces:
    a descan plane removed from the field takes a quadratic from the potential."""
    yy, xx = np.mgrid[0:NY, 0:NX].astype(float)
    basis = np.stack([np.ones_like(xx), xx, yy, xx * xx, xx * yy, yy * yy], axis=-1).reshape(-1, 6)

    def detrend(z):
        z = np.asarray(z, float).ravel()
        return z - basis @ np.linalg.lstsq(basis, z, rcond=None)[0]

    return np.corrcoef(detrend(a), detrend(b))[0, 1]


def _cube():
    """A Gaussian beam deflected by the gradient of ``_potential`` (in the
    detector's frame: the scan's field rotated by −THETA), on a descan plane."""
    phi = _potential()
    gy, gx = np.gradient(phi)
    gx, gy = 6 * gx, 6 * gy  # up to about 1.5 px
    t = math.radians(-THETA)
    dx, dy = math.cos(t) * gx - math.sin(t) * gy, math.sin(t) * gx + math.cos(t) * gy
    iy, ix = np.mgrid[0:NY, 0:NX]
    bx = K / 2 + dx + 0.04 * ix  # descan: up to 1.2 px across the scan
    by = K / 2 + dy - 0.03 * iy
    ky, kx = np.mgrid[0:K, 0:K] + 0.5
    cube = np.exp(-((kx[None, None] - bx[..., None, None]) ** 2 + (ky[None, None] - by[..., None, None]) ** 2) / (2 * 2.0**2))
    return (1000 * cube + 1).astype(np.float32), phi


def test_deflection_rotation_and_icom():
    from si_viewer.processing.dpc import integrate, rotate

    cube, phi = _cube()
    iy, ix = np.mgrid[0:NY, 0:NX]
    # The centring's shifts (what moves the beam to the centre) of the beam drawn.
    phi_g = np.gradient(phi)
    t = math.radians(-THETA)
    gx, gy = 6 * phi_g[1], 6 * phi_g[0]
    beam_x = K / 2 + math.cos(t) * gx - math.sin(t) * gy + 0.04 * ix
    beam_y = K / 2 + math.sin(t) * gx + math.cos(t) * gy - 0.03 * iy
    shifts = np.stack([K / 2 - beam_x, K / 2 - beam_y], axis=-1)
    dx, dy = com.deflection(shifts, PlaneOptions(mode="plane"))
    found = com.rotation_of(dx, dy)
    assert not found["mirror"] and abs((found["theta"] - THETA + 180) % 360 - 180) < 3
    phase = integrate(*rotate(dx, dy, found["theta"], found["mirror"]), pad=True)
    assert _corr(phase, phi) > 0.95


def _wait(client, job):
    for _ in range(600):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_com_dpc_through_the_api(tmp_path):
    from fastapi.testclient import TestClient
    from test_4dstem import _write_emdfile

    from si_viewer.app import create_app
    from si_viewer.session import Session

    cube, phi = _cube()
    path = tmp_path / "com.h5"
    _write_emdfile(path, cube, extra_2d=False)
    session = Session()
    client = TestClient(create_app(session))
    cube_id = client.post("/api/files", json={"path": str(path)}).json()["datasets"][0]["id"]
    url = f"/api/datasets/{cube_id}/4d/com/run"

    opts = {"beam": {"method": "center_of_mass", "threshold": 0}, "plane": {"mode": "plane"}}
    job = _wait(client, client.post(url, json={"options": opts}).json())
    assert job["status"] == "done", job["error"]
    r = job["result"]
    assert abs((r["theta"] - THETA + 180) % 360 - 180) < 3 and r["mirror"] is False
    made = {session.get_dataset(i).dpc["output"]: session.get_dataset(i) for i in r["ids"]}
    assert set(made) == {"colour", "x", "y", "idpc"}
    colour = made["colour"]
    assert colour.parent_id == cube_id and made["idpc"].parent_id == cube_id
    assert made["x"].parent_id == colour.id and made["y"].parent_id == colour.id
    assert made["idpc"].title == "iCOM" and colour.dpc["source"] == "4d"
    assert (made["x"].title, made["y"].title) == ("COM x", "COM y")
    icom = np.asarray(made["idpc"].signal.data)
    assert icom.shape == (NY, NX) and _corr(icom, phi) > 0.95
    # Shown as a DPC object: direction and magnitude.
    frame = client.get(f"/api/datasets/{colour.id}/dpc/frame")
    assert frame.status_code == 200 and frame.headers["X-Array-Shape"] == f"2,{NY},{NX}"
    # The quadrant DPC process doesn't take it.
    assert client.get(f"/api/datasets/{colour.id}/dpc").status_code == 409

    # Again with a rotation given: made in place (the same objects).
    opts["rotation"] = {"auto": False, "theta": THETA + 180, "mirror": False}
    job = _wait(client, client.post(url, json={"options": opts}).json())
    assert job["status"] == "done" and set(job["result"]["ids"]) == set(r["ids"])
    flipped = np.asarray(session.get_dataset(made["idpc"].id).signal.data)
    assert _corr(flipped, phi) < -0.95  # 180° off: the potential inverted
    # The blurred maximum is not a centre of mass.
    assert client.post(url, json={"options": {"beam": {"method": "blur"}}}).status_code == 422
