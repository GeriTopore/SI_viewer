"""Centre-of-mass virtual images of 4D-STEM data (as in the data), and the QR
rotation from the least curl of the Calibration's descan step."""

import math
import time

import numpy as np

from si_viewer.processing import com
from si_viewer.processing.centring import PlaneOptions, plane

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


def _beam():
    """Where ``_cube`` draws the beam ([y, x] each, pattern px)."""
    phi = _potential()
    gy, gx = np.gradient(phi)
    gx, gy = 6 * gx, 6 * gy
    t = math.radians(-THETA)
    iy, ix = np.mgrid[0:NY, 0:NX]
    bx = K / 2 + math.cos(t) * gx - math.sin(t) * gy + 0.04 * ix
    by = K / 2 + math.sin(t) * gx + math.cos(t) * gy - 0.03 * iy
    return bx, by


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


def test_rotation_of_the_deflection():
    """The beam's positions less the descan plane fitted to them, turned by
    the rotation leaving the least curl, integrate to the potential."""
    from si_viewer.processing.dpc import integrate, rotate

    _, phi = _cube()
    beam = np.stack(_beam(), axis=-1)
    d = beam - plane(beam.astype(np.float32), PlaneOptions(mode="plane"))
    found = com.rotation_of(d[..., 0], d[..., 1])
    assert not found["mirror"] and abs((found["theta"] - THETA + 180) % 360 - 180) < 3
    phase = integrate(*rotate(d[..., 0], d[..., 1], found["theta"], found["mirror"]), pad=True)
    assert _corr(phase, phi) > 0.95


def _wait(client, job):
    for _ in range(600):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def _open(tmp_path, cube):
    from fastapi.testclient import TestClient
    from test_4dstem import _write_emdfile

    from si_viewer.app import create_app
    from si_viewer.session import Session

    path = tmp_path / "com.h5"
    _write_emdfile(path, cube, extra_2d=False)
    session = Session()
    client = TestClient(create_app(session))
    cube_id = client.post("/api/files", json={"path": str(path)}).json()["datasets"][0]["id"]
    return session, client, cube_id


def test_qr_rotation_from_the_calibration(tmp_path):
    """Calibration step 2: the rotation leaving the least curl in the descan
    step's deflection (origins less their plane), then kept as the QR rotation."""
    cube, _ = _cube()
    session, client, cube_id = _open(tmp_path, cube)
    url = f"/api/datasets/{cube_id}/4d"
    # Nothing measured yet.
    assert client.post(f"{url}/calibration/rotation/find").status_code == 409
    job = client.post(f"{url}/beam/find", json={"options": {"method": "center_of_mass", "threshold": 0}}).json()
    assert _wait(client, job)["status"] == "done"
    r = client.post(f"{url}/calibration/origin", json={"source": "beam", "plane": {"mode": "plane"}})
    assert r.status_code == 200, r.text
    found = client.post(f"{url}/calibration/rotation/find")
    assert found.status_code == 200, found.text
    f = found.json()
    assert abs((f["theta"] - THETA + 180) % 360 - 180) < 3 and f["mirror"] is False and 0 <= f["curl_ratio"] < 0.5
    kept = client.post(f"{url}/calibration/rotation", json={"theta": f["theta"], "mirror": f["mirror"], "how": "curl", "curl_ratio": f["curl_ratio"]})
    assert kept.status_code == 200
    rotation = session.get_dataset(cube_id).calibration["rotation"]
    assert rotation["how"] == "curl" and abs(rotation["theta"] - f["theta"]) < 1e-9
    # The DPC / iCOM process is gone.
    assert client.post(f"{url}/com/run", json={}).status_code in (404, 405)


def test_live_com_as_in_the_data(tmp_path):
    """COM x / y and iCOM through a detector: the centre of mass of the
    pixels inside it from the pattern's centre, with no descan or rotation."""
    from si_viewer.processing import fourd
    from si_viewer.processing.dpc import integrate

    cube, _ = _cube()
    bx, by = _beam()
    # A disc around the beam's path (it moves less than 3 px), for the whole pattern.
    disc = fourd.Detector(shape="circle", cx=K / 2, cy=K / 2, r=12)
    x, y, icom = fourd.virtual_images(cube, [(disc, "com_x"), (disc, "com_y"), (disc, "icom")])
    # The beam where it is drawn (the flat background pulls it ~2 % toward the centre).
    assert np.abs(x - (bx - K / 2)).max() < 0.1 and np.abs(y - (by - K / 2)).max() < 0.1
    np.testing.assert_allclose(icom, integrate(x - x.mean(), y - y.mean(), pad=True), rtol=1e-4, atol=1e-5)

    session, client, cube_id = _open(tmp_path, cube)
    url = f"/api/datasets/{cube_id}/4d"
    detector = {"shape": "circle", "cx": K / 2, "cy": K / 2, "r": 12}
    r = client.post(f"{url}/virtual", json={"detector": detector, "response": "com_x"})
    assert r.status_code == 200
    got = np.frombuffer(r.content, np.float32).reshape(NY, NX)
    # In the pattern's units: its pixel size × the pixels.
    scale = session.get_dataset(cube_id).signal.axes_manager.signal_axes[0].scale
    np.testing.assert_allclose(got, x * scale, rtol=1e-3, atol=1e-5)
    two = client.post(f"{url}/virtuals", json={"items": [{"detector": detector, "response": "com_x"}, {"detector": detector}]})
    assert two.status_code == 200 and two.headers["X-Array-Shape"] == f"2,{NY},{NX}"
    saved = client.post(f"{url}/virtual/save", json={"detector": detector, "response": "icom"}).json()
    made = session.get_dataset(saved["id"])
    assert made.title.endswith("iCOM")


def test_live_com_colour(tmp_path):
    """The combined COM image: direction (degrees) and magnitude, as a DPC
    object is shown; saved as one (COM x and y under it)."""
    import json

    from fastapi.testclient import TestClient
    from test_4dstem import _write_emdfile

    from si_viewer.app import create_app
    from si_viewer.processing import fourd
    from si_viewer.session import Session

    cube, _ = _cube()
    disc = fourd.Detector(shape="circle", cx=K / 2, cy=K / 2, r=12)
    colour, x, y = fourd.virtual_images(cube, [(disc, "com"), (disc, "com_x"), (disc, "com_y")])
    assert colour.shape == (2, NY, NX)
    np.testing.assert_allclose(colour[1], np.hypot(x, y), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(colour[0], np.degrees(np.arctan2(y, x)), atol=1e-3)

    path = tmp_path / "colour.h5"
    _write_emdfile(path, cube, extra_2d=False)
    session = Session()
    client = TestClient(create_app(session))
    cube_id = client.post("/api/files", json={"path": str(path)}).json()["datasets"][0]["id"]
    url = f"/api/datasets/{cube_id}/4d"
    detector = {"shape": "circle", "cx": K / 2, "cy": K / 2, "r": 12}
    one = client.post(f"{url}/virtual", json={"detector": detector, "response": "com"})
    assert one.status_code == 200 and one.headers["X-Array-Shape"] == f"2,{NY},{NX}"
    # Several groups: each one's channels after the other's, counted in the meta.
    many = client.post(f"{url}/virtuals", json={"items": [{"detector": detector}, {"detector": detector, "response": "com"}]})
    assert many.headers["X-Array-Shape"] == f"3,{NY},{NX}"
    assert json.loads(many.headers["X-Array-Meta"])["channels"] == [1, 2]
    # Saved: a DPC object shown in colour, COM x and y under it; the process's own untouched.
    saved = client.post(f"{url}/virtual/save", json={"detector": detector, "response": "com"}).json()
    obj = session.get_dataset(saved["id"])
    assert obj.origin == "dpc" and obj.dpc["output"] == "colour" and obj.dpc["saved"] and obj.parent_id == cube_id
    kids = sorted(d.title for d in session.datasets.values() if d.parent_id == obj.id)
    assert kids == ["COM x", "COM y"]
    assert client.get(f"/api/datasets/{obj.id}/dpc/frame").headers["X-Array-Shape"] == f"2,{NY},{NX}"
    again = client.post(f"{url}/virtual/save", json={"detector": detector, "response": "com"}).json()
    assert again["id"] != saved["id"] and session.get_dataset(again["id"]).title.endswith(" 2")
