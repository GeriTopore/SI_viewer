"""Calibrating 4D-STEM patterns: a ring fitted as an ellipse, the pixel size
from it, a camera length or entered (processing/calibration.py, the
/4d/calibration endpoints)."""

import numpy as np
import pytest

from si_viewer.processing import calibration as cal

N = 128


def _ring(cx=63.2, cy=64.7, a=30.0, b=27.0, angle=0.5, width=1.0, n=N):
    """An elliptical ring (semi-axes a, b; a's axis at ``angle`` from x)
    centred on (cx, cy) (pixel i spanning [i, i + 1)), on a background."""
    yy, xx = np.indices((n, n)) + 0.5
    xr = (xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle)
    yr = -(xx - cx) * np.sin(angle) + (yy - cy) * np.cos(angle)
    rho = np.sqrt((xr / a) ** 2 + (yr / b) ** 2)
    return (np.exp(-(((rho - 1) * np.sqrt(a * b) / width) ** 2) / 2) * 100 + 5).astype(np.float32)


def test_fits_an_elliptical_ring():
    ring = cal.fit_ring(_ring(), cal.RingOptions(r_min=20, r_max=38), {"cx": 64, "cy": 64})
    assert ring["cx"] == pytest.approx(63.2, abs=0.02) and ring["cy"] == pytest.approx(64.7, abs=0.02)
    assert ring["a"] == pytest.approx(30, abs=0.05) and ring["b"] == pytest.approx(27, abs=0.05)
    assert ring["angle"] == pytest.approx(0.5, abs=0.01)
    assert ring["radius"] == pytest.approx(np.sqrt(30 * 27), abs=0.05)
    assert ring["ellipticity"] == pytest.approx(30 / 27 - 1, abs=0.005)
    assert ring["rms"] < 0.1
    # A circle: no ellipticity; its angle doesn't matter.
    circle = cal.fit_ring(_ring(a=25, b=25), cal.RingOptions(r_min=18, r_max=32), {"cx": 64, "cy": 64})
    assert circle["ellipticity"] < 0.003
    # pyxem's affine makes it circular, of radius √(a b) (areas kept).
    aff = np.asarray(cal.affine_of(ring))
    assert aff.shape == (3, 3) and np.allclose(aff, aff.T, atol=1e-6)
    t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    c, s = np.cos(ring["angle"]), np.sin(ring["angle"])
    pts = np.column_stack([ring["a"] * np.cos(t) * c - ring["b"] * np.sin(t) * s,
                           ring["a"] * np.cos(t) * s + ring["b"] * np.sin(t) * c])
    r = np.hypot(*(pts @ aff[:2, :2].T).T)
    assert r.max() - r.min() < 1e-6 and r.mean() == pytest.approx(ring["radius"], rel=1e-6)


def test_no_ring_is_an_error():
    with pytest.raises(ValueError):
        cal.fit_ring(np.ones((N, N), np.float32), cal.RingOptions(r_min=20, r_max=38), {"cx": 64, "cy": 64})
    with pytest.raises(ValueError):
        cal.ring_points(_ring(), 64, 64, 30, 20)


def test_pixel_size():
    assert cal.wavelength(200) == pytest.approx(0.025079, abs=1e-5)
    assert cal.wavelength(300) == pytest.approx(0.019687, abs=1e-5)
    # Au 111 (2.355 Å) on a ring of 20 px: 1/(2.355 × 20) 1/Å per px.
    assert cal.scale_from_ring(20, 2.355) == pytest.approx(1 / 47.1)
    # 15 µm pixels at 100 mm, 200 kV: 1.5e-4 rad per px / λ.
    assert cal.scale_from_camera(100, 15, 200) == pytest.approx(1.5e-4 / 0.025079, rel=1e-4)


def test_calibration_through_the_api_and_kept(tmp_path):
    from fastapi.testclient import TestClient
    from test_4dstem import _write_emdfile

    from si_viewer.app import create_app
    from si_viewer.project import open_project, save_project
    from si_viewer.session import Session

    # Every pattern: the ring and a bright-field disk at the centre.
    pattern = _ring()
    yy, xx = np.indices((N, N)) + 0.5
    pattern += 500 * (np.hypot(xx - 63.2, yy - 64.7) <= 4)
    cube = np.broadcast_to(pattern, (3, 4, N, N)).astype(np.float32)
    path = tmp_path / "ring.h5"
    _write_emdfile(path, cube, extra_2d=False)
    session = Session()
    client = TestClient(create_app(session))
    cube_id = client.post("/api/files", json={"path": str(path)}).json()["datasets"][0]["id"]

    ring = client.post(f"/api/datasets/{cube_id}/4d/calibration/ring", json={"options": {"r_min": 20, "r_max": 38}}).json()
    assert ring["a"] == pytest.approx(30, abs=0.1) and len(ring["points"]) > 100 and len(ring["affine"]) == 3
    bad = client.post(f"/api/datasets/{cube_id}/4d/calibration/ring", json={"options": {"r_min": 38, "r_max": 20}})
    assert bad.status_code == 422
    assert "Au" in client.get(f"/api/datasets/{cube_id}/4d/calibration/standards").json()

    # The ellipse kept (step 2): with pyxem's affine, switched on.
    kept = client.post(f"/api/datasets/{cube_id}/4d/calibration/ellipse", json={"ring": ring})
    assert kept.status_code == 200
    cal = session.get_dataset(cube_id).calibration
    assert cal["ring"]["a"] == pytest.approx(ring["a"]) and cal["affine"] is not None and cal["enabled"]["ellipse"]

    # From the ring: Au 111 → 1/d at the ring's radius; 0 at its centre.
    body = {"how": "ring", "ring": ring, "d": 2.355, "standard": "Au", "hkl": "111", "units": "1/nm"}
    r = client.post(f"/api/datasets/{cube_id}/4d/calibration", json=body).json()
    want = 10 / (2.355 * ring["radius"])
    assert r["scale"] == pytest.approx(want)
    ds = session.get_dataset(cube_id)
    kx, ky = ds.signal.axes_manager.signal_axes
    assert kx.scale == pytest.approx(want) and kx.units == "1/nm"
    # The ring's centre (pixel-edge position) at 0: index cx − 0.5.
    assert kx.offset + kx.scale * (ring["cx"] - 0.5) == pytest.approx(0, abs=1e-9)
    assert ky.offset + ky.scale * (ring["cy"] - 0.5) == pytest.approx(0, abs=1e-9)
    assert ds.calibration["how"]["hkl"] == "111" and ds.calibration["ring"]["a"] == pytest.approx(ring["a"])

    # Entered: kept in its units, 0 left where it was.
    r = client.post(f"/api/datasets/{cube_id}/4d/calibration", json={"how": "entered", "scale": 0.02, "units": "1/Å", "centre": False}).json()
    assert kx.scale == 0.02 and kx.units == "1/Å"
    assert kx.offset + kx.scale * (ring["cx"] - 0.5) == pytest.approx(0, abs=1e-9)
    # The pixel size set again: the ellipse kept as it was.
    assert ds.calibration["ring"]["a"] == pytest.approx(ring["a"]) and ds.calibration["enabled"]["ellipse"]
    assert client.post(f"/api/datasets/{cube_id}/4d/calibration", json={"how": "camera"}).status_code == 422

    # Kept with the analysis, also when the data is read from its file again.
    saved = save_project(session, tmp_path / "p.hspy")
    other = Session()
    open_project(other, saved)
    kx2 = other.get_dataset(cube_id).signal.axes_manager.signal_axes[0]
    assert kx2.scale == 0.02 and kx2.units == "1/Å" and other.get_dataset(cube_id).calibration["scale"] == 0.02
