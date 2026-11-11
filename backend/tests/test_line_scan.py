"""Line scans cut from a spectrum image along a line."""

import hyperspy.api as hs
import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_thickness import _link, _open, _role, _si

from si_viewer.app import create_app
from si_viewer.processing.line_scan import LineSpec, extract, sample_points


@pytest.fixture
def client():
    return TestClient(create_app())


def _ramp(h=6, w=8, c=3) -> np.ndarray:
    """Spectra whose channels are x, y and 1 at each pixel centre."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32) + 0.5
    return np.stack([x, y, np.ones_like(x)], axis=-1)


def test_horizontal_line_hits_pixel_centres():
    out = extract(_ramp(), LineSpec(x0=0.5, y0=2.5, x1=6.5, y1=2.5))
    assert out.shape == (7, 3)
    assert np.allclose(out[:, 0], np.arange(7) + 0.5) and np.allclose(out[:, 1], 2.5)


def test_any_angle_is_interpolated():
    # A diagonal: x and y follow the line (the ramp is linear, so exactly).
    line = LineSpec(x0=1.0, y0=1.0, x1=5.0, y1=4.0)
    out = extract(_ramp(), line)
    xs, ys = sample_points(line)
    assert len(out) == 6  # floor(5) + 1
    assert np.allclose(out[:, 0], xs[:, 0]) and np.allclose(out[:, 1], ys[:, 0])


def test_width_averages_across_the_line():
    data = _ramp()
    data[..., 2] = (np.arange(6) == 2)[:, None]  # channel 2: 1 on row 2 only
    out = extract(data, LineSpec(x0=0.5, y0=2.5, x1=6.5, y1=2.5, width=3))
    # Rows 1, 2 and 3 averaged: y is still 2.5 and row 2 counts a third.
    assert np.allclose(out[:, 1], 2.5) and np.allclose(out[:, 2], 1 / 3)


def test_outside_and_too_short():
    with pytest.raises(ValueError, match="leaves"):
        extract(_ramp(), LineSpec(x0=0.5, y0=2.5, x1=20, y1=2.5))
    with pytest.raises(ValueError, match="shorter"):
        extract(_ramp(), LineSpec(x0=1, y0=1, x1=1.5, y1=1))


def test_line_scan_api(client, tmp_path):
    s = hs.signals.Signal1D(_ramp(6, 8))
    s.axes_manager[0].scale = s.axes_manager[1].scale = 0.5
    s.axes_manager[0].units = s.axes_manager[1].units = "nm"
    path = tmp_path / "si.hspy"
    s.save(path)
    f = client.post("/api/files", json={"path": str(path)}).json()
    file = f.get("file", f)
    si = next(d for d in file["datasets"] if d["kind"] == "spectrum_image")
    r = client.post(
        f"/api/datasets/{si['id']}/line-scan",
        json={"line": {"x0": 0.5, "y0": 0.5, "x1": 6.5, "y1": 5.5, "width": 3}, "name": "Line 1"},
    )
    assert r.status_code == 200, r.text
    ls = r.json()["dataset"]
    assert ls["line_scan"] and ls["origin"] == "line-scan" and ls["parent_id"] == si["id"]
    assert ls["title"].endswith("· Line 1")
    distance = next(a for a in ls["axes"] if a["navigate"])
    assert distance["units"] == "nm" and distance["size"] == 8  # floor(hypot(6, 5)) + 1
    assert distance["scale"] * 7 == pytest.approx(0.5 * np.hypot(6, 5))
    # Deleting it removes it.
    assert client.delete(f"/api/datasets/{ls['id']}").status_code == 200
    assert client.post(
        f"/api/datasets/{si['id']}/line-scan", json={"line": {"x0": 0, "y0": 0, "x1": 30, "y1": 0}},
    ).status_code == 422


def _eels(tmp_path, name, offset):
    s = hs.signals.Signal1D(np.random.default_rng(0).uniform(1, 2, (6, 8, 50)).astype(np.float32))
    s.set_signal_type("EELS")
    s.axes_manager[-1].offset = offset
    s.metadata.General.title = name
    path = tmp_path / f"{name}.hspy"
    s.save(path)
    return path


def test_linked_spectrum_images_are_cut_and_linked(client, tmp_path):
    ll = _si(_open(client, _eels(tmp_path, "ll", -10)))
    hl = _si(_open(client, _eels(tmp_path, "hl", 400)))
    _role(client, ll, "low_loss")
    _link(client, hl, ll)
    r = client.post(
        f"/api/datasets/{hl['id']}/line-scan", json={"line": {"x0": 1, "y0": 5, "x1": 7, "y1": 1, "width": 2}},
    )
    assert r.status_code == 200, r.text
    ls, partners = r.json()["dataset"], r.json()["partners"]
    assert len(partners) == 1 and partners[0]["parent_id"] == ll["id"]
    assert ls["eels"]["role"] == "core_loss" and partners[0]["eels"]["role"] == "low_loss"
    assert ls["eels"]["linked"] == [partners[0]["id"]]
    # Each is cut from its own spectrum image.
    data = client.app.state.session.get_dataset(partners[0]["id"]).signal.data
    src = client.app.state.session.get_dataset(ll["id"]).signal.data
    assert data.shape[-1] == 50 and src.min() <= data.min() and data.max() <= src.max()


# ---------- rebinning a line scan ----------


def test_a_line_scan_is_rebinned_along_both_axes(client, tmp_path):
    ll = _si(_open(client, _eels(tmp_path, "ll", -10)))
    hl = _si(_open(client, _eels(tmp_path, "hl", 400)))
    _role(client, ll, "low_loss")
    _link(client, hl, ll)
    r = client.post(f"/api/datasets/{hl['id']}/line-scan", json={"line": {"x0": 0.5, "y0": 2.5, "x1": 7.5, "y1": 2.5}})
    ls, partner = r.json()["dataset"], r.json()["partners"][0]
    session = client.app.state.session
    before = np.asarray(session.get_dataset(ls["id"]).signal.data, np.float64)
    axes = {a["navigate"]: a for a in ls["axes"]}
    assert before.shape == (8, 50)

    r = client.post(f"/api/datasets/{ls['id']}/rebin", json={"factors": [2, 5]})
    assert r.status_code == 200, r.text
    now = r.json()["dataset"]
    assert now["navigation_shape"] == [4] and now["signal_shape"] == [10] and now["line_scan"]
    new = {a["navigate"]: a for a in now["axes"]}
    assert new[True]["scale"] == pytest.approx(2 * axes[True]["scale"])
    assert new[False]["scale"] == pytest.approx(5 * axes[False]["scale"])
    np.testing.assert_allclose(
        session.get_dataset(ls["id"]).signal.data, before.reshape(4, 2, 10, 5).sum(axis=(1, 3)), rtol=1e-5,
    )
    assert session.get_dataset(ls["id"]).rebinned[-1].startswith("Distance×2, ")
    # The linked line scan is binned with it.
    assert session.get_dataset(partner["id"]).signal.data.shape == (4, 10)
    # Factors outside the axes are refused.
    assert client.post(f"/api/datasets/{ls['id']}/rebin", json={"factors": [5, 1]}).status_code == 422
