"""EELS example files: spectrum images and line scans (see conftest.py for the data)."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from si_viewer.app import create_app
from si_viewer.session import DatasetKind, Session, classify


def _array(r) -> np.ndarray:
    assert r.status_code == 200, r.text
    shape = tuple(int(n) for n in r.headers["X-Array-Shape"].split(","))
    return np.frombuffer(r.content, dtype="<f4").reshape(shape)


def _open(path):
    session = Session()
    entry = session.open_file(path)
    assert len(entry.dataset_ids) == 1
    return session, session.get_dataset(entry.dataset_ids[0])


def _energy(ds):
    axis = ds.signal.axes_manager.signal_axes[0]
    return axis.offset, axis.scale, axis.size


# ---------- loading ----------


def test_cuzn_spectrum_image(eels_cuzn):
    _, ds = _open(eels_cuzn)
    assert ds.kind is DatasetKind.SPECTRUM_IMAGE and not ds.is_line_scan
    s = ds.summary()
    assert s["signal_type"] == "EELS" and s["eds"] is None
    assert s["navigation_shape"] == [40, 50] and s["signal_shape"] == [162]
    assert _energy(ds) == (700.0, 8.0, 162)
    assert ds.signal.axes_manager.signal_axes[0].units == "eV"


def test_dm4_spectrum_image(eels_dm4):
    _, ds = _open(eels_dm4)
    assert ds.kind is DatasetKind.SPECTRUM_IMAGE and ds.lazy and not ds.is_line_scan
    s = ds.summary()
    assert s["signal_type"] == "EELS"
    assert s["navigation_shape"] == [120, 120] and s["signal_shape"] == [3456]
    offset, scale, size = _energy(ds)
    assert offset == pytest.approx(362, abs=1) and scale == pytest.approx(0.09, abs=0.005)
    assert offset + scale * (size - 1) == pytest.approx(673, abs=1)


def test_dm4_pixel_spectrum(eels_dm4):
    client = TestClient(create_app())
    entry = client.post("/api/files", json={"path": str(eels_dm4)}).json()
    si = entry["datasets"][0]
    assert si["kind"] == "spectrum_image" and si["lazy"] is True
    base = f"/api/datasets/{si['id']}"
    spectrum = _array(client.get(f"{base}/pixel-spectrum", params={"x": 7, "y": 3}))
    assert spectrum.shape == (3456,)
    # (x, y) = (7, 3) is row 3, column 7 of the array.
    ds = client.app.state.session.get_dataset(si["id"])
    np.testing.assert_allclose(spectrum, ds.signal.data[3, 7], rtol=1e-6)
    assert client.get(f"{base}/frame").status_code == 200


def test_no_initial_maps_for_eels(eels_cuzn):
    session, ds = _open(eels_cuzn)
    assert session.computed_maps(ds) == []


# ---------- line scans ----------


@pytest.fixture(scope="module")
def line_scans(eels_linescan, eels_linescan_low_loss):
    client = TestClient(create_app())
    out = {}
    for key, path in (("core", eels_linescan), ("low", eels_linescan_low_loss)):
        entry = client.post("/api/files", json={"path": str(path)}).json()
        si = entry["datasets"][0]
        ds = client.app.state.session.get_dataset(si["id"])
        ds.materialise()
        out[key] = (si, np.asarray(ds.signal.data, np.float32))
    return client, out


@pytest.mark.parametrize("key, channels, offset", [("core", 586, 428.5), ("low", 1024, -50.0)])
def test_linescan_loads_as_spectrum_image(line_scans, key, channels, offset):
    client, scans = line_scans
    si, data = scans[key]
    assert si["kind"] == "spectrum_image" and si["line_scan"] is True
    assert si["signal_type"] == "EELS" and si["eds"] is None
    assert si["navigation_shape"] == [40] and si["signal_shape"] == [channels]
    energy = next(a for a in si["axes"] if not a["navigate"])
    assert (energy["offset"], energy["scale"], energy["units"]) == (offset, 0.5, "eV")
    assert data.shape == (40, channels)


def test_classify_linescan(eels_linescan):
    import hyperspy.api as hs

    assert classify(hs.load(str(eels_linescan), lazy=True)) is DatasetKind.SPECTRUM_IMAGE


@pytest.mark.parametrize("key", ["core", "low"])
def test_linescan_frame_is_the_data(line_scans, key):
    client, scans = line_scans
    si, data = scans[key]
    base = f"/api/datasets/{si['id']}"
    frame = _array(client.get(f"{base}/frame"))
    assert frame.shape == data.shape  # (40, channels): position × energy
    np.testing.assert_array_equal(frame, data)
    # Binning doesn't apply.
    np.testing.assert_array_equal(_array(client.get(f"{base}/frame", params={"bin": 4})), data)


@pytest.mark.parametrize("key", ["core", "low"])
def test_linescan_spectra(line_scans, key):
    client, scans = line_scans
    si, data = scans[key]
    base = f"/api/datasets/{si['id']}"

    for i in (0, 17, 39):
        px = client.get(f"{base}/pixel-spectrum", params={"x": i, "y": 0})
        assert '"pixels": 1' in px.headers["X-Array-Meta"]
        np.testing.assert_allclose(_array(px), data[i], rtol=1e-6)
    assert client.get(f"{base}/pixel-spectrum", params={"x": 40, "y": 0}).status_code == 416
    assert client.get(f"{base}/pixel-spectrum", params={"x": 0, "y": 1}).status_code == 416

    rect = client.post(f"{base}/region-spectrum", json={"kind": "rect", "x0": 3, "y0": 0, "x1": 8, "y1": 1})
    assert '"pixels": 5' in rect.headers["X-Array-Meta"]
    np.testing.assert_allclose(_array(rect), data[3:8].sum(0), rtol=1e-5)

    total = _array(client.get(f"{base}/spectrum"))
    np.testing.assert_allclose(total, data.sum(0, dtype=np.float64), rtol=1e-5)

    m = _array(client.get(f"{base}/energy-map", params={"lo": 440.0, "hi": 460.0}))
    assert m.shape == (1, 40)


def test_linescan_energy_map_values(line_scans):
    client, scans = line_scans
    si, data = scans["core"]
    # Channels 20..39 of the core-loss line scan: 438.5 eV up to 458 eV.
    m = _array(client.get(f"/api/datasets/{si['id']}/energy-map", params={"lo": 438.5, "hi": 448.5}))
    np.testing.assert_allclose(m[0], data[:, 20:40].sum(1), rtol=1e-4)


def test_linescan_navigators_and_thumbnail(line_scans):
    client, scans = line_scans
    si, _ = scans["core"]
    sources = client.get(f"/api/datasets/{si['id']}/navigators").json()
    assert [s["id"] for s in sources] == [si["id"]]
    r = client.get(f"/api/datasets/{si['id']}/thumbnail")
    assert r.status_code == 200 and r.content[:4] == b"\x89PNG"


def test_linescan_refuses_2d_operations(line_scans):
    client, scans = line_scans
    si, data = scans["core"]
    base = f"/api/datasets/{si['id']}"

    r = client.post(f"{base}/rebin", json={"factors": [2, 1]})
    assert r.status_code == 400 and "line scan" in r.json()["detail"]

    for spatial in ({"spatial": "gaussian"}, {"spatial": "mean", "size": 3}):
        r = client.put(f"{base}/filters", json=spatial)
        assert r.status_code == 422 and "line scan" in r.json()["detail"]

    assert client.put(f"{base}/map-options", json={}).status_code == 400
    assert client.post(f"{base}/fit-maps", json={}).status_code == 400
    assert client.put(f"{base}/elements", json={"elements": ["Fe"]}).status_code == 400  # not EDS


def test_linescan_spectral_smoothing_allowed(eels_linescan):
    client = TestClient(create_app())
    si = client.post("/api/files", json={"path": str(eels_linescan)}).json()["datasets"][0]
    base = f"/api/datasets/{si['id']}"
    raw = _array(client.get(f"{base}/frame"))
    r = client.put(f"{base}/filters", json={"spectral": "gaussian", "spectral_sigma": 2.0})
    assert r.status_code == 200, r.text
    smoothed = _array(client.get(f"{base}/frame"))
    assert smoothed.shape == raw.shape and not np.array_equal(smoothed, raw)
    assert smoothed.sum() == pytest.approx(raw.sum(), rel=1e-3)
    px = _array(client.get(f"{base}/pixel-spectrum", params={"x": 5, "y": 0}))
    np.testing.assert_allclose(px, smoothed[5], rtol=1e-4, atol=1e-3)
    client.put(f"{base}/filters", json={})
    np.testing.assert_array_equal(_array(client.get(f"{base}/frame")), raw)


# ---------- the other examples ----------


def test_haadf_emd_has_a_stack(haadf_emd):
    session = Session()
    entry = session.open_file(haadf_emd)
    stacks = [d for d in session.siblings_in_file(entry.id) if d.kind is DatasetKind.STACK]
    assert stacks and stacks[0].signal.axes_manager.navigation_size > 1
