"""Cropping a spectrum image to a rectangle and/or an energy window."""

import numpy as np
import pytest

from test_insitu import fake_dm4  # noqa: F401 (fixture)


@pytest.fixture
def client(tmp_path):
    import hyperspy.api as hs
    from fastapi.testclient import TestClient

    from si_viewer.app import create_app

    data = np.arange(12 * 10 * 50, dtype=np.float32).reshape(12, 10, 50)  # y, x, energy
    s = hs.signals.Signal1D(data)
    s.set_signal_type("EELS")
    s.axes_manager[0].scale = s.axes_manager[1].scale = 0.5
    s.axes_manager[-1].offset, s.axes_manager[-1].scale, s.axes_manager[-1].units = 100.0, 2.0, "eV"
    s.metadata.General.title = "SI"
    s.save(tmp_path / "si.hspy")
    c = TestClient(create_app())
    c.si = c.post("/api/files", json={"path": str(tmp_path / "si.hspy")}).json()["datasets"][0]
    c.data = data
    return c


def test_crop_to_a_rectangle_and_a_window(client):
    si, session = client.si, client.app.state.session
    # Pixels whose centres are inside: columns 2–5, rows 3–7.
    region = {"kind": "rect", "x0": 1.6, "y0": 2.6, "x1": 6.4, "y1": 8.2}
    r = client.post(f"/api/datasets/{si['id']}/crop", json={"region": region})
    assert r.status_code == 200, r.text
    made = session.get_dataset(r.json()["id"])
    assert made.origin == "cropped" and made.parent_id == si["id"] and made.kind.value == "spectrum_image"
    np.testing.assert_array_equal(made.signal.data, client.data[3:8, 2:6])
    nav = made.signal.axes_manager.navigation_axes
    assert (nav[0].offset, nav[1].offset) == (1.0, 1.5)  # calibrated where it starts
    assert made.title == "SI (x 2–5, y 3–7)" and made.summary()["eels"] is not None

    # An energy window: 110–130 eV, channels 5–15.
    r = client.post(f"/api/datasets/{si['id']}/crop", json={"lo": 110, "hi": 130})
    made = session.get_dataset(r.json()["id"])
    np.testing.assert_array_equal(made.signal.data, client.data[:, :, 5:16])
    assert made.signal.axes_manager.signal_axes[0].offset == 110 and made.title == "SI (110–130 eV)"

    # Both at once; and the result can be deleted.
    r = client.post(f"/api/datasets/{si['id']}/crop", json={"region": region, "lo": 110, "hi": 130})
    made = session.get_dataset(r.json()["id"])
    np.testing.assert_array_equal(made.signal.data, client.data[3:8, 2:6, 5:16])
    assert client.delete(f"/api/datasets/{made.id}").status_code == 200


def test_bad_crops(client):
    si = client.si
    url = f"/api/datasets/{si['id']}/crop"
    assert client.post(url, json={}).status_code == 422
    assert client.post(url, json={"region": {"kind": "ellipse", "x0": 1, "y0": 1, "x1": 4, "y1": 4}}).status_code == 422
    assert client.post(url, json={"region": {"kind": "rect", "x0": 20, "y0": 20, "x1": 30, "y1": 30}}).status_code == 422
    assert client.post(url, json={"lo": 120, "hi": 120}).status_code == 422


def _client_with(tmp_path, signal, name):
    from fastapi.testclient import TestClient

    from si_viewer.app import create_app

    signal.save(tmp_path / name)
    c = TestClient(create_app())
    return c, c.post("/api/files", json={"path": str(tmp_path / name)}).json()["datasets"][0]


def test_crop_an_image_and_a_stack(tmp_path):
    import hyperspy.api as hs

    image = np.arange(20 * 30, dtype=np.float32).reshape(20, 30)
    client, ds = _client_with(tmp_path, hs.signals.Signal2D(image), "im.hspy")
    region = {"kind": "rect", "x0": 4, "y0": 2, "x1": 10, "y1": 7}  # columns 4–9, rows 2–6
    r = client.post(f"/api/datasets/{ds['id']}/crop", json={"region": region})
    assert r.status_code == 200, r.text
    made = client.app.state.session.get_dataset(r.json()["id"])
    assert made.kind.value == "image" and made.parent_id == ds["id"]
    np.testing.assert_array_equal(made.signal.data, image[2:7, 4:10])
    assert client.post(f"/api/datasets/{ds['id']}/crop", json={"lo": 1, "hi": 2}).status_code == 422

    stack = np.arange(4 * 20 * 30, dtype=np.float32).reshape(4, 20, 30)
    client, ds = _client_with(tmp_path, hs.signals.Signal2D(stack), "st.hspy")
    made = client.app.state.session.get_dataset(
        client.post(f"/api/datasets/{ds['id']}/crop", json={"region": region}).json()["id"])
    assert made.kind.value == "stack"
    np.testing.assert_array_equal(made.signal.data, stack[:, 2:7, 4:10])  # every frame kept


def test_crop_a_spectrum_to_a_window(tmp_path):
    import hyperspy.api as hs

    s = hs.signals.Signal1D(np.arange(50, dtype=np.float32))
    s.set_signal_type("EELS")
    s.axes_manager[0].offset, s.axes_manager[0].scale = 100.0, 2.0
    client, ds = _client_with(tmp_path, s, "sp.hspy")
    made = client.app.state.session.get_dataset(
        client.post(f"/api/datasets/{ds['id']}/crop", json={"lo": 110, "hi": 130}).json()["id"])
    np.testing.assert_array_equal(made.signal.data, np.arange(5, 16))
    assert client.post(f"/api/datasets/{ds['id']}/crop", json={"region": {"kind": "rect"}}).status_code == 422


def test_crop_an_in_situ_series_keeps_its_frames(tmp_path, fake_dm4):
    from fastapi.testclient import TestClient

    from si_viewer.app import create_app
    from test_insitu import make_series

    dims, n = [5, 3, 4], 5  # x, y, E
    path, frames = make_series(tmp_path, fake_dm4, "SI", dims, 2, "<f4", n)
    client = TestClient(create_app())
    si = next(d for d in client.post("/api/files", json={"path": str(path)}).json()["datasets"] if d["insitu"])
    session = client.app.state.session
    region = {"kind": "rect", "x0": 1, "y0": 0, "x1": 4, "y1": 2}  # columns 1–3, rows 0–1
    axis = session.get_dataset(si["id"]).signal.axes_manager.signal_axes[0]
    lo, hi = axis.offset + axis.scale, axis.offset + 2 * axis.scale  # channels 1–2
    r = client.post(f"/api/datasets/{si['id']}/crop", json={"region": region, "lo": lo, "hi": hi})
    assert r.status_code == 200, r.text
    made = session.get_dataset(r.json()["id"])
    assert made.insitu is not None and made.insitu.n == n and made.parent_id == si["id"]
    want = lambda k: frames[k].transpose(1, 2, 0)[0:2, 1:4, 1:3]  # noqa: E731
    np.testing.assert_array_equal(np.asarray(made.signal.data), want(0))
    assert client.put(f"/api/datasets/{made.id}/insitu/frame", json={"frame": 3}).status_code == 200
    np.testing.assert_array_equal(np.asarray(made.signal.data), want(3))

    # Saved as its crop, and remade from the raw files on open.
    saved = client.post("/api/project/save", json={"path": str(tmp_path / "a.hspy"), "views": {}}).json()["path"]
    other = TestClient(create_app())
    assert other.post("/api/project/open", json={"path": saved}).status_code == 200
    again = other.app.state.session.get_dataset(made.id)
    assert again.insitu is not None and again.insitu.crop == made.insitu.crop
    np.testing.assert_array_equal(np.asarray(again.signal.data), want(3))
