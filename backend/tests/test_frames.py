import numpy as np
import pytest
import tifffile
from fastapi.testclient import TestClient

from si_viewer.app import create_app
from si_viewer.processing.frames import bin2d


def _array(r) -> np.ndarray:
    assert r.status_code == 200, r.text
    shape = tuple(int(n) for n in r.headers["X-Array-Shape"].split(","))
    return np.frombuffer(r.content, dtype="<f4").reshape(shape)


@pytest.fixture
def client():
    return TestClient(create_app())


@pytest.fixture
def stack_path(tmp_path):
    # 5 frames: a leading size of 3 or 4 would be read back as RGB(A) channels.
    data = np.arange(5 * 6 * 8, dtype="uint16").reshape(5, 6, 8)
    path = tmp_path / "stack.tif"
    tifffile.imwrite(path, data)
    return path, data


def test_bin2d():
    a = np.arange(5 * 7, dtype=float).reshape(5, 7)
    b = bin2d(a, 2)
    assert b.shape == (2, 3)
    assert b[0, 0] == a[:2, :2].mean()


def test_stack_frames(client, stack_path):
    path, data = stack_path
    ds = client.post("/api/files", json={"path": str(path)}).json()["datasets"][0]
    url = f"/api/datasets/{ds['id']}/frame"

    np.testing.assert_array_equal(_array(client.get(url, params={"index": 2})), data[2])
    np.testing.assert_allclose(_array(client.get(url, params={"projection": "mean"})), data.mean(0))
    np.testing.assert_allclose(
        _array(client.get(url, params={"index": 1, "bin": 2})), bin2d(data[1].astype(float), 2)
    )
    assert client.get(url, params={"index": 5}).status_code == 416
    assert client.get(f"/api/datasets/{ds['id']}/spectrum").status_code == 400


def test_calibrate_axes(client, stack_path):
    path, _ = stack_path
    ds = client.post("/api/files", json={"path": str(path)}).json()["datasets"][0]
    r = client.patch(
        f"/api/datasets/{ds['id']}/axes",
        json=[{"index": 1, "scale": 0.25, "units": "nm"}, {"index": 2, "scale": 0.25, "units": "nm"}],
    )
    axes = r.json()["axes"]
    assert [(a["scale"], a["units"]) for a in axes[1:]] == [(0.25, "nm"), (0.25, "nm")]
    bad = client.patch(f"/api/datasets/{ds['id']}/axes", json=[{"index": 1, "scale": 0}])
    assert bad.status_code == 422


def test_spectrum_image_views(client, example_emd):
    entry = client.post("/api/files", json={"path": str(example_emd)}).json()
    si = next(d for d in entry["datasets"] if d["kind"] == "spectrum_image")
    base = f"/api/datasets/{si['id']}"

    assert client.post(f"{base}/load").json()["lazy"] is False

    total = _array(client.get(f"{base}/frame"))
    assert total.shape == tuple(si["navigation_shape"][::-1])
    spectrum = _array(client.get(f"{base}/spectrum"))
    assert spectrum.shape == tuple(si["signal_shape"])
    # Every count lands in exactly one pixel and one channel.
    assert np.isclose(total.sum(dtype=np.float64), spectrum.sum(dtype=np.float64), rtol=1e-5)

    sources = client.get(f"{base}/navigators").json()
    labels = [s["label"] for s in sources]
    assert "Total counts" in labels
    assert "DF" in labels[0].upper()  # companion ADF image first
