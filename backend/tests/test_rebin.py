import numpy as np
import pytest
import tifffile
from fastapi.testclient import TestClient

from si_viewer.app import create_app


def _array(r) -> np.ndarray:
    assert r.status_code == 200, r.text
    shape = tuple(int(n) for n in r.headers["X-Array-Shape"].split(","))
    return np.frombuffer(r.content, "<f4").reshape(shape)


@pytest.fixture
def client():
    return TestClient(create_app())


def test_rebin_stack_in_place(client, tmp_path):
    data = np.arange(6 * 8 * 10, dtype="uint16").reshape(6, 8, 10)
    tifffile.imwrite(tmp_path / "s.tif", data)
    ds = client.post("/api/files", json={"path": str(tmp_path / "s.tif")}).json()["datasets"][0]

    # Axis order as in the summary: frames, y (height), x (width).
    r = client.post(f"/api/datasets/{ds['id']}/rebin", json={"factors": [1, 2, 5]}).json()
    new = r["dataset"]
    assert new["id"] == ds["id"] and new["revision"] == ds["revision"] + 1
    assert new["title"] == ds["title"] and new["rebinned"] == ["width×5, height×2"]
    assert new["navigation_shape"] == [6] and new["signal_shape"] == [2, 4]  # (x, y)
    assert len(r["file"]["datasets"]) == 1  # replaced, not copied
    expected = data[3].reshape(4, 2, 2, 5).sum(axis=(1, 3))
    np.testing.assert_array_equal(_array(client.get(f"/api/datasets/{ds['id']}/frame", params={"index": 3})), expected)

    # Factors now apply to the rebinned sizes.
    assert client.post(f"/api/datasets/{ds['id']}/rebin", json={"factors": [1, 1, 1]}).status_code == 422
    assert client.post(f"/api/datasets/{ds['id']}/rebin", json={"factors": [1, 5, 1]}).status_code == 422
    assert client.post(f"/api/datasets/{ds['id']}/rebin", json={"factors": [2, 2]}).status_code == 422
    # Only element maps can be deleted.
    assert client.delete(f"/api/datasets/{ds['id']}").status_code == 400


def test_rebin_spectrum_image_in_place(client, example_emd):
    entry = client.post("/api/files", json={"path": str(example_emd)}).json()
    si = next(d for d in entry["datasets"] if d["kind"] == "spectrum_image")
    maps = {d["element"]: d for d in entry["datasets"] if d["parent_id"] == si["id"]}
    total = _array(client.get(f"/api/datasets/{si['id']}/spectrum"))

    r = client.post(f"/api/datasets/{si['id']}/rebin", json={"factors": [2, 2, 1]}).json()
    new = r["dataset"]
    assert new["id"] == si["id"] and new["navigation_shape"] == [128, 128]
    assert new["axes"][0]["scale"] == pytest.approx(2 * si["axes"][0]["scale"])
    assert new["eds"]["elements"] == si["eds"]["elements"]
    assert len(r["file"]["datasets"]) == len(entry["datasets"])  # nothing added
    # Counts are conserved.
    np.testing.assert_allclose(_array(client.get(f"/api/datasets/{si['id']}/spectrum")), total, rtol=1e-6)
    # Total-counts map at the new size.
    assert _array(client.get(f"/api/datasets/{si['id']}/frame")).shape == (128, 128)

    # Its maps are recomputed in place at the new size.
    after = {d["element"]: d for d in r["file"]["datasets"] if d["parent_id"] == si["id"]}
    assert after.keys() == maps.keys()
    for el, m in after.items():
        assert m["id"] == maps[el]["id"] and m["revision"] == maps[el]["revision"] + 1
        assert m["signal_shape"] == [128, 128] and m["title"] == maps[el]["title"]
