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


def test_only_spectrum_images_rebin(client, tmp_path):
    tifffile.imwrite(tmp_path / "s.tif", np.zeros((6, 8, 10), "uint16"))
    stack = client.post("/api/files", json={"path": str(tmp_path / "s.tif")}).json()["datasets"][0]
    assert client.post(f"/api/datasets/{stack['id']}/rebin", json={"factors": [1, 2, 2]}).status_code == 400
    entry = client.post("/api/examples/fept-si").json()
    for d in entry["datasets"]:
        if d["kind"] != "spectrum_image":  # HAADF and element maps
            r = client.post(f"/api/datasets/{d['id']}/rebin", json={"factors": [2, 2]})
            assert r.status_code == 400, d["title"]
    # Only element maps can be deleted.
    assert client.delete(f"/api/datasets/{stack['id']}").status_code == 400


def test_rebin_takes_the_whole_scan(client):
    entry = client.post("/api/examples/fept-si").json()
    si = next(d for d in entry["datasets"] if d["kind"] == "spectrum_image")
    haadf = next(d for d in entry["datasets"] if d["kind"] == "image" and d["origin"] == "file")
    raw = _array(client.get(f"/api/datasets/{haadf['id']}/frame"))

    r = client.post(f"/api/datasets/{si['id']}/rebin", json={"factors": [2, 4, 1]}).json()
    datasets = {d["id"]: d for d in r["file"]["datasets"]}
    assert datasets[si["id"]]["navigation_shape"] == [16, 32]  # (x, y)
    # The HAADF on the same scan is binned the same way, so it still lines up.
    h = datasets[haadf["id"]]
    assert h["signal_shape"] == [16, 32] and h["revision"] == haadf["revision"] + 1 and h["rebinned"]
    np.testing.assert_allclose(
        _array(client.get(f"/api/datasets/{haadf['id']}/frame")),
        raw.reshape(32, 2, 16, 4).sum(axis=(1, 3)), rtol=1e-5,
    )
    assert haadf["id"] in [s["id"] for s in client.get(f"/api/datasets/{si['id']}/navigators").json()]
    for d in datasets.values():
        if d["kind"] == "image":
            assert d["signal_shape"] == [16, 32], d["title"]

    # Factors now apply to the rebinned sizes.
    assert client.post(f"/api/datasets/{si['id']}/rebin", json={"factors": [1, 1, 1]}).status_code == 422
    assert client.post(f"/api/datasets/{si['id']}/rebin", json={"factors": [1, 17, 1]}).status_code == 422
    assert client.post(f"/api/datasets/{si['id']}/rebin", json={"factors": [2, 2]}).status_code == 422


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
