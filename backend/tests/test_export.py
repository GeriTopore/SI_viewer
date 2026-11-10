import io
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from si_viewer.app import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


def _load(tmp_path, r, name):
    import hyperspy.api as hs

    assert r.status_code == 200, r.text
    path = tmp_path / name
    path.write_bytes(r.content)
    return hs.load(path)


def test_export_formats(client):
    formats = client.get("/api/export-formats").json()
    assert [f["format"] for f in formats["spectrum"]] == ["msa", "csv", "hspy"]
    assert [f["format"] for f in formats["spectrum_image"]] == ["hspy"]


def test_export_map_as_shown(client, tmp_path):
    entry = client.post("/api/examples/fept-si").json()
    si = next(d for d in entry["datasets"] if d["kind"] == "spectrum_image")
    fe = next(d for d in entry["datasets"] if d["element"] == "Fe")
    client.put(f"/api/datasets/{si['id']}/filters", json={"spatial": "gaussian", "sigma": 1})
    shown = client.get(f"/api/datasets/{fe['id']}/frame")
    shown = np.frombuffer(shown.content, "<f4").reshape(64, 64)

    r = client.get(f"/api/datasets/{fe['id']}/export", params={"format": "tiff"})
    assert "Fe_K" in r.headers["content-disposition"]
    tif = _load(tmp_path, r, "fe.tif")
    np.testing.assert_allclose(tif.data, shown)
    assert tif.axes_manager[0].scale == pytest.approx(si["axes"][1]["scale"])
    csv = client.get(f"/api/datasets/{fe['id']}/export", params={"format": "csv"}).text
    np.testing.assert_allclose(np.loadtxt(io.StringIO(csv), delimiter=","), shown, rtol=1e-6)

    # The spectrum image is saved as loaded, its processing noted.
    h = _load(tmp_path, client.get(f"/api/datasets/{si['id']}/export", params={"format": "hspy"}), "si.hspy")
    assert h.data.shape == (64, 64, si["signal_shape"][0])
    assert "Gaussian" in h.metadata.Processing.si_viewer.filters_not_applied
    assert client.get(f"/api/datasets/{si['id']}/export", params={"format": "tiff"}).status_code == 422

    # All maps in one zip.
    z = zipfile.ZipFile(io.BytesIO(client.get(f"/api/files/{entry['id']}/maps").content))
    assert sorted(z.namelist()) == ["Cu_Kα.tif", "Fe_Kα.tif", "Pt_Lα.tif"]


def test_export_spectrum(client, tmp_path):
    entry = client.post("/api/examples/fept-spectrum").json()
    spec = entry["datasets"][0]
    msa = _load(tmp_path, client.get(f"/api/datasets/{spec['id']}/export", params={"format": "msa"}), "s.msa")
    csv = client.get(f"/api/datasets/{spec['id']}/export", params={"format": "csv"}).text
    table = np.loadtxt(io.StringIO(csv), delimiter=",", skiprows=1)
    np.testing.assert_allclose(table[:, 1], msa.data, rtol=1e-6)
    assert table[1, 0] - table[0, 0] == pytest.approx(msa.axes_manager[0].scale)


def test_save_to_a_path(client, tmp_path):
    entry = client.post("/api/examples/fept-si").json()
    fe = next(d for d in entry["datasets"] if d["element"] == "Fe")
    target = tmp_path / "iron map.tif"
    r = client.post(f"/api/datasets/{fe['id']}/save", json={"format": "tiff", "path": str(target)})
    assert r.status_code == 200 and r.json()["path"] == str(target)
    downloaded = client.get(f"/api/datasets/{fe['id']}/export", params={"format": "tiff"}).content
    assert target.read_bytes() == downloaded
    # An existing file needs overwrite; a missing folder or a bad format is refused.
    again = {"format": "tiff", "path": str(target)}
    assert client.post(f"/api/datasets/{fe['id']}/save", json=again).status_code == 409
    assert client.post(f"/api/datasets/{fe['id']}/save", json={**again, "overwrite": True}).status_code == 200
    missing = {"format": "tiff", "path": str(tmp_path / "nope" / "x.tif")}
    assert client.post(f"/api/datasets/{fe['id']}/save", json=missing).status_code == 404
    assert client.post(f"/api/datasets/{fe['id']}/save", json={**again, "format": "msa"}).status_code == 422


def test_batch_export(client, tmp_path):
    entry = client.post("/api/examples/fept-si").json()
    maps = [d for d in entry["datasets"] if d["is_element_map"]]
    body = {"ids": [m["id"] for m in maps], "format": "tiff", "folder": str(tmp_path)}
    r = client.post("/api/export/batch", json=body)
    assert r.status_code == 200, r.text
    assert sorted(p.name for p in tmp_path.iterdir()) == ["Cu_Kα.tif", "Fe_Kα.tif", "Pt_Lα.tif"]
    assert client.post("/api/export/batch", json=body).status_code == 409
    assert client.post("/api/export/batch", json={**body, "overwrite": True}).status_code == 200
    assert client.post("/api/export/batch", json={**body, "folder": str(tmp_path / "no")}).status_code == 404


def test_listing_files_of_one_extension(client, tmp_path):
    (tmp_path / "a.tif").write_bytes(b"x")
    (tmp_path / "b.csv").write_bytes(b"x")
    names = [e["name"] for e in client.get("/api/fs/list", params={"path": str(tmp_path), "ext": "csv"}).json()["entries"]]
    assert names == ["b.csv"]
