import pytest
from fastapi.testclient import TestClient

from si_viewer.app import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert "hyperspy" in body["versions"]


def test_open_missing_file(client):
    assert client.post("/api/files", json={"path": "/no/such/file.emd"}).status_code == 404


def test_open_list_close(client, example_emd):
    r = client.post("/api/files", json={"path": str(example_emd)})
    assert r.status_code == 200
    entry = r.json()
    assert entry["name"] == example_emd.name
    kinds = {d["kind"] for d in entry["datasets"]}
    assert "spectrum_image" in kinds

    assert [f["id"] for f in client.get("/api/files").json()] == [entry["id"]]

    ds = entry["datasets"][0]
    assert client.get(f"/api/datasets/{ds['id']}").json()["id"] == ds["id"]
    meta = client.get(f"/api/datasets/{ds['id']}/metadata").json()
    assert "General" in meta["metadata"]

    assert client.delete(f"/api/files/{entry['id']}").status_code == 204
    assert client.get("/api/files").json() == []
