"""Undoing a deletion: the datasets come back in their places (a computed
element map with its element)."""

import pytest
from fastapi.testclient import TestClient

from si_viewer.app import create_app

from test_thickness import _open, _si, _wait

BACKGROUND = {"model": "power_law", "order": 2, "fast": True, "background": [800, 900], "signal": [900, 1100]}


@pytest.fixture
def client():
    return TestClient(create_app())


def _fept(client):
    entry = client.post("/api/examples/fept-si").json()
    si = next(d for d in entry["datasets"] if d["kind"] == "spectrum_image")
    return entry, si


def _delete(client, ds_id) -> dict:
    r = client.delete(f"/api/datasets/{ds_id}")
    assert r.status_code == 200, r.text
    return r.json()


def _undo(client, token) -> dict:
    r = client.post(f"/api/undo/{token}")
    assert r.status_code == 200, r.text
    return r.json()


def _ids(file):
    return [d["id"] for d in file["datasets"]]


def test_window_map_comes_back_in_its_place(client):
    entry, si = _fept(client)
    w = client.post(f"/api/datasets/{si['id']}/window-maps", json={"lo": 6.2, "hi": 6.6, "label": None}).json()["dataset"]
    before = _ids(client.get("/api/files").json()[0])
    gone = _delete(client, w["id"])
    assert w["id"] not in _ids(gone["file"])
    back = _undo(client, gone["undo"])["files"][0]
    assert _ids(back) == before
    # A token is used once.
    assert client.post(f"/api/undo/{gone['undo']}").status_code == 404


def test_computed_element_map_comes_back_with_its_element(client):
    entry, si = _fept(client)
    fe = next(d for d in entry["datasets"] if d["element"] == "Fe")
    gone = _delete(client, fe["id"])
    assert "Fe" not in next(d for d in gone["file"]["datasets"] if d["id"] == si["id"])["eds"]["elements"]
    back = _undo(client, gone["undo"])["files"][0]
    assert "Fe" in next(d for d in back["datasets"] if d["id"] == si["id"])["eds"]["elements"]
    assert any(d["element"] == "Fe" and d["origin"] == "computed" for d in back["datasets"])


def test_a_result_comes_back_with_what_was_made_from_it(client, eels_cuzn):
    si = _si(_open(client, eels_cuzn))
    r = client.post(f"/api/datasets/{si['id']}/background", json={"options": BACKGROUND})
    bg = _wait(client, r.json())["result"]["dataset_id"]
    client.post(f"/api/datasets/{bg}/window-maps", json={"lo": 930, "hi": 960, "label": None})
    before = _ids(client.get("/api/files").json()[0])
    made = [d for d in client.get("/api/files").json()[0]["datasets"] if d["parent_id"] == bg]
    assert len(made) == 2  # the exponent map and the window map
    gone = _delete(client, bg)
    assert set(_ids(gone["file"])) == set(before) - {bg, *(d["id"] for d in made)}
    back = _undo(client, gone["undo"])["files"][0]
    assert _ids(back) == before


def test_closing_the_file_drops_its_undo(client):
    entry, si = _fept(client)
    w = client.post(f"/api/datasets/{si['id']}/window-maps", json={"lo": 6.2, "hi": 6.6, "label": None}).json()["dataset"]
    gone = _delete(client, w["id"])
    client.delete(f"/api/files/{entry['id']}")
    assert client.post(f"/api/undo/{gone['undo']}").status_code == 404
    assert client.post("/api/undo/nope").status_code == 404
