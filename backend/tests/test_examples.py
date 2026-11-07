import numpy as np
import pytest
from fastapi.testclient import TestClient

from si_viewer.app import create_app
from si_viewer.examples import EXAMPLES


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def test_list_examples(client):
    listed = client.get("/api/examples").json()
    assert [e["id"] for e in listed] == [e.id for e in EXAMPLES]
    assert all(e["title"] and e["description"] for e in listed)
    assert client.post("/api/examples/nope").status_code == 404


@pytest.mark.parametrize(
    "example_id, kinds",
    [
        ("fept-spectrum", ["spectrum"]),
        ("tm002-spectrum", ["spectrum"]),
        ("atomic-image", ["image"]),
        ("two-gaussians", ["spectrum_image"]),
    ],
)
def test_open_examples(client, example_id, kinds):
    entry = client.post(f"/api/examples/{example_id}").json()
    assert entry["path"] == f"example:{example_id}"
    assert entry["name"] == next(e.title for e in EXAMPLES if e.id == example_id)
    assert [d["kind"] for d in entry["datasets"]] == kinds
    # Opening again returns the same entry.
    assert client.post(f"/api/examples/{example_id}").json()["id"] == entry["id"]


def test_synthetic_fept_si(client):
    entry = client.post("/api/examples/fept-si").json()
    by_kind = {}
    for d in entry["datasets"]:
        by_kind.setdefault(d["kind"], []).append(d)
    si = by_kind["spectrum_image"][0]
    assert si["navigation_shape"] == [64, 64] and si["eds"]["elements"] == ["Cu", "Fe", "Pt"]
    assert si["axes"][0]["units"] == "nm" and si["axes"][2]["units"] == "keV"
    # HAADF image and raw element maps for Fe, Pt and Cu.
    images = by_kind["image"]
    assert any(d["title"] == "HAADF" for d in images)
    assert {d["element"] for d in images if d["is_element_map"]} == {"Cu", "Fe", "Pt"}

    # Pt-rich cores and Fe-rich shells: the Pt/Fe ratio is higher at a
    # particle's centre than in its shell (particle at x=40, y=45, r=13).
    def ratio(region):
        fit = client.post(f"/api/datasets/{si['id']}/fit", json={"source": {"type": "region", "region": region}}).json()
        lines = {l["line"]: l["intensity"] for l in fit["lines"]}
        return lines["Pt_La"] / lines["Fe_Ka"]

    core = ratio({"kind": "ellipse", "x0": 36, "y0": 41, "x1": 44, "y1": 49})
    shell = ratio({"kind": "rect", "x0": 49, "y0": 40, "x1": 52, "y1": 50})
    assert core > 2 * shell


def test_synthetic_si_is_reproducible():
    from si_viewer.examples import _synthetic_fept_si_data

    a, _, _ = _synthetic_fept_si_data()
    _synthetic_fept_si_data.cache_clear()
    b, _, _ = _synthetic_fept_si_data()
    np.testing.assert_array_equal(a, b)
