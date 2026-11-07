"""Fitting tests. exSpy's FePt nanoparticle spectrum has good counts, so the
fit itself is tested on it; the example .emd covers spectrum-image sources."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from si_viewer.app import create_app


@pytest.fixture(scope="module")
def fept_path() -> Path:
    import exspy.data

    path = Path(exspy.data.__file__).parent / "EDS_TEM_FePt_nanoparticles.hspy"
    if not path.is_file():
        pytest.skip("exSpy's example data is not installed")
    return path


@pytest.fixture
def client():
    return TestClient(create_app())


def test_fit_fept_spectrum(client, fept_path):
    spec = client.post("/api/files", json={"path": str(fept_path)}).json()["datasets"][0]
    assert spec["kind"] == "spectrum" and spec["eds"]["elements"] == ["Fe", "Pt"]
    # The grid's Cu is in the spectrum: leaving it out distorts the fit.
    client.put(f"/api/datasets/{spec['id']}/elements", json={"elements": ["Cu", "Fe", "Pt"]})

    r = client.post(f"/api/datasets/{spec['id']}/fit", json={})
    assert r.status_code == 200, r.text
    fit = r.json()
    n = len(fit["x"])
    assert fit["x"][0] >= 0.5  # the noise peak is skipped
    assert len(fit["data"]) == len(fit["model"]) == len(fit["background"]) == n
    assert set(fit["elements"]) == {"Cu", "Fe", "Pt"} and all(len(c) == n for c in fit["elements"].values())

    lines = {l["line"]: l for l in fit["lines"]}
    assert {l for l, row in lines.items() if row["analysis"]} == {"Cu_Ka", "Fe_Ka", "Pt_La"}
    assert all(row["intensity"] >= 0 for row in lines.values())  # bounded fit
    assert lines["Pt_La"]["intensity"] > 10 * 1000 and lines["Fe_Ka"]["intensity"] > 1000
    assert 120 < fit["resolution_MnKa"] < 150
    assert fit["residual_ratio"] < 0.15 and fit["pixels"] == 1 and fit["skipped"] == []


def test_fit_options(client, fept_path):
    spec = client.post("/api/files", json={"path": str(fept_path)}).json()["datasets"][0]
    fit = client.post(
        f"/api/datasets/{spec['id']}/fit",
        json={"options": {"min_energy": 2.0, "calibrate_resolution": False, "calibrate_energy": False}},
    ).json()
    assert fit["x"][0] >= 2.0 - 0.02  # the channel containing 2 keV
    # Fe Lα (0.70 keV) is below the fitted range now.
    assert "Fe_La" not in {l["line"] for l in fit["lines"]}

    client.put(f"/api/datasets/{spec['id']}/elements", json={"elements": []})
    assert client.post(f"/api/datasets/{spec['id']}/fit", json={}).status_code == 422


def test_fit_spectrum_image_sources(client, example_emd):
    entry = client.post("/api/files", json={"path": str(example_emd)}).json()
    si = next(d for d in entry["datasets"] if d["kind"] == "spectrum_image")
    url = f"/api/datasets/{si['id']}/fit"

    region = {"kind": "rect", "x0": 0, "y0": 0, "x1": 64, "y1": 64}
    fit = client.post(url, json={"source": {"type": "region", "region": region}}).json()
    assert fit["pixels"] == 64 * 64 and len(fit["model"]) == len(fit["x"])
    assert fit["skipped"] == ["C"]  # C Kα (0.28 keV) is below the default 0.5 keV
    assert client.post(url, json={"source": {"type": "pixel", "x": 10, "y": 10, "size": 5}}).json()["pixels"] == 25
    assert client.post(url, json={"source": {"type": "all"}}).json()["pixels"] == 256 * 256
    empty = {"kind": "rect", "x0": 500, "y0": 500, "x1": 510, "y1": 510}
    assert client.post(url, json={"source": {"type": "region", "region": empty}}).status_code == 422
