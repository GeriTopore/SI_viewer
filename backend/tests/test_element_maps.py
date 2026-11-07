import numpy as np
import pytest
from fastapi.testclient import TestClient

from si_viewer.app import create_app
from si_viewer.processing.element_maps import (
    MapOptions, analysis_line, line_intensity, line_label, line_windows,
)
from si_viewer.session import DatasetKind, Session


@pytest.fixture(scope="module")
def si_signal(example_emd):
    session = Session()
    entry = session.open_file(example_emd)
    si = next(session.datasets[d] for d in entry.dataset_ids
              if session.datasets[d].kind is DatasetKind.SPECTRUM_IMAGE)
    si.materialise()
    return si.signal


def test_line_label():
    assert line_label("Fe_Ka") == "Fe Kα" and line_label("Bi_La") == "Bi Lα"


def test_analysis_line(si_signal):
    assert analysis_line(si_signal, "Fe") == "Fe_Ka"
    assert analysis_line(si_signal, "Bi") == "Bi_La"  # Bi Kα is beyond the 20 keV range
    assert analysis_line(si_signal, "H") is None  # below the usable energy range


@pytest.mark.parametrize("background", [False, True])
def test_matches_exspy(si_signal, background):
    """The NumPy map equals exSpy's get_lines_intensity."""
    opts = MapOptions(background=background)
    integration, bg = line_windows(si_signal, "Fe_Ka", opts)
    ours = line_intensity(si_signal, integration, bg)
    theirs = si_signal.get_lines_intensity(
        xray_lines=["Fe_Ka"], integration_windows=[integration],
        background_windows=[bg] if bg else None,
    )[0].data
    np.testing.assert_allclose(ours, theirs, rtol=1e-5, atol=1e-4)


@pytest.fixture
def client_si(example_emd):
    client = TestClient(create_app())
    entry = client.post("/api/files", json={"path": str(example_emd)}).json()
    si = next(d for d in entry["datasets"] if d["kind"] == "spectrum_image")
    return client, si


def _maps(file_summary, origin=None):
    return [d for d in file_summary["datasets"] if d["is_element_map"] and (origin is None or d["origin"] == origin)]


def _map(file_summary, element):
    return next(m for m in _maps(file_summary) if m["element"] == element)


FILE_ELEMENTS = {"Bi", "C", "Mn", "O", "Ti", "Cu", "Fe"}


def test_raw_maps_on_open(client_si, example_emd):
    """Velox's processed maps are replaced by raw maps computed from the SI."""
    client, si = client_si
    file = client.get("/api/files").json()[0]
    assert _maps(file, "file") == []
    computed = _maps(file, "computed")
    assert {m["element"] for m in computed} == FILE_ELEMENTS
    assert all(m["parent_id"] == si["id"] for m in computed)
    assert _map(file, "Bi")["title"] == "Bi Lα"
    # Raw net counts (not weight fractions): Fe Kα averages ~0.7 counts/pixel.
    fe = np.frombuffer(client.get(f"/api/datasets/{_map(file, 'Fe')['id']}/frame").content, "<f4")
    assert 0.3 < fe.mean() < 2


def test_add_and_untick_element(client_si):
    client, si = client_si
    elements = si["eds"]["elements"]

    r = client.put(f"/api/datasets/{si['id']}/elements", json={"elements": elements + ["Ga"]}).json()
    ga = _map(r["file"], "Ga")
    assert (ga["title"], ga["origin"], ga["parent_id"]) == ("Ga Kα", "computed", si["id"])
    assert ga["signal_shape"] == si["navigation_shape"] and r["warnings"] == []
    assert client.get(f"/api/datasets/{ga['id']}/frame").status_code == 200

    r = client.put(f"/api/datasets/{si['id']}/elements", json={"elements": elements}).json()
    assert {m["element"] for m in _maps(r["file"])} == FILE_ELEMENTS


def test_element_without_line_warns(client_si):
    client, si = client_si
    r = client.put(f"/api/datasets/{si['id']}/elements", json={"elements": si["eds"]["elements"] + ["H"]}).json()
    assert "H" in r["dataset"]["eds"]["elements"]
    assert {m["element"] for m in _maps(r["file"])} == FILE_ELEMENTS and r["warnings"]


def test_delete_map_unticks_element(client_si):
    client, si = client_si
    file = client.get("/api/files").json()[0]

    file = client.delete(f"/api/datasets/{_map(file, 'Fe')['id']}").json()["file"]
    assert all(m["element"] != "Fe" for m in _maps(file))
    for d in file["datasets"]:
        if d["eds"]:
            assert "Fe" not in d["eds"]["elements"]

    # Re-ticking brings the map back.
    si_now = next(d for d in file["datasets"] if d["id"] == si["id"])
    r = client.put(f"/api/datasets/{si['id']}/elements", json={"elements": si_now["eds"]["elements"] + ["Fe"]}).json()
    assert _map(r["file"], "Fe")["title"] == "Fe Kα"

    # Only element maps can be deleted.
    assert client.delete(f"/api/datasets/{si['id']}").status_code == 400


def test_map_options_recompute(client_si):
    client, si = client_si
    file = client.get("/api/files").json()[0]
    fe = _map(file, "Fe")
    before = np.frombuffer(client.get(f"/api/datasets/{fe['id']}/frame").content, "<f4")

    r = client.put(f"/api/datasets/{si['id']}/map-options", json={"background": False, "smoothing": 3}).json()
    assert r["dataset"]["eds"]["map_options"] == {"background": False, "width": 2.0, "smoothing": 3}
    fe2 = _map(r["file"], "Fe")
    assert fe2["id"] == fe["id"] and fe2["revision"] == fe["revision"] + 1
    after = np.frombuffer(client.get(f"/api/datasets/{fe['id']}/frame").content, "<f4")
    assert after.std() < before.std()  # smoothed
    assert client.put(f"/api/datasets/{si['id']}/map-options", json={"smoothing": 0}).status_code == 422
