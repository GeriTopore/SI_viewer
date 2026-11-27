import numpy as np
import pytest
from fastapi.testclient import TestClient

from si_viewer import examples
from si_viewer.app import create_app
from si_viewer.examples import EXAMPLES


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def test_list_examples(client):
    listed = client.get("/api/examples").json()
    # Every example, listed by group (see test_examples_are_grouped_in_order).
    assert sorted(e["id"] for e in listed) == sorted(e.id for e in EXAMPLES)
    assert all(e["title"] and e["description"] and e["group"] for e in listed)
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


# ---------- the menu: grouped by use, pyxem's 4D-STEM data among them ----------
def test_examples_are_grouped_in_order():
    listed = examples.list_examples()
    groups = [e["group"] for e in listed]
    assert set(groups) <= set(examples.GROUPS)
    # Each group's examples together, the groups in the menu's order.
    firsts = list(dict.fromkeys(groups))
    assert firsts == [g for g in examples.GROUPS if g in firsts]
    assert len(firsts) == len([g for i, g in enumerate(groups) if i == 0 or groups[i - 1] != g])


def test_pyxem_simulation_opens_as_4dstem():
    pytest.importorskip("pyxem")
    from si_viewer.session import DatasetKind, Session

    ids = {e["id"]: e for e in examples.list_examples()}
    assert ids["pyxem-stripes"]["group"] == "4D-STEM: DPC and beam shift"
    assert "download_mb" not in ids["pyxem-stripes"]  # simulated: nothing to download
    s = Session()
    entry = s.open_example("pyxem-stripes")
    d = s.datasets[entry.dataset_ids[0]]
    assert entry.label == "Magnetic stripes (pyxem, simulated)"
    assert d.kind is DatasetKind.DIFFRACTION4D and d.signal.data.dtype == "float32"
    # pyxem's LaTeX units ($\AA^{-1}$), as the app writes them.
    strain = s.datasets[s.open_example("pyxem-strain").dataset_ids[0]]
    assert strain.signal.axes_manager.signal_axes[0].units == "1/Å"


def test_an_analysis_of_an_example_opens_again(tmp_path):
    """An example's path is ``example:<id>``: opened from the examples."""
    from si_viewer.project import open_project, save_project
    from si_viewer.session import Session

    s = Session()
    entry = s.open_example("two-gaussians")
    ids = list(entry.dataset_ids)
    saved = save_project(s, tmp_path / "ex.hspy")
    other = Session()
    open_project(other, saved)
    again = other.files[entry.id]
    assert str(again.path) == "example:two-gaussians" and list(again.dataset_ids) == ids
    assert other.get_dataset(ids[0]).signal.data.shape == s.get_dataset(ids[0]).signal.data.shape


def test_an_edited_calibration_is_kept_with_the_analysis(tmp_path):
    """Info → Edit calibration on a file's data: saved (link level) and set
    again on the raw data when the analysis is opened."""
    from fastapi.testclient import TestClient

    from si_viewer.app import create_app
    from si_viewer.project import open_project, save_project
    from si_viewer.session import Session

    s = Session()
    client = TestClient(create_app(s))
    ds_id = client.post("/api/examples/two-gaussians").json()["datasets"][0]["id"]
    axes = s.get_dataset(ds_id).signal.axes_manager._axes
    last = len(axes) - 1
    before = axes[last].scale
    r = client.patch(f"/api/datasets/{ds_id}/axes", json=[{"index": last, "scale": before * 2, "offset": -1.5, "units": "eV"}])
    assert r.status_code == 200
    saved = save_project(s, tmp_path / "cal.hspy")
    other = Session()
    open_project(other, saved)
    ax = other.get_dataset(ds_id).signal.axes_manager._axes[last]
    assert (ax.scale, ax.offset, ax.units) == (before * 2, -1.5, "eV")
