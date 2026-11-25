"""Simulation projects: creating, loading structures, reopening, and a save by the base app."""

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import si_viewer_abtem.plugin as abtem
from si_viewer.app import create_app
from si_viewer.project import open_project, read_project, save_project
from si_viewer.session import DatasetKind, Session
from si_viewer_abtem.structures import StructureError, read_structure, structure_atoms

STO = """data_SrTiO3
_symmetry_space_group_name_H-M 'P m -3 m'
_symmetry_Int_Tables_number 221
_cell_length_a 3.905
_cell_length_b 3.905
_cell_length_c 3.905
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_symmetry_equiv_pos_as_xyz
x,y,z
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Sr1 Sr 0 0 0 1
Ti1 Ti 0.5 0.5 0.5 1
O1 O 0.5 0.5 0 1
O2 O 0.5 0 0.5 1
O3 O 0 0.5 0.5 1
"""

MIXED = STO.replace("Sr1 Sr 0 0 0 1\n", "Sr1 Sr 0 0 0 0.5\nBa1 Ba 0 0 0 0.5\n")


@pytest.fixture
def cif(tmp_path):
    path = tmp_path / "SrTiO3.cif"
    path.write_text(STO)
    return path


@pytest.fixture
def client():
    return TestClient(create_app(Session(), plugins=[abtem]))


def _new(client, folder):
    r = client.post("/api/plugins/abtem/projects", json={"path": str(folder)})
    assert r.status_code == 200, r.text
    return r.json()["file"]


def _load(client, file_id, path):
    return client.post(f"/api/plugins/abtem/projects/{file_id}/structures", json={"path": str(path)})


def test_read_structure(cif):
    info = read_structure(cif)
    assert info["formula"] == "SrTiO3" and info["atoms"] == 5 and info["mixed_sites"] == 0
    assert info["composition"] == {"O": 3, "Sr": 1, "Ti": 1}
    assert info["cell"] == [3.905, 3.905, 3.905, 90, 90, 90]
    assert info["spacegroup"]["number"] == 221


def test_mixed_site_composition(tmp_path):
    path = tmp_path / "mixed.cif"
    path.write_text(MIXED)
    info = read_structure(path)
    assert info["formula"] is None and info["mixed_sites"] == 1
    assert info["composition"]["Sr"] == pytest.approx(0.5) and info["composition"]["Ba"] == pytest.approx(0.5)


def test_bad_cif_is_refused(tmp_path):
    path = tmp_path / "bad.cif"
    path.write_text("not a cif")
    with pytest.raises(StructureError):
        read_structure(path)


def test_atoms_repeat_on_cell_boundaries(client, tmp_path, cif):
    file = _new(client, tmp_path / "project")
    item = _load(client, file["id"], cif).json()["file"]["datasets"][0]
    r = client.get(f"/api/plugins/abtem/structures/{item['id']}/atoms")
    assert r.status_code == 200, r.text
    data = r.json()
    symbols = data["atoms"]["symbol"]
    assert (symbols.count("Sr"), symbols.count("Ti"), symbols.count("O")) == (8, 1, 6)
    frac = np.array(data["atoms"]["frac"])
    assert frac.min() == 0 and frac.max() == 1
    np.testing.assert_allclose(data["atoms"]["xyz"], frac * 3.905, atol=1e-5)
    np.testing.assert_allclose(data["cell"], np.eye(3) * 3.905, atol=1e-5)
    assert len(data["sites"]) == 5 and set(data["elements"]) == {"Sr", "Ti", "O"}
    assert all(data["elements"][e]["radius"] > 0 for e in data["elements"])
    assert data["elements"]["O"]["covalent"] == pytest.approx(0.66)


def test_atoms_keep_site_occupancy(tmp_path):
    path = tmp_path / "mixed.cif"
    path.write_text(MIXED)
    data = structure_atoms(path)
    site = data["sites"][data["atoms"]["site"][0]]
    assert site == pytest.approx({"Sr": 0.5, "Ba": 0.5})
    assert {"Sr", "Ba"} <= set(data["elements"])


def test_atoms_of_something_else_is_404(client):
    assert client.get("/api/plugins/abtem/structures/nope/atoms").status_code == 404


def test_new_project_makes_the_folder(client, tmp_path):
    folder = tmp_path / "STO simulation"
    file = _new(client, folder)
    assert file["name"] == "STO simulation" and file["datasets"] == []
    assert file["plugin"] == {"id": "abtem", "type": "simulation"}
    assert (folder / "structures").is_dir() and (folder / "runs").is_dir()
    assert (folder / "STO simulation.hspy").is_file()


def test_used_folder_is_refused(client, tmp_path):
    folder = tmp_path / "used"
    folder.mkdir()
    (folder / "keep.txt").write_text("mine")
    r = client.post("/api/plugins/abtem/projects", json={"path": str(folder)})
    assert r.status_code == 422
    assert [p.name for p in folder.iterdir()] == ["keep.txt"]


def test_structure_is_copied_and_listed(client, tmp_path, cif):
    file = _new(client, tmp_path / "project")
    r = _load(client, file["id"], cif).json()
    [d] = r["file"]["datasets"]
    assert d["id"] == r["dataset_id"] and d["kind"] == "item" and d["title"] == "SrTiO3"
    assert d["plugin"]["item"] == "structure" and d["plugin"]["settings"]["file"] == "structures/SrTiO3.cif"
    assert (tmp_path / "project/structures/SrTiO3.cif").read_text() == STO
    # The same file again: a second copy, not an overwrite.
    again = _load(client, file["id"], cif).json()
    assert again["file"]["datasets"][1]["plugin"]["settings"]["file"] == "structures/SrTiO3 2.cif"


def test_bad_structure_changes_nothing(client, tmp_path):
    file = _new(client, tmp_path / "project")
    bad = tmp_path / "bad.cif"
    bad.write_text("not a cif")
    assert _load(client, file["id"], bad).status_code == 422
    assert client.get("/api/files").json()[0]["datasets"] == []
    assert list((tmp_path / "project/structures").iterdir()) == []


def test_reopened_after_a_restart(client, tmp_path, cif):
    file = _new(client, tmp_path / "project")
    item = _load(client, file["id"], cif).json()["file"]["datasets"][0]
    # Saved when loaded: a new session (a restarted server) opens it.
    session = Session()
    open_project(session, tmp_path / "project/project.hspy")
    entry = session.files[file["id"]]
    assert entry.plugin == {"id": "abtem", "type": "simulation"} and entry.path == (tmp_path / "project").resolve()
    ds = session.datasets[item["id"]]
    assert ds.kind is DatasetKind.ITEM and ds.plugin == item["plugin"] and ds.title == "SrTiO3"


def test_moved_folder_still_opens(client, tmp_path, cif):
    file = _new(client, tmp_path / "project")
    _load(client, file["id"], cif)
    moved = tmp_path / "elsewhere"
    (tmp_path / "project").rename(moved)
    session = Session()
    _, warnings = open_project(session, moved / "project.hspy")
    assert warnings == [] and session.files[file["id"]].path == moved.resolve()


def test_base_app_keeps_the_project(client, tmp_path, cif):
    """Opened and saved without the plugin, nothing is lost, and the plugin can go on."""
    file = _new(client, tmp_path / "project")
    item = _load(client, file["id"], cif).json()["file"]["datasets"][0]
    target = tmp_path / "project/project.hspy"
    views = {"marker": 1}

    base = TestClient(create_app(Session(), plugins=[]))
    assert base.post("/api/project/open", json={"path": str(target)}).status_code == 200
    listed = base.get("/api/files").json()[0]
    assert listed["plugin"]["id"] == "abtem" and listed["datasets"][0]["plugin"] == item["plugin"]
    assert base.get(f"/api/datasets/{item['id']}/thumbnail").status_code == 200
    r = base.post("/api/project/save", json={"path": str(target), "views": views, "overwrite": True, "file_ids": [file["id"]]})
    assert r.status_code == 200, r.text
    assert read_project(target)[1] == views

    again = TestClient(create_app(Session(), plugins=[abtem]))
    assert again.post("/api/project/open", json={"path": str(target)}).status_code == 200
    r = _load(again, file["id"], cif).json()
    assert [d["id"] for d in r["file"]["datasets"]][0] == item["id"] and len(r["file"]["datasets"]) == 2
    assert (tmp_path / "project/structures/SrTiO3.cif").is_file()


def test_item_can_be_deleted_and_restored(client, tmp_path, cif):
    file = _new(client, tmp_path / "project")
    item = _load(client, file["id"], cif).json()["file"]["datasets"][0]
    session = client.app.state.session
    token = session.delete_dataset(session.datasets[item["id"]])
    assert item["id"] not in session.datasets
    session.undo(token)
    assert session.datasets[item["id"]].plugin == item["plugin"]


def test_entry_point_is_declared():
    text = (Path(abtem.__file__).parents[2] / "pyproject.toml").read_text()
    assert 'abtem = "si_viewer_abtem.plugin"' in text


def test_save_project_helper_is_atomic(client, tmp_path, cif):
    file = _new(client, tmp_path / "project")
    session = client.app.state.session
    save_project(session, tmp_path / "project/project.hspy", {"a": 1}, "link", [file["id"]])
    assert not list((tmp_path / "project").glob(".*saving"))
