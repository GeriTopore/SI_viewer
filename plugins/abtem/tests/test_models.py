"""Oriented, orthogonal models of a structure."""

import numpy as np
import pytest
from ase.geometry import get_distances
from fastapi.testclient import TestClient

import si_viewer_abtem.plugin as abtem
from si_viewer.app import create_app
from si_viewer.project import open_project
from si_viewer.session import Session
from si_viewer_abtem.models import build_model, orient, preview, read_model
from si_viewer_abtem.structures import StructureError

from test_projects import MIXED, STO

A = 3.905
ZNO = """data_ZnO
_symmetry_space_group_name_H-M 'P 63 m c'
_symmetry_Int_Tables_number 186
_cell_length_a 3.25
_cell_length_b 3.25
_cell_length_c 5.207
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 120
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Zn1 Zn 0.33333333 0.66666667 0
O1 O 0.33333333 0.66666667 0.382
"""


@pytest.fixture
def sto(tmp_path):
    path = tmp_path / "SrTiO3.cif"
    path.write_text(STO)
    return path


@pytest.fixture
def zno(tmp_path):
    path = tmp_path / "ZnO.cif"
    path.write_text(ZNO)
    return path


def _nearest(atoms):
    _, d = get_distances(atoms.positions, cell=atoms.cell, pbc=True)
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


@pytest.mark.parametrize(
    "uvw, lengths, cells",
    [
        ([0, 0, 1], (A, A, A), 1),
        ([1, 0, 0], (A, A, A), 1),
        ([1, 1, 0], (A * 2**0.5, A, A * 2**0.5), 2),
        ([1, 1, 1], (A * 6**0.5, A * 2**0.5, A * 3**0.5), 6),
        ([2, 1, 0], (A * 5**0.5, A, A * 5**0.5), 5),
    ],
)
def test_cubic_orientations(uvw, lengths, cells):
    o = orient(np.eye(3) * A, uvw)
    np.testing.assert_allclose(o.lengths, lengths, atol=1e-9)
    assert o.cells == cells and o.strain_percent == 0
    assert np.linalg.det(o.matrix) > 0
    assert (o.matrix[2] == uvw).all()
    h = o.matrix * A
    np.testing.assert_allclose(h @ h.T, np.diag(np.diag(h @ h.T)), atol=1e-9)


@pytest.mark.parametrize("uvw", [[0, 0, 1], [1, 1, 0], [1, 1, 1], [1, -1, 0], [2, 1, 0]])
def test_model_keeps_the_crystal(sto, uvw):
    model, info = build_model(sto, uvw, [2, 2, 2])
    assert (model.cell.array == np.diag(np.diag(model.cell.array))).all() and model.pbc.all()
    assert len(model) == info["atoms"]
    assert model.get_volume() == pytest.approx(len(model) / 5 * A**3)
    symbols = model.get_chemical_symbols()
    assert symbols.count("O") == 3 * symbols.count("Sr") == 3 * symbols.count("Ti")
    # Every atom's nearest neighbour is at Ti–O or O–O/Sr–O distance, and no atom is doubled.
    assert _nearest(model).min() == pytest.approx(A / 2, abs=1e-6)


def test_001_columns_are_on_the_expected_sites(sto):
    model, _ = build_model(sto, [0, 0, 1], [1, 1, 1])
    frac = np.round(model.get_scaled_positions(), 6)
    got = {(s, *f) for s, f in zip(model.get_chemical_symbols(), frac.tolist())}
    assert ("Sr", 0.0, 0.0, 0.0) in got and ("Ti", 0.5, 0.5, 0.5) in got
    assert ("O", 0.5, 0.5, 0.0) in got


def test_110_atoms_form_columns_along_the_beam(sto):
    model, _ = build_model(sto, [1, 1, 0], [1, 1, 4])
    xy = np.round(model.positions[:, :2], 5)
    assert len(np.unique(xy, axis=0)) * 4 <= len(model)


def test_hexagonal_orientations(zno):
    from si_viewer_abtem.models import _read_unit

    cell = _read_unit(zno).cell.array
    o = orient(cell, [0, 0, 1])
    assert o.strain_percent == 0 and o.cells == 2
    np.testing.assert_allclose(sorted(o.lengths[:2]), [3.25, 3.25 * 3**0.5], rtol=1e-6)
    for uvw in ([1, 0, 0], [1, 1, 0], [2, 1, 0]):
        o = orient(cell, uvw)
        assert o.strain_percent < 1e-4
    model, info = build_model(zno, [0, 0, 1], [2, 2, 2])
    assert len(model) == info["atoms"] == 4 * 2 * 8
    assert 1.9 < _nearest(model).min() < 2.05  # the Zn–O bond


def test_preview_matches_build(sto):
    p = preview(sto, [1, 1, 0], [2, 3, 4])
    model, _ = build_model(sto, [1, 1, 0], [2, 3, 4])
    assert p["atoms"] == len(model)
    np.testing.assert_allclose(np.diag(model.cell.array), p["size"], atol=1e-3)


def test_default_repeats(sto):
    p = preview(sto, [1, 1, 2])
    assert p["repeats"] == [1, 1, 1] and p["atoms"] == 30


def test_112_with_six_repeats_along_the_beam(sto):
    model, info = build_model(sto, [1, 1, 2], [1, 1, 6])
    assert info["axes"][2] == [1, 1, 2] and info["strain_percent"] == 0
    np.testing.assert_allclose(info["size"], [A * 2**0.5, A * 3**0.5, 6 * A * 6**0.5], atol=1e-3)
    assert len(model) == 180
    assert model.get_volume() == pytest.approx(36 * A**3)
    assert _nearest(model).min() == pytest.approx(A / 2, abs=1e-6)


def test_mixed_sites_are_filled_exactly_and_reproducibly(tmp_path):
    path = tmp_path / "mixed.cif"
    path.write_text(MIXED)
    one, info = build_model(path, [0, 0, 1], [4, 4, 4], seed=3)
    two, _ = build_model(path, [0, 0, 1], [4, 4, 4], seed=3)
    other, _ = build_model(path, [0, 0, 1], [4, 4, 4], seed=4)
    assert info["mixed"]
    symbols = one.get_chemical_symbols()
    assert symbols.count("Sr") == symbols.count("Ba") == 32 and symbols.count("Ti") == 64
    assert symbols == two.get_chemical_symbols() and symbols != other.get_chemical_symbols()


def test_vacancies_are_removed(tmp_path):
    path = tmp_path / "vacant.cif"
    path.write_text(STO.replace("Sr1 Sr 0 0 0 1\n", "Sr1 Sr 0 0 0 0.75\n"))
    model, _ = build_model(path, [0, 0, 1], [4, 4, 4])
    symbols = model.get_chemical_symbols()
    assert symbols.count("Sr") == 48 and symbols.count("Ti") == 64 and len(model) == 64 * 5 - 16


@pytest.mark.parametrize("uvw", [[0, 0, 0], [13, 1, 0]])
def test_bad_zone_axes(uvw):
    with pytest.raises(StructureError):
        orient(np.eye(3) * A, uvw)


def test_too_many_atoms(sto):
    with pytest.raises(StructureError, match="limit"):
        build_model(sto, [0, 0, 1], [40, 40, 40])


def test_model_in_a_project(tmp_path, sto):
    client = TestClient(create_app(Session(), plugins=[abtem]))
    file = client.post("/api/plugins/abtem/projects", json={"path": str(tmp_path / "project")}).json()["file"]
    structure = client.post(f"/api/plugins/abtem/projects/{file['id']}/structures", json={"path": str(sto)}).json()["dataset_id"]
    p = client.post(f"/api/plugins/abtem/structures/{structure}/model/preview", json={"uvw": [1, 1, 0]})
    assert p.status_code == 200 and p.json()["cell"][1] == pytest.approx(A)
    r = client.post(f"/api/plugins/abtem/structures/{structure}/models", json={"uvw": [1, 1, 0], "repeats": [2, 2, 2]})
    assert r.status_code == 200, r.text
    item = next(d for d in r.json()["file"]["datasets"] if d["id"] == r.json()["dataset_id"])
    assert item["parent_id"] == structure and item["plugin"]["item"] == "model" and item["title"] == "[110] 2×2×2"
    settings = item["plugin"]["settings"]
    assert settings["atoms"] == 80 and settings["composition"] == {"O": 48, "Sr": 16, "Ti": 16}
    saved = tmp_path / "project" / settings["file"]
    assert saved.is_file() and len(read_model(saved)) == 80
    atoms = client.get(f"/api/plugins/abtem/structures/{item['id']}/atoms").json()
    assert len(atoms["atoms"]["symbol"]) == 80

    # A restart keeps it, and a model can't be made from a model.
    session = Session()
    open_project(session, tmp_path / "project/project.hspy")
    assert session.datasets[item["id"]].plugin == item["plugin"] and session.datasets[item["id"]].parent_id == structure
    assert client.post(f"/api/plugins/abtem/structures/{item['id']}/models", json={"uvw": [0, 0, 1]}).status_code == 404
    assert client.post(f"/api/plugins/abtem/structures/{structure}/models", json={"uvw": [0, 0, 0]}).status_code == 422
