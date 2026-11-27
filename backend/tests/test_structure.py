"""Crystal structures (CIFs) opened as orix phases, as pyxem takes them."""

from fastapi.testclient import TestClient

from si_viewer.app import create_app
from si_viewer.processing.structure import phase_of

AU = """data_Au
_cell_length_a 4.0782
_cell_length_b 4.0782
_cell_length_c 4.0782
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'F m -3 m'
_symmetry_Int_Tables_number 225
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Au1 Au 0 0 0 1
"""

PEROVSKITE = """data_SrTiO3
_cell_length_a 3.905
_cell_length_b 3.905
_cell_length_c 3.905
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P m -3 m'
_symmetry_Int_Tables_number 221
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
"""


def _open(client, path):
    r = client.post("/api/files", json={"path": str(path)})
    assert r.status_code == 200, r.text
    return r.json()["datasets"][0]


def test_cif_as_a_phase(tmp_path):
    (tmp_path / "Au.cif").write_text(AU)
    (tmp_path / "SrTiO3.cif").write_text(PEROVSKITE)
    client = TestClient(create_app())
    au = _open(client, tmp_path / "Au.cif")
    assert au["kind"] == "other" and au["title"] == "Au"
    s = au["structure"]
    assert s["formula"] == "Au" and s["spacegroup"] == {"number": 225, "symbol": "Fm-3m"}
    assert s["point_group"] == "m-3m" and s["cell"] == [4.0782, 4.0782, 4.0782, 90.0, 90.0, 90.0]
    # The whole cell, as orix expands it (4 atoms of fcc).
    assert s["composition"] == {"Au": 4.0} and len(s["sites"]) == 4
    sto = _open(client, tmp_path / "SrTiO3.cif")["structure"]
    assert sto["formula"] == "SrTiO3" and sto["composition"] == {"Sr": 1.0, "Ti": 1.0, "O": 3.0}
    # The orix phase diffsims simulates from.
    phase = phase_of(client.app.state.session.get_dataset(au["id"]).signal)
    assert phase.name == "Au" and phase.space_group.number == 225 and len(phase.structure) == 4
    # Listed with the files that can be opened.
    listing = client.get("/api/fs/list", params={"path": str(tmp_path)}).json()
    entries = listing.get("entries", listing) if isinstance(listing, dict) else listing
    assert any(e["name"] == "Au.cif" and e.get("supported", True) for e in entries)


def test_bad_cif(tmp_path):
    (tmp_path / "bad.cif").write_text("data_x\n_cell_length_a 1\n")
    r = TestClient(create_app()).post("/api/files", json={"path": str(tmp_path / "bad.cif")})
    assert r.status_code == 422


def test_saved_with_the_analysis(tmp_path):
    (tmp_path / "Au.cif").write_text(AU)
    client = TestClient(create_app())
    _open(client, tmp_path / "Au.cif")
    for level in ("link", "raw"):
        saved = tmp_path / f"a-{level}.hspy"
        r = client.post("/api/project/save", json={"path": str(saved), "views": {}, "level": level})
        assert r.status_code == 200, r.text
        other = TestClient(create_app())
        r = other.post("/api/project/open", json={"path": str(saved)})
        assert r.status_code == 200, r.text
        d = other.get("/api/files").json()[0]["datasets"][0]
        assert d["structure"]["formula"] == "Au" and d["kind"] == "other"
