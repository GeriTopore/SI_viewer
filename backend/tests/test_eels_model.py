"""EELS model fitting (M13): the edges of the chosen elements, the fit of one
spectrum and of every pixel against exSpy's own calls."""

import warnings
from pathlib import Path

import hyperspy.api as hs
import numpy as np
import pooch
import pytest
from fastapi.testclient import TestClient
from test_thickness import _link, _open, _role, _si, _wait

from si_viewer.app import create_app
from si_viewer.processing import eels_model as M
from si_viewer.processing.eels_model import EelsFitOptions

HYDROGENIC = {"gos": "hydrogenic"}


@pytest.fixture
def client():
    return TestClient(create_app())


def _has_dft() -> bool:
    return any(Path(pooch.os_cache("pooch")).glob("*Segger_Guzzinati_Kohl*.gosh"))


def _synthetic(tmp_path, shape=(3, 4)) -> tuple[Path, np.ndarray, np.ndarray]:
    """A spectrum image made by the model itself (O K and Mn L, hydrogenic)
    with a known intensity of each at every pixel."""
    s = hs.signals.Signal1D(np.ones((*shape, 400), np.float32))
    s.set_signal_type("EELS")
    axis = s.axes_manager.signal_axes[0]
    axis.offset, axis.scale, axis.units, axis.name = 450.0, 0.75, "eV", "Energy loss"
    s.set_microscope_parameters(beam_energy=200, convergence_angle=20, collection_angle=40)
    M.set_elements(s, ["O", "Mn"])
    built = M.build_model(s, EelsFitOptions(**HYDROGENIC))
    m = built.model
    rng = np.random.default_rng(3)
    oxygen = rng.uniform(2e3, 6e3, shape)
    manganese = rng.uniform(5e2, 2e3, shape)
    for el, values in (("O", oxygen), ("Mn", manganese)):
        p = built.masters[el].intensity
        p.map["values"][:], p.map["is_set"][:] = values, True
    power_law = next(c for c in m if hasattr(c, "r"))
    for p, value in ((power_law.A, 4e11), (power_law.r, 3.0)):
        p.map["values"][:], p.map["is_set"][:] = value, True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s.data = m.as_signal(show_progressbar=False).data.astype(np.float32)
    path = tmp_path / "synthetic_eels.hspy"
    s.save(path)
    return path, oxygen, manganese


def _elements(client, ds, elements) -> dict:
    r = client.put(f"/api/datasets/{ds['id']}/eels-elements", json={"elements": elements})
    assert r.status_code == 200, r.text
    return r.json()["dataset"]


def _fit(client, ds, source=None, **options):
    body = {"options": options} | ({"source": source} if source else {})
    return client.post(f"/api/datasets/{ds['id']}/eels-fit", json=body)


def _maps(job, origin) -> dict[str, dict]:
    return {d["element"]: d for d in job["result"]["file"]["datasets"] if d["origin"] == origin}


def _data(client, ds_id) -> np.ndarray:
    return np.asarray(client.app.state.session.get_dataset(ds_id).signal.data)


# ---------- elements and edges ----------


def test_element_table_has_edges(client):
    table = {e["symbol"]: e for e in client.get("/api/elements").json()}
    mn = {e["edge"]: e for e in table["Mn"]["edges"]}
    assert mn["L3"] == {"edge": "L3", "onset": 640.0, "major": True}
    assert not mn["L1"]["major"]
    assert [e["edge"] for e in table["O"]["edges"]] == ["K"]  # "K1a"-like entries are left out


def test_elements_and_subshells_in_range(client, eels_cuzn):
    ds = _si(_open(client, eels_cuzn))
    assert ds["edges"] == {"elements": [], "subshells": []} and ds["eds"] is None
    ds = _elements(client, ds, ["Zn", "Cu", "O"])
    assert ds["edges"]["elements"] == ["Cu", "O", "Zn"]
    # 700–1988 eV: the L edges of Cu and Zn; O K (532 eV) is below the range.
    assert [s["name"] for s in ds["edges"]["subshells"]] == ["Cu_L3", "Cu_L2", "Zn_L3", "Zn_L2", "Cu_L1", "Zn_L1"]
    ds = _elements(client, ds, ["Cu"])
    assert ds["edges"]["elements"] == ["Cu"]
    assert _elements(client, ds, [])["edges"] == {"elements": [], "subshells": []}
    assert client.put(f"/api/datasets/{ds['id']}/eels-elements", json={"elements": ["Xx"]}).status_code == 422


def test_elements_are_for_eels(client, tmp_path):
    s = hs.signals.Signal1D(np.ones((2, 2, 64), np.float32))
    s.save(tmp_path / "plain.hspy")
    ds = _si(_open(client, tmp_path / "plain.hspy"))
    assert ds["edges"] is None
    assert client.put(f"/api/datasets/{ds['id']}/eels-elements", json={"elements": ["Cu"]}).status_code == 400
    assert _fit(client, ds).status_code == 400


# ---------- one spectrum ----------


def test_fit_equals_exspy(client, eels_cuzn):
    ds = _elements(client, _si(_open(client, eels_cuzn)), ["Cu", "Zn"])
    r = _fit(client, ds, **HYDROGENIC)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["pixels"] == 2000 and not out["convolved"] and out["low_loss_id"] is None

    ref = hs.load(eels_cuzn).sum()
    ref.change_dtype(float)
    ref.add_elements(("Cu", "Zn"))
    m = ref.create_model(GOS="hydrogenic")
    m.smart_fit()
    rows = {row["element"]: row for row in out["rows"]}
    assert [row["edge"] for row in out["rows"]] == ["Cu_L3", "Zn_L3"]  # by onset
    assert rows["Cu"]["intensity"] == pytest.approx(m.components.Cu_L3.intensity.value, rel=1e-6)
    assert rows["Zn"]["intensity"] == pytest.approx(m.components.Zn_L3.intensity.value, rel=1e-6)
    assert rows["Cu"]["atomic_percent"] + rows["Zn"]["atomic_percent"] == pytest.approx(100)
    assert rows["Cu"]["onset"] == 931.0 and rows["Cu"]["std"] > 0

    # The curves add up, and describe the data.
    model = np.array(out["background"]) + sum(np.array(c) for c in out["elements"].values())
    assert np.allclose(out["model"], model)
    assert len(out["x"]) == 162 and out["x"][0] == 700.0
    assert out["residual_ratio"] < 0.02


def test_fit_options(client, eels_cuzn):
    ds = _elements(client, _si(_open(client, eels_cuzn)), ["Cu", "Zn", "O"])
    base = _fit(client, ds, **HYDROGENIC).json()
    assert base["skipped"] == ["O"]

    cropped = _fit(client, ds, lo=802, hi=1500, **HYDROGENIC).json()
    assert cropped["x"][0] == 804.0 and cropped["x"][-1] == 1500.0  # 8 eV channels from 700

    moved = _fit(client, ds, onsets={"Cu": 939.0}, **HYDROGENIC).json()
    assert {r["element"]: r["onset"] for r in moved["rows"]} == {"Cu": 939.0, "Zn": 1020.0}
    assert moved["rows"][0]["intensity"] != base["rows"][0]["intensity"]

    fine = _fit(client, ds, fine_structure=True, fine_structure_width=40, **HYDROGENIC).json()
    assert all(0 < r["fine_structure_width"] <= 40 for r in fine["rows"])
    assert base["rows"][0]["fine_structure_width"] is None

    pixel = _fit(client, ds, source={"type": "pixel", "x": 32, "y": 14, "size": 3}, **HYDROGENIC).json()
    assert pixel["pixels"] == 9


def test_fit_refusals(client, eels_cuzn):
    ds = _si(_open(client, eels_cuzn))
    r = _fit(client, ds, **HYDROGENIC)
    assert r.status_code == 422 and "Add elements" in r.json()["detail"]
    ds = _elements(client, ds, ["Cu"])
    r = _fit(client, ds, lo=1200, **HYDROGENIC)  # no Cu edge above 1200 eV
    assert r.status_code == 422 and "edge in the fitted range" in r.json()["detail"]
    r = _fit(client, ds, lo=5000, **HYDROGENIC)
    assert r.status_code == 422 and "fit range" in r.json()["detail"]
    # The cross-sections need the microscope's parameters.
    signal = client.app.state.session.get_dataset(ds["id"]).signal
    del signal.metadata.Acquisition_instrument.TEM.convergence_angle
    r = _fit(client, ds, **HYDROGENIC)
    assert r.status_code == 422 and "convergence angle" in r.json()["detail"]


def test_hydrogenic_has_no_m_edges(client, tmp_path):
    path, *_ = _synthetic(tmp_path)
    ds = _elements(client, _si(_open(client, path)), ["La"])  # only M edges... none in 450–750 eV
    assert ds["edges"]["subshells"] == []
    s = hs.signals.Signal1D(np.ones(400, np.float32))
    s.set_signal_type("EELS")
    s.axes_manager[0].offset, s.axes_manager[0].scale = 700.0, 1.0
    s.set_microscope_parameters(beam_energy=200, convergence_angle=20, collection_angle=40)
    M.set_elements(s, ["La"])
    with pytest.raises(ValueError, match="K and L edges only"):
        M.build_model(s, EelsFitOptions(gos="hydrogenic"))


@pytest.mark.skipif(not _has_dft(), reason="exSpy's DFT GOS table is not downloaded")
def test_dft_cross_sections(client, eels_cuzn):
    ds = _elements(client, _si(_open(client, eels_cuzn)), ["Cu", "Zn"])
    out = _fit(client, ds).json()  # DFT is the default
    assert out["gos"] == "dft"
    ref = hs.load(eels_cuzn).sum()
    ref.change_dtype(float)
    ref.add_elements(("Cu", "Zn"))
    m = ref.create_model()
    m.smart_fit()
    rows = {row["element"]: row for row in out["rows"]}
    assert rows["Cu"]["intensity"] == pytest.approx(m.components.Cu_L3.intensity.value, rel=1e-6)
    assert rows["Zn"]["intensity"] == pytest.approx(m.components.Zn_L3.intensity.value, rel=1e-6)


def test_fit_convolved_with_the_low_loss(client, eels_linescan, eels_linescan_low_loss):
    hl = _si(_open(client, eels_linescan))
    ll = _si(_open(client, eels_linescan_low_loss))
    hl = _elements(client, hl, ["O", "Mn"])
    alone = _fit(client, hl, **HYDROGENIC).json()
    assert not alone["convolved"]

    _role(client, ll, "low_loss")
    _link(client, hl, ll)
    out = _fit(client, hl, **HYDROGENIC).json()
    assert out["convolved"] and out["low_loss_id"] == ll["id"]
    assert not _fit(client, hl, convolve=False, **HYDROGENIC).json()["convolved"]

    s, low = hs.load(eels_linescan).sum(), hs.load(eels_linescan_low_loss).sum()
    s.change_dtype(float)
    low.change_dtype(float)
    s.add_elements(("O", "Mn"))
    m = s.create_model(low_loss=low, GOS="hydrogenic")
    m.smart_fit()
    rows = {row["element"]: row for row in out["rows"]}
    assert rows["O"]["intensity"] == pytest.approx(m.components.O_K.intensity.value, rel=1e-5)
    assert rows["Mn"]["intensity"] == pytest.approx(m.components.Mn_L3.intensity.value, rel=1e-5)
    # The low loss itself is fitted without convolution.
    assert _fit(client, _elements(client, ll, ["O"]), **HYDROGENIC).status_code in (200, 422)


# ---------- every pixel ----------


def test_maps_recover_known_intensities(client, tmp_path):
    path, oxygen, manganese = _synthetic(tmp_path)
    ds = _elements(client, _si(_open(client, path)), ["O", "Mn"])
    r = client.post(f"/api/datasets/{ds['id']}/eels-fit-maps", json={"options": HYDROGENIC})
    assert r.status_code == 200, r.text
    job = _wait(client, r.json())
    assert job["status"] == "done", job["error"]
    fitted = _maps(job, "fitted")
    assert set(fitted) == {"O", "Mn"} and fitted["O"]["quantities"] == ["counts", "at%"]
    assert fitted["O"]["title"] == "O (fit)" and fitted["O"]["parent_id"] == ds["id"] and fitted["O"]["is_element_map"]
    assert np.allclose(_data(client, fitted["O"]["id"]), oxygen, rtol=1e-3)
    assert np.allclose(_data(client, fitted["Mn"]["id"]), manganese, rtol=1e-3)
    # The fit at a pixel (and a block) adds up to that pixel's map values.
    si_now = next(d for d in job["result"]["file"]["datasets"] if d["id"] == ds["id"])
    assert si_now["pixel_fit"]
    one = client.get(f"/api/datasets/{ds['id']}/eels-pixel-fit", params={"x": 2, "y": 1}).json()
    rows = {r["element"]: r for r in one["rows"]}
    assert one["pixels"] == 1 and rows["O"]["intensity"] == pytest.approx(float(oxygen[1, 2]), rel=1e-3)
    assert one["residual_ratio"] < 0.05
    block = client.get(f"/api/datasets/{ds['id']}/eels-pixel-fit", params={"x": 1, "y": 1, "size": 2}).json()
    assert block["pixels"] == 4
    assert {r["element"]: r for r in block["rows"]}["Mn"]["intensity"] == pytest.approx(float(manganese[1:3, 1:3].sum()), rel=1e-3)
    assert client.get(f"/api/datasets/{ds['id']}/eels-pixel-fit", params={"x": 99, "y": 0}).status_code == 422
    assert client.put(f"/api/datasets/{ds['id']}/fit-quantity", json={"quantity": "at%"}).status_code == 200
    assert np.allclose(_data(client, fitted["O"]["id"]), 100 * oxygen / (oxygen + manganese), rtol=1e-3)

    # A second fit replaces the maps; without composition there are none.
    r = client.post(f"/api/datasets/{ds['id']}/eels-fit-maps", json={"options": HYDROGENIC, "composition": False})
    job = _wait(client, r.json())
    maps = _maps(job, "fitted")
    assert len(maps) == 2 and all(d["quantities"] == ["counts"] and d["quantity"] == "counts" for d in maps.values())


def test_maps_equal_exspy_multifit(eels_cuzn):
    s = hs.load(eels_cuzn).inav[30:35, 12:16]
    M.set_elements(s, ["Cu", "Zn"])
    done = []
    maps, skipped, fits = M.fit_maps(s, EelsFitOptions(**HYDROGENIC), report=done.append)
    assert skipped == [] and done[-1] == 1.0
    ref = s.deepcopy()
    ref.change_dtype("float32")
    m = ref.create_model(GOS="hydrogenic")
    m.multifit(kind="smart", show_progressbar=False)
    assert maps["Cu"].shape == (4, 5)
    assert np.allclose(maps["Cu"], m.components.Cu_L3.intensity.map["values"], rtol=1e-4)
    assert np.allclose(maps["Zn"], m.components.Zn_L3.intensity.map["values"], rtol=1e-4)


def test_maps_refusals(client, eels_cuzn, eels_linescan):
    ds = _si(_open(client, eels_cuzn))
    r = client.post(f"/api/datasets/{ds['id']}/eels-fit-maps", json={"options": HYDROGENIC})
    assert r.status_code == 422 and "Add elements" in r.json()["detail"]
    line = _elements(client, _si(_open(client, eels_linescan)), ["O"])
    assert client.post(f"/api/datasets/{line['id']}/eels-fit-maps", json={"options": HYDROGENIC}).status_code == 400


def test_atomic_percent_maps():
    at = M.atomic_percent_maps({"A": np.array([1.0, -2.0, 0.0, np.nan]), "B": np.array([3.0, 4.0, 0.0, 2.0])})
    assert np.allclose(at["A"][:2], [25, 0]) and np.allclose(at["B"][[0, 1, 3]], [75, 100, 100])
    assert np.isnan(at["A"][2]) and np.isnan(at["B"][2])


def test_microscope_parameters_set_in_the_info_tab(client, eels_cuzn, tmp_path):
    """Parameters missing from the file are set (with exSpy) and then fit;
    they are kept with a saved analysis."""
    ds = _elements(client, _si(_open(client, eels_cuzn)), ["Cu"])
    session = client.app.state.session
    signal = session.get_dataset(ds["id"]).signal
    del signal.metadata.Acquisition_instrument.TEM.convergence_angle
    assert client.get(f"/api/datasets/{ds['id']}").json()["microscope"]["convergence_angle"] is None
    assert _fit(client, ds, **HYDROGENIC).status_code == 422
    assert client.patch(f"/api/datasets/{ds['id']}/microscope", json={"convergence_angle": -1}).status_code == 422
    r = client.patch(f"/api/datasets/{ds['id']}/microscope", json={"convergence_angle": 21.5})
    assert r.status_code == 200 and r.json()["microscope"]["convergence_angle"] == 21.5
    assert r.json()["microscope"]["beam_energy"]  # the others left as they were
    assert _fit(client, ds, **HYDROGENIC).status_code == 200

    saved = client.post("/api/project/save", json={"path": str(tmp_path / "a.hspy"), "views": {}}).json()["path"]
    other = TestClient(create_app())
    assert other.post("/api/project/open", json={"path": saved}).status_code == 200
    again = other.app.state.session.get_dataset(ds["id"]).signal
    assert again.metadata.Acquisition_instrument.TEM.convergence_angle == 21.5
