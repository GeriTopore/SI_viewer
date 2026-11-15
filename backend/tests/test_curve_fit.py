"""Curve fitting (M21a): HyperSpy's Model1D on one spectrum, built from a spec."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from si_viewer.app import create_app


def _peaks(n=501):
    x = np.linspace(0, 100, n)
    clean = 50 + 400 * np.exp(-((x - 40) ** 2) / (2 * 3**2)) + 200 * np.exp(-((x - 60) ** 2) / (2 * 5**2))
    return x, clean


@pytest.fixture
def client(tmp_path):
    import hyperspy.api as hs

    x, clean = _peaks()
    rng = np.random.default_rng(0)
    # A small SI whose pixels are noisy copies of the peaks (EELS: the fit must
    # not turn into exSpy's model).
    cube = rng.poisson(np.broadcast_to(clean, (4, 5, x.size))).astype(np.float32)
    s = hs.signals.Signal1D(cube)
    s.set_signal_type("EELS")
    ax = s.axes_manager.signal_axes[0]
    ax.scale, ax.offset, ax.units = 0.2, 0.0, "eV"
    s.save(tmp_path / "si.hspy")
    c = TestClient(create_app())
    c.si = c.post("/api/files", json={"path": str(tmp_path / "si.hspy")}).json()["datasets"][0]
    return c


def _comp(name, kind, **params):
    return {"name": name, "kind": kind, "params": {k: {"value": v} for k, v in params.items()}}


def test_catalogue(client):
    cat = client.get("/api/curve-fit/components").json()
    kinds = {c["kind"]: c for c in cat["components"]}
    assert {"Gaussian", "Lorentzian", "Voigt", "PowerLaw", "Polynomial", "Offset", "Arctan"} <= set(kinds)
    assert [p["name"] for p in kinds["Gaussian"]["params"]] == ["A", "centre", "sigma"]
    assert kinds["Gaussian"]["estimate"] and kinds["Gaussian"]["group"] == "Peaks"


def test_estimate_then_fit_equals_hyperspy(client):
    import hyperspy.api as hs

    si = client.si
    model = {"components": [_comp("Peak 1", "Gaussian"), _comp("Peak 2", "Gaussian"), _comp("Background", "Offset")],
             "lo": 20, "hi": 80}
    # Estimates over a narrower range for each peak, then fit over 20–80.
    est = {}
    for name, lo, hi in (("Peak 1", 30, 50), ("Peak 2", 50, 70), ("Background", 0, 10)):
        r = client.post(f"/api/datasets/{si['id']}/curve-fit/estimate",
                        json={"model": {**model, "lo": lo, "hi": hi}, "component": name})
        assert r.status_code == 200, r.text
        est[name] = {p["name"]: p["value"] for p in r.json()["params"]}
    assert abs(est["Peak 1"]["centre"] - 40) < 1
    for c in model["components"]:
        c["params"] = {k: {"value": v} for k, v in est[c["name"]].items()}
    r = client.post(f"/api/datasets/{si['id']}/curve-fit", json={"model": model})
    assert r.status_code == 200, r.text
    got = r.json()
    assert got["pixels"] == 20 and got["fitted"] and got["poisson"] and 0.5 < got["red_chisq"] < 2
    p1 = {p["name"]: p for p in got["components"][0]["params"]}
    assert abs(p1["centre"]["value"] - 40) < 0.1 and p1["centre"]["std"] > 0
    assert len(got["x"]) == len(got["model"]) == len(got["components"][0]["curve"])
    assert got["range"] == [20, 80]

    # HyperSpy's own fit of the same sum spectrum from the same start.
    session = client.app.state.session
    data = np.asarray(session.get_dataset(si["id"]).signal.data, float).sum(axis=(0, 1))
    s = hs.signals.Signal1D(data)
    s.axes_manager[0].scale, s.axes_manager[0].is_binned = 0.2, True
    s.estimate_poissonian_noise_variance()
    m = s.create_model()
    comps = [hs.model.components1D.Gaussian(), hs.model.components1D.Gaussian(), hs.model.components1D.Offset()]
    m.extend(comps)
    for comp, name in zip(comps, ("Peak 1", "Peak 2", "Background")):
        for p in comp.parameters:
            p.value = est[name][p.name]
    m.set_signal_range(20, 80)
    m.fit()
    assert np.isclose(p1["centre"]["value"], comps[0].centre.value) and np.isclose(p1["A"]["value"], comps[0].A.value, rtol=1e-6)


def test_fixed_bounds_inactive_and_evaluate(client):
    si = client.si
    peak = _comp("Peak", "Gaussian", A=20000, centre=41, sigma=3)
    peak["params"]["centre"] = {"value": 41, "free": False}
    bg = _comp("Background", "Offset", offset=1000)
    bg["params"]["offset"]["bmax"] = 500
    other = {**_comp("Unused", "Lorentzian"), "active": False}
    r = client.post(f"/api/datasets/{si['id']}/curve-fit", json={"model": {"components": [peak, bg, other], "lo": 20, "hi": 55}})
    assert r.status_code == 200, r.text
    comps = {c["name"]: c for c in r.json()["components"]}
    assert comps["Peak"]["params"][1]["value"] == 41  # fixed
    assert comps["Background"]["params"][0]["value"] <= 500 + 1e-6  # bounded
    assert comps["Unused"]["curve"] is None
    # Evaluate only: values unchanged, no goodness of fit.
    ev = client.post(f"/api/datasets/{si['id']}/curve-fit", json={"model": {"components": [peak]}, "fit": False}).json()
    assert "fitted" in ev, ev
    assert not ev["fitted"] and "red_chisq" not in ev and ev["components"][0]["params"][0]["value"] == 20000


def test_pixel_and_polynomial(client):
    si = client.si
    model = {"components": [{"name": "Poly", "kind": "Polynomial", "order": 1}, _comp("Peak", "Gaussian", A=4000, centre=40, sigma=3)]}
    r = client.post(f"/api/datasets/{si['id']}/curve-fit",
                    json={"model": model, "source": {"type": "pixel", "x": 1, "y": 2, "size": 1}})
    assert r.status_code == 200, r.text
    got = r.json()
    assert got["pixels"] == 1 and [p["name"] for p in got["components"][0]["params"]] == ["a0", "a1"]


def test_refusals(client):
    si = client.si
    url = f"/api/datasets/{si['id']}/curve-fit"
    assert client.post(url, json={"model": {"components": []}}).status_code == 422
    assert client.post(url, json={"model": {"components": [_comp("A", "Nope")]}}).status_code == 422
    assert client.post(url, json={"model": {"components": [_comp("A", "Gaussian"), _comp("A", "Offset")]}}).status_code == 422
    assert client.post(url, json={"model": {"components": [_comp("A", "Gaussian")], "lo": 500, "hi": 600}}).status_code == 422
    fixed = _comp("A", "Offset", offset=1)
    fixed["params"]["offset"]["free"] = False
    assert client.post(url, json={"model": {"components": [fixed]}}).status_code == 422
    r = client.post(f"{url}/estimate", json={"model": {"components": [_comp("A", "Arctan")]}, "component": "A"})
    assert r.status_code == 422 and "estimate" in r.json()["detail"]


def test_fitting_test_example_recovers_the_truth():
    """The two-Gaussians fitting test (zero counts away from the peaks): the
    fit converges and finds the true centres at a pixel."""
    from si_viewer.app import create_app
    from si_viewer.examples import get_example

    client = TestClient(create_app())
    files = client.post("/api/examples/fit-two-gaussians").json()
    si = next(d for f in (files if isinstance(files, list) else [files]) for d in f["datasets"] if d["kind"] == "spectrum_image")
    truth = get_example("fit-two-gaussians").load()
    c1 = float(truth[1].data[5, 7])  # Gaussian 1 centre at x = 7, y = 5
    c2 = float(truth[3].data[5, 7])
    model = {"components": [_comp("G1", "Gaussian", A=5000, centre=256, sigma=10),
                            _comp("G2", "Gaussian", A=10000, centre=768, sigma=20)]}
    r = client.post(f"/api/datasets/{si['id']}/curve-fit", json={"model": model, "source": {"type": "pixel", "x": 7, "y": 5, "size": 1}})
    assert r.status_code == 200, r.text
    got = r.json()
    assert got["success"] and got["red_chisq"] is not None and got["red_chisq"] < 2
    p1 = {p["name"]: p for p in got["components"][0]["params"]}
    p2 = {p["name"]: p for p in got["components"][1]["params"]}
    assert abs(p1["centre"]["value"] - c1) < 4 * p1["centre"]["std"] + 0.5
    assert abs(p2["centre"]["value"] - c2) < 4 * p2["centre"]["std"] + 0.5


def test_background_window(client):
    """With a background window the Offset is fitted over the window alone
    (HyperSpy's fit there), then kept while the peaks are fitted."""
    import hyperspy.api as hs

    si = client.si
    peak = _comp("Peak", "Gaussian", A=20000, centre=40, sigma=3)
    bg = _comp("Background", "Offset", offset=1000)
    model = {"components": [bg, peak], "bg_lo": 0, "bg_hi": 20}
    r = client.post(f"/api/datasets/{si['id']}/curve-fit", json={"model": model})
    assert r.status_code == 200, r.text
    got = {c["name"]: {p["name"]: p["value"] for p in c["params"]} for c in r.json()["components"]}

    data = np.asarray(client.app.state.session.get_dataset(si["id"]).signal.data, float).sum(axis=(0, 1))
    s = hs.signals.Signal1D(data)
    s.axes_manager[0].scale, s.axes_manager[0].is_binned = 0.2, True
    s.estimate_poissonian_noise_variance()
    m = s.create_model()
    o = hs.model.components1D.Offset()
    o.offset.value = 1000
    m.append(o)
    m.set_signal_range(0, 20)
    m.fit()
    assert got["Background"]["offset"] == pytest.approx(o.offset.value, rel=1e-6)
    assert abs(got["Peak"]["centre"] - 40) < 0.1

    url = f"/api/datasets/{si['id']}/curve-fit"
    assert client.post(url, json={"model": {**model, "bg_lo": 90, "bg_hi": 200}}).status_code == 422
    assert client.post(url, json={"model": {"components": [peak], "bg_lo": 0, "bg_hi": 20}}).status_code == 422


def test_twins(client):
    """A doublet: the second peak's centre 5 eV above the first's, its area
    half of it, its width the same (HyperSpy's twins); bad links refused."""
    si = client.si
    p1 = _comp("P1", "Gaussian", A=20000, centre=39, sigma=2.5)
    p2 = _comp("P2", "Gaussian", A=1, centre=0, sigma=1)
    p2["params"]["centre"]["twin"] = {"component": "P1", "param": "centre", "expr": "x + 21"}
    p2["params"]["A"]["twin"] = {"component": "P1", "param": "A", "expr": "0.5*x"}
    p2["params"]["sigma"]["twin"] = {"component": "P1", "param": "sigma"}
    bg = _comp("Background", "Offset", offset=200)
    url = f"/api/datasets/{si['id']}/curve-fit"
    r = client.post(url, json={"model": {"components": [p1, p2, bg]}})
    assert r.status_code == 200, r.text
    got = {c["name"]: {p["name"]: p["value"] for p in c["params"]} for c in r.json()["components"]}
    assert got["P2"]["centre"] == pytest.approx(got["P1"]["centre"] + 21)
    assert got["P2"]["A"] == pytest.approx(got["P1"]["A"] / 2)
    assert got["P2"]["sigma"] == pytest.approx(got["P1"]["sigma"])
    assert abs(got["P1"]["centre"] - 40) < 0.5

    bad = {**p2, "params": {**p2["params"], "centre": {"value": 0, "twin": {"component": "Nope", "param": "centre"}}}}
    assert client.post(url, json={"model": {"components": [p1, bad]}}).status_code == 422
    loop1 = _comp("L1", "Gaussian")
    loop2 = _comp("L2", "Gaussian")
    loop1["params"]["centre"] = {"value": 1, "twin": {"component": "L2", "param": "centre"}}
    loop2["params"]["centre"] = {"value": 1, "twin": {"component": "L1", "param": "centre"}}
    assert client.post(url, json={"model": {"components": [loop1, loop2]}}).status_code == 422
    badexpr = {**p2, "params": {**p2["params"], "centre": {"value": 0, "twin": {"component": "P1", "param": "centre", "expr": "x +"}}}}
    r = client.post(url, json={"model": {"components": [p1, badexpr]}})
    assert r.status_code == 422 and "formula" in r.json()["detail"]
