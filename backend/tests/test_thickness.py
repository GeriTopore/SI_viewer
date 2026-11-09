"""Thickness of EELS low-loss spectrum images (M10d): the log-ratio estimate
against exSpy's own, the options, the dataset it makes and how it follows its
low loss."""

import time
import warnings

import exspy
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from si_viewer.app import create_app
from si_viewer.io import load_file
from si_viewer.processing import thickness as T
from si_viewer.processing.filters import Filters, filter_image, filter_spectrum
from si_viewer.processing.thickness import ThicknessOptions, thickness_map

STEP = 0.5
ENERGY = np.arange(-20.0, 60.0, STEP)  # the ZLP is at -3 eV, a plasmon at 15 eV


@pytest.fixture
def client():
    return TestClient(create_app())


def _wait(client, job, timeout=120):
    t0 = time.time()
    while job["status"] in ("queued", "running"):
        assert time.time() - t0 < timeout, "job did not finish"
        time.sleep(0.02)
        job = client.get(f"/api/jobs/{job['id']}").json()
    return job


def _open(client, path) -> dict:
    r = client.post("/api/files", json={"path": str(path)})
    assert r.status_code == 200, r.text
    return r.json()


def _si(entry) -> dict:
    return next(d for d in entry["datasets"] if d["kind"] == "spectrum_image")


def _role(client, ds, role) -> None:
    r = client.put(f"/api/datasets/{ds['id']}/eels-role", json={"role": role})
    assert r.status_code == 200, r.text


def _link(client, ds, partner) -> None:
    r = client.put(f"/api/datasets/{ds['id']}/eels-partner", json={"partner_id": partner["id"]})
    assert r.status_code == 200, r.text


def _run(client, ds, **options):
    """Estimate the thickness (POST) and wait: the job."""
    r = client.post(f"/api/datasets/{ds['id']}/thickness", json={"options": options})
    assert r.status_code == 200, r.text
    return _wait(client, r.json())


def _made(job, origin="thickness") -> list[dict]:
    return [d for d in job["result"]["file"]["datasets"] if d["origin"] == origin]


def _data(client, ds_id) -> np.ndarray:
    return np.asarray(client.app.state.session.get_dataset(ds_id).signal.data)


# ---------- synthetic low loss ----------


def _gauss(x, sigma):
    return np.exp(-0.5 * (x / sigma) ** 2)


def _spectra(shape=(4, 6)):
    """(counts, t/λ): every pixel has a ZLP of 10000 counts (a Gaussian with a
    Lorentzian tail, which gives the spectrum an inflexion to find) and a
    plasmon of r times that, so t/λ is ln(1 + r)."""
    zlp = 0.8 * _gauss(ENERGY + 3, 0.4) / _gauss(ENERGY + 3, 0.4).sum()
    tail = 0.5 / ((ENERGY + 3) ** 2 + 0.25)
    zlp = zlp + 0.2 * tail / tail.sum()
    plasmon = _gauss(ENERGY - 15, 3) / _gauss(ENERGY - 15, 3).sum()
    n = int(np.prod(shape))
    r = np.linspace(0.2, 1.5, n).reshape(shape)
    counts = 1e4 * (zlp + r[..., None] * plasmon)
    return counts.astype("float32"), np.log1p(r)


def _low_loss(shape=(4, 6), title="ll", binned=True, angles=True, units="nm"):
    counts, truth = _spectra(shape)
    s = exspy.signals.EELSSpectrum(counts)
    s.metadata.General.title = title
    ax = s.axes_manager.signal_axes[0]
    ax.offset, ax.scale, ax.units, ax.name, ax.is_binned = float(ENERGY[0]), STEP, "eV", "Energy loss", binned
    for nav in s.axes_manager.navigation_axes:
        nav.scale, nav.units = 2.0, units
    s.metadata.set_item("Acquisition_instrument.TEM.beam_energy", 200.0)
    if angles:
        s.metadata.set_item("Acquisition_instrument.TEM.convergence_angle", 10.0)
        s.metadata.set_item("Acquisition_instrument.TEM.Detector.EELS.collection_angle", 20.0)
    return s, truth


def _core_loss(shape=(4, 6), title="hl"):
    rng = np.random.default_rng(1)
    energy = np.arange(400.0, 480.0, 1.0)
    data = (50 * np.exp(-(energy - 400) / 40) + rng.random(tuple(shape) + (len(energy),))).astype("float32")
    s = exspy.signals.EELSSpectrum(data)
    s.metadata.General.title = title
    ax = s.axes_manager.signal_axes[0]
    ax.offset, ax.scale, ax.units, ax.name = 400.0, 1.0, "eV", "Energy loss"
    for nav in s.axes_manager.navigation_axes:
        nav.scale, nav.units = 2.0, "nm"
    s.metadata.set_item("Acquisition_instrument.TEM.beam_energy", 200.0)
    return s


def _save(tmp_path, s, name):
    path = tmp_path / f"{name}.hspy"
    s.save(path)
    return path


def _open_low_loss(client, tmp_path, **kw):
    s, truth = _low_loss(**kw)
    si = _si(_open(client, _save(tmp_path, s, kw.get("title", "ll"))))
    _role(client, si, "low_loss")
    return s, truth, si


def _direct(s, threshold="auto", **kw):
    """exSpy's own estimate on the signal ``s``."""
    if threshold == "auto":
        threshold = s.estimate_elastic_scattering_threshold(window=kw.pop("window", 10.0))
    return np.asarray(s.estimate_thickness(threshold=threshold, **kw).data, float)


# ---------- options ----------


def test_options_are_validated():
    o = ThicknessOptions()
    assert (o.threshold, o.window, o.output, o.mean_free_path, o.density) == ("auto", 10.0, "relative", None, None)
    assert ThicknessOptions(threshold=7).threshold == 7.0
    # Relative thickness takes no mean free path or density.
    assert ThicknessOptions(mean_free_path=100, density=3).mean_free_path is None
    assert ThicknessOptions(output="absolute", mean_free_path=120).title == "Thickness (nm)"
    assert ThicknessOptions().title == "t/λ"
    for bad in (
        dict(threshold="soon"),
        dict(threshold=float("nan")),
        dict(threshold=float("inf")),
        dict(window=0),
        dict(window=-3),
        dict(window=float("nan")),
        dict(output="absolute"),  # neither
        dict(output="absolute", mean_free_path=100, density=3),  # both
        dict(output="absolute", mean_free_path=0),
        dict(output="absolute", density=-1),
        dict(output="nm"),
    ):
        with pytest.raises(ValidationError):
            ThicknessOptions(**bad)


def test_options_must_fit_the_spectra():
    s, _ = _low_loss()
    T.check_options(s, ThicknessOptions())
    T.check_options(s, ThicknessOptions(threshold=6.0, output="absolute", density=5))
    with pytest.raises(ValueError, match="threshold must lie inside the spectrum's energy range"):
        T.check_options(s, ThicknessOptions(threshold=200.0))
    with pytest.raises(ValueError, match="threshold must lie inside"):
        T.check_options(s, ThicknessOptions(threshold=-30.0))
    with pytest.raises(ValueError, match="above the first channel"):
        T.check_options(s, ThicknessOptions(threshold=-20.0))
    with pytest.raises(ValueError, match="searches 1–111 eV, outside the spectrum's energy range"):
        T.check_options(s, ThicknessOptions(window=110))
    with pytest.raises(ValueError, match="search window must be at least 5 eV"):
        T.check_options(s, ThicknessOptions(window=3))
    bare, _ = _low_loss(angles=False)
    T.check_options(bare, ThicknessOptions(output="absolute", mean_free_path=100))
    with pytest.raises(ValueError, match="density needs the convergence angle, collection angle"):
        T.check_options(bare, ThicknessOptions(output="absolute", density=5))
    # A core loss: the zero-loss peak is out of range.
    with pytest.raises(ValueError, match="outside the spectrum's energy range"):
        T.check_options(_core_loss(), ThicknessOptions())


# ---------- the estimate ----------


@pytest.mark.parametrize("binned", [True, False])
def test_auto_and_fixed_equal_exspy(binned):
    s, truth = _low_loss(binned=binned)
    thr = s.estimate_elastic_scattering_threshold(window=10.0)
    assert np.isfinite(thr.data).all() and 3 <= thr.data.min() and thr.data.max() <= 7

    auto = thickness_map(s, ThicknessOptions())
    assert auto.shape == (4, 6) and auto.dtype == np.float32
    # exSpy on binned counts (it mixes sums and integrals on an unbinned axis).
    s.axes_manager.signal_axes[0].is_binned = True
    np.testing.assert_allclose(auto, _direct(s, thr), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(auto, truth, atol=0.01)  # t/λ = ln(I_total / I_zlp)

    for threshold in (5.0, 6.5, -1.0):
        fixed = thickness_map(s, ThicknessOptions(threshold=threshold))
        np.testing.assert_allclose(fixed, _direct(s, threshold), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(thickness_map(s, ThicknessOptions(threshold=6.0)), truth, atol=0.01)


def test_ratio_of_the_zero_loss_peak_to_everything():
    s, _ = _low_loss()
    counts = np.asarray(s.data, float)
    below = counts[..., ENERGY < 6.0].sum(-1)
    expected = np.log(counts.sum(-1) / below)
    np.testing.assert_allclose(thickness_map(s, ThicknessOptions(threshold=6.0)), expected, rtol=1e-5)


def test_the_threshold_is_estimated_over_all_the_pixels_whatever_the_chunks(monkeypatch):
    s, _ = _low_loss(shape=(5, 7))
    whole = thickness_map(s, ThicknessOptions())
    for chunk in (1, 4, 6):  # the last chunk is short
        monkeypatch.setattr(T, "CHUNK", chunk)
        np.testing.assert_array_equal(thickness_map(s, ThicknessOptions()), whole)
        fixed = thickness_map(s, ThicknessOptions(threshold=6.0))
    monkeypatch.setattr(T, "CHUNK", 2048)
    np.testing.assert_array_equal(thickness_map(s, ThicknessOptions(threshold=6.0)), fixed)
    np.testing.assert_allclose(whole, _direct(s), rtol=1e-6, atol=1e-6)


def test_absolute_thickness(client):
    s, truth = _low_loss()
    relative = thickness_map(s, ThicknessOptions(threshold=6.0))
    mfp = thickness_map(s, ThicknessOptions(threshold=6.0, output="absolute", mean_free_path=120.0))
    np.testing.assert_allclose(mfp, relative * 120.0, rtol=1e-6)
    np.testing.assert_allclose(mfp, _direct(s, 6.0, mean_free_path=120.0), rtol=1e-6)

    # A density: the Iakoubovskii mean free path and the angular correction.
    dens = thickness_map(s, ThicknessOptions(threshold=6.0, output="absolute", density=5.0))
    np.testing.assert_allclose(dens, _direct(s, 6.0, density=5.0), rtol=1e-6)
    assert (dens > 0).all() and not np.allclose(dens, relative)
    auto = thickness_map(s, ThicknessOptions(output="absolute", density=5.0))
    np.testing.assert_allclose(auto, _direct(s, "auto", density=5.0), rtol=1e-6)


def test_the_processed_data_is_used():
    s, _ = _low_loss()
    # Spectral smoothing: the smoothed spectra.
    f = Filters(spectral="gaussian", spectral_sigma=1.5)
    smoothed = s.deepcopy()
    smoothed.data = filter_spectrum(np.asarray(s.data), f)
    np.testing.assert_allclose(
        thickness_map(s, ThicknessOptions(threshold=6.0), f), _direct(smoothed, 6.0), rtol=1e-6, atol=1e-6,
    )
    np.testing.assert_allclose(
        thickness_map(s, ThicknessOptions(), f), _direct(smoothed), rtol=1e-6, atol=1e-6,
    )
    # A spatial filter blurs the map (NaN stays out of its neighbours).
    g = Filters(spatial="gaussian", sigma=1.0)
    raw = thickness_map(s, ThicknessOptions(threshold=6.0))
    np.testing.assert_allclose(thickness_map(s, ThicknessOptions(threshold=6.0), g), filter_image(raw, g), rtol=1e-6)


def test_pixels_without_counts_have_no_thickness():
    s, truth = _low_loss()
    s.data[1, 2, :] = 0
    for options in (ThicknessOptions(threshold=6.0), ThicknessOptions()):
        out = thickness_map(s, options)
        assert np.isnan(out[1, 2])
        other = np.ones((4, 6), bool)
        other[1, 2] = False
        np.testing.assert_allclose(out[other], truth[other], atol=0.01)


def test_progress_is_reported():
    s, _ = _low_loss()
    seen = []
    thickness_map(s, ThicknessOptions(), report=lambda f, m=None: seen.append((f, m)))
    fractions = [f for f, _ in seen]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0
    assert {m for _, m in seen} == {"Finding the threshold", "Estimating the thickness"}


# ---------- the dataset it makes ----------


def test_apply_makes_a_map_under_the_low_loss(client, tmp_path):
    s, truth, si = _open_low_loss(client, tmp_path)
    job = _run(client, si, threshold=6.0)
    assert job["status"] == "done", job
    (t,) = _made(job)
    assert t["id"] == job["result"]["dataset_id"]
    assert (t["kind"], t["parent_id"], t["origin"], t["title"]) == ("image", si["id"], "thickness", "t/λ")
    assert t["navigation_shape"] == [] and t["signal_shape"] == [6, 4]  # x, y
    assert t["thickness"] == {
        "source_id": si["id"], "threshold": 6.0, "window": 10.0, "output": "relative",
        "mean_free_path": None, "density": None, "units": "eV",
    }
    assert t["eels"] is None and t["technique"] is None and not t["is_element_map"]
    # The scan's calibration comes with it.
    assert [a["units"] for a in t["axes"]] == ["nm", "nm"] and [a["scale"] for a in t["axes"]] == [2.0, 2.0]
    np.testing.assert_allclose(_data(client, t["id"]), truth, atol=0.01)
    np.testing.assert_allclose(_data(client, t["id"]), thickness_map(s, ThicknessOptions(threshold=6.0)), rtol=1e-6)
    assert client.get(f"/api/datasets/{t['id']}/frame").status_code == 200
    assert client.get(f"/api/datasets/{t['id']}/thumbnail").status_code == 200


def test_auto_threshold_is_recorded(client, tmp_path):
    s, _, si = _open_low_loss(client, tmp_path)
    t = _made(_run(client, si, window=8.0))[0]
    assert t["thickness"]["threshold"] == "auto" and t["thickness"]["window"] == 8.0
    np.testing.assert_allclose(_data(client, t["id"]), thickness_map(s, ThicknessOptions(window=8.0)), rtol=1e-6)


def test_absolute_output_through_the_api(client, tmp_path):
    s, truth, si = _open_low_loss(client, tmp_path)
    job = _run(client, si, threshold=6.0, output="absolute", mean_free_path=120.0)
    (t,) = _made(job)
    assert t["title"] == "Thickness (nm)"
    assert (t["thickness"]["output"], t["thickness"]["mean_free_path"], t["thickness"]["density"]) == ("absolute", 120.0, None)
    np.testing.assert_allclose(_data(client, t["id"]), truth * 120.0, atol=0.01 * 120.0)

    job = _run(client, si, threshold=6.0, output="absolute", density=5.0)
    (t2,) = _made(job)
    assert t2["id"] == t["id"] and t2["thickness"]["density"] == 5.0 and t2["thickness"]["mean_free_path"] is None
    np.testing.assert_allclose(_data(client, t["id"]), _direct(s, 6.0, density=5.0), rtol=1e-6)


def test_running_again_replaces_it(client, tmp_path):
    _, _, si = _open_low_loss(client, tmp_path)
    first = _run(client, si, threshold=6.0)
    (t,) = _made(first)
    assert t["revision"] == 0
    before = _data(client, t["id"]).copy()

    second = _run(client, si, threshold=4.5, output="absolute", mean_free_path=50.0)
    (again,) = _made(second)  # still the only one
    assert again["id"] == t["id"] and again["revision"] == 1
    assert again["title"] == "Thickness (nm)" and again["thickness"]["threshold"] == 4.5
    assert not np.allclose(_data(client, t["id"]), before)
    assert len([d for d in second["result"]["file"]["datasets"] if d["origin"] == "thickness"]) == 1

    # A name the user gave is kept.
    client.app.state.session.get_dataset(t["id"]).signal.metadata.General.title = "my thickness"
    third = _run(client, si, threshold=6.0)
    assert _made(third)[0]["title"] == "my thickness"


def test_progress_and_cancel(client, tmp_path, monkeypatch):
    _, _, si = _open_low_loss(client, tmp_path, shape=(20, 20))
    monkeypatch.setattr(T, "CHUNK", 25)
    r = client.post(f"/api/datasets/{si['id']}/thickness", json={"options": {}})
    job = r.json()
    seen = set()
    while job["status"] in ("queued", "running"):
        seen.add(round(job["progress"], 1))
        time.sleep(0.02)
        job = client.get(f"/api/jobs/{job['id']}").json()
    assert job["status"] == "done" and len(seen) > 2, seen

    session = client.app.state.session
    (old,) = session.thickness_maps(session.get_dataset(si["id"]))
    monkeypatch.setattr(T, "CHUNK", 1)  # 800 steps
    job = client.post(f"/api/datasets/{si['id']}/thickness", json={"options": {"window": 9.0}}).json()
    client.post(f"/api/jobs/{job['id']}/cancel")
    job = _wait(client, job)
    assert job["status"] == "cancelled"
    # The earlier result is as it was.
    (still,) = session.thickness_maps(session.get_dataset(si["id"]))
    assert still is old and old.revision == 0 and old.thickness["threshold"] == "auto"


# ---------- errors ----------


def test_only_eels_spectrum_images_with_a_low_loss(client, tmp_path):
    for d in client.post("/api/examples/fept-si").json()["datasets"]:  # EDS
        r = client.post(f"/api/datasets/{d['id']}/thickness", json={"options": {}})
        assert r.status_code == 400, (d["title"], r.text)
        assert r.json()["detail"] == f"{d['title']} is not an EELS spectrum image"
        assert client.post(f"/api/datasets/{d['id']}/thickness/threshold", json={}).status_code == 400
    assert client.post("/api/datasets/nope/thickness", json={"options": {}}).status_code == 404

    # A core loss with no low loss in its group.
    si = _si(_open(client, _save(tmp_path, _core_loss(), "alone")))
    for url in ("thickness", "thickness/threshold"):
        r = client.post(f"/api/datasets/{si['id']}/{url}", json={"options": {}})
        assert r.status_code == 400 and r.json()["detail"] == "No low loss in its group", r.text


def test_bad_options_are_refused(client, tmp_path):
    _, _, si = _open_low_loss(client, tmp_path)
    url = f"/api/datasets/{si['id']}/thickness"
    for options, message in (
        ({"threshold": 500.0}, "threshold must lie inside the spectrum's energy range (-20–59.5 eV)"),
        ({"threshold": -20.0}, "above the first channel"),
        ({"window": 150.0}, "outside the spectrum's energy range"),
        ({"output": "absolute"}, "mean free path or a density"),
        ({"output": "absolute", "mean_free_path": 1, "density": 1}, "not both or neither"),
        ({"output": "absolute", "mean_free_path": -1}, "greater than 0"),
        ({"threshold": "soon"}, "threshold"),
        ({"threshold": "NaN"}, "must be a number"),
    ):
        r = client.post(url, json={"options": options})
        assert r.status_code == 422, (options, r.text)
        assert message in r.text, (options, r.text)
    r = client.post(url, json={"options": {"threshold": None}})  # what a JavaScript NaN becomes
    assert r.status_code == 422, r.text
    assert not any(d.origin == "thickness" for d in client.app.state.session.datasets.values())


def test_a_density_needs_the_microscope_parameters(client, tmp_path):
    s, _ = _low_loss(angles=False)
    si = _si(_open(client, _save(tmp_path, s, "bare")))
    _role(client, si, "low_loss")
    r = client.post(f"/api/datasets/{si['id']}/thickness", json={"options": {"output": "absolute", "density": 5}})
    assert r.status_code == 422 and "density needs the convergence angle, collection angle" in r.text
    job = _run(client, si, output="absolute", mean_free_path=100.0)
    assert job["status"] == "done"


# ---------- the threshold of the summed spectrum ----------


def test_threshold_of_the_summed_spectrum(client, tmp_path):
    s, _, si = _open_low_loss(client, tmp_path)
    total = s.sum()  # the summed spectrum, as a 1D signal
    expected = float(total.estimate_elastic_scattering_threshold(window=10.0).data.ravel()[0])
    r = client.post(f"/api/datasets/{si['id']}/thickness/threshold", json={"window": 10.0})
    assert r.status_code == 200, r.text
    assert r.json() == {"low_loss_id": si["id"], "threshold": pytest.approx(expected)}
    assert 3 <= expected <= 7

    assert client.post(f"/api/datasets/{si['id']}/thickness/threshold", json={}).json()["threshold"] == pytest.approx(expected)
    r = client.post(f"/api/datasets/{si['id']}/thickness/threshold", json={"window": 400})
    assert r.status_code == 422 and "outside the spectrum's energy range" in r.text
    assert client.post(f"/api/datasets/{si['id']}/thickness/threshold", json={"window": 0}).status_code == 422

    # The summed spectrum as shown: spectrally smoothed.
    f = {"spatial": "none", "sigma": 1.0, "size": 3, "spectral": "gaussian", "spectral_sigma": 2.0,
         "window": 7, "order": 2}
    assert client.put(f"/api/datasets/{si['id']}/filters", json=f).status_code == 200
    smoothed = total.deepcopy()
    smoothed.data = filter_spectrum(np.asarray(total.data), Filters(**f))
    expected = float(smoothed.estimate_elastic_scattering_threshold(window=10.0).data.ravel()[0])
    r = client.post(f"/api/datasets/{si['id']}/thickness/threshold", json={})
    assert r.json()["threshold"] == pytest.approx(expected)


def test_no_inflexion_gives_no_threshold(client, tmp_path):
    s, _ = _low_loss()
    s.data[:] = 0  # nothing to find
    si = _si(_open(client, _save(tmp_path, s, "empty")))
    _role(client, si, "low_loss")
    r = client.post(f"/api/datasets/{si['id']}/thickness/threshold", json={})
    assert r.status_code == 200 and r.json()["threshold"] is None


# ---------- following the low loss ----------


def test_a_linked_core_loss_runs_it_on_the_low_loss(client, tmp_path):
    ll_s, truth = _low_loss()
    ll = _si(_open(client, _save(tmp_path, ll_s, "ll")))
    hl = _si(_open(client, _save(tmp_path, _core_loss(), "hl")))
    _role(client, ll, "low_loss")
    _link(client, hl, ll)

    r = client.post(f"/api/datasets/{hl['id']}/thickness/threshold", json={})
    assert r.status_code == 200 and r.json()["low_loss_id"] == ll["id"]
    job = _run(client, hl, threshold=6.0)  # from the core loss
    assert job["status"] == "done", job
    (t,) = _made(job)
    assert t["parent_id"] == ll["id"] and t["thickness"]["source_id"] == ll["id"]
    np.testing.assert_allclose(_data(client, t["id"]), truth, atol=0.01)
    # It replaces the low loss's own, whichever of the two it is run from.
    job = _run(client, ll, threshold=5.0)
    assert _made(job)[0]["id"] == t["id"] and len(_made(job)) == 1
    listed = {d["id"]: d for f in client.get("/api/files").json() for d in f["datasets"]}
    assert listed[hl["id"]]["eels"]["linked"] == [ll["id"]]


def test_rebinning_the_low_loss_recomputes_it(client, tmp_path):
    s, _, ll = _open_low_loss(client, tmp_path, shape=(6, 8))
    hl = _si(_open(client, _save(tmp_path, _core_loss(shape=(6, 8)), "hl")))
    _link(client, hl, ll)
    auto = _made(_run(client, ll))[0]
    assert _data(client, auto["id"]).shape == (6, 8)

    # Rebinned from the core loss: the whole group is.
    r = client.post(f"/api/datasets/{hl['id']}/rebin", json={"factors": [2, 2, 1]})
    assert r.status_code == 200, r.text
    files = {d["id"]: d for f in client.get("/api/files").json() for d in f["datasets"]}
    t = files[auto["id"]]
    assert t["signal_shape"] == [4, 3] and t["revision"] == 1 and t["thickness"]["threshold"] == "auto"
    rebinned = client.app.state.session.get_dataset(ll["id"]).signal
    assert rebinned.data.shape == (3, 4, len(ENERGY))
    np.testing.assert_allclose(_data(client, t["id"]), _direct(rebinned), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(_data(client, t["id"]), thickness_map(rebinned, ThicknessOptions()), rtol=1e-6)

    # Rebinning the low loss itself does too, and a threshold that stays in range is kept.
    fixed = _made(_run(client, ll, threshold=6.0))[0]
    assert fixed["id"] == auto["id"] and fixed["revision"] == 2  # replaced, not added
    r = client.post(f"/api/datasets/{ll['id']}/rebin", json={"factors": [1, 2, 1]})
    assert r.status_code == 200, r.text
    again = {d["id"]: d for d in r.json()["file"]["datasets"]}[fixed["id"]]
    assert again["signal_shape"] == [2, 3] and again["revision"] == 3 and again["thickness"]["threshold"] == 6.0


def test_a_rebin_that_leaves_no_room_removes_it(client, tmp_path):
    _, _, ll = _open_low_loss(client, tmp_path, shape=(4, 4))
    t = _made(_run(client, ll, window=58.0))[0]  # searches 1–59 eV: just inside -20–59.5
    assert t["thickness"]["window"] == 58.0
    # 4 eV channels: the spectrum then ends at 56 eV.
    r = client.post(f"/api/datasets/{ll['id']}/rebin", json={"factors": [1, 1, 8]})
    assert r.status_code == 200, r.text
    assert t["id"] not in {d["id"] for d in r.json()["file"]["datasets"]}


def test_changing_the_filters_recomputes_it(client, tmp_path):
    s, _, ll = _open_low_loss(client, tmp_path)
    t = _made(_run(client, ll, threshold=6.0))[0]
    plain = _data(client, t["id"]).copy()
    spatial = {"spatial": "gaussian", "sigma": 1.0, "size": 3, "spectral": "none", "spectral_sigma": 1.0, "window": 7, "order": 2}
    r = client.put(f"/api/datasets/{ll['id']}/filters", json=spatial)
    assert r.status_code == 200, r.text
    after = {d["id"]: d for d in r.json()["file"]["datasets"]}[t["id"]]
    assert after["revision"] == 1
    np.testing.assert_allclose(_data(client, t["id"]), filter_image(plain, Filters(**spatial)), rtol=1e-6)

    spectral = {**spatial, "spatial": "none", "spectral": "savgol", "window": 9, "order": 3}
    assert client.put(f"/api/datasets/{ll['id']}/filters", json=spectral).status_code == 200
    smoothed = s.deepcopy()
    smoothed.data = filter_spectrum(np.asarray(s.data), Filters(**spectral))
    np.testing.assert_allclose(_data(client, t["id"]), _direct(smoothed, 6.0), rtol=1e-5, atol=1e-6)
    # The map itself can't be filtered on its own.
    r = client.put(f"/api/datasets/{t['id']}/filters", json=spatial)
    assert r.status_code == 422 and "follows the processing of" in r.text


def test_delete_and_close(client, tmp_path):
    s, _ = _low_loss()
    entry = _open(client, _save(tmp_path, s, "ll"))
    ll = _si(entry)
    _role(client, ll, "low_loss")
    session = client.app.state.session
    before = {d["id"] for d in entry["datasets"]}
    t = _made(_run(client, ll, threshold=6.0))[0]

    r = client.delete(f"/api/datasets/{t['id']}")
    assert r.status_code == 200, r.text
    assert {d["id"] for d in r.json()["file"]["datasets"]} == before
    assert t["id"] not in session.datasets and ll["id"] in session.datasets

    t = _made(_run(client, ll, threshold=6.0))[0]
    assert t["id"] in session.datasets
    assert client.delete(f"/api/files/{entry['id']}").status_code in (200, 204)
    for gone in (t["id"], ll["id"]):
        assert gone not in session.datasets
        assert client.get(f"/api/datasets/{gone}").status_code == 404


def test_deleting_the_source_takes_it(client, tmp_path):
    # A low loss made by the app (a background-removed one) can be deleted.
    s, _ = _low_loss()
    entry = _open(client, _save(tmp_path, s, "ll"))
    ll = _si(entry)
    bg = client.post(f"/api/datasets/{ll['id']}/background", json={"options": {
        "model": "offset", "background": [20, 30], "signal": [31, 50]}}).json()
    bg = _wait(client, bg)
    assert bg["status"] == "done", bg
    bg_id = bg["result"]["dataset_id"]
    _role(client, {"id": bg_id}, "low_loss")
    t = _made(_run(client, {"id": bg_id}, threshold=35.0))
    assert t and t[0]["parent_id"] == bg_id
    r = client.delete(f"/api/datasets/{bg_id}")
    assert r.status_code == 200 and t[0]["id"] not in {d["id"] for d in r.json()["file"]["datasets"]}


def test_it_goes_when_the_source_stops_being_a_low_loss(client, tmp_path):
    _, _, ll = _open_low_loss(client, tmp_path)
    t = _made(_run(client, ll, threshold=6.0))[0]
    _role(client, ll, "core_loss")
    assert t["id"] not in client.app.state.session.datasets

    _role(client, ll, "low_loss")
    t = _made(_run(client, ll, threshold=6.0))[0]
    r = client.put(f"/api/datasets/{ll['id']}/technique", json={"technique": None})
    assert r.status_code == 200, r.text
    assert t["id"] not in {d["id"] for d in r.json()["file"]["datasets"]}


# ---------- line scans ----------


def _load_exspy(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = load_file(path, lazy=False)[0]
    s.set_signal_type("EELS")
    return s


def test_line_scan_gives_a_profile(client, eels_linescan_low_loss):
    si = _si(_open(client, eels_linescan_low_loss))
    assert si["line_scan"]
    _role(client, si, "low_loss")
    job = _run(client, si)
    assert job["status"] == "done", job
    (t,) = _made(job)
    assert (t["kind"], t["origin"], t["parent_id"], t["title"]) == ("spectrum", "thickness", si["id"], "t/λ")
    assert t["signal_shape"] == [40] and t["navigation_shape"] == []
    (axis,) = t["axes"]
    assert axis["size"] == 40 and axis["units"] == "µm" and axis["scale"] == pytest.approx(0.003219, rel=1e-3)
    s = _load_exspy(eels_linescan_low_loss)
    np.testing.assert_allclose(_data(client, t["id"]), _direct(s), rtol=1e-6, atol=1e-6)
    r = client.get(f"/api/datasets/{t['id']}/spectrum")
    assert r.status_code == 200 and r.headers["X-Array-Shape"] == "40"
    assert client.get(f"/api/datasets/{t['id']}/thumbnail").status_code == 200

    # The threshold of the summed spectrum, and a fixed one.
    r = client.post(f"/api/datasets/{si['id']}/thickness/threshold", json={})
    assert r.status_code == 200 and 1 <= r.json()["threshold"] <= 11
    fixed = _made(_run(client, si, threshold=7.0, output="absolute", mean_free_path=100.0))[0]
    assert fixed["id"] == t["id"] and fixed["title"] == "Thickness (nm)"
    np.testing.assert_allclose(_data(client, t["id"]), 100 * _direct(s, 7.0), rtol=1e-6)


# ---------- the example data ----------


def test_dual_eels_low_loss(client, eels_dm4_ll):
    si = _si(_open(client, eels_dm4_ll))
    assert si["eels"]["role"] == "low_loss" and si["navigation_shape"] == [96, 415]
    # Rebinned first, to keep it quick.
    r = client.post(f"/api/datasets/{si['id']}/rebin", json={"factors": [16, 16, 1]})
    assert r.status_code == 200, r.text
    small = r.json()["dataset"]
    assert small["navigation_shape"] == [6, 25]

    job = _run(client, small)
    assert job["status"] == "done", job
    (t,) = _made(job)
    values = _data(client, t["id"])
    assert values.shape == (25, 6)
    finite = values[np.isfinite(values)]
    assert finite.size > values.size // 2 and (finite > 0).all() and finite.max() < 5
    ds = client.app.state.session.get_dataset(si["id"])
    # (Float32 sums of 3456 channels, in a different order.)
    np.testing.assert_allclose(values, _direct(ds.signal), rtol=1e-4, atol=1e-4)

    fixed = _made(_run(client, small, threshold=1.0))[0]
    assert np.isfinite(_data(client, fixed["id"])).all() and (_data(client, fixed["id"]) > 0).all()
    threshold = client.post(f"/api/datasets/{si['id']}/thickness/threshold", json={}).json()["threshold"]
    assert threshold is None or 1 <= threshold <= 11
