"""EELS background removal (M10e): the fits against HyperSpy's, the API."""

import time
import warnings

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from si_viewer.app import create_app
from si_viewer.io import load_file
from si_viewer.processing.background import (
    BackgroundOptions, EnergyAxis, channel_windows, describe, fit_block, lba_window, remove_background,
)
from si_viewer.processing.regions import window_label

CUZN = {"background": [800, 920], "signal": [925, 1010]}
LSMO = {"background": [580, 630], "signal": [635, 690]}


@pytest.fixture
def client():
    return TestClient(create_app())


def _open(client, path) -> dict:
    r = client.post("/api/files", json={"path": str(path)})
    assert r.status_code == 200, r.text
    return r.json()


def _si(entry) -> dict:
    return next(d for d in entry["datasets"] if d["kind"] == "spectrum_image")


def _wait(client, job, timeout=120):
    t0 = time.time()
    while job["status"] in ("queued", "running"):
        assert time.time() - t0 < timeout, "job did not finish"
        time.sleep(0.02)
        job = client.get(f"/api/jobs/{job['id']}").json()
    return job


def _remove(client, ds, **options):
    """Remove the background (POST) and wait: the job."""
    r = client.post(f"/api/datasets/{ds['id']}/background", json={"options": options})
    assert r.status_code == 200, r.text
    return _wait(client, r.json())


def _children(file, parent, origin):
    return [d for d in file["datasets"] if d["parent_id"] == parent["id"] and d["origin"] == origin]


def _data(client, ds) -> np.ndarray:
    return np.asarray(client.app.state.session.get_dataset(ds["id"]).signal.data)


def _signal(path):
    """The first signal of a file, in memory, as EELS."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = load_file(path, lazy=False)[0]
    s.set_signal_type("EELS")
    return s


# ---------- options ----------


def test_options_are_validated_and_sorted():
    o = BackgroundOptions(background=[920, 800], signal=[1010, 925])
    assert o.background == (800, 920) and o.signal == (925, 1010)
    assert (o.model, o.order, o.fast) == ("power_law", 2, True)
    for bad in (
        dict(background=[900, 900], signal=[925, 1010]),  # empty window
        dict(background=[800, 920], signal=[910, 1010]),  # overlap
        dict(background=[800, float("nan")], signal=[925, 1010]),
        dict(background=[800, float("inf")], signal=[925, 1010]),
        dict(model="gaussian", background=[800, 920], signal=[925, 1010]),
        dict(model="polynomial", order=0, background=[800, 920], signal=[925, 1010]),
        dict(model="polynomial", order=7, background=[800, 920], signal=[925, 1010]),
    ):
        with pytest.raises(ValidationError):
            BackgroundOptions(**bad)
    with pytest.raises(ValidationError, match="must end before the signal window starts"):
        BackgroundOptions(background=[800, 930], signal=[925, 1010])
    # Meeting at one energy is fine.
    BackgroundOptions(background=[800, 925], signal=[925, 1010])


def test_windows_must_fit_the_spectra(eels_cuzn):
    axis = EnergyAxis.of(_signal(eels_cuzn))
    assert (axis.offset, axis.scale, axis.size, axis.units) == (700.0, 8.0, 162, "eV")
    ok = channel_windows(axis, BackgroundOptions(**CUZN))
    assert (ok.b0, ok.b1, ok.s0, ok.s1) == (12, 28, 28, 39)
    with pytest.raises(ValueError, match="background window must lie inside the spectrum's energy range"):
        channel_windows(axis, BackgroundOptions(background=[600, 920], signal=[925, 1010]))
    with pytest.raises(ValueError, match="signal window must lie inside"):
        channel_windows(axis, BackgroundOptions(background=[800, 920], signal=[925, 2100]))
    with pytest.raises(ValueError, match="background window needs at least 3 channels for a power law"):
        channel_windows(axis, BackgroundOptions(background=[800, 812], signal=[925, 1010]))
    with pytest.raises(ValueError, match="background window needs at least 4 channels for a polynomial"):
        channel_windows(axis, BackgroundOptions(model="polynomial", order=3, background=[800, 816], signal=[925, 1010]))
    with pytest.raises(ValueError, match="signal window needs at least 2 channels"):
        channel_windows(axis, BackgroundOptions(background=[800, 920], signal=[925, 929]))
    channel_windows(axis, BackgroundOptions(model="offset", background=[800, 816], signal=[925, 1010]))


# ---------- the fits equal HyperSpy's ----------

HS_NAMES = {"power_law": "Power law", "exponential": "Exponential", "polynomial": "Polynomial", "offset": "Offset"}


def _hyperspy(s, win, axis, model, order, fast):
    """HyperSpy's ``remove_background`` over the same windows, cut to the signal
    window. The fast estimates cover the channels [b0, b1) as ours do. HyperSpy's
    full fit is over the channels up to and including its upper limit."""
    x_hi = axis.energies(win.b1 if fast else win.b1 - 1, 1)[0]
    ref = s.remove_background(
        signal_range=(axis.energies(win.b0, 1)[0], x_hi), background_type=HS_NAMES[model],
        polynomial_order=order, fast=fast, plot_remainder=False, show_progressbar=False, display=False,
    )
    return np.asarray(ref.isig[win.s0 : win.s1].data, float)


@pytest.mark.parametrize("model, order, fast", [
    ("power_law", 2, True), ("power_law", 2, False),
    ("exponential", 2, True), ("exponential", 2, False),
    ("polynomial", 2, True), ("polynomial", 3, True), ("polynomial", 2, False),
    ("offset", 2, True),
])
@pytest.mark.parametrize("name", ["eels_cuzn", "eels_linescan"])
def test_fit_equals_hyperspy(request, name, model, order, fast):
    s = _signal(request.getfixturevalue(name))
    windows = CUZN if name == "eels_cuzn" else LSMO
    if name == "eels_linescan":
        s = s.inav[:12]  # the full fits are slow
    else:
        s = s.inav[:10, :8]
    o = BackgroundOptions(model=model, order=order, fast=fast, **windows)
    axis = EnergyAxis.of(s)
    win = channel_windows(axis, o)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ours, r = remove_background(s, o)
        ref = _hyperspy(s, win, axis, model, order, fast)
    assert ours.data.shape == ref.shape == (*s.data.shape[:-1], win.s1 - win.s0)
    counts = np.abs(np.asarray(s.data, float)[..., win.b0 : win.b1]).mean()
    # Fast estimates are the same numbers (HyperSpy takes the logs of an exponential
    # in the data's own, single, precision); full fits end at the same minimum.
    tolerance = 5e-3 if not fast else 5e-5 if model == "exponential" else 2e-6
    np.testing.assert_allclose(ours.data, ref, atol=tolerance * counts)
    if model == "power_law":
        assert r.shape == s.data.shape[:-1] and np.isfinite(r).all()
    else:
        assert r is None


def test_power_law_exponent_equals_hyperspy(eels_cuzn):
    s = _signal(eels_cuzn).inav[:10, :8]
    o = BackgroundOptions(**CUZN)
    axis = EnergyAxis.of(s)
    win = channel_windows(axis, o)
    _, r = remove_background(s, o)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, model = s.remove_background(
            signal_range=(axis.energies(win.b0, 1)[0], axis.energies(win.b1, 1)[0]), fast=True,
            plot_remainder=False, show_progressbar=False, return_model=True, display=False,
        )
    np.testing.assert_allclose(r, model[0].r.map["values"], rtol=1e-5)


def test_a_pixel_without_counts_does_not_spoil_the_others(eels_cuzn):
    s = _signal(eels_cuzn).inav[:4, :4]
    s.data = np.asarray(s.data, float)
    clean = remove_background(s, BackgroundOptions(**CUZN))[0].data.copy()
    s.data[1, 2, 20:28] = 0  # no counts in the second half of the background window
    s.data[0, 0, :] = 0
    for fast in (True, False):
        out, r = remove_background(s, BackgroundOptions(fast=fast, **CUZN))
        assert np.isnan(r[0, 0]) and np.isnan(r[1, 2])  # no fit there
        assert np.isfinite(out.data).all()
        np.testing.assert_array_equal(out.data[0, 0], 0)  # nothing to subtract from nothing
        other = np.ones((4, 4), bool)
        other[0, 0] = other[1, 2] = False
        if fast:
            np.testing.assert_allclose(out.data[other], clean[other], rtol=1e-6, atol=1e-3)
        else:
            assert np.isfinite(r[other]).all()


def test_a_power_law_is_recovered_exactly():
    x = 400.0 + 2.0 * np.arange(300)
    truth = 5e9 * x ** -3.0
    s = _signal_from(np.tile(truth, (3, 1)) * np.array([[1], [2], [0.5]]), offset=400.0, scale=2.0)
    o = BackgroundOptions(background=[440, 520], signal=[560, 700])
    for fast in (True, False):
        out, r = remove_background(s, BackgroundOptions(**{**o.model_dump(), "fast": fast}))
        # The closed-form estimate sums the channels as an integral: a percent or so.
        np.testing.assert_allclose(r, 3.0, atol=0.02 if fast else 1e-4)
        window = truth[(560 - 400) // 2 : (700 - 400) // 2] * np.array([[1], [2], [0.5]])
        assert np.abs(out.data).max() / window.max() < (0.05 if fast else 1e-4)


def _signal_from(data, offset, scale, units="eV"):
    import hyperspy.api as hs

    s = hs.signals.Signal1D(np.asarray(data, float))
    ax = s.axes_manager.signal_axes[0]
    ax.offset, ax.scale, ax.units, ax.name, ax.is_binned = offset, scale, units, "Energy loss", True
    s.set_signal_type("EELS")
    return s


# ---------- the API ----------


def test_cuzn_spectrum_image(client, eels_cuzn):
    entry = _open(client, eels_cuzn)
    si = _si(entry)
    energy = next(a for a in si["axes"] if not a["navigate"])
    assert si["background"] is None

    job = _remove(client, si, **CUZN)
    assert job["status"] == "done", job
    assert job["kind"] == "background" and job["result"]["dataset_id"]
    file = job["result"]["file"]
    bg = next(d for d in file["datasets"] if d["id"] == job["result"]["dataset_id"])

    # A new spectrum image under the source, in the same file.
    assert (bg["origin"], bg["kind"], bg["parent_id"], bg["file_id"]) == ("background", "spectrum_image", si["id"], si["file_id"])
    assert bg["technique"] == "EELS" and bg["signal_type"] == "EELS" and not bg["line_scan"]
    assert bg["eels"]["linked"] == [] and si["eels"]["linked"] == []
    assert bg["background"] == {
        "source_id": si["id"], "model": "power_law", "order": 2, "fast": True, "log": False, "lba": 0.0,
        "background": [800, 920], "signal": [925, 1010], "units": "eV",
    }
    assert bg["title"] == f"{window_label(925, 1010, 'eV', 8.0)}, bg removed" == "925–1010 eV, bg removed"
    assert next(d for d in file["datasets"] if d["id"] == si["id"])["background"] is None

    # The signal window's channels (28 … 38), calibrated as the source's.
    assert bg["navigation_shape"] == si["navigation_shape"] and bg["signal_shape"] == [11]
    axes = {a["navigate"]: a for a in bg["axes"]}
    new = next(a for a in bg["axes"] if not a["navigate"])
    assert new["size"] == 11 and new["scale"] == energy["scale"] and new["units"] == "eV" and new["name"] == energy["name"]
    assert new["offset"] == pytest.approx(energy["offset"] + 28 * energy["scale"])
    assert [a for a in bg["axes"] if a["navigate"]] == [a for a in si["axes"] if a["navigate"]]
    assert bg["dtype"] == "float32"

    # Power-law exponents, as an image on the scan under it.
    (r,) = _children(file, bg, "background-r")
    assert r["kind"] == "image" and r["title"] == "r (power-law exponent)" and r["signal_shape"] == si["navigation_shape"]
    values = _data(client, r)
    assert values.shape == tuple(si["navigation_shape"][::-1]) and np.isfinite(values).all()
    assert np.mean((values > 2) & (values < 6)) > 0.9

    # The preview of a source spectrum is what was subtracted.
    raw = _data(client, si)
    pixel = raw[3, 5].astype(float)
    p = client.post(f"/api/datasets/{si['id']}/background/preview", json={"options": CUZN, "spectra": [pixel.tolist()]})
    assert p.status_code == 200, p.text
    body = p.json()
    assert body["start"] == 12 and len(body["background"]) == 1 and len(body["background"][0]) == 39 - 12
    np.testing.assert_allclose(
        _data(client, bg)[3, 5], pixel[28:39] - np.array(body["background"][0])[28 - 12 :], rtol=1e-5, atol=1e-2,
    )
    assert body["r"][0] == pytest.approx(values[3, 5], rel=1e-5)

    # Over the background window the remainder is small next to the counts.
    spectra = raw.reshape(-1, raw.shape[-1])[::50].astype(float)
    body = client.post(f"/api/datasets/{si['id']}/background/preview", json={"options": CUZN, "spectra": spectra.tolist()}).json()
    window = slice(0, 28 - 12)
    fitted = np.array(body["background"])[:, window]
    counts = spectra[:, 12:28]
    assert abs((counts - fitted).sum()) < 0.02 * counts.sum()
    assert len(body["r"]) == len(spectra) and all(v is None or 0 < v < 8 for v in body["r"])

    # Spectra of the new SI, a window map of it: it is a normal spectrum image.
    wm = client.post(f"/api/datasets/{bg['id']}/window-maps", json={"lo": 940, "hi": 980})
    assert wm.status_code == 200, wm.text
    assert client.post(f"/api/datasets/{bg['id']}/background/preview", json={"options": CUZN, "spectra": []}).status_code == 422


def test_preview_models_and_validation(client, eels_cuzn):
    si = _si(_open(client, eels_cuzn))
    spectrum = _data(client, si)[2, 2].astype(float).tolist()
    url = f"/api/datasets/{si['id']}/background/preview"
    for model, order in (("power_law", 2), ("exponential", 2), ("polynomial", 3), ("offset", 2)):
        for fast in (True, False):
            r = client.post(url, json={"options": {"model": model, "order": order, "fast": fast, **CUZN}, "spectra": [spectrum, spectrum]})
            assert r.status_code == 200, (model, r.text)
            b = r.json()
            assert b["start"] == 12 and np.array(b["background"]).shape == (2, 27)
            assert (b["r"][0] is None) == (model != "power_law")
            assert b["background"][0] == b["background"][1]
    # Errors: plain English, 422.
    r = client.post(url, json={"options": {**CUZN, "background": [800, 930]}, "spectra": [spectrum]})
    assert r.status_code == 422 and "must end before the signal window starts" in r.text
    r = client.post(url, json={"options": {**CUZN, "background": [100, 920]}, "spectra": [spectrum]})
    assert r.status_code == 422 and "energy range" in r.json()["detail"]
    r = client.post(url, json={"options": CUZN, "spectra": [spectrum[:-1]]})
    assert r.status_code == 422 and "162 channels" in r.json()["detail"]
    for body in ({"options": {**CUZN, "signal": [925, 925]}}, {"options": {**CUZN, "model": "bad"}}):
        assert client.post(url, json={**body, "spectra": [spectrum]}).status_code == 422
        assert client.post(f"/api/datasets/{si['id']}/background", json=body).status_code == 422
    assert client.post(f"/api/datasets/{si['id']}/background", json={"options": {**CUZN, "signal": [925, 929]}}).status_code == 422
    assert client.post("/api/datasets/nope/background/preview", json={"options": CUZN, "spectra": []}).status_code == 404


def test_full_fit_reports_progress(client, eels_cuzn):
    si = _si(_open(client, eels_cuzn))
    r = client.post(f"/api/datasets/{si['id']}/background", json={"options": {**CUZN, "fast": False}})
    job = r.json()
    assert job["status"] in ("queued", "running", "done")
    seen = set()
    while job["status"] in ("queued", "running"):
        seen.add(round(job["progress"], 1))
        time.sleep(0.02)
        job = client.get(f"/api/jobs/{job['id']}").json()
    assert job["status"] == "done" and len(seen) > 2, seen
    bg = next(d for d in job["result"]["file"]["datasets"] if d["id"] == job["result"]["dataset_id"])
    assert bg["background"]["fast"] is False


def test_job_can_be_cancelled(client, eels_cuzn):
    entry = _open(client, eels_cuzn)
    si = _si(entry)
    job = client.post(f"/api/datasets/{si['id']}/background", json={"options": {**CUZN, "fast": False}}).json()
    client.post(f"/api/jobs/{job['id']}/cancel")
    job = _wait(client, job)
    assert job["status"] == "cancelled"
    session = client.app.state.session
    assert not any(d.origin == "background" for d in session.datasets.values())


def test_line_scan(client, eels_linescan):
    si = _si(_open(client, eels_linescan))
    assert si["line_scan"]
    job = _remove(client, si, **LSMO)
    assert job["status"] == "done", job
    file = job["result"]["file"]
    bg = next(d for d in file["datasets"] if d["id"] == job["result"]["dataset_id"])
    assert bg["kind"] == "spectrum_image" and bg["line_scan"] and bg["technique"] == "EELS"
    assert bg["navigation_shape"] == si["navigation_shape"] == [40]
    assert bg["signal_shape"] == [110] and bg["title"] == "635–690 eV, bg removed"
    new = next(a for a in bg["axes"] if not a["navigate"])
    old = next(a for a in si["axes"] if not a["navigate"])
    assert new["scale"] == old["scale"] and new["offset"] == pytest.approx(old["offset"] + 413 * old["scale"])
    assert not _children(file, bg, "background-r")  # no map of a line scan
    # The line scan view's data: positions × channels.
    frame = client.get(f"/api/datasets/{bg['id']}/frame")
    assert frame.headers["X-Array-Shape"] == "40,110"
    # Other models, and the full fit, work.
    for model in ("exponential", "polynomial", "offset"):
        assert _remove(client, si, model=model, **LSMO)["status"] == "done"
    assert _remove(client, si, fast=False, **LSMO)["status"] == "done"


def test_single_spectrum(client, tmp_path):
    x = 400.0 + 2.0 * np.arange(300)
    path = tmp_path / "one.hspy"
    s = _signal_from(5e9 * x ** -3.0 + 100 * np.exp(-((x - 600) / 20) ** 2), 400.0, 2.0)
    s.save(path)
    ds = _open(client, path)["datasets"][0]
    assert ds["kind"] == "spectrum" and ds["technique"] == "EELS"
    options = {"background": [440, 520], "signal": [540, 700]}
    job = _remove(client, ds, **options)
    assert job["status"] == "done", job
    bg = next(d for d in job["result"]["file"]["datasets"] if d["id"] == job["result"]["dataset_id"])
    assert bg["kind"] == "spectrum" and bg["origin"] == "background" and bg["parent_id"] == ds["id"]
    assert bg["navigation_shape"] == [] and bg["signal_shape"] == [80] and bg["title"] == "540–700 eV, bg removed"
    assert not _children(job["result"]["file"], bg, "background-r")
    out = _data(client, bg)
    assert out.shape == (80,) and out.max() == pytest.approx(100, rel=0.05)  # the Gaussian is left
    p = client.post(f"/api/datasets/{ds['id']}/background/preview", json={"options": options, "spectra": [_data(client, ds).tolist()]})
    assert p.status_code == 200 and p.json()["r"][0] == pytest.approx(3.0, abs=0.01)


def test_rerun_replaces_in_place(client, eels_cuzn):
    si = _si(_open(client, eels_cuzn))
    job = _remove(client, si, **CUZN)
    first = job["result"]["dataset_id"]
    bg = next(d for d in job["result"]["file"]["datasets"] if d["id"] == first)
    (r,) = _children(job["result"]["file"], bg, "background-r")

    # New windows and model: same dataset, new revision, title follows.
    options = {"model": "exponential", "fast": False, "background": [780, 900], "signal": [905, 1030]}
    r2 = client.put(f"/api/datasets/{first}/background", json={"options": options})
    assert r2.status_code == 200, r2.text
    job = _wait(client, r2.json())
    assert job["status"] == "done" and job["result"]["dataset_id"] == first
    datasets = {d["id"]: d for d in job["result"]["file"]["datasets"]}
    now = datasets[first]
    assert now["revision"] == bg["revision"] + 1 and now["title"] == "905–1030 eV, bg removed"
    assert now["background"]["model"] == "exponential" and now["background"]["background"] == [780, 900]
    assert now["background"]["fast"] is False and now["signal_shape"] == [17]
    assert len([d for d in datasets.values() if d["origin"] == "background"]) == 1
    assert r["id"] not in datasets  # an exponential has no r map
    # Back to a power law: a map again.
    job = _wait(client, client.put(f"/api/datasets/{first}/background", json={"options": CUZN}).json())
    assert [d["origin"] for d in job["result"]["file"]["datasets"] if d["parent_id"] == first] == ["background-r"]

    # A name the user gave is kept.
    client.app.state.session.get_dataset(first).signal.metadata.General.title = "Cu L"
    job = _wait(client, client.put(f"/api/datasets/{first}/background", json={"options": options}).json())
    assert next(d for d in job["result"]["file"]["datasets"] if d["id"] == first)["title"] == "Cu L"

    # Only background-removed datasets can be re-run; and bad windows fail at once.
    assert client.put(f"/api/datasets/{si['id']}/background", json={"options": CUZN}).status_code == 400
    assert client.put(f"/api/datasets/{first}/background", json={"options": {**CUZN, "signal": [925, 2500]}}).status_code == 422
    assert client.put("/api/datasets/nope/background", json={"options": CUZN}).status_code == 404


def test_window_maps_of_the_result_follow_a_rerun(client, eels_cuzn):
    si = _si(_open(client, eels_cuzn))
    bg_id = _remove(client, si, **CUZN)["result"]["dataset_id"]
    wm = client.post(f"/api/datasets/{bg_id}/window-maps", json={"lo": 940, "hi": 980}).json()["dataset"]
    before = _data(client, wm).copy()
    job = _wait(client, client.put(f"/api/datasets/{bg_id}/background", json={"options": {**CUZN, "model": "offset"}}).json())
    assert job["status"] == "done"
    after = client.app.state.session.get_dataset(wm["id"])
    assert after.revision == wm["revision"] + 1 and not np.array_equal(np.asarray(after.signal.data), before)


def test_rebin_of_the_source_recomputes_the_result(client, eels_cuzn):
    entry = _open(client, eels_cuzn)
    si = _si(entry)
    job = _remove(client, si, **CUZN)
    bg_id = job["result"]["dataset_id"]
    full = _remove(client, si, model="polynomial", fast=False, order=2, background=[800, 920], signal=[925, 1010])
    other_id = full["result"]["dataset_id"]
    assert si["navigation_shape"] == [40, 50]

    r = client.post(f"/api/datasets/{si['id']}/rebin", json={"factors": [2, 2, 1]})
    assert r.status_code == 200, r.text
    datasets = {d["id"]: d for d in r.json()["file"]["datasets"]}
    bg = datasets[bg_id]
    assert bg["navigation_shape"] == [20, 25] and bg["signal_shape"] == [11] and bg["revision"] == 1
    assert datasets[other_id]["navigation_shape"] == [20, 25] and datasets[other_id]["background"]["fast"] is False
    (rmap,) = [d for d in datasets.values() if d["parent_id"] == bg_id]
    assert rmap["signal_shape"] == [20, 25] and rmap["revision"] == 1

    # The same as removing the background from the rebinned spectra.
    again = _remove(client, datasets[si["id"]], **CUZN)
    new = next(d for d in again["result"]["file"]["datasets"] if d["id"] == again["result"]["dataset_id"])
    np.testing.assert_allclose(_data(client, bg), _data(client, new), rtol=1e-5, atol=1e-3)

    # Rebinning the result itself is allowed (it is a spectrum image); its map goes.
    r = client.post(f"/api/datasets/{bg_id}/rebin", json={"factors": [2, 1, 1]})
    assert r.status_code == 200, r.text
    ids = {d["id"] for d in r.json()["file"]["datasets"]}
    assert rmap["id"] not in ids and bg_id in ids


def test_result_has_its_own_processing(client, eels_cuzn):
    si = _si(_open(client, eels_cuzn))
    bg = _remove(client, si, **CUZN)["result"]["dataset_id"]
    filters = client.get(f"/api/datasets/{bg}").json()["filters"]
    r = client.put(f"/api/datasets/{bg}/filters", json={**filters, "spectral": "none", "spatial": "gaussian", "sigma": 1.0})
    assert r.status_code == 200, r.text
    assert client.get(f"/api/datasets/{bg}").json()["filters"]["spatial"] == "gaussian"
    assert client.get(f"/api/datasets/{si['id']}").json()["filters"]["spatial"] == "none"


def test_delete_and_close(client, eels_cuzn):
    entry = _open(client, eels_cuzn)
    si = _si(entry)
    session = client.app.state.session
    job = _remove(client, si, **CUZN)
    bg_id = job["result"]["dataset_id"]
    wm = client.post(f"/api/datasets/{bg_id}/window-maps", json={"lo": 940, "hi": 980}).json()["dataset"]
    before = {d["id"] for d in entry["datasets"]}

    # Deleting a result takes what was made from it, and nothing else.
    r = client.delete(f"/api/datasets/{bg_id}")
    assert r.status_code == 200, r.text
    assert {d["id"] for d in r.json()["file"]["datasets"]} == before
    assert bg_id not in session.datasets and wm["id"] not in session.datasets

    # Closing the file deletes the lot.
    job = _remove(client, si, **CUZN)
    bg_id = job["result"]["dataset_id"]
    (r_id,) = [d["id"] for d in job["result"]["file"]["datasets"] if d["parent_id"] == bg_id]
    assert client.delete(f"/api/files/{entry['id']}").status_code in (200, 204)
    for gone in (bg_id, r_id, si["id"]):
        assert gone not in session.datasets
        assert client.get(f"/api/datasets/{gone}").status_code == 404


def test_leaving_eels_removes_the_results(client, eels_cuzn):
    si = _si(_open(client, eels_cuzn))
    bg_id = _remove(client, si, **CUZN)["result"]["dataset_id"]
    r = client.put(f"/api/datasets/{si['id']}/technique", json={"technique": None})
    assert r.status_code == 200, r.text
    assert bg_id not in {d["id"] for d in r.json()["file"]["datasets"]}
    assert client.post(f"/api/datasets/{si['id']}/background", json={"options": CUZN}).status_code == 400


def test_only_eels_spectra(client, tmp_path):
    entry = client.post("/api/examples/fept-si").json()  # EDS
    for d in entry["datasets"]:
        for r in (
            client.post(f"/api/datasets/{d['id']}/background", json={"options": CUZN}),
            client.post(f"/api/datasets/{d['id']}/background/preview", json={"options": CUZN, "spectra": []}),
        ):
            assert r.status_code == 400, (d["title"], r.text)
            assert r.json()["detail"] == "Background removal is for EELS spectra"


def test_saved_files_record_the_removal(client, eels_cuzn):
    si = _si(_open(client, eels_cuzn))
    bg = _remove(client, si, **CUZN)["result"]["dataset_id"]
    meta = client.get(f"/api/datasets/{bg}/metadata").json()["metadata"]
    assert meta["Processing"]["si_viewer"]["background_removal"] == (
        "power law (fast), background 800–920 eV, signal 925–1010 eV"
    )
    r = client.get(f"/api/datasets/{bg}/export", params={"format": "hspy"})
    assert r.status_code == 200 and len(r.content) > 1000


def test_dm4_spectrum_image(client, eels_dm4):
    si = _si(_open(client, eels_dm4))
    axis = next(a for a in si["axes"] if not a["navigate"])
    lo = axis["offset"] + axis["scale"] * axis["size"] * 0.2
    span = axis["scale"] * axis["size"] * 0.1
    options = {"background": [lo, lo + span], "signal": [lo + 1.2 * span, lo + 2 * span]}
    p = client.post(
        f"/api/datasets/{si['id']}/background/preview",
        json={"options": options, "spectra": [np.ones(axis["size"]).tolist()]},
    )
    assert p.status_code == 200, p.text


# ---------- log fit and LBA ----------


def test_log_fit_recovers_a_power_law_and_an_exponential():
    x = 400.0 + 2.0 * np.arange(300)
    scales = np.array([[1], [2], [0.5]])
    o = {"background": [440, 520], "signal": [560, 700], "log": True}
    window = slice((560 - 400) // 2, (700 - 400) // 2)
    for model, truth in (("power_law", 5e9 * x**-3.0), ("exponential", 1e4 * np.exp(-x / 150.0))):
        s = _signal_from(np.tile(truth, (3, 1)) * scales, offset=400.0, scale=2.0)
        out, r = remove_background(s, BackgroundOptions(model=model, **o))
        assert np.abs(out.data).max() / (truth[window] * scales).max() < 1e-5, model
        if model == "power_law":
            np.testing.assert_allclose(r, 3.0, rtol=1e-6)


def test_log_fit_skips_channels_without_counts():
    x = 400.0 + 2.0 * np.arange(300)
    truth = 5e9 * x**-3.0
    data = np.tile(truth, (2, 1))
    data[1, 25] = 0  # one empty channel in the background window
    data[1, 26] = -3
    out, r = remove_background(
        _signal_from(data, 400.0, 2.0), BackgroundOptions(background=[440, 520], signal=[560, 700], log=True),
    )
    np.testing.assert_allclose(r, 3.0, rtol=1e-6)
    no_counts = _signal_from(np.zeros((1, 300)), 400.0, 2.0)
    out, r = remove_background(no_counts, BackgroundOptions(background=[440, 520], signal=[560, 700], log=True))
    assert np.isnan(r).all() and (out.data == 0).all()


def test_lba_keeps_spectra_of_one_shape_and_steadies_noisy_ones():
    x = 400.0 + 2.0 * np.arange(200)
    truth = 5e9 * x**-3.0
    amounts = np.linspace(1, 3, 20 * 16).reshape(20, 16, 1)
    window = slice(20, 60)
    # The same shape everywhere: LBA changes nothing.
    exact = amounts * truth
    np.testing.assert_allclose(lba_window(exact[..., window], 3.0), exact[..., window], rtol=1e-9)

    rng = np.random.default_rng(0)
    noisy = _signal_from(rng.poisson(amounts * truth * 0.2).astype(float), 400.0, 2.0)
    o = {"background": [440, 520], "signal": [530, 700]}
    _, plain = remove_background(noisy, BackgroundOptions(**o))
    _, lba = remove_background(noisy, BackgroundOptions(lba=4.0, **o))
    assert np.nanstd(lba) < 0.5 * np.nanstd(plain)
    assert abs(np.nanmean(lba) - 3.0) < 0.1
    assert describe(BackgroundOptions(lba=4.0, log=True, **o), "eV").startswith("power law (log fit), LBA 4 px,")


def test_lba_preview_of_a_pixel_is_what_is_subtracted(client, eels_cuzn):
    si = _si(_open(client, eels_cuzn))
    options = {**CUZN, "lba": 3.0, "log": True}
    job = _remove(client, si, **options)
    assert job["status"] == "done", job
    file = job["result"]["file"]
    bg = next(d for d in file["datasets"] if d["id"] == job["result"]["dataset_id"])
    raw = _data(client, si).astype(float)
    url = f"/api/datasets/{si['id']}/background/preview"
    for x, y in ((5, 3), (0, 0)):
        body = client.post(url, json={
            "options": options, "spectra": [raw[y, x].tolist(), raw[y, x].tolist()], "pixels": [{"x": x, "y": y}, None],
        }).json()
        np.testing.assert_allclose(
            _data(client, bg)[y, x], raw[y, x, 28:39] - np.array(body["background"][0])[28 - 12 :], rtol=1e-4, atol=1e-2,
        )
        assert body["background"][0] != body["background"][1]  # without the pixel: no LBA
