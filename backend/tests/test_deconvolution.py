"""Plural-scattering deconvolution (M10d): Fourier-log, Fourier-ratio and
Richardson–Lucy with the group's low loss, its preview and its results."""

import warnings

import hyperspy.api as hs
import numpy as np
import pytest
from fastapi.testclient import TestClient

from si_viewer.app import create_app
from si_viewer.processing.background import EnergyAxis
from si_viewer.processing.deconvolution import power_law_extrapolation, richardson_lucy

from test_thickness import _link, _open, _role, _si, _wait

BACKGROUND = {"model": "power_law", "fast": True, "background": [590, 630], "signal": [635, 700]}


@pytest.fixture
def client():
    return TestClient(create_app())


@pytest.fixture
def pair(client, eels_linescan, eels_linescan_low_loss):
    """The LSMO line scans, linked: (low loss, core loss) summaries."""
    ll = _si(_open(client, eels_linescan_low_loss))
    hl = _si(_open(client, eels_linescan))
    _role(client, ll, "low_loss")
    _link(client, hl, ll)
    return ll, hl


def _signal(client, ds_id):
    return client.app.state.session.get_dataset(ds_id).signal


def _deconvolve(client, ds, **options):
    r = client.post(f"/api/datasets/{ds['id']}/deconvolution", json={"options": options})
    assert r.status_code == 200, r.text
    job = _wait(client, r.json())
    assert job["status"] == "done", job
    return next(d for d in job["result"]["file"]["datasets"] if d["id"] == job["result"]["dataset_id"])


def _remove_background(client, ds):
    r = client.post(f"/api/datasets/{ds['id']}/background", json={"options": BACKGROUND})
    job = _wait(client, r.json())
    return next(d for d in job["result"]["file"]["datasets"] if d["id"] == job["result"]["dataset_id"])


def _quiet():
    warnings.filterwarnings("ignore")


# ---------- Richardson–Lucy and the extrapolation ----------


def test_richardson_lucy_sharpens_and_keeps_counts():
    x = np.arange(200.0)
    truth = np.zeros((2, 200))
    truth[0, [60, 64, 120]] = [100, 50, 80]
    truth[1, [30, 150]] = [40, 90]
    # A broad kernel whose maximum sits at channel 25 (any position works).
    kernel = np.exp(-0.5 * ((x[:60] - 25) / 3.0) ** 2)
    k = kernel / kernel.sum()
    blurred = np.stack([np.convolve(t, k)[25 : 25 + 200] for t in truth])
    out = richardson_lucy(blurred, np.stack([kernel, 5 * kernel]), 200)
    np.testing.assert_allclose(out.sum(axis=1), blurred.sum(axis=1), rtol=1e-6)
    # The peaks come back where they were, much sharper than blurred.
    for row, at, height in [(0, 60, 100), (0, 120, 80), (1, 30, 40), (1, 150, 90)]:
        assert abs(int(np.argmax(out[row, at - 3 : at + 4])) - 3) <= 1
        assert out[row, at - 1 : at + 2].sum() > 0.8 * height > 1.5 * blurred[row, at - 1 : at + 2].sum()


def test_extrapolation_matches_exspy_and_survives_empty_tails(eels_linescan_low_loss):
    _quiet()
    s = hs.load(eels_linescan_low_loss)
    s.set_signal_type("EELS")
    ref = s.power_law_extrapolation(window_size=100, extrapolation_size=100).data
    mine = power_law_extrapolation(s.data.astype(float), EnergyAxis.of(s), 100, 100)
    np.testing.assert_allclose(mine, ref, rtol=1e-9)
    empty = s.data.astype(float)
    empty[3, -100:] = 0  # exSpy gives up on the whole signal here
    out = power_law_extrapolation(empty, EnergyAxis.of(s), 100, 100)
    assert np.isfinite(out).all() and not out[3, -100:].any()
    np.testing.assert_allclose(out[0], mine[0])


# ---------- the three methods, as exSpy does them ----------


def test_fourier_log_of_the_low_loss(client, pair):
    _quiet()
    ll, hl = pair
    d = _deconvolve(client, ll, method="fourier_log", threshold=3.0)
    assert d["origin"] == "deconvolved" and d["parent_id"] == ll["id"] and d["kind"] == "spectrum_image"
    assert d["deconvolution"]["method"] == "fourier_log" and d["deconvolution"]["low_loss_id"] == ll["id"]
    assert d["title"].endswith("(Fourier-log)") and d["eels"] is not None
    s = _signal(client, ll["id"])
    zlp = s.deepcopy()
    zlp.data = np.where(s.axes_manager.signal_axes[0].axis < 3.0, s.data, 0)
    ref = s.fourier_log_deconvolution(zlp).data
    np.testing.assert_allclose(_signal(client, d["id"]).data, ref, rtol=1e-4, atol=1e-3 * np.abs(ref).max())
    # Only the low loss.
    r = client.post(f"/api/datasets/{hl['id']}/deconvolution", json={"options": {"method": "fourier_log"}})
    assert r.status_code == 400 and "low loss" in r.json()["detail"]


def test_fourier_ratio_needs_the_background_removed(client, pair):
    _quiet()
    ll, hl = pair
    r = client.post(f"/api/datasets/{hl['id']}/deconvolution", json={"options": {"method": "fourier_ratio"}})
    assert r.status_code == 400 and "background" in r.json()["detail"]
    bg = _remove_background(client, hl)
    d = _deconvolve(client, bg, method="fourier_ratio", threshold=3.0, fwhm=1.0)
    assert d["parent_id"] == bg["id"] and d["deconvolution"]["low_loss_id"] == ll["id"]
    ref = _signal(client, bg["id"]).fourier_ratio_deconvolution(_signal(client, ll["id"]), fwhm=1.0, threshold=3.0).data
    np.testing.assert_allclose(_signal(client, d["id"]).data, ref, rtol=1e-4, atol=1e-4 * np.abs(ref).max())
    # Automatic threshold and FWHM run too.
    auto = _deconvolve(client, bg, method="fourier_ratio")
    assert np.isfinite(_signal(client, auto["id"]).data).all()


def test_richardson_lucy_with_the_low_loss_or_its_zero_loss_peak(client, pair):
    _quiet()
    ll, hl = pair
    d = _deconvolve(client, hl, method="richardson_lucy", iterations=10)
    out, raw = _signal(client, d["id"]).data, _signal(client, hl["id"]).data
    assert out.shape == raw.shape
    np.testing.assert_allclose(out.sum(axis=-1), np.clip(raw, 0, None).sum(axis=-1), rtol=1e-3)
    r = client.post(f"/api/datasets/{ll['id']}/deconvolution", json={"options": {"method": "richardson_lucy"}})
    assert r.status_code == 400 and "zero-loss peak" in r.json()["detail"]
    sharp = _deconvolve(client, ll, method="richardson_lucy", psf="zlp", threshold=3.0, iterations=5)
    assert _signal(client, sharp["id"]).data.shape == _signal(client, ll["id"]).data.shape


def test_without_a_low_loss(client, eels_linescan):
    hl = _si(_open(client, eels_linescan))
    r = client.post(f"/api/datasets/{hl['id']}/deconvolution", json={"options": {"method": "richardson_lucy"}})
    assert r.status_code == 400 and r.json()["detail"] == "No low loss in its group"


# ---------- preview, re-run and delete ----------


def test_preview_sums_the_selected_pixels(client, pair):
    _quiet()
    ll, hl = pair
    body = {
        "options": {"method": "fourier_log", "threshold": 3.0},
        "selections": [{"kind": "whole"}, {"kind": "pixel", "x": 4, "y": 0, "size": 1}],
    }
    r = client.post(f"/api/datasets/{ll['id']}/deconvolution/preview", json=body)
    assert r.status_code == 200, r.text
    p = r.json()
    assert p["pixels"] == [40, 1] and p["threshold"] == 3.0 and p["fwhm"] is None
    s = _signal(client, ll["id"])
    one = s.inav[4:5].deepcopy()
    zlp = one.deepcopy()
    zlp.data = np.where(s.axes_manager.signal_axes[0].axis < 3.0, one.data, 0)
    np.testing.assert_allclose(p["spectra"][1], one.fourier_log_deconvolution(zlp).data[0], rtol=1e-4, atol=1e-2)
    # Fourier-ratio reports the FWHM it measured.
    bg = _remove_background(client, hl)
    body = {"options": {"method": "fourier_ratio"}, "selections": [{"kind": "whole"}]}
    p = client.post(f"/api/datasets/{bg['id']}/deconvolution/preview", json=body).json()
    assert p["fwhm"] > 0 and p["threshold"] > 0 and len(p["spectra"][0]) == len(_signal(client, bg["id"]).data[0])


def test_rerun_follows_the_method_and_delete(client, pair):
    _quiet()
    ll, hl = pair
    d = _deconvolve(client, hl, method="richardson_lucy", iterations=3)
    r = client.put(f"/api/datasets/{d['id']}/deconvolution", json={"options": {"method": "richardson_lucy", "iterations": 6}})
    job = _wait(client, r.json())
    again = next(x for x in job["result"]["file"]["datasets"] if x["id"] == d["id"])
    assert again["deconvolution"]["iterations"] == 6 and again["revision"] == d["revision"] + 1
    assert client.delete(f"/api/datasets/{d['id']}").status_code == 200
    assert d["id"] not in client.app.state.session.datasets
