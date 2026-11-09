"""Zero-loss peak alignment of EELS low-loss spectrum images (M10d): the shifts
against exSpy's own, the linked core losses that follow, the shift map, the
history, and what is remade from the aligned data."""

import time

import exspy
import numpy as np
import pytest
from fastapi.testclient import TestClient

from si_viewer.app import create_app
from si_viewer.processing.thickness import ThicknessOptions, thickness_map
from si_viewer.processing.zlp import ZlpOptions, align_zero_loss_peak

STEP = 0.1
ENERGY = np.arange(-20.0, 60.0, STEP)  # the ZLP is near -3 eV, a plasmon 18 eV above it
CORE = np.arange(400.0, 480.0, 0.5)  # a feature 43 eV above the ZLP, at 443 eV
SHAPE = (4, 6)


@pytest.fixture
def client():
    return TestClient(create_app())


def _wait(client, job, timeout=300):
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


def _post(client, path, **body):
    r = client.post(path, json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _link(client, ds, partner) -> None:
    r = client.put(f"/api/datasets/{ds['id']}/eels-partner", json={"partner_id": partner["id"]})
    assert r.status_code == 200, r.text


def _align(client, ds, **options):
    r = client.post(f"/api/datasets/{ds['id']}/zlp-alignment", json={"options": options})
    assert r.status_code == 200, r.text
    return _wait(client, r.json())


def _dataset(client, ds):
    return client.app.state.session.get_dataset(ds["id"] if isinstance(ds, dict) else ds)


def _summary(client, ds) -> dict:
    return _dataset(client, ds).summary()


def _file(client, ds) -> dict:
    session = client.app.state.session
    return session.file_of(_dataset(client, ds)).summary(session)


def _children(client, ds, origin) -> list[dict]:
    return [d for d in _file(client, ds)["datasets"] if d["origin"] == origin and d["parent_id"] == ds["id"]]


# ---------- synthetic spectrum images ----------


def _centres(shape=SHAPE, seed=0) -> np.ndarray:
    """Where each pixel's zero-loss peak is: -3 eV, give or take 0.5."""
    return -3.0 + 0.5 * (2 * np.random.default_rng(seed).random(shape) - 1)


def _low_loss(c, units="nm", noise=True) -> exspy.signals.EELSSpectrum:
    e = ENERGY - c[..., None]
    data = 1000 * np.exp(-0.5 * (e / 0.5) ** 2) + 300 * np.exp(-0.5 * ((e - 18) / 3) ** 2)
    if noise:
        data = data + np.random.default_rng(1).random(data.shape)
    s = exspy.signals.EELSSpectrum(data.astype("float32"))
    s.metadata.General.title = "ll"
    ax = s.axes_manager.signal_axes[0]
    ax.offset, ax.scale, ax.units, ax.name = float(ENERGY[0]), STEP, "eV", "Energy loss"
    for nav in s.axes_manager.navigation_axes:
        nav.scale, nav.units = 2.0, units
    s.metadata.set_item("Acquisition_instrument.TEM.beam_energy", 200.0)
    return s


def _core_loss(c, background=False) -> exspy.signals.EELSSpectrum:
    """A feature at 443 eV that drifts with the zero-loss peak (it is at 443 + c + 3 ... 443 + c
    on the axis, as the spectrometer saw it), on a power-law background if asked."""
    data = 100 * np.exp(-0.5 * ((CORE - (443.0 + c[..., None])) / 2.0) ** 2)
    if background:
        data = data + 200 * (CORE / 400.0) ** -3
    s = exspy.signals.EELSSpectrum(data.astype("float32"))
    s.metadata.General.title = "hl"
    ax = s.axes_manager.signal_axes[0]
    ax.offset, ax.scale, ax.units, ax.name = float(CORE[0]), 0.5, "eV", "Energy loss"
    for nav in s.axes_manager.navigation_axes:
        nav.scale, nav.units = 2.0, "nm"
    s.metadata.set_item("Acquisition_instrument.TEM.beam_energy", 200.0)
    return s


def _save(tmp_path, s, name):
    path = tmp_path / f"{name}.hspy"
    s.save(path)
    return path


def _open_pair(client, tmp_path, c=None, background=False):
    """A low loss and a core loss on the same scan, opened and linked."""
    c = _centres() if c is None else c
    ll = _si(_open(client, _save(tmp_path, _low_loss(c), "ll")))
    hl = _si(_open(client, _save(tmp_path, _core_loss(c, background), "hl")))
    r = client.put(f"/api/datasets/{ll['id']}/eels-role", json={"role": "low_loss"})
    assert r.status_code == 200, r.text
    _link(client, hl, ll)
    return c, ll, hl


def _centroid(data, offset, scale, lo, hi) -> np.ndarray:
    """The centre of mass of every spectrum in [lo, hi] (energy units)."""
    energy = offset + scale * np.arange(data.shape[-1])
    keep = (energy >= lo) & (energy <= hi)
    w = np.asarray(data, float)[..., keep]
    return (w * energy[keep]).sum(-1) / w.sum(-1)


def _axis(signal):
    ax = signal.axes_manager.signal_axes[0]
    return float(ax.offset), float(ax.scale)


# ---------- the alignment ----------


def test_peaks_end_up_at_zero_and_core_losses_follow(client, tmp_path):
    c, ll, hl = _open_pair(client, tmp_path)
    before = _centroid(_dataset(client, hl).signal.data, *_axis(_dataset(client, hl).signal), 430, 455)
    assert np.ptp(before) > 0.5  # the feature drifts with the peak

    job = _align(client, hl)  # any member of the group
    assert job["status"] == "done", job
    assert job["kind"] == "zlp-alignment" and job["result"]["aligned"] == [ll["id"], hl["id"]]

    low, core = _dataset(client, ll).signal, _dataset(client, hl).signal
    peak = _centroid(low.data, *_axis(low), -2, 2)
    assert np.abs(peak).max() < 0.05  # every pixel's peak is at 0 eV
    feature = _centroid(core.data, *_axis(core), 430, 455)
    assert np.ptp(feature) < 0.15  # the same energy in every pixel
    assert feature.mean() == pytest.approx(443.0, abs=0.1)

    # exSpy's own result, on the same data: the same channels, axes and counts.
    direct_ll, direct_hl = _low_loss(c), _core_loss(c)
    direct_ll.align_zero_loss_peak(also_align=[direct_hl], print_stats=False, show_progressbar=False)
    for mine, theirs in ((low, direct_ll), (core, direct_hl)):
        assert mine.data.shape == theirs.data.shape
        assert _axis(mine) == pytest.approx(_axis(theirs))
        np.testing.assert_allclose(mine.data, theirs.data, rtol=1e-4, atol=1e-3)
    assert low.data.shape[-1] < len(ENERGY) and not np.isnan(low.data).any()  # cropped

    # Both files' summaries say so; the energy axes start at the new offsets.
    for ds in (ll, hl):
        s = _summary(client, ds)
        assert s["zlp_aligned"] == ["sub-pixel, cropped"] and s["revision"] > 0
    assert _summary(client, hl)["axes"][-1]["offset"] == pytest.approx(_axis(core)[0])


def test_peak_maximum_only_equals_exspy(client, tmp_path):
    c, ll, hl = _open_pair(client, tmp_path)
    job = _align(client, ll, subpixel=False)
    assert job["status"] == "done", job
    direct_ll, direct_hl = _low_loss(c), _core_loss(c)
    direct_ll.align_zero_loss_peak(subpixel=False, also_align=[direct_hl], print_stats=False, show_progressbar=False)
    for ds, theirs in ((ll, direct_ll), (hl, direct_hl)):
        mine = _dataset(client, ds).signal
        assert mine.data.shape == theirs.data.shape and _axis(mine) == pytest.approx(_axis(theirs))
        np.testing.assert_allclose(mine.data, theirs.data, rtol=1e-4, atol=1e-3)
        assert _summary(client, ds)["zlp_aligned"] == ["peak maximum, cropped"]
    # A peak maximum is only as good as the channels: within one of them.
    peak = _centroid(_dataset(client, ll).signal.data, *_axis(_dataset(client, ll).signal), -2, 2)
    assert np.abs(peak).max() < STEP


def test_without_cropping_the_channels_stay(client, tmp_path):
    c, ll, hl = _open_pair(client, tmp_path)
    job = _align(client, ll, crop=False)
    assert job["status"] == "done", job
    low, core = _dataset(client, ll).signal, _dataset(client, hl).signal
    assert low.data.shape[-1] == len(ENERGY) and core.data.shape[-1] == len(CORE)
    assert np.isfinite(low.data).all() and np.isfinite(core.data).all()
    # The channels the shifts emptied are 0, at the end the spectrum was moved away from.
    assert (low.data[..., :8] == 0).any() and (low.data[..., -8:] == 0).any()
    assert np.abs(_centroid(low.data, *_axis(low), -2, 2)).max() < 0.05
    assert _summary(client, ll)["zlp_aligned"] == ["sub-pixel, not cropped"]


def test_the_shift_map_is_what_was_applied(client, tmp_path):
    c, ll, hl = _open_pair(client, tmp_path)
    original = np.asarray(_dataset(client, ll).signal.data).copy()
    job = _align(client, ll)
    result = job["result"]
    (made,) = _children(client, ll, "zlp-shift")
    assert result["dataset_id"] == made["id"]
    assert (made["title"], made["kind"], made["parent_id"], made["file_id"]) == ("ZLP shift", "image", ll["id"], ll["file_id"])
    assert made["is_element_map"] is False and made["technique"] is None
    assert made["id"] in [d["id"] for d in result["file"]["datasets"]] and result["partner_files"] == [
        f for f in result["partner_files"] if f["id"] != ll["file_id"]
    ]
    assert result["partner_files"][0]["id"] == hl["file_id"]

    shifts = np.asarray(_dataset(client, made).signal.data)
    assert shifts.shape == SHAPE and shifts.dtype == np.float32
    # The peak at c moved to 0: a shift of -c (eV) at every pixel.
    np.testing.assert_allclose(shifts, -c, atol=0.03)
    assert result["shift"] == pytest.approx(float(shifts.mean()), abs=1e-4)
    assert _dataset(client, made).signal.metadata.Signal.quantity == "Shift (eV)"

    # The aligned spectrum is the original moved by it.
    low = _dataset(client, ll).signal
    offset, scale = _axis(low)
    energy = offset + scale * np.arange(low.data.shape[-1])
    expected = np.array(
        [[np.interp(energy - shifts[y, x], ENERGY, original[y, x]) for x in range(SHAPE[1])] for y in range(SHAPE[0])]
    )
    np.testing.assert_allclose(low.data, expected, atol=30)


def test_running_again_leaves_nothing_to_shift(client, tmp_path):
    c, ll, hl = _open_pair(client, tmp_path)
    first = _align(client, ll)["result"]
    (made,) = _children(client, ll, "zlp-shift")
    revision = made["revision"]
    assert abs(first["shift"]) > 2

    second = _align(client, ll)
    assert second["status"] == "done", second
    (again,) = _children(client, ll, "zlp-shift")  # replaced, not added
    assert again["id"] == made["id"] and again["revision"] > revision
    assert second["result"]["dataset_id"] == made["id"]
    shifts = np.asarray(_dataset(client, made).signal.data)
    assert np.abs(shifts).max() < 0.05 and abs(second["result"]["shift"]) < 0.02
    assert _summary(client, ll)["zlp_aligned"] == ["sub-pixel, cropped"] * 2
    assert _summary(client, hl)["zlp_aligned"] == ["sub-pixel, cropped"] * 2
    peak = _centroid(_dataset(client, ll).signal.data, *_axis(_dataset(client, ll).signal), -2, 2)
    assert np.abs(peak).max() < 0.05


def test_the_shift_map_can_be_deleted(client, tmp_path):
    c, ll, hl = _open_pair(client, tmp_path)
    _align(client, ll)
    (made,) = _children(client, ll, "zlp-shift")
    r = client.delete(f"/api/datasets/{made['id']}")
    assert r.status_code == 200, r.text
    assert made["id"] not in [d["id"] for d in r.json()["file"]["datasets"]]
    assert not _children(client, ll, "zlp-shift")


def test_a_core_loss_that_is_not_linked_is_left_alone(client, tmp_path):
    c = _centres()
    ll = _si(_open(client, _save(tmp_path, _low_loss(c), "ll")))
    hl = _si(_open(client, _save(tmp_path, _core_loss(c), "hl")))
    client.put(f"/api/datasets/{ll['id']}/eels-role", json={"role": "low_loss"})
    _dataset(client, hl).materialise()
    before = _dataset(client, hl).signal
    data = np.asarray(before.data).copy()
    job = _align(client, ll)
    assert job["status"] == "done" and job["result"]["aligned"] == [ll["id"]] and job["result"]["partner_files"] == []
    assert _dataset(client, hl).signal is before and np.array_equal(before.data, data)
    assert _summary(client, hl)["zlp_aligned"] == []


def test_line_scan(client, tmp_path):
    c = _centres((12,))
    ll = _si(_open(client, _save(tmp_path, _low_loss(c), "ll")))
    assert ll["line_scan"]
    client.put(f"/api/datasets/{ll['id']}/eels-role", json={"role": "low_loss"})
    job = _align(client, ll)
    assert job["status"] == "done", job
    low = _dataset(client, ll).signal
    assert np.abs(_centroid(low.data, *_axis(low), -2, 2)).max() < 0.05
    (made,) = _children(client, ll, "zlp-shift")
    assert made["kind"] == "spectrum" and made["title"] == "ZLP shift"
    shifts = np.asarray(_dataset(client, made).signal.data)
    assert shifts.shape == (12,)
    np.testing.assert_allclose(shifts, -c, atol=0.05)
    # Its axis is the line scan's position.
    assert made["axes"][0]["units"] == "nm" and made["axes"][0]["size"] == 12


def test_progress_is_reported_and_the_inputs_are_not_changed():
    c = _centres()
    low, core = _low_loss(c), _core_loss(c)
    steps = []
    result = align_zero_loss_peak(low, [core], ZlpOptions(), lambda f, m=None: steps.append((f, m)))
    fractions = [f for f, _ in steps]
    assert fractions == sorted(fractions) and fractions[0] == 0 and fractions[-1] == 1
    assert {m for _, m in steps if m} >= {"Finding the zero-loss peaks", "Refining the alignment"}
    assert len(result.signals) == 2 and result.shifts.shape == SHAPE
    # The inputs are copied, not changed.
    assert low.data.shape[-1] == len(ENERGY) and _axis(low)[0] == pytest.approx(float(ENERGY[0]))


# ---------- what is made from the aligned data ----------


def test_what_was_made_from_them_is_remade(client, tmp_path):
    c, ll, hl = _open_pair(client, tmp_path, background=True)
    # A window on the plasmon (at 15 eV on the axis), a fixed threshold, a background removal.
    window = _post(client, f"/api/datasets/{ll['id']}/window-maps", lo=12.0, hi=18.0)["dataset"]
    job = _wait(client, _post(client, f"/api/datasets/{ll['id']}/thickness", options={"threshold": 1.5}))
    assert job["status"] == "done", job
    (thickness,) = _children(client, ll, "thickness")
    before = np.asarray(_dataset(client, thickness).signal.data).copy()
    job = _wait(client, _post(
        client, f"/api/datasets/{hl['id']}/background", options={"background": [410, 425], "signal": [430, 470]},
    ))
    assert job["status"] == "done", job
    (bg,) = _children(client, hl, "background")
    assert bg["title"] == "430–470 eV, bg removed"

    job = _align(client, ll)
    assert job["status"] == "done", job
    shift = job["result"]["shift"]
    assert shift == pytest.approx(3.0, abs=0.1)

    # The window map: the same dataset, on the plasmon again (the window moved with the axis).
    (win,) = _children(client, ll, "window")
    assert win["id"] == window["id"] and win["revision"] > window["revision"]
    assert win["window"]["lo"] == pytest.approx(12.0 + shift, abs=1e-5)
    assert win["window"]["hi"] == pytest.approx(18.0 + shift, abs=1e-5)
    assert win["title"] != window["title"]
    sums = np.asarray(_dataset(client, win).signal.data)
    assert sums.shape == SHAPE and (sums > 0).all()

    # The thickness: a fixed threshold stays as far from the peak.
    (t,) = _children(client, ll, "thickness")
    assert t["id"] == thickness["id"] and t["revision"] > thickness["revision"]
    assert t["thickness"]["threshold"] == pytest.approx(1.5 + shift, abs=1e-5)
    after = np.asarray(_dataset(client, t).signal.data)
    assert after.shape == before.shape and np.isfinite(after).all()
    expected = thickness_map(_dataset(client, ll).signal, ThicknessOptions(threshold=1.5 + shift))
    np.testing.assert_allclose(after, expected, rtol=1e-5)

    # The background-removed spectra: windows moved with the axis, its title with them.
    (new_bg,) = _children(client, hl, "background")
    assert new_bg["id"] == bg["id"] and new_bg["revision"] > bg["revision"]
    assert new_bg["background"]["background"] == pytest.approx([410 + shift, 425 + shift], abs=1e-5)
    assert new_bg["background"]["signal"] == pytest.approx([430 + shift, 470 + shift], abs=1e-5)
    assert new_bg["title"].endswith("eV, bg removed") and new_bg["title"] != bg["title"]
    # The edge is where the signal window leaves it: at the same place in every pixel.
    signal = _dataset(client, new_bg).signal
    centre = _centroid(signal.data, *_axis(signal), 433, 473)
    assert np.ptp(centre) < 0.15


def test_what_no_longer_fits_is_removed(client, tmp_path):
    c, ll, hl = _open_pair(client, tmp_path)
    top = float(ENERGY[-1])
    _post(client, f"/api/datasets/{ll['id']}/window-maps", lo=top - 0.1, hi=top)
    _post(client, f"/api/datasets/{ll['id']}/window-maps", lo=10.0, hi=12.0)
    job = _wait(client, _post(client, f"/api/datasets/{ll['id']}/thickness", options={"threshold": top}))
    assert job["status"] == "done", job
    assert len(_children(client, ll, "window")) == 2 and len(_children(client, ll, "thickness")) == 1

    assert _align(client, ll)["status"] == "done"
    # The axis moved up by 3 eV: the window at its top and the threshold at its end left it.
    (window,) = _children(client, ll, "window")
    assert window["window"]["lo"] == pytest.approx(13.0, abs=0.1)
    assert _children(client, ll, "thickness") == []
    assert len(_children(client, ll, "zlp-shift")) == 1


def test_rebinning_removes_the_shift_map(client, tmp_path):
    c, ll, hl = _open_pair(client, tmp_path)
    _align(client, ll)
    assert len(_children(client, ll, "zlp-shift")) == 1
    r = client.post(f"/api/datasets/{ll['id']}/rebin", json={"factors": [2, 2, 1]})
    assert r.status_code == 200, r.text
    assert _children(client, ll, "zlp-shift") == []
    assert _summary(client, ll)["zlp_aligned"] == ["sub-pixel, cropped"]  # the data stays aligned


def test_the_shift_map_goes_with_the_role(client, tmp_path):
    c, ll, hl = _open_pair(client, tmp_path)
    _align(client, ll)
    client.put(f"/api/datasets/{hl['id']}/eels-partner", json={"partner_id": None})
    r = client.put(f"/api/datasets/{ll['id']}/eels-role", json={"role": "core_loss"})
    assert r.status_code == 200, r.text
    assert _children(client, ll, "zlp-shift") == []


# ---------- errors ----------


def test_errors(client, tmp_path):
    c = _centres()
    hl = _si(_open(client, _save(tmp_path, _core_loss(c), "hl")))
    # A core loss with no low loss in its group.
    r = client.post(f"/api/datasets/{hl['id']}/zlp-alignment", json={})
    assert r.status_code == 400 and "No low loss" in r.json()["detail"]

    # Not EELS.
    plain = exspy.signals.EELSSpectrum(np.ones((3, 4, 32), "float32"))
    plain.set_signal_type("")
    plain.metadata.General.title = "plain"
    other = _si(_open(client, _save(tmp_path, plain, "plain")))
    assert other["eels"] is None
    r = client.post(f"/api/datasets/{other['id']}/zlp-alignment", json={})
    assert r.status_code == 400 and "not an EELS" in r.json()["detail"]
    assert client.post("/api/datasets/nope/zlp-alignment", json={}).status_code == 404

    # Bad options.
    ll = _si(_open(client, _save(tmp_path, _low_loss(c), "ll")))
    client.put(f"/api/datasets/{ll['id']}/eels-role", json={"role": "low_loss"})
    url = f"/api/datasets/{ll['id']}/zlp-alignment"
    assert client.post(url, json={"options": {"calibrate": False}}).status_code == 422
    assert client.post(url, json={"options": {"subpixel": "maybe"}}).status_code == 422
    assert client.post(url, json={"options": {"crop": None}}).status_code == 422

    # Nothing to align along: a handful of channels, or one spectrum.
    short = exspy.signals.EELSSpectrum(np.ones((3, 4, 8), "float32"))
    short.metadata.General.title = "short"
    short.axes_manager.signal_axes[0].units = "eV"
    short = _si(_open(client, _save(tmp_path, short, "short")))
    client.put(f"/api/datasets/{short['id']}/eels-role", json={"role": "low_loss"})
    r = client.post(f"/api/datasets/{short['id']}/zlp-alignment", json={})
    assert r.status_code == 422 and "channels" in r.json()["detail"]
    # Nothing was changed by any of them.
    assert _summary(client, ll)["zlp_aligned"] == [] and _summary(client, hl)["zlp_aligned"] == []


def test_job_can_be_cancelled(client, tmp_path):
    c, ll, hl = _open_pair(client, tmp_path)
    before = _dataset(client, ll).signal
    job = client.post(f"/api/datasets/{ll['id']}/zlp-alignment", json={}).json()
    client.post(f"/api/jobs/{job['id']}/cancel")
    job = _wait(client, job)
    if job["status"] == "cancelled":  # (a job this small may finish first)
        assert _dataset(client, ll).signal is before and _summary(client, ll)["zlp_aligned"] == []
        assert _children(client, ll, "zlp-shift") == []
    else:
        assert job["status"] == "done"


# ---------- the example data ----------


def test_dual_eels_example(client, eels_dm4_ll, eels_dm4_hl):
    ll = _si(_open(client, eels_dm4_ll))
    hl = _si(_open(client, eels_dm4_hl))
    assert ll["eels"]["role"] == "low_loss" and hl["eels"]["role"] == "core_loss"
    _link(client, hl, ll)
    # Rebinned first, to keep it quick (both: they are linked).
    r = client.post(f"/api/datasets/{ll['id']}/rebin", json={"factors": [16, 16, 1]})
    assert r.status_code == 200, r.text
    assert _summary(client, ll)["navigation_shape"] == [6, 25] == _summary(client, hl)["navigation_shape"]

    session = client.app.state.session
    low, core = _dataset(client, ll), _dataset(client, hl)
    options = ThicknessOptions()
    before_peak = np.asarray(low.signal.data, float).sum((0, 1))
    offset, scale = _axis(low.signal)
    assert offset + scale * before_peak.argmax() < -1.0  # the ZLP is at about -3 eV
    nan_before = int(np.isnan(thickness_map(low.signal, options)).sum())
    core_offset = _axis(core.signal)[0]

    job = _align(client, hl)
    assert job["status"] == "done", job
    assert job["result"]["aligned"] == [ll["id"], hl["id"]]

    offset, scale = _axis(low.signal)
    summed = np.asarray(low.signal.data, float).sum((0, 1))
    assert abs(offset + scale * summed.argmax()) <= scale
    assert low.signal.data.shape[:2] == (25, 6) and core.signal.data.shape[:2] == (25, 6)
    assert _axis(core.signal)[0] == pytest.approx(core_offset + job["result"]["shift"], abs=1.0)

    nan_after = int(np.isnan(thickness_map(low.signal, options)).sum())
    assert nan_after < nan_before / 2, (nan_before, nan_after)
    assert session.get_dataset(ll["id"]).zlp_aligned == ["sub-pixel, cropped"]
    (made,) = _children(client, ll, "zlp-shift")
    shifts = np.asarray(_dataset(client, made).signal.data)
    assert shifts.shape == (25, 6) and np.isfinite(shifts).all()
