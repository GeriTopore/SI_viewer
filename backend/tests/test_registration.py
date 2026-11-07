import numpy as np
import pytest
import tifffile
from fastapi.testclient import TestClient

from si_viewer.app import create_app


def _array(r) -> np.ndarray:
    assert r.status_code == 200, r.text
    shape = tuple(int(n) for n in r.headers["X-Array-Shape"].split(","))
    return np.frombuffer(r.content, "<f4").reshape(shape)


def _drifting_stack(shifts, size=96, seed=0):
    """Frames of one noisy lattice-and-blobs image, each moved by an integer
    (dy, dx) shift (periodic)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:size, :size]
    base = 1 + 0.5 * np.cos(2 * np.pi * x / 8) * np.cos(2 * np.pi * y / 8)
    for cy, cx in rng.uniform(10, size - 10, (6, 2)):
        base += 3 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 30)
    frames = [np.roll(base, (dy, dx), axis=(0, 1)) for dy, dx in shifts]
    return (np.array(frames) * 1000 + rng.normal(0, 30, (len(shifts), size, size))).astype("float32")


SHIFTS = [(0, 0), (1, 2), (3, 3), (4, 5), (6, 6), (7, 8)]


@pytest.fixture
def stack(tmp_path):
    client = TestClient(create_app())
    tifffile.imwrite(tmp_path / "drift.tif", _drifting_stack(SHIFTS))
    ds = client.post("/api/files", json={"path": str(tmp_path / "drift.tif")}).json()["datasets"][0]
    assert ds["kind"] == "stack"
    return client, ds


def test_session_and_previews(stack):
    client, ds = stack
    url = f"/api/datasets/{ds['id']}/registration"
    assert client.get(url).status_code == 404  # none yet

    state = client.post(url, json={}).json()
    assert (state["first"], state["last"], state["shape"]) == (0, 6, [96, 96])

    assert _array(client.get(f"{url}/frame", params={"i": 2})).shape == (96, 96)

    # The cross-correlation of frames 0 and 3 peaks at their shift.
    cc = _array(client.get(f"{url}/correlation", params={"i": 0, "j": 3, "type": "hann", "n": 4}))
    assert cc.shape == (96, 96)
    peak = np.unravel_index(np.argmax(cc), cc.shape)
    assert (peak[0] - 48, peak[1] - 48) == SHIFTS[3]

    # Comparing cutoffs; the session's own mask is unchanged by previews.
    ccs = _array(client.get(f"{url}/compare-n", params={"i": 0, "j": 1, "type": "hann", "n": [3, 6, 9]}))
    assert ccs.shape == (3, 96, 96)
    mask = client.get(url).json()["mask"]
    assert (mask["type"], mask["n"]) == ("hann", 6.0)

    state = client.put(url, json={"mask": {"type": "gaussian", "n": 3}, "bad_frames": [4, 99]}).json()
    assert state["mask"]["type"] == "gaussian" and state["bad_frames"] == [4]

    # A frame range keeps the settings; frames outside it are refused.
    state = client.post(url, json={"first": 1, "last": 5}).json()
    assert state["bad_frames"] == [4] and state["mask"]["n"] == 3
    assert client.get(f"{url}/frame", params={"i": 0}).status_code == 422
    assert client.post(url, json={"first": 3, "last": 4}).status_code == 422

    assert client.delete(url).status_code == 204
    assert client.get(url).status_code == 404


def test_only_stacks(stack):
    client, _ = stack
    entry = client.post("/api/examples/fept-si").json()
    haadf = next(d for d in entry["datasets"] if d["kind"] == "image")
    assert client.post(f"/api/datasets/{haadf['id']}/registration", json={}).status_code == 400


def _wait(client, job, timeout=120):
    import time

    t0 = time.time()
    while job["status"] in ("queued", "running"):
        assert time.time() - t0 < timeout, "job did not finish"
        time.sleep(0.05)
        job = client.get(f"/api/jobs/{job['id']}").json()
    return job


@pytest.mark.parametrize("method", ["pixel", "gf", "com"])
def test_peak_preview_matches_the_shift_matrices(stack, method):
    """The peak step's shift for a pair is the one the registration uses."""
    client, ds = stack
    url = f"/api/datasets/{ds['id']}/registration"
    client.post(url, json={})
    mask = {"type": "hann", "n": 4}
    peaks = {"method": method, "num_peaks": 3, "sigma_guess": 2, "window_radius": 5}
    client.put(url, json={"mask": mask, "peaks": peaks})

    previews = {}
    for i, j in [(0, 3), (1, 5), (4, 2)]:
        p = client.post(f"{url}/peak", json={"i": i, "j": j, "mask": mask, "peaks": peaks}).json()
        previews[i, j] = p
        if method == "pixel":  # the true shift (the others are biased by this lattice-like test image)
            (dy_i, dx_i), (dy_j, dx_j) = SHIFTS[i], SHIFTS[j]
            assert (p["dx"], p["dy"]) == (dx_j - dx_i, dy_j - dy_i)
        if method == "gf":
            assert len(p["candidates"]) == 3 and sum(c["chosen"] for c in p["candidates"]) == 1
            chosen = next(c for c in p["candidates"] if c["chosen"])
            assert len(chosen["data"]) == 11 and len(chosen["model"][0]) == 11
            assert chosen["fit_x"] == pytest.approx(p["x"]) and chosen["fit_y"] == pytest.approx(p["y"])

    job = _wait(client, client.post(f"{url}/shifts").json())
    assert job["status"] == "done", job
    assert job["result"]["shifts"]["stale"] is False
    m = _array(client.get(f"{url}/matrices"))
    assert m.shape == (2, 6, 6)
    np.testing.assert_allclose(m[0], -m[0].T, atol=1e-5)  # antisymmetric
    # Previews use the same FFTs as the job, so the numbers are the same.
    for (i, j), p in previews.items():
        assert m[0][i, j] == pytest.approx(p["dx"], abs=1e-5) and m[1][i, j] == pytest.approx(p["dy"], abs=1e-5)

    # Changing a setting marks the matrices stale.
    state = client.put(url, json={"mask": {"type": "hann", "n": 5}}).json()
    assert state["shifts"]["stale"] is True


def test_correlation_types_and_errors(stack):
    client, ds = stack
    url = f"/api/datasets/{ds['id']}/registration"
    client.post(url, json={})
    assert client.get(f"{url}/matrices").status_code == 422  # not computed yet
    for corr in ["cc", "mc", "pc"]:
        cc = _array(client.get(f"{url}/correlation", params={"i": 0, "j": 2, "corr": corr}))
        peak = np.unravel_index(np.argmax(cc), cc.shape)
        assert (peak[0] - 48, peak[1] - 48) == SHIFTS[2], corr
    bad = {"i": 0, "j": 1, "peaks": {"method": "gf", "window_radius": 60}}
    assert client.post(f"{url}/peak", json=bad).status_code == 422


def test_outlier_mask(stack):
    client, ds = stack
    url = f"/api/datasets/{ds['id']}/registration"
    client.post(url, json={})
    body = {"outliers": {"method": "transitivity", "threshold": 1}, "bad_frames": []}
    assert client.post(f"{url}/outlier-mask", json=body).status_code == 422  # no shifts yet
    client.put(url, json={"mask": {"type": "hann", "n": 4}})
    assert _wait(client, client.post(f"{url}/shifts").json())["status"] == "done"

    # Clean synthetic shifts: transitivity holds, nothing is masked.
    m = _array(client.post(f"{url}/outlier-mask", json=body))
    assert m.shape == (6, 6) and m.min() == 1
    # A bad frame masks its whole row and column.
    m = _array(client.post(f"{url}/outlier-mask", json={**body, "bad_frames": [2]}))
    assert m[2].sum() == 0 and m[:, 2].sum() == 0 and m[0, 1] == 1
    # Only pairs at most one frame apart; the diagonal is kept.
    m = _array(client.post(f"{url}/outlier-mask", json={"outliers": {"method": "none", "off_diagonal": 1}}))
    assert m[0, 1] == 1 and m[0, 2] == 0 and np.all(np.diag(m) == 1)
    # Neighbour check, and settings stored with the session.
    r = client.post(f"{url}/outlier-mask", json={"outliers": {"method": "neighbours", "max_shift": 0.5}})
    assert '"pairs": 36' in r.headers["X-Array-Meta"]
    assert client.put(url, json={"outliers": {"method": "neighbours", "max_shift": 2}}).json()["outliers"]["max_shift"] == 2


def test_correction_and_drift(stack):
    client, ds = stack
    url = f"/api/datasets/{ds['id']}/registration"
    client.post(url, json={})
    client.put(url, json={"mask": {"type": "hann", "n": 4}})
    assert client.post(f"{url}/correct", json={}).status_code == 422  # no shifts yet
    assert _wait(client, client.post(f"{url}/shifts").json())["status"] == "done"

    # Keep only pairs up to two frames apart, with frame 3 bad: correction
    # fills the rest from paths of good pairs, and the drift is the true one.
    body = {"outliers": {"method": "none", "off_diagonal": 2}, "bad_frames": [3]}
    drift = client.post(f"{url}/correct", json=body).json()
    assert drift["frames"] == [0, 1, 2, 4, 5]
    true = np.array([SHIFTS[f] for f in drift["frames"]], float)  # (dy, dx)
    rel = lambda v: np.array(v) - np.mean(v)  # noqa: E731
    np.testing.assert_allclose(rel(drift["x"]), rel(true[:, 1]), atol=1e-6)
    np.testing.assert_allclose(rel(drift["y"]), rel(true[:, 0]), atol=1e-6)

    c = _array(client.get(f"{url}/corrected"))
    assert c.shape == (3, 6, 6)
    assert c[2][3].sum() == 0 and c[2][0, 5] == 1  # bad frame unknown, far pair filled
    assert c[0][0, 5] == pytest.approx(SHIFTS[5][1] - SHIFTS[0][1])

    # Only neighbouring pairs, frame 3 bad: frames 0–2 and 4–5 share no good
    # pair, so they can't be related. An error, not an endless loop.
    r = client.post(f"{url}/correct", json={"outliers": {"method": "none", "off_diagonal": 1}, "bad_frames": [3]})
    assert r.status_code == 422 and "too few good pairs" in r.json()["detail"]


def test_elliptical_mask_is_the_packages(stack):
    """The elliptical Gaussian mask (makeFourierMask_eg) with theta in degrees."""
    import rigidregistration

    from si_viewer.processing.registration import MaskSettings, RegistrationSession, make_mask

    frames = _drifting_stack(SHIFTS)
    s = RegistrationSession(frames)
    mask = MaskSettings(type="elliptical", n=12, n2=9.106, theta=81.36)
    ops = s._ops(mask, s.peaks)
    ref = rigidregistration.stackregistration.imstack(np.moveaxis(frames.astype(float), 0, -1))
    ref.makeFourierMask_eg(n1=12, n2=9.106, theta=np.radians(81.36))
    np.testing.assert_allclose(ops.mask_fourierspace, ref.mask_fourierspace)
    make_mask(ref, MaskSettings(type="hann", n=4))
    assert ref.mask_params["masktype"] == "hann"

    client, ds = stack
    url = f"/api/datasets/{ds['id']}/registration"
    client.post(url, json={})
    params = {"i": 0, "j": 3, "type": "elliptical", "n": 4, "n2": 3, "theta": 30}
    cc = _array(client.get(f"{url}/correlation", params=params))
    peak = np.unravel_index(np.argmax(cc), cc.shape)
    assert (peak[0] - 48, peak[1] - 48) == SHIFTS[3]
    ccs = _array(client.get(f"{url}/compare-n", params={**params, "n": [3, 6], "n1": 4}))
    assert ccs.shape == (2, 96, 96)


NB3CL8 = "../../../rigidRegistration/samplefiles/sample_data/Nb3Cl8_cross_section.tif"


def test_matches_the_detailed_walkthrough():
    """The DetailedWalkThrough notebook's Nb3Cl8 registration (elliptical
    mask, Gaussian fit, transitivity outliers, correction), run with the
    package directly, gives the same matrices, mask and drift as the app."""
    import os

    import rigidregistration

    from si_viewer.processing.registration import (
        MaskSettings, OutlierSettings, PeakSettings, RegistrationSession,
    )

    path = os.path.join(os.path.dirname(__file__), NB3CL8)
    if not os.path.exists(path):
        pytest.skip("rigidRegistration's sample data not found next to this repository")
    raw = tifffile.imread(path)

    ref = rigidregistration.stackregistration.imstack(np.rollaxis(raw, 0, 3) / float(2**16))
    ref.getFFTs()
    ref.makeFourierMask_eg(n1=12, n2=9.106, theta=1.42)
    ref.setGaussianFitParams(num_peaks=3, sigma_guess=3, window_radius=4)
    ref.findImageShifts(findMaxima="gf")
    ref.get_outliers(10, 10)
    ref.refine_mask()
    ref.make_corrected_Rij()
    ref.get_imshifts()

    s = RegistrationSession(raw)
    s.set_settings(
        mask=MaskSettings(type="elliptical", n=12, n2=9.106, theta=float(np.degrees(1.42))),
        peaks=PeakSettings(method="gf", num_peaks=3, sigma_guess=3, window_radius=4),
    )
    s.compute_shifts(lambda f: None)
    m = s.matrices()
    np.testing.assert_allclose(m[0], ref.Y_ij, atol=1e-4)
    np.testing.assert_allclose(m[1], ref.X_ij, atol=1e-4)
    outliers = OutlierSettings(threshold=10, max_paths=10)
    np.testing.assert_array_equal(s.outlier_mask(outliers), ref.Rij_mask)
    drift = s.correct(outliers)
    np.testing.assert_allclose(drift["x"], -ref.shifts_y, atol=1e-4)
    np.testing.assert_allclose(drift["y"], -ref.shifts_x, atol=1e-4)
