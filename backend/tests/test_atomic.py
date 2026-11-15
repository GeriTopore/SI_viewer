"""M22a: kemstem's example images and FFT peaks, checked against kemstem."""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.ndimage import gaussian_filter

from si_viewer.app import create_app
from si_viewer.examples import EXAMPLES, kemstem_data
from si_viewer.processing import atomic

pytestmark = pytest.mark.skipif(kemstem_data() is None, reason="kemstem's clone (example data) is missing")

SITES = "rI_2138 B1_RT 5.3 Mx_0_HAADF.tif"


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


@pytest.fixture(scope="module")
def crop():
    """A 256 × 256 crop of the double perovskite (fast tests)."""
    import hyperspy.api as hs

    return hs.load(kemstem_data() / SITES).data[300:556, 300:556].astype(np.float64)


def _strongest_peak(image):
    """(y, x) pixel of the brightest non-central spot of the log FFT."""
    pattern = np.log1p(np.abs(atomic.fourier_pattern(image)))
    h, w = pattern.shape
    yy, xx = np.mgrid[:h, :w]
    pattern[np.hypot(yy - h // 2, xx - w // 2) < 8] = 0
    return np.array(np.unravel_index(np.argmax(pattern), pattern.shape), dtype=float)


def test_examples_listed_and_opened(client):
    ids = {e.id for e in EXAMPLES}
    assert {"kemstem-strain", "kemstem-sites", "kemstem-pld"} <= ids
    entry = client.post("/api/examples/kemstem-pld").json()
    [d] = entry["datasets"]
    assert d["kind"] == "image"
    assert [a["size"] for a in d["axes"]] == [898, 963]
    assert all(a["units"] == "nm" and a["scale"] == pytest.approx(0.0184) for a in d["axes"])


def test_frequency_round_trip():
    for shape in [(256, 256), (255, 300)]:
        p = np.array([[10.3, 200.7], [128, 128], [0, 299]])
        assert np.allclose(atomic.to_pixels(atomic.to_freqs(p, shape), shape), p)
    # The centre is frequency 0; one pixel right is 1/W cycles per pixel.
    assert np.allclose(atomic.to_freqs([[128, 129]], (256, 256)), [[0, 1 / 256]])


def test_gaussian_refinement_matches_kemstem(crop):
    import kemstem

    peak = _strongest_peak(crop)
    start = peak + [1.4, -1.2]  # a click a little off the spot
    opts = atomic.PeakOptions(method="gauss", window=9, blur=1.0)
    [got] = atomic.refine_peaks(crop, atomic.to_freqs(start, crop.shape), opts)
    assert not got["failed"]

    # The notebooks' calls on the same pattern.
    pattern = kemstem.prepare_fourier_pattern(kemstem.normalize(crop), log=True, log_offset=1e0)
    ref, errs, popt, _ = kemstem.refine_peaks_gf(gaussian_filter(pattern, 1), start[None], window_dimension=9,
                                                 remove_unfit=False, verbose=False)
    assert not errs[0]
    assert np.allclose(atomic.to_pixels([[got["fy"], got["fx"]]], crop.shape), ref, atol=1e-6)
    assert got["background"] == pytest.approx(popt[0, 6])
    # And it lands on the spot.
    assert np.hypot(*(ref[0] - peak)) < 1.0


def test_centre_of_mass_and_edges(crop):
    peak = _strongest_peak(crop)
    opts = atomic.PeakOptions(method="com", window=9, blur=1.0)
    near, edge = atomic.refine_peaks(crop, atomic.to_freqs([peak + [1, 1], [1, 1]], crop.shape), opts)
    assert not near["failed"] and "background" not in near
    assert np.hypot(*(atomic.to_pixels([[near["fy"], near["fx"]]], crop.shape)[0] - peak)) < 1.5
    # Too near the FFT's edge for the window: kept, marked failed.
    assert edge["failed"]
    with pytest.raises(ValueError):
        atomic.PeakOptions(window=8).check()


def test_endpoint(client):
    d = client.post("/api/examples/kemstem-sites").json()["datasets"][0]
    image = np.asarray(client.app.state.session.get_dataset(d["id"]).signal.data, dtype=np.float64)
    peak = _strongest_peak(image)
    f = atomic.to_freqs(peak + [1, -1], image.shape)[0]
    r = client.post(f"/api/datasets/{d['id']}/atomic/fft-peaks", json={"peaks": [list(f)], "window": 9})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["shape"] == [1017, 1017]
    [p] = body["peaks"]
    assert not p["failed"]
    assert np.hypot(*(atomic.to_pixels([[p["fy"], p["fx"]]], image.shape)[0] - peak)) < 1.0
    assert client.post(f"/api/datasets/{d['id']}/atomic/fft-peaks", json={"peaks": [list(f)], "window": 4}).status_code == 422
    assert client.post("/api/datasets/nope/atomic/fft-peaks", json={"peaks": []}).status_code == 404


# ---------- M22b: Fourier filtering, phase and strain ----------

STRAIN = "rI_1341 2.6 Mx B_C270_0_HAADF.tif"


@pytest.fixture(scope="module")
def film():
    """A 512 × 512 crop (even: kemstem's N/2 and the array's N//2 agree) of
    the relaxed film, across the interface."""
    import hyperspy.api as hs

    return hs.load(kemstem_data() / STRAIN).data[700:1212, 700:1212].astype(np.float64)


def _two_peaks(image):
    """The two strongest first-order spots in different directions (frequencies)."""
    pattern = np.log1p(np.abs(atomic.fourier_pattern(image)))
    h, w = pattern.shape
    yy, xx = np.mgrid[:h, :w]
    pattern[np.hypot(yy - h // 2, xx - w // 2) < 8] = 0
    first = np.array(np.unravel_index(np.argmax(pattern), pattern.shape), dtype=float)
    d = first - [h // 2, w // 2]
    # Mask the first spot's line (both signs), then take the next.
    along = np.abs((yy - h // 2) * d[1] - (xx - w // 2) * d[0]) / np.hypot(*d) < 6
    pattern[along] = 0
    second = np.array(np.unravel_index(np.argmax(pattern), pattern.shape), dtype=float)
    return atomic.to_freqs([first, second], (h, w))


def test_fourier_maps_match_kemstem(film):
    import kemstem

    f1, f2 = _two_peaks(film)
    opts = atomic.FourierOptions(sigma=12)
    made = atomic.fourier_maps(film, {1: tuple(f1), 2: tuple(f2)}, {"amplitude", "grating", "phase", "strain"},
                               opts, per_peak=[1], tensor=(1, 2))
    by = {m["output"]: m for m in made}
    assert set(by) == {"amplitude", "grating", "phase", "strain-par", "strain-perp", "exx", "eyy", "exy", "oxy"}
    assert by["exx"]["peaks"] == [1, 2] and by["phase"]["title"] == "Phase g1"

    # The notebook's calls, with peaks as FFT pixels.
    p1, p2 = atomic.to_pixels([f1, f2], film.shape)
    ft = kemstem.prepare_fourier_pattern(kemstem.normalize(film))
    fim1 = kemstem.fourier_filter(ft, p1, sigma=12)[0]
    fim2 = kemstem.fourier_filter(ft, p2, sigma=12)[0]
    ph1 = kemstem.phaselock(fim1, p1, sigma=24)
    ph2 = kemstem.phaselock(fim2, p2, sigma=24)
    par, perp = kemstem.phase_to_strain(ph1, p1)
    exx, eyy, exy, oxy = kemstem.calculate_strain_tensor(ph1, ph2, p1, p2)
    for key, ref in (("amplitude", np.abs(fim1)), ("grating", np.real(fim1)), ("strain-par", par),
                     ("strain-perp", perp), ("exx", exx), ("eyy", eyy), ("exy", exy), ("oxy", oxy)):
        assert np.allclose(by[key]["data"], ref, atol=1e-5 * max(1, np.abs(ref).max())), key
    # Phases agree modulo 2π.
    assert np.allclose(np.angle(np.exp(1j * (by["phase"]["data"] - ph1))), 0, atol=1e-4)
    # A film on its substrate: away from the edges, strains of about a percent.
    assert abs(np.median(exx[64:-64, 64:-64])) < 0.03 and abs(np.median(eyy[64:-64, 64:-64])) < 0.03


def test_gpa_and_checks(film):
    f1, f2 = _two_peaks(film)
    made = atomic.fourier_maps(film, {1: tuple(f1)}, {"phase"}, atomic.FourierOptions(sigma=12, phase="gpa"))
    [m] = made
    assert m["data"].min() >= 0 and m["data"].max() < 2 * np.pi + 1e-6
    with pytest.raises(ValueError):
        atomic.fourier_maps(film, {1: tuple(f1), 2: tuple(2 * f1)}, set(), atomic.FourierOptions(), [], (1, 2))
    with pytest.raises(ValueError):
        atomic.FourierOptions(sigma=0).check()
    assert atomic.coarsening((512, 512), 12) == pytest.approx(512 / (2 * np.pi * 12) * np.sqrt(np.log(4)))


def test_odd_sizes_measure_no_false_strain(film):
    """A peak exactly on an FFT pixel of an odd-sized, unstrained lattice:
    ε∥ ≈ 0 (kemstem's N/2 references, given the true frequency)."""
    n = 255
    yy, xx = np.mgrid[:n, :n]
    f = (0, 25 / n)  # 25 cycles across
    lattice = 1 + np.cos(2 * np.pi * f[1] * xx)
    [par] = [m for m in atomic.fourier_maps(lattice, {1: f}, {"strain"}, atomic.FourierOptions(sigma=6))
             if m["output"] == "strain-par"]
    assert abs(np.median(par["data"])) < 1e-3


def _wait(client, job, timeout=120):
    import time

    t0 = time.time()
    while job["status"] in ("queued", "running"):
        assert time.time() - t0 < timeout, "job did not finish"
        time.sleep(0.05)
        job = client.get(f"/api/jobs/{job['id']}").json()
    return job


def test_fourier_endpoint(client):
    d = client.post("/api/examples/kemstem-sites").json()["datasets"][0]
    image = np.asarray(client.app.state.session.get_dataset(d["id"]).signal.data, dtype=np.float64)
    f1, f2 = _two_peaks(image)
    body = {"peaks": {"1": list(f1), "2": list(f2)}, "per_peak": [1], "outputs": ["amplitude", "phase"],
            "tensor": [1, 2], "sigma": 10}
    job = _wait(client, client.post(f"/api/datasets/{d['id']}/atomic/fourier", json=body).json())
    assert job["status"] == "done", job
    ids = job["result"]["ids"]
    assert len(ids) == 6
    files = job["result"]["file"]["datasets"]
    made = [x for x in files if x["id"] in ids]
    assert {x["title"] for x in made} == {"Amplitude g1", "Phase g1", "εxx (g1, g2)", "εyy (g1, g2)",
                                           "εxy (g1, g2)", "ωxy (g1, g2)"}
    assert all(x["parent_id"] == d["id"] and x["origin"] == "atomic" and x["kind"] == "image" for x in made)
    assert made[0]["atomic"]["sigma"] == 10 and made[0]["atomic"]["coarsening"] > 0
    # Again: the same maps, replaced in place.
    again = _wait(client, client.post(f"/api/datasets/{d['id']}/atomic/fourier", json=body).json())
    assert again["result"]["ids"] == ids
    # Checks before the job.
    bad = {**body, "tensor": [1, 1]}
    assert client.post(f"/api/datasets/{d['id']}/atomic/fourier", json=bad).status_code == 422
    assert client.post(f"/api/datasets/{d['id']}/atomic/fourier", json={**body, "per_peak": [], "tensor": None}).status_code == 422
    # Deletable.
    assert client.delete(f"/api/datasets/{ids[0]}").status_code == 200
