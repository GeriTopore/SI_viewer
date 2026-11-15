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
