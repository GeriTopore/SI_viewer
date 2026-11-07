import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from scipy import ndimage

from si_viewer.app import create_app
from si_viewer.processing.filters import (
    Filters, filter_image, filter_spectrum, filtered_pixel_spectrum, spatial_kernel,
)
from si_viewer.processing.regions import pixel_spectrum


def _array(r) -> np.ndarray:
    assert r.status_code == 200, r.text
    shape = tuple(int(n) for n in r.headers["X-Array-Shape"].split(","))
    return np.frombuffer(r.content, "<f4").reshape(shape)


def test_validation():
    with pytest.raises(ValidationError):
        Filters(spatial="median", size=4)
    with pytest.raises(ValidationError):
        Filters(spectral="savgol", window=5, order=5)
    assert Filters().describe() == ""
    assert Filters(spatial="gaussian", sigma=1.5, spectral="savgol").describe() == (
        "Gaussian σ 1.5 px; Savitzky–Golay 7 ch, order 2"
    )


def test_filter_image_keeps_nan():
    rng = np.random.default_rng(0)
    img = rng.random((20, 20)).astype(np.float32)
    img[5, 5] = np.nan
    for f in (Filters(spatial="gaussian", sigma=1), Filters(spatial="mean"), Filters(spatial="median")):
        out = filter_image(img, f)
        assert np.isnan(out[5, 5]) and np.isfinite(np.delete(out.ravel(), 5 * 20 + 5)).all()
    clean = rng.random((20, 20)).astype(np.float32)
    np.testing.assert_allclose(
        filter_image(clean, Filters(spatial="median", size=5)), ndimage.median_filter(clean, 5, mode="reflect")
    )


@pytest.mark.parametrize("f", [Filters(spatial="gaussian", sigma=1.3), Filters(spatial="mean", size=5)])
@pytest.mark.parametrize("x,y,size", [(10, 12, 1), (0, 0, 3), (23, 19, 2)])
def test_filtered_pixel_spectrum_matches_blurred_cube(f, x, y, size):
    """A pixel spectrum of the filtered SI equals the block sum of the SI
    blurred channel by channel (away from the edges: same up to rescaling)."""
    rng = np.random.default_rng(1)
    data = rng.poisson(3, (20, 24, 16)).astype(np.uint16)
    k = spatial_kernel(f)
    blurred = np.stack([ndimage.convolve(data[..., c].astype(float), k, mode="constant") for c in range(16)], -1)
    got, n = filtered_pixel_spectrum(data, x, y, size, f)
    expected, n_raw = pixel_spectrum(blurred, x, y, size)
    assert n == n_raw
    if 4 <= x < 20 and 4 <= y < 16:
        np.testing.assert_allclose(got, expected, rtol=1e-9)
    else:  # near the edge, weights are rescaled to keep the total
        assert got.sum() == pytest.approx(pixel_spectrum(data, x, y, size)[0].sum(), rel=0.3)


def test_filter_spectrum():
    s = np.zeros(50)
    s[25] = 10
    g = filter_spectrum(s, Filters(spectral="gaussian", spectral_sigma=2))
    assert g.sum() == pytest.approx(10) and g[25] < 10
    sg = filter_spectrum(np.arange(50.0), Filters(spectral="savgol"))
    np.testing.assert_allclose(sg, np.arange(50.0), atol=1e-9)  # keeps polynomials


@pytest.fixture
def client_si():
    client = TestClient(create_app())
    entry = client.post("/api/examples/fept-si").json()
    si = next(d for d in entry["datasets"] if d["kind"] == "spectrum_image")
    return client, si, entry


def test_filters_api(client_si):
    client, si, entry = client_si
    haadf = next(d for d in entry["datasets"] if d["kind"] == "image" and d["origin"] == "file")
    assert haadf["filters"]["spatial"] == "none" and haadf["filter_rev"] == 0

    raw = _array(client.get(f"/api/datasets/{haadf['id']}/frame"))
    r = client.put(f"/api/datasets/{haadf['id']}/filters", json={"spatial": "gaussian", "sigma": 2})
    assert r.status_code == 200, r.text
    assert r.json()["dataset"]["filter_rev"] == 1 and r.json()["dataset"]["revision"] == 0
    blurred = _array(client.get(f"/api/datasets/{haadf['id']}/frame"))
    np.testing.assert_allclose(blurred, ndimage.gaussian_filter(raw, 2, mode="reflect"), rtol=1e-5, atol=1e-5)
    # Removing the filter gives the raw data back.
    client.put(f"/api/datasets/{haadf['id']}/filters", json={})
    np.testing.assert_array_equal(_array(client.get(f"/api/datasets/{haadf['id']}/frame")), raw)

    # Not applicable: spectral smoothing on an image, a median on an SI.
    assert client.put(f"/api/datasets/{haadf['id']}/filters", json={"spectral": "gaussian"}).status_code == 422
    assert client.put(f"/api/datasets/{si['id']}/filters", json={"spatial": "median"}).status_code == 422
    assert client.put(f"/api/datasets/{si['id']}/filters", json={"spatial": "mean", "size": 4}).status_code == 422


def test_si_filters_reach_maps_and_spectra(client_si):
    client, si, entry = client_si
    maps = {d["element"]: d for d in entry["datasets"] if d["origin"] == "computed"}
    fe = maps["Fe"]
    raw_map = _array(client.get(f"/api/datasets/{fe['id']}/frame"))
    raw_sum = _array(client.get(f"/api/datasets/{si['id']}/spectrum"))
    raw_px = _array(client.get(f"/api/datasets/{si['id']}/pixel-spectrum?x=30&y=30"))

    r = client.put(f"/api/datasets/{si['id']}/filters", json={"spatial": "gaussian", "sigma": 1.5}).json()
    fe_now = next(d for d in r["file"]["datasets"] if d["id"] == fe["id"])
    assert fe_now["revision"] == fe["revision"] + 1  # recomputed
    np.testing.assert_allclose(
        _array(client.get(f"/api/datasets/{fe['id']}/frame")),
        ndimage.gaussian_filter(raw_map, 1.5, mode="reflect"), rtol=1e-4, atol=1e-3,
    )
    # The pixel spectrum is smoothed spatially; the sum spectrum is unchanged.
    px = _array(client.get(f"/api/datasets/{si['id']}/pixel-spectrum?x=30&y=30"))
    assert not np.allclose(px, raw_px)
    np.testing.assert_array_equal(_array(client.get(f"/api/datasets/{si['id']}/spectrum")), raw_sum)

    # Spectral smoothing applies to every spectrum shown (counts kept, bar
    # the edges of the energy range); keeping the same
    # spatial filter doesn't recompute the maps.
    body = {"spatial": "gaussian", "sigma": 1.5, "spectral": "gaussian", "spectral_sigma": 2}
    r = client.put(f"/api/datasets/{si['id']}/filters", json=body).json()
    assert next(d for d in r["file"]["datasets"] if d["id"] == fe["id"])["revision"] == fe_now["revision"]
    smooth = _array(client.get(f"/api/datasets/{si['id']}/spectrum"))
    assert smooth.sum() == pytest.approx(raw_sum.sum(), rel=1e-2) and smooth.max() < raw_sum.max()

    # Fitted maps made before a spatial filter change are removed.
    job = client.post(f"/api/datasets/{si['id']}/fit-maps", json={}).json()
    while job["status"] in ("queued", "running"):
        job = client.get(f"/api/jobs/{job['id']}").json()
    assert job["status"] == "done"
    r = client.put(f"/api/datasets/{si['id']}/filters", json={}).json()
    assert r["notes"] and not [d for d in r["file"]["datasets"] if d["origin"] == "fitted"]
