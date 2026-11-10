import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from scipy import ndimage

from si_viewer.app import create_app
from si_viewer.processing.filters import (
    Filters, bandpass, bandpass_adjoint, bandpass_transfer, check_applicable, filter_image, filter_spectrum,
    spatial_weights, spectral_adjoint,
)
from si_viewer.processing.regions import (
    Region, pixel_spectrum, processed_energy_map, processed_pixel_spectrum, processed_region_spectrum,
    region_spectrum,
)


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
    for bad in ({"small": 40, "large": 40}, {"small": 50, "large": 10}, {"small": 0}, {"large": 20000}):
        with pytest.raises(ValidationError):
            Filters(spatial="bandpass", **bad)
    with pytest.raises(ValidationError, match="less than"):
        Filters(spatial="bandpass", small=5, large=3)
    assert Filters(spatial="bandpass").describe() == "Bandpass 2–40 px"
    assert Filters(spatial="bandpass", small=1.5, large=25).describe() == "Bandpass 1.5–25 px"
    check_applicable("spectrum_image", Filters(spatial="bandpass"))
    check_applicable("image", Filters(spatial="bandpass"))
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


def _processed_cube(data: np.ndarray, f: Filters) -> np.ndarray:
    """The filtered spectrum image, brute force: every channel blurred, then
    every spectrum smoothed."""
    cube = data.astype(np.float64)
    if f.spatial == "gaussian":
        cube = ndimage.gaussian_filter(cube, (f.sigma, f.sigma, 0), mode="reflect")
    elif f.spatial == "mean":
        cube = ndimage.uniform_filter(cube, (f.size, f.size, 1), mode="reflect")
    elif f.spatial == "bandpass":
        cube = np.stack([bandpass(cube[..., c], f.small, f.large) for c in range(cube.shape[-1])], axis=-1)
    return filter_spectrum(cube, f)


PROCESSING = [
    Filters(spatial="bandpass", small=1.5, large=8),
    Filters(spatial="bandpass", small=2, large=40, spectral="gaussian", spectral_sigma=1),
    Filters(spatial="gaussian", sigma=1.3),
    Filters(spatial="mean", size=5),
    Filters(spatial="gaussian", sigma=0.8, spectral="savgol", window=5),
    Filters(spectral="gaussian", spectral_sigma=1.5),
]


@pytest.mark.parametrize("f", PROCESSING)
def test_processed_spectra_match_processed_cube(f):
    """Pixel and region spectra (edges included) and window maps of the
    processed SI, computed from the raw data, equal the brute-force ones."""
    rng = np.random.default_rng(1)
    data = rng.poisson(3, (20, 24, 16)).astype(np.uint16)
    cube = _processed_cube(data, f)
    for x, y, size in [(10, 12, 1), (0, 0, 3), (23, 19, 2)]:
        got, n = processed_pixel_spectrum(data, x, y, size, f)
        expected, n_raw = pixel_spectrum(cube, x, y, size)
        assert n == n_raw
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-9)
    for region in [
        Region(kind="ellipse", x0=-2, y0=3, x1=11, y1=19),
        Region(kind="polygon", points=[(3, 2), (22, 5), (9, 18)]),
        Region(kind="rect", x0=0, y0=0, x1=24, y1=20),
    ]:
        got, _ = processed_region_spectrum(data, region, f)
        np.testing.assert_allclose(got, region_spectrum(cube, region)[0], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(
        processed_energy_map(data, 0.0, 1.0, 4, 9, f), cube[..., 4:9].sum(-1), rtol=1e-5, atol=1e-4  # float32 maps
    )


def test_bandpass_transfer():
    h = bandpass_transfer((256, 256), 4, 64)
    assert h[0, 0] == 1
    assert (h >= 0).all() and (h <= 1).all()
    # Half power (amplitude 2^-½) at a period of "large" px (high-pass side)...
    assert h[0, 4] == pytest.approx(2**-0.5, abs=0.01)  # f = 4/256 = 1/64 cycles/px
    # ... and at a period of "small" px (low-pass side).
    assert bandpass_transfer((256, 256), 4, 1000)[0, 64] == pytest.approx(2**-0.5, abs=1e-6)
    # Isotropic.
    assert h[4, 0] == pytest.approx(h[0, 4], rel=1e-12)


def test_bandpass_image():
    rng = np.random.default_rng(3)
    # A flat image stays flat (and keeps its level), whatever its size.
    for shape in [(40, 50), (7, 90), (1, 30)]:
        flat = np.full(shape, 5.0, np.float32)
        np.testing.assert_allclose(filter_image(flat, Filters(spatial="bandpass")), 5.0, atol=1e-4)
    img = rng.random((64, 64)).astype(np.float32)
    out = filter_image(img, Filters(spatial="bandpass", small=2, large=30))
    assert out.dtype == np.float32 and out.shape == img.shape
    assert abs(float(out.mean()) - float(img.mean())) < 0.02
    assert out.std() < img.std()
    # A slow ramp is removed (up to the mean); a fine checkerboard too.
    yy, xx = np.mgrid[:128, :128]
    ramp = (0.2 * xx).astype(np.float32)
    got = filter_image(ramp, Filters(spatial="bandpass", small=1, large=10))
    assert np.abs(got[40:90, 40:90] - ramp.mean()).max() < 0.3 * np.ptp(ramp)
    assert np.abs(got[40:90, 40:90] - got[40:90, 40:90].mean()).max() < 0.5
    checker = ((xx + yy) % 2).astype(np.float32)
    got = filter_image(checker, Filters(spatial="bandpass", small=4, large=60))
    assert np.abs(got[20:100, 20:100] - 0.5).max() < 0.05


def test_bandpass_nan():
    rng = np.random.default_rng(4)
    img = rng.random((30, 30)).astype(np.float32)
    img[5, 5] = np.nan
    img[20:22, 10:14] = np.nan
    f = Filters(spatial="bandpass", small=1.5, large=12)
    out = filter_image(img, f)
    assert np.isnan(out[5, 5]) and np.isnan(out[20:22, 10:14]).all()
    assert np.isfinite(out[~np.isnan(img)]).all()
    # NaN pixels are filled with the valid mean for the transform.
    filled = np.where(np.isnan(img), np.nanmean(img), img)
    expected = filter_image(filled, f)
    np.testing.assert_allclose(out[~np.isnan(img)], expected[~np.isnan(img)], rtol=1e-5, atol=1e-6)
    assert np.isnan(filter_image(np.full((8, 8), np.nan, np.float32), f)).all()


def test_bandpass_adjoint():
    rng = np.random.default_rng(5)
    for shape in [(20, 24), (9, 31)]:
        a, b = rng.random(shape), rng.random(shape)
        lhs = (bandpass(a, 1.5, 9) * b).sum()
        rhs = (a * bandpass_adjoint(b, 1.5, 9)).sum()
        assert lhs == pytest.approx(rhs, rel=1e-10)
        mask = np.zeros(shape)
        mask[3:7, 2:9] = 1
        np.testing.assert_allclose(spatial_weights(mask, Filters(spatial="bandpass", small=1.5, large=9)),
                                   bandpass_adjoint(mask, 1.5, 9))


def test_spectral_adjoint():
    f = Filters(spectral="savgol", window=7, order=2)
    rng = np.random.default_rng(2)
    spectra = rng.random((5, 40))
    weights = rng.random((12, 3))  # on channels 10..21
    via_adjoint = spectra @ spectral_adjoint(weights, f, 40, offset=10)
    direct = filter_spectrum(spectra, f)[:, 10:22] @ weights
    np.testing.assert_allclose(via_adjoint, direct, rtol=1e-10)


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
    assert client.put(f"/api/datasets/{si['id']}/filters", json={"spatial": "bandpass", "small": 9, "large": 3}).status_code == 422
    # A bandpass is linear, so a spectrum image takes it.
    assert client.put(f"/api/datasets/{si['id']}/filters", json={"spatial": "bandpass"}).status_code == 200
    client.put(f"/api/datasets/{si['id']}/filters", json={})


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

    # Spectral smoothing applies to every spectrum (counts kept, bar the
    # edges of the energy range) and remakes the maps from smoothed spectra.
    body = {"spatial": "gaussian", "sigma": 1.5, "spectral": "gaussian", "spectral_sigma": 2}
    r = client.put(f"/api/datasets/{si['id']}/filters", json=body).json()
    assert next(d for d in r["file"]["datasets"] if d["id"] == fe["id"])["revision"] == fe_now["revision"] + 1
    smooth = _array(client.get(f"/api/datasets/{si['id']}/spectrum"))
    assert smooth.sum() == pytest.approx(raw_sum.sum(), rel=1e-2) and smooth.max() < raw_sum.max()

    # Maps follow the spectrum image: they can't be filtered on their own.
    assert client.put(f"/api/datasets/{fe['id']}/filters", json={"spatial": "median"}).status_code == 422

    # A single fit uses the processed spectrum: of all pixels, the smoothed sum.
    fit = client.post(f"/api/datasets/{si['id']}/fit", json={}).json()
    np.testing.assert_allclose(np.array(fit["data"])[-200:], smooth[-200:], rtol=1e-4)

    # Fitted maps made before a processing change are removed.
    job = client.post(f"/api/datasets/{si['id']}/fit-maps", json={}).json()
    while job["status"] in ("queued", "running"):
        job = client.get(f"/api/jobs/{job['id']}").json()
    assert job["status"] == "done"
    r = client.put(f"/api/datasets/{si['id']}/filters", json={}).json()
    assert r["notes"] and not [d for d in r["file"]["datasets"] if d["origin"] == "fitted"]
