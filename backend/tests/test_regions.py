import numpy as np
import pytest
from fastapi.testclient import TestClient

from si_viewer.app import create_app
from si_viewer.processing.regions import (
    Region, energy_map, pixel_spectrum, region_mask, region_spectrum,
)


@pytest.fixture
def cube():
    rng = np.random.default_rng(1)
    return rng.integers(0, 50, size=(20, 30, 16), dtype=np.uint16)


def test_rect_mask_uses_pixel_centres(cube):
    ys, xs, mask = region_mask(cube.shape[:2], Region(kind="rect", x0=2.2, y0=3.0, x1=5.6, y1=4.9))
    # Centres 2.5..5.5 in x (pixels 2-5) and 3.5..4.5 in y (pixels 3-4).
    assert (xs, ys) == (slice(2, 6), slice(3, 5)) and mask.all()
    spectrum, n = region_spectrum(cube, Region(kind="rect", x0=5.6, y0=4.9, x1=2.2, y1=3.0))
    assert n == 8
    np.testing.assert_array_equal(spectrum, cube[3:5, 2:6].sum(axis=(0, 1)))


def test_ellipse_and_polygon(cube):
    ell = Region(kind="ellipse", x0=10, y0=5, x1=20, y1=15)
    ys, xs, mask = region_mask(cube.shape[:2], ell)
    spectrum, n = region_spectrum(cube, ell)
    assert n == mask.sum() and 60 < n < 100  # ≈ π·5·5
    np.testing.assert_array_equal(spectrum, cube[ys, xs][mask].sum(axis=0))

    # A triangle covering the lower-left half of a 4 × 4 square.
    tri = Region(kind="polygon", points=[(0, 0), (0, 4), (4, 4)])
    _, _, mask = region_mask(cube.shape[:2], tri)
    assert mask.sum() == 6  # centres strictly below the diagonal


def test_region_outside_is_empty(cube):
    spectrum, n = region_spectrum(cube, Region(kind="rect", x0=100, y0=100, x1=120, y1=110))
    assert n == 0 and not spectrum.any()


def test_pixel_spectrum_block(cube):
    s, n = pixel_spectrum(cube, 0, 0, size=3)  # clipped at the corner
    assert n == 9
    np.testing.assert_array_equal(s, cube[0:3, 0:3].sum(axis=(0, 1)))
    s, n = pixel_spectrum(cube, 10, 7)
    np.testing.assert_array_equal(s, cube[7, 10])


def test_energy_map(cube):
    m = energy_map(cube, offset=0.0, scale=0.5, lo=1.0, hi=2.0)  # channels 2-3
    np.testing.assert_array_equal(m, cube[..., 2:4].sum(-1))


@pytest.fixture(scope="module")
def si_client(example_emd):
    client = TestClient(create_app())
    entry = client.post("/api/files", json={"path": str(example_emd)}).json()
    si = next(d for d in entry["datasets"] if d["kind"] == "spectrum_image")
    return client, si


def _array(r):
    assert r.status_code == 200, r.text
    return np.frombuffer(r.content, "<f4")


def test_si_endpoints(si_client):
    client, si = si_client
    base = f"/api/datasets/{si['id']}"
    total = _array(client.get(f"{base}/spectrum"))

    whole = client.post(f"{base}/region-spectrum", json={"kind": "rect", "x0": 0, "y0": 0, "x1": 256, "y1": 256})
    assert '"pixels": 65536' in whole.headers["X-Array-Meta"]
    np.testing.assert_allclose(_array(whole), total, rtol=1e-6)

    px = client.get(f"{base}/pixel-spectrum", params={"x": 10, "y": 20, "size": 3})
    assert '"pixels": 9' in px.headers["X-Array-Meta"]
    assert client.get(f"{base}/pixel-spectrum", params={"x": 999, "y": 0}).status_code == 416

    m = _array(client.get(f"{base}/energy-map", params={"lo": 6.2, "hi": 6.6}))  # Fe Kα
    assert m.size == 256 * 256 and m.sum() > 0


def test_elements(si_client):
    client, si = si_client
    table = client.get("/api/elements").json()
    fe = next(e for e in table if e["symbol"] == "Fe")
    assert fe["Z"] == 26 and any(l["line"] == "Ka" and abs(l["energy"] - 6.404) < 1e-3 for l in fe["lines"])

    assert si["eds"]["beam_energy"] == 300.0
    assert "Fe" in si["eds"]["elements"]
    r = client.put(f"/api/datasets/{si['id']}/elements", json={"elements": ["Fe", "O"]})
    eds = r.json()["dataset"]["eds"]
    assert eds["elements"] == ["Fe", "O"] and "Fe_Ka" in eds["xray_lines"]
    r = client.put(f"/api/datasets/{si['id']}/elements", json={"elements": []})
    assert r.json()["dataset"]["eds"]["elements"] == []
    assert client.put(f"/api/datasets/{si['id']}/elements", json={"elements": ["Xx"]}).status_code == 422
