"""Masks made from images: thresholds, smoothing, clean-up, the datasets."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from si_viewer.app import create_app
from si_viewer.processing import masks


def _blobs(n=120):
    """Two bright discs (r 20 and r 3) on a dim noisy background."""
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[:n, :n]
    image = rng.normal(0, 0.05, (n, n))
    image[np.hypot(yy - 40, xx - 40) < 20] += 1
    image[np.hypot(yy - 90, xx - 90) < 3] += 1
    return image


def test_thresholds_and_cleaning():
    image = _blobs()
    above = masks.make_mask(image, masks.MaskOptions(keep="above", lo=0.5))
    assert above.dtype == bool and 1200 < above.sum() < 1350  # π·20² + π·3² ≈ 1285
    below = masks.make_mask(image, masks.MaskOptions(keep="below", hi=0.5))
    assert np.array_equal(below, ~above)
    inv = masks.make_mask(image, masks.MaskOptions(keep="above", lo=0.5, invert=True))
    assert np.array_equal(inv, below)
    both = masks.make_mask(image, masks.MaskOptions(keep="between", lo=-10, hi=0.5))
    assert np.array_equal(both, below)
    # The small disc goes when regions under 100 px are removed.
    big = masks.make_mask(image, masks.MaskOptions(keep="above", lo=0.5, min_area=100))
    assert big[40, 40] and not big[90, 90] and 1200 < big.sum() < 1280
    # Smoothing: thresholds on the blurred image.
    blurred = masks.make_mask(image, masks.MaskOptions(keep="above", lo=0.5, smooth=2))
    assert blurred[40, 40] and abs(int(blurred.sum()) - 1257) < 120
    with pytest.raises(ValueError):
        masks.MaskOptions(keep="between", lo=1, hi=0).check()
    with pytest.raises(ValueError):
        masks.MaskOptions(keep="above").check()


def test_values_and_otsu():
    image = _blobs()
    v = masks.threshold_values(image, 0, sample=500)
    assert len(v["values"]) == 500 and 0.1 < v["otsu"] < 0.9
    assert v["min"] < 0 and v["max"] > 1


def test_inside():
    m = np.zeros((10, 10), bool)
    m[2:5, 2:5] = True
    got = masks.inside(m, np.array([[3, 3], [3.4, 4.4], [6, 6], [np.nan, 3], [-1, 3], [3, 20]]))
    assert got.tolist() == [True, True, False, False, False, False]


def test_mask_datasets(tmp_path):
    import tifffile

    path = tmp_path / "blobs.tif"
    tifffile.imwrite(path, _blobs().astype(np.float32))
    c = TestClient(create_app())
    d = c.post("/api/files", json={"path": str(path)}).json()["datasets"][0]
    url = f"/api/datasets/{d['id']}/mask"

    v = c.post(f"{url}/values", json={"smooth": 1}).json()
    assert 0.1 < v["otsu"] < 0.9
    r = c.post(url, json={"keep": "above", "lo": v["otsu"]})
    assert r.status_code == 200, r.text
    made = {x["id"]: x for x in r.json()["file"]["datasets"]}
    m = made[r.json()["id"]]
    assert m["title"] == "Mask" and m["kind"] == "image" and m["origin"] == "mask" and m["parent_id"] == d["id"]
    assert m["mask"]["source_id"] == d["id"] and m["mask"]["lo"] == pytest.approx(v["otsu"])
    assert m["mask"]["fraction"] == pytest.approx(1285 / 120**2, abs=0.01)
    session = c.app.state.session
    data = np.asarray(session.get_dataset(m["id"]).signal.data)
    assert data.dtype == np.float32 and set(np.unique(data)) == {0.0, 1.0}

    # A second one is "Mask 2"; making the first again keeps its name.
    r2 = c.post(url, json={"keep": "below", "hi": 0.5}).json()
    assert next(x for x in r2["file"]["datasets"] if x["id"] == r2["id"])["title"] == "Mask 2"
    r3 = c.post(url, json={"keep": "above", "lo": 0.5, "min_area": 100, "replace": m["id"]}).json()
    again = next(x for x in r3["file"]["datasets"] if x["id"] == m["id"])
    assert r3["id"] == m["id"] and again["title"] == "Mask" and again["mask"]["min_area"] == 100
    assert again["revision"] > m["revision"]
    # Errors: bad options, not a mask of this image, not an image.
    assert c.post(url, json={"keep": "between", "lo": 1, "hi": 0}).status_code == 422
    assert c.post(url, json={"keep": "above", "lo": 0, "replace": d["id"]}).status_code == 422
    # Deleting one removes it (and undo brings it back).
    assert c.delete(f"/api/datasets/{r2['id']}").status_code == 200
    assert r2["id"] not in session.datasets
