"""Energy-window maps kept from a spectrum image ("Keep as map")."""

import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from si_viewer.app import create_app
from si_viewer.processing.filters import Filters, filter_image
from si_viewer.processing.regions import window_label


@pytest.fixture
def client():
    return TestClient(create_app())


def _open(client, path) -> dict:
    r = client.post("/api/files", json={"path": str(path)})
    assert r.status_code == 200, r.text
    return r.json()


def _si(entry) -> dict:
    return next(d for d in entry["datasets"] if d["kind"] == "spectrum_image")


def _windows(entry) -> list[dict]:
    return [d for d in entry["datasets"] if d["origin"] == "window"]


def _data(client, ds) -> np.ndarray:
    return np.asarray(client.app.state.session.get_dataset(ds["id"]).signal.data)


def _window(si_ds, lo_frac=0.31, hi_frac=0.43):
    """A window inside the energy axis that doesn't sit on channel edges."""
    ax = next(a for a in si_ds["axes"] if not a["navigate"])
    lo = ax["offset"] + ax["scale"] * ax["size"] * lo_frac
    hi = ax["offset"] + ax["scale"] * ax["size"] * hi_frac
    return float(lo), float(hi), ax


def _expected(cube, ax, lo, hi):
    """The window's channels summed, by the rule of the viewer's energy map."""
    n = cube.shape[-1]
    i0 = int(np.clip(math.floor((lo - ax["offset"]) / ax["scale"]), 0, n - 1))
    i1 = int(np.clip(math.ceil((hi - ax["offset"]) / ax["scale"]), i0 + 1, n))
    return cube[..., i0:i1].sum(axis=-1, dtype=np.float32)


def _scan_axes(ds) -> list[dict]:
    """The calibration of the scan: a spectrum image's navigation axes, an
    image's own axes (array order)."""
    keys = ("name", "size", "scale", "offset", "units")
    scan = ds["kind"] == "image"
    return [{k: a[k] for k in keys} for a in ds["axes"] if scan or a["navigate"]]


def _keep(client, si, lo, hi, **body):
    return client.post(f"/api/datasets/{si['id']}/window-maps", json={"lo": lo, "hi": hi, **body})


@pytest.fixture(params=["eels_cuzn", "example_emd"])
def spectrum_image_path(request):
    return request.getfixturevalue(request.param)


def test_window_map_is_the_sum_of_the_window_channels(client, spectrum_image_path):
    entry = _open(client, spectrum_image_path)
    si = _si(entry)
    lo, hi, ax = _window(si)
    r = _keep(client, si, lo, hi)
    assert r.status_code == 200, r.text
    body = r.json()
    ds = body["dataset"]

    # The summary: an image under the SI, with its window.
    assert ds["kind"] == "image" and ds["origin"] == "window" and ds["parent_id"] == si["id"]
    assert ds["window"] == {"lo": lo, "hi": hi, "units": ax["units"]}
    assert ds["title"] == window_label(lo, hi, ax["units"], ax["scale"]) and not ds["is_element_map"]
    assert ds["element"] is None and ds["signal_shape"] == si["navigation_shape"]
    assert si["window"] is None
    assert [d["id"] for d in _windows(body["file"])] == [ds["id"]]

    # The values, and the scan's calibration.
    np.testing.assert_array_equal(_data(client, ds), _expected(_data(client, si), ax, lo, hi))
    assert _scan_axes(ds) == _scan_axes(si)
    assert len(ds["axes"]) == 2 and not any(a["navigate"] for a in ds["axes"])

    # Limits in either order give the same map.
    swapped = _keep(client, si, hi, lo).json()["dataset"]
    assert swapped["window"] == ds["window"]
    np.testing.assert_array_equal(_data(client, swapped), _data(client, ds))


def test_label_and_default_names(client, eels_cuzn):
    si = _si(_open(client, eels_cuzn))
    lo, hi, ax = _window(si)
    mid = round(ax["offset"] + ax["scale"] * ax["size"] / 2)
    assert _keep(client, si, mid, mid + 20).json()["dataset"]["title"] == f"{mid}–{mid + 20} {ax['units']}".strip()
    named = _keep(client, si, lo, hi, label="  Cu L  ").json()["dataset"]
    assert named["title"] == "Cu L"
    assert _keep(client, si, lo, hi, label="").json()["dataset"]["title"] == window_label(lo, hi, ax["units"], ax["scale"])
    assert window_label(660, 640.5, "eV") == "640.5–660 eV"
    assert window_label(6.2, 6.6, "keV") == "6.2–6.6 keV"
    assert window_label(1, 2.34567, None) == "1–2.346"
    assert window_label(916.692, 1068.579, "eV", 8.0) == "917–1069 eV"
    assert window_label(6.2047, 6.6512, "keV", 0.005) == "6.205–6.651 keV"
    assert window_label(531.234, 540.0, "eV", 0.09) == "531.23–540 eV"


def test_window_map_can_be_deleted_alone(client, spectrum_image_path):
    entry = _open(client, spectrum_image_path)
    si = _si(entry)
    lo, hi, _ = _window(si)
    ds = _keep(client, si, lo, hi).json()["dataset"]
    elements = si["eds"]["elements"] if si["eds"] else None
    before = {d["id"] for d in entry["datasets"]}

    r = client.delete(f"/api/datasets/{ds['id']}")
    assert r.status_code == 200, r.text
    assert {d["id"] for d in r.json()["file"]["datasets"]} == before
    # The spectrum image's elements and maps are untouched.
    si_now = next(d for d in r.json()["file"]["datasets"] if d["id"] == si["id"])
    assert (si_now["eds"]["elements"] if si_now["eds"] else None) == elements
    assert client.delete(f"/api/datasets/{ds['id']}").status_code == 404


def test_window_map_is_deleted_with_its_spectrum_image(client, spectrum_image_path):
    entry = _open(client, spectrum_image_path)
    si = _si(entry)
    lo, hi, _ = _window(si)
    ds = _keep(client, si, lo, hi).json()["dataset"]
    assert client.delete(f"/api/files/{entry['id']}").status_code in (200, 204)
    assert ds["id"] not in client.app.state.session.datasets
    assert client.get(f"/api/datasets/{ds['id']}").status_code == 404


def test_window_map_follows_rebinning(client, spectrum_image_path):
    entry = _open(client, spectrum_image_path)
    si = _si(entry)
    lo, hi, ax = _window(si)
    ds = _keep(client, si, lo, hi, label="kept").json()["dataset"]

    r = client.post(f"/api/datasets/{si['id']}/rebin", json={"factors": [2, 2, 1]})
    assert r.status_code == 200, r.text
    file = r.json()["file"]
    (now,) = _windows(file)
    si_now = r.json()["dataset"]
    # Same dataset, remade at the new size from the rebinned spectrum image.
    assert now["id"] == ds["id"] and now["revision"] == ds["revision"] + 1 and now["title"] == "kept"
    assert now["window"] == ds["window"] and now["parent_id"] == si["id"]
    assert now["signal_shape"] == si_now["navigation_shape"] != ds["signal_shape"]
    np.testing.assert_array_equal(_data(client, now), _expected(_data(client, si_now), ax, lo, hi))
    assert _scan_axes(now) == _scan_axes(si_now)


def test_window_map_follows_the_spatial_filter(client, eels_cuzn):
    si = _si(_open(client, eels_cuzn))
    lo, hi, ax = _window(si)
    ds = _keep(client, si, lo, hi).json()["dataset"]
    raw = _data(client, ds).copy()

    f = {"spatial": "gaussian", "sigma": 1.5}
    r = client.put(f"/api/datasets/{si['id']}/filters", json=f)
    assert r.status_code == 200, r.text
    (now,) = _windows(r.json()["file"])
    assert now["revision"] == ds["revision"] + 1
    spec = Filters(**f)
    np.testing.assert_allclose(_data(client, now), filter_image(raw, spec), rtol=1e-5)

    # The map can't be filtered on its own; removing the filter restores it.
    assert client.put(f"/api/datasets/{ds['id']}/filters", json={"spatial": "median", "size": 3}).status_code >= 400
    r = client.put(f"/api/datasets/{si['id']}/filters", json={})
    np.testing.assert_allclose(_data(client, _windows(r.json()["file"])[0]), raw, rtol=1e-6)


def test_window_map_errors(client, eels_cuzn, eels_linescan):
    si = _si(_open(client, eels_cuzn))
    lo, hi, ax = _window(si)
    top = ax["offset"] + ax["scale"] * ax["size"]
    # Outside the energy range, or not numbers.
    assert _keep(client, si, top + 10, top + 20).status_code == 422
    assert _keep(client, si, ax["offset"] - 50, ax["offset"] - 5).status_code == 422
    assert client.post(f"/api/datasets/{si['id']}/window-maps", json={"lo": "a", "hi": 1}).status_code == 422
    assert client.post("/api/datasets/nope/window-maps", json={"lo": 1, "hi": 2}).status_code == 404
    # Only spectrum images: not an image (or its window map).
    ds = _keep(client, si, lo, hi).json()["dataset"]
    assert client.post(f"/api/datasets/{ds['id']}/window-maps", json={"lo": 1, "hi": 2}).status_code == 400

    # A line scan has no maps.
    scan = _si(_open(client, eels_linescan))
    assert scan["line_scan"]
    l, h, _ = _window(scan)
    r = _keep(client, scan, l, h)
    assert r.status_code == 400 and "line scan" in r.json()["detail"]


def _move(client, ds, lo, hi):
    return client.patch(f"/api/datasets/{ds['id']}/window", json={"lo": lo, "hi": hi})


def test_window_map_can_be_moved_in_place(client, spectrum_image_path):
    entry = _open(client, spectrum_image_path)
    si = _si(entry)
    lo, hi, ax = _window(si)
    ds = _keep(client, si, lo, hi).json()["dataset"]
    lo2, hi2, _ = _window(si, 0.52, 0.66)

    r = _move(client, ds, hi2, lo2)  # limits in either order
    assert r.status_code == 200, r.text
    now = r.json()["dataset"]
    # Same dataset, recomputed: new window, revision, and (automatic) name.
    assert now["id"] == ds["id"] and now["revision"] == ds["revision"] + 1 and now["parent_id"] == si["id"]
    assert now["window"] == {"lo": lo2, "hi": hi2, "units": ax["units"]}
    assert now["title"] == window_label(lo2, hi2, ax["units"], ax["scale"]) != ds["title"]
    np.testing.assert_array_equal(_data(client, now), _expected(_data(client, si), ax, lo2, hi2))
    (listed,) = _windows(r.json()["file"])
    assert listed["id"] == ds["id"] and listed["window"] == now["window"]
    assert len(r.json()["file"]["datasets"]) == len(entry["datasets"]) + 1


def test_moving_a_window_map_keeps_a_label_the_user_gave(client, eels_cuzn):
    si = _si(_open(client, eels_cuzn))
    lo, hi, _ = _window(si)
    ds = _keep(client, si, lo, hi, label="Cu L").json()["dataset"]
    lo2, hi2, _ = _window(si, 0.5, 0.6)
    assert _move(client, ds, lo2, hi2).json()["dataset"]["title"] == "Cu L"


def test_a_renamed_window_map_keeps_its_name_when_moved_or_refiltered(client, eels_cuzn):
    si = _si(_open(client, eels_cuzn))
    lo, hi, _ = _window(si)
    ds = _keep(client, si, lo, hi).json()["dataset"]
    r = client.put(f"/api/datasets/{ds['id']}/title", json={"title": "  Cu   L edge "})
    assert r.status_code == 200 and r.json()["title"] == "Cu L edge"
    lo2, hi2, _ = _window(si, 0.5, 0.6)
    assert _move(client, ds, lo2, hi2).json()["dataset"]["title"] == "Cu L edge"
    f = {"spatial": "gaussian", "sigma": 1.5}
    assert client.put(f"/api/datasets/{si['id']}/filters", json=f).status_code == 200
    m = client.app.state.session.get_dataset(ds["id"])
    assert m.title == "Cu L edge" and m.signal.metadata.General.title == "Cu L edge"


def test_rename_errors(client, eels_cuzn):
    si = _si(_open(client, eels_cuzn))
    assert client.put(f"/api/datasets/{si['id']}/title", json={"title": "   "}).status_code == 422
    assert client.put("/api/datasets/nope/title", json={"title": "x"}).status_code == 404
    r = client.put(f"/api/datasets/{si['id']}/title", json={"title": "Cu–Zn SI"})
    assert r.json()["title"] == "Cu–Zn SI"


def test_moved_window_map_follows_the_spatial_filter(client, eels_cuzn):
    si = _si(_open(client, eels_cuzn))
    lo, hi, ax = _window(si)
    ds = _keep(client, si, lo, hi).json()["dataset"]
    f = {"spatial": "gaussian", "sigma": 1.5}
    assert client.put(f"/api/datasets/{si['id']}/filters", json=f).status_code == 200
    lo2, hi2, _ = _window(si, 0.5, 0.6)
    now = _move(client, ds, lo2, hi2).json()["dataset"]
    raw = _expected(_data(client, si), ax, lo2, hi2)
    np.testing.assert_allclose(_data(client, now), filter_image(raw, Filters(**f)), rtol=1e-5)


def test_move_window_map_errors(client, eels_cuzn):
    entry = _open(client, eels_cuzn)
    si = _si(entry)
    lo, hi, ax = _window(si)
    ds = _keep(client, si, lo, hi).json()["dataset"]
    top = ax["offset"] + ax["scale"] * ax["size"]
    # Outside the energy range (the map is left as it was), or not numbers.
    assert _move(client, ds, top + 10, top + 20).status_code == 422
    assert _move(client, ds, ax["offset"] - 50, ax["offset"] - 5).status_code == 422
    assert client.patch(f"/api/datasets/{ds['id']}/window", json={"lo": "a", "hi": 1}).status_code == 422
    still = client.app.state.session.get_dataset(ds["id"])
    assert still.window == ds["window"] and still.revision == ds["revision"]
    # Only window maps.
    assert _move(client, si, lo, hi).status_code == 400
    assert client.patch("/api/datasets/nope/window", json={"lo": lo, "hi": hi}).status_code == 404


def test_window_snapped_to_channels_sums_exactly_those_channels(client, eels_cuzn):
    si = _si(_open(client, eels_cuzn))
    ax = next(a for a in si["axes"] if not a["navigate"])
    i0, i1 = 7, 23
    lo, hi = ax["offset"] + i0 * ax["scale"], ax["offset"] + i1 * ax["scale"]
    ds = _keep(client, si, lo, hi).json()["dataset"]
    cube = _data(client, si)
    np.testing.assert_array_equal(_data(client, ds), cube[..., i0:i1].sum(axis=-1, dtype=np.float32))
