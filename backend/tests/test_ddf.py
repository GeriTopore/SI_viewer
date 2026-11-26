"""Digital dark field (py4DSTEM's DDFimage): virtual images from the Bragg
vectors, a disk counting where it lies inside the detector."""

import numpy as np
import pytest

from si_viewer.processing import disks
from si_viewer.processing.fourd import Detector, Point


def _table(rows):
    """Rows of (scan x, scan y, kx, ky, score) → the vectors table (scan y, x, ky, kx, score)."""
    return np.array([[sy + 0.5, sx + 0.5, ky, kx, s] for sx, sy, kx, ky, s in rows], np.float32)


TABLE = _table([
    (0, 0, 10, 10, 5.0),   # on the aperture at (10, 10)
    (0, 0, 20, 20, 3.0),   # on the aperture at (20, 20)
    (1, 0, 10.8, 10, 2.0),  # within r = 1 of (10, 10)
    (1, 0, 30, 30, 9.0),   # on no aperture
    (0, 1, 11.5, 10, 4.0),  # outside r = 1
])
APERTURES = Detector(shape="apertures", r=1.0, apertures=[Point(x=10, y=10), Point(x=20, y=20)])


def test_ddf_sums_the_matched_disks():
    img = disks.ddf_image(TABLE, (2, 2), APERTURES)
    np.testing.assert_allclose(img, [[8.0, 2.0], [0.0, 0.0]])
    count = disks.ddf_image(TABLE, (2, 2), APERTURES, weight="count")
    np.testing.assert_allclose(count, [[2, 1], [0, 0]])
    np.testing.assert_allclose(disks.ddf_image(TABLE, (2, 2), APERTURES, "max"), [[5.0, 2.0], [0, 0]])
    np.testing.assert_allclose(disks.ddf_image(TABLE, (2, 2), APERTURES, "mean"), [[4.0, 2.0], [0, 0]])


def test_ddf_counts_a_disk_once_and_other_shapes():
    # Two apertures over the same disk: it counts once.
    twice = Detector(shape="apertures", r=2.0, apertures=[Point(x=10, y=10), Point(x=11, y=10)])
    assert disks.ddf_image(TABLE, (2, 2), twice, weight="count")[0, 0] == 1
    circle = Detector(shape="circle", cx=20, cy=20, r=15)
    np.testing.assert_allclose(disks.ddf_image(TABLE, (2, 2), circle, weight="count"), [[2, 2], [1, 0]])
    annulus = Detector(shape="annulus", cx=20, cy=20, r=15, r_inner=5)
    np.testing.assert_allclose(disks.ddf_image(TABLE, (2, 2), annulus, weight="count"), [[1, 2], [1, 0]])


def test_an_aperture_keeps_its_own_radius():
    # Merged groups: an aperture with its own radius, the others the detector's.
    from si_viewer.processing.fourd import detector_mask

    d = Detector(shape="apertures", r=1.0, apertures=[Point(x=10, y=10), Point(x=20, y=20, r=3.0)])
    mask = detector_mask((32, 32), d)
    assert mask[9, 9] and not mask[9, 11] and mask[19, 22]  # pixel centres at +0.5
    big = Detector(shape="apertures", r=1.0, apertures=[Point(x=10, y=10, r=2.0), Point(x=20, y=20)])
    # The disk at 11.5 is within 2 of (10, 10) now.
    assert disks.ddf_image(TABLE, (2, 2), big, weight="count")[1, 0] == 1


def test_ddf_follows_the_centring():
    # Position (1, 0)'s beam is 3 px right of position (0, 0)'s, and so are
    # its disks. Centred on the mean origin (x 33.5), both disks land at 11.5.
    table = _table([(0, 0, 10, 10, 1.0), (1, 0, 13, 10, 1.0)])
    origins = np.array([[[32, 32], [35, 32]]], float)  # [y, x, (x, y)]
    aperture = Detector(shape="apertures", r=1.0, apertures=[Point(x=11.5, y=10)])
    assert disks.ddf_image(table, (1, 2), aperture, weight="count").tolist() == [[0, 0]]
    np.testing.assert_allclose(disks.ddf_image(table, (1, 2), aperture, weight="count", origins=origins), [[1, 1]])


def test_ddf_through_the_api(tmp_path):
    from fastapi.testclient import TestClient
    from test_4dstem import _write_emdfile
    from test_bragg_calibration import NX, NY, _cube, _origin, _wait

    from si_viewer.app import create_app
    from si_viewer.session import Session

    path = tmp_path / "ring.h5"
    _write_emdfile(path, _cube(), extra_2d=False)
    session = Session()
    client = TestClient(create_app(session))
    cube_id = client.post("/api/files", json={"path": str(path)}).json()["datasets"][0]["id"]
    url = f"/api/datasets/{cube_id}/4d"

    def image(body):
        r = client.post(f"{url}/virtual", json=body)
        assert r.status_code == 200, r.text
        return np.frombuffer(r.content, np.float32).reshape([int(v) for v in r.headers["X-Array-Shape"].split(",")])

    # A small aperture on the direct beam where it is on average (it drifts
    # over a pixel across the scan).
    mean = np.mean([_origin(iy, ix) for iy in range(NY) for ix in range(NX)], axis=0)
    beam = {"shape": "apertures", "r": 0.5, "apertures": [{"x": mean[0], "y": mean[1]}]}
    body = {"detector": beam, "source": "vectors", "weight": "count"}
    assert client.post(f"{url}/virtual", json=body).status_code == 409  # no disks yet
    job = _wait(client, client.post(f"{url}/disks/run", json={"options": {"threshold": 0.3, "min_spacing": 3}}).json())
    assert job["status"] == "done", job["error"]
    # Not centred: the beam's disk misses the aperture at the scan's edges.
    raw = image(body)
    assert raw.shape == (NY, NX) and 0 < raw.sum() < NY * NX
    # Centred (each position's origin from its disks, a descan plane): it is caught everywhere, once.
    r = client.post(f"{url}/calibration/origin", json={"source": "vectors", "plane": {"mode": "plane"}})
    assert r.status_code == 200, r.text
    np.testing.assert_array_equal(image(body), np.ones((NY, NX)))
    # Its intensity: the disks' scores; the pixel image of the same detector is another thing.
    assert image({**body, "weight": "intensity"}).min() > 0
    two = client.post(f"{url}/virtuals", json={"items": [body, {"detector": beam}]})
    assert two.status_code == 200 and two.headers["X-Array-Shape"] == f"2,{NY},{NX}"
    # Saved: titled as a DDF, with how it was made.
    saved = client.post(f"{url}/virtual/save", json=body).json()
    made = session.get_dataset(saved["id"])
    assert made.title.startswith("DDF ") and made.virtual["source"] == "vectors" and made.virtual["weight"] == "count"
