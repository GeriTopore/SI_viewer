"""Sub-scan images: images sampling a spectrum image's field of view k times
more finely are offered as navigators at their own resolution."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import hyperspy.api as hs

from si_viewer.app import create_app
from si_viewer.subscan import Plane, scan_factor

SCAN = Plane(5, 3, 0.25, 0.25, "nm", "nm")


def finer(w, h, scale=0.0625, units="nm") -> Plane:
    return Plane(w, h, scale, scale, units, units)


# ---------- the arithmetic ----------


def test_the_scan_itself_is_k_1():
    assert scan_factor(SCAN, SCAN) == 1
    # k = 1 never looks at the calibration (as before sub-scan support).
    assert scan_factor(SCAN, Plane(5, 3, 7.0, 7.0, "µm", "µm")) == 1


@pytest.mark.parametrize("k", [2, 4, 7])
def test_whole_multiples_over_the_same_field_of_view(k):
    assert scan_factor(SCAN, finer(5 * k, 3 * k, 0.25 / k)) == k


@pytest.mark.parametrize(
    "image",
    [
        finer(18, 10),  # not a multiple
        finer(20, 9),  # a multiple on x (4), another on y (3)
        finer(10, 12),  # 2 on x, 4 on y
        finer(4, 3),  # coarser than the scan
        finer(2, 3),
        finer(0, 0),
    ],
)
def test_other_shapes_are_not_sub_scans(image):
    assert scan_factor(SCAN, image) is None


def test_field_of_view_must_agree_within_one_percent():
    assert scan_factor(SCAN, finer(20, 12, 0.0625 * 1.009)) == 4
    assert scan_factor(SCAN, finer(20, 12, 0.0625 * 1.02)) is None
    assert scan_factor(SCAN, finer(20, 12, 0.25)) is None  # same scale: 4 × the field of view
    # Units are converted: 0.0625 nm = 62.5 pm.
    assert scan_factor(SCAN, finer(20, 12, 62.5, "pm")) == 4
    assert scan_factor(SCAN, finer(20, 12, 0.0625, "µm")) is None
    # One axis off is enough.
    assert scan_factor(SCAN, Plane(20, 12, 0.0625, 0.0625 * 1.5, "nm", "nm")) is None


def test_uncalibrated_planes_are_taken_for_the_same_field_of_view():
    assert scan_factor(Plane(5, 3), Plane(20, 12)) == 4  # no units: nothing to compare
    assert scan_factor(SCAN, Plane(20, 12, 1.0, 1.0, "", "")) == 4
    assert scan_factor(SCAN, Plane(20, 12, 1.0, 1.0, "<undefined>", "<undefined>")) == 4


# ---------- navigator sources of an ordinary spectrum image ----------


def image(w, h, title, scale, fill=1.0):
    s = hs.signals.Signal2D(np.full((h, w), fill, dtype=np.float32))
    for ax in s.axes_manager.signal_axes:
        ax.scale, ax.units = scale, "nm"
    s.metadata.General.title = title
    return s


@pytest.fixture
def client_with_subscan():
    si = hs.signals.Signal1D(np.arange(3 * 5 * 8, dtype=np.float32).reshape(3, 5, 8))
    si.set_signal_type("EELS")
    for ax in si.axes_manager.navigation_axes:
        ax.scale, ax.units = 0.25, "nm"
    si.metadata.General.title = "SI"
    signals = [
        si,
        image(20, 12, "ADF sub-scan", 0.0625, 3.0),  # 4 ×, the companion
        image(10, 6, "Half sub-scan", 0.125),  # 2 ×
        image(5, 3, "Same grid", 0.25),
        image(20, 12, "Wrong field of view", 0.25),  # 4 × the pixels, 4 × the field of view
        image(18, 10, "Odd size", 0.0625),
    ]
    app = create_app()
    entry = app.state.session.open_signals(Path("subscan.test"), signals)
    return TestClient(app), entry


def test_sub_scan_images_are_navigators_at_their_own_resolution(client_with_subscan):
    client, entry = client_with_subscan
    session = client.app.state.session
    ids = {session.datasets[i].title: i for i in entry.dataset_ids}
    si = ids["SI"]
    sources = client.get(f"/api/datasets/{si}/navigators").json()
    by_label = {s["label"]: s for s in sources}
    assert sources[0]["label"] == "ADF sub-scan"  # the companion image comes first
    assert by_label["ADF sub-scan"] == {
        "id": ids["ADF sub-scan"], "label": "ADF sub-scan", "factor": 4, "dataset_id": ids["ADF sub-scan"],
    }
    assert by_label["Half sub-scan"]["factor"] == 2
    assert by_label["Same grid"]["factor"] == 1
    assert by_label["Total counts"] == {"id": si, "label": "Total counts", "factor": 1, "dataset_id": si}
    # A multiple over another field of view, and a size that is no multiple, are not offered.
    assert "Wrong field of view" not in by_label and "Odd size" not in by_label

    # The image is served at its own size, with its own calibration.
    adf = ids["ADF sub-scan"]
    r = client.get(f"/api/datasets/{adf}/frame")
    assert r.headers["X-Array-Shape"] == "12,20"
    summary = client.get(f"/api/datasets/{adf}").json()
    x = next(a for a in summary["axes"] if a["name"] == "x" or a["size"] == 20)
    assert x["size"] == 20 and x["scale"] == pytest.approx(0.0625) and x["units"] == "nm"
    # The spectrum image's own scan is untouched.
    assert client.get(f"/api/datasets/{si}").json()["navigation_shape"] == [5, 3]


def test_sub_scan_companion_is_the_thumbnail(client_with_subscan):
    client, entry = client_with_subscan
    session = client.app.state.session
    si = next(session.datasets[i] for i in entry.dataset_ids if session.datasets[i].title == "SI")
    r = client.get(f"/api/datasets/{si.id}/thumbnail")
    assert r.status_code == 200 and r.content[:4] == b"\x89PNG"
