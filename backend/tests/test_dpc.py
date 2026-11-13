"""DPC and iDPC from four detector quadrants (processing/dpc.py)."""

import math

from pathlib import Path

import numpy as np
import pytest

from si_viewer.processing import dpc


def _potential(n=128, seed=0):
    """Gaussian 'atoms' on a smooth background: a potential and its gradient."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n].astype(float)
    phi = np.zeros((n, n))
    for cy, cx in rng.uniform(8, n - 8, (25, 2)):
        phi += np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 2.5**2))
    gy, gx = np.gradient(phi)
    return phi, gx, gy


def _segments(gx, gy, theta, mirror):
    """Four quadrant images whose normalised differences are the field
    (gx, gy) seen in a detector frame rotated by -theta (and mirrored)."""
    t = math.radians(-theta)
    dx = math.cos(t) * gx - math.sin(t) * gy
    dy = math.sin(t) * gx + math.cos(t) * gy
    if mirror:
        dy = -dy
    return 1 + dx / 2, 1 + dy / 2, 1 - dx / 2, 1 - dy / 2  # A, B, C, D (sum 4)


def test_quadrants_and_pairs():
    assert [dpc.quadrant(s) for s in ([0, 4], [1, 5], [2, 6], [3, 7])] == [0, 1, 2, 3]
    assert dpc.quadrant([2]) == 2
    with pytest.raises(ValueError):
        dpc.quadrant([0, 5])
    with pytest.raises(ValueError):
        dpc.quadrant([])


@pytest.mark.parametrize("theta,mirror", [(0, False), (37, False), (135.7, True), (250, False), (300, True)])
def test_the_rotation_is_found_and_the_potential_recovered(theta, mirror):
    phi, gx, gy = _potential()
    a, b, c, d = _segments(gx, gy, theta, mirror)
    dx, dy = dpc.differences(a, b, c, d, normalise=True)
    np.testing.assert_allclose(dx, (a - c) / 4)
    found = dpc.find_rotation(dx, dy, coarse=1000)
    assert found["mirror"] == mirror
    assert abs((found["theta"] - theta + 180) % 360 - 180) < 0.5
    assert found["curl_ratio"] < 0.05
    rx, ry = dpc.rotate(dx, dy, found["theta"], found["mirror"])
    np.testing.assert_allclose(rx * 4, gx, atol=1e-6)
    np.testing.assert_allclose(ry * 4, gy, atol=1e-6)
    i = dpc.integrate(rx, ry)
    assert np.corrcoef(i.ravel(), phi.ravel())[0, 1] > 0.98


# ---------- through the API: the DPC and iDPC objects ----------


def test_dpc_and_idpc_objects(tmp_path):
    import hyperspy.api as hs
    from fastapi.testclient import TestClient

    from si_viewer.app import create_app

    phi, gx, gy = _potential(64)
    folder = tmp_path / "run"
    folder.mkdir()
    for letter, segs, data in zip("ABCD", ([0, 4], [1, 5], [2, 6], [3, 7]), _segments(gx, gy, 30, False)):
        s = hs.signals.Signal2D(data.astype(np.float32))
        s.metadata.General.title = f"DF-{letter}"
        s.metadata.set_item("Acquisition_instrument.TEM.Detector.name", "DFS")
        s.metadata.set_item("Acquisition_instrument.TEM.Detector.segments", segs)
        s.save(folder / f"{letter}.hspy")
    client = TestClient(create_app())
    r = client.post("/api/files/folder", json={"path": str(folder)}).json()
    ids = {}
    for p in r["paths"]:
        f = client.post("/api/files", json={"path": p, "experiment_id": r["experiment"]["id"]}).json()
        ids[f["datasets"][0]["title"]] = f["datasets"][0]["id"]

    info = client.get(f"/api/datasets/{ids['DF-B']}/dpc").json()
    assert info["choice"] == {k: ids[f"DF-{k}"] for k in "ABCD"} and info["settings"] is None
    wrong = {**info["choice"], "A": ids["DF-B"]}
    assert client.post("/api/dpc", json={"inputs": wrong}).status_code == 422
    rot = client.post("/api/dpc/rotation", json={"inputs": info["choice"]}).json()
    assert abs((rot["theta"] - 30 + 180) % 360 - 180) < 1 and not rot["mirror"]

    made = client.post("/api/dpc", json={"inputs": info["choice"], "theta": 30}).json()["ids"]
    session = client.app.state.session
    colour, x, y, idpc = (session.get_dataset(made[k]) for k in ("colour", "x", "y", "idpc"))
    assert colour.title == "DPC" and idpc.title == "iDPC"
    assert x.parent_id == colour.id and y.parent_id == colour.id
    assert colour.parent_id is None and idpc.parent_id is None and idpc.derived_from is None
    strong = np.hypot(gx, gy) > 0.01 * np.hypot(gx, gy).max()  # the direction of ~0 is float32 noise
    np.testing.assert_allclose(np.asarray(colour.signal.data)[strong], np.degrees(np.arctan2(gy, gx))[strong], atol=0.5)
    mag = client.get(f"/api/datasets/{colour.id}/dpc/magnitude")
    assert mag.status_code == 200
    np.testing.assert_allclose(
        np.frombuffer(mag.content, "<f4").reshape(64, 64), np.hypot(gx, gy) / 4, rtol=1e-3, atol=1e-5)  # normalised: sum 4
    # Made again in place; the settings come back with the choice.
    again = client.post("/api/dpc", json={"inputs": info["choice"], "theta": 10}).json()["ids"]
    assert again == made and session.get_dataset(made["colour"]).dpc["theta"] == 10
    assert client.get(f"/api/datasets/{made['idpc']}/dpc").json()["settings"]["theta"] == 10
    # Deleting the DPC object takes x and y with it.
    assert client.delete(f"/api/datasets/{made['colour']}").status_code == 200
    assert made["x"] not in session.datasets and made["idpc"] in session.datasets


def test_dpc_of_stacks(tmp_path):
    """Four stacks: DPC, DPC x, DPC y and iDPC stacks, each frame from the
    same frames; a frame of the DPC stack (and its mean) in colour."""
    import hyperspy.api as hs
    from fastapi.testclient import TestClient

    from si_viewer.app import create_app

    fields = [_potential(64, seed=k)[1:] for k in range(3)]
    folder = tmp_path / "run"
    folder.mkdir()
    for q, (letter, segs) in enumerate(zip("ABCD", ([0, 4], [1, 5], [2, 6], [3, 7]))):
        frames = np.stack([_segments(gx, gy, 30, False)[q] for gx, gy in fields]).astype(np.float32)
        s = hs.signals.Signal2D(frames)
        s.metadata.General.title = f"DF-{letter}"
        s.metadata.set_item("Acquisition_instrument.TEM.Detector.name", "DFS")
        s.metadata.set_item("Acquisition_instrument.TEM.Detector.segments", segs)
        s.save(folder / f"{letter}.hspy")
    client = TestClient(create_app())
    r = client.post("/api/files/folder", json={"path": str(folder)}).json()
    ids = {}
    for path in r["paths"]:
        f = client.post("/api/files", json={"path": path, "experiment_id": r["experiment"]["id"]}).json()
        assert f["datasets"][0]["kind"] == "stack"
        ids[f["datasets"][0]["title"]] = f["datasets"][0]["id"]

    info = client.get(f"/api/datasets/{ids['DF-A']}/dpc").json()
    assert info["choice"] == {k: ids[f"DF-{k}"] for k in "ABCD"}
    rot = client.post("/api/dpc/rotation", json={"inputs": info["choice"]}).json()
    assert abs((rot["theta"] - 30 + 180) % 360 - 180) < 1 and not rot["mirror"]
    made = client.post("/api/dpc", json={"inputs": info["choice"], "theta": 30}).json()["ids"]
    session = client.app.state.session
    colour, idpc = session.get_dataset(made["colour"]), session.get_dataset(made["idpc"])
    assert colour.kind.value == idpc.kind.value == "stack" and idpc.signal.data.shape == (3, 64, 64)
    for k, (gx, gy) in enumerate(fields):  # each frame its own
        phi = _potential(64, seed=k)[0]
        assert np.corrcoef(np.asarray(idpc.signal.data[k]).ravel(), phi.ravel())[0, 1] > 0.95
    shown = client.get(f"/api/datasets/{colour.id}/dpc/frame", params={"index": 2})
    assert shown.headers["X-Array-Shape"] == "2,64,64"
    angle, magnitude = np.frombuffer(shown.content, "<f4").reshape(2, 64, 64)
    gx, gy = fields[2]
    gx, gy = gx - gx.mean(), gy - gy.mean()
    np.testing.assert_allclose(magnitude, np.hypot(gx, gy) / 4, rtol=1e-3, atol=1e-5)
    mean = client.get(f"/api/datasets/{colour.id}/dpc/frame", params={"projection": "mean", "bin": 2})
    assert mean.status_code == 200 and mean.headers["X-Array-Shape"] == "2,32,32"
    assert client.get(f"/api/datasets/{colour.id}/dpc/frame", params={"index": 3}).status_code == 422
    # Images and stacks are not mixed.
    assert client.post("/api/dpc", json={"inputs": {**info["choice"], "A": made["x"]}}).status_code == 422


def test_dpc_stacks_aligned_from_the_haadf_shifts(tmp_path):
    """DPC stacks from unaligned segments, aligned afterwards with the shifts
    measured on the HAADF: shifted back and cropped as the HAADF is (the DPC
    object through its x and y)."""
    import hyperspy.api as hs
    from fastapi.testclient import TestClient

    from si_viewer.app import create_app

    fields = [_potential(64, seed=k)[1:] for k in range(3)]
    drift = [(0, 0), (2, -3), (-1, 4)]  # (rows, columns) each frame moved by
    folder = tmp_path / "run"
    folder.mkdir()
    for q, (letter, segs) in enumerate(zip("ABCD", ([0, 4], [1, 5], [2, 6], [3, 7]))):
        frames = np.stack([
            np.roll(_segments(gx, gy, 0, False)[q], drift[k], axis=(0, 1)) for k, (gx, gy) in enumerate(fields)
        ]).astype(np.float32)
        s = hs.signals.Signal2D(frames)
        s.metadata.General.title = f"DF-{letter}"
        s.metadata.set_item("Acquisition_instrument.TEM.Detector.segments", segs)
        s.save(folder / f"{letter}.hspy")
    hs.signals.Signal2D(np.ones((3, 64, 64), np.float32)).save(folder / "HAADF.hspy")
    client = TestClient(create_app())
    r = client.post("/api/files/folder", json={"path": str(folder)}).json()
    ids = {}
    for path in r["paths"]:
        f = client.post("/api/files", json={"path": path, "experiment_id": r["experiment"]["id"]}).json()
        ids[Path(path).stem] = f["datasets"][0]["id"]
    session = client.app.state.session
    info = client.get(f"/api/datasets/{ids['A']}/dpc").json()
    made = client.post("/api/dpc", json={"inputs": info["choice"], "normalise": False}).json()["ids"]

    assert client.get(f"/api/datasets/{made['idpc']}/alignment/sources").json() == []  # nothing measured yet
    square = [5, 6, 50, 48]  # x, y, width, height
    session.get_dataset(ids["HAADF"]).alignment = {
        "frames": [0, 1, 2], "square": square,
        "drift": {"frames": [0, 1, 2], "x": [c for _, c in drift], "y": [r for r, _ in drift]},
    }
    sources = client.get(f"/api/datasets/{made['colour']}/alignment/sources").json()
    assert [s["id"] for s in sources] == [ids["HAADF"]]
    for k in ("colour", "idpc"):
        r = client.post(f"/api/datasets/{made[k]}/alignment/from", json={"source": ids["HAADF"]})
        assert r.status_code == 200, r.text
    x0, y0, w, h = square
    x = session.get_dataset(made["x"])
    assert x.signal.data.shape == (3, 48, 50)
    for k, (gx, gy) in enumerate(fields):
        # The deflection of either sign, shifted back (each frame's mean was removed before).
        want = (gx - gx.mean())[y0:y0 + h, x0:x0 + w]
        np.testing.assert_allclose(np.asarray(x.signal.data[k]), want, atol=1e-4)
    colour = session.get_dataset(made["colour"])
    angle, _ = np.frombuffer(client.get(f"/api/datasets/{colour.id}/dpc/frame", params={"index": 1}).content, "<f4").reshape(2, 48, 50)
    np.testing.assert_allclose(angle, np.asarray(colour.signal.data[1]), atol=1e-3)
    idpc = session.get_dataset(made["idpc"])
    assert idpc.signal.data.shape == (3, 48, 50) and idpc.alignment is not None
    # The frames as loaded again: the DPC object with its x and y.
    assert client.put(f"/api/datasets/{colour.id}/alignment", json={"active": False}).status_code == 200
    assert x.signal.data.shape == (3, 64, 64)
    assert client.delete(f"/api/datasets/{colour.id}/alignment").status_code == 200
    assert x.alignment is None and colour.alignment is None


# ---------- against Velox's iDPC ----------

VELOX_IDPC = Path(__file__).resolve().parents[2] / "examples/STO/0025-20261112_1629_9.00_Mx_HAADF-DF.emd"


@pytest.mark.skipif(not VELOX_IDPC.exists(), reason="example file not present")
def test_matches_velox_idpc():
    """Velox's iDPC: the plain differences rotated by the recorded detector
    rotation (its rotationalOffset), no mirror, integrated as periodic."""
    import math

    from si_viewer.io import load_file

    images = {s.metadata.General.title: s for s in load_file(str(VELOX_IDPC), lazy=False)}
    a, b, c, d = (np.asarray(images[f"DF-{k}"].data, np.float64) for k in "ABCD")
    theta = math.degrees(float(images["DF-A"].original_metadata.CustomProperties.DetectorRotation.value))
    ours = dpc.integrate(*dpc.rotate(*dpc.differences(a, b, c, d, normalise=False), theta))
    velox = np.asarray(images["iDPC"].data, np.float64)
    scale = (ours * velox).sum() / (ours * ours).sum()
    assert np.std(velox - scale * ours) < 0.002 * np.std(velox)
