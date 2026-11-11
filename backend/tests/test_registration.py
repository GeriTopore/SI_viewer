import numpy as np
import pytest
import tifffile
from fastapi.testclient import TestClient

from si_viewer.app import create_app


def _array(r) -> np.ndarray:
    assert r.status_code == 200, r.text
    shape = tuple(int(n) for n in r.headers["X-Array-Shape"].split(","))
    return np.frombuffer(r.content, "<f4").reshape(shape)


def _drifting_stack(shifts, size=96, seed=0):
    """Frames of one noisy lattice-and-blobs image, each moved by an integer
    (dy, dx) shift (periodic)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:size, :size]
    base = 1 + 0.5 * np.cos(2 * np.pi * x / 8) * np.cos(2 * np.pi * y / 8)
    for cy, cx in rng.uniform(10, size - 10, (6, 2)):
        base += 3 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 30)
    frames = [np.roll(base, (dy, dx), axis=(0, 1)) for dy, dx in shifts]
    return (np.array(frames) * 1000 + rng.normal(0, 30, (len(shifts), size, size))).astype("float32")


SHIFTS = [(0, 0), (1, 2), (3, 3), (4, 5), (6, 6), (7, 8)]


@pytest.fixture
def stack(tmp_path):
    client = TestClient(create_app())
    tifffile.imwrite(tmp_path / "drift.tif", _drifting_stack(SHIFTS))
    ds = client.post("/api/files", json={"path": str(tmp_path / "drift.tif")}).json()["datasets"][0]
    assert ds["kind"] == "stack"
    return client, ds


def test_session_and_previews(stack):
    client, ds = stack
    url = f"/api/datasets/{ds['id']}/registration"
    assert client.get(url).status_code == 404  # none yet

    state = client.post(url, json={}).json()
    assert (state["first"], state["last"], state["shape"]) == (0, 6, [96, 96])

    assert _array(client.get(f"{url}/frame", params={"i": 2})).shape == (96, 96)

    # The cross-correlation of frames 0 and 3 peaks at their shift.
    cc = _array(client.get(f"{url}/correlation", params={"i": 0, "j": 3, "type": "hann", "n": 4}))
    assert cc.shape == (96, 96)
    peak = np.unravel_index(np.argmax(cc), cc.shape)
    assert (peak[0] - 48, peak[1] - 48) == SHIFTS[3]

    # Comparing cutoffs; the session's own mask is unchanged by previews.
    ccs = _array(client.get(f"{url}/compare-n", params={"i": 0, "j": 1, "type": "hann", "n": [3, 6, 9]}))
    assert ccs.shape == (3, 96, 96)
    mask = client.get(url).json()["mask"]
    assert (mask["type"], mask["n"]) == ("hann", 6.0)

    state = client.put(url, json={"mask": {"type": "gaussian", "n": 3}, "bad_frames": [4, 99]}).json()
    assert state["mask"]["type"] == "gaussian" and state["bad_frames"] == [4]

    # A frame range keeps the settings; frames outside it are refused.
    state = client.post(url, json={"first": 1, "last": 5}).json()
    assert state["bad_frames"] == [4] and state["mask"]["n"] == 3
    assert client.get(f"{url}/frame", params={"i": 0}).status_code == 422
    assert client.post(url, json={"first": 3, "last": 4}).status_code == 422

    assert client.delete(url).status_code == 204
    assert client.get(url).status_code == 404


def test_only_stacks(stack):
    client, _ = stack
    entry = client.post("/api/examples/fept-si").json()
    haadf = next(d for d in entry["datasets"] if d["kind"] == "image")
    assert client.post(f"/api/datasets/{haadf['id']}/registration", json={}).status_code == 400


def _wait(client, job, timeout=120):
    import time

    t0 = time.time()
    while job["status"] in ("queued", "running"):
        assert time.time() - t0 < timeout, "job did not finish"
        time.sleep(0.05)
        job = client.get(f"/api/jobs/{job['id']}").json()
    return job


@pytest.mark.parametrize("method", ["pixel", "gf", "com"])
def test_peak_preview_matches_the_shift_matrices(stack, method):
    """The peak step's shift for a pair is the one the registration uses."""
    client, ds = stack
    url = f"/api/datasets/{ds['id']}/registration"
    client.post(url, json={})
    mask = {"type": "hann", "n": 4}
    peaks = {"method": method, "num_peaks": 3, "sigma_guess": 2, "window_radius": 5}
    client.put(url, json={"mask": mask, "peaks": peaks})

    previews = {}
    for i, j in [(0, 3), (1, 5), (4, 2)]:
        p = client.post(f"{url}/peak", json={"i": i, "j": j, "mask": mask, "peaks": peaks}).json()
        previews[i, j] = p
        if method == "pixel":  # the true shift (the others are biased by this lattice-like test image)
            (dy_i, dx_i), (dy_j, dx_j) = SHIFTS[i], SHIFTS[j]
            assert (p["dx"], p["dy"]) == (dx_j - dx_i, dy_j - dy_i)
        if method == "gf":
            assert len(p["candidates"]) == 3 and sum(c["chosen"] for c in p["candidates"]) == 1
            chosen = next(c for c in p["candidates"] if c["chosen"])
            assert len(chosen["data"]) == 11 and len(chosen["model"][0]) == 11
            assert chosen["fit_x"] == pytest.approx(p["x"]) and chosen["fit_y"] == pytest.approx(p["y"])

    job = _wait(client, client.post(f"{url}/shifts").json())
    assert job["status"] == "done", job
    assert job["result"]["shifts"]["stale"] is False
    m = _array(client.get(f"{url}/matrices"))
    assert m.shape == (2, 6, 6)
    np.testing.assert_allclose(m[0], -m[0].T, atol=1e-5)  # antisymmetric
    # Previews use the same FFTs as the job, so the numbers are the same.
    for (i, j), p in previews.items():
        assert m[0][i, j] == pytest.approx(p["dx"], abs=1e-5) and m[1][i, j] == pytest.approx(p["dy"], abs=1e-5)

    # Changing a setting marks the matrices stale.
    state = client.put(url, json={"mask": {"type": "hann", "n": 5}}).json()
    assert state["shifts"]["stale"] is True


def test_correlation_types_and_errors(stack):
    client, ds = stack
    url = f"/api/datasets/{ds['id']}/registration"
    client.post(url, json={})
    assert client.get(f"{url}/matrices").status_code == 422  # not computed yet
    for corr in ["cc", "mc", "pc"]:
        cc = _array(client.get(f"{url}/correlation", params={"i": 0, "j": 2, "corr": corr}))
        peak = np.unravel_index(np.argmax(cc), cc.shape)
        assert (peak[0] - 48, peak[1] - 48) == SHIFTS[2], corr
    bad = {"i": 0, "j": 1, "peaks": {"method": "gf", "window_radius": 60}}
    assert client.post(f"{url}/peak", json=bad).status_code == 422


def test_outlier_mask(stack):
    client, ds = stack
    url = f"/api/datasets/{ds['id']}/registration"
    client.post(url, json={})
    body = {"outliers": {"method": "transitivity", "threshold": 1}, "bad_frames": []}
    assert client.post(f"{url}/outlier-mask", json=body).status_code == 422  # no shifts yet
    client.put(url, json={"mask": {"type": "hann", "n": 4}})
    assert _wait(client, client.post(f"{url}/shifts").json())["status"] == "done"

    # Clean synthetic shifts: transitivity holds, nothing is masked.
    m = _array(client.post(f"{url}/outlier-mask", json=body))
    assert m.shape == (6, 6) and m.min() == 1
    # A bad frame masks its whole row and column.
    m = _array(client.post(f"{url}/outlier-mask", json={**body, "bad_frames": [2]}))
    assert m[2].sum() == 0 and m[:, 2].sum() == 0 and m[0, 1] == 1
    # Only pairs at most one frame apart; the diagonal is kept.
    m = _array(client.post(f"{url}/outlier-mask", json={"outliers": {"method": "none", "off_diagonal": 1}}))
    assert m[0, 1] == 1 and m[0, 2] == 0 and np.all(np.diag(m) == 1)
    # Neighbour check, and settings stored with the session.
    r = client.post(f"{url}/outlier-mask", json={"outliers": {"method": "neighbours", "max_shift": 0.5}})
    assert '"pairs": 36' in r.headers["X-Array-Meta"]
    assert client.put(url, json={"outliers": {"method": "neighbours", "max_shift": 2}}).json()["outliers"]["max_shift"] == 2


def test_outlier_mask_edits(stack):
    """Entries kept or masked by hand, on top of the outlier settings."""
    client, ds = stack
    url = f"/api/datasets/{ds['id']}/registration"
    client.post(url, json={})
    client.put(url, json={"mask": {"type": "hann", "n": 4}})
    assert _wait(client, client.post(f"{url}/shifts").json())["status"] == "done"
    mask_url = f"{url}/outlier-mask"

    # A masked pair is masked in both (i, j) and (j, i); nothing else changes.
    m = _array(client.post(mask_url, json={"outliers": {"method": "none"}, "edits": {"mask": [[0, 3]], "keep": []}}))
    assert m[0, 3] == 0 and m[3, 0] == 0
    assert m.sum() == 36 - 2
    # Either order of the pair is the same pair.
    m = _array(client.post(mask_url, json={"outliers": {"method": "none"}, "edits": {"mask": [[3, 0]]}}))
    assert m[0, 3] == 0 and m[3, 0] == 0 and m.sum() == 34

    # A kept pair overrides the off-diagonal limit.
    none1 = {"method": "none", "off_diagonal": 1}
    m = _array(client.post(mask_url, json={"outliers": none1, "edits": {"keep": [[0, 2]]}}))
    assert m[0, 2] == 1 and m[2, 0] == 1 and m[0, 3] == 0 and m[0, 1] == 1
    # A pair in both lists is masked.
    m = _array(client.post(mask_url, json={"outliers": none1, "edits": {"keep": [[0, 2]], "mask": [[0, 2]]}}))
    assert m[0, 2] == 0 and m[2, 0] == 0

    # Bad frames win over kept pairs; the diagonal of good frames stays.
    body = {"outliers": none1, "bad_frames": [2], "edits": {"keep": [[0, 2], [2, 4], [0, 4]]}}
    m = _array(client.post(mask_url, json=body))
    assert m[2].sum() == 0 and m[:, 2].sum() == 0 and m[0, 4] == 1 and m[4, 0] == 1
    assert np.all(np.diag(m)[[0, 1, 3, 4, 5]] == 1)
    # Masking a diagonal entry is ignored.
    m = _array(client.post(mask_url, json={"outliers": {"method": "none"}, "edits": {"mask": [[1, 1], [2, 4]]}}))
    assert np.all(np.diag(m) == 1) and m[2, 4] == 0 and m.sum() == 34

    # Frames outside the registered ones, and pairs on the diagonal, are ignored.
    edits = {"keep": [[0, 9], [-1, 2], [3, 3]], "mask": [[7, 8], [5, 5]]}
    m = _array(client.post(mask_url, json={"outliers": none1, "edits": edits}))
    expected = _array(client.post(mask_url, json={"outliers": none1}))
    np.testing.assert_array_equal(m, expected)

    # Stored with the session by PUT; pairs out of range are dropped.
    state = client.put(url, json={"edits": {"keep": [[0, 2], [0, 9]], "mask": [[1, 4]]}}).json()
    assert state["edits"] == {"keep": [[0, 2]], "mask": [[1, 4]]}
    assert client.get(url).json()["edits"] == state["edits"]
    # A request carries its own edits (empty here), as it does its outliers.
    m = _array(client.post(mask_url, json={"outliers": {"method": "none"}}))
    assert m.sum() == 36
    assert client.put(url, json={"mask": {"type": "hann", "n": 4}}).json()["edits"] == state["edits"]


def test_correction_with_edits(stack):
    client, ds = stack
    url = f"/api/datasets/{ds['id']}/registration"
    client.post(url, json={})
    client.put(url, json={"mask": {"type": "hann", "n": 4}})
    assert _wait(client, client.post(f"{url}/shifts").json())["status"] == "done"

    # Masking the pair (0, 5) as well: the drift is still the true one, as
    # the pair is filled from paths of good ones.
    edits = {"mask": [[5, 0]], "keep": [[1, 4]]}
    body = {"outliers": {"method": "none", "off_diagonal": 2}, "bad_frames": [3], "edits": edits}
    drift = client.post(f"{url}/correct", json=body).json()
    assert drift["frames"] == [0, 1, 2, 4, 5]
    true = np.array([SHIFTS[f] for f in drift["frames"]], float)  # (dy, dx)
    rel = lambda v: np.array(v) - np.mean(v)  # noqa: E731
    np.testing.assert_allclose(rel(drift["x"]), rel(true[:, 1]), atol=1e-6)
    np.testing.assert_allclose(rel(drift["y"]), rel(true[:, 0]), atol=1e-6)
    assert client.get(url).json()["edits"] == {"mask": [[5, 0]], "keep": [[1, 4]]}
    c = _array(client.get(f"{url}/corrected"))
    assert c[2][1, 4] == 1 and c[0][1, 4] == pytest.approx(SHIFTS[4][1] - SHIFTS[1][1])

    # Applied, the edits are stored with the alignment's settings.
    r = client.post(f"{url}/apply").json()
    assert r["dataset"]["alignment"]["settings"]["edits"] == edits
    assert r["dataset"]["alignment"]["settings"]["bad_frames"] == [3]


def test_correction_and_drift(stack):
    client, ds = stack
    url = f"/api/datasets/{ds['id']}/registration"
    client.post(url, json={})
    client.put(url, json={"mask": {"type": "hann", "n": 4}})
    assert client.post(f"{url}/correct", json={}).status_code == 422  # no shifts yet
    assert _wait(client, client.post(f"{url}/shifts").json())["status"] == "done"

    # Keep only pairs up to two frames apart, with frame 3 bad: correction
    # fills the rest from paths of good pairs, and the drift is the true one.
    body = {"outliers": {"method": "none", "off_diagonal": 2}, "bad_frames": [3]}
    drift = client.post(f"{url}/correct", json=body).json()
    assert drift["frames"] == [0, 1, 2, 4, 5]
    true = np.array([SHIFTS[f] for f in drift["frames"]], float)  # (dy, dx)
    rel = lambda v: np.array(v) - np.mean(v)  # noqa: E731
    np.testing.assert_allclose(rel(drift["x"]), rel(true[:, 1]), atol=1e-6)
    np.testing.assert_allclose(rel(drift["y"]), rel(true[:, 0]), atol=1e-6)

    c = _array(client.get(f"{url}/corrected"))
    assert c.shape == (3, 6, 6)
    assert c[2][3].sum() == 0 and c[2][0, 5] == 1  # bad frame unknown, far pair filled
    assert c[0][0, 5] == pytest.approx(SHIFTS[5][1] - SHIFTS[0][1])

    # Only neighbouring pairs, frame 3 bad: frames 0–2 and 4–5 share no good
    # pair, so they can't be related. An error, not an endless loop.
    r = client.post(f"{url}/correct", json={"outliers": {"method": "none", "off_diagonal": 1}, "bad_frames": [3]})
    assert r.status_code == 422 and "too few good pairs" in r.json()["detail"]


def test_result(stack):
    client, ds = stack
    url = f"/api/datasets/{ds['id']}/registration"
    client.post(url, json={})
    client.put(url, json={"mask": {"type": "hann", "n": 4}})
    assert _wait(client, client.post(f"{url}/shifts").json())["status"] == "done"
    assert client.get(f"{url}/result").status_code == 422  # not corrected yet
    client.post(f"{url}/correct", json={"outliers": {"method": "none", "off_diagonal": 2}, "bad_frames": [3]})

    # Bad frame 3 is left out; the frames' (dy, dx) shifts span 7 and 8 px,
    # relative to their mean (not whole pixels), and crop_image rounds
    # inwards at both ends: the area inside every frame is 87 × 88.
    r = client.get(f"{url}/result").json()
    assert r["frames"] == [0, 1, 2, 4, 5]
    assert r["rect"][2:] == [87, 88] and r["square"][2:] == [87, 87]
    assert client.get(f"{url}/aligned", params={"i": 3}).status_code == 422

    # Aligned, the frames agree to within their noise (σ 30), and the
    # average is their mean.
    frames = [_array(client.get(f"{url}/aligned", params={"i": f})) for f in r["frames"]]
    assert frames[0].shape == (87, 87)
    for f in frames[1:]:
        assert np.abs(f - frames[0]).mean() < 60
    np.testing.assert_allclose(_array(client.get(f"{url}/average")), np.mean(frames, axis=0), rtol=1e-5)

    # The square sits at square + drift on each frame as loaded.
    sx, sy, side, _ = r["square"]
    d = r["drift"]
    for k, f in enumerate(r["frames"]):
        x, y = round(sx + d["x"][k]), round(sy + d["y"][k])
        raw = _array(client.get(f"{url}/frame", params={"i": f}))[y:y + side, x:x + side]
        assert np.corrcoef(raw.ravel(), frames[k].ravel())[0, 1] > 0.95
    assert _array(client.get(f"{url}/average", params={"crop": False})).shape == (96, 96)


def test_apply(stack):
    client, ds = stack
    url = f"/api/datasets/{ds['id']}/registration"
    client.post(url, json={})
    client.put(url, json={"mask": {"type": "hann", "n": 4}})
    assert client.post(f"{url}/apply").status_code == 422  # nothing computed yet
    assert _wait(client, client.post(f"{url}/shifts").json())["status"] == "done"
    client.post(f"{url}/correct", json={"outliers": {"method": "none", "off_diagonal": 2}, "bad_frames": [3]})
    aligned = [_array(client.get(f"{url}/aligned", params={"i": f})) for f in (0, 1, 2, 4, 5)]
    average = _array(client.get(f"{url}/average"))

    # Applied: the stack's frames are the aligned ones (bad frame 3 left
    # out, cropped), and the average is a new image next to it.
    r = client.post(f"{url}/apply").json()
    stack_ = r["dataset"]
    assert stack_["navigation_shape"] == [5] and stack_["signal_shape"] == [87, 87]
    assert stack_["alignment"]["active"] and stack_["alignment"]["frames"] == [0, 1, 2, 4, 5]
    assert stack_["alignment"]["loaded_frames"] == 6
    names = [d["title"] for d in r["file"]["datasets"]]
    assert names == ["drift", "drift (aligned average)"]
    assert r["average"]["signal_shape"] == [87, 87] and r["average"]["derived_from"] == ds["id"]
    frame = lambda **q: _array(client.get(f"/api/datasets/{ds['id']}/frame", params=q))  # noqa: E731
    np.testing.assert_allclose(frame(index=3), aligned[3])
    np.testing.assert_allclose(frame(projection="mean"), average, rtol=1e-5)
    # The image axes start at the square's corner.
    x0 = r["dataset"]["axes"][2]["offset"] - ds["axes"][2]["offset"]
    assert x0 == pytest.approx(client.get(f"{url}/result").json()["square"][0] * ds["axes"][2]["scale"])
    # The registration carries on from the loaded frames.
    assert client.get(url).status_code == 200

    # The switch: frames as loaded, then aligned again.
    off = client.put(f"/api/datasets/{ds['id']}/alignment", json={"active": False}).json()["dataset"]
    assert off["navigation_shape"] == [6] and off["signal_shape"] == [96, 96] and not off["alignment"]["active"]
    assert off["filter_rev"] > stack_["filter_rev"]
    np.testing.assert_allclose(frame(index=3), _drifting_stack(SHIFTS)[3])
    on = client.put(f"/api/datasets/{ds['id']}/alignment", json={"active": True}).json()["dataset"]
    assert on["navigation_shape"] == [5]

    # Applying again replaces the average; removing restores the stack.
    assert len(client.post(f"{url}/apply").json()["file"]["datasets"]) == 2
    gone = client.delete(f"/api/datasets/{ds['id']}/alignment").json()
    assert gone["dataset"]["alignment"] is None and gone["dataset"]["navigation_shape"] == [6]
    assert [d["title"] for d in gone["file"]["datasets"]] == ["drift"]


def _csv_rows(r):
    import csv
    import io

    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/csv")
    text = r.text
    comments = [line for line in text.splitlines() if line.startswith("# ")]
    rows = list(csv.DictReader(io.StringIO("\n".join(line for line in text.splitlines() if not line.startswith("#")))))
    return comments, rows


def test_export_shifts_and_report(stack):
    import re

    client, ds = stack
    url = f"/api/datasets/{ds['id']}/registration"
    client.post(url, json={})
    client.put(url, json={"mask": {"type": "hann", "n": 4}})
    assert _wait(client, client.post(f"{url}/shifts").json())["status"] == "done"
    assert client.get(f"{url}/shifts.csv").status_code == 422  # not corrected yet
    assert client.get(f"{url}/report.pdf").status_code == 422
    assert client.get(f"/api/datasets/{ds['id']}/alignment/shifts.csv").status_code == 404  # nothing applied

    body = {"outliers": {"method": "none", "off_diagonal": 2}, "bad_frames": [3]}
    drift = client.post(f"{url}/correct", json=body).json()

    r = client.get(f"{url}/shifts.csv")
    assert 'filename="drift_shifts.csv"' in r.headers["content-disposition"]
    comments, rows = _csv_rows(r)
    assert [int(row["frame"]) for row in rows] == [0, 1, 2, 4, 5] == drift["frames"]
    for key, values in (("drift_x_px", drift["x"]), ("drift_y_px", drift["y"])):
        np.testing.assert_allclose([float(row[key]) for row in rows], values, atol=1e-4)
    for key, values in (("shift_x_px", drift["x"]), ("shift_y_px", drift["y"])):
        np.testing.assert_allclose([float(row[key]) for row in rows], -np.array(values), atol=1e-4)
    assert "shift_x_A" not in rows[0]  # the test stack has no length units
    text = "\n".join(comments)
    assert "Frames left out: 3" in text and "Frames registered: 0 to 5" in text
    assert "hann" in text and "only pairs up to 2 frames apart" in text and "Crop (square)" in text

    r = client.get(f"{url}/report.pdf")
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    assert r.headers["content-disposition"].startswith('attachment; filename="drift_alignment_report.pdf"')
    assert r.content.startswith(b"%PDF")
    assert len(re.findall(rb"/Type\s*/Page[^s]", r.content)) == 2

    # Applied, the same rows come from the stack itself.
    assert client.post(f"{url}/apply").status_code == 200
    _, applied = _csv_rows(client.get(f"/api/datasets/{ds['id']}/alignment/shifts.csv"))
    assert applied == rows


def test_export_calibrated_columns():
    from si_viewer.processing.registration_export import angstrom_per_pixel, shifts_csv

    info = {
        "frames": [0, 2], "first": 0, "last": 3, "square": [1, 2, 50, 50],
        "drift": {"frames": [0, 2], "x": [0.5, -0.5], "y": [-0.25, 0.25]},
        "settings": {
            "mask": {"type": "elliptical", "n": 12, "n2": 9.1, "theta": 81.4},
            "peaks": {"correlation": "cc", "method": "gf", "num_peaks": 3, "sigma_guess": 3, "window_radius": 4},
            "outliers": {"method": "transitivity", "threshold": 10, "max_paths": 5, "max_shift": 5, "off_diagonal": None},
            "bad_frames": [1],
        },  # no "edits": data stored before they existed
    }
    out = shifts_csv(info, "Nb3Cl8", 0.02, "nm")
    assert "n1 = 12, n2 = 9.1, θ = 81.4°" in out and "Frames left out: 1" in out and "Manual edits" not in out
    rows = [line.split(",") for line in out.splitlines() if not line.startswith("#")]
    assert rows[0][-4:] == ["shift_x_A", "shift_y_A", "drift_x_A", "drift_y_A"]
    assert rows[1] == ["0", "-0.5", "0.25", "0.5", "-0.25", "-0.1", "0.05", "0.1", "-0.05"]
    assert angstrom_per_pixel(1, "µm") == angstrom_per_pixel(1, "μm") == angstrom_per_pixel(1, "um") == 1e4
    assert angstrom_per_pixel(1, "Å") == angstrom_per_pixel(1, "A") == angstrom_per_pixel(1, "Angstrom") == 1.0
    assert angstrom_per_pixel(1, "pm") == 0.01 and angstrom_per_pixel(1, "m") == 1e10
    assert angstrom_per_pixel(1, "") is None and angstrom_per_pixel(1, "<undefined>") is None
    assert angstrom_per_pixel(None, "nm") is None and angstrom_per_pixel(1, "1/nm") is None


def test_elliptical_mask_is_the_packages(stack):
    """The elliptical Gaussian mask (makeFourierMask_eg) with theta in degrees."""
    import rigidregistration

    from si_viewer.processing.registration import MaskSettings, RegistrationSession, make_mask

    frames = _drifting_stack(SHIFTS)
    s = RegistrationSession(frames)
    mask = MaskSettings(type="elliptical", n=12, n2=9.106, theta=81.36)
    ops = s._ops(mask, s.peaks)
    ref = rigidregistration.stackregistration.imstack(np.moveaxis(frames.astype(float), 0, -1))
    ref.makeFourierMask_eg(n1=12, n2=9.106, theta=np.radians(81.36))
    np.testing.assert_allclose(ops.mask_fourierspace, ref.mask_fourierspace)
    make_mask(ref, MaskSettings(type="hann", n=4))
    assert ref.mask_params["masktype"] == "hann"

    client, ds = stack
    url = f"/api/datasets/{ds['id']}/registration"
    client.post(url, json={})
    params = {"i": 0, "j": 3, "type": "elliptical", "n": 4, "n2": 3, "theta": 30}
    cc = _array(client.get(f"{url}/correlation", params=params))
    peak = np.unravel_index(np.argmax(cc), cc.shape)
    assert (peak[0] - 48, peak[1] - 48) == SHIFTS[3]
    ccs = _array(client.get(f"{url}/compare-n", params={**params, "n": [3, 6], "n1": 4}))
    assert ccs.shape == (2, 96, 96)


NB3CL8 = "../../../rigidRegistration/samplefiles/sample_data/Nb3Cl8_cross_section.tif"


def test_matches_the_detailed_walkthrough():
    """The DetailedWalkThrough notebook's Nb3Cl8 registration (elliptical
    mask, Gaussian fit, transitivity outliers, correction), run with the
    package directly, gives the same matrices, mask and drift as the app."""
    import os

    import rigidregistration

    from si_viewer.processing.registration import (
        MaskEdits, MaskSettings, OutlierSettings, PeakSettings, RegistrationSession,
    )

    path = os.path.join(os.path.dirname(__file__), NB3CL8)
    if not os.path.exists(path):
        pytest.skip("rigidRegistration's sample data not found next to this repository")
    raw = tifffile.imread(path)

    ref = rigidregistration.stackregistration.imstack(np.rollaxis(raw, 0, 3) / float(2**16))
    ref.getFFTs()
    ref.makeFourierMask_eg(n1=12, n2=9.106, theta=1.42)
    ref.setGaussianFitParams(num_peaks=3, sigma_guess=3, window_radius=4)
    ref.findImageShifts(findMaxima="gf")
    ref.get_outliers(10, 10)
    ref.refine_mask()
    ref.make_corrected_Rij()
    ref.get_imshifts()

    s = RegistrationSession(raw)
    s.set_settings(
        mask=MaskSettings(type="elliptical", n=12, n2=9.106, theta=float(np.degrees(1.42))),
        peaks=PeakSettings(method="gf", num_peaks=3, sigma_guess=3, window_radius=4),
    )
    s.compute_shifts(lambda f: None)
    m = s.matrices()
    np.testing.assert_allclose(m[0], ref.Y_ij, atol=1e-4)
    np.testing.assert_allclose(m[1], ref.X_ij, atol=1e-4)
    outliers = OutlierSettings(threshold=10, max_paths=10)
    np.testing.assert_array_equal(s.outlier_mask(outliers), ref.Rij_mask)
    drift = s.correct(outliers)
    np.testing.assert_allclose(drift["x"], -ref.shifts_y, atol=1e-4)
    np.testing.assert_allclose(drift["y"], -ref.shifts_x, atol=1e-4)

    # The result: the package's average and crop (its data is normalised).
    ref.get_averaged_image(get_shifts=False, correct_Rij=False)
    ref.crop_image()
    r = s.result_summary()
    assert r["frames"] == list(range(raw.shape[0]))
    np.testing.assert_allclose(s.average(crop=False) / 2**16, ref.average_image, rtol=1e-4, atol=1e-6)
    x, y, w, h = r["rect"]
    np.testing.assert_allclose(s.average(crop=False)[y:y + h, x:x + w] / 2**16, ref.cropped_image, rtol=1e-4, atol=1e-6)
    # The square: as large as the rectangle allows, centred in it.
    sx, sy, side, _ = r["square"]
    assert side == min(w, h) and x <= sx and sx + side <= x + w and y <= sy and sy + side <= y + h
    assert s.average().shape == (side, side) and s.aligned_frame(5).shape == (side, side)

    # The notebook's hand edits after get_outliers (cell 24): single pairs
    # kept or masked, and blocks of masked pairs.
    ref.get_outliers(10, 10)
    for i, j in [[11, 12], [11, 13]]:
        ref.Rij_mask[i, j] = 1
        ref.Rij_mask[j, i] = 1
    for i, j in [[1, 9]]:
        ref.Rij_mask[i, j] = 0
        ref.Rij_mask[j, i] = 0
    ref.Rij_mask[:5, 15] = 0
    ref.Rij_mask[:3, 18] = 0
    ref.Rij_mask[:8, 24:26] = 0
    ref.Rij_mask[15, :5] = 0
    ref.Rij_mask[18, :3] = 0
    ref.Rij_mask[24:26, :8] = 0
    ref.refine_mask()
    ref.make_corrected_Rij()
    ref.get_imshifts()

    # The same as pairs (the blocks' mirror images are implied).
    edits = MaskEdits(
        keep=[(11, 12), (11, 13)],
        mask=[(1, 9)] + [(r, 15) for r in range(5)] + [(r, 18) for r in range(3)]
        + [(r, c) for r in range(8) for c in (24, 25)],
    )
    np.testing.assert_array_equal(s.outlier_mask(outliers, edits=edits), ref.Rij_mask)
    drift = s.correct(outliers, edits=edits)
    np.testing.assert_allclose(drift["x"], -ref.shifts_y, atol=1e-4)
    np.testing.assert_allclose(drift["y"], -ref.shifts_x, atol=1e-4)
    assert s.correction["edits"] == edits.model_dump()


def test_apply_as_a_new_stack(stack):
    client, ds = stack
    url = f"/api/datasets/{ds['id']}/registration"
    client.post(url, json={})
    client.put(url, json={"mask": {"type": "hann", "n": 4}})
    assert _wait(client, client.post(f"{url}/shifts").json())["status"] == "done"
    client.post(f"{url}/correct", json={"outliers": {"method": "none", "off_diagonal": 2}, "bad_frames": [3]})
    aligned3 = _array(client.get(f"{url}/aligned", params={"i": 4}))

    r = client.post(f"{url}/apply", json={"in_place": False}).json()
    # The stack is as loaded; the aligned frames are a new stack under it, the average under that.
    assert r["dataset"]["alignment"] is None and r["dataset"]["navigation_shape"] == [6]
    datasets = {d["id"]: d for d in r["file"]["datasets"]}
    new = datasets[r["aligned_id"]]
    assert new["origin"] == "aligned" and new["parent_id"] == ds["id"] and new["kind"] == "stack"
    assert new["navigation_shape"] == [5] and new["title"] == "drift (aligned)"
    assert r["average"]["parent_id"] == new["id"]
    np.testing.assert_allclose(_array(client.get(f"/api/datasets/{new['id']}/frame", params={"index": 3})), aligned3)
    session = client.app.state.session
    assert session.get_dataset(new["id"]).ops[-1]["op"] == "align"
    assert session.get_dataset(ds["id"]).ops == []
    # Deleting it takes its average with it.
    assert client.delete(f"/api/datasets/{new['id']}").status_code == 200
    assert [session.datasets[i].title for i in session.files[ds["file_id"]].dataset_ids] == ["drift"]
