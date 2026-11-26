"""Strain from Bragg vectors, step 1: the Bragg vector map's peaks and the
lattice (O, g1, g2) proposed from them."""

import numpy as np

from si_viewer.processing import strain

N = 64
O = (31.6, 32.3)
G1 = (9.2, 1.1)
G2 = (-2.3, 10.4)


def _table(ny=6, nx=7, jitter=0.15, seed=0):
    """Disks at every lattice point O + h g1 + k g2 on the pattern, at every
    scan position, a little scattered (rows: scan y, x, ky, kx, score)."""
    rng = np.random.default_rng(seed)
    rows = []
    for iy in range(ny):
        for ix in range(nx):
            for h in range(-4, 5):
                for k in range(-4, 5):
                    x = O[0] + h * G1[0] + k * G2[0] + rng.normal(0, jitter)
                    y = O[1] + h * G1[1] + k * G2[1] + rng.normal(0, jitter)
                    if 1 < x < N - 1 and 1 < y < N - 1:
                        rows.append((iy, ix, y, x, 1.0 if (h, k) != (0, 0) else 5.0))
    return np.array(rows, np.float32)


def test_peaks_on_the_lattice_points():
    peaks = strain.bvm_peaks(_table(), (N, N), strain.BvmPeakOptions(sigma=0.5, threshold=0.05, min_spacing=4))
    # The direct beam among them, where it is.
    assert np.min(np.hypot(peaks[:, 0] - O[0], peaks[:, 1] - O[1])) < 0.15
    # Every peak within a fraction of a pixel of a lattice point.
    m = np.array([[G1[0], G2[0]], [G1[1], G2[1]]])
    hk = np.linalg.solve(m, (peaks[:, :2] - O).T).T
    near = np.round(hk) @ m.T + O
    assert np.max(np.hypot(*(peaks[:, :2] - near).T)) < 0.3
    assert len(peaks) >= 25


def test_lattice_proposed_from_the_peaks():
    peaks = strain.bvm_peaks(_table(), (N, N), strain.BvmPeakOptions(sigma=0.5, threshold=0.05, min_spacing=4))
    # O: the peak nearest the direct beam's guess (the map counts disks, so not the brightest).
    lat = strain.guess_lattice(peaks, (33.0, 31.0))
    assert lat is not None
    np.testing.assert_allclose((lat["ox"], lat["oy"]), O, atol=0.15)
    g1, g2 = np.array([lat["g1x"], lat["g1y"]]), np.array([lat["g2x"], lat["g2y"]])
    # The shortest basis, one of ±G1 and ±G2 each, g2 counter-clockwise on screen (y down).
    for g in (g1, g2):
        assert min(np.hypot(*(g - s * np.array(v))) for v in (G1, G2) for s in (1, -1)) < 0.3
    assert g1[0] * g2[1] - g1[1] * g2[0] < 0


def test_too_few_peaks():
    assert strain.guess_lattice(np.array([[1.0, 1.0, 5.0], [5.0, 1.0, 1.0]]), (1, 1)) is None
    # All on one line.
    line = np.array([[0.0, 0.0, 5.0], [5.0, 0.0, 1.0], [10.0, 0.0, 1.0], [-5.0, 0.0, 1.0]])
    assert strain.guess_lattice(line, (0, 0)) is None
    empty = strain.bvm_peaks(np.zeros((0, 5), np.float32), (N, N), strain.BvmPeakOptions())
    assert empty.shape == (0, 3)


# ---------- step 2: indexing and the lattice fit ----------

NY, NX = 4, 5


def _strained(iy, ix):
    """The lattice at a scan position: O drifting (descan), g1 and g2
    stretched along x by up to 3 % across the scan and turned a little."""
    exx = 0.03 * ix / (NX - 1)
    rot = 0.01 * iy
    f = np.array([[1 + exx, -rot], [rot, 1.0]])
    o = np.array(O) + (0.4 * ix, -0.3 * iy)
    return o, f @ np.array(G1), f @ np.array(G2)


def _strained_table(jitter=0.05, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for iy in range(NY):
        for ix in range(NX):
            o, g1, g2 = _strained(iy, ix)
            for h in range(-4, 5):
                for k in range(-4, 5):
                    x, y = o + h * g1 + k * g2 + rng.normal(0, jitter, 2)
                    if 1 < x < N - 1 and 1 < y < N - 1:
                        rows.append((iy, ix, y, x, 1.0))
            # A stray disk off the lattice: not indexed.
            rows.append((iy, ix, o[1] + 0.5 * g1[1] + 0.5 * g2[1], o[0] + 0.5 * g1[0] + 0.5 * g2[0], 1.0))
    return np.array(rows, np.float32)


def _picked():
    return strain.StrainLattice(ox=O[0], oy=O[1], g1x=G1[0], g1y=G1[1], g2x=G2[0], g2y=G2[1])


def test_fit_recovers_each_positions_lattice():
    table = _strained_table()
    # The tolerance covers the drift and strain (outer disks move ~2 px), under half a g.
    res = strain.fit_lattice(table, (NY, NX), _picked(), strain.FitOptions(tolerance=4), np.eye(2))
    fit = res["fit"]
    assert fit.shape == (6, NY, NX) and np.isfinite(fit).all()
    for iy in range(NY):
        for ix in range(NX):
            o, g1, g2 = _strained(iy, ix)
            np.testing.assert_allclose(fit[0:2, iy, ix], g1, atol=0.03)
            np.testing.assert_allclose(fit[2:4, iy, ix], g2, atol=0.03)
            # O' is the origin's offset from the picked O.
            np.testing.assert_allclose(fit[4:6, iy, ix], o - O, atol=0.05)
    # Every disk but the stray one indexed; the residual about the jitter.
    per_position = np.bincount((table[:, 0] * NX + table[:, 1]).astype(int), minlength=NY * NX).reshape(NY, NX)
    np.testing.assert_array_equal(res["count"], per_position - 1)
    assert np.nanmax(res["residual"]) < 0.15


def test_fit_needs_enough_disks_off_one_line():
    o, g1, g2 = _strained(0, 0)
    # Position (0, 0): three disks along g1 only; (0, 1): two disks.
    rows = [(0, 0, o[1] + h * g1[1], o[0] + h * g1[0], 1.0) for h in (-1, 0, 1, 2)]
    rows += [(0, 1, o[1] + h * g2[1], o[0] + h * g2[0], 1.0) for h in (0, 1)]
    res = strain.fit_lattice(np.array(rows, np.float32), (1, 2), _picked(), strain.FitOptions(min_disks=3), np.eye(2))
    assert np.isnan(res["fit"]).all() and np.isnan(res["residual"]).all()
    np.testing.assert_array_equal(res["count"], [[4, 2]])


def test_fit_in_the_rotated_frame():
    table = _strained_table()
    rot = {"theta": 90.0, "mirror": False}
    frame = strain.frame_matrix(None, rot)
    res = strain.fit_lattice(table, (NY, NX), _picked(), strain.FitOptions(tolerance=4), frame)
    _, g1, _ = _strained(2, 3)
    # As processing.dpc.rotate: (x, y) → (−y, x).
    np.testing.assert_allclose(res["fit"][0:2, 2, 3], (-g1[1], g1[0]), atol=0.03)
    # A mirror flips y first.
    m = strain.frame_matrix(None, {"theta": 0.0, "mirror": True})
    np.testing.assert_allclose(m, np.diag([1.0, -1.0]))
    # The ellipse made circular: its a axis shrunk, b grown, area kept.
    e = strain.frame_matrix({"a": 11.0, "b": 9.0, "angle": 0.0}, None)
    np.testing.assert_allclose(np.diag(e), (np.sqrt(9 / 11), np.sqrt(11 / 9)))


def test_fit_through_the_api(tmp_path):
    """Patterns with disks on the strained lattice: disks found, the lattice
    fitted at every position, previewed at one, kept with the analysis."""
    import time

    from fastapi.testclient import TestClient
    from test_4dstem import _write_emdfile

    from si_viewer.app import create_app
    from si_viewer.project import open_project, save_project
    from si_viewer.session import Session

    ss = 4
    yy, xx = (np.mgrid[0 : N * ss, 0 : N * ss] + 0.5) / ss
    cube = np.zeros((NY, NX, N, N), np.float32)
    for iy in range(NY):
        for ix in range(NX):
            o, g1, g2 = _strained(iy, ix)
            fine = np.zeros_like(xx)
            for h in range(-4, 5):
                for k in range(-4, 5):
                    c = o + h * g1 + k * g2
                    fine += (60.0 if (h, k) == (0, 0) else 30.0) * (np.hypot(xx - c[0], yy - c[1]) <= 2.0)
            cube[iy, ix] = fine.reshape(N, ss, N, ss).mean(axis=(1, 3)) + 1
    path = tmp_path / "lattice.h5"
    _write_emdfile(path, cube, extra_2d=False)
    session = Session()
    client = TestClient(create_app(session))
    cube_id = client.post("/api/files", json={"path": str(path)}).json()["datasets"][0]["id"]
    url = f"/api/datasets/{cube_id}"

    def wait(job):
        for _ in range(600):
            job = client.get(f"/api/jobs/{job['id']}").json()
            if job["status"] not in ("queued", "running"):
                return job
            time.sleep(0.05)
        raise AssertionError("job did not finish")

    job = wait(client.post(f"{url}/4d/disks/run", json={"options": {"threshold": 0.3, "min_spacing": 3, "edge": 1}}).json())
    assert job["status"] == "done", job["error"]
    # Step 1: the lattice proposed from the map's peaks (the strain smears them a little).
    peaks = client.post(f"{url}/4d/strain/bvm-peaks", json={"options": {"sigma": 1, "min_spacing": 4}}).json()
    lat = peaks["lattice"]
    assert lat is not None and abs(lat["ox"] - O[0]) < 1.5 and abs(lat["oy"] - O[1]) < 1.5
    # Step 2: the fit (job), its maps under the Bragg vectors.
    body = {"lattice": _picked().model_dump(), "options": {"tolerance": 4}}
    job = wait(client.post(f"{url}/4d/strain/fit", json=body).json())
    assert job["status"] == "done", job["error"]
    r = job["result"]
    assert r["fitted"] == NY * NX and r["indexed"] <= r["disks"]
    made = {d["title"]: d for d in r["file"]["datasets"] if d["origin"] == "strain-fit"}
    assert sorted(made) == ["Fit residual", "Indexed disks", "Lattice fit"]
    vectors = next(d for d in r["file"]["datasets"] if d["origin"] == "disks")
    assert made["Lattice fit"]["parent_id"] == vectors["id"]
    assert made["Indexed disks"]["parent_id"] == made["Lattice fit"]["id"]
    assert made["Lattice fit"]["kind"] == "stack" and made["Lattice fit"]["strain"]["lattice"] == body["lattice"]
    fit = np.asarray(session.get_dataset(made["Lattice fit"]["id"]).signal.data)
    for iy, ix in ((0, 0), (3, 4)):
        _, g1, g2 = _strained(iy, ix)
        np.testing.assert_allclose(fit[0:2, iy, ix], g1, atol=0.1)
        np.testing.assert_allclose(fit[2:4, iy, ix], g2, atol=0.1)
    # Made again in place.
    again = wait(client.post(f"{url}/4d/strain/fit", json=body).json())["result"]
    assert sorted(again["ids"]) == sorted(d["id"] for d in made.values())

    # The preview at a position: its disks indexed, the fitted lattice on its own pattern.
    pv = client.post(f"{url}/4d/strain/preview", json={**body, "positions": [[4, 3], [99, 0]]}).json()
    assert list(pv) == ["4,3"]
    e = pv["4,3"]
    o, g1, g2 = _strained(3, 4)
    assert e["count"] == sum(d[4] for d in e["disks"]) and e["residual"] < 0.3
    p00 = next(p for p in e["points"] if p[2:] == [0, 0])
    p10 = next(p for p in e["points"] if p[2:] == [1, 0])
    np.testing.assert_allclose(p00[:2], o, atol=0.15)
    np.testing.assert_allclose(np.subtract(p10[:2], p00[:2]), g1, atol=0.15)

    # Kept with the analysis.
    saved = save_project(session, tmp_path / "p.hspy")
    other = Session()
    open_project(other, saved)
    back = other.get_dataset(made["Lattice fit"]["id"])
    assert back.strain["output"] == "lattice" and back.strain["components"][0] == "g1 x"
    np.testing.assert_allclose(np.asarray(back.signal.data), fit, equal_nan=True)
