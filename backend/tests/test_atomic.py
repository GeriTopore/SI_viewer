"""M22a: kemstem's example images and FFT peaks, checked against kemstem."""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.ndimage import gaussian_filter

from si_viewer.app import create_app
from si_viewer.examples import EXAMPLES, kemstem_data
from si_viewer.processing import atomic

pytestmark = pytest.mark.skipif(kemstem_data() is None, reason="kemstem's clone (example data) is missing")

SITES = "rI_2138 B1_RT 5.3 Mx_0_HAADF.tif"


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


@pytest.fixture(scope="module")
def crop():
    """A 256 × 256 crop of the double perovskite (fast tests)."""
    import hyperspy.api as hs

    return hs.load(kemstem_data() / SITES).data[300:556, 300:556].astype(np.float64)


def _strongest_peak(image):
    """(y, x) pixel of the brightest non-central spot of the log FFT."""
    pattern = np.log1p(np.abs(atomic.fourier_pattern(image)))
    h, w = pattern.shape
    yy, xx = np.mgrid[:h, :w]
    pattern[np.hypot(yy - h // 2, xx - w // 2) < 8] = 0
    return np.array(np.unravel_index(np.argmax(pattern), pattern.shape), dtype=float)


def test_examples_listed_and_opened(client):
    ids = {e.id for e in EXAMPLES}
    assert {"kemstem-strain", "kemstem-sites", "kemstem-pld"} <= ids
    entry = client.post("/api/examples/kemstem-pld").json()
    [d] = entry["datasets"]
    assert d["kind"] == "image"
    assert [a["size"] for a in d["axes"]] == [898, 963]
    assert all(a["units"] == "nm" and a["scale"] == pytest.approx(0.0184) for a in d["axes"])


def test_frequency_round_trip():
    for shape in [(256, 256), (255, 300)]:
        p = np.array([[10.3, 200.7], [128, 128], [0, 299]])
        assert np.allclose(atomic.to_pixels(atomic.to_freqs(p, shape), shape), p)
    # The centre is frequency 0; one pixel right is 1/W cycles per pixel.
    assert np.allclose(atomic.to_freqs([[128, 129]], (256, 256)), [[0, 1 / 256]])


def test_gaussian_refinement_matches_kemstem(crop):
    import kemstem

    peak = _strongest_peak(crop)
    start = peak + [1.4, -1.2]  # a click a little off the spot
    opts = atomic.PeakOptions(method="gauss", window=9, blur=1.0)
    [got] = atomic.refine_peaks(crop, atomic.to_freqs(start, crop.shape), opts)
    assert not got["failed"]

    # The notebooks' calls on the same pattern.
    pattern = kemstem.prepare_fourier_pattern(kemstem.normalize(crop), log=True, log_offset=1e0)
    ref, errs, popt, _ = kemstem.refine_peaks_gf(gaussian_filter(pattern, 1), start[None], window_dimension=9,
                                                 remove_unfit=False, verbose=False)
    assert not errs[0]
    assert np.allclose(atomic.to_pixels([[got["fy"], got["fx"]]], crop.shape), ref, atol=1e-6)
    assert got["background"] == pytest.approx(popt[0, 6])
    # And it lands on the spot.
    assert np.hypot(*(ref[0] - peak)) < 1.0


def test_centre_of_mass_and_edges(crop):
    peak = _strongest_peak(crop)
    opts = atomic.PeakOptions(method="com", window=9, blur=1.0)
    near, edge = atomic.refine_peaks(crop, atomic.to_freqs([peak + [1, 1], [1, 1]], crop.shape), opts)
    assert not near["failed"] and "background" not in near
    assert np.hypot(*(atomic.to_pixels([[near["fy"], near["fx"]]], crop.shape)[0] - peak)) < 1.5
    # Too near the FFT's edge for the window: kept, marked failed.
    assert edge["failed"]
    with pytest.raises(ValueError):
        atomic.PeakOptions(window=8).check()


def test_endpoint(client):
    d = client.post("/api/examples/kemstem-sites").json()["datasets"][0]
    image = np.asarray(client.app.state.session.get_dataset(d["id"]).signal.data, dtype=np.float64)
    peak = _strongest_peak(image)
    f = atomic.to_freqs(peak + [1, -1], image.shape)[0]
    r = client.post(f"/api/datasets/{d['id']}/atomic/fft-peaks", json={"peaks": [list(f)], "window": 9})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["shape"] == [1017, 1017]
    [p] = body["peaks"]
    assert not p["failed"]
    assert np.hypot(*(atomic.to_pixels([[p["fy"], p["fx"]]], image.shape)[0] - peak)) < 1.0
    assert client.post(f"/api/datasets/{d['id']}/atomic/fft-peaks", json={"peaks": [list(f)], "window": 4}).status_code == 422
    assert client.post("/api/datasets/nope/atomic/fft-peaks", json={"peaks": []}).status_code == 404


# ---------- M22b: Fourier filtering, phase and strain ----------

STRAIN = "rI_1341 2.6 Mx B_C270_0_HAADF.tif"


@pytest.fixture(scope="module")
def film():
    """A 512 × 512 crop (even: kemstem's N/2 and the array's N//2 agree) of
    the relaxed film, across the interface."""
    import hyperspy.api as hs

    return hs.load(kemstem_data() / STRAIN).data[700:1212, 700:1212].astype(np.float64)


def _two_peaks(image):
    """The two strongest first-order spots in different directions (frequencies)."""
    pattern = np.log1p(np.abs(atomic.fourier_pattern(image)))
    h, w = pattern.shape
    yy, xx = np.mgrid[:h, :w]
    pattern[np.hypot(yy - h // 2, xx - w // 2) < 8] = 0
    first = np.array(np.unravel_index(np.argmax(pattern), pattern.shape), dtype=float)
    d = first - [h // 2, w // 2]
    # Mask the first spot's line (both signs), then take the next.
    along = np.abs((yy - h // 2) * d[1] - (xx - w // 2) * d[0]) / np.hypot(*d) < 6
    pattern[along] = 0
    second = np.array(np.unravel_index(np.argmax(pattern), pattern.shape), dtype=float)
    return atomic.to_freqs([first, second], (h, w))


def test_fourier_maps_match_kemstem(film):
    import kemstem

    f1, f2 = _two_peaks(film)
    opts = atomic.FourierOptions(sigma=12)
    made = atomic.fourier_maps(film, {1: tuple(f1), 2: tuple(f2)}, {"amplitude", "grating", "phase", "strain"},
                               opts, per_peak=[1], tensor=(1, 2))
    by = {m["output"]: m for m in made}
    assert set(by) == {"amplitude", "grating", "phase", "strain-par", "strain-perp", "exx", "eyy", "exy", "oxy"}
    assert by["exx"]["peaks"] == [1, 2] and by["phase"]["title"] == "Phase g1"

    # The notebook's calls, with peaks as FFT pixels.
    p1, p2 = atomic.to_pixels([f1, f2], film.shape)
    ft = kemstem.prepare_fourier_pattern(kemstem.normalize(film))
    fim1 = kemstem.fourier_filter(ft, p1, sigma=12)[0]
    fim2 = kemstem.fourier_filter(ft, p2, sigma=12)[0]
    ph1 = kemstem.phaselock(fim1, p1, sigma=24)
    ph2 = kemstem.phaselock(fim2, p2, sigma=24)
    par, perp = kemstem.phase_to_strain(ph1, p1)
    exx, eyy, exy, oxy = kemstem.calculate_strain_tensor(ph1, ph2, p1, p2)
    for key, ref in (("amplitude", np.abs(fim1)), ("grating", np.real(fim1)), ("strain-par", par),
                     ("strain-perp", perp), ("exx", exx), ("eyy", eyy), ("exy", exy), ("oxy", oxy)):
        assert np.allclose(by[key]["data"], ref, atol=1e-5 * max(1, np.abs(ref).max())), key
    # Phases agree modulo 2π.
    assert np.allclose(np.angle(np.exp(1j * (by["phase"]["data"] - ph1))), 0, atol=1e-4)
    # A film on its substrate: away from the edges, strains of about a percent.
    assert abs(np.median(exx[64:-64, 64:-64])) < 0.03 and abs(np.median(eyy[64:-64, 64:-64])) < 0.03


def test_gpa_and_checks(film):
    f1, f2 = _two_peaks(film)
    made = atomic.fourier_maps(film, {1: tuple(f1)}, {"phase"}, atomic.FourierOptions(sigma=12, phase="gpa"))
    [m] = made
    assert m["data"].min() >= 0 and m["data"].max() < 2 * np.pi + 1e-6
    with pytest.raises(ValueError):
        atomic.fourier_maps(film, {1: tuple(f1), 2: tuple(2 * f1)}, set(), atomic.FourierOptions(), [], (1, 2))
    with pytest.raises(ValueError):
        atomic.FourierOptions(sigma=0).check()
    assert atomic.coarsening((512, 512), 12) == pytest.approx(512 / (2 * np.pi * 12) * np.sqrt(np.log(4)))


def test_odd_sizes_measure_no_false_strain(film):
    """A peak exactly on an FFT pixel of an odd-sized, unstrained lattice:
    ε∥ ≈ 0 (kemstem's N/2 references, given the true frequency)."""
    n = 255
    yy, xx = np.mgrid[:n, :n]
    f = (0, 25 / n)  # 25 cycles across
    lattice = 1 + np.cos(2 * np.pi * f[1] * xx)
    [par] = [m for m in atomic.fourier_maps(lattice, {1: f}, {"strain"}, atomic.FourierOptions(sigma=6))
             if m["output"] == "strain-par"]
    assert abs(np.median(par["data"])) < 1e-3


def _wait(client, job, timeout=120):
    import time

    t0 = time.time()
    while job["status"] in ("queued", "running"):
        assert time.time() - t0 < timeout, "job did not finish"
        time.sleep(0.05)
        job = client.get(f"/api/jobs/{job['id']}").json()
    return job


def test_fourier_endpoint(client):
    d = client.post("/api/examples/kemstem-sites").json()["datasets"][0]
    image = np.asarray(client.app.state.session.get_dataset(d["id"]).signal.data, dtype=np.float64)
    f1, f2 = _two_peaks(image)
    body = {"peaks": {"1": list(f1), "2": list(f2)}, "per_peak": [1], "outputs": ["amplitude", "phase"],
            "tensor": [1, 2], "sigma": 10}
    job = _wait(client, client.post(f"/api/datasets/{d['id']}/atomic/fourier", json=body).json())
    assert job["status"] == "done", job
    ids = job["result"]["ids"]
    assert len(ids) == 6
    files = job["result"]["file"]["datasets"]
    made = [x for x in files if x["id"] in ids]
    assert {x["title"] for x in made} == {"Amplitude g1", "Phase g1", "εxx (g1, g2)", "εyy (g1, g2)",
                                           "εxy (g1, g2)", "ωxy (g1, g2)"}
    assert all(x["parent_id"] == d["id"] and x["origin"] == "atomic" and x["kind"] == "image" for x in made)
    assert made[0]["atomic"]["sigma"] == 10 and made[0]["atomic"]["coarsening"] > 0
    # Again: the same maps, replaced in place.
    again = _wait(client, client.post(f"/api/datasets/{d['id']}/atomic/fourier", json=body).json())
    assert again["result"]["ids"] == ids
    # Checks before the job.
    bad = {**body, "tensor": [1, 1]}
    assert client.post(f"/api/datasets/{d['id']}/atomic/fourier", json=bad).status_code == 422
    assert client.post(f"/api/datasets/{d['id']}/atomic/fourier", json={**body, "per_peak": [], "tensor": None}).status_code == 422
    # Deletable.
    assert client.delete(f"/api/datasets/{ids[0]}").status_code == 200


# ---------- M22d: atomic columns ----------

def test_find_and_refine_columns_match_kemstem(crop):
    """Notebook 2's settings on a crop of the double perovskite."""
    import kemstem

    opts = atomic.ColumnOptions(blur=0.5, distance=4, threshold=0.1, window=7)
    found = atomic.find_columns(crop, opts)
    blurred = gaussian_filter(kemstem.normalize(crop), 0.5)
    ref = kemstem.find_columns(blurred, distance=4, threshold=0.1)
    assert np.array_equal(found, ref) and len(found) > 100

    table = atomic.refine_columns(crop, found, opts)
    assert table.shape == (len(found), len(atomic.COLUMN_FIELDS))
    pos, errs, params, _ = kemstem.refine_columns(blurred, found, window_dimension=7, remove_unfit=False, verbose=False)
    assert np.allclose(table[:, :2], pos) and np.array_equal(table[:, 8].astype(bool), errs)
    ok = ~errs
    assert np.allclose(table[ok, 2], params[ok, 0]) and np.allclose(table[ok, 7], params[ok, 0] + params[ok, 6])
    # Refined columns stay near where they were found.
    assert np.median(np.hypot(*(table[ok, :2] - found[ok]).T)) < 1.0

    t = atomic.test_column(crop, tuple(found[len(found) // 2]), opts)
    assert np.array(t["data"]).shape == (7, 7) == np.array(t["fit"]).shape
    assert t["row"][:2] == pytest.approx(list(table[len(found) // 2, :2]))
    with pytest.raises(ValueError):
        atomic.ColumnOptions(window=6).check()


def test_columns_endpoints(client):
    d = client.post("/api/examples/kemstem-sites").json()["datasets"][0]
    url = f"/api/datasets/{d['id']}/atomic/columns"
    found = client.post(f"{url}/find", json={"distance": 4, "threshold": 0.1}).json()["columns"]
    assert len(found) > 1000
    # A small part, to keep the test quick.
    part = [c for c in found if 300 <= c[0] < 420 and 300 <= c[1] < 420]
    t = client.post(f"{url}/test", json={"at": part[0], "window": 7}).json()
    assert t["fields"][0] == "y" and len(t["data"]) == 7
    job = _wait(client, client.post(f"{url}/refine", json={"columns": part, "window": 7}).json())
    assert job["status"] == "done", job
    cid = job["result"]["id"]
    made = next(x for x in job["result"]["file"]["datasets"] if x["id"] == cid)
    assert made["kind"] == "other" and made["parent_id"] == d["id"] and made["title"] == "Columns"
    assert made["atomic"]["output"] == "columns" and made["atomic"]["count"] == len(part)
    table = client.get(f"/api/datasets/{cid}/atomic/columns").json()
    assert len(table["rows"]) == len(part) and table["fields"][-1] == "failed"
    # Again: the same dataset, in place.
    again = _wait(client, client.post(f"{url}/refine", json={"columns": part[:10], "window": 7}).json())
    assert again["result"]["id"] == cid
    assert len(client.get(f"/api/datasets/{cid}/atomic/columns").json()["rows"]) == 10
    # Sublattices: datasets under the Columns, the table's "sublattice" field from them.
    rows = client.get(f"/api/datasets/{cid}/atomic/columns").json()["rows"]
    intensity = [r[7] for r in rows]
    centres = client.post("/api/atomic/kmeans", json={"values": intensity, "k": 2}).json()["centres"]
    assert centres[0] > centres[1]
    cut = sum(centres) / 2
    labels = [None if v is None else (0 if v > cut else 1) for v in intensity]
    sub_url = f"/api/datasets/{cid}/atomic/columns/sublattices"
    r = client.put(sub_url, json={"labels": labels, "names": ["A", "B"]})
    assert r.status_code == 200, r.text
    subs = [x for x in r.json()["file"]["datasets"] if x["parent_id"] == cid]
    assert [x["title"] for x in subs] == ["Sublattice A", "Sublattice B"] and [x["id"] for x in subs] == r.json()["ids"]
    assert all(x["atomic"]["output"] == "sublattice" and x["kind"] == "other" for x in subs)
    assert sum(x["atomic"]["count"] for x in subs) == sum(v is not None for v in labels)
    table = client.get(f"/api/datasets/{cid}/atomic/columns").json()
    assert table["fields"][-1] == "sublattice" and [row[-1] for row in table["rows"]] == labels
    # Deleting one leaves its columns in none; again replaces them.
    assert client.delete(f"/api/datasets/{subs[1]['id']}").status_code == 200
    assert [row[-1] for row in client.get(f"/api/datasets/{cid}/atomic/columns").json()["rows"]] == [
        x if x == 0 else None for x in labels]
    r = client.put(sub_url, json={"labels": [0] * 10, "names": ["X"]})
    assert [x["title"] for x in r.json()["file"]["datasets"] if x["parent_id"] == cid] == ["Sublattice X"]
    assert client.put(sub_url, json={"labels": [2] * 10, "names": ["A"]}).status_code == 422
    assert client.put(sub_url, json={"labels": [0], "names": ["A"]}).status_code == 422
    assert client.post("/api/atomic/kmeans", json={"values": [1.0], "k": 2}).status_code == 422
    # Cleared: none left, no field.
    r = client.put(sub_url, json={"labels": [None] * 10, "names": []})
    assert not [x for x in r.json()["file"]["datasets"] if x["parent_id"] == cid]
    assert client.get(f"/api/datasets/{cid}/atomic/columns").json()["fields"][-1] == "failed"
    client.put(sub_url, json={"labels": [0] * 10, "names": ["A"]})
    # Fitting again: the old sublattices go.
    again = _wait(client, client.post(f"{url}/refine", json={"columns": part[:10], "window": 7}).json())
    assert not [x for x in again["result"]["file"]["datasets"] if x["parent_id"] == cid]
    assert client.post(f"{url}/refine", json={"columns": [], "window": 7}).status_code == 422
    assert client.post(f"{url}/find", json={"window": 4}).status_code == 422
    assert client.get(f"/api/datasets/{d['id']}/atomic/columns").status_code == 400
    assert client.delete(f"/api/datasets/{cid}").status_code == 200


def test_columns_and_maps_saved_in_analyses(tmp_path):
    """A Columns dataset and an atomic map come back when an analysis opens."""
    import shutil

    raw = shutil.copy(kemstem_data() / SITES, tmp_path / "sites.tif")
    a = TestClient(create_app())
    d = a.post("/api/files", json={"path": str(raw)}).json()["datasets"][0]
    session = a.app.state.session
    image = session.get_dataset(d["id"])
    table = np.arange(18, dtype=float).reshape(2, 9)
    table[:, 8] = [0, 1]
    cols = session.set_columns(image, table, {"blur": 0.5, "distance": 4, "threshold": 0.1, "window": 7})
    sub = session.set_sublattices(cols, [None, 0], ["A"])[0]
    f1, f2 = _two_peaks(np.asarray(image.signal.data, dtype=np.float64))
    _wait(a, a.post(f"/api/datasets/{d['id']}/atomic/fourier",
                    json={"peaks": {"1": list(f1)}, "per_peak": [1], "outputs": ["amplitude"], "sigma": 10}).json())
    saved = a.post("/api/project/save", json={"path": str(tmp_path / "an"), "views": {}}).json()["path"]

    b = TestClient(create_app())
    r = b.post("/api/project/open", json={"path": saved})
    assert r.status_code == 200, r.text
    got = b.app.state.session.get_dataset(cols.id)
    assert got.kind.value == "other" and got.atomic["output"] == "columns" and got.parent_id == d["id"]
    assert np.allclose(np.asarray(got.signal.data), table)
    rows = b.get(f"/api/datasets/{cols.id}/atomic/columns").json()["rows"]
    assert rows[1][8] == 1 and [r[9] for r in rows] == [None, 0]
    got_sub = b.app.state.session.get_dataset(sub.id)
    assert got_sub.parent_id == cols.id and got_sub.atomic["output"] == "sublattice"
    amp = [x for x in b.app.state.session.datasets.values() if x.atomic and x.atomic["output"] == "amplitude"]
    assert len(amp) == 1 and amp[0].parent_id == d["id"]


def test_neighbourhoods_and_bonds(client):
    """Neighbourhoods, clusters and bonds of a square lattice (two
    sublattices, checkerboard), against kemstem's own calls."""
    import kemstem

    d = client.post("/api/examples/kemstem-sites").json()["datasets"][0]
    session = client.app.state.session
    image = session.get_dataset(d["id"])
    yy, xx = np.mgrid[20:120:10, 30:130:10]
    n = yy.size
    table = np.zeros((n, 9))
    table[:, 0] = yy.ravel() + 0.2 * np.sin(xx.ravel())  # a little disorder
    table[:, 1] = xx.ravel()
    table[:, 7] = np.where((yy.ravel() + xx.ravel()) % 20 == 0, 0.8, 0.3)
    cols = session.set_columns(image, table, {"blur": 0.5, "distance": 4, "threshold": 0.1, "window": 7})
    labels = [0 if v > 0.5 else 1 for v in table[:, 7]]
    a, b = session.set_sublattices(cols, labels, ["A", "B"])

    r = client.post(f"/api/datasets/{a.id}/atomic/neighbourhood", json={"neighbours": cols.id, "k": 8})
    assert r.status_code == 200, r.text
    nb_id = r.json()["id"]
    made = next(x for x in r.json()["file"]["datasets"] if x["id"] == nb_id)
    assert made["parent_id"] == cols.id and made["atomic"]["origins"] == a.id and made["atomic"]["k"] == 8
    assert made["title"] == "Neighbourhood A → all"
    got = client.get(f"/api/datasets/{nb_id}/atomic/neighbourhood").json()
    pos = np.asarray(cols.signal.data, dtype=np.float64)[:, :2]  # as stored (float32)
    want = kemstem.get_neighborhood(pos, pos[np.array(labels) == 0], k=8)
    assert np.allclose(np.column_stack([got["dy"], got["dx"]]), want, atol=1e-3)

    r = client.post(f"/api/datasets/{nb_id}/atomic/neighbourhood/clusters", json={"n": 9})
    clusters = np.array(next(x for x in r.json()["file"]["datasets"] if x["id"] == nb_id)["atomic"]["clusters"])
    assert clusters.shape == (9, 2)
    assert np.allclose(clusters, kemstem.cluster_neighbors(want, n_clusters=9), atol=1e-3)
    v1 = np.array([0.0, 10.0])  # the in-plane lattice vector

    r = client.post(f"/api/datasets/{nb_id}/atomic/bonds", json={"name": "v1", "vector": list(v1), "threshold": 3})
    assert r.status_code == 200, r.text
    bid = r.json()["id"]
    bonds = next(x for x in r.json()["file"]["datasets"] if x["id"] == bid)
    assert bonds["title"] == "Bonds v1" and bonds["parent_id"] == nb_id
    rows = np.array(client.get(f"/api/datasets/{bid}/atomic/bonds").json()["rows"], dtype=float)
    want = kemstem.get_vector_to_neighbors(pos, pos[np.array(labels) == 0], v1, threshold=3)
    assert np.allclose(rows[:, 1:3], want, atol=1e-3, equal_nan=True)
    assert np.allclose(rows[:, 0], np.flatnonzero(np.array(labels) == 0))
    ok = np.isfinite(rows[:, 3])
    assert bonds["atomic"]["missing"] == int((~ok).sum()) > 0  # the right edge has none
    assert np.allclose(rows[ok, 3], 10, atol=0.5) and np.allclose(rows[ok, 4], 0, atol=3)
    # Again: in place. Made again with other settings: in place, its name
    # kept, its bonds and clusters gone.
    again = client.post(f"/api/datasets/{nb_id}/atomic/bonds", json={"name": "v1", "vector": list(v1)}).json()
    assert again["id"] == bid
    client.put(f"/api/datasets/{nb_id}/title", json={"title": "AA tilt"})
    r = client.post(f"/api/datasets/{a.id}/atomic/neighbourhood", json={"neighbours": cols.id, "k": 4, "replace": nb_id})
    assert r.json()["id"] == nb_id
    assert not [x for x in r.json()["file"]["datasets"] if x["parent_id"] == nb_id]
    made = next(x for x in r.json()["file"]["datasets"] if x["id"] == nb_id)
    assert "clusters" not in made["atomic"] and made["atomic"]["k"] == 4 and made["title"] == "AA tilt"
    # Without ``replace``: a new one each time, named apart.
    r2 = client.post(f"/api/datasets/{a.id}/atomic/neighbourhood", json={"neighbours": cols.id, "k": 8}).json()
    r3 = client.post(f"/api/datasets/{a.id}/atomic/neighbourhood", json={"neighbours": cols.id, "k": 8}).json()
    titles = {x["id"]: x["title"] for x in r3["file"]["datasets"]}
    assert len({nb_id, r2["id"], r3["id"]}) == 3
    assert titles[r2["id"]] == "Neighbourhood A → all" and titles[r3["id"]] == "Neighbourhood A → all 2"
    # Replaced with other origins (B → A): in place under the Columns, name kept.
    r4 = client.post(f"/api/datasets/{b.id}/atomic/neighbourhood", json={"neighbours": a.id, "replace": nb_id}).json()
    moved = next(x for x in r4["file"]["datasets"] if x["id"] == r4["id"])
    assert r4["id"] == nb_id and moved["parent_id"] == cols.id and moved["title"] == "AA tilt"
    assert moved["atomic"]["origins"] == b.id and moved["atomic"]["neighbours"] == a.id
    nb_id = r2["id"]
    # Checks.
    assert client.post(f"/api/datasets/{a.id}/atomic/neighbourhood", json={"neighbours": cols.id, "k": 0}).status_code == 422
    assert client.post(f"/api/datasets/{d['id']}/atomic/neighbourhood", json={"neighbours": cols.id}).status_code == 400
    assert client.post(f"/api/datasets/{nb_id}/atomic/bonds", json={"name": "x", "vector": [0, 10], "threshold": 0}).status_code == 422
    # Deleting a sublattice: the neighbourhoods using it go too, and come back with an undo.
    r5 = client.post(f"/api/datasets/{b.id}/atomic/neighbourhood", json={"neighbours": a.id}).json()
    gone = client.delete(f"/api/datasets/{a.id}").json()
    assert r5["id"] not in session.datasets and nb_id not in session.datasets
    assert client.post(f"/api/undo/{gone['undo']}").status_code == 200
    assert r5["id"] in session.datasets and nb_id in session.datasets and a.id in session.datasets
    # Splitting again: neighbourhoods of the old sublattices go.
    session.set_sublattices(cols, labels, ["A", "B"])
    assert nb_id not in session.datasets
    assert client.delete(f"/api/datasets/{cols.id}").status_code == 200


def _pld_lattice(n=160, a=10, amp=0.6):
    """A square lattice of Gaussian columns, displaced along y by ±amp in turn
    along x (a period of two columns): (image, true positions, displacements)."""
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    image = np.zeros((n, n))
    pos, disp = [], []
    for y0 in np.arange(a / 2, n, a):
        for x0 in np.arange(a / 2, n, a):
            d = amp * (-1) ** int(x0 // a)  # ±amp, alternating along x
            image += np.exp(-((yy - y0 - d) ** 2 + (xx - x0) ** 2) / (2 * 1.6 ** 2))
            pos.append((y0 + d, x0))
            disp.append((d, 0.0))
    return image, np.array(pos), np.array(disp)


def test_pld_detect_reference_and_displacements(tmp_path):
    """Superlattice peaks found and grouped, damped into a reference, and the
    displacements measured back (against the known ones)."""
    import tifffile

    image, pos, disp = _pld_lattice()
    path = tmp_path / "pld.tif"
    tifffile.imwrite(path, image.astype(np.float32))
    c = TestClient(create_app())
    d = c.post("/api/files", json={"path": str(path)}).json()["datasets"][0]
    url = f"/api/datasets/{d['id']}/atomic/pld"

    r = c.post(f"{url}/detect", json={"count": 60, "within": 0.3}).json()
    basis = np.array(r["basis"])
    assert np.allclose(sorted(np.hypot(*basis.T)), [0.1, 0.1], atol=0.01)  # the 10 px lattice
    fam = r["families"][0]
    # The modulation: half a lattice vector along x (q = (0, ½) or (½, 0) in the basis).
    assert sorted(np.abs(fam["q"])) == pytest.approx([0, 0.5], abs=0.05)
    sl = [p for p in r["peaks"] if p["family"] == fam["id"]]
    assert len(sl) >= 4 and all(p["label"] == "superlattice" for p in sl)
    assert any(p["family"] == 0 for p in r["peaks"])

    r = c.post(f"{url}/reference", json={"peaks": [[p["fy"], p["fx"]] for p in sl], "radius": 4})
    assert r.status_code == 200, r.text
    made = {x["id"]: x for x in r.json()["file"]["datasets"]}
    ref, diff = (made[i] for i in r.json()["ids"])
    assert ref["title"] == "PLD reference" and ref["kind"] == "image" and ref["parent_id"] == d["id"]
    assert diff["atomic"]["output"] == "pld-difference"
    assert c.post(f"{url}/reference", json={"peaks": [[0.2, 0.2]], "radius": 0}).status_code == 422

    session = c.app.state.session
    image_ds = session.get_dataset(d["id"])
    table = np.zeros((len(pos), 9))
    table[:, :2] = pos
    cols = session.set_columns(image_ds, table, {"blur": 0.0, "distance": 5, "threshold": 0.1, "window": 7})
    job = _wait(c, c.post(f"/api/datasets/{cols.id}/atomic/pld/displacements", json={"threshold": 1.5}).json())
    assert job["status"] == "done", job
    out = next(x for x in job["result"]["file"]["datasets"] if x["id"] == job["result"]["id"])
    assert out["title"] == "Displacements" and out["parent_id"] == cols.id and out["atomic"]["output"] == "displacements"
    rows = np.array(c.get(f"/api/datasets/{out['id']}/atomic/displacements").json()["rows"], dtype=float)
    # Away from the edges, the measured displacement is the fitted position less
    # the reference's: the known wave (fitted positions are the true ones here).
    inner = (pos[:, 0] > 20) & (pos[:, 0] < 140) & (pos[:, 1] > 20) & (pos[:, 1] < 140)
    ok = inner & np.isfinite(rows[:, 0])
    assert ok.sum() > 50
    assert np.corrcoef(rows[ok, 0], disp[ok, 0])[0, 1] > 0.95
    assert np.allclose(np.median(np.abs(rows[ok, 0])), 0.6, atol=0.2)
    assert np.allclose(rows[ok, 2], np.hypot(rows[ok, 0], rows[ok, 1]), atol=1e-4)
    # Refitting the columns takes them away.
    session.set_columns(image_ds, table, {"blur": 0.0, "distance": 5, "threshold": 0.1, "window": 7})
    assert out["id"] not in session.datasets


def test_sublattices_from_a_mask(tmp_path):
    """A sublattice of the columns in a mask, added next to the others."""
    import tifffile

    image, pos, _ = _pld_lattice(n=100)
    path = tmp_path / "lattice.tif"
    tifffile.imwrite(path, image.astype(np.float32))
    c = TestClient(create_app())
    d = c.post("/api/files", json={"path": str(path)}).json()["datasets"][0]
    session = c.app.state.session
    image_ds = session.get_dataset(d["id"])
    table = np.zeros((len(pos), 9))
    table[:, :2] = pos
    table[0, :2] = np.nan  # a column without a position: in neither
    cols = session.set_columns(image_ds, table, {"blur": 0.0, "distance": 5, "threshold": 0.1, "window": 7})
    # A mask of the left half, as a dataset under the image.
    m = np.zeros(image.shape, bool)
    m[:, :50] = True
    mask = session.set_mask(image_ds, m, {"keep": "above", "lo": 0.5})

    url = f"/api/datasets/{cols.id}/atomic/columns/sublattices/mask"
    r = c.post(url, json={"mask": mask.id})
    assert r.status_code == 200, r.text
    made = {x["id"]: x for x in r.json()["file"]["datasets"]}
    a = made[r.json()["id"]]
    left = np.isfinite(pos[:, 0]) & (pos[:, 1] < 49.5)
    left[0] = False
    assert a["title"] == "Sublattice A" and a["atomic"]["count"] == left.sum() and a["atomic"]["mask"] == mask.id
    assert sorted(np.asarray(session.get_dataset(a["id"]).signal.data).tolist()) == np.flatnonzero(left).tolist()
    # Only that one: no sublattice of the rest.
    assert [x["title"] for x in made.values() if x["atomic"] and x["atomic"].get("output") == "sublattice"] == ["Sublattice A"]
    # A second is added next to it (B), the first kept.
    r2 = c.post(url, json={"mask": mask.id}).json()
    subs = [x for x in r2["file"]["datasets"] if x["atomic"] and x["atomic"].get("output") == "sublattice"]
    assert [x["title"] for x in subs] == ["Sublattice A", "Sublattice B"]
    # Not an image of the same size: refused.
    assert c.post(url, json={"mask": cols.id}).status_code == 422


def _perovskite(n=200, a=20, shift=(1.0, -0.6)):
    """Bright A columns on a square lattice (a px), dimmer B columns at the
    cell centres displaced by ``shift`` (dy, dx) px, the sign alternating by
    row of cells: (image, A positions, B positions, B displacements)."""
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    image = np.zeros((n, n))
    A, Bs, D = [], [], []
    for y0 in np.arange(a / 2, n, a):
        for x0 in np.arange(a / 2, n, a):
            image += np.exp(-((yy - y0) ** 2 + (xx - x0) ** 2) / (2 * 2.0 ** 2))
            A.append((y0, x0))
            if y0 + a < n and x0 + a < n:
                s = 1 if int(y0 // a) % 2 == 0 else -1
                d = (s * shift[0], s * shift[1])
                b = (y0 + a / 2 + d[0], x0 + a / 2 + d[1])
                image += 0.4 * np.exp(-((yy - b[0]) ** 2 + (xx - b[1]) ** 2) / (2 * 2.0 ** 2))
                Bs.append(b)
                D.append(d)
    return image, np.array(A), np.array(Bs), np.array(D)


def test_cage_centres_and_displacement(tmp_path):
    """B sites seeded at the centres of four A sites (a neighbourhood's picked
    clusters), fitted as a sublattice of their own, then each B's displacement
    from its A cage measured back."""
    import tifffile

    image, A, Bs, D = _perovskite()
    path = tmp_path / "perovskite.tif"
    tifffile.imwrite(path, image.astype(np.float32))
    c = TestClient(create_app())
    d = c.post("/api/files", json={"path": str(path)}).json()["datasets"][0]
    url = f"/api/datasets/{d['id']}/atomic/columns"

    # Only the bright A columns: a high-pass and a high threshold.
    found = np.array(c.post(f"{url}/find", json={"blur": 1, "background": 30, "distance": 6, "threshold": 0.5}).json()["columns"])
    assert len(found) == len(A)
    job = _wait(c, c.post(f"{url}/refine", json={"columns": found.tolist(), "window": 7}).json())
    cols = job["result"]["id"]

    # A → A neighbourhood; the cage: the origin, right, down and the corner.
    nb = c.post(f"/api/datasets/{cols}/atomic/neighbourhood", json={"neighbours": cols, "k": 8}).json()["id"]
    cage = [[0, 0], [0, 20], [20, 0], [20, 20]]
    r = c.post(f"/api/datasets/{nb}/atomic/neighbourhood/centres", json={"vectors": cage, "threshold": 4}).json()
    assert r["name"] == "B" and r["added"] == len(Bs) and r["id"] == cols
    seeds = np.array(r["columns"])
    # Each seed at an undisplaced cell centre.
    centres = Bs - D
    assert np.allclose(np.sort(seeds, axis=0), np.sort(centres, axis=0), atol=0.05)
    assert c.post(f"/api/datasets/{nb}/atomic/neighbourhood/centres", json={"vectors": [[0, 0], [7, 7]], "threshold": 2}).status_code == 422

    # Fit only the new ones, added after the A columns (which keep their fit).
    before = np.array(c.get(f"/api/datasets/{cols}/atomic/columns").json()["rows"], dtype=float)
    job = _wait(c, c.post(f"/api/datasets/{cols}/atomic/columns/append", json={"columns": r["columns"], "window": 7}).json())
    assert job["status"] == "done", job
    assert job["result"]["id"] == cols
    files = job["result"]["file"]["datasets"]
    subs = {x["title"]: x for x in files if x["atomic"] and x["atomic"].get("output") == "sublattice"}
    assert subs["Sublattice A"]["atomic"]["count"] == len(A) and subs["Sublattice B"]["atomic"]["count"] == len(Bs)
    assert subs["Sublattice B"]["id"] == job["result"]["sublattice"]
    after = np.array(c.get(f"/api/datasets/{cols}/atomic/columns").json()["rows"], dtype=float)
    assert len(after) == len(A) + len(Bs)
    assert np.allclose(after[:len(A), :len(before[0])], before, equal_nan=True)  # untouched
    # The A → A neighbourhood stays, now of Sublattice A.
    kept = next(x for x in files if x["id"] == nb)
    assert kept["atomic"]["origins"] == subs["Sublattice A"]["id"] and kept["atomic"]["neighbours"] == subs["Sublattice A"]["id"]
    assert kept["title"] == "Neighbourhood A → A"

    # B → A: the four diagonal A sites make the cage; displacement = B − their mean.
    b, a = subs["Sublattice B"]["id"], subs["Sublattice A"]["id"]
    nb2 = c.post(f"/api/datasets/{b}/atomic/neighbourhood", json={"neighbours": a, "k": 4}).json()["id"]
    diag = [[-10, -10], [-10, 10], [10, -10], [10, 10]]
    r = c.post(f"/api/datasets/{nb2}/atomic/neighbourhood/displacement", json={"vectors": diag, "threshold": 4})
    assert r.status_code == 200, r.text
    out = next(x for x in r.json()["file"]["datasets"] if x["id"] == r.json()["id"])
    assert out["title"] == "Displacement" and out["parent_id"] == nb2 and out["atomic"]["output"] == "displacements"
    rows = np.array(c.get(f"/api/datasets/{out['id']}/atomic/displacements").json()["rows"], dtype=float)
    table = np.array(c.get(f"/api/datasets/{job['result']['id']}/atomic/columns").json()["rows"], dtype=float)
    # Rows of B columns: compare with the known shift of the nearest true B.
    got = 0
    for i in range(len(A), len(table)):
        j = np.argmin(np.hypot(*(Bs - table[i, :2]).T))
        assert np.allclose(rows[i, :2], D[j], atol=0.15)
        got += 1
    assert got == len(Bs)
    assert np.isnan(rows[:len(A), 0]).all()  # A rows: none
    assert out["atomic"]["median"] == pytest.approx(np.hypot(1.0, 0.6), abs=0.1)
    # Again: the same dataset.
    again = c.post(f"/api/datasets/{nb2}/atomic/neighbourhood/displacement", json={"vectors": diag, "threshold": 4}).json()
    assert again["id"] == out["id"]


def test_template_finding(crop, client):
    """Finding by correlation with a Gaussian template matches
    ptycho-quant-analysis's ``find_peaks`` (match_template + peak_local_max)
    on the same prepared image, and reaches the dim columns too."""
    from skimage.feature import match_template, peak_local_max

    opts = atomic.ColumnOptions(blur=1.0, distance=3, threshold=0.6, method="template", template=1.5)
    got = atomic.find_columns(crop, opts)
    im = atomic.prepared(crop, 1.0)
    w = 2 * int(np.ceil(2 * 1.5)) + 1
    yy, xx = np.indices((w, w))
    tmpl = np.exp(-((xx - (w - 1) / 2) ** 2 + (yy - (w - 1) / 2) ** 2) / (2 * 1.5 ** 2))
    cc = match_template(im, tmpl, pad_input=True, mode="edge")
    ref = peak_local_max(cc, min_distance=3, threshold_abs=0.6, exclude_border=2)
    assert np.array_equal(got, ref) and len(got) > 100
    # The correlation doesn't care how bright a column is: it finds more than
    # a 0.6 threshold on the image itself.
    assert len(got) > len(atomic.find_columns(crop, atomic.ColumnOptions(blur=1.0, distance=3, threshold=0.6)))
    with pytest.raises(ValueError):
        atomic.ColumnOptions(method="template", threshold=1.5).check()
    with pytest.raises(ValueError):
        atomic.ColumnOptions(method="template", template=0.1).check()

    d = client.post("/api/examples/kemstem-sites").json()["datasets"][0]
    url = f"/api/datasets/{d['id']}/atomic/columns/find"
    r = client.post(url, json={"method": "template", "template": 1.5, "threshold": 0.6, "distance": 3, "blur": 1.0})
    assert r.status_code == 200 and len(r.json()["columns"]) > 1000
    assert client.post(url, json={"method": "template", "threshold": -2}).status_code == 422
    assert client.post(url, json={"method": "nope"}).status_code == 422


def test_sublattices_by_position_in_cell(tmp_path):
    """Columns placed in the cell of a reference sublattice's lattice, then
    split by the nearest picked site (the cell repeating)."""
    import tifffile

    image, A, Bs, D = _perovskite()
    path = tmp_path / "perovskite.tif"
    tifffile.imwrite(path, image.astype(np.float32))
    c = TestClient(create_app())
    d = c.post("/api/files", json={"path": str(path)}).json()["datasets"][0]
    session = c.app.state.session
    pos = np.concatenate([A, Bs])
    table = np.zeros((len(pos) + 1, 9))
    table[:-1, :2] = pos
    table[-1, :2] = np.nan  # no position: no place, no site
    cols = session.set_columns(session.get_dataset(d["id"]), table, {"window": 7})
    # The reference: the A columns (a sublattice), lattice vectors along x and y.
    [ref] = session.set_sublattices(cols, [0] * len(A) + [None] * (len(Bs) + 1), ["A"])
    vectors = [[0, 20], [20, 0]]

    r = c.post(f"/api/datasets/{cols.id}/atomic/columns/cell", json={"reference": ref.id, "vectors": vectors}).json()
    u, v = np.array(r["u"][:-1], dtype=float), np.array(r["v"][:-1], dtype=float)
    assert r["u"][-1] is None
    assert np.allclose(u[:len(A)], 0, atol=1e-6) and np.allclose(v[:len(A)], 0, atol=1e-6)
    # B: half a cell along both, less its shift (±1, ∓0.6 px of 20).
    assert np.allclose(np.abs(v[len(A):]), np.abs(0.5 - np.abs(D[:, 0]) / 20), atol=1e-3)
    assert np.allclose(np.abs(u[len(A):]), np.abs(0.5 - np.abs(D[:, 1]) / 20), atol=1e-3)

    url = f"/api/datasets/{cols.id}/atomic/columns/sublattices/cell"
    r = c.put(url, json={"reference": ref.id, "vectors": vectors, "sites": [[0, 0], [0.5, 0.5]], "radius": 3})
    assert r.status_code == 200, r.text
    subs = {x["title"]: x["atomic"]["count"] for x in r.json()["file"]["datasets"]
            if x["atomic"] and x["atomic"].get("output") == "sublattice"}
    assert subs == {"Sublattice A": len(A), "Sublattice B": len(Bs)}
    # The split replaces the sublattices (the reference too): the new A is the reference now.
    a = lambda r: next(x["id"] for x in r["file"]["datasets"] if x["title"] == "Sublattice A")
    assert c.put(url, json={"reference": ref.id, "vectors": vectors, "sites": [[0, 0]]}).status_code == 404
    # A site at (−½, −½) is the same one, the cell repeating.
    r = c.put(url, json={"reference": a(r.json()), "vectors": vectors, "sites": [[0, 0], [-0.5, -0.5]], "radius": 3}).json()
    subs = {x["title"]: x["atomic"]["count"] for x in r["file"]["datasets"]
            if x["atomic"] and x["atomic"].get("output") == "sublattice"}
    assert subs["Sublattice B"] == len(Bs)
    # Too small a radius: the shifted B columns are left out.
    r = c.put(url, json={"reference": a(r), "vectors": vectors, "sites": [[0, 0], [0.5, 0.5]], "radius": 0.5}).json()
    subs = {x["title"]: x["atomic"]["count"] for x in r["file"]["datasets"]
            if x["atomic"] and x["atomic"].get("output") == "sublattice"}
    assert subs["Sublattice B"] == 0 and subs["Sublattice A"] == len(A)
    # Errors: parallel vectors, no sites.
    assert c.put(url, json={"reference": a(r), "vectors": [[0, 20], [0, 40]], "sites": [[0, 0]]}).status_code == 422
    assert c.put(url, json={"reference": a(r), "vectors": vectors, "sites": []}).status_code == 422
    # Two sites of one sublattice (named alike).
    r = c.put(url, json={"reference": a(r), "vectors": vectors, "sites": [[0, 0], [0.5, 0.5], [0.5, -0.5]],
                         "names": ["A", "B", "B"], "radius": 3}).json()
    subs = {x["title"]: x["atomic"]["count"] for x in r["file"]["datasets"]
            if x["atomic"] and x["atomic"].get("output") == "sublattice"}
    assert subs == {"Sublattice A": len(A), "Sublattice B": len(Bs)}
    assert c.put(url, json={"reference": a(r), "vectors": vectors, "sites": [[0, 0]], "names": ["A", "B"]}).status_code == 422


def test_cage_of_the_k_nearest(tmp_path):
    """A cage of each origin's k nearest neighbours (no clusters picked):
    sites moving up and down in turn (as oxygen with octahedral tilts) all
    measured from the centre of their four A columns; a site without a whole
    cage (at the edge) is left out."""
    import tifffile

    a = 20
    A = np.array([(y, x) for y in np.arange(10, 200, a) for x in np.arange(10, 200, a)], dtype=float)
    C, D = [], []
    for i, y in enumerate(np.arange(20, 180, a)):
        for j, x in enumerate(np.arange(20, 180, a)):
            d = 1.0 if (i + j) % 2 else -1.0  # up and down in turn
            C.append((y + d, x))
            D.append((d, 0.0))
    C.append((1.0, 100.0))  # above the first A row: no whole cage
    C, D = np.array(C), np.array(D)
    path = tmp_path / "tilts.tif"
    tifffile.imwrite(path, np.zeros((200, 200), np.float32))
    c = TestClient(create_app())
    d = c.post("/api/files", json={"path": str(path)}).json()["datasets"][0]
    session = c.app.state.session
    table = np.zeros((len(A) + len(C), 9))
    table[:, :2] = np.concatenate([A, C])
    cols = session.set_columns(session.get_dataset(d["id"]), table, {"window": 7})
    sa, sc = session.set_sublattices(cols, [0] * len(A) + [1] * len(C), ["A", "C"])
    nb = c.post(f"/api/datasets/{sc.id}/atomic/neighbourhood", json={"neighbours": sa.id, "k": 4}).json()["id"]

    url = f"/api/datasets/{nb}/atomic/neighbourhood/displacement"
    assert c.post(url, json={"threshold": 3}).status_code == 422  # neither clusters nor nearest
    r = c.post(url, json={"nearest": True, "threshold": 3})
    assert r.status_code == 200, r.text
    out = next(x for x in r.json()["file"]["datasets"] if x["id"] == r.json()["id"])
    assert out["atomic"]["nearest"] and out["atomic"]["missing"] == 1
    rows = np.array(c.get(f"/api/datasets/{out['id']}/atomic/displacements").json()["rows"], dtype=float)
    got = rows[len(A):len(A) + len(D), :2]
    assert np.allclose(got, D, atol=1e-6)
    assert np.isnan(rows[-1, 0]) and np.isnan(rows[:len(A), 0]).all()
    # As seeds: the cage centres (undisplaced sites).
    r = c.post(f"/api/datasets/{nb}/atomic/neighbourhood/centres", json={"nearest": True, "threshold": 3}).json()
    assert r["added"] == len(D) and np.allclose(np.sort(np.array(r["columns"]), axis=0), np.sort(C[:-1] - D, axis=0))


def test_picks_kept_with_datasets(tmp_path):
    """Picks made in the page (lattice vectors, cell sites …) kept on their
    dataset, in its summary, through later changes, and in saved analyses."""
    import tifffile

    image, A, Bs, _ = _perovskite(n=100)
    path = tmp_path / "picks.tif"
    tifffile.imwrite(path, image.astype(np.float32))
    c = TestClient(create_app())
    d = c.post("/api/files", json={"path": str(path)}).json()["datasets"][0]
    session = c.app.state.session
    table = np.zeros((len(A), 9))
    table[:, :2] = A
    cols = session.set_columns(session.get_dataset(d["id"]), table, {"window": 7})
    nb = c.post(f"/api/datasets/{cols.id}/atomic/neighbourhood", json={"neighbours": cols.id, "k": 4}).json()["id"]
    picks = {"vectors": [{"name": "v₁", "dy": 0.0, "dx": 20.0}], "cage": [[0, 0], [20, 20]], "cageNearest": False}
    assert c.put(f"/api/datasets/{nb}/atomic/picks", json={"picks": picks}).status_code == 204
    assert c.put(f"/api/datasets/{cols.id}/atomic/picks", json={"picks": {"cell": {"sites": [[0, 0, "A"]]}}}).status_code == 204
    assert c.put(f"/api/datasets/{d['id']}/atomic/picks", json={"picks": {}}).status_code == 400  # not of the analysis
    assert c.put("/api/datasets/nope/atomic/picks", json={"picks": {}}).status_code == 404
    # In the summary, and kept through clustering.
    c.post(f"/api/datasets/{nb}/atomic/neighbourhood/clusters", json={"n": 4})
    files = c.get("/api/files").json()
    ds = {x["id"]: x for f in files for x in f["datasets"]}
    assert ds[nb]["atomic"]["picks"] == picks and ds[nb]["atomic"]["clusters"]
    assert ds[cols.id]["atomic"]["picks"]["cell"]["sites"] == [[0, 0, "A"]]
    # Saved with the analysis and back on opening it.
    out = tmp_path / "a.h5"
    assert c.post("/api/project/save", json={"path": str(out)}).status_code == 200
    c2 = TestClient(create_app())
    r = c2.post("/api/project/open", json={"path": str(out)})
    assert r.status_code == 200, r.text
    ds2 = {x["id"]: x for f in c2.get("/api/files").json() for x in f["datasets"]}
    assert ds2[nb]["atomic"]["picks"] == picks


def _streaked_cell(seed=0):
    """Columns of a 24 × 71 px cell (as the iDPC image's): A on the lattice,
    B tight at (0, ½), C at (½, ½) displaced up or down by 6–14 px (a tilt
    streak, across the cell's corners), and a few strays at (½, 0) of no
    site. (positions, true site per column: 0 A, 1 B, 2 C, None stray)."""
    rng = np.random.default_rng(seed)
    pos, site = [], []
    h = 71.0
    for i in range(12):
        for j in range(12):
            y0, x0 = h * i, 24.0 * j
            pos.append((y0 + rng.normal(0, 0.3), x0 + rng.normal(0, 0.3)))
            site.append(0)
            pos.append((y0 + h / 2 + rng.normal(0, 0.5), x0 + rng.normal(0, 0.5)))
            site.append(1)
            d = rng.choice([-1, 1]) * rng.uniform(6, 14)
            pos.append((y0 + h / 2 + d, x0 + 12 + rng.normal(0, 0.5)))
            site.append(2)
            if (i + j) % 9 == 0:
                pos.append((y0 + rng.normal(0, 1), x0 + 12 + rng.normal(0, 1)))
                site.append(None)
    return np.array(pos), site


def test_site_clusters_follow_streaks():
    """Each site takes the dense group of columns it sits in: a streaked
    site whole (where a circle loses its ends), across the cell's edges;
    strays apart from any site are left out and reported; an empty site
    takes nothing."""
    pos, site = _streaked_cell()
    A = pos[[s == 0 for s in site]]
    vectors = [[0, 24], [71, 0]]
    cell = atomic.cell_coordinates(pos, A, vectors)
    sites = [[0, 0], [0, 0.5], [0.5, 0.5]]
    truth = np.array([-1 if s is None else s for s in site])

    # A circle small enough to leave the strays out loses C's ends.
    circle = np.array([-1 if x is None else x for x in atomic.assign_sites(cell, sites, vectors, 5)])
    assert (circle[truth == 2] == 2).mean() < 0.2
    labels, fits, unpicked = atomic.cluster_sites(cell, sites, vectors, link=4, least=5)
    got = np.array([-1 if x is None else x for x in labels])
    for k in range(3):
        assert (got[truth == k] == k).all(), k
    assert (got[truth == -1] == -1).all()
    assert [f["count"] for f in fits] == [(truth == k).sum() for k in range(3)]
    assert np.allclose([abs(fits[2]["u"])], [0.5], atol=0.02)
    # The strays: a group no site took, at (½, 0).
    assert unpicked and unpicked[0]["count"] == (truth == -1).sum()
    assert abs(abs(unpicked[0]["u"]) - 0.5) < 0.05 and abs(unpicked[0]["v"]) < 0.05
    # An empty site takes nothing; two sites in one group split it by nearness.
    labels, fits, _ = atomic.cluster_sites(cell, sites + [[0.25, 0.25]], vectors, reach=3)
    assert fits[3]["empty"] and fits[3]["count"] == 0
    labels, fits, _ = atomic.cluster_sites(cell, [[0, 0], [0, 0.5], [0.5, 0.4], [0.5, 0.6]], vectors)
    assert fits[2]["count"] + fits[3]["count"] == (truth == 2).sum() and fits[2]["count"] > 0 and fits[3]["count"] > 0
    with pytest.raises(ValueError):
        atomic.cluster_sites(cell, [], vectors)


def test_cell_fit_and_split_by_clusters(tmp_path):
    import tifffile

    pos, site = _streaked_cell(seed=1)
    path = tmp_path / "streaks.tif"
    tifffile.imwrite(path, np.zeros((860, 288), np.float32))
    c = TestClient(create_app())
    d = c.post("/api/files", json={"path": str(path)}).json()["datasets"][0]
    session = c.app.state.session
    table = np.zeros((len(pos), 9))
    table[:, :2] = pos
    cols = session.set_columns(session.get_dataset(d["id"]), table, {"window": 7})
    [ref] = session.set_sublattices(cols, [0 if s == 0 else None for s in site], ["A"])
    body = {"reference": ref.id, "vectors": [[0, 24], [71, 0]], "sites": [[0, 0], [0, 0.5], [0.5, 0.5]],
            "mode": "clusters", "link": 4, "least": 5, "radius": 8}
    r = c.post(f"/api/datasets/{cols.id}/atomic/columns/cell/fit", json=body)
    assert r.status_code == 200, r.text
    fit = r.json()
    assert fit["left_out"] == sum(1 for s in site if s is None)
    assert [x["count"] for x in fit["sites"]] == [site.count(0), site.count(1), site.count(2)]
    assert len(fit["labels"]) == len(pos) and fit["unpicked"][0]["count"] == sum(1 for s in site if s is None)
    r = c.put(f"/api/datasets/{cols.id}/atomic/columns/sublattices/cell", json=body)
    assert r.status_code == 200, r.text
    subs = {x["title"]: x["atomic"]["count"] for x in r.json()["file"]["datasets"]
            if x["atomic"] and x["atomic"].get("output") == "sublattice"}
    assert subs == {"Sublattice A": site.count(0), "Sublattice B": site.count(1), "Sublattice C": site.count(2)}
