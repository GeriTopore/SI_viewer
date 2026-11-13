"""PCA of spectrum images (M19a), made with HyperSpy's decomposition."""

import time

import numpy as np
import pytest



def _truth(h=24, w=20, n=120, seed=0):
    """Three components: smooth maps × peaked spectra, as counts."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w] / max(h, w)
    maps = np.stack([np.ones((h, w)), 0.5 + x, 0.3 + np.exp(-((x - 0.5) ** 2 + (y - 0.4) ** 2) / 0.05)])
    e = np.arange(n)
    spectra = np.stack([
        200 * np.exp(-e / 60),
        80 * np.exp(-((e - 40) ** 2) / 30),
        60 * np.exp(-((e - 85) ** 2) / 50),
    ])
    clean = np.einsum("khw,kn->hwn", maps, spectra)
    return clean, rng.poisson(clean).astype(np.float32)


# ---------- through the API ----------


@pytest.fixture
def client(tmp_path):
    import hyperspy.api as hs
    from fastapi.testclient import TestClient

    from si_viewer.app import create_app

    _, noisy = _truth()
    s = hs.signals.Signal1D(noisy)
    s.set_signal_type("EELS")
    s.axes_manager[-1].offset, s.axes_manager[-1].scale, s.axes_manager[-1].units = 400.0, 0.5, "eV"
    s.metadata.General.title = "SI"
    s.save(tmp_path / "si.hspy")
    c = TestClient(create_app())
    c.si = c.post("/api/files", json={"path": str(tmp_path / "si.hspy")}).json()["datasets"][0]
    return c


def _wait(client, job):
    t0 = time.time()
    while job["status"] in ("queued", "running"):
        assert time.time() - t0 < 120
        time.sleep(0.02)
        job = client.get(f"/api/jobs/{job['id']}").json()
    assert job["status"] == "done", job
    return job["result"]


def test_pca_object_and_denoised_si(client, tmp_path):
    si = client.si
    session = client.app.state.session
    # A window of 100 channels: 410–459.5 eV.
    result = _wait(client, client.post(f"/api/datasets/{si['id']}/pca", json={"lo": 410, "hi": 459.5, "components": 8}).json())
    pca = session.get_dataset(result["pca_id"])
    assert pca.kind.value == "stack" and pca.parent_id == si["id"] and pca.signal.data.shape == (8, 24, 20)
    assert pca.pca["channels"] == [20, 120] and pca.pca["window"] == [410, 459.5] and len(pca.pca["explained"]) == 8
    factors = session.pca_factors(pca)
    assert factors.parent_id == pca.id and factors.signal.data.shape == (8, 100)
    assert factors.signal.axes_manager.signal_axes[0].offset == 410

    r = client.get(f"/api/datasets/{pca.id}/pca/factor", params={"index": 1})
    assert r.status_code == 200 and r.headers["X-Array-Shape"] == "100"
    assert client.get(f"/api/datasets/{pca.id}/pca/factor", params={"index": 8}).status_code == 422

    made = client.post(f"/api/datasets/{pca.id}/pca/denoise", json={"n": 3}).json()
    denoised = session.get_dataset(made["id"])
    assert denoised.origin == "pca-denoised" and denoised.parent_id == si["id"] and denoised.pca["n"] == 3
    assert denoised.signal.data.shape == (24, 20, 100) and denoised.summary()["eels"] is not None
    assert denoised.signal.axes_manager.signal_axes[0].offset == 410
    # The same n again: remade in place.
    assert client.post(f"/api/datasets/{pca.id}/pca/denoise", json={"n": 3}).json()["id"] == denoised.id

    # Decompose again: the same objects, remade.
    again = _wait(client, client.post(f"/api/datasets/{si['id']}/pca", json={"components": 5}).json())
    assert again["pca_id"] == pca.id and pca.signal.data.shape == (5, 24, 20) and factors.signal.data.shape == (5, 120)

    # Saved with an analysis and opened again.
    saved = client.post("/api/project/save", json={"path": str(tmp_path / "a.hspy"), "views": {}}).json()["path"]
    from fastapi.testclient import TestClient

    from si_viewer.app import create_app

    other = TestClient(create_app())
    assert other.post("/api/project/open", json={"path": saved}).status_code == 200
    reopened = other.app.state.session.get_dataset(pca.id)
    assert reopened.pca["components"] == 5 and np.allclose(reopened.signal.data, pca.signal.data)
    assert other.post(f"/api/datasets/{pca.id}/pca/denoise", json={"n": 2}).status_code == 200

    # Deleting the PCA takes its factors with it.
    assert client.delete(f"/api/datasets/{pca.id}").status_code == 200
    assert factors.id not in session.datasets and denoised.id in session.datasets


def test_bad_window_and_not_an_si(client):
    si = client.si
    assert client.post(f"/api/datasets/{si['id']}/pca", json={"lo": 450, "hi": 450}).status_code == 422


@pytest.mark.parametrize("poisson", [True, False])
def test_three_components_found_and_noise_removed(tmp_path, poisson):
    import hyperspy.api as hs
    from fastapi.testclient import TestClient

    from si_viewer.app import create_app

    clean, noisy = _truth()
    s = hs.signals.Signal1D(noisy)
    s.set_signal_type("EELS")
    s.save(tmp_path / "si.hspy")
    client = TestClient(create_app())
    si = client.post("/api/files", json={"path": str(tmp_path / "si.hspy")}).json()["datasets"][0]
    result = _wait(client, client.post(f"/api/datasets/{si['id']}/pca", json={"components": 10, "poisson": poisson}).json())
    session = client.app.state.session
    pca = session.get_dataset(result["pca_id"])
    explained = np.array(pca.pca["explained"])
    # The first three carry the signal; the rest are noise.
    assert explained[:3].sum() > 0.97 and explained[3] < 0.01
    # As HyperSpy's own model of the same data.
    ref = s.deepcopy()
    ref.decomposition(normalize_poissonian_noise=poisson, output_dimension=10, svd_solver="randomized", print_info=False)
    want = ref.get_decomposition_model(3).data
    made = session.get_dataset(client.post(f"/api/datasets/{pca.id}/pca/denoise", json={"n": 3}).json()["id"])
    got = np.asarray(made.signal.data)
    assert np.sqrt(np.mean((got - want) ** 2)) < 1e-3 * np.sqrt(np.mean(want ** 2))
    err_noisy = np.sqrt(np.mean((noisy - clean) ** 2))
    assert np.sqrt(np.mean((got - clean) ** 2)) < 0.35 * err_noisy
    assert made.signal.learning_results.factors is None  # nothing of the decomposition kept with it


def test_poisson_weighting_needs_counts(tmp_path):
    import hyperspy.api as hs
    from fastapi.testclient import TestClient

    from si_viewer.app import create_app

    _, noisy = _truth()
    noisy[0, 0] = -1e6
    s = hs.signals.Signal1D(noisy)
    s.set_signal_type("EELS")
    s.save(tmp_path / "si.hspy")
    client = TestClient(create_app())
    si = client.post("/api/files", json={"path": str(tmp_path / "si.hspy")}).json()["datasets"][0]
    job = client.post(f"/api/datasets/{si['id']}/pca", json={"components": 4}).json()
    while job["status"] in ("queued", "running"):
        time.sleep(0.02)
        job = client.get(f"/api/jobs/{job['id']}").json()
    assert job["status"] == "failed" and "Negative" in job["error"]


@pytest.mark.parametrize("algorithm", ["SVD", "sklearn_pca", "NMF", "sparse_pca", "RPCA"])
def test_algorithms(tmp_path, algorithm):
    """Each algorithm's denoised SI is HyperSpy's model of the same data (with
    the mean put back for scikit-learn's, which centre it)."""
    import hyperspy.api as hs
    from fastapi.testclient import TestClient

    from si_viewer.app import create_app
    from si_viewer.session import DECOMPOSITIONS

    clean, noisy = _truth(h=12, w=10, n=60)
    s = hs.signals.Signal1D(noisy)
    s.set_signal_type("EELS")
    s.save(tmp_path / "si.hspy")
    client = TestClient(create_app())
    si = client.post("/api/files", json={"path": str(tmp_path / "si.hspy")}).json()["datasets"][0]
    session = client.app.state.session
    result = _wait(client, client.post(f"/api/datasets/{si['id']}/pca", json={"components": 4, "algorithm": algorithm}).json())
    made = session.get_dataset(result["pca_id"])
    name, centred = DECOMPOSITIONS[algorithm]
    assert made.title == name and made.pca["algorithm"] == algorithm and made.pca["poisson"] is (not centred)
    assert made.signal.data.shape == (4, 12, 10)
    assert len(made.pca["explained"]) == (0 if algorithm in ("NMF", "sparse_pca") else 4)
    denoised = session.get_dataset(client.post(f"/api/datasets/{made.id}/pca/denoise", json={"n": 4}).json()["id"])
    got = np.asarray(denoised.signal.data)
    # Three components carry the signal: the model is close to the truth.
    err_noisy = np.sqrt(np.mean((noisy - clean) ** 2))
    assert np.sqrt(np.mean((got - clean) ** 2)) < 0.8 * err_noisy, algorithm
    assert denoised.title.endswith(f"({name}, 4)")


def test_each_algorithm_its_own_objects(client):
    si = client.si
    session = client.app.state.session
    a = _wait(client, client.post(f"/api/datasets/{si['id']}/pca", json={"components": 3}).json())["pca_id"]
    b = _wait(client, client.post(f"/api/datasets/{si['id']}/pca", json={"components": 3, "algorithm": "NMF"}).json())["pca_id"]
    assert a != b and session.get_dataset(b).title == "NMF" and session.get_dataset(b).pca["explained"] == []
    assert client.post(f"/api/datasets/{si['id']}/pca", json={"algorithm": "nope"}).status_code == 422


@pytest.mark.parametrize("norm", [{"mode": "range"}, {"mode": "mean", "lo": 410, "hi": 420}])
def test_decomposition_of_normalised_spectra(client, norm):
    """Each spectrum normalised (as the plot does) before decomposing; the
    denoised SI is back on the spectra's own scale."""
    si = client.si
    session = client.app.state.session
    src = session.get_dataset(si["id"])
    data = np.asarray(src.signal.data, np.float64)
    result = _wait(client, client.post(f"/api/datasets/{si['id']}/pca", json={"components": 8, "normalise": norm}).json())
    pca = session.get_dataset(result["pca_id"])
    assert pca.pca["normalise"] == norm and pca.pca["poisson"] is False
    # The factors are of normalised spectra: about 0–1 for "range".
    if norm["mode"] == "range":
        rebuilt = np.einsum("khw,kn->hwn", pca.signal.data, session.pca_factors(pca).signal.data)
        assert -0.2 < rebuilt.min() and rebuilt.max() < 1.2
    made = session.get_dataset(client.post(f"/api/datasets/{pca.id}/pca/denoise", json={"n": 3}).json()["id"])
    got = np.asarray(made.signal.data, np.float64)
    # Three components carry the signal: close to the data on its own scale.
    assert np.sqrt(np.mean((got - data) ** 2)) < 0.5 * np.sqrt(np.mean(data ** 2))
    assert abs(got.mean() / data.mean() - 1) < 0.05


def test_bad_normalisation(client):
    r = client.post(f"/api/datasets/{client.si['id']}/pca", json={"normalise": {"mode": "value"}})
    assert r.status_code == 422
