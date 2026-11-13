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
