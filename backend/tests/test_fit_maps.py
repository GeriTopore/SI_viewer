import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from si_viewer.app import create_app
from si_viewer.jobs import JobManager
from si_viewer.processing.fit_maps import composition_maps


def _array(r) -> np.ndarray:
    assert r.status_code == 200, r.text
    shape = tuple(int(n) for n in r.headers["X-Array-Shape"].split(","))
    return np.frombuffer(r.content, "<f4").reshape(shape)


def _wait(client, job, timeout=120):
    t0 = time.time()
    while job["status"] in ("queued", "running"):
        assert time.time() - t0 < timeout, "job did not finish"
        time.sleep(0.05)
        job = client.get(f"/api/jobs/{job['id']}").json()
    return job


def test_composition_maps():
    i = {"A": np.array([[2.0, 0.0]]), "B": np.array([[2.0, -1.0]])}
    at = composition_maps(i, {"A": 1.0, "B": 3.0}, {"A": 10.0, "B": 10.0})
    assert at["A"][0, 0] == pytest.approx(25) and at["B"][0, 0] == pytest.approx(75)
    assert np.isnan(at["A"][0, 1])  # no signal (negative counts as 0)
    # Too few counts: masked.
    masked = composition_maps(i, {}, {"A": 10.0, "B": 10.0}, min_counts=5)
    assert np.isnan(masked["A"]).all()


def test_job_cancel():
    jobs = JobManager()
    started = threading.Event()

    def slow(job):
        started.set()
        for k in range(1000):
            job.report(k / 1000)
            time.sleep(0.01)
        return {}

    job = jobs.submit("test", "slow", slow)
    started.wait(5)
    jobs.cancel(job.id)
    for _ in range(200):
        if job.status == "cancelled":
            break
        time.sleep(0.01)
    assert job.status == "cancelled" and job.result is None


@pytest.fixture
def client_si():
    client = TestClient(create_app())
    entry = client.post("/api/examples/fept-si").json()
    si = next(d for d in entry["datasets"] if d["kind"] == "spectrum_image")
    return client, si


def _maps(file, origin):
    return {d["element"]: d for d in file["datasets"] if d["origin"] == origin}


def test_fit_maps_job(client_si):
    client, si = client_si
    body = {"quant": {"composition": True, "exclude": ["Cu"], "kfactors": {"Fe_Ka": 1.2, "Pt_La": 1.8}}}
    job = _wait(client, client.post(f"/api/datasets/{si['id']}/fit-maps", json=body).json())
    assert job["status"] == "done" and job["progress"] == 1, job
    file = job["result"]["file"]

    fitted = _maps(file, "fitted")
    assert set(fitted) == {"Cu", "Fe", "Pt"}
    assert fitted["Pt"]["title"] == "Pt Lα (fit)" and fitted["Pt"]["signal_shape"] == [64, 64]
    assert fitted["Pt"]["quantities"] == ["counts", "at%"] and fitted["Pt"]["quantity"] == "counts"
    assert fitted["Cu"]["quantities"] == ["counts"]  # Cu excluded
    counts_fe = _array(client.get(f"/api/datasets/{fitted['Fe']['id']}/frame"))

    # The switch shows at% on every map that has it (a new revision).
    r = client.put(f"/api/datasets/{si['id']}/fit-quantity", json={"quantity": "at%"})
    assert r.status_code == 200, r.text
    comp = _maps(r.json()["file"], "fitted")
    assert comp["Fe"]["quantity"] == "at%" and comp["Fe"]["title"] == "Fe at%"
    assert comp["Fe"]["revision"] > fitted["Fe"]["revision"] and comp["Cu"]["quantity"] == "counts"
    fe = _array(client.get(f"/api/datasets/{comp['Fe']['id']}/frame"))
    pt = _array(client.get(f"/api/datasets/{comp['Pt']['id']}/frame"))
    assert not np.allclose(np.nan_to_num(fe), counts_fe)
    total = fe + pt
    assert np.allclose(total[np.isfinite(total)], 100, atol=1e-3)
    # The film between particles (few counts) is masked by default.
    assert np.isnan(pt[0:6, 28:34]).mean() > 0.8
    # Pt-rich core vs Fe-rich shell of the particle at x=40, y=45 (r=13).
    assert np.nanmean(pt[43:47, 38:42]) > np.nanmean(pt[40:50, 50:52]) + 15

    # Running again replaces the previous results, still showing at%.
    job = _wait(client, client.post(f"/api/datasets/{si['id']}/fit-maps", json=body).json())
    file = job["result"]["file"]
    again = _maps(file, "fitted")
    assert set(again) == {"Cu", "Fe", "Pt"} and again["Fe"]["quantity"] == "at%"
    # Without composition there is no at%.
    job = _wait(client, client.post(f"/api/datasets/{si['id']}/fit-maps", json={"quant": {"composition": False}}).json())
    file = job["result"]["file"]
    assert all(d["quantities"] == ["counts"] for d in _maps(file, "fitted").values())
    assert client.put(f"/api/datasets/{si['id']}/fit-quantity", json={"quantity": "at%"}).status_code == 422

    # Deleting a fitted map leaves the element ticked (and its raw map).
    file = client.delete(f"/api/datasets/{_maps(file, 'fitted')['Fe']['id']}").json()["file"]
    si_now = next(d for d in file["datasets"] if d["id"] == si["id"])
    assert "Fe" in si_now["eds"]["elements"] and "Fe" in _maps(file, "computed")

    # Rebinning the spectrum image clears per-pixel results made at the old size.
    r = client.post(f"/api/datasets/{si['id']}/rebin", json={"factors": [2, 2, 1]}).json()
    assert _maps(r["file"], "fitted") == {}


def test_fit_maps_errors(client_si):
    client, si = client_si
    images = [d for d in client.get("/api/files").json()[0]["datasets"] if d["kind"] == "image"]
    assert client.post(f"/api/datasets/{images[0]['id']}/fit-maps", json={}).status_code == 400
    client.put(f"/api/datasets/{si['id']}/elements", json={"elements": []})
    assert client.post(f"/api/datasets/{si['id']}/fit-maps", json={}).status_code == 422
    assert client.get("/api/jobs/nope").status_code == 404
