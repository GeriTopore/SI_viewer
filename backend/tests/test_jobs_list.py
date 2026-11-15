"""The list of background jobs (the jobs flap): running ones, then the latest to finish."""

import threading
import time

from fastapi.testclient import TestClient

from si_viewer.app import create_app
from si_viewer.jobs import JobManager


def _wait(cond, timeout=5.0):
    t0 = time.time()
    while not cond():
        assert time.time() - t0 < timeout
        time.sleep(0.01)


def test_listing_keeps_running_and_three_recent():
    jobs = JobManager()
    gate = threading.Event()
    done = [jobs.submit("t", f"done {i}", lambda job: {}) for i in range(4)]
    failed = jobs.submit("t", "failed", lambda job: (_ for _ in ()).throw(ValueError("bad")))
    _wait(lambda: all(not j.active for j in [*done, failed]))
    running = jobs.submit("t", "running", lambda job: (gate.wait(5), {})[1])
    queued = jobs.submit("t", "queued", lambda job: {})
    _wait(lambda: running.status == "running")
    out = jobs.listing(3)
    assert [j.title for j in out[:2]] == ["running", "queued"]  # active first, oldest first
    assert [j.title for j in out[2:]] == ["failed", "done 3", "done 2"]  # newest finished first
    assert failed.error == "bad" and failed.finished >= failed.started
    gate.set()
    _wait(lambda: not queued.active)


def test_list_endpoint_leaves_results_out():
    client = TestClient(create_app())
    jobs = client.app.state.jobs
    job = jobs.submit("t", "with result", lambda job: {"big": list(range(10))})
    _wait(lambda: not job.active)
    listed = client.get("/api/jobs").json()
    assert listed[0]["title"] == "with result" and "result" not in listed[0]
    assert listed[0]["finished"] is not None
    assert client.get(f"/api/jobs/{job.id}").json()["result"] == {"big": list(range(10))}
