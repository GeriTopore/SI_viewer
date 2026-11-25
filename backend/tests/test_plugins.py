"""The plugin host: discovery, isolation, routes, process jobs, saved results."""

import sys
import time
import types
from pathlib import Path

import hyperspy.api as hs
import numpy as np
import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from si_viewer.app import create_app
from si_viewer.jobs import JobManager
from si_viewer.plugins import API_VERSION, PluginManifest
from si_viewer.project import open_project, save_project
from si_viewer.session import Session

sys.path.insert(0, str(Path(__file__).parent / "plugin_fixture"))
import si_viewer_test_plugin as test_plugin  # noqa: E402


def _wait(cond, timeout=20.0):
    t0 = time.time()
    while not cond():
        assert time.time() - t0 < timeout
        time.sleep(0.01)


def _plugin(id="demo", register=lambda ctx: None, **manifest):
    return types.SimpleNamespace(manifest=PluginManifest(id=id, name=id, version="1", **manifest), register=register)


@pytest.fixture
def image_session(tmp_path):
    path = tmp_path / "image.hspy"
    signal = hs.signals.Signal2D(np.arange(64.0).reshape(8, 8))
    signal.save(path)
    session = Session()
    entry = session.open_file(path)
    return session, session.datasets[entry.dataset_ids[0]]


def test_no_plugins():
    assert TestClient(create_app(plugins=[])).get("/api/plugins").json() == []


def test_broken_plugins_do_not_stop_the_app():
    def boom(ctx):
        router = APIRouter()
        router.get("/left-behind")(lambda: {})
        ctx.add_router(router)
        raise RuntimeError("boom")

    broken = _plugin("broken", boom)
    old = _plugin("old", api_version=API_VERSION + 1)
    no_manifest = types.SimpleNamespace(register=lambda ctx: None)
    client = TestClient(create_app(plugins=[broken, old, no_manifest, _plugin("fine")]))
    listed = client.get("/api/plugins").json()
    assert [p["status"] for p in listed] == ["failed", "incompatible", "failed", "active"]
    assert listed[0]["error"] == "boom"
    assert client.get("/api/health").status_code == 200
    assert not any(getattr(r, "path", "") == "/api/plugins/broken/left-behind" for r in client.app.routes)


def test_duplicate_id_is_refused():
    listed = TestClient(create_app(plugins=[_plugin("same"), _plugin("same")])).get("/api/plugins").json()
    assert [p["status"] for p in listed] == ["active", "failed"]


def test_routes_and_frontend_are_served():
    client = TestClient(create_app(plugins=[test_plugin]))
    [record] = client.get("/api/plugins").json()
    assert record["status"] == "active" and record["frontend"] == "/plugins/test-plugin/index.js"
    r = client.get(record["frontend"])
    assert r.status_code == 200 and "export function activate" in r.text
    # Rebuilt under the same name: browsers must revalidate it.
    assert r.headers["cache-control"] == "no-cache"
    assert client.get(record["frontend"], headers={"If-None-Match": r.headers["etag"]}).status_code == 304
    assert client.post("/api/plugins/test-plugin/datasets/nope/run", json={}).status_code == 404


def test_process_job_makes_a_dataset_that_is_saved(image_session, tmp_path):
    session, image = image_session
    client = TestClient(create_app(session, plugins=[test_plugin]))
    job = client.post(f"/api/plugins/test-plugin/datasets/{image.id}/run", json={"steps": 3, "delay": 0}).json()
    jobs = client.app.state.jobs
    _wait(lambda: not jobs.jobs[job["id"]].active)
    done = client.get(f"/api/jobs/{job['id']}").json()
    assert done["status"] == "done" and done["progress"] == 1 and done["message"] == "Step 3 of 3"
    made = session.datasets[done["result"]["dataset_id"]]
    assert made.origin == "plugin:test-plugin" and made.parent_id == image.id
    assert made.plugin == {"id": "test-plugin", "settings": {"steps": 3, "delay": 0.0, "fail": False}}
    assert made.summary()["plugin"]["id"] == "test-plugin"

    path = tmp_path / "analysis.hspy"
    save_project(session, path, {})
    reopened = Session()
    open_project(reopened, path)
    again = reopened.datasets[made.id]
    assert again.plugin == made.plugin and again.origin == made.origin
    np.testing.assert_array_equal(again.signal.data, made.signal.data)


def test_process_failure_is_reported():
    jobs = JobManager()
    job = jobs.submit_process("t", "fails", [sys.executable, "-c", "import sys; sys.exit('went wrong')"])
    _wait(lambda: not job.active)
    assert job.status == "failed" and job.error == "went wrong"


def test_cancel_kills_the_process_and_its_children():
    jobs = JobManager()
    code = (
        "import json,subprocess,sys,time;"
        "c=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']);"
        "print(json.dumps({'progress':.5,'message':str(c.pid)}),flush=True);time.sleep(60)"
    )
    job = jobs.submit_process("t", "long", [sys.executable, "-c", code])
    _wait(lambda: job.message != "")
    child = int(job.message)
    t0 = time.time()
    jobs.cancel(job.id)
    _wait(lambda: not job.active)
    assert job.status == "cancelled" and time.time() - t0 < 5

    def gone():
        try:
            import os

            os.kill(child, 0)
            return False
        except ProcessLookupError:
            return True

    _wait(gone)


def test_process_jobs_do_not_block_normal_jobs():
    jobs = JobManager()
    long = jobs.submit_process("t", "long", [sys.executable, "-c", "import time; time.sleep(30)"])
    _wait(lambda: long.status == "running")
    quick = jobs.submit("t", "quick", lambda job: {"ok": True})
    _wait(lambda: not quick.active, timeout=5)
    assert quick.status == "done" and long.status == "running"
    jobs.cancel(long.id)
    _wait(lambda: not long.active)
