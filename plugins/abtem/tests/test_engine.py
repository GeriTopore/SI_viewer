"""abTEM's environment: status, the install job, and never replacing other folders."""

import stat
import subprocess
import sys
import time

import pytest
from fastapi.testclient import TestClient

import si_viewer_abtem.plugin as abtem
from si_viewer.app import create_app
from si_viewer.session import Session
from si_viewer_abtem import engine


def _wait(cond, timeout=20.0):
    t0 = time.time()
    while not cond():
        assert time.time() - t0 < timeout
        time.sleep(0.01)


@pytest.fixture
def env(tmp_path, monkeypatch):
    path = tmp_path / "abtem-env"
    monkeypatch.setenv(engine.ENV_VAR, str(path))
    engine._checked.clear()
    return path


@pytest.fixture
def client(env):
    return TestClient(create_app(Session(), plugins=[abtem]))


def _fake_python(env, body):
    """An executable at the environment's python path running ``body``."""
    python = engine.python_path(env)
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text(f"#!{sys.executable}\n{body}\n")
    python.chmod(python.stat().st_mode | stat.S_IEXEC)
    return python


def test_not_installed(client, env):
    out = client.get("/api/plugins/abtem/engine").json()
    assert out["installed"] is False and out["path"] == str(env)
    assert out["version"] == engine.ABTEM_VERSION and out["job"] is None


def test_interrupted_install_does_not_count(client, env):
    _fake_python(env, "print('{\"result\": {\"abtem\": \"1\"}}')")
    assert client.get("/api/plugins/abtem/engine").json()["installed"] is False


def test_installed(client, env):
    _fake_python(env, "print('some library noise'); print('{\"result\": {\"abtem\": \"1.0.11\"}}')")
    (env / engine.READY).write_text("{}")
    out = client.get("/api/plugins/abtem/engine").json()
    assert out["installed"] is True and out["versions"] == {"abtem": "1.0.11"} and out["error"] is None


def test_broken_environment_is_reported(client, env):
    _fake_python(env, "import sys; sys.exit('no module named abtem')")
    (env / engine.READY).write_text("{}")
    out = client.get("/api/plugins/abtem/engine").json()
    assert out["installed"] is False and "no module named abtem" in out["error"]


def test_install_job(client, env, tmp_path, monkeypatch):
    script = tmp_path / "fake_setup.py"
    script.write_text(
        "import json, pathlib, sys\n"
        "print(json.dumps({'progress': 0.5, 'message': 'Downloading'}), flush=True)\n"
        f"pathlib.Path({str(env)!r}).mkdir()\n"
        f"pathlib.Path({str(env / engine.READY)!r}).write_text('{{}}')\n"
        "print(json.dumps({'result': {'abtem': '1.0.11'}}), flush=True)\n"
    )
    monkeypatch.setattr(engine, "setup_command", lambda: [sys.executable, "-I", str(script)])
    job = client.post("/api/plugins/abtem/engine/install").json()
    jobs = client.app.state.jobs
    _wait(lambda: not jobs.jobs[job["id"]].active)
    assert jobs.jobs[job["id"]].status == "done"


def test_second_install_is_refused(client, env, tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "setup_command", lambda: [sys.executable, "-c", "import time; time.sleep(30)"])
    job = client.post("/api/plugins/abtem/engine/install").json()
    assert client.post("/api/plugins/abtem/engine/install").status_code == 409
    assert client.get("/api/plugins/abtem/engine").json()["job"]["id"] == job["id"]
    client.post(f"/api/jobs/{job['id']}/cancel")
    jobs = client.app.state.jobs
    _wait(lambda: not jobs.jobs[job["id"]].active)
    assert jobs.jobs[job["id"]].status == "cancelled"


def test_setup_never_replaces_another_folder(tmp_path):
    folder = tmp_path / "mine"
    folder.mkdir()
    (folder / "keep.txt").write_text("mine")
    done = subprocess.run(
        [sys.executable, "-I", str(engine.SETUP), str(folder), "abtem==0", str(engine.WORKER)],
        capture_output=True, text=True,
    )
    assert done.returncode != 0 and "in use" in done.stderr
    assert [p.name for p in folder.iterdir()] == ["keep.txt"]


def test_worker_refuses_unknown_commands():
    done = subprocess.run([sys.executable, "-I", str(engine.WORKER), "nope"], capture_output=True, text=True)
    assert done.returncode == 2 and "Unknown" in done.stderr


def test_worker_does_not_import_the_app():
    text = "\n".join(p.read_text() for p in engine.WORKER_DIR.glob("*.py"))
    assert "import si_viewer" not in text and "from si_viewer" not in text
