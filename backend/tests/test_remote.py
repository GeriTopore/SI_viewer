"""Running remotely (R1): the token every request needs, a free port with a
session file, stopping when standard input closes, the in-memory limit."""

import json
import os
import re
import stat
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from fastapi.testclient import TestClient

import si_viewer
from si_viewer.app import create_app, set_in_memory_limit
from si_viewer.auth import cookie_name
from si_viewer.session import Session

TOKEN = "s3cret-token"
PAGE = {"accept": "text/html,application/xhtml+xml"}


def _client(token=TOKEN):
    return TestClient(create_app(Session(), token=token), follow_redirects=False)


def test_requests_need_the_token():
    c = _client()
    r = c.get("/api/health")
    assert r.status_code == 403 and "token" in r.json()["detail"]
    # A page without it: a page saying so, not the app.
    r = c.get("/", headers=PAGE)
    assert r.status_code == 403 and "token needed" in r.text
    assert c.get("/api/health", headers={"authorization": f"token {TOKEN}"}).status_code == 200
    assert c.get("/api/health", headers={"authorization": "token wrong"}).status_code == 403
    # A script may give it in the query (no redirect: it keeps no cookie).
    assert c.get(f"/api/health?token={TOKEN}").status_code == 200
    assert c.get("/api/health?token=wrong").status_code == 403


def test_a_page_with_the_token_sets_the_cookie_and_drops_it_from_the_address():
    c = _client()
    r = c.get(f"/?token={TOKEN}&x=1", headers=PAGE)
    assert r.status_code == 303 and r.headers["location"] == "/?x=1"
    cookie = r.headers["set-cookie"]
    assert cookie.startswith(f"{cookie_name(TOKEN)}={TOKEN};")
    assert "HttpOnly" in cookie and "SameSite=Strict" in cookie
    # With the cookie, everything is open (the client keeps it).
    assert c.get("/api/health").status_code == 200
    assert c.get("/api/files").status_code == 200
    # Another server's cookie (another token) doesn't count.
    other = _client("another-token")
    other.cookies.set(cookie_name(TOKEN), TOKEN)
    assert other.get("/api/health").status_code == 403
    assert cookie_name(TOKEN) != cookie_name("another-token")


def test_the_tab_icon_needs_no_token():
    assert _client().get("/favicon.ico").status_code != 403


def test_no_token_is_open():
    assert _client(None).get("/api/health").status_code == 200


def test_in_memory_limit():
    from si_viewer import session as session_mod
    from si_viewer.io import py4dstem
    from si_viewer.processing import frames

    old = (session_mod.SI_IN_MEMORY_LIMIT, session_mod.DIFFRACTION4D_IN_MEMORY_LIMIT, py4dstem.IN_MEMORY_LIMIT,
           frames.STACK_IN_MEMORY_LIMIT)
    try:
        set_in_memory_limit(64 << 30)
        assert session_mod.SI_IN_MEMORY_LIMIT == session_mod.DIFFRACTION4D_IN_MEMORY_LIMIT == 64 << 30
        assert py4dstem.IN_MEMORY_LIMIT == 64 << 30 and frames.STACK_IN_MEMORY_LIMIT == 16 << 30
    finally:
        (session_mod.SI_IN_MEMORY_LIMIT, session_mod.DIFFRACTION4D_IN_MEMORY_LIMIT, py4dstem.IN_MEMORY_LIMIT,
         frames.STACK_IN_MEMORY_LIMIT) = old


def test_server_on_a_free_port_with_a_session_file_stops_when_stdin_closes(tmp_path):
    """As a launcher starts it over SSH: ``--port 0 --exit-on-eof``."""
    backend = Path(si_viewer.__file__).resolve().parents[1]
    env = {**os.environ, "HOME": str(tmp_path), "PYTHONPATH": str(backend)}
    env.pop("SI_VIEWER_TOKEN", None)
    proc = subprocess.Popen(
        [sys.executable, "-m", "si_viewer.app", "--no-browser", "--port", "0", "--exit-on-eof"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, text=True,
    )
    try:
        url = None
        deadline = time.time() + 90
        while time.time() < deadline and url is None:
            line = proc.stdout.readline()
            m = re.search(r"SI Viewer running at (\S+)", line)
            if m:
                url = m.group(1)
        assert url, "no address printed"
        m = re.match(r"http://127\.0\.0\.1:(\d+)/\?token=(\S+)$", url)
        assert m
        port, token = int(m.group(1)), m.group(2)
        assert port != 0 and len(token) >= 24

        session_file = tmp_path / ".si-viewer" / "sessions" / f"{proc.pid}.json"
        info = json.loads(session_file.read_text())
        assert info["port"] == port and info["token"] == token and info["url"] == url
        # Only its user can read it (it holds the token).
        assert stat.S_IMODE(session_file.stat().st_mode) == 0o600
        assert stat.S_IMODE(session_file.parent.stat().st_mode) == 0o700

        def get(path, auth=True):
            req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                         headers={"Authorization": f"token {token}"} if auth else {})
            try:
                with urllib.request.urlopen(req, timeout=5) as r:
                    return r.status
            except urllib.error.HTTPError as e:
                return e.code

        for _ in range(100):
            try:
                status = get("/api/health")
                break
            except OSError:
                time.sleep(0.2)
        assert status == 200
        assert get("/api/health", auth=False) == 403

        # The connection that started it ends: it stops and its file goes.
        proc.stdin.close()
        proc.wait(timeout=30)
        assert not session_file.exists()
    finally:
        if proc.poll() is None:
            proc.kill()


def test_a_new_session_file_clears_those_of_servers_gone(tmp_path, monkeypatch):
    import si_viewer.app as app_mod

    monkeypatch.setattr(app_mod, "SESSIONS_DIR", tmp_path)
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    (tmp_path / f"{gone.pid}.json").write_text("{}")
    (tmp_path / f"{os.getppid()}.json").write_text("{}")
    (tmp_path / "notes.json").write_text("{}")
    app_mod._write_session_file({"pid": os.getpid()})
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [f"{os.getpid()}.json", f"{os.getppid()}.json", "notes.json"])


# ---------- R2: the launcher, ``si-viewer remote HOST`` ----------

FAKE_SSH = r'''#!/usr/bin/env python3
"""A stand-in ssh: runs the command here; control commands (-O) succeed."""
import os, sys
args = sys.argv[1:]
pos, i = [], 0
while i < len(args):
    a = args[i]
    if a in ("-o", "-L", "-O"):
        if a == "-O":
            sys.exit(0)
        i += 2
        continue
    if a.startswith("-"):
        i += 1
        continue
    pos.append(a)
    i += 1
os.execvp("sh", ["sh", "-c", " ".join(pos[1:])])
'''


def test_the_server_command_and_its_address_line():
    from si_viewer.remote import parse_ready, server_command

    line = server_command("~/env/bin/si-viewer", 64, ["/data/a b.h5"])
    assert line == "exec env ~/env/bin/si-viewer --no-browser --port 0 --exit-on-eof --in-memory 64 '/data/a b.h5'"
    assert parse_ready("SI Viewer running at http://127.0.0.1:40123/?token=abc_-1\n") == ("127.0.0.1", 40123, "abc_-1")
    assert parse_ready("SI Viewer running at http://127.0.0.1:5/\n") == ("127.0.0.1", 5, None)
    assert parse_ready("INFO: something else") is None


def _fake_ssh(tmp_path):
    path = tmp_path / "ssh"
    path.write_text(FAKE_SSH)
    path.chmod(0o755)
    return str(path)


def test_launcher_starts_the_server_and_stops_it(tmp_path, monkeypatch):
    import io

    from si_viewer import remote

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("SI_VIEWER_TOKEN", raising=False)
    monkeypatch.setattr(remote, "LOG_DIR", tmp_path / ".si-viewer")
    backend = Path(si_viewer.__file__).resolve().parents[1]
    out, log = io.StringIO(), io.StringIO()
    r = remote.Remote("somehost", f"PYTHONPATH={backend} {sys.executable} -m si_viewer.app",
                      ssh=_fake_ssh(tmp_path), open_browser=False, log=log, out=out)
    try:
        url = r.start()
        m = re.match(r"http://localhost:(\d+)/\?token=(\S+)$", url)
        assert m and r.remote_port and int(m.group(1)) != 0
        token = m.group(2)
        req = urllib.request.Request(f"http://127.0.0.1:{r.remote_port}/api/health",
                                     headers={"Authorization": f"token {token}"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 200
        assert "SI Viewer on somehost" in out.getvalue() and "SI Viewer running at" in log.getvalue()
        sessions = list((tmp_path / ".si-viewer" / "sessions").glob("*.json"))
        assert len(sessions) == 1
    finally:
        r.stop()
    assert r.proc.poll() is not None
    assert not list((tmp_path / ".si-viewer" / "sessions").glob("*.json"))


def test_launcher_reports_a_server_that_fails_to_start(tmp_path, monkeypatch):
    import io

    import pytest

    from si_viewer import remote

    monkeypatch.setattr(remote, "LOG_DIR", tmp_path / ".si-viewer")
    r = remote.Remote("somehost", "/no/such/si-viewer", ssh=_fake_ssh(tmp_path), open_browser=False,
                      log=io.StringIO(), out=io.StringIO())
    with pytest.raises(RuntimeError, match="stopped before starting"):
        r.start()
    r.stop()


# ---------- installing there: ``si-viewer remote HOST --install`` ----------


def test_the_environment_keeps_only_conda_packages(tmp_path):
    import yaml

    from si_viewer.remote_install import conda_environment

    env = tmp_path / "environment.yml"
    env.write_text("name: dev\nchannels: [conda-forge]\ndependencies:\n  - python=3.12\n  - hyperspy>=2.4\n"
                   "  - pip\n  - pip:\n      - -e .\n      - -e ../kemstem\n")
    got = yaml.safe_load(conda_environment(env, "si-viewer"))
    assert got == {"name": "si-viewer", "channels": ["conda-forge"], "dependencies": ["python=3.12", "hyperspy>=2.4", "pip"]}


def _stub(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(0o755)


def test_the_install_script_creates_then_keeps_then_updates_the_environment(tmp_path):
    """``install.sh`` against stand-ins for conda, mamba and the env's python."""
    from si_viewer.remote_install import install_script

    home = tmp_path / "home"
    prefix = home / "mf"
    log = tmp_path / "calls.log"
    py = prefix / "envs" / "si-viewer" / "bin" / "python"
    _stub(prefix / "bin" / "conda", "exit 0\n")
    # ``env create`` makes the environment's python; it logs its calls.
    _stub(prefix / "bin" / "mamba",
          f'echo "mamba $*" >> {log}\nif [ "$2" = create ]; then mkdir -p {py.parent}; '
          f'printf \'#!/usr/bin/env bash\\necho "python $*" >> {log}\\n\' > {py}; chmod +x {py}; fi\n')
    d = tmp_path / "install"
    (d / "wheels").mkdir(parents=True)
    (d / "wheels" / "si_viewer-0.1.0-py3-none-any.whl").write_bytes(b"")
    (d / "environment.yml").write_text("name: si-viewer\ndependencies: [python=3.12]\n")
    (d / "build.json").write_text('{"commit": "abc"}')
    (d / "install.sh").write_text(install_script("~/mf"))
    (home / ".si-viewer").mkdir(parents=True)

    def run():
        r = subprocess.run(["bash", str(d / "install.sh")], env={**os.environ, "HOME": str(home)},
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        return r.stdout

    out = run()
    assert "Creating the conda environment si-viewer" in out
    calls = log.read_text()
    assert "mamba env create -y -q -n si-viewer" in calls
    assert "--force-reinstall --no-deps" in calls and calls.count("python -m pip install") == 2
    assert json.loads((home / ".si-viewer" / "installed.json").read_text()) == {"commit": "abc"}
    # Again, the packages unchanged: the environment is left alone.
    log.write_text("")
    assert "up to date" in run() and "mamba" not in log.read_text()
    # Its packages changed: updated.
    (d / "environment.yml").write_text("name: si-viewer\ndependencies: [python=3.12, numba]\n")
    assert "Updating the conda environment" in run() and "mamba env update -q -n si-viewer" in log.read_text()


def test_the_bundle_holds_the_app_with_its_frontend(tmp_path):
    import io
    import tarfile
    import zipfile

    import pytest

    from si_viewer.remote_install import app_root, bundle

    if not (app_root() / "backend" / "si_viewer" / "static" / "index.html").is_file():
        pytest.skip("frontend not built")
    with tarfile.open(fileobj=io.BytesIO(bundle("~/mf", say=lambda *_: None))) as tar:
        names = tar.getnames()
        assert {"environment.yml", "install.sh", "build.json"} <= set(names)
        assert "-e " not in tar.extractfile("environment.yml").read().decode()
        assert tar.getmember("install.sh").mode & 0o111
        wheel = next(n for n in names if n.startswith("wheels/si_viewer-"))
        with zipfile.ZipFile(io.BytesIO(tar.extractfile(wheel).read())) as z:
            inside = set(z.namelist())
    assert "si_viewer/static/index.html" in inside and "si_viewer/remote.py" in inside and "si_viewer/auth.py" in inside
