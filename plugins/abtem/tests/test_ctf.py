"""CTFs: settings in the project, and their evaluation by a long-running worker."""

import math
import stat
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

import si_viewer_abtem.plugin as abtem
from si_viewer.app import create_app
from si_viewer.project import open_project
from si_viewer.session import DatasetKind, Session
from si_viewer_abtem import engine
from si_viewer_abtem.ctfs import CtfSettings, worker_request

# Serves requests: echoes them, fails on energy 98 kV, stops on 99 kV.
FAKE_WORKER = r"""#!{python}
import json, sys
if sys.argv[-1] == "ping":
    print(json.dumps({{"result": {{"abtem": "1.0.11"}}}}))
    sys.exit()
assert sys.argv[-1] == "serve"
print(json.dumps({{"ready": True}}), flush=True)
for line in sys.stdin:
    request = json.loads(line)["request"]
    if request["energy"] == 99e3:
        sys.exit("stopped on purpose")
    if request["energy"] == 98e3:
        print(json.dumps({{"error": "failed on purpose"}}), flush=True)
        continue
    print("not json", flush=True)
    out = {{"request": request}}
    if request["image_points"]:
        import base64
        import numpy as np
        # [kx, ky], each value its defocus, so a redraw shows.
        array = np.full((4, 4), request["defocus"], np.float32)
        out.update(image=base64.b64encode(array.tobytes()).decode(), image_shape=[4, 4], image_step=[0.1, 0.1])
    print(json.dumps({{"result": out}}), flush=True)
"""


@pytest.fixture
def setup(tmp_path, monkeypatch):
    env = tmp_path / "abtem-env"
    monkeypatch.setenv(engine.ENV_VAR, str(env))
    engine._checked.clear()
    python = engine.python_path(env)
    python.parent.mkdir(parents=True)
    python.write_text(FAKE_WORKER.format(python=sys.executable))
    python.chmod(python.stat().st_mode | stat.S_IEXEC)
    (env / engine.READY).write_text("{}")
    client = TestClient(create_app(Session(), plugins=[abtem]))
    file = client.post("/api/plugins/abtem/projects", json={"path": str(tmp_path / "project")}).json()["file"]
    return client, file["id"], tmp_path / "project"


def test_units_for_abtem():
    r = worker_request(CtfSettings(energy_kv=200, cs_um=10, c5_mm=2, astigmatism_angle=90, profile_angle=180))
    assert r["energy"] == 200e3 and r["cs"] == 1e5 and r["c5"] == 2e7
    assert r["phi12"] == pytest.approx(math.pi / 2) and r["phi"] == pytest.approx(math.pi)
    assert r["semiangle"] is None


def test_image_matches_kx_and_ky():
    import base64

    from si_viewer_abtem.ctfs import to_signal

    array = np.zeros((4, 4), np.float32)
    array[3, 0] = 1  # largest kx, smallest ky
    result = {"image": base64.b64encode(array.tobytes()).decode(), "image_shape": [4, 4], "image_step": [0.5, 0.25]}
    signal = to_signal(result, "t")
    kx, ky = signal.axes_manager.signal_axes
    assert (kx.name, kx.units, kx.scale, kx.offset) == ("kx", "1/Å", 0.5, -1.0)
    assert (ky.name, ky.scale, ky.offset) == ("ky", 0.25, -0.5)
    # Rows are ky from the top (most negative first), columns kx.
    assert signal.data[0, 3] == 1 and signal.data.sum() == 1


def test_made_changed_and_kept(setup):
    client, file_id, project = setup
    r = client.post(f"/api/plugins/abtem/projects/{file_id}/ctfs", json={"defocus": 20}).json()
    ds = next(d for d in r["file"]["datasets"] if d["id"] == r["dataset_id"])
    assert ds["kind"] == "image" and ds["plugin"]["label"] == "CTF" and ds["parent_id"] is None
    assert ds["title"] == "CTF 300 kV" and ds["plugin"]["colormap"] == "rdbu"
    assert {"filters", "mask", "columns"} <= set(ds["plugin"]["hidden_processes"])
    session = client.app.state.session
    np.testing.assert_allclose(session.datasets[ds["id"]].signal.data, 20)
    saved_at = (project / "project.hspy").stat().st_mtime_ns

    # A redraw without saving: new data, the same file on disk.
    client.put(f"/api/plugins/abtem/ctfs/{ds['id']}?save=false", json={"defocus": 5})
    np.testing.assert_allclose(session.datasets[ds["id"]].signal.data, 5)
    assert (project / "project.hspy").stat().st_mtime_ns == saved_at

    changed = client.put(f"/api/plugins/abtem/ctfs/{ds['id']}", json={"defocus": -10, "semiangle": 15}).json()
    after = next(d for d in changed["file"]["datasets"] if d["id"] == ds["id"])
    assert after["plugin"]["settings"]["defocus"] == -10 and after["plugin"]["settings"]["semiangle"] == 15
    assert after["data_rev"] > ds["data_rev"]

    again = Session()
    open_project(again, project / "project.hspy")
    kept = again.datasets[ds["id"]]
    assert kept.kind is DatasetKind.IMAGE and kept.plugin["settings"]["semiangle"] == 15
    np.testing.assert_allclose(kept.signal.data, -10)


@pytest.mark.parametrize("bad", [{"energy_kv": 1}, {"semiangle": 0}, {"focal_spread": -1}, {"max_angle": 0}])
def test_bad_settings(setup, bad):
    client, file_id, _ = setup
    assert client.post(f"/api/plugins/abtem/projects/{file_id}/ctfs", json=bad).status_code == 422
    assert client.post("/api/plugins/abtem/ctfs/evaluate", json=bad).status_code == 422


def test_only_a_ctf_can_be_changed(setup):
    client, _, _ = setup
    assert client.put("/api/plugins/abtem/ctfs/nope", json={}).status_code == 404


def test_evaluated_by_one_worker_that_survives_errors(setup):
    client, _, _ = setup
    r = client.post("/api/plugins/abtem/ctfs/evaluate", json={"energy_kv": 200, "cs_um": 1})
    assert r.status_code == 200 and r.json()["request"]["cs"] == 1e4 and "image" in r.json()
    profile = client.post("/api/plugins/abtem/ctfs/evaluate?image=false", json={}).json()
    assert profile["request"]["image_points"] == 0 and "image" not in profile
    failed = client.post("/api/plugins/abtem/ctfs/evaluate", json={"energy_kv": 98})
    assert failed.status_code == 502 and "on purpose" in failed.json()["detail"]
    assert client.post("/api/plugins/abtem/ctfs/evaluate", json={}).status_code == 200
    # A worker that stops is started again on the next request.
    stopped = client.post("/api/plugins/abtem/ctfs/evaluate", json={"energy_kv": 99})
    assert stopped.status_code == 502 and "stopped on purpose" in stopped.json()["detail"]
    assert client.post("/api/plugins/abtem/ctfs/evaluate", json={}).status_code == 200


def test_needs_abtem(setup, tmp_path, monkeypatch):
    client, _, _ = setup
    monkeypatch.setenv(engine.ENV_VAR, str(tmp_path / "none"))
    assert client.post("/api/plugins/abtem/ctfs/evaluate", json={}).status_code == 409


@pytest.mark.skipif(not engine.python_path(engine.DEFAULT_ENV).exists(), reason="abTEM isn't installed")
def test_real_abtem_matches_the_phase_error(monkeypatch):
    """abTEM's CTF profile is -sin χ, χ = 2π/λ (C10 α²/2 + C30 α⁴/4), C10 = -defocus."""
    monkeypatch.setenv(engine.ENV_VAR, str(engine.DEFAULT_ENV))
    service = engine.Service()
    try:
        r = service.call("ctf", worker_request(CtfSettings(energy_kv=300, defocus=54.3, cs_um=10)))
    finally:
        service.close()
    lam = r["wavelength"]
    k = np.arange(len(r["curves"]["ctf"])) * r["k_step"]
    alpha = k * lam
    chi = 2 * np.pi / lam * (0.5 * -54.3 * alpha**2 + 0.25 * 1e5 * alpha**4)
    np.testing.assert_allclose(r["curves"]["ctf"], -np.sin(chi), atol=1e-4)
    assert r["scherzer_defocus"] == pytest.approx(math.sqrt(1.5 * 1e5 * lam), rel=1e-6)
    assert r["image_shape"] == [256, 256]
