"""Wavefunctions through depth, with a stand-in for the abTEM worker."""

import json
import stat
import sys
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

import si_viewer_abtem.plugin as abtem
from si_viewer.app import create_app
from si_viewer.project import open_project
from si_viewer.session import DatasetKind, Session
from si_viewer_abtem import engine

from test_projects import STO

FAKE_WORKER = r"""#!{python}
import json, pathlib, sys, time
import numpy as np
if sys.argv[-1] == "ping":
    print(json.dumps({{"result": {{"abtem": "1.0.11"}}}}))
    sys.exit()
request = json.load(open(sys.argv[-1]))
if sys.argv[-2] == "potential":
    np.save(request["output"], np.ones((2, 4, 3), np.float32))
    print(json.dumps({{"result": {{"path": request["output"], "shape": [2, 4, 3], "sampling": [0.1, 0.2], "slice_thickness": [1.0, 1.0], "abtem": "1.0.11"}}}}))
    sys.exit()
if request["energy"] == 99e3:
    sys.exit("worker failed on purpose")
if request["energy"] == 98e3:
    time.sleep(30)
nx, ny = 4 * request["tile"][0], 3 * request["tile"][1]
wave = np.ones((3, nx, ny), np.complex64) * (1 + 1j)
configs = request.get("phonon_configs", 0)
if configs:
    # [config, plane, x, y]; configuration i has |ψ|² = 2 (i + 1).
    wave = np.stack([wave * np.sqrt(i + 1) for i in range(configs)]).astype(np.complex64)
np.save(pathlib.Path(request["folder"]) / "wave.npy", wave)
probe = request["beam"] == "probe"
print(json.dumps({{"result": {{
    "shape": list(wave.shape), "configs": configs, "sampling": [0.1, 0.2], "extent": [nx * 0.1, ny * 0.2], "slice_thickness": 1.0,
    "position": request["position"] or [nx * 0.05, ny * 0.1] if probe else None,
    "intensity_scale": 0.25 if probe else 1.0, "abtem": "1.0.11",
}}}}))
"""


def _wait(client, job):
    jobs = client.app.state.jobs
    t0 = time.time()
    while jobs.jobs[job["id"]].active:
        assert time.time() - t0 < 20
        time.sleep(0.01)
    return client.get(f"/api/jobs/{job['id']}").json()


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
    cif = tmp_path / "SrTiO3.cif"
    cif.write_text(STO)
    file = client.post("/api/plugins/abtem/projects", json={"path": str(tmp_path / "project")}).json()["file"]
    structure = client.post(f"/api/plugins/abtem/projects/{file['id']}/structures", json={"path": str(cif)}).json()["dataset_id"]
    model = client.post(f"/api/plugins/abtem/structures/{structure}/models", json={"uvw": [0, 0, 1], "repeats": [2, 2, 2]}).json()["dataset_id"]
    built = _wait(client, client.post(f"/api/plugins/abtem/models/{model}/potential", json={}).json())
    return client, built["result"]["dataset_id"], tmp_path / "project"


def _url(potential):
    return f"/api/plugins/abtem/potentials/{potential}/wavefunction"


@pytest.mark.parametrize("bad", [
    {"beam": "nope"}, {"energy_kv": 1}, {"semiangle": 0}, {"tile": [1]}, {"tile": [0, 1]}, {"tile": [65, 1]},
    {"position": [1.0]},
])
def test_bad_settings(setup, bad):
    client, potential, _ = setup
    assert client.post(_url(potential), json=bad).status_code == 422


def test_convergence_beyond_the_sampling_is_refused(setup):
    client, potential, project = setup
    r = client.post(_url(potential), json={"beam": "probe", "semiangle": 200})
    assert r.status_code == 422 and "131 mrad" in r.json()["detail"]
    assert list((project / "runs").iterdir()) == []


def test_too_large_is_refused(setup, monkeypatch):
    client, potential, _ = setup
    monkeypatch.setattr(abtem, "MAX_STACK_BYTES", 1024)
    r = client.post(_url(potential), json={"tile": [64, 64]})
    assert r.status_code == 422 and "GB" in r.json()["detail"]


def test_probe_is_saved_as_stacks_through_depth(setup):
    client, potential, project = setup
    body = {"beam": "probe", "energy_kv": 200, "semiangle": 25, "defocus": 10, "tile": [2, 1], "position": [0.2, 0.3]}
    done = _wait(client, client.post(_url(potential), json=body).json())
    assert done["status"] == "done", done["error"]
    datasets = {d["id"]: d for d in done["result"]["file"]["datasets"]}
    shown = datasets[done["result"]["dataset_id"]]
    assert shown["plugin"]["label"] == "Intensity" and shown["kind"] == "stack"
    wave = datasets[shown["parent_id"]]
    assert wave["kind"] == "item" and wave["parent_id"] == potential
    assert wave["plugin"]["item"] == "wavefunction" and wave["title"] == "Probe 200 kV, 25 mrad, Δf 10 Å"
    assert wave["plugin"]["settings"]["planes"] == 3 and wave["plugin"]["settings"]["position"] == [0.2, 0.3]

    children = {d["plugin"]["label"]: d for d in datasets.values() if d["parent_id"] == wave["id"]}
    assert set(children) == {"Intensity", "Phase"}
    assert children["Intensity"]["plugin"]["stack"]["start"] == "last"
    assert children["Phase"]["plugin"]["stack"]["projections"] is False

    session = client.app.state.session
    # |1+i|² = 2, scaled so that the probe's total is 1.
    np.testing.assert_allclose(session.datasets[children["Intensity"]["id"]].signal.data, 0.5)
    np.testing.assert_allclose(session.datasets[children["Phase"]["id"]].signal.data, np.pi / 4)
    assert session.datasets[children["Phase"]["id"]].signal.data.shape == (3, 3, 8)

    folder = project / wave["plugin"]["settings"]["run"]
    assert sorted(p.name for p in folder.iterdir()) == ["manifest.json", "settings.json", "wave.npy"]
    assert np.iscomplexobj(np.load(folder / "wave.npy"))
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["mode"] == "wavefunction" and manifest["axes"]["order"] == ["plane", "x", "y"]

    again = Session()
    open_project(again, project / "project.hspy")
    assert again.datasets[wave["id"]].kind is DatasetKind.ITEM
    assert again.datasets[children["Phase"]["id"]].kind is DatasetKind.STACK


def _phonon_potential(client, potential, configs=3):
    model = client.app.state.session.datasets[potential].plugin["settings"]["model"]
    built = _wait(client, client.post(f"/api/plugins/abtem/models/{model}/potential", json={"phonon_configs": configs}).json())
    assert built["status"] == "done", built["error"]
    return built["result"]["dataset_id"]


def test_frozen_phonons_give_the_mean_intensity_and_the_first_phase(setup):
    client, potential, project = setup
    phonons = _phonon_potential(client, potential)
    body = {"beam": "probe", "energy_kv": 200, "semiangle": 25, "tile": [2, 1], "position": [0.2, 0.3]}
    done = _wait(client, client.post(_url(phonons), json=body).json())
    assert done["status"] == "done", done["error"]
    datasets = {d["id"]: d for d in done["result"]["file"]["datasets"]}
    wave = datasets[datasets[done["result"]["dataset_id"]]["parent_id"]]
    settings = wave["plugin"]["settings"]
    assert settings["configs"] == 3 and settings["planes"] == 3 and settings["gpts"] == [8, 3]
    children = {d["plugin"]["label"]: d for d in datasets.values() if d["parent_id"] == wave["id"]}
    assert children["Phase"]["title"] == "Wave phase (config 1)"
    assert children["Intensity"]["title"] == "Wave intensity (mean of 3 configs)"

    session = client.app.state.session
    # The mean of 2, 4 and 6 is 4, times the worker's intensity scale (0.25).
    np.testing.assert_allclose(session.datasets[children["Intensity"]["id"]].signal.data, 1.0, rtol=1e-5)
    np.testing.assert_allclose(session.datasets[children["Phase"]["id"]].signal.data, np.pi / 4, rtol=1e-5)
    assert session.datasets[children["Intensity"]["id"]].signal.data.shape == (3, 3, 8)

    # wave.npy is [config, plane, x, y]; the manifest says so.
    folder = project / settings["run"]
    assert np.load(folder / "wave.npy").shape == (3, 3, 8, 3)
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["axes"]["order"] == ["config", "plane", "x", "y"] and manifest["configs"] == 3


def test_without_phonons_the_wave_keeps_its_old_form(setup):
    client, potential, project = setup
    done = _wait(client, client.post(_url(potential), json={}).json())
    datasets = {d["id"]: d for d in done["result"]["file"]["datasets"]}
    wave = datasets[datasets[done["result"]["dataset_id"]]["parent_id"]]
    assert wave["plugin"]["settings"]["configs"] == 0 and wave["plugin"]["settings"]["gpts"] == [4, 3]
    assert np.load(project / wave["plugin"]["settings"]["run"] / "wave.npy").shape == (3, 4, 3)
    assert {d["title"] for d in datasets.values() if d["parent_id"] == wave["id"]} == {"Wave phase", "Wave intensity"}


def test_too_many_configurations_are_refused(setup, monkeypatch):
    client, potential, project = setup
    phonons = _phonon_potential(client, potential, configs=64)
    monkeypatch.setattr(abtem, "MAX_STACK_BYTES", 4000)
    r = client.post(_url(phonons), json={})
    assert r.status_code == 422 and "64 configs" in r.json()["detail"]
    assert list((project / "runs").iterdir()) == []


def test_plane_wave_keeps_unit_scale(setup):
    client, potential, _ = setup
    done = _wait(client, client.post(_url(potential), json={}).json())
    datasets = {d["id"]: d for d in done["result"]["file"]["datasets"]}
    shown = datasets[done["result"]["dataset_id"]]
    assert datasets[shown["parent_id"]]["title"] == "Plane wave 300 kV"
    np.testing.assert_allclose(client.app.state.session.datasets[shown["id"]].signal.data, 2)


def test_failed_run_adds_nothing(setup):
    client, potential, project = setup
    before = len(client.get("/api/files").json()[0]["datasets"])
    done = _wait(client, client.post(_url(potential), json={"energy_kv": 99}).json())
    assert done["status"] == "failed" and "on purpose" in done["error"]
    assert len(client.get("/api/files").json()[0]["datasets"]) == before
    assert all(p.name.startswith(".tmp-") for p in (project / "runs").iterdir())


def test_a_run_keeps_the_folder_of_one_under_way(setup):
    client, potential, project = setup
    job = client.post(_url(potential), json={"energy_kv": 98}).json()
    assert client.post(_url(potential), json={}).status_code == 409
    assert client.get(_url(potential)).json()["job"]["id"] == job["id"]
    running = [p for p in (project / "runs").iterdir() if p.name.startswith(".tmp-")]
    r = client.post(f"/api/plugins/abtem/potentials/{potential}/hrtem", json={"defocus": 1})
    assert r.status_code == 200 and all(p.is_dir() for p in running)
    client.post(f"/api/jobs/{r.json()['id']}/cancel")
    client.post(f"/api/jobs/{job['id']}/cancel")
    assert _wait(client, job)["status"] == "cancelled"
