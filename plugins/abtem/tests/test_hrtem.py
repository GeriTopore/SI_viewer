"""HRTEM runs and the images CTFs form of them, with a stand-in for the abTEM worker."""

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
from si_viewer_abtem.runs import max_semiangle, wavelength

from test_projects import STO

# The image of a wave (1 + 0.5i everywhere): |ψ|² = 1.25, times 2 with an
# envelope, 3 for a focal series, so tests can tell what was asked.
FAKE_WORKER = r"""#!{python}
import base64, json, pathlib, sys, time
import numpy as np
if sys.argv[-1] == "ping":
    print(json.dumps({{"result": {{"abtem": "1.0.11"}}}}))
    sys.exit()
if sys.argv[-1] == "serve":
    print(json.dumps({{"ready": True}}), flush=True)
    for line in sys.stdin:
        message = json.loads(line)
        request = message["request"]
        if message["command"] == "ctf":
            array = np.zeros((4, 4), np.float32)
            out = {{"image": base64.b64encode(array.tobytes()).decode(), "image_shape": [4, 4], "image_step": [0.1, 0.1]}}
        else:
            wave = np.load(request["wave"])
            factor = 3 if request["coherence"] == "focal-series" else 2
            configs = wave.shape[0] if wave.ndim == 3 else 0
            # A leading axis of configurations: their images' mean.
            image = (np.abs(wave) ** 2 * factor).astype(np.float32)
            image = image.mean(axis=0) if configs else image
            out = {{"image": base64.b64encode(image.tobytes()).decode(), "shape": list(image.shape), "configs": configs,
                    "mean": float(image.mean()), "request": request}}
        print(json.dumps({{"result": out}}), flush=True)
    sys.exit()
request = json.load(open(sys.argv[-1]))
if sys.argv[-2] == "potential":
    np.save(request["output"], np.ones((2, 4, 3), np.float32))
    print(json.dumps({{"result": {{"path": request["output"], "shape": [2, 4, 3], "sampling": [0.1, 0.2], "slice_thickness": [1.0, 1.0], "abtem": "1.0.11"}}}}))
    sys.exit()
folder = pathlib.Path(request["folder"])
if request["energy"] == 99e3:
    sys.exit("worker failed on purpose")
if request["energy"] == 98e3:
    print(json.dumps({{"progress": 0.2, "message": "Running"}}), flush=True)
    time.sleep(30)
pathlib.Path(request["model"]).read_text()
wave = (1 + 0.5j) * np.ones((4, 3), np.complex64)
configs = request.get("phonon_configs", 0)
if configs:
    # Configuration i has |ψ|² = 1.25 (i + 1).
    wave = np.stack([wave * np.sqrt(i + 1) for i in range(configs)]).astype(np.complex64)
np.save(folder / "exit_wave.npy", wave)
print(json.dumps({{"result": {{"shape": list(wave.shape), "configs": configs, "sampling": [0.1, 0.2], "slices": 8, "exit_intensity": float((np.abs(wave) ** 2).mean()), "abtem": "1.0.11"}}}}))
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
    return client, built["result"]["dataset_id"], tmp_path / "project", file["id"]


def _url(potential):
    return f"/api/plugins/abtem/potentials/{potential}/hrtem"


def _ctf(client, file_id, **settings):
    return client.post(f"/api/plugins/abtem/projects/{file_id}/ctfs", json=settings).json()["dataset_id"]


def _by_id(done):
    return {d["id"]: d for d in done["result"]["file"]["datasets"]}


def test_wavelength_and_aperture_limit():
    assert wavelength(300) == pytest.approx(0.0196875, abs=1e-6)
    assert wavelength(200) == pytest.approx(0.0250793, abs=1e-6)
    assert max_semiangle(300, 0.05) == pytest.approx(131.25, abs=0.05)


def test_only_a_potential_can_be_simulated(setup):
    client, potential, _, _ = setup
    datasets = client.get("/api/files").json()[0]["datasets"]
    model = next(d for d in datasets if d["id"] != potential and d["plugin"].get("item") == "model")
    assert client.post(_url(model["id"]), json={}).status_code == 404


@pytest.mark.parametrize("bad", [{"energy_kv": 1}, {"coherence": "nope"}, {"ctf": "nope"}])
def test_bad_settings(setup, bad):
    client, potential, _, _ = setup
    assert client.post(_url(potential), json=bad).status_code in (404, 422)


def test_ctf_aperture_beyond_the_sampling_is_refused(setup):
    client, potential, project, file_id = setup
    ctf = _ctf(client, file_id, semiangle=200)
    r = client.post(_url(potential), json={"ctf": ctf})
    assert r.status_code == 422 and "131 mrad" in r.json()["detail"]
    assert list((project / "runs").iterdir()) == []


def test_needs_abtem(setup, tmp_path, monkeypatch):
    client, potential, _, _ = setup
    monkeypatch.setenv(engine.ENV_VAR, str(tmp_path / "none"))
    assert client.post(_url(potential), json={}).status_code == 409


def test_run_without_a_ctf_keeps_the_exit_wave(setup):
    client, potential, project, _ = setup
    done = _wait(client, client.post(_url(potential), json={"energy_kv": 200}).json())
    assert done["status"] == "done", done["error"]
    datasets = _by_id(done)
    shown = datasets[done["result"]["dataset_id"]]
    run = datasets[shown["parent_id"]]
    assert run["kind"] == "item" and run["parent_id"] == potential and run["plugin"]["item"] == "run"
    assert run["title"] == "HRTEM 200 kV" and run["plugin"]["settings"]["energy_kv"] == 200
    assert {d["plugin"]["label"] for d in datasets.values() if d["parent_id"] == run["id"]} == {"Amplitude", "Phase"}
    folder = project / run["plugin"]["settings"]["run"]
    assert sorted(p.name for p in folder.iterdir()) == ["exit_wave.npy", "manifest.json", "settings.json"]
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["mode"] == "hrtem" and manifest["axes"]["order"] == ["x", "y"]


def test_run_with_a_ctf_forms_its_image(setup):
    client, potential, project, file_id = setup
    ctf = _ctf(client, file_id, energy_kv=200, defocus=30)
    done = _wait(client, client.post(_url(potential), json={"ctf": ctf, "energy_kv": 300}).json())
    assert done["status"] == "done", done["error"]
    datasets = _by_id(done)
    image = datasets[done["result"]["dataset_id"]]
    run = datasets[image["parent_id"]]
    # The CTF's energy, not the one sent.
    assert run["title"] == "HRTEM 200 kV" and image["plugin"]["label"] == "Image" and image["kind"] == "image"
    assert image["title"] == "Image · CTF 200 kV, Δf 30 Å"
    settings = image["plugin"]["settings"]
    assert settings["ctf"] == ctf and settings["ctf_settings"]["defocus"] == 30 and settings["coherence"] == "envelope"
    session = client.app.state.session
    np.testing.assert_allclose(session.datasets[image["id"]].signal.data, 2.5)  # |1 + 0.5i|² × 2
    assert session.datasets[image["id"]].signal.data.shape == (3, 4)

    again = Session()
    open_project(again, project / "project.hspy")
    assert again.datasets[image["id"]].kind is DatasetKind.IMAGE
    np.testing.assert_allclose(again.datasets[image["id"]].signal.data, 2.5)


def test_focal_series_only_with_a_focal_spread(setup):
    client, potential, _, file_id = setup
    plain = _ctf(client, file_id)
    spread = _ctf(client, file_id, focal_spread=30)
    first = _wait(client, client.post(_url(potential), json={"ctf": plain, "coherence": "focal-series"}).json())
    assert _by_id(first)[first["result"]["dataset_id"]]["plugin"]["settings"]["coherence"] == "envelope"
    run = _by_id(first)[first["result"]["dataset_id"]]["parent_id"]
    r = client.post(f"/api/plugins/abtem/runs/{run}/images", json={"ctf": spread, "coherence": "focal-series"}).json()
    image = next(d for d in r["file"]["datasets"] if d["id"] == r["dataset_id"])
    assert image["title"].endswith(", focal series") and image["parent_id"] == run
    np.testing.assert_allclose(client.app.state.session.datasets[image["id"]].signal.data, 3.75)


def test_apply_ctf_to_a_finished_run(setup):
    client, potential, project, file_id = setup
    done = _wait(client, client.post(_url(potential), json={}).json())
    run = _by_id(done)[done["result"]["dataset_id"]]["parent_id"]
    ctf = _ctf(client, file_id, defocus=-20, semiangle=20)
    r = client.post(f"/api/plugins/abtem/runs/{run}/images", json={"ctf": ctf})
    assert r.status_code == 200
    image = next(d for d in r.json()["file"]["datasets"] if d["id"] == r.json()["dataset_id"])
    assert image["parent_id"] == run and image["plugin"]["settings"]["ctf_settings"]["semiangle"] == 20
    again = Session()
    open_project(again, project / "project.hspy")
    assert image["id"] in again.datasets

    other = _ctf(client, file_id, energy_kv=80)
    wrong = client.post(f"/api/plugins/abtem/runs/{run}/images", json={"ctf": other})
    assert wrong.status_code == 422 and "80 kV" in wrong.json()["detail"]
    assert client.post(f"/api/plugins/abtem/runs/{potential}/images", json={"ctf": ctf}).status_code == 404
    assert client.post(f"/api/plugins/abtem/runs/{run}/images", json={"ctf": potential}).status_code == 404


def test_project_ctfs_are_listed(setup):
    client, _, _, file_id = setup
    a = _ctf(client, file_id, defocus=10)
    b = _ctf(client, file_id, energy_kv=80)
    listed = client.get(f"/api/plugins/abtem/projects/{file_id}/ctfs").json()
    assert [(c["id"], c["settings"]["energy_kv"]) for c in listed] == [(a, 300), (b, 80)]


def _phonon_potential(client, potential, configs=3):
    """A potential of the same model with frozen phonons."""
    model = client.app.state.session.datasets[potential].plugin["settings"]["model"]
    built = _wait(client, client.post(f"/api/plugins/abtem/models/{model}/potential", json={"phonon_configs": configs}).json())
    assert built["status"] == "done", built["error"]
    return built["result"]["dataset_id"]


def test_frozen_phonons_keep_a_wave_per_configuration(setup):
    client, potential, project, file_id = setup
    phonons = _phonon_potential(client, potential)
    ctf = _ctf(client, file_id, defocus=30)
    done = _wait(client, client.post(_url(phonons), json={"ctf": ctf}).json())
    assert done["status"] == "done", done["error"]
    datasets = _by_id(done)
    image = datasets[done["result"]["dataset_id"]]
    run = datasets[image["parent_id"]]
    settings = run["plugin"]["settings"]
    assert settings["configs"] == 3 and settings["gpts"] == [4, 3] and settings["potential"]["phonon_configs"] == 3
    assert settings["exit_intensity"] == pytest.approx(1.25 * 2)  # the mean over configurations

    # exit_wave.npy is [config, x, y]; the manifest says so.
    folder = project / settings["run"]
    assert np.load(folder / "exit_wave.npy").shape == (3, 4, 3)
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["axes"]["order"] == ["config", "x", "y"] and manifest["configs"] == 3

    # The exit wave shown is the first configuration's, and says so.
    shown = {d["plugin"]["label"]: d for d in datasets.values() if d["parent_id"] == run["id"]}
    assert shown["Phase"]["title"] == "Exit wave phase (config 1)" and shown["Amplitude"]["title"] == "Exit wave amplitude (config 1)"
    session = client.app.state.session
    np.testing.assert_allclose(session.datasets[shown["Amplitude"]["id"]].signal.data, np.abs(1 + 0.5j), rtol=1e-6)
    assert session.datasets[shown["Phase"]["id"]].signal.data.shape == (3, 4)

    # The image is the mean of the configurations' images: |ψ|² = 1.25 × (1, 2, 3), × 2.
    assert shown["Image"]["title"] == "Image · CTF 300 kV, Δf 30 Å, 3 phonon configs"
    assert shown["Image"]["plugin"]["settings"]["configs"] == 3
    np.testing.assert_allclose(session.datasets[shown["Image"]["id"]].signal.data, 1.25 * 2 * 2, rtol=1e-6)

    # Apply CTF to the finished run averages them too.
    other = _ctf(client, file_id, defocus=-20)
    r = client.post(f"/api/plugins/abtem/runs/{run['id']}/images", json={"ctf": other})
    assert r.status_code == 200
    np.testing.assert_allclose(session.datasets[r.json()["dataset_id"]].signal.data, 1.25 * 2 * 2, rtol=1e-6)

    again = Session()
    open_project(again, project / "project.hspy")
    assert again.datasets[run["id"]].plugin["settings"]["configs"] == 3


def test_runs_without_phonons_keep_their_old_form(setup):
    client, potential, project, _ = setup
    done = _wait(client, client.post(_url(potential), json={}).json())
    run = _by_id(done)[_by_id(done)[done["result"]["dataset_id"]]["parent_id"]]
    assert run["plugin"]["settings"]["configs"] == 0 and run["plugin"]["settings"]["gpts"] == [4, 3]
    folder = project / run["plugin"]["settings"]["run"]
    assert np.load(folder / "exit_wave.npy").shape == (4, 3)
    titles = {d["title"] for d in _by_id(done).values() if d["parent_id"] == run["id"]}
    assert titles == {"Exit wave phase", "Exit wave amplitude"}


def test_too_many_configurations_are_refused(setup, monkeypatch):
    client, potential, project, _ = setup
    phonons = _phonon_potential(client, potential, configs=64)
    monkeypatch.setattr(abtem, "MAX_STACK_BYTES", 1024)
    r = client.post(_url(phonons), json={})
    assert r.status_code == 422 and "64 configs" in r.json()["detail"]
    assert list((project / "runs").iterdir()) == []


def test_failed_run_adds_nothing_and_is_cleared(setup):
    client, potential, project, _ = setup
    before = len(client.get("/api/files").json()[0]["datasets"])
    done = _wait(client, client.post(_url(potential), json={"energy_kv": 99}).json())
    assert done["status"] == "failed" and "on purpose" in done["error"]
    assert len(client.get("/api/files").json()[0]["datasets"]) == before
    assert all(p.name.startswith(".tmp-") for p in (project / "runs").iterdir())
    finished = _wait(client, client.post(_url(potential), json={}).json())
    assert finished["status"] == "done"
    names = [p.name for p in (project / "runs").iterdir()]
    assert len(names) == 1 and not names[0].startswith(".tmp-")


def test_cancelled_run_and_second_run(setup):
    client, potential, _, _ = setup
    job = client.post(_url(potential), json={"energy_kv": 98}).json()
    assert client.post(_url(potential), json={}).status_code == 409
    assert client.get(_url(potential)).json()["job"]["id"] == job["id"]
    client.post(f"/api/jobs/{job['id']}/cancel")
    assert _wait(client, job)["status"] == "cancelled"
    assert client.get(_url(potential)).json()["job"] is None
