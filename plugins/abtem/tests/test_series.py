"""Thickness and defocus series, with a stand-in for the abTEM worker."""

import importlib.util
import json
import stat
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import si_viewer_abtem.plugin as abtem
from si_viewer.app import create_app
from si_viewer.project import open_project
from si_viewer.session import DatasetKind, Session
from si_viewer_abtem import engine
from si_viewer_abtem.series import MAX_DEFOCI, defocus_values, exit_planes_of

from test_projects import STO

# A potential of 6 slices (1 Å each) on a 4 × 3 grid. A thickness series keeps
# plane k as (1 + 0.1 k) e^{0.2 i k} (the first configuration; the next ones
# are scaled), and its image is |ψ|² × 2 (× 3 for a focal series). The exit
# wave of HRTEM is 1 + 0.5i everywhere; its defocus series image is
# |ψ|² × 2 (× 3) + defocus / 1000.
FAKE_WORKER = r"""#!PYTHON
import base64, json, pathlib, sys, time
import numpy as np
if sys.argv[-1] == "ping":
    print(json.dumps({"result": {"abtem": "1.0.11"}}))
    sys.exit()
if sys.argv[-1] == "serve":
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        message = json.loads(line)
        request = message["request"]
        if message["command"] == "ctf":
            array = np.zeros((4, 4), np.float32)
            out = {"image": base64.b64encode(array.tobytes()).decode(), "image_shape": [4, 4], "image_step": [0.1, 0.1]}
        elif message["command"] == "defocus_series":
            wave = np.load(request["wave"])
            factor = 3 if request["coherence"] == "focal-series" else 2
            intensity = (np.abs(wave) ** 2).mean(axis=0) if wave.ndim == 3 else np.abs(wave) ** 2
            images = np.stack([intensity * factor + d / 1000 for d in request["defoci"]]).astype(np.float32)
            np.save(request["output"], images)
            out = {"path": request["output"], "shape": list(images.shape)}
        else:
            wave = np.load(request["wave"])
            image = (np.abs(wave) ** 2 * 2).astype(np.float32)
            out = {"image": base64.b64encode(image.tobytes()).decode(), "shape": list(image.shape), "mean": float(image.mean())}
        print(json.dumps({"result": out}), flush=True)
    sys.exit()
request = json.load(open(sys.argv[-1]))
command = sys.argv[-2]
if command == "potential":
    np.save(request["output"], np.ones((6, 4, 3), np.float32))
    print(json.dumps({"result": {"path": request["output"], "shape": [6, 4, 3], "sampling": [0.1, 0.2], "slice_thickness": [1.0] * 6, "abtem": "1.0.11"}}))
    sys.exit()
folder = pathlib.Path(request["folder"])
if request["energy"] == 99e3:
    sys.exit("worker failed on purpose")
if request["energy"] == 98e3:
    print(json.dumps({"progress": 0.2, "message": "Running"}), flush=True)
    time.sleep(30)
configs = request.get("phonon_configs") or 0
if command == "hrtem":
    wave = (1 + 0.5j) * np.ones((4, 3), np.complex64)
    if configs:
        wave = np.stack([wave * (c + 1) for c in range(configs)])
    np.save(folder / "exit_wave.npy", wave)
    print(json.dumps({"result": {"shape": list(wave.shape), "configs": configs, "sampling": [0.1, 0.2], "slices": 6, "exit_intensity": 1.25, "abtem": "1.0.11"}}))
    sys.exit()
assert command == "thickness_series", command
planes = request["exit_planes"]
(folder / "seen.json").write_text(json.dumps(request))
one = np.stack([(1 + 0.1 * k) * np.exp(0.2j * k) * np.ones((4, 3)) for k in range(len(planes))]).astype(np.complex64)
waves = np.stack([one * (c + 1) for c in range(configs)]) if configs else one
np.save(folder / "exit_waves.npy", waves)
if request["ctf"]:
    factor = 3 if request["ctf"]["coherence"] == "focal-series" else 2
    intensity = (np.abs(waves) ** 2).mean(axis=0) if configs else np.abs(waves) ** 2
    np.save(folder / "images.npy", (intensity * factor).astype(np.float32))
print(json.dumps({"result": {
    "shape": list(one.shape), "configs": configs, "sampling": [0.1, 0.2], "slice_thickness": 1.0,
    "thicknesses": [0.0] + [float(p + 1) for p in planes[1:]], "exit_intensity": 1.0, "abtem": "1.0.11",
}}))
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
    python.write_text(FAKE_WORKER.replace("PYTHON", sys.executable))
    python.chmod(python.stat().st_mode | stat.S_IEXEC)
    (env / engine.READY).write_text("{}")
    client = TestClient(create_app(Session(), plugins=[abtem]))
    cif = tmp_path / "SrTiO3.cif"
    cif.write_text(STO)
    file = client.post("/api/plugins/abtem/projects", json={"path": str(tmp_path / "project")}).json()["file"]
    structure = client.post(f"/api/plugins/abtem/projects/{file['id']}/structures", json={"path": str(cif)}).json()["dataset_id"]
    model = client.post(f"/api/plugins/abtem/structures/{structure}/models", json={"uvw": [0, 0, 1], "repeats": [2, 2, 2]}).json()["dataset_id"]
    return client, model, tmp_path / "project", file["id"]


def _potential(client, model, **settings):
    built = _wait(client, client.post(f"/api/plugins/abtem/models/{model}/potential", json=settings).json())
    assert built["status"] == "done", built["error"]
    return built["result"]["dataset_id"]


def _series_url(potential):
    return f"/api/plugins/abtem/potentials/{potential}/thickness-series"


def _ctf(client, file_id, **settings):
    return client.post(f"/api/plugins/abtem/projects/{file_id}/ctfs", json=settings).json()["dataset_id"]


def _by_id(done):
    return {d["id"]: d for d in done["result"]["file"]["datasets"]}


def _children(datasets, parent):
    return [d for d in datasets.values() if d["parent_id"] == parent]


def _axis(client, dataset_id):
    return client.app.state.session.datasets[dataset_id].signal.axes_manager.navigation_axes[0]


def test_planes_and_defocus_values():
    assert exit_planes_of(6, 1) == [-1, 0, 1, 2, 3, 4, 5]
    assert exit_planes_of(6, 2) == [-1, 1, 3, 5]
    # Only the multiples of the step: even spacing, so the last slice may be left out.
    assert exit_planes_of(7, 3) == [-1, 2, 5]
    assert defocus_values(-100, 100, 20) == [-100 + 20 * i for i in range(11)]
    assert defocus_values(0, 1, 0.1)[-1] == 1.0 and len(defocus_values(0, 1, 0.1)) == 11
    with pytest.raises(ValueError):
        defocus_values(0, 10, 20)
    with pytest.raises(ValueError):
        defocus_values(0, MAX_DEFOCI + 1, 1)
    assert len(defocus_values(0, MAX_DEFOCI - 1, 1)) == MAX_DEFOCI


def test_worker_has_the_series_commands():
    path = Path(engine.WORKER)
    spec = importlib.util.spec_from_file_location("abtem_worker_main_for_test", path)
    main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main)
    assert main.COMMANDS["thickness_series"] == (main.thickness_series, True)
    assert main.SERVED["defocus_series"] is main.defocus_series
    module = main._series()
    assert callable(module.thickness_series) and callable(module.defocus_series)


def test_only_a_potential_can_have_a_series(setup):
    client, model, _, _ = setup
    assert client.post(_series_url(model), json={}).status_code == 404
    assert client.get(_series_url(model)).status_code == 404


@pytest.mark.parametrize("bad", [{"energy_kv": 1}, {"step": 0}, {"step": 7}, {"coherence": "nope"}, {"ctf": "nope"}])
def test_bad_thickness_settings(setup, bad):
    client, model, project, _ = setup
    potential = _potential(client, model)
    assert client.post(_series_url(potential), json=bad).status_code in (404, 422)
    assert list((project / "runs").iterdir()) == []


def test_needs_abtem(setup, tmp_path, monkeypatch):
    client, model, _, _ = setup
    potential = _potential(client, model)
    monkeypatch.setenv(engine.ENV_VAR, str(tmp_path / "none"))
    assert client.post(_series_url(potential), json={}).status_code == 409


def test_ctf_aperture_beyond_the_sampling_is_refused(setup):
    client, model, project, file_id = setup
    potential = _potential(client, model)
    r = client.post(_series_url(potential), json={"ctf": _ctf(client, file_id, semiangle=200)})
    assert r.status_code == 422 and "131 mrad" in r.json()["detail"]
    assert list((project / "runs").iterdir()) == []


def test_series_without_a_ctf_keeps_the_exit_waves(setup):
    client, model, project, _ = setup
    potential = _potential(client, model)
    done = _wait(client, client.post(_series_url(potential), json={"energy_kv": 200}).json())
    assert done["status"] == "done", done["error"]
    datasets = _by_id(done)
    shown = datasets[done["result"]["dataset_id"]]
    series = datasets[shown["parent_id"]]
    assert series["kind"] == "item" and series["parent_id"] == potential and series["plugin"]["item"] == "series"
    assert series["plugin"]["label"] == "Thickness series" and series["title"] == "Thickness series 200 kV"
    settings = series["plugin"]["settings"]
    assert settings["energy_kv"] == 200 and settings["planes"] == 7 and settings["step"] == 1
    assert settings["thicknesses"] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0] and settings["ctf"] is None

    children = {d["plugin"]["label"]: d for d in _children(datasets, series["id"])}
    assert set(children) == {"Amplitude", "Phase"}
    assert shown["id"] == children["Amplitude"]["id"] and shown["kind"] == "stack"
    session = client.app.state.session
    amplitude = session.datasets[children["Amplitude"]["id"]]
    assert amplitude.signal.data.shape == (7, 3, 4)  # [plane, y, x]
    np.testing.assert_allclose(amplitude.signal.data[:, 0, 0], [1 + 0.1 * k for k in range(7)], rtol=1e-5)
    np.testing.assert_allclose(session.datasets[children["Phase"]["id"]].signal.data[:, 0, 0], [0.2 * k for k in range(7)], atol=1e-5)
    axis = _axis(client, shown["id"])
    assert (axis.name, axis.units, axis.scale, axis.offset) == ("t", "Å", 1.0, 0.0)
    assert amplitude.plugin["stack"] == {"frame": "Thickness", "frames": "thicknesses", "sum": "Sum", "start": "last", "planes": True}
    # A sum of phases means nothing.
    assert session.datasets[children["Phase"]["id"]].plugin["stack"]["projections"] is False

    folder = project / settings["run"]
    assert {p.name for p in folder.iterdir()} >= {"exit_waves.npy", "manifest.json", "settings.json"}
    assert not (folder / "images.npy").exists() and not (folder / "request.json").exists()
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["mode"] == "thickness-series" and manifest["axes"]["order"] == ["plane", "x", "y"]
    assert manifest["outputs"] == {"exit_waves": "exit_waves.npy"}
    assert np.load(folder / "exit_waves.npy").shape == (7, 4, 3)


def test_series_with_a_ctf_adds_the_image_stack(setup):
    client, model, project, file_id = setup
    potential = _potential(client, model)
    ctf = _ctf(client, file_id, energy_kv=200, defocus=30, focal_spread=20)
    done = _wait(client, client.post(_series_url(potential), json={"ctf": ctf, "energy_kv": 300, "step": 2, "coherence": "focal-series"}).json())
    assert done["status"] == "done", done["error"]
    datasets = _by_id(done)
    image = datasets[done["result"]["dataset_id"]]
    series = datasets[image["parent_id"]]
    # The CTF's energy, not the one sent; planes at the incident wave, then every 2 slices.
    assert series["title"] == "Thickness series 200 kV, every 2 slices"
    assert image["plugin"]["label"] == "Image" and image["kind"] == "stack"
    assert image["title"] == "Image · CTF 200 kV, Δf 30 Å, focal series"
    assert [d["plugin"]["label"] for d in _children(datasets, series["id"])][0] == "Image"
    settings = series["plugin"]["settings"]
    assert settings["thicknesses"] == [0.0, 2.0, 4.0, 6.0] and settings["step"] == 2 and settings["planes"] == 4
    assert settings["ctf"] == ctf and settings["ctf_settings"]["defocus"] == 30 and settings["coherence"] == "focal-series"
    session = client.app.state.session
    data = session.datasets[image["id"]].signal.data
    assert data.shape == (4, 3, 4)
    np.testing.assert_allclose(data[:, 0, 0], [3 * (1 + 0.1 * k) ** 2 for k in range(4)], rtol=1e-5)  # |ψ|² × 3 of each plane
    np.testing.assert_allclose(data[0], 3.0)  # the incident wave
    axis = _axis(client, image["id"])
    assert (axis.name, axis.scale) == ("t", 2.0)
    folder = project / settings["run"]
    request = json.loads((folder / "seen.json").read_text())
    assert request["exit_planes"] == [-1, 1, 3, 5] and request["ctf"]["coherence"] == "focal-series"
    assert request["ctf"]["defocus"] == 30 and request["energy"] == 200e3
    assert not (folder / "images.npy").exists()

    again = Session()
    open_project(again, project / "project.hspy")
    stack = again.datasets[image["id"]]
    assert stack.kind is DatasetKind.STACK and stack.plugin["stack"]["planes"] is True
    np.testing.assert_allclose(stack.signal.data, data)
    assert stack.signal.axes_manager.navigation_axes[0].name == "t"
    assert again.datasets[series["id"]].plugin["item"] == "series"


def test_focal_series_needs_a_focal_spread(setup):
    client, model, project, file_id = setup
    potential = _potential(client, model)
    ctf = _ctf(client, file_id)
    done = _wait(client, client.post(_series_url(potential), json={"ctf": ctf, "coherence": "focal-series"}).json())
    image = _by_id(done)[done["result"]["dataset_id"]]
    assert image["plugin"]["settings"] == {"result": "thickness-series-image", "run": image["parent_id"]}
    series = _by_id(done)[image["parent_id"]]
    assert series["plugin"]["settings"]["coherence"] == "envelope"
    np.testing.assert_allclose(client.app.state.session.datasets[image["id"]].signal.data[0], 2.0)


def test_series_of_frozen_phonons_shows_the_first_configuration(setup):
    client, model, project, file_id = setup
    potential = _potential(client, model, phonon_configs=2)
    ctf = _ctf(client, file_id)
    done = _wait(client, client.post(_series_url(potential), json={"ctf": ctf}).json())
    assert done["status"] == "done", done["error"]
    datasets = _by_id(done)
    series = datasets[_by_id(done)[done["result"]["dataset_id"]]["parent_id"]]
    assert series["plugin"]["settings"]["configs"] == 2 and series["plugin"]["settings"]["gpts"] == [4, 3]
    children = {d["plugin"]["label"]: d for d in _children(datasets, series["id"])}
    session = client.app.state.session
    # The first configuration's wave, not an average of waves.
    np.testing.assert_allclose(session.datasets[children["Amplitude"]["id"]].signal.data[:, 0, 0], [1 + 0.1 * k for k in range(7)], rtol=1e-5)
    assert children["Amplitude"]["title"].endswith("(config 1)") and "2 phonon configs" in children["Image"]["title"]
    # Intensities are averaged over the configurations: (1² + 2²) / 2 = 2.5, × 2.
    np.testing.assert_allclose(session.datasets[children["Image"]["id"]].signal.data[0], 5.0)
    folder = project / series["plugin"]["settings"]["run"]
    assert np.load(folder / "exit_waves.npy").shape == (2, 7, 4, 3)
    assert json.loads((folder / "manifest.json").read_text())["axes"]["order"] == ["config", "plane", "x", "y"]


def test_failed_series_adds_nothing_and_is_cleared(setup):
    client, model, project, _ = setup
    potential = _potential(client, model)
    before = len(client.get("/api/files").json()[0]["datasets"])
    done = _wait(client, client.post(_series_url(potential), json={"energy_kv": 99}).json())
    assert done["status"] == "failed" and "on purpose" in done["error"]
    assert len(client.get("/api/files").json()[0]["datasets"]) == before
    assert all(p.name.startswith(".tmp-") for p in (project / "runs").iterdir())
    finished = _wait(client, client.post(_series_url(potential), json={}).json())
    assert finished["status"] == "done"
    names = [p.name for p in (project / "runs").iterdir()]
    assert len(names) == 1 and not names[0].startswith(".tmp-")


def test_cancelled_series_and_second_series(setup):
    client, model, project, _ = setup
    potential = _potential(client, model)
    job = client.post(_series_url(potential), json={"energy_kv": 98}).json()
    assert client.post(_series_url(potential), json={}).status_code == 409
    assert client.get(_series_url(potential)).json()["job"]["id"] == job["id"]
    # Its temporary folder is not cleared by an HRTEM run started meanwhile.
    folders = list((project / "runs").iterdir())
    assert len(folders) == 1
    other = client.post(f"/api/plugins/abtem/potentials/{potential}/hrtem", json={"energy_kv": 98}).json()
    assert folders[0].exists()
    client.post(f"/api/jobs/{other['id']}/cancel")
    client.post(f"/api/jobs/{job['id']}/cancel")
    _wait(client, other)
    assert _wait(client, job)["status"] == "cancelled"
    assert client.get(_series_url(potential)).json()["job"] is None


def _hrtem(client, potential, **body):
    done = _wait(client, client.post(f"/api/plugins/abtem/potentials/{potential}/hrtem", json=body).json())
    assert done["status"] == "done", done["error"]
    return _by_id(done)[done["result"]["dataset_id"]]["parent_id"]


def test_defocus_series_of_an_hrtem_run(setup):
    client, model, project, file_id = setup
    potential = _potential(client, model)
    run = _hrtem(client, potential)
    ctf = _ctf(client, file_id, defocus=-20, semiangle=20)
    url = f"/api/plugins/abtem/runs/{run}/defocus-series"
    r = client.post(url, json={"ctf": ctf, "start": -100, "stop": 100, "step": 20})
    assert r.status_code == 200, r.text
    datasets = {d["id"]: d for d in r.json()["file"]["datasets"]}
    series = datasets[r.json()["dataset_id"]]
    assert series["parent_id"] == run and series["kind"] == "stack" and series["plugin"]["label"] == "Defocus series"
    assert series["title"] == "Defocus series · CTF 300 kV, Δf -100 to 100 Å"
    assert series["plugin"]["stack"] == {"frame": "Defocus", "frames": "defoci", "sum": "Sum", "planes": True}
    settings = series["plugin"]["settings"]
    assert settings["start"] == -100 and settings["stop"] == 100 and settings["step"] == 20 and settings["defoci"] == 11
    assert settings["ctf"] == ctf and settings["ctf_settings"]["defocus"] == -20 and settings["coherence"] == "envelope"
    session = client.app.state.session
    stack = session.datasets[series["id"]]
    assert stack.signal.data.shape == (11, 3, 4)
    np.testing.assert_allclose(stack.signal.data[:, 0, 0], [2.5 + d / 1000 for d in range(-100, 101, 20)], rtol=1e-6)
    axis = _axis(client, series["id"])
    assert (axis.name, axis.units, axis.scale, axis.offset) == ("Δf", "Å", 20.0, -100.0)
    # The worker's temporary file is gone.
    assert not list((Path(tempfile.gettempdir()) / "si-viewer-abtem").glob("defocus-series-*.npy"))

    again = Session()
    open_project(again, project / "project.hspy")
    reopened = again.datasets[series["id"]]
    assert reopened.kind is DatasetKind.STACK and reopened.parent_id == run
    axis = reopened.signal.axes_manager.navigation_axes[0]
    assert (axis.name, axis.scale, axis.offset) == ("Δf", 20.0, -100.0)
    np.testing.assert_allclose(reopened.signal.data, stack.signal.data)


def test_defocus_series_focal_series_and_phonon_wave(setup):
    client, model, _, file_id = setup
    potential = _potential(client, model, phonon_configs=2)
    run = _hrtem(client, potential)
    ctf = _ctf(client, file_id, focal_spread=30)
    url = f"/api/plugins/abtem/runs/{run}/defocus-series"
    r = client.post(url, json={"ctf": ctf, "start": 0, "stop": 10, "step": 10, "coherence": "focal-series"})
    assert r.status_code == 200, r.text
    series = next(d for d in r.json()["file"]["datasets"] if d["id"] == r.json()["dataset_id"])
    assert series["title"].endswith(", focal series, 2 phonon configs")
    # Intensities averaged over the configurations (1.25 × (1 + 4) / 2), × 3.
    np.testing.assert_allclose(client.app.state.session.datasets[series["id"]].signal.data[:, 0, 0], [9.375, 9.385], rtol=1e-6)
    # Without a focal spread there is nothing to average over.
    plain = _ctf(client, file_id)
    r = client.post(url, json={"ctf": plain, "start": 0, "stop": 10, "step": 10, "coherence": "focal-series"})
    assert client.app.state.session.datasets[r.json()["dataset_id"]].plugin["settings"]["coherence"] == "envelope"


@pytest.mark.parametrize("bad", [
    {"start": 0, "stop": 10, "step": 20},
    {"start": 0, "stop": 1000, "step": 1},
    {"start": 0, "stop": 10, "step": 0},
    {"start": 10, "stop": 0, "step": 5},
    {"start": 0, "stop": 10},
])
def test_bad_defocus_series(setup, bad):
    client, model, _, file_id = setup
    run = _hrtem(client, _potential(client, model))
    ctf = _ctf(client, file_id)
    before = len(client.get("/api/files").json()[0]["datasets"])
    r = client.post(f"/api/plugins/abtem/runs/{run}/defocus-series", json={"ctf": ctf, **bad})
    assert r.status_code == 422
    assert len(client.get("/api/files").json()[0]["datasets"]) == before


def test_defocus_series_only_of_hrtem_runs_with_a_matching_ctf(setup):
    client, model, project, file_id = setup
    potential = _potential(client, model)
    run = _hrtem(client, potential)
    ctf = _ctf(client, file_id)
    body = {"ctf": ctf, "start": 0, "stop": 20, "step": 10}
    assert client.post(f"/api/plugins/abtem/runs/{potential}/defocus-series", json=body).status_code == 404
    other = _ctf(client, file_id, energy_kv=80)
    wrong = client.post(f"/api/plugins/abtem/runs/{run}/defocus-series", json={**body, "ctf": other})
    assert wrong.status_code == 422 and "80 kV" in wrong.json()["detail"]
    assert client.post(f"/api/plugins/abtem/runs/{run}/defocus-series", json={**body, "ctf": run}).status_code == 404
    # Not of a thickness series: its image stacks are not HRTEM runs.
    done = _wait(client, client.post(_series_url(potential), json={}).json())
    series = _by_id(done)[_by_id(done)[done["result"]["dataset_id"]]["parent_id"]]["id"]
    assert client.post(f"/api/plugins/abtem/runs/{series}/defocus-series", json=body).status_code == 404
    assert client.post(f"/api/plugins/abtem/runs/{run}/images", json={"ctf": ctf}).status_code == 200


def test_defocus_series_needs_abtem(setup, tmp_path, monkeypatch):
    client, model, _, file_id = setup
    run = _hrtem(client, _potential(client, model))
    ctf = _ctf(client, file_id)
    monkeypatch.setenv(engine.ENV_VAR, str(tmp_path / "none"))
    r = client.post(f"/api/plugins/abtem/runs/{run}/defocus-series", json={"ctf": ctf, "start": 0, "stop": 20, "step": 10})
    assert r.status_code == 409
