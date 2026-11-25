"""The projected potential job, with a stand-in for the abTEM worker."""

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
from si_viewer_abtem.potentials import PotentialSettings, configs_of, grid, normalized, title_of, to_signal, to_stack

from test_projects import STO

FAKE_WORKER = r"""#!{python}
import json, sys
import numpy as np
if sys.argv[-1] == "ping":
    print(json.dumps({{"result": {{"abtem": "1.0.11"}}}}))
    sys.exit()
request = json.load(open(sys.argv[-1]))
if request["sampling"] == 0.09:
    sys.exit("worker failed on purpose")
configs = request["phonon_configs"]
# The mean over configurations (here: the data times the configurations + 1, so tests see they were asked for).
array = np.arange(24.0).reshape(2, 4, 3) * (configs + 1)
np.save(request["output"], array)
print(json.dumps({{"progress": 0.5, "message": "Building"}}), flush=True)
print(json.dumps({{"result": {{"path": request["output"], "shape": [2, 4, 3], "configs": configs, "sampling": [0.1, 0.2], "slice_thickness": [1.5, 1.5], "abtem": "1.0.11"}}}}))
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
    return client, model, tmp_path


def _url(model):
    return f"/api/plugins/abtem/models/{model}/potential"


def test_grid_matches_abtem_rounding():
    assert grid([5.523, 6.764, 57.392], PotentialSettings(sampling=0.1, slice_thickness=1)) == (56, 68, 58)


def test_titles_tell_settings_apart():
    assert title_of(PotentialSettings(sampling=0.1)) == "Potential 0.1 Å, 1 Å slices"
    other = PotentialSettings(sampling=0.1, parametrization="kirkland", projection="finite")
    assert title_of(other) == "Potential 0.1 Å, 1 Å slices, Kirkland, finite"


def test_image_matches_the_beam_view():
    # abTEM's array is [x, y]; the image has x to the right and y up.
    array = np.zeros((4, 3))
    array[3, 0] = 1  # largest x, smallest y: the bottom right corner
    array[0, 2] = 2  # smallest x, largest y: the top left corner
    signal = to_signal(array, (0.1, 0.2), "t")
    assert signal.data.shape == (3, 4)
    assert signal.data[2, 3] == 1 and signal.data[0, 0] == 2
    x, y = signal.axes_manager.signal_axes
    assert (x.name, x.units, x.scale, x.size) == ("x", "Å", pytest.approx(0.1), 4)
    assert (y.name, y.units, y.scale, y.size) == ("y", "Å", pytest.approx(0.2), 3)


def test_frozen_phonon_titles_and_defaults():
    off = PotentialSettings()
    assert (off.phonon_configs, off.phonon_sigma, off.phonon_seed) == (0, 0.1, 0)
    # One configuration is no ensemble: it is off, and what is unused is reset.
    assert PotentialSettings(phonon_configs=1, phonon_sigma=0.3, phonon_seed=4) == off
    on = PotentialSettings(sampling=0.1, phonon_configs=8)
    assert title_of(on) == "Potential 0.1 Å, 1 Å slices, 8 phonon configs"
    other = PotentialSettings(sampling=0.1, phonon_configs=8, phonon_sigma=0.05, phonon_seed=3)
    assert title_of(other) == "Potential 0.1 Å, 1 Å slices, 8 phonon configs, σ 0.05 Å, seed 3"
    assert normalized({"sampling": 0.1, "slice_thickness": 1.0, "parametrization": "lobato", "projection": "infinite"}) == PotentialSettings(sampling=0.1).model_dump()
    assert configs_of(normalized({})) == 0 and configs_of({"phonon_configs": 8}) == 8


@pytest.mark.parametrize("bad", [
    {"sampling": 0.001}, {"slice_thickness": 0}, {"parametrization": "nope"}, {"projection": "nope"},
    {"phonon_configs": 65}, {"phonon_configs": -1}, {"phonon_configs": 8, "phonon_sigma": 0}, {"phonon_configs": 8, "phonon_sigma": 2},
    {"phonon_configs": 8, "phonon_seed": -1},
])
def test_bad_settings(setup, bad):
    client, model, _ = setup
    assert client.post(_url(model), json=bad).status_code == 422


def test_too_many_pixels(tmp_path, setup):
    client, _, _ = setup
    file = client.get("/api/files").json()[0]
    structure = file["datasets"][0]["id"]
    big = client.post(f"/api/plugins/abtem/structures/{structure}/models", json={"uvw": [0, 0, 1], "repeats": [12, 1, 1]}).json()["dataset_id"]
    assert client.post(_url(big), json={"sampling": 0.01}).status_code == 422


def test_too_much_memory(setup):
    client, _, _ = setup
    structure = client.get("/api/files").json()[0]["datasets"][0]["id"]
    big = client.post(f"/api/plugins/abtem/structures/{structure}/models", json={"uvw": [0, 0, 1], "repeats": [5, 5, 10]}).json()["dataset_id"]
    r = client.post(_url(big), json={"sampling": 0.01, "slice_thickness": 0.1})
    assert r.status_code == 422 and "GB" in r.json()["detail"]


def test_needs_abtem(setup, tmp_path, monkeypatch):
    client, model, _ = setup
    monkeypatch.setenv(engine.ENV_VAR, str(tmp_path / "none"))
    assert client.post(_url(model), json={}).status_code == 409


def test_stack_matches_the_beam_view_slice_by_slice():
    array = np.zeros((2, 4, 3))
    array[1, 3, 0] = 1  # second slice, largest x, smallest y: its bottom right corner
    signal = to_stack(array, (0.1, 0.2), 1.5, "t")
    assert signal.data.shape == (2, 3, 4) and signal.data[1, 2, 3] == 1
    z = signal.axes_manager.navigation_axes[0]
    assert (z.name, z.units, z.scale, z.size) == ("z", "Å", pytest.approx(1.5), 2)
    x, y = signal.axes_manager.signal_axes
    assert (x.name, x.size, y.name, y.size) == ("x", 4, "y", 3)


def test_result_is_a_potential_stack(setup, tmp_path):
    client, model, _ = setup
    done = _wait(client, client.post(_url(model), json={"sampling": 0.1}).json())
    assert done["status"] == "done", done["error"]
    datasets = {d["id"]: d for d in done["result"]["file"]["datasets"]}
    made = datasets[done["result"]["dataset_id"]]
    assert made["kind"] == "stack" and made["parent_id"] == model
    assert made["title"] == "Potential 0.1 Å, 1 Å slices"
    plugin = made["plugin"]
    assert plugin["label"] == "Potential" and plugin["stack"]["sum"] == "Projected"
    assert set(plugin["hidden_processes"]) == {"align", "rebin"}
    settings = plugin["settings"]
    assert settings["result"] == "potential" and settings["units"] == "V·Å" and settings["slices"] == 2
    assert settings["gpts"] == [4, 3] and settings["slice_thickness"] == 1.5
    assert settings["potential"]["sampling"] == 0.1

    # No separate projection: the stack's Sum is the slices' sum.
    assert not [d for d in datasets.values() if d["parent_id"] == made["id"]]
    session = client.app.state.session
    from si_viewer.processing.frames import get_frame

    expected = np.flipud(np.arange(24.0).reshape(2, 4, 3).sum(axis=0).T)
    np.testing.assert_allclose(get_frame(session.datasets[made["id"]], projection="sum"), expected)

    listed = client.get(_url(model)).json()
    assert [r["dataset_id"] for r in listed["results"]] == [made["id"]] and listed["job"] is None

    # A restart keeps it.
    session = Session()
    open_project(session, tmp_path / "project/project.hspy")
    again = session.datasets[made["id"]]
    assert again.kind is DatasetKind.STACK and again.plugin == made["plugin"]
    assert again.signal.data.shape == (2, 3, 4)


def test_same_settings_reuse_the_image_and_new_ones_add_one(setup):
    client, model, _ = setup
    first = _wait(client, client.post(_url(model), json={"sampling": 0.1}).json())
    same = client.post(_url(model), json={"sampling": 0.1}).json()
    assert same["id"] is None and same["status"] == "done"
    assert same["result"]["dataset_id"] == first["result"]["dataset_id"]
    other = _wait(client, client.post(_url(model), json={"sampling": 0.2}).json())
    assert other["result"]["dataset_id"] != first["result"]["dataset_id"]
    assert len(client.get(_url(model)).json()["results"]) == 2


def test_frozen_phonons_make_their_own_potential(setup, tmp_path):
    client, model, _ = setup
    plain = _wait(client, client.post(_url(model), json={"sampling": 0.1}).json())
    # One configuration is off: the same potential.
    one = client.post(_url(model), json={"sampling": 0.1, "phonon_configs": 1, "phonon_seed": 3}).json()
    assert one["id"] is None and one["result"]["dataset_id"] == plain["result"]["dataset_id"]

    body = {"sampling": 0.1, "phonon_configs": 4, "phonon_sigma": 0.07, "phonon_seed": 2}
    done = _wait(client, client.post(_url(model), json=body).json())
    assert done["status"] == "done", done["error"]
    assert done["result"]["dataset_id"] != plain["result"]["dataset_id"]
    made = next(d for d in done["result"]["file"]["datasets"] if d["id"] == done["result"]["dataset_id"])
    assert made["title"] == "Potential 0.1 Å, 1 Å slices, 4 phonon configs, σ 0.07 Å, seed 2"
    stored = made["plugin"]["settings"]["potential"]
    assert (stored["phonon_configs"], stored["phonon_sigma"], stored["phonon_seed"]) == (4, 0.07, 2)
    # The stack is the worker's mean over configurations, [slice, x, y] as without them.
    session = client.app.state.session
    assert session.datasets[made["id"]].signal.data.shape == (2, 3, 4)
    np.testing.assert_allclose(session.datasets[made["id"]].signal.data.sum(axis=0), np.flipud(np.arange(24.0).reshape(2, 4, 3).sum(axis=0).T) * 5)

    # The same ones again give it; another seed is another potential.
    same = client.post(_url(model), json=body).json()
    assert same["id"] is None and same["result"]["dataset_id"] == made["id"]
    other = _wait(client, client.post(_url(model), json={**body, "phonon_seed": 3}).json())
    assert other["result"]["dataset_id"] not in (made["id"], plain["result"]["dataset_id"])
    assert len(client.get(_url(model)).json()["results"]) == 3

    # A restart keeps them.
    again = Session()
    open_project(again, tmp_path / "project/project.hspy")
    assert again.datasets[made["id"]].plugin["settings"]["potential"]["phonon_configs"] == 4


def test_potentials_made_before_frozen_phonons_are_reused(setup):
    client, model, _ = setup
    first = _wait(client, client.post(_url(model), json={"sampling": 0.1}).json())
    ds = client.app.state.session.datasets[first["result"]["dataset_id"]]
    old = {k: v for k, v in ds.plugin["settings"]["potential"].items() if not k.startswith("phonon_")}
    ds.plugin = {**ds.plugin, "settings": {**ds.plugin["settings"], "potential": old}}
    same = client.post(_url(model), json={"sampling": 0.1}).json()
    assert same["id"] is None and same["result"]["dataset_id"] == first["result"]["dataset_id"]


def test_worker_failure_adds_nothing(setup):
    client, model, _ = setup
    done = _wait(client, client.post(_url(model), json={"sampling": 0.09}).json())
    assert done["status"] == "failed" and "on purpose" in done["error"]
    assert client.get(_url(model)).json()["results"] == []
