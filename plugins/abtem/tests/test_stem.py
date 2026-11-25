"""STEM imaging, with a stand-in for the abTEM worker."""

import base64
import json
import stat
import sys
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

import si_viewer_abtem.plugin as abtem
import si_viewer_abtem.stem as stem
from si_viewer.app import create_app
from si_viewer.project import open_project
from si_viewer.session import DatasetKind, Session
from si_viewer_abtem import engine
from si_viewer_abtem.runs import wavelength

from test_projects import STO

# The worker's scan: 3 × 2 positions, 131 bins of 1 mrad holding 1 + 10x + 100y
# each, so an annulus of n bins is n times that, and tests can tell which
# annulus and which orientation they got. It writes what it was asked to
# ``seen.json`` next to itself.
FAKE_WORKER = r"""#!{python}
import base64, json, pathlib, sys, time
import numpy as np
SEEN = pathlib.Path(sys.argv[0]).with_name("seen.json")
X, Y, BINS = 3, 2, 131


def polar():
    p = np.zeros((X, Y, BINS), np.float32)
    for x in range(X):
        for y in range(Y):
            p[x, y] = 1 + 10 * x + 100 * y
    return p


def encode(images):
    return {{"images": [base64.b64encode(i.astype(np.float32).tobytes()).decode() for i in images], "shape": list(images[0].shape)}}


if sys.argv[-1] == "ping":
    print(json.dumps({{"result": {{"abtem": "1.0.11"}}}}))
    sys.exit()
if sys.argv[-1] == "serve":
    print(json.dumps({{"ready": True}}), flush=True)
    for line in sys.stdin:
        message = json.loads(line)
        request = message["request"]
        SEEN.write_text(json.dumps(message))
        if message["command"] == "stem-integrate":
            p = np.load(request["polar"])
            out = encode([p[..., int(i):int(o)].sum(axis=-1) for i, o in request["limits"]])
        else:
            image = np.load(request["image"])
            out = {{**encode([image * 2]), "sampling": [s / 2 for s in request["sampling"]]}}
        print(json.dumps({{"result": out}}), flush=True)
    sys.exit()
request = json.load(open(sys.argv[-1]))
if sys.argv[-2] == "potential":
    np.save(request["output"], np.ones((2, 4, 3), np.float32))
    print(json.dumps({{"result": {{"path": request["output"], "shape": [2, 4, 3], "sampling": [0.1, 0.2], "slice_thickness": [1.0, 1.0], "abtem": "1.0.11"}}}}))
    sys.exit()
SEEN.write_text(json.dumps(request))
if request["energy"] == 99e3:
    sys.exit("worker failed on purpose")
if request["energy"] == 98e3:
    print(json.dumps({{"progress": 0.2, "message": "Scanning"}}), flush=True)
    time.sleep(30)
folder = pathlib.Path(request["folder"])
p = polar()
np.save(folder / "polar.npy", p)
images = [p[..., int(d["inner"]):int(d["outer"])].sum(axis=-1) for d in request["detectors"]]
np.save(folder / "images.npy", np.stack(images))
print(json.dumps({{"result": {{
    "shape": [X, Y, BINS], "scan_sampling": [0.2, 0.3], "scan_extent": [0.6, 0.6], "step": 1.0, "max_angle": 131.0,
    "positions": X * Y, "gpts": [4, 3], "abtem": "1.0.11",
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
    return client, built["result"]["dataset_id"], tmp_path / "project", python.with_name("seen.json")


def _url(potential):
    return f"/api/plugins/abtem/potentials/{potential}/stem"


DETECTORS = [
    {"name": "BF", "inner": 0, "outer": 20},
    {"name": "ABF", "inner": 10, "outer": 20},
    {"name": "HAADF", "inner": 90, "outer": 130},
]


def _body(**extra):
    return {"detectors": DETECTORS, **extra}


def _run(client, potential, **extra):
    done = _wait(client, client.post(_url(potential), json=_body(**extra)).json())
    assert done["status"] == "done", done["error"]
    datasets = {d["id"]: d for d in done["result"]["file"]["datasets"]}
    run = next(d for d in datasets.values() if d["plugin"] and d["plugin"].get("item") == "stem")
    return done, datasets, run


@pytest.mark.parametrize("bad", [
    {"detectors": []},
    {"detectors": [{"name": "X", "inner": 20, "outer": 20}]},
    {"detectors": [{"name": "", "inner": 0, "outer": 20}]},
    {"detectors": [{"name": "X", "inner": -1, "outer": 20}]},
    {"semiangle": 0}, {"energy_kv": 1}, {"tile": [1]}, {"tile": [0, 1]}, {"tile": [65, 1]},
    {"sampling": 0.001}, {"step": 0}, {"start": [0.5, 0.0], "end": [0.5, 1.0]}, {"end": [1.5, 1.0]},
])
def test_bad_settings(setup, bad):
    client, potential, project, _ = setup
    r = client.post(_url(potential), json={"detectors": DETECTORS, **bad})
    assert r.status_code == 422
    assert list((project / "runs").iterdir()) == []


def test_detector_beyond_the_sampling_is_refused_before_a_run(setup):
    client, potential, project, seen = setup
    r = client.post(_url(potential), json=_body(detectors=[*DETECTORS, {"name": "HAADF 2", "inner": 90, "outer": 200}]))
    assert r.status_code == 422 and "HAADF 2" in r.json()["detail"] and "131 mrad" in r.json()["detail"]
    assert list((project / "runs").iterdir()) == []
    assert not seen.exists()


def test_convergence_beyond_the_sampling_is_refused(setup):
    client, potential, project, _ = setup
    r = client.post(_url(potential), json=_body(semiangle=200))
    assert r.status_code == 422 and "131 mrad" in r.json()["detail"]
    assert list((project / "runs").iterdir()) == []


def test_too_much_polar_data_is_refused(setup, monkeypatch):
    client, potential, project, _ = setup
    monkeypatch.setattr(stem, "MAX_STACK_BYTES", 1024)
    r = client.post(_url(potential), json=_body(sampling=0.01))
    assert r.status_code == 422 and "GB" in r.json()["detail"]
    assert list((project / "runs").iterdir()) == []


def test_run_is_saved_with_an_image_for_each_detector(setup):
    client, potential, project, seen = setup
    done, datasets, run = _run(client, potential, energy_kv=200, semiangle=25, defocus=10, tile=[2, 3], start=[0.0, 0.5], end=[0.5, 1.0])
    assert run["parent_id"] == potential and run["plugin"]["label"] == "STEM"
    assert run["title"] == "STEM 200 kV, 25 mrad, Δf 10 Å"

    request = json.loads(seen.read_text())
    assert request["energy"] == 200e3 and request["semiangle"] == 25 and request["defocus"] == 10
    assert request["tile"] == [2, 3] and request["start"] == [0.0, 0.5] and request["end"] == [0.5, 1.0]
    assert request["detectors"] == DETECTORS and request["step"] == 1.0
    # The Nyquist sampling of the probe, when none is asked for.
    nyquist = wavelength(200) * 1000 / (4 * 25)
    assert request["scan_sampling"] == pytest.approx(nyquist) and request["sampling"] == 0.05
    assert run["plugin"]["settings"]["sampling"] == pytest.approx(nyquist)

    images = [d for d in datasets.values() if d["parent_id"] == run["id"]]
    assert sorted(d["plugin"]["label"] for d in images) == ["ABF", "BF", "HAADF"]
    assert done["result"]["dataset_id"] == next(d["id"] for d in images if d["plugin"]["label"] == "HAADF")
    session = client.app.state.session
    haadf = next(d for d in images if d["plugin"]["label"] == "HAADF")
    assert haadf["title"] == "HAADF 90–130 mrad" and haadf["kind"] == "image"
    signal = session.datasets[haadf["id"]].signal
    assert [a.scale for a in signal.axes_manager.signal_axes] == [0.2, 0.3] and signal.axes_manager.signal_axes[0].units == "Å"
    # [x, y] with y up becomes rows from the top: 3 columns, 2 rows, 40 bins of 1 + 10x + 100y.
    assert signal.data.shape == (2, 3)
    for x in range(3):
        for y in range(2):
            assert signal.data[1 - y, x] == pytest.approx(40 * (1 + 10 * x + 100 * y))

    folder = project / run["plugin"]["settings"]["run"]
    assert sorted(p.name for p in folder.iterdir()) == ["manifest.json", "polar.npy", "settings.json"]
    assert np.load(folder / "polar.npy").shape == (3, 2, 131)
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["mode"] == "stem" and manifest["axes"]["order"] == ["x", "y", "radial"]
    assert manifest["axes"]["sampling"] == [0.2, 0.3] and manifest["axes"]["radial_step"] == 1.0


def test_default_sampling_and_explicit_sampling(setup):
    client, potential, _, seen = setup
    _run(client, potential, sampling=0.4)
    assert json.loads(seen.read_text())["scan_sampling"] == 0.4


def test_add_detector_integrates_the_saved_polar_data(setup):
    client, potential, project, seen = setup
    _, datasets, run = _run(client, potential)
    r = client.post(f"/api/plugins/abtem/stems/{run['id']}/detectors", json={"name": "MAADF", "inner": 50, "outer": 100})
    assert r.status_code == 200, r.text
    added = {d["id"]: d for d in r.json()["file"]["datasets"]}[r.json()["dataset_id"]]
    assert added["parent_id"] == run["id"] and added["plugin"]["label"] == "MAADF" and added["title"] == "MAADF 50–100 mrad"
    sent = json.loads(seen.read_text())
    assert sent["command"] == "stem-integrate" and sent["request"]["limits"] == [[50, 100]]
    assert sent["request"]["step"] == 1.0 and sent["request"]["sampling"] == [0.2, 0.3]
    data = client.app.state.session.datasets[added["id"]].signal.data
    assert data[0, 0] == pytest.approx(50 * (1 + 100))  # x = 0, y = 1 is the top row

    # A name with no colour of its own is a plain detector.
    r = client.post(f"/api/plugins/abtem/stems/{run['id']}/detectors", json={"name": "Mine", "inner": 10, "outer": 30})
    added = {d["id"]: d for d in r.json()["file"]["datasets"]}[r.json()["dataset_id"]]
    assert added["plugin"]["label"] == "Detector" and added["title"] == "Mine 10–30 mrad"


@pytest.mark.parametrize("bad", [
    {"inner": 50, "outer": 140}, {"inner": 60, "outer": 60}, {"inner": -1, "outer": 10}, {"inner": 0, "outer": 0},
])
def test_add_detector_beyond_the_data_is_refused(setup, bad):
    client, potential, _, _ = setup
    _, datasets, run = _run(client, potential)
    before = len(client.get("/api/files").json()[0]["datasets"])
    r = client.post(f"/api/plugins/abtem/stems/{run['id']}/detectors", json=bad)
    assert r.status_code == 422
    assert len(client.get("/api/files").json()[0]["datasets"]) == before


def test_add_detector_needs_a_stem_run(setup):
    client, potential, _, _ = setup
    assert client.post(f"/api/plugins/abtem/stems/{potential}/detectors", json={"inner": 1, "outer": 2}).status_code == 404


def test_post_processing_adds_an_image_under_the_run(setup):
    client, potential, project, seen = setup
    _, datasets, run = _run(client, potential)
    haadf = next(d for d in datasets.values() if d["parent_id"] == run["id"] and d["plugin"]["label"] == "HAADF")
    r = client.post(f"/api/plugins/abtem/stem-images/{haadf['id']}/post", json={"interpolate": 0.1, "source_size": 0.5, "dose": 1e5, "seed": 3})
    assert r.status_code == 200, r.text
    made = {d["id"]: d for d in r.json()["file"]["datasets"]}[r.json()["dataset_id"]]
    assert made["parent_id"] == run["id"] and made["plugin"]["label"] == "HAADF"
    assert made["title"] == "HAADF 90–130 mrad · 0.1 Å, σ 0.5 Å, 100000 e⁻/Å²"
    sent = json.loads(seen.read_text())["request"]
    assert sent["sampling"] == [0.2, 0.3] and sent["interpolate"] == 0.1 and sent["source_size"] == 0.5
    assert sent["dose"] == 1e5 and sent["seed"] == 3
    session = client.app.state.session
    # The worker doubles the [x, y] image; it comes back the way it went.
    np.testing.assert_allclose(session.datasets[made["id"]].signal.data, 2 * session.datasets[haadf["id"]].signal.data)
    assert [a.scale for a in session.datasets[made["id"]].signal.axes_manager.signal_axes] == [0.1, 0.15]
    assert made["plugin"]["settings"]["post"]["dose"] == 1e5


@pytest.mark.parametrize("bad", [{}, {"source_size": -1}, {"dose": -1}, {"interpolate": -0.1}])
def test_post_processing_needs_a_step(setup, bad):
    client, potential, _, _ = setup
    _, datasets, run = _run(client, potential)
    image = next(d for d in datasets.values() if d["parent_id"] == run["id"])
    assert client.post(f"/api/plugins/abtem/stem-images/{image['id']}/post", json=bad).status_code == 422


def test_post_processing_needs_a_stem_image(setup):
    client, potential, _, _ = setup
    assert client.post(f"/api/plugins/abtem/stem-images/{potential}/post", json={"source_size": 0.3}).status_code == 404


def test_run_survives_reopening_the_project(setup):
    client, potential, project, _ = setup
    _, datasets, run = _run(client, potential)
    again = Session()
    open_project(again, project / "project.hspy")
    assert again.datasets[run["id"]].kind is DatasetKind.ITEM
    assert again.datasets[run["id"]].plugin["settings"]["max_angle"] == 131.0
    children = [d for d in again.datasets.values() if d.parent_id == run["id"]]
    assert sorted(d.plugin["label"] for d in children) == ["ABF", "BF", "HAADF"]
    assert all(d.kind is DatasetKind.IMAGE for d in children)

    # A detector can be added to the reopened run, from its polar data.
    reopened = TestClient(create_app(again, plugins=[abtem]))
    r = reopened.post(f"/api/plugins/abtem/stems/{run['id']}/detectors", json={"name": "ADF", "inner": 20, "outer": 60})
    assert r.status_code == 200, r.text


def test_failed_run_adds_nothing(setup):
    client, potential, project, _ = setup
    before = len(client.get("/api/files").json()[0]["datasets"])
    done = _wait(client, client.post(_url(potential), json=_body(energy_kv=99)).json())
    assert done["status"] == "failed" and "on purpose" in done["error"]
    assert len(client.get("/api/files").json()[0]["datasets"]) == before
    assert all(p.name.startswith(".tmp-") for p in (project / "runs").iterdir())


def test_one_run_at_a_time_on_a_potential(setup):
    client, potential, project, _ = setup
    job = client.post(_url(potential), json=_body(energy_kv=98)).json()
    assert client.post(_url(potential), json=_body()).status_code == 409
    assert client.get(_url(potential)).json()["job"]["id"] == job["id"]
    client.post(f"/api/jobs/{job['id']}/cancel")
    assert _wait(client, job)["status"] == "cancelled"
