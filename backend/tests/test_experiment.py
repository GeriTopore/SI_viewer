"""Experiments: the files of one folder opened together (a DM survey with
the region of its spectrum image, EELS spectrum images linked on opening)."""

import hyperspy.api as hs
import numpy as np
import pytest
from fastapi.testclient import TestClient

from si_viewer.app import create_app
from si_viewer.io.dm_experiment import (
    acquisition_label, dual_sibling_id, experiment_ids, is_survey, survey_regions, unique_id,
)

from test_eels_links import DUAL, _signal

IMAGE = "ImageList.TagGroup0"
KEYWORDS = f"{IMAGE}.ImageTags.Meta Data.Experiment keywords"
RUN = "Spectrum Imaging_27/01/2026_15:43:07"
ANNOTATIONS = "DocumentObjectList.TagGroup0.AnnotationGroupList"


@pytest.fixture
def client():
    return TestClient(create_app())


def _keywords(s, label, run=RUN):
    s.original_metadata.set_item(f"{KEYWORDS}.TagGroup0.Experiment ID", run)
    s.original_metadata.set_item(f"{KEYWORDS}.TagGroup0.Label", "Spectrum Imaging")
    s.original_metadata.set_item(f"{KEYWORDS}.TagGroup1.Label", label)


def _uid(s, *uid):
    for i, v in enumerate(uid):
        s.original_metadata.set_item(f"{IMAGE}.UniqueID.Data{i}", v)


def _survey():
    s = hs.signals.Signal2D(np.random.default_rng(0).random((64, 64)).astype("float32"))
    s.metadata.General.title = "ADF Image (SI Survey)"
    _keywords(s, "Survey")
    s.original_metadata.set_item(f"{IMAGE}.ImageTags.DataBar.Signal Name", "ADF")
    rects = [("Spectrum Image", (10.0, 20.0, 30.0, 44.0), 23), ("Spatial Drift (pre-measure)", (40.0, 2.0, 60.0, 22.0), 23)]
    for i, (label, rect, kind) in enumerate(rects):
        s.original_metadata.set_item(f"{ANNOTATIONS}.TagGroup{i}.AnnotationType", kind)
        s.original_metadata.set_item(f"{ANNOTATIONS}.TagGroup{i}.Label", label)
        s.original_metadata.set_item(f"{ANNOTATIONS}.TagGroup{i}.Rectangle", rect)
    s.original_metadata.set_item(f"{ANNOTATIONS}.TagGroup2.AnnotationType", 31)  # a scale bar
    s.original_metadata.set_item(f"{ANNOTATIONS}.TagGroup2.Rectangle", (1.0, 1.0, 5.0, 20.0))
    return s


def _dual(low: bool, own, sibling, run=RUN):
    s = _signal("EELS LL" if low else "EELS HL", low, role="low_loss" if low else "core_loss")
    _keywords(s, "EELS", run)
    _uid(s, *own)
    s.original_metadata.set_item(f"{DUAL}.Dual acquire sibling.Unique Image ID", sibling)
    return s


def _folder(tmp_path, run_b=RUN):
    folder = tmp_path / "Run 1"
    folder.mkdir()
    _survey().save(folder / "survey.hspy")
    _dual(True, (1, 2, 3, 4), (5, 6, 7, 8)).save(folder / "b_ll.hspy")
    _dual(False, (5, 6, 7, 8), (1, 2, 3, 4), run_b).save(folder / "a_hl.hspy")
    (folder / "notes.txt").write_text("not data")
    return folder


def _open_folder(client, folder):
    r = client.post("/api/files/folder", json={"path": str(folder)})
    assert r.status_code == 200, r.text
    plan = r.json()
    entries = []
    for p in plan["paths"]:
        r = client.post("/api/files", json={"path": p, "experiment_id": plan["experiment"]["id"]})
        assert r.status_code == 200, r.text
        entries.append(r.json())
    return plan, entries


def test_dm_tags_of_a_survey():
    s = _survey()
    assert is_survey(s) and acquisition_label(s) == "Survey" and experiment_ids(s) == [RUN]
    regions = survey_regions(s)
    # (top, left, bottom, right) -> x, y, width, height; the scale bar is no region.
    assert regions == [
        {"label": "Spectrum Image", "kind": "si", "x": 20.0, "y": 10.0, "width": 24.0, "height": 20.0},
        {"label": "Spatial Drift (pre-measure)", "kind": "drift", "x": 2.0, "y": 40.0, "width": 20.0, "height": 20.0},
    ]
    hl = _dual(False, (5, 6, 7, 8), (1, 2, 3, 4))
    assert unique_id(hl) == (5, 6, 7, 8) and dual_sibling_id(hl) == (1, 2, 3, 4)
    assert not is_survey(hl) and survey_regions(hl) == []


def test_open_folder_groups_and_links(client, tmp_path):
    folder = _folder(tmp_path)
    plan, entries = _open_folder(client, folder)
    assert plan["experiment"]["name"] == "Run 1"
    assert [p.split("/")[-1] for p in plan["paths"]] == ["a_hl.hspy", "b_ll.hspy", "survey.hspy"]
    files = client.get("/api/files").json()
    assert {f["experiment"]["id"] for f in files} == {plan["experiment"]["id"]}
    roles = {f["path"].split("/")[-1]: f["role"] for f in files}
    assert roles == {"a_hl.hspy": "EELS HL", "b_ll.hspy": "EELS LL", "survey.hspy": "Survey"}
    by_name = {f["path"].split("/")[-1]: f["datasets"][0] for f in files}
    assert by_name["a_hl.hspy"]["eels"]["linked"] == [by_name["b_ll.hspy"]["id"]]
    assert by_name["survey.hspy"]["survey"]["regions"][0]["kind"] == "si"
    assert by_name["a_hl.hspy"]["survey"] is None

    # The same folder again: the same experiment, nothing opened twice.
    again, _ = _open_folder(client, folder)
    assert again["experiment"]["id"] == plan["experiment"]["id"]
    assert len(client.get("/api/files").json()) == 3

    # The last file closed ends the experiment.
    for f in files:
        assert client.delete(f"/api/files/{f['id']}").status_code == 204
    assert client.app.state.session.experiments == {}


def test_spectrum_images_of_other_runs_are_not_linked(client, tmp_path):
    folder = _folder(tmp_path, run_b="Spectrum Imaging_27/01/2026_16:00:00")
    # No dual ids to go by: only the runs tell.
    for name in ("a_hl", "b_ll"):
        s = hs.load(folder / f"{name}.hspy")
        s.original_metadata.set_item(f"{DUAL}.Dual acquire sibling.Unique Image ID", (0, 0, 0, 0))
        s.save(folder / f"{name}.hspy", overwrite=True)
    _open_folder(client, folder)
    for f in client.get("/api/files").json():
        if f["datasets"][0]["eels"]:
            assert f["datasets"][0]["eels"]["linked"] == []


def test_opening_files_alone_links_nothing(client, tmp_path):
    folder = _folder(tmp_path)
    for name in ("a_hl", "b_ll"):
        r = client.post("/api/files", json={"path": str(folder / f"{name}.hspy")})
        assert r.json()["experiment"] is None and r.json()["role"] is None
        assert r.json()["datasets"][0]["eels"]["linked"] == []


def test_unknown_experiment_and_folder(client, tmp_path):
    folder = _folder(tmp_path)
    r = client.post("/api/files", json={"path": str(folder / "survey.hspy"), "experiment_id": "nope"})
    assert r.status_code == 404
    assert client.post("/api/files/folder", json={"path": str(folder / "missing")}).status_code == 404
