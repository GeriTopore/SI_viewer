"""Saving and opening an analysis (M18b)."""

import shutil
import time

import hyperspy.api as hs
import numpy as np
import pytest
import tifffile
from fastapi.testclient import TestClient

from si_viewer.app import create_app

from test_insitu import fake_dm4  # noqa: F401 (fixture)
from test_registration import SHIFTS, _drifting_stack
from test_zlp_alignment import _open_pair


def _client():
    return TestClient(create_app())


@pytest.fixture
def client():
    return _client()


def _wait(client, job, timeout=300):
    t0 = time.time()
    while job["status"] in ("queued", "running"):
        assert time.time() - t0 < timeout, "job did not finish"
        time.sleep(0.02)
        job = client.get(f"/api/jobs/{job['id']}").json()
    assert job["status"] == "done", job
    return job


def _ok(r):
    assert r.status_code == 200, r.text
    return r.json()


def _si(entry) -> dict:
    return next(d for d in entry["datasets"] if d["kind"] == "spectrum_image")


def _tree(client) -> list[tuple]:
    """Every dataset as (file id, id, title, origin, parent, shape), in order."""
    out = []
    for f in client.get("/api/files").json():
        for d in f["datasets"]:
            out.append((f["id"], d["id"], d["title"], d["origin"], d["parent_id"],
                        tuple(d["navigation_shape"]), tuple(d["signal_shape"])))
    return out


def _arrays(client) -> dict:
    session = client.app.state.session
    return {i: np.asarray(ds.signal.data, np.float64) for i, ds in session.datasets.items()}


def _save(client, path, views=None, **body):
    return client.post("/api/project/save", json={"path": str(path), "views": views or {}, **body})


def _reopen(path) -> tuple[TestClient, dict]:
    other = _client()
    return other, _ok(other.post("/api/project/open", json={"path": str(path)}))


def _same(a: TestClient, b: TestClient) -> None:
    assert _tree(a) == _tree(b)
    xa, xb = _arrays(a), _arrays(b)
    for i in xa:
        np.testing.assert_allclose(xb[i], xa[i], rtol=1e-5, atol=1e-5, err_msg=i)


def test_round_trip(client, tmp_path, eels_cuzn):
    raw = shutil.copy(eels_cuzn, tmp_path / "cuzn.hspy")
    si = _si(_ok(client.post("/api/files", json={"path": str(raw)})))
    _ok(client.post(f"/api/datasets/{si['id']}/rebin", json={"factors": [2, 2, 1]}))
    _ok(client.post(f"/api/datasets/{si['id']}/window-maps", json={"lo": 940, "hi": 980}))
    job = _ok(client.post(f"/api/datasets/{si['id']}/background",
                          json={"options": {"background": [800, 920], "signal": [925, 1010]}}))
    _wait(client, job)
    _ok(client.post(f"/api/datasets/{si['id']}/line-scan", json={"line": {"x0": 0.5, "y0": 2.5, "x1": 15.5, "y1": 2.5}}))

    views = {"si": {"cursor": [3, 4]}, "layout": "x"}
    path = tmp_path / "analysis"
    saved = _ok(_save(client, path, views))["path"]
    assert saved.endswith("analysis.hspy")
    assert _save(client, path).status_code == 409  # exists
    assert _ok(_save(client, path, views, overwrite=True))["path"] == saved

    # HyperSpy reads the results it stored.
    stored = hs.load(saved)
    stored = stored if isinstance(stored, list) else [stored]
    assert len(stored) >= 3

    other, body = _reopen(saved)
    assert body["views"] == views and body["warnings"] == []
    _same(client, other)
    # The raw SI is rebinned again, not stored.
    ds = other.app.state.session.get_dataset(si["id"])
    assert ds.rebinned and ds.signal.data.shape[:2] == (25, 20)

    # A project can't be opened as data.
    r = client.post("/api/files", json={"path": saved})
    assert r.status_code == 422 and "Open analysis" in r.text


def test_new_dataset_copies(client, tmp_path, eels_cuzn):
    raw = shutil.copy(eels_cuzn, tmp_path / "cuzn.hspy")
    si = _si(_ok(client.post("/api/files", json={"path": str(raw)})))
    _ok(client.post(f"/api/datasets/{si['id']}/rebin", json={"factors": [2, 2, 1], "in_place": False}))
    saved = _ok(_save(client, tmp_path / "a.hspy"))["path"]
    other, _ = _reopen(saved)
    _same(client, other)


def test_stack_alignment_is_redone(client, tmp_path):
    tifffile.imwrite(tmp_path / "drift.tif", _drifting_stack(SHIFTS))
    ds = _ok(client.post("/api/files", json={"path": str(tmp_path / "drift.tif")}))["datasets"][0]
    url = f"/api/datasets/{ds['id']}/registration"
    client.post(url, json={})
    _wait(client, client.post(f"{url}/shifts").json())
    client.post(url + "/correct", json={"outliers": {"method": "none", "off_diagonal": 2}})
    _ok(client.post(f"{url}/apply", json={}))
    saved = _ok(_save(client, tmp_path / "a.hspy"))["path"]
    other, _ = _reopen(saved)
    _same(client, other)
    assert other.app.state.session.get_dataset(ds["id"]).alignment is not None


def test_zero_loss_alignment_is_redone(client, tmp_path):
    _, ll, hl = _open_pair(client, tmp_path)
    _wait(client, _ok(client.post(f"/api/datasets/{hl['id']}/zlp-alignment", json={"options": {}})))
    saved = _ok(_save(client, tmp_path / "a.hspy"))["path"]
    other, _ = _reopen(saved)
    _same(client, other)


def test_missing_and_changed_raw_data(client, tmp_path):
    s = hs.signals.Signal1D(np.random.default_rng(0).random((4, 5, 30)).astype(np.float32))
    s.set_signal_type("EELS")
    raw = tmp_path / "raw.hspy"
    s.save(raw)
    _ok(client.post("/api/files", json={"path": str(raw)}))
    saved = _ok(_save(client, tmp_path / "a.hspy"))["path"]

    # Changed since: opened, with a warning.
    s.data[0, 0, 0] += 1
    s.save(raw, overwrite=True)
    _, body = _reopen(saved)
    assert len(body["warnings"]) == 1 and "raw.hspy" in body["warnings"][0]

    # Moved together with the project: found next to it.
    moved = tmp_path / "moved"
    moved.mkdir()
    shutil.move(raw, moved / "raw.hspy")
    shutil.move(saved, moved / "a.hspy")
    _reopen(moved / "a.hspy")

    # Gone: nothing opened, and what was open stays open.
    (moved / "raw.hspy").unlink()
    r = client.post("/api/project/open", json={"path": str(moved / "a.hspy")})
    assert r.status_code == 422 and "raw.hspy" in r.text
    assert len(client.get("/api/files").json()) == 1


def test_not_a_project(client, tmp_path):
    s = hs.signals.Signal1D(np.zeros((3, 10), np.float32))
    s.save(tmp_path / "plain.hspy")
    r = client.post("/api/project/open", json={"path": str(tmp_path / "plain.hspy")})
    assert r.status_code == 422
    _ok(client.post("/api/files", json={"path": str(tmp_path / "plain.hspy")}))
    # Not overwritten either.
    r = _save(client, tmp_path / "plain.hspy", overwrite=True)
    assert r.status_code == 409


# ---------- "Include results": opening without the raw data ----------


def test_results_open_without_the_raw_data(client, tmp_path, eels_cuzn):
    raw = shutil.copy(eels_cuzn, tmp_path / "cuzn.hspy")
    si = _si(_ok(client.post("/api/files", json={"path": str(raw)})))
    _ok(client.post(f"/api/datasets/{si['id']}/rebin", json={"factors": [2, 2, 1]}))
    _ok(client.post(f"/api/datasets/{si['id']}/window-maps", json={"lo": 940, "hi": 980}))
    sizes = _ok(client.post("/api/project/estimate", json={}))
    assert sizes["results"] > sizes["link"] > 0
    saved = _ok(_save(client, tmp_path / "a.hspy", level="results"))["path"]
    (tmp_path / "cuzn.hspy").unlink()

    other, body = _reopen(saved)
    assert len(body["warnings"]) == 1 and "not found" in body["warnings"][0]
    assert body["files"][0]["raw_missing"]
    _same(client, other)
    # It can be worked on, and saved again (its data stored whatever the level).
    _ok(other.post(f"/api/datasets/{si['id']}/window-maps", json={"lo": 1000, "hi": 1040}))
    again = _ok(_save(other, tmp_path / "b.hspy"))["path"]
    third, _ = _reopen(again)
    _same(other, third)


def test_changed_raw_data_uses_what_was_stored(client, tmp_path):
    s = hs.signals.Signal1D(np.random.default_rng(0).random((4, 5, 30)).astype(np.float32))
    s.set_signal_type("EELS")
    raw = tmp_path / "raw.hspy"
    s.save(raw)
    _ok(client.post("/api/files", json={"path": str(raw)}))
    saved = _ok(_save(client, tmp_path / "a.hspy", level="results"))["path"]
    s.data[:] = 0
    s.save(raw, overwrite=True)
    other, body = _reopen(saved)
    assert "changed" in body["warnings"][0]
    _same(client, other)


def test_results_keep_an_aligned_stack(client, tmp_path):
    tifffile.imwrite(tmp_path / "drift.tif", _drifting_stack(SHIFTS))
    ds = _ok(client.post("/api/files", json={"path": str(tmp_path / "drift.tif")}))["datasets"][0]
    url = f"/api/datasets/{ds['id']}/registration"
    client.post(url, json={})
    _wait(client, client.post(f"{url}/shifts").json())
    client.post(url + "/correct", json={"outliers": {"method": "none", "off_diagonal": 2}})
    _ok(client.post(f"{url}/apply", json={}))
    saved = _ok(_save(client, tmp_path / "a.hspy", level="results"))["path"]
    (tmp_path / "drift.tif").unlink()
    other, _ = _reopen(saved)
    _same(client, other)
    stack = other.app.state.session.get_dataset(ds["id"])
    assert stack.alignment is not None and stack.signal is stack.aligned_signal
    np.testing.assert_allclose(
        stack.original_signal.data, client.app.state.session.get_dataset(ds["id"]).original_signal.data)


def test_unknown_level(client, tmp_path, eels_cuzn):
    _ok(client.post("/api/files", json={"path": str(eels_cuzn)}))
    assert _save(client, tmp_path / "a.hspy", level="all").status_code == 422


def test_results_keep_the_range_an_in_situ_series_showed(tmp_path, fake_dm4):
    from si_viewer.project import open_project, save_project
    from si_viewer.session import Session

    from test_insitu import open_pair

    session, si, _, frames, _ = open_pair(tmp_path, fake_dm4)
    total = si.insitu.reduce("sum", 0, 3)
    session.apply_insitu_range(si, total, "sum", 0, 3)
    saved = save_project(session, tmp_path / "a.hspy", level="results")
    for p in list(tmp_path.iterdir()):
        if p.name != "a.hspy":
            p.unlink() if p.is_file() else shutil.rmtree(p)

    other = Session()
    _, warnings = open_project(other, saved)
    assert warnings and all("not found" in w for w in warnings)
    ds = other.get_dataset(si.id)
    assert ds.insitu is None and other.files[si.file_id].raw_missing
    np.testing.assert_allclose(ds.signal.data, np.stack(frames).astype(np.float64).sum(axis=0), rtol=1e-6)


# ---------- "Include raw data" and locating raw files ----------


def test_embedded_raw_data_is_used_when_the_file_is_gone(client, tmp_path, eels_cuzn):
    raw = shutil.copy(eels_cuzn, tmp_path / "cuzn.hspy")
    si = _si(_ok(client.post("/api/files", json={"path": str(raw)})))
    _ok(client.post(f"/api/datasets/{si['id']}/rebin", json={"factors": [2, 2, 1]}))
    _ok(client.post(f"/api/datasets/{si['id']}/window-maps", json={"lo": 940, "hi": 980}))
    sizes = _ok(client.post("/api/project/estimate", json={}))
    assert sizes["raw"] > (tmp_path / "cuzn.hspy").stat().st_size
    saved = _ok(_save(client, tmp_path / "a.hspy", level="raw"))["path"]
    (tmp_path / "cuzn.hspy").unlink()

    other, body = _reopen(saved)
    f = body["files"][0]
    assert f["embedded"] and not f["raw_missing"] and body["warnings"] == []
    _same(client, other)  # rebinned again from the embedded copy
    # Saved again at any level, the copy is embedded again, and the raw file
    # is still referenced where it was.
    again = _ok(_save(other, tmp_path / "b.hspy"))["path"]
    third, body = _reopen(again)
    assert body["files"][0]["embedded"] and body["files"][0]["path"].endswith("cuzn.hspy")
    _same(other, third)


def test_raw_files_are_located(client, tmp_path):
    def eels(name):
        s = hs.signals.Signal1D(np.random.default_rng(0).random((3, 4, 20)).astype(np.float32))
        s.set_signal_type("EELS")
        s.save(tmp_path / f"{name}.hspy")
        return _ok(client.post("/api/files", json={"path": str(tmp_path / f"{name}.hspy")}))

    a, b = eels("one"), eels("two")
    saved = _ok(_save(client, tmp_path / "a.hspy"))["path"]
    moved = tmp_path.parent / f"{tmp_path.name}-moved"
    moved.mkdir()
    for name in ("one", "two"):
        shutil.move(tmp_path / f"{name}.hspy", moved / f"{name}.hspy")

    missing = _ok(client.post("/api/project/check", json={"path": saved}))["missing"]
    assert [m["name"] for m in missing] == ["one.hspy", "two.hspy"]
    # Locating one finds the other in its folder.
    locations = {a["id"]: str(moved / "one.hspy")}
    assert _ok(client.post("/api/project/check", json={"path": saved, "locations": locations}))["missing"] == []
    other = _client()
    body = _ok(other.post("/api/project/open", json={"path": saved, "locations": locations}))
    assert [f["id"] for f in body["files"]] == [a["id"], b["id"]] and body["warnings"] == []
    assert other.post("/api/project/open", json={"path": saved}).status_code == 422


def test_in_situ_series_are_never_embedded(tmp_path, fake_dm4):
    import json

    import h5py

    from si_viewer.project import GROUP, save_project
    from test_insitu import open_pair

    session, si, _, _, _ = open_pair(tmp_path, fake_dm4)
    saved = save_project(session, tmp_path / "a.hspy", level="raw")
    with h5py.File(saved, "r") as f:
        files = json.loads(f[GROUP].attrs["project"])["files"]
        assert "raw" not in f[GROUP]
    assert all(fe["stored"] and not fe["embedded"] for fe in files)


# ---------- saving one workspace; opening next to what is open ----------


def _eels_file(client, tmp_path, name, offset=0.0):
    s = hs.signals.Signal1D(np.random.default_rng(0).random((3, 4, 20)).astype(np.float32))
    s.set_signal_type("EELS")
    s.axes_manager[-1].offset = offset
    s.save(tmp_path / f"{name}.hspy")
    return _ok(client.post("/api/files", json={"path": str(tmp_path / f"{name}.hspy")}))


def test_only_the_files_asked_for_are_saved(client, tmp_path):
    one, two = _eels_file(client, tmp_path, "one"), _eels_file(client, tmp_path, "two")
    _ok(client.post(f"/api/datasets/{_si(one)['id']}/window-maps", json={"lo": 5, "hi": 10}))
    sizes = _ok(client.post("/api/project/estimate", json={"file_ids": [one["id"]]}))
    assert sizes["file_ids"] == [one["id"]]
    saved = _ok(_save(client, tmp_path / "a.hspy", file_ids=[one["id"]]))["path"]

    # Opened next to what is open, not twice.
    r = client.post("/api/project/open", json={"path": saved})
    assert r.status_code == 422 and "already open" in r.text
    assert client.delete(f"/api/files/{one['id']}").status_code == 204
    body = _ok(client.post("/api/project/open", json={"path": saved}))
    assert body["file_ids"] == [one["id"]]
    assert {f["id"] for f in body["files"]} == {one["id"], two["id"]}
    titles = [d["title"] for d in next(f for f in body["files"] if f["id"] == one["id"])["datasets"]]
    assert len(titles) == 2  # the SI and its window map


def test_linked_files_are_saved_with_it(client, tmp_path):
    ll, hl = _eels_file(client, tmp_path, "ll", -5), _eels_file(client, tmp_path, "hl", 400)
    _eels_file(client, tmp_path, "other")
    assert client.put(f"/api/datasets/{_si(ll)['id']}/eels-role", json={"role": "low_loss"}).status_code == 200
    r = client.put(f"/api/datasets/{_si(hl)['id']}/eels-partner", json={"partner_id": _si(ll)["id"]})
    assert r.status_code == 200, r.text
    sizes = _ok(client.post("/api/project/estimate", json={"file_ids": [hl["id"]]}))
    assert sorted(sizes["file_ids"]) == sorted([ll["id"], hl["id"]])
    saved = _ok(_save(client, tmp_path / "a.hspy", file_ids=[hl["id"]]))["path"]
    other, body = _reopen(saved)
    assert sorted(f["id"] for f in body["files"]) == sorted([ll["id"], hl["id"]])
    hl_ds = other.app.state.session.get_dataset(_si(hl)["id"])
    assert hl_ds.eels_linked == [_si(ll)["id"]]
