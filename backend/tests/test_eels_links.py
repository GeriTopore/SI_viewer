"""Linked EELS spectrum images (M10c): roles, link groups, and shared processing."""

import exspy
import numpy as np
import pytest
from fastapi.testclient import TestClient

from si_viewer.app import create_app

ACQ = "ImageList.TagGroup0.ImageTags.SI.Acquisition"
DUAL = "ImageList.TagGroup0.ImageTags.EELS.Acquisition"
SAME_ACQUISITION = {f"{ACQ}.Date": "11/07/2026", f"{ACQ}.Start time": "10:15:00"}


@pytest.fixture
def client():
    return TestClient(create_app())


def _open(client, path) -> dict:
    r = client.post("/api/files", json={"path": str(path)})
    assert r.status_code == 200, r.text
    return r.json()


def _files(client) -> dict:
    return {f["id"]: f for f in client.get("/api/files").json()}


def _listed(client) -> dict:
    """Every dataset's current summary by id."""
    return {d["id"]: d for f in _files(client).values() for d in f["datasets"]}


def _si(entry) -> dict:
    return next(d for d in entry["datasets"] if d["kind"] == "spectrum_image")


def _link(client, ds_id, partner_id) -> dict:
    """Link by hand (nothing is ever linked on opening)."""
    r = client.put(f"/api/datasets/{ds_id}/eels-partner", json={"partner_id": partner_id})
    assert r.status_code == 200, r.text
    return r.json()


# ---------- synthetic 2D spectrum images ----------


def _signal(title, low=False, shape=(6, 8), scale=1.0, beam=80.0, tags=None, role=None, offset=None):
    """A small EELS spectrum image; ``low`` puts a ZLP at 0 eV (which means
    nothing to the viewer), ``role`` writes DM's dual-acquisition marker."""
    rng = np.random.default_rng(0 if low else 1)
    start = offset if offset is not None else (-20 if low else 400)
    energy = np.arange(start, start + 80, 1.0)
    base = np.exp(-0.5 * (energy / 2.0) ** 2) * 1000 if low else 50 * np.exp(-(energy - start) / 40)
    data = (base + rng.random(shape + (len(energy),))).astype("float32")
    s = exspy.signals.EELSSpectrum(data)
    s.metadata.General.title = title
    s.axes_manager.signal_axes[0].offset = float(energy[0])
    s.axes_manager.signal_axes[0].scale = 1.0
    s.axes_manager.signal_axes[0].units = "eV"
    for ax in s.axes_manager.navigation_axes:
        ax.scale, ax.units = scale, "nm"
    if beam is not None:
        s.metadata.set_item("Acquisition_instrument.TEM.beam_energy", beam)
    tags = dict(tags or {})
    if role is not None:
        tags[f"{DUAL}.Is dual acquire low-loss"] = int(role == "low_loss")
    for path, value in tags.items():
        s.original_metadata.set_item(path, value)
    return s


def _save(tmp_path, name, title, low, **kw):
    path = tmp_path / f"{name}.hspy"
    _signal(title, low, **kw).save(path)
    return path


def _pair(tmp_path, ll_name="x_ll", hl_name="x_hl", ll_title="x (low-loss)", hl_title="x (high-loss)", **kw):
    return (
        _save(tmp_path, ll_name, ll_title, True, **kw),
        _save(tmp_path, hl_name, hl_title, False, **kw),
    )


def _trio(tmp_path, **kw):
    """One low loss and two core losses of one acquisition (by DM tags),
    their names unrelated."""
    kw = {"tags": SAME_ACQUISITION, **kw}
    return (
        _save(tmp_path, "one", "first", True, role="low_loss", **kw),
        _save(tmp_path, "two", "second", False, role="core_loss", offset=400, **kw),
        _save(tmp_path, "three", "third", False, role="core_loss", offset=800, **kw),
    )


def _group_of(client, ds_id) -> set:
    d = _listed(client)[ds_id]
    return {ds_id, *d["eels"]["linked"]}


# ---------- roles ----------


def test_roles_without_markers_are_core_loss_by_default(client, eels_linescan, eels_linescan_low_loss, eels_cuzn, eels_dm4):
    # No role guessing: not even the low-loss example (ZLP at 0 eV) is a low loss.
    ids = [_si(_open(client, p))["id"] for p in (eels_linescan_low_loss, eels_linescan, eels_cuzn, eels_dm4)]
    listed = _listed(client)
    for i in ids:
        eels = listed[i]["eels"]
        assert (eels["role"], eels["role_source"]) == ("core_loss", "default")
        assert set(eels) == {
            "role", "role_source", "linked", "beam_energy", "convergence_angle", "collection_angle",
        }
    assert listed[ids[0]]["eels"]["beam_energy"] == 80


def test_role_comes_from_the_dual_acquisition_tag(client, tmp_path):
    ll = _save(tmp_path, "a", "a", False, role="low_loss")  # a core-loss-looking spectrum marked low loss
    hl = _save(tmp_path, "b", "b", True, role="core_loss", shape=(3, 3))  # a ZLP marked core loss
    plain = _save(tmp_path, "c", "c", True, shape=(2, 2))
    a, b, c = (_si(_open(client, p)) for p in (ll, hl, plain))
    assert (a["eels"]["role"], a["eels"]["role_source"]) == ("low_loss", "metadata")
    assert (b["eels"]["role"], b["eels"]["role_source"]) == ("core_loss", "metadata")
    assert (c["eels"]["role"], c["eels"]["role_source"]) == ("core_loss", "default")


def test_non_eels_has_no_eels_info(client, example_emd):
    for d in client.post("/api/examples/fept-si").json()["datasets"]:
        assert d["eels"] is None, d["title"]


def test_set_role_by_hand(client, tmp_path):
    ll, hl = _pair(tmp_path)  # no markers: both core loss
    low_entry, core_entry = _open(client, ll), _open(client, hl)
    low, core = _si(low_entry), _si(core_entry)
    assert _group_of(client, low["id"]) == {low["id"]}
    _link(client, low["id"], core["id"])
    assert _group_of(client, low["id"]) == {low["id"], core["id"]}

    r = client.put(f"/api/datasets/{low['id']}/eels-role", json={"role": "low_loss"})
    assert r.status_code == 200, r.text
    by_id = {d["id"]: d for f in r.json()["files"] for d in f["datasets"]}
    assert {f["id"] for f in r.json()["files"]} == {low_entry["id"], core_entry["id"]}
    assert (by_id[low["id"]]["eels"]["role"], by_id[low["id"]]["eels"]["role_source"]) == ("low_loss", "user")
    assert by_id[core["id"]]["eels"]["role"] == "core_loss"
    assert by_id[core["id"]]["eels"]["role_source"] == "default"
    assert by_id[core["id"]]["eels"]["linked"] == [low["id"]]

    # One low loss per group.
    r = client.put(f"/api/datasets/{core['id']}/eels-role", json={"role": "low_loss"})
    assert r.status_code == 400
    assert r.json()["detail"] == f"{low['title']} is already the low loss of this group"
    assert _listed(client)[core["id"]]["eels"]["role"] == "core_loss"

    # Marking it as a core loss again is fine, and then the other can take over.
    assert client.put(f"/api/datasets/{low['id']}/eels-role", json={"role": "core_loss"}).status_code == 200
    assert client.put(f"/api/datasets/{core['id']}/eels-role", json={"role": "low_loss"}).status_code == 200
    assert _listed(client)[core["id"]]["eels"]["linked"] == [low["id"]]

    assert client.put(f"/api/datasets/{low['id']}/eels-role", json={"role": "nope"}).status_code == 422
    assert client.put("/api/datasets/nope/eels-role", json={"role": "low_loss"}).status_code == 404


def test_set_role_without_a_group_and_for_non_eels(client, tmp_path, example_emd):
    si = _si(_open(client, _save(tmp_path, "solo", "solo", False)))
    r = client.put(f"/api/datasets/{si['id']}/eels-role", json={"role": "low_loss"})
    assert r.status_code == 200
    assert r.json()["files"][0]["datasets"][0]["eels"]["role"] == "low_loss"
    eds = next(d for d in client.post("/api/examples/fept-si").json()["datasets"] if d["kind"] == "spectrum_image")
    assert client.put(f"/api/datasets/{eds['id']}/eels-role", json={"role": "low_loss"}).status_code == 400


# ---------- opening never links ----------


def test_opening_compatible_files_links_nothing(
    client, tmp_path, eels_linescan, eels_linescan_low_loss, eels_dm4_hl, eels_dm4_ll, eels_dm4, eels_cuzn
):
    # The LSMO pair (no role markers: two core losses).
    lsmo = [_si(_open(client, p)) for p in (eels_linescan, eels_linescan_low_loss)]
    # The dual-EELS DM4 pair: roles come from the metadata, the spectra are never read.
    hl = _si(_open(client, eels_dm4_hl))
    ll = _si(_open(client, eels_dm4_ll))
    old = _si(_open(client, eels_dm4))
    cuzn = _si(_open(client, eels_cuzn))
    # Pairs that every old sign would have linked: names, one acquisition by DM tags, a survey image,
    # sibling IDs.
    synthetic = []
    for sub, ll_kw, hl_kw in [
        ("names", {}, {}),
        ("acq", {"tags": SAME_ACQUISITION}, {"tags": SAME_ACQUISITION}),
        ("survey", {"tags": {f"{ACQ}.Survey Image.Unique Image ID": "ABC-123"}},
         {"tags": {f"{ACQ}.Survey Image.Unique Image ID": "ABC-123"}}),
        ("sibling", {"role": "low_loss", "tags": {f"{DUAL}.Dual acquire sibling.Unique Image ID": (7, 8, 9, 10)}},
         {"role": "core_loss", "tags": {f"{DUAL}.Dual acquire sibling.Unique Image ID": (1, 2, 3, 4)}}),
    ]:
        (tmp_path / sub).mkdir()
        synthetic += [
            _si(_open(client, _save(tmp_path / sub, "x_ll", "x (low-loss)", True, **ll_kw))),
            _si(_open(client, _save(tmp_path / sub, "x_hl", "x (high-loss)", False, **hl_kw))),
        ]

    assert hl["lazy"] and ll["lazy"]
    assert (hl["eels"]["role"], hl["eels"]["role_source"]) == ("core_loss", "metadata")
    assert (ll["eels"]["role"], ll["eels"]["role_source"]) == ("low_loss", "metadata")
    assert (old["eels"]["role"], old["eels"]["role_source"]) == ("core_loss", "default")
    assert (synthetic[6]["eels"]["role"], synthetic[7]["eels"]["role"]) == ("low_loss", "core_loss")

    listed = _listed(client)
    for d in [*lsmo, hl, ll, old, cuzn, *synthetic]:
        assert d["eels"]["linked"] == [], d["title"]
        assert listed[d["id"]]["eels"]["linked"] == [], d["title"]
    assert client.app.state.session.eels_groups == {}

    # They are only offered as candidates, none in a group.
    for d, other in ((lsmo[0], lsmo[1]), (lsmo[1], lsmo[0]), (hl, ll), (ll, hl)):
        partners = client.get(f"/api/datasets/{d['id']}/eels-partners").json()
        assert {"id": other["id"], "title": other["title"], "file_id": other["file_id"],
                "role": listed[other["id"]]["eels"]["role"], "group": None} in partners
        assert all(p["group"] is None for p in partners)
    assert client.get(f"/api/datasets/{cuzn['id']}/eels-partners").json() == []  # shape mismatch


def test_marking_the_low_loss_puts_it_first(client, eels_linescan, eels_linescan_low_loss):
    core, low = _si(_open(client, eels_linescan)), _si(_open(client, eels_linescan_low_loss))
    _link(client, core["id"], low["id"])
    assert client.put(f"/api/datasets/{low['id']}/eels-role", json={"role": "low_loss"}).status_code == 200
    listed = _listed(client)
    assert listed[core["id"]]["eels"]["linked"] == [low["id"]]
    assert listed[low["id"]]["eels"]["linked"] == [core["id"]]
    session = client.app.state.session
    assert session.eels_groups[session.get_dataset(low["id"]).eels_group][0] == low["id"]


def test_manual_unlink_and_link(client, eels_linescan, eels_linescan_low_loss):
    core_entry, low_entry = _open(client, eels_linescan), _open(client, eels_linescan_low_loss)
    core, low = _si(core_entry), _si(low_entry)
    _link(client, core["id"], low["id"])
    assert len(client.app.state.session.eels_groups) == 1

    r = client.put(f"/api/datasets/{core['id']}/eels-partner", json={"partner_id": None})
    assert r.status_code == 200
    assert {f["id"] for f in r.json()["files"]} == {core_entry["id"], low_entry["id"]}
    assert all(d["eels"]["linked"] == [] for f in r.json()["files"] for d in f["datasets"])
    assert client.app.state.session.eels_groups == {}
    listed = client.get(f"/api/datasets/{core['id']}/eels-partners").json()
    assert listed == [
        {"id": low["id"], "title": low["title"], "file_id": low_entry["id"], "role": "core_loss", "group": None}
    ]

    r = client.put(f"/api/datasets/{low['id']}/eels-partner", json={"partner_id": core["id"]})
    assert r.status_code == 200
    by_id = {d["id"]: d for f in r.json()["files"] for d in f["datasets"]}
    assert by_id[core["id"]]["eels"]["linked"] == [low["id"]]
    assert by_id[low["id"]]["eels"]["linked"] == [core["id"]]
    assert len(client.app.state.session.eels_groups) == 1

    # Linking what is linked already changes nothing.
    assert client.put(f"/api/datasets/{low['id']}/eels-partner", json={"partner_id": core["id"]}).status_code == 200
    assert len(client.app.state.session.eels_groups) == 1

    assert client.put("/api/datasets/nope/eels-partner", json={"partner_id": None}).status_code == 404
    assert client.put(f"/api/datasets/{low['id']}/eels-partner", json={"partner_id": "nope"}).status_code == 404
    assert client.get("/api/datasets/nope/eels-partners").status_code == 404


def test_incompatible_link_is_refused(client, eels_linescan, eels_linescan_low_loss, eels_cuzn, example_emd):
    core, low = _si(_open(client, eels_linescan)), _si(_open(client, eels_linescan_low_loss))
    cuzn = _si(_open(client, eels_cuzn))
    _link(client, core["id"], low["id"])
    groups = {k: list(v) for k, v in client.app.state.session.eels_groups.items()}
    # Shape mismatch.
    assert client.put(f"/api/datasets/{cuzn['id']}/eels-partner", json={"partner_id": low["id"]}).status_code == 400
    # Not EELS at all.
    eds = next(d for d in client.post("/api/examples/fept-si").json()["datasets"] if d["kind"] == "spectrum_image")
    assert client.put(f"/api/datasets/{eds['id']}/eels-partner", json={"partner_id": low["id"]}).status_code == 400
    assert client.put(f"/api/datasets/{core['id']}/eels-partner", json={"partner_id": eds["id"]}).status_code == 400
    assert client.get(f"/api/datasets/{eds['id']}/eels-partners").status_code == 400
    # Nothing changed.
    assert client.app.state.session.eels_groups == groups


def test_closing_a_file_drops_the_link(client, eels_linescan, eels_linescan_low_loss):
    core_entry, low_entry = _open(client, eels_linescan), _open(client, eels_linescan_low_loss)
    _link(client, _si(core_entry)["id"], _si(low_entry)["id"])
    assert client.app.state.session.eels_groups
    assert client.delete(f"/api/files/{low_entry['id']}").status_code == 204
    assert client.app.state.session.eels_groups == {}
    core = _files(client)[core_entry["id"]]["datasets"][0]
    assert core["eels"]["linked"] == []


def test_changing_the_technique_drops_the_link(client, eels_linescan, eels_linescan_low_loss):
    core_entry, low_entry = _open(client, eels_linescan), _open(client, eels_linescan_low_loss)
    core, low = _si(core_entry), _si(low_entry)
    _link(client, core["id"], low["id"])
    r = client.put(f"/api/datasets/{core['id']}/technique", json={"technique": "EDS"})
    assert r.status_code == 200
    body = r.json()
    assert body["dataset"]["eels"] is None
    assert [p["id"] for p in body["partners"]] == [low["id"]]
    assert body["partners"][0]["eels"]["linked"] == []
    assert [f["id"] for f in body["partner_files"]] == [low_entry["id"]]
    assert client.app.state.session.eels_groups == {}
    # Back to EELS: the role is read again (no link, though).
    r = client.put(f"/api/datasets/{core['id']}/technique", json={"technique": "EELS"})
    assert r.json()["dataset"]["eels"]["role"] == "core_loss"
    assert r.json()["dataset"]["eels"]["linked"] == []
    assert r.json()["partners"] == []


def test_technique_change_resets_a_role_set_by_hand(client, tmp_path):
    path = _save(tmp_path, "solo", "solo", True, role="core_loss")
    si = _si(_open(client, path))
    client.put(f"/api/datasets/{si['id']}/eels-role", json={"role": "low_loss"})
    assert _listed(client)[si["id"]]["eels"]["role_source"] == "user"
    client.put(f"/api/datasets/{si['id']}/technique", json={"technique": "EDS"})
    r = client.put(f"/api/datasets/{si['id']}/technique", json={"technique": "EELS"}).json()
    assert (r["dataset"]["eels"]["role"], r["dataset"]["eels"]["role_source"]) == ("core_loss", "metadata")


# ---------- synthetic spectrum images: shared processing ----------


def test_rebin_and_spatial_filter_follow_the_group(client, tmp_path):
    entries = [_open(client, p) for p in _trio(tmp_path)]
    low, core, core2 = (_si(e) for e in entries)
    _link(client, core["id"], low["id"])
    _link(client, core2["id"], low["id"])
    assert _group_of(client, low["id"]) == {low["id"], core["id"], core2["id"]}
    session = client.app.state.session
    raw_low = np.array(session.get_dataset(low["id"]).signal.data)

    r = client.post(f"/api/datasets/{core['id']}/rebin", json={"factors": [2, 2, 1]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dataset"]["navigation_shape"] == [4, 3]
    assert {p["id"] for p in body["partners"]} == {low["id"], core2["id"]}
    for p in body["partners"]:
        assert p["navigation_shape"] == [4, 3] and p["rebinned"]
    assert next(p for p in body["partners"] if p["id"] == low["id"])["signal_shape"] == low["signal_shape"]
    assert {f["id"] for f in body["partner_files"]} == {entries[0]["id"], entries[2]["id"]}
    assert body["dataset"]["eels"]["linked"] == [low["id"], core2["id"]]
    binned = session.get_dataset(low["id"]).signal.data
    np.testing.assert_allclose(binned, raw_low.reshape(3, 2, 4, 2, -1).sum(axis=(1, 3)), rtol=1e-5)

    # Rebinning from the low loss moves the core losses along too.
    r = client.post(f"/api/datasets/{low['id']}/rebin", json={"factors": [1, 2, 1]}).json()
    assert r["dataset"]["navigation_shape"] == [2, 3]
    assert {p["id"] for p in r["partners"]} == {core["id"], core2["id"]}
    assert all(p["navigation_shape"] == [2, 3] for p in r["partners"])

    # The energy factor applies to the others too.
    r = client.post(f"/api/datasets/{core['id']}/rebin", json={"factors": [1, 1, 2]}).json()
    assert r["dataset"]["signal_shape"] == [40] and len(r["dataset"]["rebinned"]) == 3
    assert all(p["signal_shape"] == [40] and len(p["rebinned"]) == 3 for p in r["partners"])

    # The spatial filter follows; the spectral smoothing stays per spectrum image.
    r = client.put(
        f"/api/datasets/{core['id']}/filters",
        json={"spatial": "gaussian", "sigma": 1.5, "spectral": "gaussian", "spectral_sigma": 2.0},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dataset"]["filters"]["spatial"] == "gaussian"
    assert len(body["partners"]) == 2
    for p in body["partners"]:
        assert p["filters"]["spatial"] == "gaussian" and p["filters"]["sigma"] == 1.5
        assert p["filters"]["spectral"] == "none"
    assert {f["id"] for f in body["partner_files"]} == {entries[0]["id"], entries[2]["id"]}
    assert session.get_dataset(low["id"]).filter_rev > 0

    # A spectral-only change does not touch the others' spatial filter.
    r = client.put(
        f"/api/datasets/{low['id']}/filters",
        json={"spatial": "gaussian", "sigma": 1.5, "spectral": "savgol"},
    ).json()
    assert all(p["filters"]["spectral"] == "gaussian" for p in r["partners"] if p["id"] == core["id"])
    assert all(p["filters"]["spatial"] == "gaussian" for p in r["partners"])

    # Removing the spatial filter removes it from all.
    r = client.put(f"/api/datasets/{core['id']}/filters", json={"spectral": "gaussian"}).json()
    assert all(p["filters"]["spatial"] == "none" for p in r["partners"])


def test_rebin_without_links_has_no_partners(client, tmp_path):
    path = _save(tmp_path, "solo", "solo (low-loss)", True)
    si = _si(_open(client, path))
    r = client.post(f"/api/datasets/{si['id']}/rebin", json={"factors": [2, 2, 1]}).json()
    assert r["partners"] == [] and r["partner_files"] == []


def test_rebin_with_a_group_in_one_file_bins_everything_once():
    import hyperspy.api as hs

    from si_viewer.session import Session

    session = Session()
    signals = [
        _signal("x (low-loss)", True), _signal("x (high-loss)"), _signal("x (high-loss 2)", offset=800),
        hs.signals.Signal2D(np.ones((6, 8))),
    ]
    from pathlib import Path

    entry = session.open_signals(Path("both.hspy"), signals)
    low, core, core2, image = (session.get_dataset(i) for i in entry.dataset_ids)
    assert core.eels_linked == []  # opening links nothing
    session.set_eels_partner(core, low)
    session.set_eels_partner(core2, low)
    assert set(core.eels_linked) == {low.id, core2.id}
    done = session.rebin(core, [2, 2, 1])
    assert [d.id for d in done] == [core.id, image.id, low.id, core2.id]
    for d in (core, low, core2):
        assert tuple(d.signal.axes_manager.navigation_shape) == (4, 3)
        assert len(d.rebinned) == 1
    assert tuple(image.signal.axes_manager.signal_shape) == (4, 3)
    assert len(image.rebinned) == 1


def test_rebin_too_coarse_for_a_linked_si_changes_nothing():
    from pathlib import Path

    from si_viewer.session import Session

    session = Session()
    short = _signal("x (low-loss)", True).isig[0:30]  # 30 channels, the core loss 80
    entry = session.open_signals(Path("both.hspy"), [short, _signal("x (high-loss)")])
    low, core = (session.get_dataset(i) for i in entry.dataset_ids)
    session.set_eels_partner(core, low)
    assert core.eels_linked == [low.id]
    with pytest.raises(ValueError, match=r"at most 30 .*\(linked\)"):
        session.rebin(core, [1, 1, 40])
    for d in (low, core):
        assert tuple(d.signal.axes_manager.navigation_shape) == (8, 6) and not d.rebinned


# ---------- link groups ----------


@pytest.mark.parametrize("order", [(0, 1, 2), (2, 1, 0), (1, 2, 0)])
def test_three_link_by_hand_in_any_order(client, tmp_path, order):
    paths = _trio(tmp_path)
    ids = {i: _si(_open(client, paths[i]))["id"] for i in range(3)}
    assert all(d["eels"]["linked"] == [] for d in _listed(client).values())
    for i in order[1:]:  # each joins the first one's group
        _link(client, ids[i], ids[order[0]])
    low, core, core2 = ids[0], ids[1], ids[2]
    listed = _listed(client)
    # Low loss first, then the core losses by energy offset.
    assert listed[low]["eels"]["linked"] == [core, core2]
    assert listed[core]["eels"]["linked"] == [low, core2]
    assert listed[core2]["eels"]["linked"] == [low, core]
    assert len(client.app.state.session.eels_groups) == 1


def test_several_in_one_file_link_only_by_hand():
    from pathlib import Path

    from si_viewer.session import Session

    session = Session()
    signals = [_signal("a", True, role="low_loss", tags=SAME_ACQUISITION),
               _signal("b", offset=400, tags=SAME_ACQUISITION),
               _signal("c", offset=800, tags=SAME_ACQUISITION)]
    entry = session.open_signals(Path("all.hspy"), signals)
    a, b, c = (session.get_dataset(i) for i in entry.dataset_ids)
    assert a.eels_linked == b.eels_linked == c.eels_linked == [] and session.eels_groups == {}
    session.set_eels_partner(b, a)
    session.set_eels_partner(c, a)
    assert a.eels_linked == [b.id, c.id] and b.eels_linked == [a.id, c.id]
    assert len(session.eels_groups) == 1


def test_group_with_two_low_losses_is_refused(client, tmp_path):
    other = tmp_path / "o"
    other.mkdir()
    ll1 = _save(tmp_path, "x_ll", "x (low-loss)", True, role="low_loss")
    ll2 = _save(other, "x_ll", "x (low-loss)", True, role="low_loss")
    hl = _save(tmp_path, "x_hl", "x (high-loss)", False, role="core_loss")
    a, b, c = (_si(_open(client, p)) for p in (ll1, ll2, hl))
    assert client.app.state.session.eels_groups == {}
    # Two low losses are not one group.
    r = client.put(f"/api/datasets/{b['id']}/eels-partner", json={"partner_id": a["id"]})
    assert r.status_code == 400 and "both low loss" in r.json()["detail"]
    # One low loss with the core loss works, the second is refused.
    _link(client, c["id"], a["id"])
    r = client.put(f"/api/datasets/{b['id']}/eels-partner", json={"partner_id": c["id"]})
    assert r.status_code == 400 and "both low loss" in r.json()["detail"]
    assert _group_of(client, a["id"]) == {a["id"], c["id"]}


def test_joining_leaves_the_old_group_and_merges_into_the_new_one(client, tmp_path):
    one, two, three = (_open(client, p) for p in _trio(tmp_path))
    low, core, core2 = _si(one), _si(two), _si(three)
    _link(client, core["id"], low["id"])
    _link(client, core2["id"], low["id"])
    # core2 leaves: low + core remain.
    assert client.put(f"/api/datasets/{core2['id']}/eels-partner", json={"partner_id": None}).status_code == 200
    assert _group_of(client, low["id"]) == {low["id"], core["id"]}
    assert _group_of(client, core2["id"]) == {core2["id"]}
    # core2 joins core's group.
    r = client.put(f"/api/datasets/{core2['id']}/eels-partner", json={"partner_id": core["id"]})
    assert r.status_code == 200
    assert _group_of(client, core2["id"]) == {low["id"], core["id"], core2["id"]}
    assert len(client.app.state.session.eels_groups) == 1
    # Unlinking one of three leaves a pair; unlinking again dissolves it.
    client.put(f"/api/datasets/{low['id']}/eels-partner", json={"partner_id": None})
    assert _group_of(client, core["id"]) == {core["id"], core2["id"]}
    client.put(f"/api/datasets/{core['id']}/eels-partner", json={"partner_id": None})
    assert client.app.state.session.eels_groups == {}
    assert all(d["eels"]["linked"] == [] for d in _listed(client).values())


def test_linking_a_second_low_loss_into_a_group_is_refused(client, tmp_path):
    one, two, three = (_open(client, p) for p in _trio(tmp_path))
    low, core, core2 = _si(one), _si(two), _si(three)
    _link(client, core["id"], low["id"])
    assert client.put(f"/api/datasets/{core2['id']}/eels-role", json={"role": "low_loss"}).status_code == 200
    r = client.put(f"/api/datasets/{core2['id']}/eels-partner", json={"partner_id": core["id"]})
    assert r.status_code == 400
    assert _group_of(client, core2["id"]) == {core2["id"]}
    assert _group_of(client, core["id"]) == {core["id"], low["id"]}


def test_closing_one_of_three_keeps_the_others_linked(client, tmp_path):
    one, two, three = (_open(client, p) for p in _trio(tmp_path))
    low, core, core2 = _si(one), _si(two), _si(three)
    _link(client, core["id"], low["id"])
    _link(client, core2["id"], low["id"])
    assert client.delete(f"/api/files/{two['id']}").status_code == 204
    listed = _listed(client)
    assert listed[low["id"]]["eels"]["linked"] == [core2["id"]]
    assert listed[core2["id"]]["eels"]["linked"] == [low["id"]]
    assert client.delete(f"/api/files/{three['id']}").status_code == 204
    assert _listed(client)[low["id"]]["eels"]["linked"] == []
    assert client.app.state.session.eels_groups == {}


# ---------- links are made by hand only ----------


def test_unrelated_pair_is_listed_but_not_linked(client, tmp_path):
    ll = _save(tmp_path, "alpha", "alpha scan", True)
    hl = _save(tmp_path, "beta", "beta scan", False)
    low, core = _si(_open(client, ll)), _si(_open(client, hl))
    assert core["eels"]["linked"] == []
    listed = client.get(f"/api/datasets/{core['id']}/eels-partners").json()
    assert [(p["id"], p["group"]) for p in listed] == [(low["id"], None)]
    r = client.put(f"/api/datasets/{core['id']}/eels-partner", json={"partner_id": low["id"]})
    assert r.status_code == 200
    by_id = {d["id"]: d for f in r.json()["files"] for d in f["datasets"]}
    assert by_id[low["id"]]["eels"]["linked"] == [core["id"]]


def test_two_core_losses_can_be_linked_by_hand(client, tmp_path):
    a = _si(_open(client, _save(tmp_path, "alpha", "alpha scan", False)))
    b = _si(_open(client, _save(tmp_path, "beta", "beta scan", False, offset=800)))
    assert client.put(f"/api/datasets/{a['id']}/eels-partner", json={"partner_id": b["id"]}).status_code == 200
    assert _group_of(client, a["id"]) == {a["id"], b["id"]}


def test_different_beam_energy_or_scale_is_incompatible(client, tmp_path):
    ll = _save(tmp_path, "x_ll", "x (low-loss)", True, beam=80.0)
    hl = _save(tmp_path, "x_hl", "x (high-loss)", False, beam=200.0)
    low, core = _si(_open(client, ll)), _si(_open(client, hl))
    assert core["eels"]["linked"] == []
    assert client.get(f"/api/datasets/{core['id']}/eels-partners").json() == []
    assert client.put(f"/api/datasets/{core['id']}/eels-partner", json={"partner_id": low["id"]}).status_code == 400

    other = tmp_path / "scale"
    other.mkdir()
    ll2 = _save(other, "y_ll", "y (low-loss)", True, scale=1.0)
    hl2 = _save(other, "y_hl", "y (high-loss)", False, scale=1.05)
    a, b = _si(_open(client, ll2)), _si(_open(client, hl2))
    assert b["eels"]["linked"] == [] and a["eels"]["linked"] == []


# ---------- the dual-EELS DM4 pair ----------


def test_dm4_pair_links_by_hand_without_reading_the_spectra(client, eels_dm4_ll, eels_dm4_hl):
    hl = _si(_open(client, eels_dm4_hl))
    ll = _si(_open(client, eels_dm4_ll))
    _link(client, hl["id"], ll["id"])
    listed = _listed(client)
    assert listed[hl["id"]]["eels"]["linked"] == [ll["id"]]
    assert listed[ll["id"]]["eels"]["linked"] == [hl["id"]]
    assert (listed[ll["id"]]["eels"]["role"], listed[hl["id"]]["eels"]["role"]) == ("low_loss", "core_loss")
    assert listed[hl["id"]]["lazy"] and listed[ll["id"]]["lazy"]
