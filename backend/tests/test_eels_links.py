"""Low-loss / core-loss EELS pairs (M10c): roles, linking, and shared processing."""

import exspy
import numpy as np
import pytest
from fastapi.testclient import TestClient

from si_viewer.app import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


def _open(client, path) -> dict:
    r = client.post("/api/files", json={"path": str(path)})
    assert r.status_code == 200, r.text
    return r.json()


def _files(client) -> dict:
    return {f["id"]: f for f in client.get("/api/files").json()}


def _si(entry) -> dict:
    return next(d for d in entry["datasets"] if d["kind"] == "spectrum_image")


# ---------- synthetic 2D pairs ----------


def _save(tmp_path, name, title, low, shape=(6, 8), scale=1.0, beam=80.0, tags=None):
    """A small EELS spectrum image saved as .hspy; a low loss has its ZLP at 0 eV."""
    rng = np.random.default_rng(0 if low else 1)
    energy = np.arange(-20, 60, 1.0) if low else np.arange(400, 480, 1.0)
    base = np.exp(-0.5 * (energy / 2.0) ** 2) * 1000 if low else 50 * np.exp(-(energy - 400) / 40)
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
    for path, value in (tags or {}).items():
        s.original_metadata.set_item(path, value)
    path = tmp_path / f"{name}.hspy"
    s.save(path)
    return path


def _pair(tmp_path, ll_name="x_ll", hl_name="x_hl", ll_title="x (low-loss)", hl_title="x (high-loss)", **kw):
    return (
        _save(tmp_path, ll_name, ll_title, True, **kw),
        _save(tmp_path, hl_name, hl_title, False, **kw),
    )


# ---------- roles ----------


def test_roles_of_the_examples(client, eels_linescan, eels_linescan_low_loss, eels_cuzn, eels_dm4):
    roles = {
        "low": _si(_open(client, eels_linescan_low_loss)),
        "core": _si(_open(client, eels_linescan)),
        "cuzn": _si(_open(client, eels_cuzn)),
        "dm4": _si(_open(client, eels_dm4)),
    }
    # Read the roles back from the listing (opening the pair links them).
    listed = {d["id"]: d for f in _files(client).values() for d in f["datasets"]}
    assert listed[roles["low"]["id"]]["eels"]["role"] == "low_loss"
    for key in ("core", "cuzn", "dm4"):
        assert listed[roles[key]["id"]]["eels"]["role"] == "core_loss", key
    for d in listed.values():
        assert set(d["eels"]) == {"role", "partner_id", "beam_energy", "convergence_angle", "collection_angle"}
    assert listed[roles["low"]["id"]]["eels"]["beam_energy"] == 80


def test_non_eels_has_no_eels_info(client, example_emd):
    for d in client.post("/api/examples/fept-si").json()["datasets"]:
        assert d["eels"] is None, d["title"]


# ---------- the LSMO pair ----------


@pytest.mark.parametrize("low_first", [True, False])
def test_lsmo_pair_links_in_either_order(client, eels_linescan, eels_linescan_low_loss, low_first):
    paths = [eels_linescan_low_loss, eels_linescan] if low_first else [eels_linescan, eels_linescan_low_loss]
    first, second = (_si(_open(client, p)) for p in paths)
    files = _files(client)
    summaries = {d["id"]: d for f in files.values() for d in f["datasets"]}
    a, b = summaries[first["id"]], summaries[second["id"]]
    assert a["eels"]["partner_id"] == b["id"] and b["eels"]["partner_id"] == a["id"]
    assert {a["eels"]["role"], b["eels"]["role"]} == {"low_loss", "core_loss"}
    # The newly opened file's own summary already says so.
    assert second["eels"]["partner_id"] == first["id"]
    links = client.app.state.session.eels_links
    core, low = (a, b) if a["eels"]["role"] == "core_loss" else (b, a)
    assert links == {core["id"]: low["id"]}


def test_other_examples_are_not_linked(client, eels_linescan, eels_linescan_low_loss, eels_cuzn, eels_dm4):
    lsmo = [_si(_open(client, p)) for p in (eels_linescan, eels_linescan_low_loss)]
    others = [_si(_open(client, p)) for p in (eels_cuzn, eels_dm4)]
    for d in others:
        assert d["eels"]["partner_id"] is None
        assert client.get(f"/api/datasets/{d['id']}/eels-partners").json() == []
    for d in lsmo:  # shape mismatch with the 2D ones
        partners = client.get(f"/api/datasets/{d['id']}/eels-partners").json()
        assert [p["id"] for p in partners] == [lsmo[1 - lsmo.index(d)]["id"]]
        assert set(partners[0]) == {"id", "title", "file_id", "linked_to"}
        assert partners[0]["linked_to"] == d["id"]


def test_manual_unlink_and_link(client, eels_linescan, eels_linescan_low_loss):
    core_entry, low_entry = _open(client, eels_linescan), _open(client, eels_linescan_low_loss)
    core, low = _si(core_entry), _si(low_entry)

    r = client.put(f"/api/datasets/{core['id']}/eels-partner", json={"partner_id": None})
    assert r.status_code == 200
    assert {f["id"] for f in r.json()["files"]} == {core_entry["id"], low_entry["id"]}
    assert all(d["eels"]["partner_id"] is None for f in r.json()["files"] for d in f["datasets"])
    assert client.app.state.session.eels_links == {}
    listed = client.get(f"/api/datasets/{core['id']}/eels-partners").json()
    assert listed == [{"id": low["id"], "title": low["title"], "file_id": low_entry["id"], "linked_to": None}]

    r = client.put(f"/api/datasets/{low['id']}/eels-partner", json={"partner_id": core["id"]})
    assert r.status_code == 200
    by_id = {d["id"]: d for f in r.json()["files"] for d in f["datasets"]}
    assert by_id[core["id"]]["eels"]["partner_id"] == low["id"]
    assert by_id[low["id"]]["eels"]["partner_id"] == core["id"]
    assert client.app.state.session.eels_links == {core["id"]: low["id"]}

    assert client.put("/api/datasets/nope/eels-partner", json={"partner_id": None}).status_code == 404
    assert client.put(f"/api/datasets/{low['id']}/eels-partner", json={"partner_id": "nope"}).status_code == 404
    assert client.get("/api/datasets/nope/eels-partners").status_code == 404


def test_incompatible_link_is_refused(client, eels_linescan, eels_linescan_low_loss, eels_cuzn, example_emd):
    core, low = _si(_open(client, eels_linescan)), _si(_open(client, eels_linescan_low_loss))
    cuzn = _si(_open(client, eels_cuzn))
    # Shape mismatch.
    assert client.put(f"/api/datasets/{cuzn['id']}/eels-partner", json={"partner_id": low["id"]}).status_code == 400
    # Two core losses.
    assert client.put(f"/api/datasets/{cuzn['id']}/eels-partner", json={"partner_id": core["id"]}).status_code == 400
    # Not EELS at all.
    eds = next(d for d in client.post("/api/examples/fept-si").json()["datasets"] if d["kind"] == "spectrum_image")
    assert client.put(f"/api/datasets/{eds['id']}/eels-partner", json={"partner_id": low["id"]}).status_code == 400
    assert client.put(f"/api/datasets/{core['id']}/eels-partner", json={"partner_id": eds["id"]}).status_code == 400
    assert client.get(f"/api/datasets/{eds['id']}/eels-partners").status_code == 400
    # Nothing changed.
    assert client.app.state.session.eels_links == {core["id"]: low["id"]}


def test_closing_a_file_drops_the_link(client, eels_linescan, eels_linescan_low_loss):
    core_entry, low_entry = _open(client, eels_linescan), _open(client, eels_linescan_low_loss)
    assert client.app.state.session.eels_links
    assert client.delete(f"/api/files/{low_entry['id']}").status_code == 204
    assert client.app.state.session.eels_links == {}
    core = _files(client)[core_entry["id"]]["datasets"][0]
    assert core["eels"]["partner_id"] is None


def test_changing_the_technique_drops_the_link(client, eels_linescan, eels_linescan_low_loss):
    core_entry, low_entry = _open(client, eels_linescan), _open(client, eels_linescan_low_loss)
    core, low = _si(core_entry), _si(low_entry)
    r = client.put(f"/api/datasets/{core['id']}/technique", json={"technique": "EDS"})
    assert r.status_code == 200
    body = r.json()
    assert body["dataset"]["eels"] is None
    assert body["partner"]["eels"]["partner_id"] is None and body["partner_file"]["id"] == low_entry["id"]
    assert client.app.state.session.eels_links == {}
    # Back to EELS: the role is worked out again (no link, though).
    r = client.put(f"/api/datasets/{core['id']}/technique", json={"technique": "EELS"})
    assert r.json()["dataset"]["eels"]["role"] == "core_loss"
    assert r.json()["dataset"]["eels"]["partner_id"] is None


# ---------- synthetic 2D pairs: automatic linking and shared processing ----------


@pytest.mark.parametrize("low_first", [True, False])
def test_synthetic_pair_auto_links(client, tmp_path, low_first):
    ll, hl = _pair(tmp_path)
    first, second = (ll, hl) if low_first else (hl, ll)
    a, b = _si(_open(client, first)), _si(_open(client, second))
    assert b["eels"]["partner_id"] == a["id"]
    assert {b["eels"]["role"]} == ({"core_loss"} if low_first else {"low_loss"})
    assert _files(client)[a["file_id"]]["datasets"][0]["eels"]["partner_id"] == b["id"]


def test_rebin_and_spatial_filter_follow_the_partner(client, tmp_path):
    ll, hl = _pair(tmp_path)
    low_entry, core_entry = _open(client, ll), _open(client, hl)
    low, core = _si(low_entry), _si(core_entry)
    assert core["eels"]["partner_id"] == low["id"]
    session = client.app.state.session
    raw_low = np.array(session.get_dataset(low["id"]).signal.data)

    r = client.post(f"/api/datasets/{core['id']}/rebin", json={"factors": [2, 2, 1]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dataset"]["navigation_shape"] == [4, 3]
    assert body["partner"]["id"] == low["id"] and body["partner"]["navigation_shape"] == [4, 3]
    assert body["partner"]["signal_shape"] == low["signal_shape"]  # energy untouched
    assert body["partner"]["rebinned"] and body["partner_file"]["id"] == low_entry["id"]
    assert body["dataset"]["eels"]["partner_id"] == low["id"]
    assert session.eels_links == {core["id"]: low["id"]}
    binned = session.get_dataset(low["id"]).signal.data
    np.testing.assert_allclose(binned, raw_low.reshape(3, 2, 4, 2, -1).sum(axis=(1, 3)), rtol=1e-5)

    # Rebinning from the low loss moves the core loss along too.
    r = client.post(f"/api/datasets/{low['id']}/rebin", json={"factors": [1, 2, 1]}).json()
    assert r["dataset"]["navigation_shape"] == [2, 3]
    assert r["partner"]["navigation_shape"] == [2, 3] and r["partner"]["id"] == core["id"]

    # An energy-only rebin leaves the partner alone.
    r = client.post(f"/api/datasets/{core['id']}/rebin", json={"factors": [1, 1, 2]}).json()
    assert len(r["dataset"]["rebinned"]) == 3 and len(r["partner"]["rebinned"]) == 2

    # The spatial filter follows; the spectral smoothing stays per spectrum image.
    r = client.put(
        f"/api/datasets/{core['id']}/filters",
        json={"spatial": "gaussian", "sigma": 1.5, "spectral": "gaussian", "spectral_sigma": 2.0},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dataset"]["filters"]["spatial"] == "gaussian"
    assert body["partner"]["filters"]["spatial"] == "gaussian" and body["partner"]["filters"]["sigma"] == 1.5
    assert body["partner"]["filters"]["spectral"] == "none"
    assert body["partner_file"]["id"] == low_entry["id"]
    assert session.get_dataset(low["id"]).filter_rev > 0

    # A spectral-only change does not touch the partner's spatial filter.
    r = client.put(
        f"/api/datasets/{low['id']}/filters",
        json={"spatial": "gaussian", "sigma": 1.5, "spectral": "savgol"},
    ).json()
    assert r["partner"]["filters"]["spectral"] == "gaussian"
    assert r["partner"]["filters"]["spatial"] == "gaussian"

    # Removing the spatial filter removes it from both.
    r = client.put(f"/api/datasets/{core['id']}/filters", json={"spectral": "gaussian"}).json()
    assert r["partner"]["filters"]["spatial"] == "none"


def test_rebin_without_a_partner_has_null_partner(client, tmp_path):
    path = _save(tmp_path, "solo", "solo (low-loss)", True)
    si = _si(_open(client, path))
    r = client.post(f"/api/datasets/{si['id']}/rebin", json={"factors": [2, 2, 1]}).json()
    assert r["partner"] is None and r["partner_file"] is None


def test_rebin_with_pair_in_one_file_bins_everything_once(tmp_path):
    import hyperspy.api as hs

    from si_viewer.session import Session

    session = Session()
    ll, hl = _pair(tmp_path)
    signals = [hs.load(str(ll)), hs.load(str(hl)), hs.signals.Signal2D(np.ones((6, 8)))]
    entry = session.open_signals(tmp_path / "both.hspy", signals)
    low, core, image = (session.get_dataset(i) for i in entry.dataset_ids)
    assert core.eels_partner_id == low.id
    done = session.rebin(core, [2, 2, 1])
    assert [d.id for d in done] == [core.id, image.id, low.id]
    assert tuple(core.signal.axes_manager.navigation_shape) == (4, 3)
    assert tuple(low.signal.axes_manager.navigation_shape) == (4, 3)
    assert tuple(image.signal.axes_manager.signal_shape) == (4, 3)
    assert len(low.rebinned) == len(core.rebinned) == len(image.rebinned) == 1


# ---------- strong signs and no guessing ----------


def test_unrelated_pair_is_listed_but_not_linked(client, tmp_path):
    ll = _save(tmp_path, "alpha", "alpha scan", True)
    hl = _save(tmp_path, "beta", "beta scan", False)
    low, core = _si(_open(client, ll)), _si(_open(client, hl))
    assert core["eels"]["partner_id"] is None
    listed = client.get(f"/api/datasets/{core['id']}/eels-partners").json()
    assert [(p["id"], p["linked_to"]) for p in listed] == [(low["id"], None)]
    r = client.put(f"/api/datasets/{core['id']}/eels-partner", json={"partner_id": low["id"]})
    assert r.status_code == 200
    by_id = {d["id"]: d for f in r.json()["files"] for d in f["datasets"]}
    assert by_id[low["id"]]["eels"]["partner_id"] == core["id"]
    assert client.app.state.session.eels_links == {core["id"]: low["id"]}


def test_different_beam_energy_or_scale_is_incompatible(client, tmp_path):
    ll = _save(tmp_path, "x_ll", "x (low-loss)", True, beam=80.0)
    hl = _save(tmp_path, "x_hl", "x (high-loss)", False, beam=200.0)
    low, core = _si(_open(client, ll)), _si(_open(client, hl))
    assert core["eels"]["partner_id"] is None
    assert client.get(f"/api/datasets/{core['id']}/eels-partners").json() == []
    assert client.put(f"/api/datasets/{core['id']}/eels-partner", json={"partner_id": low["id"]}).status_code == 400

    other = tmp_path / "scale"
    other.mkdir()
    ll2 = _save(other, "y_ll", "y (low-loss)", True, scale=1.0)
    hl2 = _save(other, "y_hl", "y (high-loss)", False, scale=1.05)
    a, b = _si(_open(client, ll2)), _si(_open(client, hl2))
    assert b["eels"]["partner_id"] is None and a["eels"]["partner_id"] is None


def test_dm_tags_link_pairs_with_unrelated_names(client, tmp_path):
    acq = "ImageList.TagGroup0.ImageTags.SI.Acquisition"
    tags = {f"{acq}.Date": "11/07/2026", f"{acq}.Start time": "10:15:00"}
    ll = _save(tmp_path, "one", "first", True, tags=tags)
    hl = _save(tmp_path, "two", "second", False, tags=tags)
    low, core = _si(_open(client, ll)), _si(_open(client, hl))
    assert core["eels"]["partner_id"] == low["id"]

    other = tmp_path / "survey"
    other.mkdir()
    tags = {f"{acq}.Survey Image.Unique Image ID": "ABC-123"}
    ll = _save(other, "one", "first", True, tags=tags)
    hl = _save(other, "two", "second", False, tags=tags)
    low2, core2 = _si(_open(client, ll)), _si(_open(client, hl))
    assert core2["eels"]["partner_id"] == low2["id"]

    # Different acquisitions: not linked.
    third = tmp_path / "different"
    third.mkdir()
    ll = _save(third, "one", "first", True, tags={f"{acq}.Date": "d1", f"{acq}.Start time": "t1"})
    hl = _save(third, "two", "second", False, tags={f"{acq}.Date": "d1", f"{acq}.Start time": "t2"})
    low3, core3 = _si(_open(client, ll)), _si(_open(client, hl))
    assert core3["eels"]["partner_id"] is None


def test_no_guessing_among_several_candidates(client, tmp_path):
    # Two core losses fit one low loss by name: neither is linked.
    ll = _save(tmp_path, "x_ll", "x (low-loss)", True)
    other = tmp_path / "other"
    other.mkdir()
    hl1 = _save(tmp_path, "x_hl", "x (high-loss)", False)
    hl2 = _save(other, "x_hl", "x (high-loss)", False)
    _open(client, ll)
    a = _si(_open(client, hl1))
    assert a["eels"]["partner_id"] is not None  # the only candidate so far
    b = _si(_open(client, hl2))
    assert b["eels"]["partner_id"] is None  # the low loss is taken: never re-linked


def test_ambiguous_candidates_are_left_alone(client, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    ll1 = _save(tmp_path, "x_ll", "x (low-loss)", True)
    ll2 = _save(other, "x_ll", "x (low-loss)", True)
    hl = _save(tmp_path, "x_hl", "x (high-loss)", False)
    _open(client, ll1)
    _open(client, ll2)
    c = _si(_open(client, hl))
    assert c["eels"]["partner_id"] is None
    assert len(client.get(f"/api/datasets/{c['id']}/eels-partners").json()) == 2


@pytest.mark.parametrize(
    "a, b",
    [
        ("LSMO low-loss", "LSMO high-loss"),
        ("Scan_01_low_loss", "Scan 01 core-loss"),
        ("scan1 lowloss", "scan1 coreloss"),
        ("Sample LL", "Sample HL"),
        ("run (Low Loss)", "run (high loss)"),
    ],
)
def test_name_normalisation(a, b):
    from si_viewer.processing.eels import normalised_name

    assert normalised_name(a) == normalised_name(b) != ""
