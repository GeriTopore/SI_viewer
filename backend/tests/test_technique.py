"""EDS or EELS: detection, the summary fields and setting it by hand."""

from pathlib import Path

import hyperspy.api as hs
import numpy as np
import pytest
from fastapi.testclient import TestClient

from si_viewer.app import create_app
from si_viewer.processing.technique import detect_technique
from si_viewer.session import DatasetKind, Session


def _open(path):
    session = Session()
    entry = session.open_file(path)
    return session, [session.get_dataset(i) for i in entry.dataset_ids]


def _spectra(name, shape, offset, scale, units, signal_type=""):
    """A synthetic spectrum (or spectrum image) with a named energy axis."""
    s = hs.signals.Signal1D(np.random.default_rng(0).poisson(3.0, shape).astype("float32"))
    axis = s.axes_manager.signal_axes[0]
    axis.name, axis.offset, axis.scale, axis.units = name, offset, scale, units
    if signal_type:
        s.metadata.Signal.signal_type = signal_type
    s.metadata.General.title = "synthetic"
    return s


def _open_synthetic(signal):
    session = Session()
    entry = session.open_signals(Path("synthetic.hspy"), [signal])
    return session, session.get_dataset(entry.dataset_ids[0])


# ---------- the example files ----------


@pytest.mark.parametrize("fixture", ["eels_dm4", "eels_cuzn", "eels_linescan", "eels_linescan_low_loss"])
def test_eels_examples(fixture, request):
    _, [ds] = _open(request.getfixturevalue(fixture))
    s = ds.summary()
    assert (s["technique"], s["technique_source"]) == ("EELS", "metadata")
    assert s["eds"] is None


def test_velox_file(example_emd):
    _, datasets = _open(example_emd)
    for ds in datasets:
        s = ds.summary()
        if ds.kind in (DatasetKind.SPECTRUM, DatasetKind.SPECTRUM_IMAGE):
            assert (s["technique"], s["technique_source"]) == ("EDS", "metadata")
            assert s["eds"] is not None
        else:  # images, stacks and the maps computed from the SI
            assert (s["technique"], s["technique_source"]) == (None, None)
    assert any(d.origin == "computed" for d in datasets)  # EDS SIs still get their maps


# ---------- detection on synthetic signals ----------


def test_energy_loss_axis_is_eels():
    s = _spectra("Energy loss", (4, 5, 100), 100.0, 1.0, "eV")
    assert detect_technique(s) == ("EELS", "energy_axis")
    _, ds = _open_synthetic(s)
    assert ds.signal.metadata.Signal.signal_type == "EELS"
    assert ds.summary()["technique"] == "EELS" and ds.summary()["technique_source"] == "energy_axis"


def test_kev_axis_is_eds():
    s = _spectra("Energy", (4, 5, 2000), 0.0, 0.01, "keV")
    assert detect_technique(s) == ("EDS", "energy_axis")
    _, ds = _open_synthetic(s)
    summary = ds.summary()
    assert summary["technique"] == "EDS" and summary["technique_source"] == "energy_axis"
    assert summary["signal_type"] == "EDS_SEM" and summary["eds"] is not None
    assert type(ds.signal).__name__ == "EDSSEMSpectrum"


def test_eds_type_follows_instrument():
    s = _spectra("Energy", (3, 4, 1000), 0.0, 0.02, "keV")
    s.metadata.set_item("Acquisition_instrument.TEM.beam_energy", 200.0)
    _, ds = _open_synthetic(s)
    assert ds.signal.metadata.Signal.signal_type == "EDS_TEM"


def test_kev_axis_out_of_range_is_not_eds():
    assert detect_technique(_spectra("Energy", (3, 4, 100), 50.0, 1.0, "keV")) == (None, None)
    assert detect_technique(_spectra("Energy", (3, 4, 100), -3.0, 0.1, "keV")) == (None, None)


def test_ambiguous_ev_axis_is_unknown():
    s = _spectra("Energy", (4, 5, 500), -10.0, 1.0, "eV")
    assert detect_technique(s) == (None, None)
    _, ds = _open_synthetic(s)
    summary = ds.summary()
    assert summary["technique"] is None and summary["technique_source"] is None
    assert summary["signal_type"] == "" and summary["eds"] is None


def test_single_spectrum():
    assert detect_technique(_spectra("Energy loss", (100,), 0.0, 1.0, "eV")) == ("EELS", "energy_axis")


def test_dm_tag():
    for tag, expected in (("EELS", "EELS"), ("X-ray", "EDS")):
        s = _spectra("Energy", (3, 4, 100), -10.0, 1.0, "eV")  # axis alone wouldn't say
        s.original_metadata.add_dictionary(
            {"ImageList": {"TagGroup0": {"ImageTags": {"Meta Data": {"Signal": tag}}}}}
        )
        assert detect_technique(s) == (expected, "dm_tag")
    _, ds = _open_synthetic(s)
    assert ds.summary()["technique"] == "EDS" and ds.summary()["eds"] is not None


def test_dm_tag_missing_keys():
    s = _spectra("Energy", (3, 4, 100), -10.0, 1.0, "eV")
    s.original_metadata.add_dictionary({"ImageList": {"TagGroup0": {"ImageTags": {"Other": 1}}}})
    assert detect_technique(s) == (None, None)
    s.original_metadata.add_dictionary({"ImageList": 3})
    assert detect_technique(s) == (None, None)


def test_metadata_wins_over_axis():
    s = _spectra("Energy loss", (3, 4, 100), 0.0, 1.0, "eV", signal_type="EDS_TEM")
    assert detect_technique(s) == ("EDS", "metadata")


def test_other_signal_type_is_left_alone():
    s = _spectra("Energy loss", (3, 4, 100), 0.0, 1.0, "eV", signal_type="CL")
    assert detect_technique(s) == (None, None)


def test_images_have_no_technique():
    img = hs.signals.Signal2D(np.zeros((8, 8), dtype="float32"))
    assert detect_technique(img) == (None, None)
    stack = hs.signals.Signal2D(np.zeros((3, 8, 8), dtype="float32"))
    stack.metadata.Signal.signal_type = "EELS"
    assert detect_technique(stack) == (None, None)
    _, ds = _open_synthetic(stack)
    assert ds.summary()["technique"] is None


def test_detected_eds_line_scan_has_no_maps():
    s = _spectra("Energy", (6, 1000), 0.0, 0.02, "keV")
    session, ds = _open_synthetic(s)
    assert ds.is_line_scan and ds.summary()["technique"] == "EDS"
    assert len(session.files[ds.file_id].dataset_ids) == 1


# ---------- setting it by hand ----------


@pytest.fixture
def client():
    return TestClient(create_app())


def _open_api(client, signal):
    session = client.app.state.session
    entry = session.open_signals(Path("synthetic.hspy"), [signal])
    return session.get_dataset(entry.dataset_ids[0])


def _put(client, ds, technique):
    return client.put(f"/api/datasets/{ds.id}/technique", json={"technique": technique})


def test_set_unknown_spectrum_to_eds(client):
    ds = _open_api(client, _spectra("Energy", (4, 5, 300), -10.0, 1.0, "eV"))
    assert ds.summary()["eds"] is None
    r = _put(client, ds, "EDS")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dataset"]["technique"] == "EDS" and body["dataset"]["technique_source"] == "user"
    assert body["dataset"]["eds"] is not None and body["dataset"]["signal_type"] == "EDS_SEM"
    assert [d["id"] for d in body["file"]["datasets"]] == [ds.id]  # nothing computed
    assert type(ds.signal).__name__ == "EDSSEMSpectrum"


def test_set_eels_clears_eds(client):
    ds = _open_api(client, _spectra("Energy", (4, 5, 300), 0.0, 0.05, "keV"))
    assert ds.summary()["eds"] is not None
    before = ds.filter_rev
    r = _put(client, ds, "EELS")
    body = r.json()["dataset"]
    assert body["technique"] == "EELS" and body["technique_source"] == "user" and body["eds"] is None
    assert body["signal_type"] == "EELS" and body["filter_rev"] > before
    assert type(ds.signal).__name__ == "EELSSpectrum"


def test_set_null(client):
    ds = _open_api(client, _spectra("Energy loss", (4, 5, 100), 100.0, 1.0, "eV"))
    body = _put(client, ds, None).json()["dataset"]
    assert body["technique"] is None and body["technique_source"] == "user" and body["signal_type"] == ""
    assert ds.signal.axes_manager.signal_axes[0].name == "Energy loss"  # the data is untouched


def test_leaving_eds_removes_maps(client, example_emd):
    session = client.app.state.session
    entry = session.open_file(example_emd)
    si = next(
        session.get_dataset(i) for i in entry.dataset_ids
        if session.get_dataset(i).kind is DatasetKind.SPECTRUM_IMAGE
    )
    assert session.computed_maps(si)
    body = _put(client, si, "EELS").json()
    assert body["dataset"]["eds"] is None
    assert not session.computed_maps(si)
    assert all(d["origin"] != "computed" for d in body["file"]["datasets"])
    # and back to EDS: no maps until elements are picked
    body = _put(client, si, "EDS").json()
    assert body["dataset"]["eds"] is not None and not session.computed_maps(si)


def test_set_technique_clears_caches(client):
    ds = _open_api(client, _spectra("Energy", (4, 5, 300), -10.0, 1.0, "eV"))
    ds.cache["x"] = 1
    ds.thumbnails[64] = b"x"
    _put(client, ds, "EELS")
    assert not ds.cache and not ds.thumbnails


def test_set_technique_errors(client):
    img = _open_api(client, hs.signals.Signal2D(np.zeros((8, 8), dtype="float32")))
    assert _put(client, img, "EDS").status_code == 400
    ds = _open_api(client, _spectra("Energy", (4, 5, 300), -10.0, 1.0, "eV"))
    assert _put(client, ds, "CL").status_code == 422
    assert client.put("/api/datasets/nope/technique", json={"technique": "EDS"}).status_code == 404
