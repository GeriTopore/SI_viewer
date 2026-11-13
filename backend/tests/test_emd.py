import numpy as np

from si_viewer.io.emd import load_emd, velox_format
from si_viewer.session import DatasetKind, Session, classify


def test_velox_format(example_emd):
    assert velox_format(example_emd) in ("Velox", "DevelopersKit", "VeloxLite")


def test_load_emd_kinds(example_emd):
    signals = load_emd(example_emd)
    kinds = [classify(s) for s in signals]
    assert DatasetKind.SPECTRUM_IMAGE in kinds
    assert DatasetKind.IMAGE in kinds

    si = signals[kinds.index(DatasetKind.SPECTRUM_IMAGE)]
    assert si.metadata.Signal.signal_type == "EDS_TEM"
    assert si.axes_manager.signal_axes[0].units == "keV"


def test_si_image_stack_loaded(example_emd):
    """Multi-frame images (the HAADF acquired with the SI) come back as stacks."""
    signals = load_emd(example_emd)
    stacks = [s for s in signals if classify(s) is DatasetKind.STACK]
    if not stacks:
        import pytest

        pytest.skip("Example file has no multi-frame images")
    assert stacks[0].axes_manager.navigation_size > 1


def test_session_open(example_emd):
    session = Session()
    entry = session.open_file(example_emd)
    assert entry.dataset_ids
    # Opening the same path again returns the same entry.
    assert session.open_file(example_emd) is entry

    si = next(
        session.datasets[d]
        for d in entry.dataset_ids
        if session.datasets[d].kind is DatasetKind.SPECTRUM_IMAGE
    )
    # Loaded on open (raw element maps are computed from it straight away).
    assert not si.lazy
    assert isinstance(si.signal.data, np.ndarray)
    assert si.summary()["navigation_shape"] == list(si.signal.axes_manager.navigation_shape)

    session.close_file(entry.id)
    assert not session.files and not session.datasets


def test_image_stacks_are_loaded_once_with_velox_names():
    """A Velox file of image stacks alone (HAADF and the four DF segments):
    each stack once, titled as Velox shows it."""
    from conftest import _example

    from si_viewer.io import load_file

    signals = load_file(_example("20261112_1302_2.25_Mx_HAADF-DF.emd"))
    titles = sorted(s.metadata.General.title for s in signals)
    assert titles == ["DF-A", "DF-B", "DF-C", "DF-D", "HAADF"]
    assert all(s.axes_manager.navigation_dimension == 1 for s in signals)
    seg = {s.metadata.General.title: s.metadata.get_item("General.original_title") for s in signals}
    assert seg["DF-A"] == "DFS(0,4)" and seg["DF-D"] == "DFS(3,7)"


def test_detector_segments_are_recorded():
    """Each segment image carries the segments summed into it, from Velox's
    acquisition record, and keeps them through a saved analysis."""
    import tempfile

    from conftest import _example

    from si_viewer.project import open_project, save_project
    from si_viewer.session import Session

    s = Session()
    entry = s.open_file(_example("20261112_1302_2.25_Mx_HAADF-DF.emd"))
    by_title = {s.datasets[i].title: s.datasets[i].summary()["detector"] for i in entry.dataset_ids}
    assert by_title["HAADF"] is None
    assert by_title["DF-A"]["segments"] == [0, 4] and by_title["DF-B"]["segments"] == [1, 5]
    assert by_title["DF-C"]["segments"] == [2, 6] and by_title["DF-D"]["segments"] == [3, 7]
    assert by_title["DF-A"]["polarities"] == ["Positive", "Positive"] and by_title["DF-A"]["name"] == "DFS"

    with tempfile.TemporaryDirectory() as tmp:
        saved = save_project(s, f"{tmp}/a.hspy", level="results")
        other = Session()
        open_project(other, saved)
        assert other.datasets[entry.dataset_ids[0]].summary()["detector"] == s.datasets[entry.dataset_ids[0]].summary()["detector"]
