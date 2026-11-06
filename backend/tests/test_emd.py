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


def test_session_open_and_materialise(example_emd):
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
    assert si.lazy
    si.materialise()
    assert not si.lazy
    assert isinstance(si.signal.data, np.ndarray)
    assert si.summary()["navigation_shape"] == list(si.signal.axes_manager.navigation_shape)

    session.close_file(entry.id)
    assert not session.files and not session.datasets
