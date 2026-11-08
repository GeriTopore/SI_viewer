import os
from pathlib import Path

import h5py
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "examples"
# Where example data is looked for, in order.
DATA_DIRS = (EXAMPLES, REPO_ROOT, REPO_ROOT / "data")


def _has_eds_spectrum_image(path: Path) -> bool:
    """A Velox file with a spectrum image (not just images or a lone spectrum)."""
    try:
        with h5py.File(path, "r") as f:
            return len(f.get("Data/SpectrumImage", {})) > 0
    except OSError:
        return False


def _find_example_emd() -> Path | None:
    env = os.environ.get("SI_VIEWER_TEST_EMD")
    if env:
        return Path(env)
    for folder in DATA_DIRS:
        for path in sorted(folder.glob("*.emd")):
            if _has_eds_spectrum_image(path):
                return path
    return None


def _example(name: str) -> Path:
    path = EXAMPLES / name
    if not path.is_file():
        pytest.skip(f"No {name} in examples/")
    return path


@pytest.fixture(scope="session")
def example_emd() -> Path:
    """A Velox .emd with an EDS spectrum image."""
    path = _find_example_emd()
    if path is None or not path.is_file():
        pytest.skip("No example .emd with a spectrum image (set SI_VIEWER_TEST_EMD or put one in examples/)")
    return path


@pytest.fixture(scope="session")
def haadf_emd() -> Path:
    """A Velox .emd holding a HAADF image stack."""
    return _example("0016-20261105_1651_13.0_Mx_HAADF.emd")


@pytest.fixture(scope="session")
def eels_dm4() -> Path:
    """EELS spectrum image, 120 × 120 × 3456 (190 MB)."""
    return _example("STEM SI_EELS Spectrum Image_summed.dm4")


@pytest.fixture(scope="session")
def eels_cuzn() -> Path:
    """EELS spectrum image, 40 × 50 × 162."""
    return _example("CuZn_EELS_mapping_tutorial.hspy")


@pytest.fixture(scope="session")
def eels_linescan() -> Path:
    """EELS line scan (core loss), 40 points × 586 channels."""
    return _example("LSMO_linescan.hspy")


@pytest.fixture(scope="session")
def eels_linescan_low_loss() -> Path:
    """EELS line scan (low loss), 40 points × 1024 channels."""
    return _example("LSMO_linescan_low_loss.hspy")
