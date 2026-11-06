import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _find_example_emd() -> Path | None:
    env = os.environ.get("SI_VIEWER_TEST_EMD")
    if env:
        return Path(env)
    found = sorted(REPO_ROOT.glob("*.emd")) + sorted((REPO_ROOT / "data").glob("*.emd"))
    return found[0] if found else None


@pytest.fixture(scope="session")
def example_emd() -> Path:
    path = _find_example_emd()
    if path is None or not path.is_file():
        pytest.skip("No example .emd (set SI_VIEWER_TEST_EMD or put one in the repo root)")
    return path
