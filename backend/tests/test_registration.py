import numpy as np
import pytest
import tifffile
from fastapi.testclient import TestClient

from si_viewer.app import create_app


def _array(r) -> np.ndarray:
    assert r.status_code == 200, r.text
    shape = tuple(int(n) for n in r.headers["X-Array-Shape"].split(","))
    return np.frombuffer(r.content, "<f4").reshape(shape)


def _drifting_stack(shifts, size=96, seed=0):
    """Frames of one noisy lattice-and-blobs image, each moved by an integer
    (dy, dx) shift (periodic)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:size, :size]
    base = 1 + 0.5 * np.cos(2 * np.pi * x / 8) * np.cos(2 * np.pi * y / 8)
    for cy, cx in rng.uniform(10, size - 10, (6, 2)):
        base += 3 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 30)
    frames = [np.roll(base, (dy, dx), axis=(0, 1)) for dy, dx in shifts]
    return (np.array(frames) * 1000 + rng.normal(0, 30, (len(shifts), size, size))).astype("float32")


SHIFTS = [(0, 0), (1, 2), (3, 3), (4, 5), (6, 6), (7, 8)]


@pytest.fixture
def stack(tmp_path):
    client = TestClient(create_app())
    tifffile.imwrite(tmp_path / "drift.tif", _drifting_stack(SHIFTS))
    ds = client.post("/api/files", json={"path": str(tmp_path / "drift.tif")}).json()["datasets"][0]
    assert ds["kind"] == "stack"
    return client, ds


def test_session_and_previews(stack):
    client, ds = stack
    url = f"/api/datasets/{ds['id']}/registration"
    assert client.get(url).status_code == 404  # none yet

    state = client.post(url, json={}).json()
    assert (state["first"], state["last"], state["shape"]) == (0, 6, [96, 96])

    assert _array(client.get(f"{url}/frame", params={"i": 2})).shape == (96, 96)

    # The cross-correlation of frames 0 and 3 peaks at their shift.
    cc = _array(client.get(f"{url}/correlation", params={"i": 0, "j": 3, "type": "hann", "n": 4}))
    assert cc.shape == (96, 96)
    peak = np.unravel_index(np.argmax(cc), cc.shape)
    assert (peak[0] - 48, peak[1] - 48) == SHIFTS[3]

    # Comparing cutoffs; the session's own mask is unchanged by previews.
    ccs = _array(client.get(f"{url}/compare-n", params={"i": 0, "j": 1, "type": "hann", "n": [3, 6, 9]}))
    assert ccs.shape == (3, 96, 96)
    assert client.get(url).json()["mask"] == {"type": "hann", "n": 6.0}

    state = client.put(url, json={"mask": {"type": "gaussian", "n": 3}, "bad_frames": [4, 99]}).json()
    assert state["mask"]["type"] == "gaussian" and state["bad_frames"] == [4]

    # A frame range keeps the settings; frames outside it are refused.
    state = client.post(url, json={"first": 1, "last": 5}).json()
    assert state["bad_frames"] == [4] and state["mask"]["n"] == 3
    assert client.get(f"{url}/frame", params={"i": 0}).status_code == 422
    assert client.post(url, json={"first": 3, "last": 4}).status_code == 422

    assert client.delete(url).status_code == 204
    assert client.get(url).status_code == 404


def test_only_stacks(stack):
    client, _ = stack
    entry = client.post("/api/examples/fept-si").json()
    haadf = next(d for d in entry["datasets"] if d["kind"] == "image")
    assert client.post(f"/api/datasets/{haadf['id']}/registration", json={}).status_code == 400
