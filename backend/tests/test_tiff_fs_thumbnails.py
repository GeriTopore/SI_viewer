import io

import numpy as np
import pytest
import tifffile
from fastapi.testclient import TestClient
from PIL import Image

from si_viewer.app import create_app
from si_viewer.session import DatasetKind, Session
from si_viewer.thumbnails import si_companion_image


@pytest.fixture
def tiff_files(tmp_path):
    rng = np.random.default_rng(0)
    stack = (rng.random((5, 160, 200)) * 1000).astype("uint16")
    tifffile.imwrite(tmp_path / "stack.tif", stack)
    tifffile.imwrite(tmp_path / "single.tiff", stack[0])
    (tmp_path / "notes.txt").write_text("not data")
    (tmp_path / "sub").mkdir()
    return tmp_path


@pytest.fixture
def client():
    return TestClient(create_app())


def test_tiff_stack_and_image(tiff_files):
    session = Session()
    stack = session.datasets[session.open_file(tiff_files / "stack.tif").dataset_ids[0]]
    assert stack.kind is DatasetKind.STACK
    assert stack.title == "stack"
    assert stack.summary()["navigation_shape"] == [5]
    assert all(ax["units"] == "" for ax in stack.summary()["axes"])

    single = session.datasets[session.open_file(tiff_files / "single.tiff").dataset_ids[0]]
    assert single.kind is DatasetKind.IMAGE


def test_fs_list(client, tiff_files):
    body = client.get("/api/fs/list", params={"path": str(tiff_files)}).json()
    names = [e["name"] for e in body["entries"]]
    assert names == ["sub", "single.tiff", "stack.tif"]  # dirs first, unsupported hidden
    assert body["parent"] == str(tiff_files.parent)
    assert client.get("/api/fs/list", params={"path": str(tiff_files / "nope")}).status_code == 404


def _png_size(content: bytes) -> tuple[int, int]:
    return Image.open(io.BytesIO(content)).size


def test_thumbnails_tiff(client, tiff_files):
    entry = client.post("/api/files", json={"path": str(tiff_files / "stack.tif")}).json()
    r = client.get(f"/api/datasets/{entry['datasets'][0]['id']}/thumbnail", params={"size": 128})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert _png_size(r.content) == (128, 102)  # aspect ratio 200:160 kept


def test_thumbnails_emd(client, example_emd):
    entry = client.post("/api/files", json={"path": str(example_emd)}).json()
    for d in entry["datasets"]:
        r = client.get(f"/api/datasets/{d['id']}/thumbnail", params={"size": 96})
        assert r.status_code == 200, d["title"]
        w, h = _png_size(r.content)
        assert max(w, h) <= 96


def test_si_companion_is_adf(example_emd):
    session = Session()
    entry = session.open_file(example_emd)
    datasets = [session.datasets[d] for d in entry.dataset_ids]
    si = next(d for d in datasets if d.kind is DatasetKind.SPECTRUM_IMAGE)
    companion = si_companion_image(si, datasets)
    assert companion is not None and "DF" in companion.title.upper()


def test_fs_locate_dropped_items(client, tmp_path, monkeypatch):
    """Dropped files and folders are found on disk by name, size and time."""
    from si_viewer.api import fs

    monkeypatch.setattr(fs, "_spotlight", lambda name: [])
    monkeypatch.setattr(fs, "LOCATE_BUDGET", 0.0)  # hints only: no walk of the home folder
    folder = tmp_path / "run 1"
    folder.mkdir()
    (folder / "a.dm4").write_bytes(b"x" * 10)
    (folder / "b.dm4").write_bytes(b"y" * 20)
    stat = (folder / "a.dm4").stat()
    file_item = {"name": "a.dm4", "size": 10, "modified": stat.st_mtime * 1000}
    folder_item = {"name": "run 1", "is_dir": True, "children": [{"name": "a.dm4", "size": 10}, {"name": "b.dm4", "size": 20}]}

    def locate(items, hints):
        r = client.post("/api/fs/locate", json={"items": items, "hints": hints})
        assert r.status_code == 200, r.text
        return r.json()["paths"]

    # The file in a hinted folder; the folder beside a hinted one (its parent is tried too).
    assert locate([file_item, folder_item], [str(folder)]) == [str((folder / "a.dm4").resolve()), str(folder.resolve())]
    # Another size, another time or other contents: not that one.
    assert locate([{**file_item, "size": 11}], [str(folder)]) == [None]
    assert locate([{**file_item, "modified": file_item["modified"] + 60_000}], [str(folder)]) == [None]
    wrong = {**folder_item, "children": [{"name": "a.dm4", "size": 99}]}
    assert locate([wrong], [str(folder)]) == [None]

    # Without a hint it is found by walking (here a stand-in for the home folder).
    monkeypatch.setattr(fs, "LOCATE_BUDGET", 5.0)
    monkeypatch.setattr(fs.Path, "home", classmethod(lambda cls: tmp_path))
    assert locate([file_item], []) == [str((folder / "a.dm4").resolve())]
