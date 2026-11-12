import numpy as np
import pytest
import tifffile
from fastapi.testclient import TestClient

from si_viewer.app import create_app


def _array(r) -> np.ndarray:
    assert r.status_code == 200, r.text
    shape = tuple(int(n) for n in r.headers["X-Array-Shape"].split(","))
    return np.frombuffer(r.content, "<f4").reshape(shape)


@pytest.fixture
def client():
    return TestClient(create_app())


def test_only_spectrum_images_and_stacks_rebin(client, tmp_path):
    tifffile.imwrite(tmp_path / "s.tif", np.zeros((6, 8, 10), "uint16"))
    stack = client.post("/api/files", json={"path": str(tmp_path / "s.tif")}).json()["datasets"][0]
    assert stack["kind"] == "stack"
    entry = client.post("/api/examples/fept-si").json()
    for d in entry["datasets"]:
        if d["kind"] != "spectrum_image":  # HAADF and element maps
            r = client.post(f"/api/datasets/{d['id']}/rebin", json={"factors": [2, 2]})
            assert r.status_code == 400, d["title"]
    # Only element maps can be deleted.
    assert client.delete(f"/api/datasets/{stack['id']}").status_code == 400


def test_rebin_takes_the_whole_scan(client):
    entry = client.post("/api/examples/fept-si").json()
    si = next(d for d in entry["datasets"] if d["kind"] == "spectrum_image")
    haadf = next(d for d in entry["datasets"] if d["kind"] == "image" and d["origin"] == "file")
    raw = _array(client.get(f"/api/datasets/{haadf['id']}/frame"))

    r = client.post(f"/api/datasets/{si['id']}/rebin", json={"factors": [2, 4, 1]}).json()
    datasets = {d["id"]: d for d in r["file"]["datasets"]}
    assert datasets[si["id"]]["navigation_shape"] == [16, 32]  # (x, y)
    # The HAADF on the same scan is binned the same way, so it still lines up.
    h = datasets[haadf["id"]]
    assert h["signal_shape"] == [16, 32] and h["revision"] == haadf["revision"] + 1 and h["rebinned"]
    np.testing.assert_allclose(
        _array(client.get(f"/api/datasets/{haadf['id']}/frame")),
        raw.reshape(32, 2, 16, 4).sum(axis=(1, 3)), rtol=1e-5,
    )
    assert haadf["id"] in [s["id"] for s in client.get(f"/api/datasets/{si['id']}/navigators").json()]
    for d in datasets.values():
        if d["kind"] == "image":
            assert d["signal_shape"] == [16, 32], d["title"]

    # Factors now apply to the rebinned sizes.
    assert client.post(f"/api/datasets/{si['id']}/rebin", json={"factors": [1, 1, 1]}).status_code == 422
    assert client.post(f"/api/datasets/{si['id']}/rebin", json={"factors": [1, 17, 1]}).status_code == 422
    assert client.post(f"/api/datasets/{si['id']}/rebin", json={"factors": [2, 2]}).status_code == 422


def test_rebin_spectrum_image_in_place(client, example_emd):
    entry = client.post("/api/files", json={"path": str(example_emd)}).json()
    si = next(d for d in entry["datasets"] if d["kind"] == "spectrum_image")
    maps = {d["element"]: d for d in entry["datasets"] if d["parent_id"] == si["id"]}
    total = _array(client.get(f"/api/datasets/{si['id']}/spectrum"))

    r = client.post(f"/api/datasets/{si['id']}/rebin", json={"factors": [2, 2, 1]}).json()
    new = r["dataset"]
    assert new["id"] == si["id"] and new["navigation_shape"] == [128, 128]
    assert new["axes"][0]["scale"] == pytest.approx(2 * si["axes"][0]["scale"])
    assert new["eds"]["elements"] == si["eds"]["elements"]
    assert len(r["file"]["datasets"]) == len(entry["datasets"])  # nothing added
    # Counts are conserved.
    np.testing.assert_allclose(_array(client.get(f"/api/datasets/{si['id']}/spectrum")), total, rtol=1e-6)
    # Total-counts map at the new size.
    assert _array(client.get(f"/api/datasets/{si['id']}/frame")).shape == (128, 128)

    # Its maps are recomputed in place at the new size.
    after = {d["element"]: d for d in r["file"]["datasets"] if d["parent_id"] == si["id"]}
    assert after.keys() == maps.keys()
    for el, m in after.items():
        assert m["id"] == maps[el]["id"] and m["revision"] == maps[el]["revision"] + 1
        assert m["signal_shape"] == [128, 128] and m["title"] == maps[el]["title"]


# ---------- stacks ----------


def _stack(client, tmp_path, data):
    tifffile.imwrite(tmp_path / "stack.tif", data)
    d = client.post("/api/files", json={"path": str(tmp_path / "stack.tif")}).json()["datasets"][0]
    assert d["kind"] == "stack"
    return d


def test_a_stack_is_rebinned_along_frames_y_and_x(client, tmp_path):
    rng = np.random.default_rng(1)
    data = rng.integers(0, 1000, (7, 9, 10)).astype("uint16")  # frames, y, x
    stack = _stack(client, tmp_path, data)
    r = client.post(f"/api/datasets/{stack['id']}/rebin", json={"factors": [2, 3, 2]})
    assert r.status_code == 200, r.text
    now = r.json()["dataset"]
    assert now["kind"] == "stack" and now["navigation_shape"] == [3] and now["signal_shape"] == [5, 3]
    expected = data[:6, :9, :10].astype(np.int64).reshape(3, 2, 3, 3, 5, 2).sum(axis=(1, 3, 5))
    session = client.app.state.session
    np.testing.assert_array_equal(session.get_dataset(stack["id"]).signal.data, expected)
    assert now["rebinned"] and now["revision"] > stack["revision"]
    # A frame of the rebinned stack.
    frame = _array(client.get(f"/api/datasets/{stack['id']}/frame?index=1"))
    np.testing.assert_allclose(frame, expected[1])
    # Out of range, nothing to do, or a time factor: refused.
    for body in ({"factors": [4, 1, 1]}, {"factors": [1, 1, 1]}, {"factors": [1, 1]}, {"factors": [1, 2, 2], "time": 2}):
        assert client.post(f"/api/datasets/{stack['id']}/rebin", json=body).status_code == 422, body


def test_rebinning_a_stack_drops_its_alignment(client, tmp_path):
    stack = _stack(client, tmp_path, np.ones((5, 8, 10), "uint16"))
    session = client.app.state.session
    ds = session.get_dataset(stack["id"])
    ds.materialise()
    # As apply_alignment leaves it: the aligned frames shown, the loaded ones kept.
    ds.original_signal = ds.signal
    ds.aligned_signal = ds.signal.deepcopy()
    ds.signal = ds.aligned_signal
    ds.alignment = {"square": [0, 0, 8, 8]}
    session.rebin(ds, [1, 2, 2])
    assert ds.alignment is None and ds.original_signal is None and ds.aligned_signal is None
    assert ds.signal.data.shape == (5, 4, 5) and (ds.signal.data == 4).all()


def test_two_d_images_are_not_rebinned(client, tmp_path):
    tifffile.imwrite(tmp_path / "i.tif", np.zeros((8, 10), "uint16"))
    image = client.post("/api/files", json={"path": str(tmp_path / "i.tif")}).json()["datasets"][0]
    assert image["kind"] == "image"
    assert client.post(f"/api/datasets/{image['id']}/rebin", json={"factors": [2, 2]}).status_code == 400


# ---------- as a new dataset ----------


def test_rebin_as_a_new_dataset(client, tmp_path):
    data = np.random.default_rng(2).integers(0, 1000, (6, 9, 10)).astype("uint16")
    stack = _stack(client, tmp_path, data)
    r = client.post(f"/api/datasets/{stack['id']}/rebin", json={"factors": [1, 3, 2], "in_place": False})
    assert r.status_code == 200, r.text
    new = r.json()["dataset"]
    assert new["origin"] == "rebinned" and new["parent_id"] == stack["id"] and new["title"] == "stack (rebinned)"
    assert new["signal_shape"] == [5, 3]
    session = client.app.state.session
    src = session.get_dataset(stack["id"])
    assert src.signal.data.shape == (6, 9, 10) and src.ops == [] and src.rebinned == []  # left as it is
    copy = session.get_dataset(new["id"])
    assert [{k: v for k, v in o.items() if k != "seq"} for o in copy.ops] == [{"op": "rebin", "scale": [1, 2, 3]}] and copy.copied_at == 0
    np.testing.assert_allclose(copy.signal.data, data.astype(np.int64).reshape(6, 3, 3, 5, 2).sum(axis=(2, 4)))
    # Bad factors make no copy.
    n = len(session.files[stack["file_id"]].dataset_ids)
    assert client.post(f"/api/datasets/{stack['id']}/rebin", json={"factors": [9, 1, 1], "in_place": False}).status_code == 422
    assert len(session.files[stack["file_id"]].dataset_ids) == n
    assert client.delete(f"/api/datasets/{new['id']}").status_code == 200


def test_rebin_in_place_is_recorded(client):
    entry = client.post("/api/examples/fept-si").json()
    si = next(d for d in entry["datasets"] if d["kind"] == "spectrum_image")
    client.post(f"/api/datasets/{si['id']}/rebin", json={"factors": [2, 2, 1]})
    ops = client.app.state.session.get_dataset(si["id"]).ops
    assert [{k: v for k, v in o.items() if k != "seq"} for o in ops] == [{"op": "rebin", "scale": [2, 2, 1]}]


def test_rebin_a_spectrum_image_as_a_new_dataset_leaves_the_scan_alone(client):
    entry = client.post("/api/examples/fept-si").json()
    si = next(d for d in entry["datasets"] if d["kind"] == "spectrum_image")
    haadf = next(d for d in entry["datasets"] if d["kind"] == "image" and d["origin"] == "file")
    before = _array(client.get(f"/api/datasets/{haadf['id']}/frame"))
    r = client.post(f"/api/datasets/{si['id']}/rebin", json={"factors": [2, 2, 1], "in_place": False}).json()
    assert r["dataset"]["origin"] == "rebinned"
    assert r["dataset"]["navigation_shape"] == [si["navigation_shape"][0] // 2, si["navigation_shape"][1] // 2]
    files = {d["id"]: d for d in r["file"]["datasets"]}
    assert files[si["id"]]["navigation_shape"] == si["navigation_shape"]
    np.testing.assert_array_equal(_array(client.get(f"/api/datasets/{haadf['id']}/frame")), before)


def test_rebin_as_a_new_dataset_copies_the_linked_ones(client, tmp_path):
    import hyperspy.api as hs

    def eels(name, offset):
        s = hs.signals.Signal1D(np.random.default_rng(0).uniform(1, 2, (6, 8, 50)).astype(np.float32))
        s.set_signal_type("EELS")
        s.axes_manager[-1].offset = offset
        s.metadata.General.title = name
        s.save(tmp_path / f"{name}.hspy")
        r = client.post("/api/files", json={"path": str(tmp_path / f"{name}.hspy")})
        return next(d for d in r.json()["datasets"] if d["kind"] == "spectrum_image")

    ll, hl = eels("ll", -10), eels("hl", 400)
    assert client.put(f"/api/datasets/{ll['id']}/eels-role", json={"role": "low_loss"}).status_code == 200
    assert client.put(f"/api/datasets/{hl['id']}/eels-partner", json={"partner_id": ll["id"]}).status_code == 200

    r = client.post(f"/api/datasets/{hl['id']}/rebin", json={"factors": [2, 2, 1], "in_place": False})
    assert r.status_code == 200, r.text
    new, partners = r.json()["dataset"], r.json()["partners"]
    assert new["parent_id"] == hl["id"] and new["navigation_shape"] == [4, 3]
    assert len(partners) == 1 and partners[0]["parent_id"] == ll["id"] and partners[0]["navigation_shape"] == [4, 3]
    # The copies are linked to each other; the sources are left as they were.
    assert new["eels"]["linked"] == [partners[0]["id"]]
    session = client.app.state.session
    assert session.get_dataset(ll["id"]).signal.data.shape == (6, 8, 50)
    assert [d.id for d in session.linked_of(session.get_dataset(hl["id"]))] == [ll["id"]]
