"""Orientation and phase mapping as pyxem does it (its SPED-Ag tutorial):
Structures' templates, their views, the polar transform."""

import numpy as np
from fastapi.testclient import TestClient

from si_viewer.app import create_app
from si_viewer.processing import orientation
from si_viewer.processing.structure import phase_from_cif
from test_com import _wait
from test_structure import AU, PEROVSKITE


def test_zone_template_obeys_the_zone_law():
    """Every reflection of the [uvw] template is in the zero-order Laue zone."""
    sto = phase_from_cif(PEROVSKITE, "SrTiO3")
    o = orientation.TemplateOptions(reciprocal_radius=1.5, max_excitation_error=0.02, precession_angle=0)
    for uvw in ([0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 2, 3]):
        t = orientation.zone_template(sto, uvw, o)
        assert t["hkl"] and np.all(np.asarray(t["hkl"]) @ np.asarray(uvw) == 0)
        assert max(t["intensity"]) == 1.0
    # [001] of a cube: |g| = |hkl| / a, all in the plane of the pattern.
    t = orientation.zone_template(sto, [0, 0, 1], o)
    np.testing.assert_allclose(np.hypot(t["x"], t["y"]), np.linalg.norm(t["hkl"], axis=1) / 3.905, rtol=1e-4)


def _setup(tmp_path, q=(1.0, "pixels")):
    from test_4dstem import _write_emdfile

    (tmp_path / "Au.cif").write_text(AU)
    (tmp_path / "SrTiO3.cif").write_text(PEROVSKITE)
    cube = np.random.default_rng(0).random((4, 4, 32, 32)).astype(np.float32)
    _write_emdfile(tmp_path / "c.h5", cube, q=q, extra_2d=False)
    client = TestClient(create_app())
    ids = [client.post("/api/files", json={"path": str(tmp_path / n)}).json()["datasets"][0]["id"]
           for n in ("c.h5", "Au.cif", "SrTiO3.cif")]
    return client, *ids


AG = AU.replace("Au", "Ag").replace("4.0782", "4.08")


def test_zone_axes_grid():
    """The tutorial's reduced grid: 0.5° near [101] and [114] (within 5°)."""
    ag = phase_from_cif(AG, "Ag")
    full = orientation.template_rotations(ag, orientation.TemplateOptions(resolution=0.5))
    near = orientation.template_rotations(ag, orientation.TemplateOptions(
        resolution=0.5, zone_axes=[{"uvw": [1, 0, 1], "angle": 5}, {"uvw": [1, 1, 4], "angle": 5}]))
    assert 0 < near.size < full.size / 10
    from orix.vector import Miller, Vector3d

    beam = (near * Vector3d.zvector()).in_fundamental_sector(ag.point_group)
    to101 = np.degrees(np.asarray(beam.angle_with(Miller(uvw=[1, 0, 1], phase=ag).in_fundamental_sector(ag.point_group))))
    to114 = np.degrees(np.asarray(beam.angle_with(Miller(uvw=[1, 1, 4], phase=ag).in_fundamental_sector(ag.point_group))))
    assert np.all(np.minimum(to101, to114) <= 5 + 1e-6)
    import pytest

    with pytest.raises(ValueError):
        orientation.template_rotations(ag, orientation.TemplateOptions(zone_axes=[{"uvw": [0, 0, 0]}]))


def test_polar_transform_of_a_ring(tmp_path):
    """The Polar transform process (pyxem's): a ring at g = 0.4 1/Å around
    the axes' 0 is a line at 0.4 1/Å at every angle; an elliptical one too,
    once the calibration's ellipse is set. As a copy under the data, or in
    place."""
    from test_4dstem import _write_emdfile

    K, scale = 96, 0.01  # 1/Å per px
    yy, xx = np.mgrid[0:K, 0:K] + 0.5
    c = K / 2

    def ring(a, b):
        r = np.hypot((xx - c) / a, (yy - c) / b)
        return np.exp(-((r - 1) ** 2) / (2 * 0.04**2)).astype(np.float32)

    cube = np.zeros((2, 2, K, K), np.float32)
    cube[0, 0] = ring(40, 40)
    cube[0, 1] = ring(44, 36.4)  # the same ring stretched along x
    _write_emdfile(tmp_path / "r.h5", cube, q=(scale, "A^-1"), extra_2d=False)
    client = TestClient(create_app())
    session = client.app.state.session
    cube_id = client.post("/api/files", json={"path": str(tmp_path / "r.h5")}).json()["datasets"][0]["id"]
    ds = session.get_dataset(cube_id)
    for ax in ds.signal.axes_manager.signal_axes:
        ax.offset = -c * scale
    url = f"/api/datasets/{cube_id}/4d/polar"

    def transform(**opts):
        job = _wait(client, client.post(url, json={"options": {"npt": 100, "r_max": 0.6, **opts}}).json())
        assert job["status"] == "done", job
        made = session.get_dataset(job["result"]["id"])
        assert made.origin == "polar" and made.parent_id == cube_id and made.kind.value == "4dstem"
        az, rad = made.signal.axes_manager.signal_axes
        assert made.signal.data.shape == (2, 2, 100, 360) and az.units == "Rad" and rad.units == "1/Å"
        g = rad.offset + rad.scale * (np.arange(100) + 0.5)
        return made, [g[np.asarray(made.signal.data[0, x]).argmax(axis=0)] for x in (0, 1)]

    made, (circle, stretched) = transform()
    assert np.abs(circle - 0.4).max() < 0.012  # two bins (0.006 1/Å)
    assert stretched.max() - stretched.min() > 0.06  # elliptical: 0.364 to 0.44
    assert made.signal.metadata.Signal.signal_type == "polar_diffraction"
    # It records what it was made from (the source's pattern axes then), for
    # Orientation mapping to tell when the source's calibration has changed.
    rec = client.get(f"/api/datasets/{made.id}").json()["polar"]
    assert rec["source"] == cube_id and rec["options"]["npt"] == 100
    assert [round(a["scale"], 6) for a in rec["axes"]] == [scale, scale] and rec["axes"][0]["units"] == "1/Å"
    # The calibration's ellipse makes it circular, at √(a b) (pyxem's affine, areas kept).
    from si_viewer.processing.calibration import affine_of

    ell = {"cx": c, "cy": c, "a": 44.0, "b": 36.4, "angle": 0.0}
    ds.calibration = {"ring": ell, "affine": affine_of(ell)}
    _, (_, fixed) = transform()
    assert np.abs(fixed - 0.4).max() < 0.019
    _, (_, unfixed) = transform(ellipse=False)
    assert unfixed.max() - unfixed.min() > 0.06
    # An empty range; a polar transform not transformed again; in place.
    bad = _wait(client, client.post(url, json={"options": {"r_min": 0.7, "r_max": 0.6}}).json())
    assert bad["status"] == "failed"
    assert client.post(f"/api/datasets/{made.id}/4d/polar", json={}).status_code == 409
    job = _wait(client, client.post(url, json={"in_place": True}).json())
    assert job["result"]["id"] == cube_id and ds.signal.data.shape == (2, 2, 112, 360)
    assert ds.polar["source"] is None


def test_templates_of_a_structure(tmp_path):
    """A Structure's Templates process (the tutorial's settings): a new
    Templates object under it each time; its inverse pole figure (orix), a
    template's pattern, the nearest to a zone axis; saved with the analysis."""
    (tmp_path / "Ag.cif").write_text(AG)
    client = TestClient(create_app())
    ag = client.post("/api/files", json={"path": str(tmp_path / "Ag.cif")}).json()["datasets"][0]["id"]
    url = f"/api/datasets/{ag}/templates/make"
    ids = []
    for opts in ({}, {"resolution": 0.5, "max_excitation_error": 0.01,
                      "zone_axes": [{"uvw": [1, 0, 1], "angle": 5}, {"uvw": [1, 1, 4], "angle": 5}]}):
        job = _wait(client, client.post(url, json={"options": opts}).json())
        assert job["status"] == "done", job
        ids.append(job["result"]["id"])
    session = client.app.state.session
    full, near = (session.get_dataset(i) for i in ids)
    assert full.parent_id == ag == near.parent_id and full.origin == "templates" and full.kind.value == "other"
    assert full.title == "Templates · 2°" and near.title == "Templates · 0.5°, [1 0 1] [1 1 4]"
    o = full.templates["options"]
    assert (o["resolution"], o["reciprocal_radius"], o["max_excitation_error"], o["precession_angle"], o["minimum_intensity"], o["kv"]) \
        == (2.0, 1.5, 0.001, 1.0, 1e-5, 200.0)
    assert full.templates["templates"] > near.templates["templates"] > 0
    assert client.post(url, json={"options": {"zone_axes": [{"uvw": [0, 0, 0]}]}}).status_code == 422
    assert client.post(f"/api/datasets/{full.id}/templates/make", json={}).status_code == 400

    ipf = client.get(f"/api/datasets/{full.id}/templates").json()
    n = full.templates["templates"]
    assert len(ipf["points"]["x"]) == n and len(ipf["points"]["rgb"]) == n
    assert sorted(c["label"] for c in ipf["sector"]["corners"]) == ["0 0 1", "1 0 1", "1 1 1"]
    i = client.get(f"/api/datasets/{full.id}/templates/nearest", params={"u": 1, "v": 0, "w": 1}).json()["index"]
    one = client.get(f"/api/datasets/{full.id}/templates/{i}").json()
    assert np.allclose(one["uvw"], [1, 0, 1], atol=0.05) and one["hkl"] and one["reciprocal_radius"] == 1.5
    assert max(np.hypot(one["x"], one["y"])) <= 1.5 + 1e-6
    assert client.get(f"/api/datasets/{full.id}/templates/{n}").status_code == 404
    # Made again from the record when not kept (an analysis reopened).
    session.template_sims.clear()
    assert client.get(f"/api/datasets/{near.id}/templates/0").status_code == 200
    saved = tmp_path / "a.hspy"
    assert client.post("/api/project/save", json={"path": str(saved), "views": {}}).status_code == 200
    other = TestClient(create_app())
    assert other.post("/api/project/open", json={"path": str(saved)}).status_code == 200
    reopened = next(d for f in other.get("/api/files").json() for d in f["datasets"] if d["origin"] == "templates")
    assert other.get(f"/api/datasets/{reopened['id']}/templates/0").status_code == 200

    plots = client.get(f"/api/datasets/{ag}/structure/plots").json()
    assert plots["symmetry"]["name"] == "m-3m"
    assert len(plots["symmetry"]["upper"]["x"]) + len(plots["symmetry"]["lower"]["x"]) == 48
    atoms = client.get(f"/api/datasets/{ag}/structure/atoms").json()
    assert set(atoms["elements"]) == {"Ag"} and len(atoms["atoms"]["symbol"]) == 14  # 4 + boundary copies
    assert client.delete(f"/api/datasets/{near.id}").status_code in (200, 204)


def test_template_toward_a_view():
    """The template nearest a beam direction, its reflections taken onto that
    direction's own zone (any of the symmetry-equivalent ones)."""
    ag = phase_from_cif(AG, "Ag")
    sim = orientation.simulate(ag, orientation.TemplateOptions())
    for uvw in ([0, 1, 1], [1, 0, 1], [-1, 1, 0], [0, 0, 1], [0, -1, 0], [1, 1, 4], [2, -1, 7]):
        r = orientation.template_toward(sim, ag, uvw)
        hkl = np.asarray(r["hkl"])
        assert len(hkl) == len(r["intensity"]) > 0 and r["angle"] < 2  # within the 2° sampling
        assert np.all(hkl @ np.asarray(uvw) == 0)  # in [uvw]'s zero-order Laue zone
    # Equivalent directions: the same template.
    same = {orientation.template_toward(sim, ag, uvw)["index"] for uvw in ([0, 1, 1], [1, 0, 1], [-1, 1, 0])}
    assert len(same) == 1


def test_polar_as_pyxem_makes_it():
    """One pattern's polar transform with pyxem's slices kept: as pyxem's
    ``get_azimuthal_integral2d`` (the tutorial's npt=112), again from the
    slices kept."""
    rng = np.random.default_rng(1)
    img = rng.random((112, 112)).astype(np.float32)
    axes = [(0.0267, -1.4968, "1/Å")] * 2
    o = orientation.PolarOptions(npt=112, gamma=1)
    for affine in (None, [[1.02, 0.01, 0], [0.01, 0.98, 0], [0, 0, 1]]):
        # Each signal fresh: pyxem changes the calibration's affine in place
        # whenever it makes the slices, so only the first use is as set.
        ref = orientation.diffraction2d(img, axes, affine).get_azimuthal_integral2d(npt=112, show_progressbar=False)
        for _ in range(2):  # made, then from the slices kept
            p = orientation.polar_of(orientation.diffraction2d(img, axes, affine), o, 1.0)
            np.testing.assert_allclose(p["data"], np.asarray(ref.data), rtol=1e-5, atol=1e-6)
        r = ref.axes_manager.signal_axes[1]
        assert abs(p["radius"]["scale"] - r.scale) < 1e-12 and p["radius"]["offset"] == r.offset


def test_match_at_positions(tmp_path):
    """pyxem's template matching at scan positions of the polar transform:
    patterns drawn from Ag's [101] template, turned in-plane, match a
    template near [101]; every template's correlation is kept."""
    from test_4dstem import _write_emdfile

    ag = phase_from_cif(AG, "Ag")
    K, scale = 96, 0.025  # 1/Å per px
    t = orientation.zone_template(ag, [1, 0, 1], orientation.TemplateOptions(precession_angle=1, max_excitation_error=0.02))
    yy, xx = np.mgrid[0:K, 0:K] + 0.5

    def pattern(turn):
        c, s = np.cos(np.radians(turn)), np.sin(np.radians(turn))
        img = np.zeros((K, K), np.float32)
        for x, y, i in zip(t["x"], t["y"], t["intensity"]):
            px, py = K / 2 + (c * x - s * y) / scale, K / 2 + (s * x + c * y) / scale
            img += i * np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * 1.0**2))
        return img

    cube = np.stack([np.stack([pattern(0), pattern(30)])] * 2).astype(np.float32)  # 2 × 2 scan
    _write_emdfile(tmp_path / "ag.h5", cube, q=(scale, "A^-1"), extra_2d=False)
    (tmp_path / "Ag.cif").write_text(AG)
    client = TestClient(create_app())
    session = client.app.state.session
    cube_id, ag_id = (client.post("/api/files", json={"path": str(tmp_path / n)}).json()["datasets"][0]["id"]
                      for n in ("ag.h5", "Ag.cif"))
    for ax in session.get_dataset(cube_id).signal.axes_manager.signal_axes:
        ax.offset = -K / 2 * scale
    made = _wait(client, client.post(f"/api/datasets/{ag_id}/templates/make", json={}).json())
    templates = made["result"]["id"]
    url = f"/api/datasets/{cube_id}/4d/orientation/match"
    body = {"templates": templates, "positions": [[0, 0], [1, 0]]}
    assert client.post(url, json=body).status_code == 409  # no polar transform yet
    polar = _wait(client, client.post(f"/api/datasets/{cube_id}/4d/polar", json={}).json())["result"]["id"]
    r = client.post(url, json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["polar"] == polar and len(out["matches"]) == 2
    n = session.get_dataset(templates).templates["templates"]
    sim = session.template_sims[templates][1]
    for m in out["matches"]:
        b = m["best"]
        uvw = orientation.direction_of(sim.rotations[b["index"]], ag)
        assert np.allclose(sorted(np.abs(uvw)), [0, 1, 1], atol=0.06)  # near <101>
        corr = [c for c in m["correlations"] if c is not None]
        assert len(m["correlations"]) == n and len(corr) == n and abs(max(corr) - b["correlation"]) < 1e-5
        assert len(m["vectors"]["x"]) == len(m["vectors"]["hkl"]) == len(m["polar"]["r"]) > 0
        assert len(b["euler"]) == 3
    # The way the crystal is seen at each (view towards the viewer, up the
    # screen, as [uvw]): Ag's reflections across the view project (right,
    # against the rows) onto the spots each pattern was drawn with, turned
    # 0° and 30° (the matched orientation may be any equivalent one).
    import itertools

    A = np.asarray(ag.structure.lattice.base)  # rows a, b, c (Å)
    fcc = np.array([h for h in itertools.product(range(-6, 7), repeat=3) if len({v % 2 for v in h}) == 1 and any(h)], float)
    G = fcc @ np.linalg.inv(A).T
    for m, turn in zip(out["matches"], (0, 30)):
        view, up = (np.asarray(m["best"][k]) @ A for k in ("view", "up"))
        view, up = view / np.linalg.norm(view), up / np.linalg.norm(up)
        assert abs(view @ up) < 1e-3
        right, on = np.cross(up, view), np.abs(G @ view) < 0.01
        x, y = G[on] @ right, -(G[on] @ up)  # y down, as the rows
        c, s = np.cos(np.radians(turn)), np.sin(np.radians(turn))
        for tx, ty in zip(t["x"], t["y"]):
            assert np.min(np.hypot(x - (c * tx - s * ty), y - (s * tx + c * ty))) < 0.04
        # The reflections' labels: the crystal's, on the zone seen, allowed (fcc).
        hkl = np.asarray(m["vectors"]["hkl"])
        assert np.abs(hkl @ np.asarray(m["best"]["view"])).max() < 0.15
        assert all(len({v % 2 for v in h}) == 1 for h in hkl)
    # The same polar transform, asked for directly; outside the scan.
    assert client.post(f"/api/datasets/{polar}/4d/orientation/match", json=body).status_code == 200
    assert client.post(url, json={**body, "positions": [[5, 0]]}).status_code == 422


def test_templates_without_precession_say_what_to_change():
    """diffsims fails where an orientation has no reflection (a narrow
    excitation error without precession): the error says what to raise."""
    au = phase_from_cif(AU, "Au")
    narrow = orientation.TemplateOptions(precession_angle=0, resolution=5, max_excitation_error=0.001)
    try:
        orientation.simulate(au, narrow)
    except ValueError as e:
        assert "raise the excitation error" in str(e) and "add precession" in str(e)
    else:
        raise AssertionError("expected the narrow bank to fail")
    wide = orientation.TemplateOptions(precession_angle=0, resolution=5, max_excitation_error=0.01)
    assert orientation.simulate(au, wide).rotations.size > 0


def test_the_matched_view_shows_the_template_on_the_match(tmp_path):
    """Match orientation: the Structure view turned to a match's view and up
    picks a template (``templates/toward``) that, projected as the view
    projects it (x right, y up), lands on the match's spots. Off the zone
    axis (the excited reflections not in ±g pairs), so a view turned half
    way round about the beam can't pass."""
    from test_4dstem import _write_emdfile

    ag = phase_from_cif(AG, "Ag")
    K, scale = 96, 0.025
    t = orientation.zone_template(ag, [1, 0, 0.9], orientation.TemplateOptions(precession_angle=1, max_excitation_error=0.02))
    yy, xx = np.mgrid[0:K, 0:K] + 0.5

    def pattern(turn):
        c, s = np.cos(np.radians(turn)), np.sin(np.radians(turn))
        img = np.zeros((K, K), np.float32)
        for x, y, i in zip(t["x"], t["y"], t["intensity"]):
            px, py = K / 2 + (c * x - s * y) / scale, K / 2 + (s * x + c * y) / scale
            img += i * np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / 2)
        return img

    cube = np.stack([np.stack([pattern(0), pattern(30)])] * 2).astype(np.float32)
    _write_emdfile(tmp_path / "ag.h5", cube, q=(scale, "A^-1"), extra_2d=False)
    (tmp_path / "Ag.cif").write_text(AG)
    client = TestClient(create_app())
    session = client.app.state.session
    cube_id, ag_id = (client.post("/api/files", json={"path": str(tmp_path / n)}).json()["datasets"][0]["id"]
                      for n in ("ag.h5", "Ag.cif"))
    for ax in session.get_dataset(cube_id).signal.axes_manager.signal_axes:
        ax.offset = -K / 2 * scale
    # Templates keeping more reflections (s 0.02 1/Å), so they aren't in ±g pairs off the axis.
    made = client.post(f"/api/datasets/{ag_id}/templates/make", json={"options": {"max_excitation_error": 0.02}}).json()
    templates = _wait(client, made)["result"]["id"]
    _wait(client, client.post(f"/api/datasets/{cube_id}/4d/polar", json={}).json())
    out = client.post(f"/api/datasets/{cube_id}/4d/orientation/match",
                      json={"templates": templates, "positions": [[0, 0], [1, 0]]}).json()
    A = np.asarray(ag.structure.lattice.base)
    for m in out["matches"]:
        view, up = (np.asarray(m["best"][k]) @ A for k in ("view", "up"))
        view, up = view / np.linalg.norm(view), up / np.linalg.norm(up)
        right = np.cross(up, view)
        uvw = view / np.abs(view).max()
        toward = client.get(f"/api/datasets/{templates}/templates/toward", params=dict(zip("uvw", uvw))).json()
        G = np.asarray(toward["hkl"], float) @ np.linalg.inv(A).T
        drawn = np.column_stack([G @ right, -(G @ up)])  # y down, as the rows
        spots = np.column_stack([m["vectors"]["x"], m["vectors"]["y"]])
        near = [np.min(np.hypot(*(spots - p).T)) for p in drawn]
        assert np.mean(np.asarray(near) < 0.03) > 0.9, sorted(near)[-5:]
        # Turned half way round, it doesn't.
        assert np.mean([np.min(np.hypot(*(spots + p).T)) < 0.03 for p in drawn]) < 0.8
