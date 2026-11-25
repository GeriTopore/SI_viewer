"""Cross-sections of a stack through depth."""

import base64

import numpy as np
import pytest
from fastapi.testclient import TestClient

import si_viewer_abtem.plugin as abtem
from si_viewer.app import create_app
from si_viewer.session import Session
from si_viewer_abtem.potentials import to_stack
from si_viewer_abtem.sections import SectionLine, section


def _stack():
    # Plane p, row r, column c holds 100 p + 10 r + c.
    p, r, c = np.meshgrid(np.arange(3), np.arange(4), np.arange(5), indexing="ij")
    return (100 * p + 10 * r + c).astype(np.float32)


def test_a_row_is_the_xz_section():
    values, n = section(_stack(), SectionLine(x0=0, y0=2.5, x1=5, y1=2.5))
    assert n == 5
    np.testing.assert_allclose(values, _stack()[:, 2, :])


def test_a_column_is_the_yz_section():
    values, n = section(_stack(), SectionLine(x0=1.5, y0=0, x1=1.5, y1=4))
    assert n == 4
    np.testing.assert_allclose(values, _stack()[:, :, 1])


def test_width_averages_across_the_line():
    values, _ = section(_stack(), SectionLine(x0=0, y0=2, x1=5, y1=2, width=2))
    np.testing.assert_allclose(values, _stack()[:, 1:3, :].mean(axis=1))


def test_too_short():
    with pytest.raises(ValueError):
        section(_stack(), SectionLine(x0=1, y0=1, x1=1.2, y1=1))


def test_route_returns_the_section_in_angstrom(tmp_path):
    client = TestClient(create_app(Session(), plugins=[abtem]))
    session = client.app.state.session
    # A potential-like stack under a plugin item, as the plugin makes them.
    entry = session.add_plugin_project(tmp_path, "abtem", "simulation")
    item = session.add_plugin_item(entry, "abtem", "model", "Model", "m", {})
    array = np.transpose(np.flip(_stack(), axis=1), (0, 2, 1))  # back to abTEM's [slice, x, y]
    ds = session.add_plugin_dataset(item, to_stack(array, (0.5, 0.25), 2.0, "t"), "abtem", "t", {}, label="Potential")
    r = client.post(f"/api/plugins/abtem/stacks/{ds.id}/section", json={"x0": 0, "y0": 2.5, "x1": 5, "y1": 2.5})
    assert r.status_code == 200, r.text
    body = r.json()
    values = np.frombuffer(base64.b64decode(body["data"]), np.float32).reshape(body["shape"])
    np.testing.assert_allclose(values, _stack()[:, 2, :])
    assert body["step"] == pytest.approx(0.5) and body["z_step"] == pytest.approx(2.0)
    assert client.post(f"/api/plugins/abtem/stacks/{item.id}/section", json={"x0": 0, "y0": 0, "x1": 4, "y1": 0}).status_code == 404
