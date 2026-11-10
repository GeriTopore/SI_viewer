import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

from si_viewer.app import create_app

PNG = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def client():
    return TestClient(create_app())


def _series(label="A", n=100, color="#d62728", **extra):
    x = np.linspace(0, 10, n)
    return {"label": label, "x": x.tolist(), "y": (100 + 50 * np.sin(x)).tolist(), "color": color, **extra}


def _render(client, body):
    return client.post("/api/plot/render", json=body)


def _pixels(png: bytes) -> np.ndarray:
    import matplotlib.image as mpimg

    return mpimg.imread(io.BytesIO(png), format="png")


def test_png_svg_pdf(client):
    body = {"series": [_series()], "options": {"dpi": 100, "width": 4, "height": 3}}
    r = _render(client, body)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "image/png"
    assert r.content.startswith(PNG)
    assert _pixels(r.content).shape[:2] == (300, 400)  # exactly the figure size

    r = _render(client, {**body, "options": {**body["options"], "format": "svg"}})
    assert r.headers["content-type"] == "image/svg+xml"
    assert b"<svg" in r.content[:600]

    r = _render(client, {**body, "options": {**body["options"], "format": "pdf"}})
    assert r.headers["content-type"] == "application/pdf"
    assert r.content.startswith(b"%PDF")


def test_dpi_scales_png(client):
    opts = {"width": 4, "height": 3}
    low = _pixels(_render(client, {"series": [_series()], "options": {**opts, "dpi": 50}}).content)
    high = _pixels(_render(client, {"series": [_series()], "options": {**opts, "dpi": 150}}).content)
    assert low.shape[:2] == (150, 200)
    assert high.shape[:2] == (450, 600)


def test_cascade_and_reverse(client):
    body = {"series": [_series("A"), _series("B", color="#1f77b4")], "options": {"style": "cascade", "dpi": 80}}
    r = _render(client, body)
    assert r.status_code == 200, r.text
    assert r.content.startswith(PNG)
    reverse = _render(client, {**body, "options": {**body["options"], "reverse": True, "padding": 0.5, "normalise": True}})
    assert reverse.status_code == 200, reverse.text
    assert reverse.content != r.content


def test_cascade_offsets_follow_hyperspy():
    from si_viewer.processing.plot_export import PlotRequest, _draw

    request = PlotRequest(
        series=[_series("A"), _series("B")], options={"style": "cascade", "padding": 1, "dpi": 50},
    )
    ax = _draw(request).axes[0]
    first, second = (np.asarray(line.get_ydata()) for line in ax.get_lines())
    assert first.min() == pytest.approx(0)
    assert second.min() == pytest.approx(1)  # one padding above the first
    assert ax.get_yticks().size == 0  # arbitrary units: no y tick values

    reverse = PlotRequest(series=request.series, options={**request.options.model_dump(), "reverse": True})
    ax = _draw(reverse).axes[0]
    first, second = (np.asarray(line.get_ydata()) for line in ax.get_lines())
    assert first.min() > second.min()


def test_right_axis(client):
    from si_viewer.processing.plot_export import PlotRequest, _draw

    request = {
        "series": [_series("counts"), {**_series("sum", color="#1f77b4"), "axis": "right"}],
        "options": {"dpi": 60, "y_label": "Counts", "y2_label": "Sum", "y2lim": [0, 500], "log": False},
    }
    assert _render(client, request).status_code == 200
    fig = _draw(PlotRequest(**request))
    assert len(fig.axes) == 2
    assert fig.axes[1].get_ylim() == (0, 500)
    assert fig.axes[1].get_ylabel() == "Sum"
    assert fig.axes[0].get_ylabel() == "Counts"
    assert len(fig.axes[0].get_lines()) == 1 and len(fig.axes[1].get_lines()) == 1


def test_right_axis_ignored_in_cascade(client):
    from si_viewer.processing.plot_export import PlotRequest, _draw

    request = {
        "series": [_series("A"), {**_series("B"), "axis": "right"}],
        "options": {"style": "cascade", "dpi": 50},
    }
    assert _render(client, request).status_code == 200
    assert len(_draw(PlotRequest(**request)).axes) == 1


def test_only_right_series_use_the_left_axis():
    from si_viewer.processing.plot_export import PlotRequest, _draw

    fig = _draw(PlotRequest(series=[{**_series("B"), "axis": "right"}], options={"dpi": 50}))
    assert len(fig.axes) == 1


def test_limits_and_log():
    from si_viewer.processing.plot_export import PlotRequest, _draw

    request = PlotRequest(
        series=[_series()],
        options={"dpi": 50, "xlim": [2, None], "ylim": [10, 400], "title": "T", "x_label": "Energy (eV)"},
    )
    ax = _draw(request).axes[0]
    assert ax.get_xlim() == (2, 10)
    assert ax.get_ylim() == (10, 400)
    assert ax.get_title() == "T"
    assert ax.get_xlabel() == "Energy (eV)"

    # Automatic: counts from zero, fitted to the part of the curve in view.
    auto = _draw(PlotRequest(series=[_series()], options={"dpi": 50})).axes[0]
    assert auto.get_ylim()[0] == 0
    assert auto.get_xlim() == (0, 10)

    logged = _draw(PlotRequest(series=[_series()], options={"dpi": 50, "log": True})).axes[0]
    assert logged.get_yscale() == "log"
    assert logged.get_ylim()[0] > 0


def test_log_with_explicit_limits(client):
    r = _render(client, {"series": [_series()], "options": {"log": True, "ylim": [50, 200], "dpi": 50}})
    assert r.status_code == 200, r.text


def test_non_uniform_x(client):
    x = np.array([0, 1, 3, 7, 12, 20.0])
    body = {
        "series": [{"label": "A", "x": x.tolist(), "y": [1, 3, 2, 5, 4, 6], "color": "#000000"}],
        "options": {"dpi": 60},
    }
    assert _render(client, body).status_code == 200
    from si_viewer.processing.plot_export import PlotRequest, _draw

    ax = _draw(PlotRequest(**body)).axes[0]
    np.testing.assert_allclose(ax.get_lines()[0].get_xdata(), x)
    # And a cascade with series of different lengths and spacings.
    body["series"].append(_series("B", n=50))
    body["options"]["style"] = "cascade"
    assert _render(client, body).status_code == 200


def test_different_x_ranges_overlap(client):
    a = {"label": "A", "x": [0, 1, 2, 3], "y": [1, 2, 3, 4], "color": "#ff0000"}
    b = {"label": "B", "x": [2, 4, 6], "y": [4, 3, 2], "color": "#0000ff", "axis": "right"}
    from si_viewer.processing.plot_export import PlotRequest, _draw

    request = {"series": [a, b], "options": {"dpi": 50}}
    assert _render(client, request).status_code == 200
    assert _draw(PlotRequest(**request)).axes[0].get_xlim() == (0, 6)


def test_nulls_become_gaps(client):
    from si_viewer.processing.plot_export import PlotRequest, _draw

    s = _series(n=20)
    s["y"][5:9] = [None] * 4
    request = {"series": [s], "options": {"dpi": 50}}
    assert _render(client, request).status_code == 200
    y = np.asarray(_draw(PlotRequest(**request)).axes[0].get_lines()[0].get_ydata())
    assert np.isnan(y[5:9]).all() and np.isfinite(y[:5]).all() and np.isfinite(y[9:]).all()
    # The same through the cascade and with normalisation.
    for opts in ({"style": "cascade"}, {"normalise": True}, {"log": True}):
        r = _render(client, {"series": [s, _series("B")], "options": {"dpi": 50, **opts}})
        assert r.status_code == 200, (opts, r.text)


def test_log_masks_non_positive_values():
    from si_viewer.processing.plot_export import PlotRequest, _draw

    s = {"label": "A", "x": [0, 1, 2, 3], "y": [0, 10, -5, 1000], "color": "#000000"}
    ax = _draw(PlotRequest(series=[s], options={"log": True, "dpi": 50})).axes[0]
    y = np.asarray(ax.get_lines()[0].get_ydata())
    assert np.isnan(y[[0, 2]]).all()
    assert ax.get_ylim() == (10, 1000)


def test_markers_windows_and_legend(client):
    body = {
        "series": [_series("A"), _series("B", color="#2ca02c", dash=[6, 4], width=2)],
        "markers": [
            {"x": 3, "label": "Fe Kα", "color": "#ff7f0e", "weight": 1},
            {"x": 3.1, "label": "Fe Kβ", "color": "#ff7f0e", "weight": 0.3},
            {"x": 5, "color": "#ff7f0e", "weight": 0.1},
            {"x": 50, "label": "off", "color": "#000000", "weight": 1},  # outside the range
        ],
        "windows": [
            {"lo": 2, "hi": 4, "color": "#1f77b4", "label": "Window"},
            {"lo": 7, "hi": 7, "color": "#d62728", "label": "Threshold"},
        ],
        "options": {"dpi": 80, "legend": True, "legend_loc": "upper left", "legend_size": 8},
    }
    r = _render(client, body)
    assert r.status_code == 200, r.text
    plain = _render(client, {**body, "markers": [], "windows": []})
    assert r.content != plain.content

    from si_viewer.processing.plot_export import PlotRequest, _draw

    ax = _draw(PlotRequest(**body)).axes[0]
    legend = ax.get_legend()
    assert [t.get_text() for t in legend.get_texts()] == ["A", "B"]
    # Series 2 is drawn thicker (width 2 against the reference 1.25).
    widths = [line.get_linewidth() for line in ax.get_lines()[:2]]
    assert widths[1] == pytest.approx(widths[0] * 2 / 1.25)

    off = _draw(PlotRequest(**{**body, "options": {**body["options"], "legend": False}})).axes[0]
    assert off.get_legend() is None


def test_font_sizes_and_line_width():
    from si_viewer.processing.plot_export import PlotRequest, _draw

    request = PlotRequest(
        series=[_series()],
        options={"dpi": 50, "title": "T", "x_label": "x", "y_label": "y", "title_size": 20, "label_size": 15,
                 "tick_size": 7, "line_width": 3},
    )
    ax = _draw(request).axes[0]
    assert ax.title.get_fontsize() == 20
    assert ax.xaxis.label.get_fontsize() == 15
    assert ax.xaxis.get_ticklabels()[0].get_fontsize() == 7
    assert ax.get_lines()[0].get_linewidth() == pytest.approx(3 * 1.25 / 1.25)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda b: b.update(series=[]), "no series"),
        (lambda b: b["series"][0].update(y=[1, 2]), "x values but 2 y values"),
        (lambda b: b["series"][0].update(x=[1, 0], y=[1, 2]), "increasing"),
        (lambda b: b["series"][0].update(color="notacolour"), "unknown colour"),
        (lambda b: b["series"][0].update(y=[None] * 100), "has no values"),
        (lambda b: b["options"].update(style="cascade", log=True), "cascade"),
        (lambda b: b["options"].update(xlim=[5, 5]), "X minimum"),
        (lambda b: b["options"].update(ylim=[0, 10], log=True), "positive"),
        (lambda b: b["options"].update(xlim=[20, None]), "x limits"),
        (lambda b: b["options"].update(ylim=[500, None]), "y limits"),
        (lambda b: b["options"].update(legend_loc="nowhere"), "legend location"),
        (lambda b: b["options"].update(dpi=5), "dpi"),
        (lambda b: b["options"].update(width=100), "inches"),
        (lambda b: b["options"].update(width=40, height=40, dpi=1000), "too large"),
        (lambda b: b["options"].update(title_size=0), "font size"),
        (lambda b: b["options"].update(padding=-1), "Spacing"),
        (lambda b: b["series"][0].update(y=[3.0] * 100) or b["options"].update(style="cascade"), "constant"),
    ],
)
def test_validation_errors(client, mutate, message):
    body = {"series": [_series()], "options": {"dpi": 50}}
    mutate(body)
    r = _render(client, body)
    assert r.status_code == 422, r.text
    assert message in str(r.json()["detail"])


def test_bad_format_is_rejected(client):
    r = _render(client, {"series": [_series()], "options": {"format": "gif"}})
    assert r.status_code == 422


def test_concurrent_requests(client):
    from concurrent.futures import ThreadPoolExecutor

    body = {"series": [_series(), _series("B")], "markers": [{"x": 3, "label": "a", "color": "#000"}], "options": {"dpi": 50}}
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda _: _render(client, body), range(8)))
    assert all(r.status_code == 200 and r.content.startswith(PNG) for r in results)
    assert len({r.content for r in results}) == 1
