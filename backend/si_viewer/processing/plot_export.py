"""Publication-style export of 1D plots (spectra, line profiles) as PNG, SVG or
PDF, drawn with matplotlib.

The curves are drawn by HyperSpy's ``plot_spectra`` (overlap or cascade, with
its normalisation and cascade offsets); the secondary axis, the X-ray line
markers, the energy windows and the legend are added with matplotlib on the
axes it returns.
"""

from __future__ import annotations

import io
import threading
from typing import Literal

import numpy as np
from pydantic import BaseModel

Axis = Literal["left", "right"]
Limits = list[float | None]

LEGEND_LOCATIONS = (
    "best", "upper right", "upper left", "lower left", "lower right", "right", "center left", "center right",
    "lower center", "upper center", "center",
)
MEDIA_TYPES = {"png": "image/png", "svg": "image/svg+xml", "pdf": "application/pdf"}
# Largest raster (pixels) and number of points we draw.
MAX_PIXELS = 50_000_000
MAX_POINTS = 2_000_000
# Series line width (css px of the live plot) that gets the base line width.
REFERENCE_WIDTH = 1.25

# Matplotlib figures are not thread-safe (rc settings, the text cache).
_lock = threading.Lock()


class PlotSeries(BaseModel):
    label: str = ""
    x: list[float]
    y: list[float | None]
    color: str = "#1f77b4"
    #: Dash pattern (on, off, ...) as in the live plot.
    dash: list[float] | None = None
    #: Line width relative to the live plot's (1.25 = the base width).
    width: float | None = None
    axis: Axis = "left"


class PlotMarker(BaseModel):
    """A vertical line (an X-ray line): its height follows ``weight``."""

    x: float
    label: str | None = None
    color: str = "#888888"
    weight: float = 1.0


class PlotWindow(BaseModel):
    """A shaded band (an energy window); lo == hi is a single line."""

    lo: float
    hi: float
    color: str = "#888888"
    label: str = ""


class PlotOptions(BaseModel):
    style: Literal["overlap", "cascade"] = "overlap"
    #: Spacing between the curves of a cascade (1 = no overlap).
    padding: float = 1.0
    normalise: bool = False
    #: Cascade: the first series on top.
    reverse: bool = False
    log: bool = False
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    y2_label: str = ""
    xlim: Limits = [None, None]
    ylim: Limits = [None, None]
    y2lim: Limits = [None, None]
    title_size: float = 12
    label_size: float = 11
    tick_size: float = 10
    legend_size: float = 10
    legend: bool = True
    legend_loc: str = "best"
    #: Base line width (pt).
    line_width: float = 1.5
    #: Figure size (inches), resolution of a PNG, and the file format.
    width: float = 6
    height: float = 4
    dpi: int = 300
    format: Literal["png", "svg", "pdf"] = "png"


class PlotRequest(BaseModel):
    series: list[PlotSeries]
    markers: list[PlotMarker] = []
    windows: list[PlotWindow] = []
    options: PlotOptions = PlotOptions()


def _check(request: PlotRequest) -> None:
    """Raise ValueError with a plain-English message for a request we can't draw."""
    from matplotlib.colors import is_color_like

    o = request.options
    if not request.series:
        raise ValueError("There is nothing to plot: no series")
    if sum(len(s.x) for s in request.series) > MAX_POINTS:
        raise ValueError("Too many points to plot")
    for i, s in enumerate(request.series):
        name = s.label or f"Series {i + 1}"
        if len(s.x) < 2:
            raise ValueError(f"{name} needs at least two points")
        if len(s.x) != len(s.y):
            raise ValueError(f"{name} has {len(s.x)} x values but {len(s.y)} y values")
        x = np.asarray(s.x, float)
        if not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0):
            raise ValueError(f"The x values of {name} must be finite and increasing")
        if not is_color_like(s.color):
            raise ValueError(f"{name} has an unknown colour: {s.color}")
        if s.dash is not None and (len(s.dash) > 8 or any(not (d >= 0) or d > 1000 for d in s.dash) or
                                   (s.dash and sum(s.dash) <= 0)):
            raise ValueError(f"{name} has an invalid dash pattern")
        if s.width is not None and not 0 < s.width <= 50:
            raise ValueError(f"The line width of {name} must be between 0 and 50")
    for m in request.markers:
        if not is_color_like(m.color):
            raise ValueError(f"Unknown marker colour: {m.color}")
    for w in request.windows:
        if not is_color_like(w.color):
            raise ValueError(f"Unknown window colour: {w.color}")
        if w.hi < w.lo:
            raise ValueError("A window's upper limit is below its lower limit")
    if not 0 <= o.padding <= 100:
        raise ValueError("Spacing must be between 0 and 100")
    if o.log and o.style == "cascade":
        raise ValueError("A cascade can't have a logarithmic axis")
    for name, size in (("Title", o.title_size), ("Label", o.label_size), ("Tick", o.tick_size), ("Legend", o.legend_size)):
        if not 1 <= size <= 100:
            raise ValueError(f"{name} font size must be between 1 and 100 pt")
    if not 0.1 <= o.line_width <= 20:
        raise ValueError("Line width must be between 0.1 and 20 pt")
    if o.legend_loc not in LEGEND_LOCATIONS:
        raise ValueError(f"Unknown legend location: {o.legend_loc}")
    if not (1 <= o.width <= 40 and 1 <= o.height <= 40):
        raise ValueError("The figure must be between 1 and 40 inches on each side")
    if not 30 <= o.dpi <= 1200:
        raise ValueError("Resolution must be between 30 and 1200 dpi")
    if o.format == "png" and o.width * o.height * o.dpi**2 > MAX_PIXELS:
        raise ValueError("The image is too large: use a smaller figure or a lower resolution")
    for name, lim in (("X", o.xlim), ("Y", o.ylim), ("Y2", o.y2lim)):
        if len(lim) != 2:
            raise ValueError(f"{name} limits need a minimum and a maximum")
        if any(v is not None and not np.isfinite(v) for v in lim):
            raise ValueError(f"{name} limits must be numbers")
        if lim[0] is not None and lim[1] is not None and lim[0] >= lim[1]:
            raise ValueError(f"{name} minimum must be below the maximum")
        if o.log and name != "X" and any(v is not None and v <= 0 for v in lim):
            raise ValueError(f"{name} limits must be positive on a logarithmic axis")


def _signal(label: str, x: np.ndarray, y: np.ndarray):
    """A Signal1D along ``x``: a uniform axis when the steps are equal, else a
    non-uniform one."""
    import hyperspy.api as hs

    step = np.diff(x)
    uniform = np.max(np.abs(step - step.mean())) <= 1e-9 * step.mean()
    axis = (
        {"scale": float(step.mean()), "offset": float(x[0]), "size": len(x)} if uniform else {"axis": x}
    )
    signal = hs.signals.Signal1D(y, axes=[{**axis, "name": "x"}])
    signal.metadata.General.title = label
    return signal


def _auto_limits(lines, xlim: tuple[float, float], log: bool, cascade: bool) -> tuple[float, float]:
    """The y range that fits what is drawn between ``xlim``: from zero for
    counts (as the live plot), to powers of ten on a log axis."""
    values = []
    for line in lines:
        x, y = (np.asarray(a, float) for a in line.get_data())
        keep = np.isfinite(y) & (x >= xlim[0]) & (x <= xlim[1])
        if log:
            keep &= y > 0
        values.append(y[keep])
    v = np.concatenate(values) if values else np.empty(0)
    if not v.size:
        return (1.0, 10.0) if log else (0.0, 1.0)
    lo, hi = float(v.min()), float(v.max())
    if log:
        lo, hi = 10 ** np.floor(np.log10(lo)), 10 ** np.ceil(np.log10(hi))
        return (lo, hi * 10 if hi <= lo else hi)
    span = (hi - lo) or abs(hi) or 1.0
    if lo >= 0 and not cascade:
        return 0.0, hi + 0.05 * (hi if hi > 0 else 1.0)
    return lo - 0.05 * span, hi + 0.05 * span


def _set_limits(ax, lines, user: Limits, xlim: tuple[float, float], log: bool, cascade: bool) -> None:
    lo, hi = _auto_limits(lines, xlim, log, cascade)
    lo = lo if user[0] is None else user[0]
    hi = hi if user[1] is None else user[1]
    if not lo < hi:
        raise ValueError("The y limits leave no range: the minimum is not below the maximum")
    ax.set_ylim(lo, hi)


def _dashes(dash: list[float] | None):
    """A matplotlib line style for a canvas dash pattern (px of the live plot)."""
    if not dash:
        return "-"
    pattern = [d * 0.5 for d in dash]
    if len(pattern) % 2:
        pattern = pattern * 2
    return (0, tuple(pattern))


def _normalise(signal) -> np.ndarray:
    """Min-max normalisation (a constant curve becomes zero)."""
    data = np.asarray(signal.data, float)
    span = data.max() - data.min()
    return (data - data.min()) / span if span > 0 else np.zeros_like(data)


def _plot_group(hs, ax, group, o: PlotOptions, cascade: bool, padding: float):
    """Draw ``group`` ((index, series) pairs) on ``ax`` with HyperSpy's
    plot_spectra; the lines come back with their series, NaN restored."""
    signals, masks = [], []
    for i, s in group:
        y = np.array([np.nan if v is None else v for v in s.y], float)
        bad = ~np.isfinite(y)
        if bad.all():
            raise ValueError(f"{s.label or f'Series {i + 1}'} has no values")
        if o.log:
            bad |= y <= 0
            if bad.all():
                raise ValueError(f"{s.label or f'Series {i + 1}'} has no positive values for a logarithmic axis")
        # HyperSpy needs numbers: gaps are filled with the curve's minimum
        # (leaving its range, hence the cascade offsets, unchanged) and
        # blanked again on the drawn lines.
        filled = np.where(bad, np.min(y[~bad]), y)
        if cascade or o.normalise:
            if filled.max() == filled.min() and cascade:
                raise ValueError(f"{s.label or f'Series {i + 1}'} is constant: it can't be drawn in a cascade")
        signals.append(_signal(s.label, np.asarray(s.x, float), filled))
        masks.append(bad)
    width = o.line_width
    hs.plot.plot_spectra(
        signals,
        style=o.style,
        padding=padding,
        normalise=_normalise if o.normalise else False,
        color=[s.color for _, s in group],
        linestyle=[_dashes(s.dash) for _, s in group],
        legend=None,
        auto_update=False,
        fig=ax.get_figure(),
        ax=ax,
    )
    lines = ax.get_lines()[-len(group):]
    for line, (_, s), bad in zip(lines, group, masks):
        y = np.array(line.get_ydata(), float)
        y[bad] = np.nan
        line.set_ydata(y)
        line.set_linewidth(width * (s.width if s.width is not None else REFERENCE_WIDTH) / REFERENCE_WIDTH)
        line.set_solid_capstyle("butt")
    return lines


def _draw(request: PlotRequest):
    """The figure for a request (not yet saved)."""
    import hyperspy.api as hs
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.transforms import blended_transform_factory

    o = request.options
    cascade = o.style == "cascade"
    indexed = list(enumerate(request.series))
    # A cascade has one (offset) axis; with nothing on the left, so has an overlap.
    left = [(i, s) for i, s in indexed if cascade or s.axis == "left"]
    right = [(i, s) for i, s in indexed if not cascade and s.axis == "right"]
    if not left:
        left, right = right, []

    fig = Figure(figsize=(o.width, o.height), dpi=o.dpi, facecolor="white", layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax2 = ax.twinx() if right else None
    if ax2 is not None:
        ax2.set_zorder(ax.get_zorder() + 1)
        ax2.patch.set_visible(False)

    padding = -o.padding if cascade and o.reverse else o.padding
    drawn = []  # (index, series, line)
    on_axis = []  # (axes, its lines, its y limits)
    for target, group, user in ((ax, left, o.ylim), (ax2, right, o.y2lim)):
        if not group:
            continue
        lines = _plot_group(hs, target, group, o, cascade, padding)
        drawn += [(i, s, line) for (i, s), line in zip(group, lines)]
        on_axis.append((target, lines, user))

    # Energy range: what the series cover, or the limits given.
    xs = [np.asarray(s.x, float) for s in request.series]
    xlim = [min(x[0] for x in xs), max(x[-1] for x in xs)]
    for k in (0, 1):
        if o.xlim[k] is not None:
            xlim[k] = o.xlim[k]
    if not xlim[0] < xlim[1]:
        raise ValueError("The x limits leave no range: the minimum is not below the maximum")
    ax.set_xlim(*xlim)

    for target, lines, user in on_axis:
        if o.log:
            target.set_yscale("log", nonpositive="mask")
        _set_limits(target, lines, user, tuple(xlim), o.log, cascade)

    # Labels and text.
    ax.tick_params(labelsize=o.tick_size)
    ax.set_xlabel(o.x_label, fontsize=o.label_size)
    ax.set_ylabel(o.y_label, fontsize=o.label_size)
    if cascade:
        ax.set_yticks([])
    if ax2 is not None:
        ax2.tick_params(labelsize=o.tick_size)
        ax2.set_ylabel(o.y2_label, fontsize=o.label_size)
    if o.title:
        ax.set_title(o.title, fontsize=o.title_size)

    # Energy windows: translucent bands with solid edges.
    top = blended_transform_factory(ax.transData, ax.transAxes)
    for w in request.windows:
        if w.hi < xlim[0] or w.lo > xlim[1]:
            continue
        if w.hi > w.lo:
            ax.axvspan(w.lo, w.hi, facecolor=w.color, alpha=0.14, linewidth=0, zorder=0.5)
        for edge in {w.lo, w.hi}:
            ax.axvline(edge, color=w.color, alpha=0.85, linewidth=1 if w.hi > w.lo else 1.5, zorder=0.6)
        if w.label:
            # At the bottom, clear of the X-ray line labels at the top.
            ax.annotate(
                w.label, (max(w.lo, xlim[0]), 0), xycoords=top, xytext=(3, 3), textcoords="offset points",
                va="bottom", ha="left", fontsize=o.tick_size * 0.9, color=w.color, annotation_clip=False,
            )

    # Settle the layout, then place the X-ray lines (labels staggered into
    # rows so they don't overlap).
    canvas = fig.canvas
    canvas.draw()
    visible = [m for m in request.markers if xlim[0] <= m.x <= xlim[1]]
    if visible:
        renderer = canvas.get_renderer()
        height_pt = ax.bbox.height / fig.dpi * 72
        row_pt = o.tick_size * 1.4
        row_ends: list[float] = []
        for m in sorted(visible, key=lambda m: m.x):
            h = 0.25 + 0.6 * min(1.0, max(0.0, m.weight))
            ax.axvline(m.x, ymin=0, ymax=h, color=m.color, alpha=0.6 if m.label else 0.35, linewidth=1, zorder=1.5)
            if not m.label:
                continue
            px = float(ax.transData.transform((m.x, 0))[0])
            text = ax.text(m.x, 1, m.label, transform=top, va="top", ha="left", fontsize=o.tick_size, color=m.color)
            width_px = text.get_window_extent(renderer).width
            row = next((r for r, end in enumerate(row_ends) if end < px - 2), None)
            if row is None:
                row = min(len(row_ends), 3)
                if row == len(row_ends):
                    row_ends.append(0.0)
            row_ends[row] = px + 3 + width_px
            text.set_y(1 - (4 + row * row_pt) / height_pt)

    # One legend for both axes, in the order of the series (a cascade lists
    # the top curve first, as HyperSpy's own legend does).
    if o.legend:
        entries = sorted(((i, s, line) for i, s, line in drawn if s.label), key=lambda e: e[0])
        if cascade and not o.reverse:
            entries.reverse()
        if entries:
            host = ax2 if ax2 is not None else ax
            host.legend(
                [line for _, _, line in entries], [s.label for _, s, _ in entries],
                loc=o.legend_loc, fontsize=o.legend_size, frameon=False,
            )
    return fig


def render_plot(request: PlotRequest) -> tuple[bytes, str]:
    """The plot as image bytes and their media type. Raises ValueError with a
    plain-English message when the request can't be drawn."""
    import matplotlib

    _check(request)
    o = request.options
    with _lock:
        with matplotlib.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path"}):
            fig = _draw(request)
            try:
                buf = io.BytesIO()
                fig.savefig(buf, format=o.format, dpi=o.dpi, facecolor="white")
            finally:
                fig.clear()
    return buf.getvalue(), MEDIA_TYPES[o.format]
