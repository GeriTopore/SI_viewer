"""Saving a drift correction: its shifts as a CSV and a two-page PDF report.

Both describe the correction the way the app shows it: drift is each frame's
position relative to the mean of the frames, in image pixels (x right, y
down); the shift that aligns a frame is minus its drift. Lengths in Å need
the image axis' scale and units (other or missing units: pixels only).
"""

from __future__ import annotations

import csv
import io
import textwrap
import unicodedata

import numpy as np

# Image axis units → Å.
_TO_ANGSTROM = {
    "m": 1e10, "mm": 1e7, "µm": 1e4, "μm": 1e4, "um": 1e4, "nm": 10.0, "pm": 0.01,
    "Å": 1.0, "A": 1.0, "angstrom": 1.0,
}


def angstrom_per_pixel(scale: float | None, units: str | None) -> float | None:
    """Å per pixel from an axis' scale and units (None: not a known length)."""
    if scale is None or not np.isfinite(scale) or scale <= 0 or not isinstance(units, str):
        return None
    u = unicodedata.normalize("NFC", units.strip())
    factor = _TO_ANGSTROM.get(u) or (1.0 if u.lower() == "angstrom" else None)
    return None if factor is None else float(scale) * factor


def _g(v: float) -> str:
    """6 significant digits (and no "-0")."""
    return f"{float(v) + 0.0:.6g}"


def _frames_text(frames: list[int]) -> str:
    return ", ".join(map(str, frames)) if frames else "none"


def settings_lines(info: dict, title: str, pixel: float | None = None) -> list[str]:
    """What the correction was made with, one fact per line (plain text)."""
    s = info.get("settings", {})
    frames = list(info["frames"])
    first, last = info["first"], info["last"]
    lines = [f"Dataset: {title}", f"Frames registered: {first} to {last - 1}"]
    kept = set(frames)
    out = [f for f in range(first, last) if f not in kept]
    lines.append(f"Frames left out: {_frames_text(out)}")

    m = s.get("mask", {})
    if m.get("type") == "elliptical":
        lines.append(
            f"Fourier mask: elliptical Gaussian, n1 = {_g(m['n'])}, n2 = {_g(m['n2'])}, θ = {_g(m['theta'])}°"
        )
    elif m.get("type") == "none":
        lines.append("Fourier mask: none")
    elif m:
        lines.append(f"Fourier mask: {m['type']}, n = {_g(m['n'])}")

    p = s.get("peaks", {})
    if p:
        corr = {"cc": "cross-correlation", "mc": "mutual correlation", "pc": "phase correlation"}
        method = {"pixel": "brightest pixel", "gf": "Gaussian fit", "com": "centre of mass"}
        text = f"Peak finding: {corr.get(p['correlation'], p['correlation'])}, {method.get(p['method'], p['method'])}"
        if p["method"] == "gf":
            text += (
                f" ({p['num_peaks']} peaks, starting σ = {_g(p['sigma_guess'])} px,"
                f" window radius {p['window_radius']} px)"
            )
        lines.append(text)

    o = s.get("outliers", {})
    if o:
        if o["method"] == "transitivity":
            text = f"Outliers: transitivity, threshold {_g(o['threshold'])} px, up to {o['max_paths']} paths"
        elif o["method"] == "neighbours":
            text = f"Outliers: neighbours, allowed difference {_g(o['max_shift'])} px"
        else:
            text = "Outliers: not tested"
        if o.get("off_diagonal"):
            text += f"; only pairs up to {o['off_diagonal']} frames apart"
        lines.append(text)

    e = s.get("edits") or {}
    if e.get("mask") or e.get("keep"):
        masked, kept = len(e.get("mask", [])), len(e.get("keep", []))
        lines.append(f"Manual edits: {masked} masked, {kept} kept (pairs of frames)")

    x, y, w, h = info["square"]
    lines.append(f"Crop (square): x = {x}, y = {y}, size = {w} px")
    if pixel is not None:
        lines.append(f"Pixel size: {_g(pixel)} Å")
    lines.append(
        "Drift is a frame's position relative to the mean of the frames, in image pixels (x right, y down); "
        "the shift that aligns a frame is minus its drift."
    )
    return lines


def shifts_csv(info: dict, title: str, scale: float | None, units: str) -> str:
    """Each kept frame's shift and drift, after comment lines (``# …``) that
    say how the correction was made. Columns in Å are added when the image
    axis has a known length unit."""
    pixel = angstrom_per_pixel(scale, units)
    out = io.StringIO()
    for line in settings_lines(info, " ".join(title.split()), pixel):
        out.write(f"# {line}\n")
    drift = info["drift"]
    frames = list(drift["frames"])
    dx, dy = np.asarray(drift["x"], float), np.asarray(drift["y"], float)
    header = ["frame", "shift_x_px", "shift_y_px", "drift_x_px", "drift_y_px"]
    if pixel is not None:
        header += ["shift_x_A", "shift_y_A", "drift_x_A", "drift_y_A"]
    w = csv.writer(out, lineterminator="\n")
    w.writerow(header)
    for k, f in enumerate(frames):
        row = [f, _g(-dx[k]), _g(-dy[k]), _g(dx[k]), _g(dy[k])]
        if pixel is not None:
            row += [_g(-dx[k] * pixel), _g(-dy[k] * pixel), _g(dx[k] * pixel), _g(dy[k] * pixel)]
        w.writerow(row)
    return out.getvalue()


# ---------- PDF report ----------

_A4_LANDSCAPE = (11.69, 8.27)
_INK = "#222222"
_MUTED = "#666666"
_MASKED = "#3b3b3b"


def _wrapped(lines: list[str], width: int) -> str:
    return "\n".join(
        textwrap.fill(line, width=width, subsequent_indent="    ", break_long_words=False) for line in lines
    )


def _drift_plot(ax, fig, drift: dict, pixel: float | None) -> None:
    from matplotlib.ticker import MaxNLocator

    k = pixel or 1.0
    unit = "Å" if pixel else "px"
    x, y = np.asarray(drift["x"], float) * k, -np.asarray(drift["y"], float) * k
    frames = drift["frames"]
    ax.plot(x, y, "-", color="#9a9a9a", lw=0.8, zorder=1)
    sc = ax.scatter(x, y, c=frames, cmap="viridis", s=26, zorder=2, edgecolors="white", linewidths=0.4)
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.12)
    ax.set_xlabel(f"drift x ({unit}, right)")
    ax.set_ylabel(f"drift −y ({unit}, up in the image)")
    ax.set_title("Drift of each frame", fontsize=10, color=_INK)
    ax.grid(True, lw=0.3, color="#cccccc")
    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("frame")
    cb.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if len(frames) > 1:
        for i in (0, len(frames) - 1):
            ax.annotate(str(frames[i]), (x[i], y[i]), textcoords="offset points", xytext=(5, 5), fontsize=8, color=_INK)


def _image_axes(ax, image: np.ndarray, pixel: float | None, title: str) -> None:
    k = pixel or 1.0
    lo, hi = np.percentile(image, [0.5, 99.5])
    ax.imshow(image, cmap="gray", vmin=lo, vmax=hi, extent=(0, image.shape[1] * k, image.shape[0] * k, 0), interpolation="nearest")
    ax.set_title(title, fontsize=10, color=_INK)
    unit = "Å" if pixel else "px"
    ax.set_xlabel(f"x ({unit})", fontsize=8)
    ax.set_ylabel(f"y ({unit})", fontsize=8)
    ax.tick_params(labelsize=7)


def _fft_axes(ax, image: np.ndarray, pixel: float | None) -> None:
    ny, nx = image.shape
    window = np.outer(np.hanning(ny), np.hanning(nx))
    f = np.fft.fftshift(np.fft.fft2((image - image.mean()) * window))
    power = np.log10(np.abs(f) ** 2 + 1e-12)
    lo, hi = np.percentile(power, [20, 99.9])
    # Spatial frequency per pixel, ±½ cycle per pixel at the edges.
    k = pixel or 1.0
    fx, fy = 0.5 / k, 0.5 / k
    ax.imshow(power, cmap="gray", vmin=lo, vmax=hi, extent=(-fx, fx, -fy, fy), interpolation="nearest")
    ax.set_title("FFT of the average (log power)", fontsize=10, color=_INK)
    unit = "Å⁻¹" if pixel else "px⁻¹"
    ax.set_xlabel(f"kx ({unit})", fontsize=8)
    ax.set_ylabel(f"ky ({unit})", fontsize=8)
    ax.tick_params(labelsize=7)


def _page1(session, info: dict, title: str, pixel: float | None):
    from matplotlib.figure import Figure

    fig = Figure(figsize=_A4_LANDSCAPE)
    fig.text(0.05, 0.935, "Drift correction report", fontsize=17, fontweight="bold", color=_INK, va="center")
    fig.text(0.05, 0.895, " ".join(title.split()), fontsize=11, color=_MUTED, va="center")

    lines = settings_lines(info, " ".join(title.split()), pixel)[1:]  # the title is above
    fig.text(0.05, 0.84, _wrapped(lines, 42), fontsize=9, color=_INK, va="top", linespacing=1.45)

    gs = fig.add_gridspec(
        2, 2, left=0.40, right=0.97, top=0.86, bottom=0.08, width_ratios=[1.25, 1], wspace=0.28, hspace=0.32,
    )
    _drift_plot(fig.add_subplot(gs[:, 0]), fig, info["drift"], pixel)
    average = session.average()
    _image_axes(fig.add_subplot(gs[0, 1]), average, pixel, f"Aligned average ({len(info['frames'])} frames)")
    _fft_axes(fig.add_subplot(gs[1, 1]), average, pixel)
    return fig


def _page2(session):
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    from .registration import MaskEdits, OutlierSettings

    c = session.correction
    measured = session.matrices()
    corrected = session.corrected_matrices()
    # The mask the correction used (so the panels match the drift above).
    mask = session.outlier_mask(OutlierSettings(**c["outliers"]), c["bad_frames"], MaskEdits(**c["edits"])) > 0
    known = corrected[2] > 0

    cmap = matplotlib.colormaps["RdBu_r"].with_extremes(bad=_MASKED)
    first, last = session.first, session.last
    extent = (first - 0.5, last - 0.5, last - 0.5, first - 0.5)

    fig = Figure(figsize=_A4_LANDSCAPE)
    fig.text(0.05, 0.945, "Shift matrices", fontsize=15, fontweight="bold", color=_INK, va="center")
    fig.text(
        0.05, 0.905, "Shift (px) of frame j (column) relative to frame i (row), before and after correcting the masked pairs.",
        fontsize=9, color=_MUTED, va="center",
    )
    gs = fig.add_gridspec(2, 2, left=0.15, right=0.87, top=0.86, bottom=0.12, wspace=0.16, hspace=0.30)
    for col, axis in enumerate(("x", "y")):
        lim = float(np.abs(measured[col]).max()) or 1.0
        panels = (
            ("measured", np.ma.masked_where(~mask, measured[col])),
            ("corrected", np.ma.masked_where(~known, corrected[col])),
        )
        for row, (what, data) in enumerate(panels):
            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(data, cmap=cmap, vmin=-lim, vmax=lim, extent=extent, interpolation="nearest")
            ax.set_title(f"Δ{axis} {what}", fontsize=10, color=_INK)
            ax.set_xlabel("frame j", fontsize=8)
            ax.set_ylabel("frame i", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.set_label("px", fontsize=8)
            cb.ax.tick_params(labelsize=7)
    fig.text(
        0.05, 0.045,
        "Dark entries are masked (outliers, pairs too far apart, bad frames, manual edits) in the measured shifts,\n"
        "and unknown in the corrected ones. Measured and corrected share one colour range per axis.",
        fontsize=8, color=_MUTED, va="center",
    )
    return fig


def report_pdf(session, title: str, scale: float | None, units: str) -> bytes:
    """A two-page A4 PDF of the session's correction: its settings, the
    drift, the aligned average and its FFT, then the shift matrices."""
    from matplotlib.backends.backend_pdf import PdfPages

    if session.correction is None:
        raise ValueError("Correct the shift matrices first")
    pixel = angstrom_per_pixel(scale, units)
    info = session.applied()["info"]
    buffer = io.BytesIO()
    with PdfPages(buffer, metadata={"Title": f"Drift correction report: {title}", "Creator": "SI Viewer"}) as pdf:
        for make in (lambda: _page1(session, info, title, pixel), lambda: _page2(session)):
            fig = make()
            try:
                pdf.savefig(fig)
            finally:
                fig.clear()
    return buffer.getvalue()
