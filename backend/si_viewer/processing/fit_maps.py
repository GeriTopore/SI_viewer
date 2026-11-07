"""Per-pixel fitted maps of an EDS spectrum image (M4, phase 2).

A full non-linear fit at every pixel is far too slow, and pixels have too few
counts for it anyway. Instead the summed spectrum is fitted once (line
positions, widths and minor-line ratios), and each pixel is then solved for
the line amplitudes and a smooth background only. That is linear least
squares with one design matrix for all pixels, done in chunks as matrix
products. Because it is linear, the per-pixel intensities add up to what
the same linear fit gives for the summed spectrum.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .fitting import FitOptions, FittedModel


def design_matrix(fitted: FittedModel, options: FitOptions) -> tuple[list[str], np.ndarray]:
    """Columns: each line family (main line plus its minor lines at fixed
    ratios, scaled so the coefficient is the main line's intensity), then a
    Legendre polynomial background. Returns the family names and the matrix
    (channels × columns)."""
    m = fitted.model
    families: dict[str, list] = {}
    for c in m:
        if c.name.startswith("background"):
            continue
        twin = c.A.twin
        root = twin.component if twin is not None else c
        families.setdefault(root.name, []).append(c)

    columns, names = [], []
    for name, comps in families.items():
        root = next(c for c in comps if c.name == name)
        saved = root.A.value
        root.A.value = 1.0  # minor lines follow through their twins
        try:
            columns.append(np.asarray(m.as_signal(component_list=comps, show_progressbar=False).data, float))
        finally:
            root.A.value = saved
        names.append(name)

    x = fitted.spectrum.axes_manager.signal_axes[0].axis
    t = 2 * (x - x[0]) / (x[-1] - x[0]) - 1 if len(x) > 1 else np.zeros_like(x)
    background = np.polynomial.legendre.legvander(t, options.background_order)
    return names, np.column_stack([*columns, background])


def fit_pixels(
    si_signal,
    fitted: FittedModel,
    options: FitOptions,
    report: Callable[[float], None] = lambda f: None,
    chunk: int = 4096,
) -> dict[str, np.ndarray]:
    """Fitted intensity map (y, x) for every line family of ``fitted``."""
    names, matrix = design_matrix(fitted, options)
    solve = np.linalg.pinv(matrix).T.astype(np.float32)  # (channels, columns)

    full_axis = si_signal.axes_manager.signal_axes[0]
    x0 = fitted.spectrum.axes_manager.signal_axes[0].axis[0]
    i0 = int(full_axis.value2index(x0))
    n = matrix.shape[0]
    data = si_signal.data
    h, w = data.shape[:2]
    flat = data.reshape(h * w, data.shape[-1])

    coeffs = np.empty((h * w, len(names)), np.float32)
    for start in range(0, h * w, chunk):
        stop = min(start + chunk, h * w)
        block = flat[start:stop, i0:i0 + n].astype(np.float32)
        coeffs[start:stop] = (block @ solve)[:, : len(names)]
        report(stop / (h * w))
    return {name: coeffs[:, j].reshape(h, w) for j, name in enumerate(names)}


def composition_maps(
    intensities: dict[str, np.ndarray],
    kfactors: dict[str, float],
    atomic_weights: dict[str, float],
    min_counts: float = 0.0,
) -> dict[str, np.ndarray]:
    """Cliff-Lorimer at% per pixel: wtᵢ ∝ kᵢ·Iᵢ (negative intensities count as
    0), atᵢ ∝ wtᵢ/Aᵢ. ``intensities`` and ``kfactors`` are keyed by element.
    Pixels whose element counts add up to less than ``min_counts`` (e.g.
    vacuum or the support film, where a stray count would read as 100%) and
    pixels with no signal are NaN."""
    elements = list(intensities)
    counts = np.stack([np.clip(intensities[el], 0, None) for el in elements])
    weighted = counts * np.array([kfactors.get(el, 1.0) for el in elements])[:, None, None]
    molar = weighted / np.array([atomic_weights[el] for el in elements])[:, None, None]
    total = molar.sum(axis=0)
    keep = (total > 0) & (counts.sum(axis=0) >= min_counts)
    with np.errstate(invalid="ignore", divide="ignore"):
        at = np.where(keep, 100 * molar / total, np.nan)
    return {el: at[i].astype(np.float32) for i, el in enumerate(elements)}
