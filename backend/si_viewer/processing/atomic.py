"""Atomic-resolution image analysis with kemstem (M22): glue only.

kemstem works on plain arrays: images normalised to 0–1, points as (n, 2)
arrays of (y, x) pixels, FFTs centred (``fftshift``). FFT peaks are kept here
and in the frontend as spatial frequencies in cycles per image pixel
``(fy, fx)``: pixel (r, c) of the centred FFT of an H × W image is
``((r - H // 2) / H, (c - W // 2) / W)``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import numpy as np

# kemstem's loops print tqdm bars; keep the server log clean.
os.environ.setdefault("TQDM_DISABLE", "1")

Method = Literal["gauss", "com"]
Edge = Literal["none", "hann", "pplus"]
Pattern = Literal["log", "abs"]


def _kemstem():
    import kemstem

    return kemstem


@dataclass(frozen=True)
class PeakOptions:
    method: Method = "gauss"
    # Full width of the fit window (Gaussian) or of the centre-of-mass crop (px of the FFT).
    window: int = 9
    # Gaussian blur (σ, px) of the pattern before refining, as in kemstem's notebooks.
    blur: float = 1.0
    edge: Edge = "none"
    # The pattern refined: log(1 + |F|) (positions) or |F| (backgrounds, for damping).
    pattern: Pattern = "log"

    def check(self) -> "PeakOptions":
        if self.window < 3 or self.window % 2 == 0:
            raise ValueError("The window must be an odd number of pixels, 3 or more")
        if self.blur < 0:
            raise ValueError("The blur must be 0 or more")
        return self


def fourier_pattern(image: np.ndarray, edge: Edge = "none") -> np.ndarray:
    """kemstem's centred complex FFT of the normalised image."""
    k = _kemstem()
    return k.prepare_fourier_pattern(k.normalize(np.asarray(image, dtype=np.float64)),
                                     edge_reduction=None if edge == "none" else edge)


def to_pixels(freqs: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """(n, 2) frequencies (fy, fx; cycles/px) → (y, x) pixels of the centred FFT."""
    h, w = shape
    f = np.asarray(freqs, dtype=np.float64).reshape(-1, 2)
    return np.stack([f[:, 0] * h + h // 2, f[:, 1] * w + w // 2], axis=1)


def to_freqs(pixels: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    p = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    return np.stack([(p[:, 0] - h // 2) / h, (p[:, 1] - w // 2) / w], axis=1)


def refine_peaks(image: np.ndarray, freqs: np.ndarray, opts: PeakOptions) -> list[dict]:
    """Refine FFT peaks of ``image`` (frequencies) with kemstem: 2D Gaussian
    fits (``refine_peaks_gf``) or centre of mass (``refine_peaks_com``) on the
    blurred log (or amplitude) pattern. One dict per peak, in order: ``fy``,
    ``fx`` and ``failed`` (the input kept), and for Gaussian fits
    ``amplitude``, ``background``, ``sigma`` (px of the FFT)."""
    from scipy.ndimage import gaussian_filter

    opts.check()
    k = _kemstem()
    ft = fourier_pattern(image, opts.edge)
    pattern = np.abs(ft) if opts.pattern == "abs" else np.log1p(np.abs(ft))
    if opts.blur > 0:
        pattern = gaussian_filter(pattern, opts.blur)
    shape = pattern.shape
    p0 = to_pixels(freqs, shape)
    if not len(p0):
        return []
    half = opts.window // 2
    inside = (p0[:, 0] >= half + 1) & (p0[:, 0] < shape[0] - half - 1) & (p0[:, 1] >= half + 1) & (p0[:, 1] < shape[1] - half - 1)
    out = [{"fy": float(f[0]), "fx": float(f[1]), "failed": True} for f in to_freqs(p0, shape)]
    if not inside.any():
        return out
    if opts.method == "com":
        refined = k.refine_peaks_com(pattern, p0[inside], half)
        failed = ~np.isfinite(refined).all(axis=1)
        params = None
    else:
        refined, failed, params, _ = k.refine_peaks_gf(pattern, p0[inside], window_dimension=opts.window,
                                                       remove_unfit=False, verbose=False)
    for j, i in enumerate(np.flatnonzero(inside)):
        if failed[j]:
            continue
        fy, fx = to_freqs(refined[j], shape)[0]
        peak = {"fy": float(fy), "fx": float(fx), "failed": False}
        if params is not None:
            amp, _, _, s1, s2, _, offset = params[j]
            peak.update(amplitude=float(amp), background=float(offset), sigma=[float(s1), float(s2)])
        out[i] = peak
    return out


# ---------- M22b: Fourier filtering, phase and strain ----------

PhaseMethod = Literal["lockin", "gpa"]
# What can be made: per peak (amplitude, grating, phase, strain from one
# peak) and from two peaks (the strain tensor).
PER_PEAK = ("amplitude", "grating", "phase", "strain")


@dataclass(frozen=True)
class FourierOptions:
    # Gaussian filter round each peak (σ, FFT px).
    sigma: float = 20.0
    phase: PhaseMethod = "lockin"
    # The lock-in's low-pass (σ, FFT px); None: twice the filter's.
    lock_sigma: float | None = None
    edge: Edge = "none"
    # Strain values beyond this (|ε|) are replaced by their median-filtered
    # surroundings (kemstem's mask_threshold); None: kept.
    mask_threshold: float | None = None

    def check(self) -> "FourierOptions":
        if not self.sigma > 0:
            raise ValueError("The filter's σ must be more than 0")
        if self.lock_sigma is not None and not self.lock_sigma > 0:
            raise ValueError("The lock-in σ must be more than 0")
        if self.mask_threshold is not None and not self.mask_threshold > 0:
            raise ValueError("The outlier limit must be more than 0")
        return self


def coarsening(shape: tuple[int, int], sigma: float) -> float:
    """Real-space coarsening radius (image px) of a Gaussian of σ FFT px
    (kemstem's ``coarsening_length``, with the image's first axis as it uses)."""
    return float(_kemstem().coarsening_length(shape[0], sigma))


def _array_pixel(f, shape):
    """Frequency → pixel of the centred FFT array (zero at N // 2): where the
    filter goes."""
    return to_pixels([f], shape)[0]


def _half_pixel(f, shape):
    """Frequency → kemstem's pixel convention for references and strain
    (zero at N / 2): ``(f·H + H/2, f·W + W/2)``. With these, the lock-in
    reference, the GPA carrier and the strain scale are the peak's true
    frequency also for odd sizes."""
    h, w = shape
    return np.array([f[0] * h + h / 2, f[1] * w + w / 2])


def fourier_maps(image: np.ndarray, peaks: dict[int, tuple[float, float]], outputs: set[str],
                 opts: FourierOptions, per_peak: list[int] | None = None,
                 tensor: tuple[int, int] | None = None, report=None) -> list[dict]:
    """kemstem's Fourier filtering of the normalised image round each peak
    (``fourier_filter``), and from it: the amplitude |filtered|, the grating
    Re(filtered), the phase (``phaselock`` or ``gpa``, wrapped to 0–2π),
    ε∥ and ε⊥ (``phase_to_strain``), and, for the two peaks ``tensor``, εxx,
    εyy, εxy and ωxy (``calculate_strain_tensor``). ``peaks``: number (1-based,
    as listed) → frequency, for every peak used; ``per_peak``: the peaks
    ``outputs`` are made for (default all). Returns dicts ``output``,
    ``peaks`` (numbers), ``title`` and ``data`` (float32)."""
    opts.check()
    per_peak = list(peaks) if per_peak is None else per_peak
    if tensor is not None:
        g = np.array([peaks[tensor[0]], peaks[tensor[1]]])
        if tensor[0] == tensor[1] or abs(np.linalg.det(g)) < 1e-12:
            raise ValueError("The strain tensor needs two peaks in different directions")
    k = _kemstem()
    ft = fourier_pattern(image, opts.edge)
    shape = ft.shape
    lock = opts.lock_sigma or 2 * opts.sigma
    need_phase = bool({"phase", "strain"} & outputs)
    numbers = sorted(set(per_peak) | set(tensor or ()))
    phases: dict[int, np.ndarray] = {}
    out: list[dict] = []
    steps = len(numbers) + (1 if tensor else 0)

    def add(output, nums, title, data):
        out.append({"output": output, "peaks": list(nums), "title": title,
                    "data": np.ascontiguousarray(data, dtype=np.float32)})

    for i, n in enumerate(numbers):
        if report:
            report(i / steps, f"Peak {n}")
        f = peaks[n]
        filtered, _, _ = k.fourier_filter(ft, _array_pixel(f, shape), opts.sigma)
        mine = n in per_peak
        if "amplitude" in outputs and mine:
            add("amplitude", [n], f"Amplitude g{n}", np.abs(filtered))
        if "grating" in outputs and mine:
            add("grating", [n], f"Grating g{n}", np.real(filtered))
        if (need_phase and mine) or (tensor and n in tensor):
            ph = (k.phaselock(filtered, _half_pixel(f, shape), lock) if opts.phase == "lockin"
                  else np.mod(k.gpa(filtered, _half_pixel(f, shape)), 2 * np.pi))
            phases[n] = ph
            if "phase" in outputs and mine:
                add("phase", [n], f"Phase g{n}", ph)
            if "strain" in outputs and mine:
                par, perp = k.phase_to_strain(ph, _half_pixel(f, shape), opts.mask_threshold)
                add("strain-par", [n], f"ε∥ g{n}", par)
                add("strain-perp", [n], f"ε⊥ g{n}", perp)
    if tensor:
        if report:
            report(len(numbers) / steps, "Strain tensor")
        a, b = tensor
        exx, eyy, exy, oxy = k.calculate_strain_tensor(
            phases[a], phases[b], _half_pixel(peaks[a], shape), _half_pixel(peaks[b], shape), opts.mask_threshold)
        label = f"g{a}, g{b}"
        for output, title, data in (("exx", "εxx", exx), ("eyy", "εyy", eyy), ("exy", "εxy", exy), ("oxy", "ωxy", oxy)):
            add(output, [a, b], f"{title} ({label})", data)
    return out


# ---------- M22d: atomic columns ----------

# The columns' table, one row per column (kemstem's (y, x) pixels, centres at
# integers; the 2D Gaussian's parameters; intensity = amplitude + offset).
COLUMN_FIELDS = ("y", "x", "amplitude", "sigma_1", "sigma_2", "angle", "offset", "intensity", "failed")
# Columns refined per step (progress and Cancel come between steps).
REFINE_CHUNK = 256


@dataclass(frozen=True)
class ColumnOptions:
    # Gaussian blur (σ, px) of the normalised image before finding and fitting.
    blur: float = 0.5
    # Least distance between columns (px; peak_local_max's min_distance).
    distance: int = 4
    # Intensity threshold (0–1 of the normalised image).
    threshold: float = 0.1
    # Fit window round each column (odd, px).
    window: int = 7
    # Finding only: a broad Gaussian (σ, px) subtracted first, a high-pass
    # that leaves the brightest columns (the notebooks' σ 3 − σ 75); 0: none.
    background: float = 0.0
    # How columns are found: "maxima", kemstem's local maxima above
    # ``threshold`` (0–1 of the image); or "template", local maxima of the
    # normalised cross-correlation with a Gaussian of σ ``template`` px
    # (scikit-image's ``match_template``, as in ptycho-quant-analysis),
    # ``threshold`` then being a correlation (−1 to 1).
    method: str = "maxima"
    template: float = 1.5

    def check(self) -> "ColumnOptions":
        if self.blur < 0:
            raise ValueError("The blur must be 0 or more")
        if self.background < 0:
            raise ValueError("The background σ must be 0 or more")
        if self.distance < 1:
            raise ValueError("The distance must be 1 px or more")
        if self.method not in ("maxima", "template"):
            raise ValueError(f"Unknown way of finding columns: {self.method}")
        if self.method == "template":
            if not -1 < self.threshold < 1:
                raise ValueError("The correlation threshold must be between −1 and 1")
            if not 0.3 <= self.template <= 50:
                raise ValueError("The template σ must be from 0.3 to 50 px")
        elif not 0 <= self.threshold < 1:
            raise ValueError("The threshold must be from 0 to 1")
        if self.window < 3 or self.window % 2 == 0:
            raise ValueError("The fit window must be an odd number of pixels, 3 or more")
        return self


def prepared(image: np.ndarray, blur: float) -> np.ndarray:
    """The image as the notebooks fit it: normalised to 0–1, blurred by σ."""
    from scipy.ndimage import gaussian_filter

    im = _kemstem().normalize(np.asarray(image, dtype=np.float64))
    return gaussian_filter(im, blur) if blur > 0 else im


def find_columns(image: np.ndarray, opts: ColumnOptions) -> np.ndarray:
    """The columns of the prepared image (less a broad background, when
    ``opts.background``): kemstem's ``find_columns``, or the maxima of the
    correlation with a Gaussian template (``opts.method``). (n, 2) (y, x)
    pixels."""
    opts.check()
    k = _kemstem()
    im = prepared(image, opts.blur)
    if opts.background > 0:
        from scipy.ndimage import gaussian_filter

        im = k.normalize(im - gaussian_filter(k.normalize(np.asarray(image, dtype=np.float64)), opts.background))
    if opts.method == "template":
        return template_peaks(im, opts.template, opts.threshold, opts.distance)
    return k.find_columns(im, distance=opts.distance, threshold=opts.threshold).astype(np.float64)


def gaussian_template(sigma: float) -> np.ndarray:
    """A Gaussian of σ px on an odd square, about 4σ wide, centred on its middle pixel."""
    w = 2 * int(np.ceil(2 * sigma)) + 1
    c = (w - 1) / 2
    yy, xx = np.indices((w, w))
    return np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * sigma ** 2))


def template_correlation(image: np.ndarray, sigma: float) -> np.ndarray:
    """The normalised cross-correlation (−1 to 1) of the image with a
    Gaussian of σ px at every pixel (scikit-image's ``match_template``)."""
    from skimage.feature import match_template

    return match_template(np.asarray(image, dtype=np.float64), gaussian_template(sigma), pad_input=True, mode="edge")


def template_peaks(image: np.ndarray, sigma: float, threshold: float, distance: int) -> np.ndarray:
    """Columns as local maxima of the correlation with a Gaussian of σ px,
    at least ``distance`` px apart and correlating above ``threshold``
    (ptycho-quant-analysis's ``find_peaks``): (n, 2) (y, x) pixels."""
    from skimage.feature import peak_local_max

    cc = template_correlation(image, sigma)
    return peak_local_max(cc, min_distance=distance, threshold_abs=threshold, exclude_border=2).astype(np.float64)


def _rows(positions, errors, params) -> np.ndarray:
    n = len(positions)
    out = np.zeros((n, len(COLUMN_FIELDS)), dtype=np.float64)
    out[:, 0:2] = positions
    # kemstem's parameters: amplitude, x0, y0, σ1, σ2, angle, offset.
    out[:, 2] = params[:, 0]
    out[:, 3:6] = params[:, 3:6]
    out[:, 6] = params[:, 6]
    out[:, 7] = params[:, 0] + params[:, 6]
    out[:, 8] = errors
    out[errors.astype(bool), 2:8] = np.nan
    return out


def refine_columns(image: np.ndarray, columns: np.ndarray, opts: ColumnOptions, report=None) -> np.ndarray:
    """kemstem's ``refine_columns`` (a 2D Gaussian fit to each) on the
    prepared image, in steps of REFINE_CHUNK columns. Returns the table
    (COLUMN_FIELDS) in the columns' order; a failed fit keeps its column's
    position and has NaN parameters."""
    opts.check()
    k = _kemstem()
    im = prepared(image, opts.blur)
    columns = np.asarray(columns, dtype=np.float64).reshape(-1, 2)
    parts = []
    for i in range(0, len(columns), REFINE_CHUNK):
        if report:
            report(i / max(1, len(columns)))
        chunk = columns[i:i + REFINE_CHUNK]
        pos, errs, params, _ = k.refine_columns(im, chunk, window_dimension=opts.window, remove_unfit=False,
                                                verbose=False)
        parts.append(_rows(pos, errs, params))
    return np.concatenate(parts) if parts else np.zeros((0, len(COLUMN_FIELDS)))


def test_column(image: np.ndarray, at: tuple[float, float], opts: ColumnOptions) -> dict:
    """One column's fit, to choose the window: the data patch, the fit and
    their row of the table (``row``, by COLUMN_FIELDS)."""
    opts.check()
    k = _kemstem()
    im = prepared(image, opts.blur)
    pos, errs, params, fits = k.refine_columns(im, np.array([at], dtype=np.float64), window_dimension=opts.window,
                                               remove_unfit=False, verbose=False)
    row = _rows(pos, errs, params)[0]
    return {
        "data": fits[:, :, 0, 0].tolist(), "fit": fits[:, :, 0, 1].tolist(),
        "row": [None if not np.isfinite(v) else float(v) for v in row],
    }


# ---------- Sublattices ----------

#: Default names of sublattices, the brightest (largest value) first.
SUBLATTICE_NAMES = ("A", "B", "C", "D", "E", "F")


def kmeans_centres(values: np.ndarray, k: int) -> list[float]:
    """Centres of ``k`` clusters of a column value (scikit-learn's k-means on
    the finite values), largest first: thresholds go half-way between them."""
    from sklearn.cluster import KMeans

    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)].reshape(-1, 1)
    if not 2 <= k <= len(SUBLATTICE_NAMES):
        raise ValueError(f"Between 2 and {len(SUBLATTICE_NAMES)} sublattices")
    if len(v) < k:
        raise ValueError("Fewer columns than sublattices")
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(v)
    return sorted((float(c) for c in km.cluster_centers_.ravel()), reverse=True)


# ---------- Neighbourhoods and bonds (M22e) ----------

#: Fields of a bonds table: the origin column's row in its Columns table, the
#: vector to its neighbour (px, y down), its length (px) and angle (° from +x,
#: y down as in the image, as kemstem's notebook).
BOND_FIELDS = ("origin", "dy", "dx", "length", "angle")


def neighbourhood(neighbours: np.ndarray, origins: np.ndarray, k: int) -> np.ndarray:
    """The vectors from each origin to its ``k`` nearest neighbours (kemstem's
    ``get_neighborhood``, [y, x] px): (k × m, 2), neighbour j of origin i at
    row j·m + i."""
    if not 1 <= k <= 200:
        raise ValueError("Between 1 and 200 neighbours")
    if len(neighbours) < k:
        raise ValueError(f"Fewer than {k} neighbours")
    if not len(origins):
        raise ValueError("No origins")
    return np.asarray(_kemstem().get_neighborhood(neighbours, origins, k=k), dtype=np.float64)


def cluster_neighbourhood(vectors: np.ndarray, n: int) -> np.ndarray:
    """``n`` cluster centres of a neighbourhood (kemstem's ``cluster_neighbors``,
    k-means), [y, x] px."""
    if not 2 <= n <= 200:
        raise ValueError("Between 2 and 200 clusters")
    if len(vectors) < n:
        raise ValueError("Fewer vectors than clusters")
    return np.asarray(_kemstem().cluster_neighbors(vectors, n_clusters=n), dtype=np.float64)


def bonds(neighbours: np.ndarray, origins: np.ndarray, rows: np.ndarray, guess: tuple[float, float],
          threshold: float) -> np.ndarray:
    """From each origin, the vector to the neighbour nearest ``origin + guess``
    (kemstem's ``get_vector_to_neighbors``; NaN when none is within
    ``threshold`` px), as a table of BOND_FIELDS; ``rows`` are the origins'
    rows in their Columns table."""
    if threshold <= 0:
        raise ValueError("The threshold must be more than 0")
    v = np.asarray(_kemstem().get_vector_to_neighbors(neighbours, origins, np.asarray(guess, dtype=np.float64),
                                                    threshold=threshold), dtype=np.float64)
    length = np.hypot(v[:, 0], v[:, 1])
    angle = np.degrees(np.arctan2(v[:, 0], v[:, 1]))
    return np.column_stack([rows.astype(np.float64), v, length, angle])


def cage_vectors(neighbours: np.ndarray, origins: np.ndarray, guesses: np.ndarray, threshold: float) -> np.ndarray:
    """From each origin, the mean of the vectors to the neighbours nearest
    ``origin + guess`` for each guess (kemstem's ``get_vector_to_neighbors``):
    the centre of a cage of neighbours (a guess of (0, 0) is the origin
    itself, when it is among the neighbours). NaN where any is not within
    ``threshold`` px. (m, 2) [dy, dx] px."""
    if threshold <= 0:
        raise ValueError("The threshold must be more than 0")
    guesses = np.asarray(guesses, dtype=np.float64).reshape(-1, 2)
    if not len(guesses):
        raise ValueError("Pick the cage's clusters first")
    k = _kemstem()
    v = np.array([k.get_vector_to_neighbors(neighbours, origins, g, threshold=threshold) for g in guesses],
                 dtype=np.float64)
    return v.mean(axis=0)  # NaN where any is missing


def _basis(vectors) -> np.ndarray:
    """The cell's two vectors ([dy, dx] px each) as the columns of a 2 × 2 matrix."""
    b = np.asarray(vectors, dtype=np.float64).reshape(2, 2).T
    if abs(np.linalg.det(b)) < 1e-6:
        raise ValueError("The two lattice vectors must point different ways")
    return b


def cell_coordinates(positions: np.ndarray, reference: np.ndarray, vectors) -> np.ndarray:
    """Each position's place in the cell of the reference columns' lattice
    (``vectors``: two [dy, dx] px): its offset from the nearest reference
    column in fractions of the vectors, wrapped to [−½, ½). NaN where a
    position is. (n, 2) (u, v), u along the first vector."""
    from scipy.spatial import cKDTree

    b = _basis(vectors)
    ref = np.asarray(reference, dtype=np.float64)
    ref = ref[np.isfinite(ref).all(axis=1)]
    if not len(ref):
        raise ValueError("The reference columns have no positions")
    pos = np.asarray(positions, dtype=np.float64)
    out = np.full((len(pos), 2), np.nan)
    ok = np.isfinite(pos).all(axis=1)
    _, j = cKDTree(ref).query(pos[ok])
    f = np.linalg.solve(b, (pos[ok] - ref[j]).T).T
    out[ok] = f - np.round(f)
    return out


def assign_sites(cell: np.ndarray, sites, vectors, radius: float) -> list[int | None]:
    """Each column's site: the nearest of ``sites`` ((u, v) in the cell) by
    distance in px, the cell repeating; None when none is within ``radius``
    px (or the column has no place)."""
    if radius <= 0:
        raise ValueError("The radius must be more than 0")
    sites = np.asarray(sites, dtype=np.float64).reshape(-1, 2)
    if not len(sites):
        raise ValueError("Pick the sites first")
    b = _basis(vectors)
    d = cell[:, None, :] - sites[None, :, :]
    d -= np.round(d)
    dist = np.hypot(*np.einsum("ij,nkj->ink", b, d))  # (n, k) px
    best = np.nanargmin(np.where(np.isfinite(dist), dist, np.inf), axis=1)
    near = dist[np.arange(len(dist)), best]
    return [int(k) if np.isfinite(r) and r <= radius else None for k, r in zip(best, near)]


def nearest_cage(vectors: np.ndarray, k: int, threshold: float) -> np.ndarray:
    """The centre of each origin's cage of its ``k`` nearest neighbours: the
    mean of a neighbourhood's vectors (``neighbourhood``'s rows, neighbour j
    of origin i at row j·m + i). NaN where the cage is not the usual shape:
    its neighbours, about their own centre and in order round it, more than
    ``threshold`` px from the usual cage's (the median over the origins), as
    when one is missing at the image's edge, or the origin outside it (a
    whole cage beside it). The shape doesn't depend on where the origin is
    in its cage, so displaced origins are kept. (m, 2) [dy, dx] px."""
    if threshold <= 0:
        raise ValueError("The threshold must be more than 0")
    v = np.asarray(vectors, dtype=np.float64).reshape(k, -1, 2).transpose(1, 0, 2)  # (m, k, 2)
    centre = v.mean(axis=1)
    rel = v - centre[:, None, :]
    order = np.argsort(np.arctan2(rel[..., 0], rel[..., 1]), axis=1)
    shape = np.take_along_axis(rel, order[..., None], axis=1)
    usual = np.nanmedian(shape, axis=0)
    off = np.hypot(*(shape - usual).transpose(2, 0, 1)).max(axis=1)
    # And the origin inside its cage (the polygon of its neighbours, in order):
    # not a whole cage beside it, as at the edge.
    o = -centre[:, None, :]  # the origin, about the cage's centre
    a, b = shape, np.roll(shape, -1, axis=1)
    cross = (b[..., 1] - a[..., 1]) * (o[..., 0] - a[..., 0]) - (b[..., 0] - a[..., 0]) * (o[..., 1] - a[..., 1])
    inside = (cross >= 0).all(axis=1) | (cross <= 0).all(axis=1)
    out = centre.copy()
    out[~((off <= threshold) & inside)] = np.nan
    return out


def cluster_sites(cell: np.ndarray, sites, vectors, link: float = 4.0, least: int = 5,
                  reach: float = 8.0) -> tuple[list[int | None], list[dict], list[dict]]:
    """Each picked site's cluster in the cell: the dense, connected group of
    columns it sits in (scikit-learn's DBSCAN on the columns' places in px:
    columns within ``link`` px of each other, at least ``least`` round a
    core), whatever its shape (e.g. a long tilt streak), the cell repeating
    (the cell tiled 3 × 3, so a group across its edges stays whole). A site
    takes every group with a column within ``reach`` px of it (both lobes of
    a site displaced up and down); columns of a group two sites share go to
    the nearer; groups no site takes, and columns in no group, are left out. Returns the labels (an index into
    ``sites``, or None), per site its ``u``, ``v`` (the members' mean place)
    and ``count`` (``empty`` when no group was near it), and the groups no
    site took (``u``, ``v``, ``count``), largest first."""
    from sklearn.cluster import DBSCAN

    if link <= 0 or least < 1 or reach <= 0:
        raise ValueError("The link distance and smallest group must be more than 0")
    sites = np.asarray(sites, dtype=np.float64).reshape(-1, 2)
    if not len(sites):
        raise ValueError("Pick the sites first")
    b = _basis(vectors)
    cell = np.asarray(cell, dtype=np.float64)
    ok = np.flatnonzero(np.isfinite(cell).all(axis=1))
    f = cell[ok]
    n = len(f)
    labels: list[int | None] = [None] * len(cell)
    if not n:
        return labels, [{"u": float(u), "v": float(v), "count": 0, "empty": True} for u, v in sites], []
    tiles = np.array([(a, c) for a in (-1, 0, 1) for c in (-1, 0, 1)], dtype=np.float64)
    px = np.concatenate([(f + t) @ b.T for t in tiles])  # 9 n, [dy, dx] px; tile 4 is the cell itself
    group = DBSCAN(eps=link, min_samples=least).fit_predict(px).reshape(9, n)
    # A column's groups: those of any of its copies (a group across an edge is one).
    def members(g):
        return (group == g).any(axis=0)

    claimed: dict[int, list[int]] = {}
    flat = group.ravel()
    for k, site in enumerate(sites):
        d = np.hypot(*(px - b @ site).T)
        for g in set(flat[(d <= reach) & (flat >= 0)].tolist()):
            claimed.setdefault(int(g), []).append(k)
    site_of = np.full(n, -1)
    for g, ks in claimed.items():
        m = members(g)
        if len(ks) == 1:
            site_of[m] = ks[0]
            continue
        # Shared: each column to the nearer site (the cell repeating).
        dd = []
        for k in ks:
            d = f[m] - sites[k]
            d -= np.round(d)
            dd.append(np.hypot(*(d @ b.T).T))
        site_of[m] = np.array(ks)[np.argmin(np.array(dd), axis=0)]
    for i, k in zip(ok, site_of):
        labels[i] = None if k < 0 else int(k)

    def mean_place(rows):
        # Mean of places about the first, the cell repeating.
        p = f[rows]
        d = p - p[0]
        d -= np.round(d)
        c = p[0] + d.mean(axis=0)
        return c - np.round(c)

    fits = []
    for k, site in enumerate(sites):
        rows = np.flatnonzero(site_of == k)
        c = mean_place(rows) if len(rows) else site
        fits.append({"u": float(c[0]), "v": float(c[1]), "count": int(len(rows)), "empty": not len(rows)})
    unpicked = []
    for g in sorted(set(group[4][group[4] >= 0]) - set(claimed)):
        rows = np.flatnonzero(members(g) & (site_of < 0))
        if len(rows):
            c = mean_place(rows)
            unpicked.append({"u": float(c[0]), "v": float(c[1]), "count": int(len(rows))})
    # A group across an edge appears from several tiles' copies only once per column set.
    seen, unique = set(), []
    for x in sorted(unpicked, key=lambda x: -x["count"]):
        key = (round(x["u"], 2), round(x["v"], 2), x["count"])
        if key not in seen:
            seen.add(key)
            unique.append(x)
    return labels, fits, unique


# ---------- M22f: periodic lattice displacements ----------

def detect_fft_peaks(image: np.ndarray, count: int = 120, within: float = 0.25, min_distance: int = 3,
                     edge: Edge = "none", noise: float = 4.0) -> list[dict]:
    """The strongest local maxima of the image's FFT (log |F|, blurred by 1
    px, less its broad background: a Gaussian of 8 px), the centre left out:
    up to ``count`` within ``within`` cycles/px of it, standing out by more
    than ``noise`` × the background's robust spread. Each: ``fy``, ``fx``
    (cycles/px) and ``height`` (above the background, log units), strongest
    first. Found with scikit-image's ``peak_local_max``."""
    from scipy.ndimage import gaussian_filter
    from skimage.feature import peak_local_max

    if not 1 <= count <= 500:
        raise ValueError("Between 1 and 500 peaks")
    if not 0 < within <= 0.5:
        raise ValueError("Within must be between 0 and 0.5 cycles/px")
    pattern = np.log1p(np.abs(fourier_pattern(image, edge)))
    prominence = gaussian_filter(pattern, 1.0) - gaussian_filter(pattern, 8.0)
    spread = 1.4826 * np.median(np.abs(prominence - np.median(prominence)))
    h, w = pattern.shape
    yy, xx = np.mgrid[0:h, 0:w]
    f = np.hypot((yy - h // 2) / h, (xx - w // 2) / w)
    floor = float(prominence.min())
    prominence[(f > within) | (f < 0.01)] = floor
    found = peak_local_max(prominence, min_distance=min_distance, num_peaks=count, exclude_border=False,
                           threshold_abs=floor + 1e-9 if noise <= 0 else noise * spread)
    out = []
    for (y, x), (fy, fx) in zip(found, to_freqs(np.array(found, dtype=np.float64).reshape(-1, 2), (h, w))):
        out.append({"fy": float(fy), "fx": float(fx), "height": float(prominence[y, x])})
    return out


def lattice_basis(peaks: list[dict], strongest: int = 8, tol: float = 0.12) -> list[list[float]] | None:
    """Two reciprocal lattice vectors (fy, fx) for detected peaks: the
    lattice the ``strongest`` peaks generate, i.e. the pair (from them and
    their differences) of smallest cell that gives every one of them whole
    indices. None without two independent peaks."""
    strong = sorted(peaks, key=lambda p: -p["height"])[:strongest]
    pts = np.array([[p["fy"], p["fx"]] for p in strong], dtype=np.float64)
    if len(pts) < 2:
        return None
    cands = [v for v in pts] + [a - b for i, a in enumerate(pts) for b in pts[i + 1:]]
    cands = [v for v in cands if np.hypot(*v) > 1e-6]
    best, best_det = None, np.inf
    for i, u in enumerate(cands):
        for v in cands[i + 1:]:
            det = abs(u[0] * v[1] - u[1] * v[0])
            if det < 1e-9 or det >= best_det:
                continue
            hk = np.linalg.solve(np.column_stack([u, v]), pts.T)
            if np.all(np.abs(hk - np.round(hk)) < tol):
                best, best_det = (u, v), det
    if best is None:
        return None
    # The shortest pair spanning the same lattice (Lagrange–Gauss reduction).
    u, v = (np.array(x) for x in best)
    for _ in range(20):
        if np.dot(u, u) > np.dot(v, v):
            u, v = v, u
        m = round(np.dot(u, v) / np.dot(u, u))
        if m == 0:
            break
        v = v - m * u
    return [u.tolist(), v.tolist()]


def label_peaks(peaks: list[dict], basis: list[list[float]] | None, tol: float = 0.12) -> list[dict]:
    """Each peak's (h, k) in ``basis`` and its label: "bragg" when both are
    near whole numbers (within ``tol``), else "superlattice" (all
    superlattice without a basis)."""
    out = []
    m = np.array(basis, dtype=np.float64).T if basis else None  # columns: the two vectors
    for p in peaks:
        q = dict(p)
        if m is not None and abs(np.linalg.det(m)) > 1e-12:
            h, k = np.linalg.solve(m, [p["fy"], p["fx"]])
            q["hk"] = [round(float(h), 3), round(float(k), 3)]
            whole = abs(h - round(h)) < tol and abs(k - round(k)) < tol
            q["label"] = "bragg" if whole else "superlattice"
        else:
            q["label"] = "superlattice"
        out.append(q)
    return out


def peak_families(peaks: list[dict], tol: float = 0.06, least: int = 4) -> tuple[list[dict], list[dict]]:
    """Groups the off-lattice peaks (labelled by ``label_peaks``) by their
    offset from the lattice: peaks at G ± q (whole G) share a family q.
    Families of ``least`` peaks or more are numbered by strength (summed
    height) from 1; each peak's ``family`` is that number, 0 for Bragg peaks
    and None for the rest. Returns the peaks and the families (``id``, ``q``
    as (h, k) fractions, ``count``, ``strength``)."""
    def wrap(v):
        return v - np.round(v)

    groups: list[list] = []  # [q, members]
    order = sorted(range(len(peaks)), key=lambda i: -peaks[i]["height"])
    for i in order:
        p = peaks[i]
        if p.get("label") != "superlattice" or "hk" not in p:
            continue
        fr = wrap(np.array(p["hk"], dtype=np.float64))
        for g in groups:
            if np.all(np.abs(wrap(fr - g[0])) < tol) or np.all(np.abs(wrap(fr + g[0])) < tol):
                g[1].append(i)
                break
        else:
            groups.append([fr, [i]])
    kept = [g for g in groups if len(g[1]) >= least]
    kept.sort(key=lambda g: -sum(peaks[i]["height"] for i in g[1]))
    out = [{**p, "family": 0 if p.get("label") == "bragg" else None} for p in peaks]
    families = []
    for n, (q, members) in enumerate(kept, start=1):
        for i in members:
            out[i]["family"] = n
        families.append({"id": n, "q": [round(float(q[0]), 3), round(float(q[1]), 3)], "count": len(members),
                         "strength": round(float(sum(peaks[i]["height"] for i in members)), 2)})
    return out, families


def damp_peaks(image: np.ndarray, freqs: np.ndarray, radius: float, window: int = 9, blur: float = 1.0,
               edge: Edge = "none") -> dict:
    """The reference image without the superlattice peaks (kemstem's PLD
    method): the peaks refined with 2D Gaussians on the blurred |F| (for
    their positions and backgrounds), then set to their background within
    ``radius`` FFT px (``damp_peaks_circle``). Returns ``reference`` (in the
    normalised image's units), ``difference`` (reference − normalised image)
    and the refined ``peaks`` (fy, fx, background, failed)."""
    if not radius > 0:
        raise ValueError("The radius must be more than 0")
    freqs = np.asarray(freqs, dtype=np.float64).reshape(-1, 2)
    if not len(freqs):
        raise ValueError("No superlattice peaks")
    k = _kemstem()
    refined = refine_peaks(image, freqs, PeakOptions("gauss", window, blur, edge, "abs"))
    ok = [p for p in refined if not p["failed"]]
    if not ok:
        raise ValueError("None of the peaks could be fitted")
    ft = fourier_pattern(image, edge)
    shape = ft.shape
    pos = to_pixels(np.array([[p["fy"], p["fx"]] for p in ok]), shape)
    levels = np.array([p["background"] for p in ok])
    _, reference = k.damp_peaks_circle(ft, pos, levels, radius)
    normalised = k.normalize(np.asarray(image, dtype=np.float64))
    return {"reference": reference, "difference": reference - normalised, "peaks": refined}


def measure_displacements(reference: np.ndarray, distorted: np.ndarray, threshold: float) -> np.ndarray:
    """kemstem's ``measure_displacements``: from each reference column, the
    vector to its nearest distorted one (NaN beyond ``threshold`` px), as a
    table of DISPLACEMENT_FIELDS."""
    if not threshold > 0:
        raise ValueError("The threshold must be more than 0")
    ref = np.asarray(reference, dtype=np.float64)
    ok = np.isfinite(ref).all(axis=1)
    d = np.full((len(ref), 2), np.nan)
    if ok.any():
        dist = np.asarray(distorted, dtype=np.float64)
        dist = dist[np.isfinite(dist).all(axis=1)]
        d[ok] = _kemstem().measure_displacements(ref[ok], dist, threshold=threshold)
    mag = np.hypot(d[:, 0], d[:, 1])
    angle = np.degrees(np.arctan2(d[:, 0], d[:, 1]))
    return np.column_stack([d, mag, angle])


#: Fields of a displacements table: the vector (px, y down), its magnitude
#: (px) and angle (° from +x, y down); one row per column of the Columns.
DISPLACEMENT_FIELDS = ("dy", "dx", "magnitude", "angle")
