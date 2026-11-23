"""Atomic-resolution images with kemstem (M22): FFT peaks."""

from __future__ import annotations

from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..processing import atomic
from ..processing.filters import filter_image
from ..session import DatasetKind

router = APIRouter(tags=["atomic"])


def image_of(request: Request, dataset_id: str) -> np.ndarray:
    """The image as shown (its filters applied); 404/400 for anything else."""
    session = request.app.state.session
    try:
        ds = session.get_dataset(dataset_id)
    except KeyError:
        raise HTTPException(404, f"Unknown dataset id: {dataset_id}")
    if ds.kind is not DatasetKind.IMAGE:
        raise HTTPException(400, "Atomic analysis needs an image")
    ds.materialise()
    return filter_image(np.asarray(ds.signal.data, dtype=np.float64), ds.filter_spec)


class PeaksBody(BaseModel):
    # Spatial frequencies (fy, fx) in cycles per image pixel.
    peaks: list[tuple[float, float]] = Field(max_length=500)
    method: Literal["gauss", "com"] = "gauss"
    window: int = 9
    blur: float = 1.0
    edge: Literal["none", "hann", "pplus"] = "none"
    pattern: Literal["log", "abs"] = "log"


@router.post("/api/datasets/{dataset_id}/atomic/fft-peaks")
def refine_fft_peaks(dataset_id: str, body: PeaksBody, request: Request) -> dict:
    """Refine FFT peaks of the image: ``peaks`` in order, each ``fy``, ``fx``,
    ``failed`` and, for Gaussian fits, ``amplitude``, ``background`` and
    ``sigma``; ``shape``, the FFT's (rows, columns)."""
    image = image_of(request, dataset_id)
    opts = atomic.PeakOptions(body.method, body.window, body.blur, body.edge, body.pattern)
    try:
        opts.check()
    except ValueError as e:
        raise HTTPException(422, str(e))
    peaks = atomic.refine_peaks(image, np.array(body.peaks, dtype=np.float64).reshape(-1, 2), opts)
    return {"peaks": peaks, "shape": list(image.shape)}


class FourierBody(BaseModel):
    # Every peak used: its number as listed (1-based) → frequency (fy, fx).
    peaks: dict[int, tuple[float, float]] = Field(max_length=50)
    # Per peak (those in ``per_peak``): amplitude, grating, phase, strain (ε∥, ε⊥).
    per_peak: list[int] = []
    outputs: list[Literal["amplitude", "grating", "phase", "strain"]] = []
    # Two peaks: the strain tensor from them.
    tensor: tuple[int, int] | None = None
    sigma: float = 20.0
    phase: Literal["lockin", "gpa"] = "lockin"
    lock_sigma: float | None = None
    edge: Literal["none", "hann", "pplus"] = "none"
    mask_threshold: float | None = None


@router.post("/api/datasets/{dataset_id}/atomic/fourier")
def fourier(dataset_id: str, body: FourierBody, request: Request) -> dict:
    """Fourier filtering round FFT peaks and what follows (kemstem), as a
    job: the maps (amplitude, grating, phase, ε∥ and ε⊥ per peak; the strain
    tensor from two) under the image, replacing ones made before from the
    same peaks. The job's result: ``file`` (the image's file) and ``ids``."""
    session = request.app.state.session
    image = image_of(request, dataset_id)
    ds = session.get_dataset(dataset_id)
    opts = atomic.FourierOptions(body.sigma, body.phase, body.lock_sigma, body.edge, body.mask_threshold)
    used = set(body.per_peak) | set(body.tensor or ())
    try:
        opts.check()
        if not ((body.per_peak and body.outputs) or body.tensor):
            raise ValueError("Choose what to make")
        missing = used - set(body.peaks)
        if missing:
            raise ValueError(f"Unknown peak {min(missing)}")
        if body.tensor:
            a, b = body.tensor
            if a == b or abs(np.linalg.det(np.array([body.peaks[a], body.peaks[b]]))) < 1e-12:
                raise ValueError("The strain tensor needs two peaks in different directions")
    except ValueError as e:
        raise HTTPException(422, str(e))
    peaks = {n: body.peaks[n] for n in used}
    settings = {
        "sigma": opts.sigma, "phase": opts.phase, "lock_sigma": opts.lock_sigma or 2 * opts.sigma,
        "edge": opts.edge, "mask_threshold": opts.mask_threshold,
        "coarsening": atomic.coarsening(image.shape, opts.sigma),
        "freqs": {str(n): list(f) for n, f in peaks.items()},
    }

    def run(job) -> dict:
        made = atomic.fourier_maps(image, peaks, set(body.outputs), opts, body.per_peak, body.tensor,
                                   report=lambda f, msg: job.report(0.95 * f, msg))
        job.report(0.97, "Saving the maps")
        out = session.add_atomic_maps(ds, made, settings)
        return {"file": session.file_of(ds).summary(session), "ids": [d.id for d in out]}

    return request.app.state.jobs.submit("atomic-fourier", f"Fourier filtering of {ds.title}", run).summary()


# ---------- M22d: atomic columns ----------

class ColumnsBody(BaseModel):
    blur: float = 0.5
    distance: int = 4
    threshold: float = 0.1
    window: int = 7
    background: float = 0.0
    method: Literal["maxima", "template"] = "maxima"
    template: float = 1.5

    def options(self) -> atomic.ColumnOptions:
        try:
            return atomic.ColumnOptions(self.blur, self.distance, self.threshold, self.window, self.background,
                                        self.method, self.template).check()
        except ValueError as e:
            raise HTTPException(422, str(e))


@router.post("/api/datasets/{dataset_id}/atomic/columns/find")
def find_columns(dataset_id: str, body: ColumnsBody, request: Request) -> dict:
    """The image's atomic columns found by kemstem (``find_columns``), as
    ``columns``: [y, x] pixels (centres at integers)."""
    image = image_of(request, dataset_id)
    found = atomic.find_columns(image, body.options())
    return {"columns": found.tolist()}


class TestBody(ColumnsBody):
    at: tuple[float, float]


@router.post("/api/datasets/{dataset_id}/atomic/columns/test")
def test_column(dataset_id: str, body: TestBody, request: Request) -> dict:
    """One column fitted (at [y, x]): ``data`` and ``fit`` patches and its
    ``row`` of the table (``fields``)."""
    image = image_of(request, dataset_id)
    return {**atomic.test_column(image, body.at, body.options()), "fields": list(atomic.COLUMN_FIELDS)}


class RefineBody(ColumnsBody):
    columns: list[tuple[float, float]] = Field(max_length=500_000)


@router.post("/api/datasets/{dataset_id}/atomic/columns/refine")
def refine_columns(dataset_id: str, body: RefineBody, request: Request) -> dict:
    """Fit every column with a 2D Gaussian (kemstem's ``refine_columns``), as
    a job; the table goes into the image's Columns dataset (made again in
    place). The job's result: ``file`` and ``id`` (the Columns dataset)."""
    session = request.app.state.session
    image = image_of(request, dataset_id)
    ds = session.get_dataset(dataset_id)
    opts = body.options()
    if not body.columns:
        raise HTTPException(422, "Find the columns first")
    columns = np.array(body.columns, dtype=np.float64)
    settings = {"blur": opts.blur, "distance": opts.distance, "threshold": opts.threshold, "window": opts.window}

    def run(job) -> dict:
        table = atomic.refine_columns(image, columns, opts,
                                      report=lambda f: job.report(0.97 * f, f"Fitting {len(columns)} columns"))
        job.report(0.98, "Saving the columns")
        made = session.set_columns(ds, table, settings)
        return {"file": session.file_of(ds).summary(session), "id": made.id}

    return request.app.state.jobs.submit("atomic-columns", f"Fit the columns of {ds.title}", run).summary()


@router.post("/api/datasets/{dataset_id}/atomic/columns/append")
def append_columns(dataset_id: str, body: RefineBody, request: Request) -> dict:
    """Fit new columns (e.g. sites at cage centres) and add them after the
    Columns dataset ``dataset_id``'s own, which stay fitted as they are: a
    job. The new ones are a sublattice of their own. The job's result:
    ``file``, ``id`` (the Columns) and ``sublattice``."""
    session = request.app.state.session
    cols = _columns_of(request, dataset_id)
    image = image_of(request, cols.parent_id)
    opts = body.options()
    if not body.columns:
        raise HTTPException(422, "No columns to add")
    columns = np.array(body.columns, dtype=np.float64)

    def run(job) -> dict:
        table = atomic.refine_columns(image, columns, opts,
                                      report=lambda f: job.report(0.97 * f, f"Fitting {len(columns)} new columns"))
        job.report(0.98, "Adding the columns")
        sub = session.append_columns(cols, table, {"window": opts.window, "blur": opts.blur})
        return {"file": session.file_of(cols).summary(session), "id": cols.id, "sublattice": sub.id}

    return request.app.state.jobs.submit("atomic-columns", f"Fit {len(columns)} new columns of {cols.title}", run).summary()


@router.get("/api/datasets/{dataset_id}/atomic/columns")
def get_columns(dataset_id: str, request: Request) -> dict:
    """A Columns dataset's table: ``fields`` and ``rows`` (NaN as null)."""
    session = request.app.state.session
    try:
        ds = session.get_dataset(dataset_id)
    except KeyError:
        raise HTTPException(404, f"Unknown dataset id: {dataset_id}")
    if not (ds.atomic and ds.atomic.get("output") == "columns"):
        raise HTTPException(400, f"{ds.title} is not a table of columns")
    table = np.asarray(ds.signal.data, dtype=np.float64)
    fields = list(atomic.COLUMN_FIELDS)
    # The sublattice of each column, from the sublattices under it.
    labels = session.column_labels(ds)
    if labels is not None:
        table = np.column_stack([table, labels])
        fields.append("sublattice")
    rows = np.where(np.isfinite(table), np.round(table, 4), None).tolist()
    return {"fields": fields, "rows": rows}


def _columns_of(request: Request, dataset_id: str):
    session = request.app.state.session
    try:
        ds = session.get_dataset(dataset_id)
    except KeyError:
        raise HTTPException(404, f"Unknown dataset id: {dataset_id}")
    if not (ds.atomic and ds.atomic.get("output") == "columns"):
        raise HTTPException(400, f"{ds.title} is not a table of columns")
    return ds


class SublatticesBody(BaseModel):
    labels: list[int | None] = Field(max_length=500_000)
    names: list[str] = Field(max_length=len(atomic.SUBLATTICE_NAMES))


@router.put("/api/datasets/{dataset_id}/atomic/columns/sublattices")
def set_sublattices(dataset_id: str, body: SublatticesBody, request: Request) -> dict:
    """Split the columns of a Columns dataset into sublattices (each column's
    index into ``names``; null: none), datasets under it; no names removes
    them. Returns its ``file`` and the sublattices' ``ids``."""
    session = request.app.state.session
    ds = _columns_of(request, dataset_id)
    try:
        made = session.set_sublattices(ds, body.labels, body.names)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"file": session.file_of(ds).summary(session), "ids": [d.id for d in made]}


class MaskBody(BaseModel):
    mask: str


@router.post("/api/datasets/{dataset_id}/atomic/columns/sublattices/mask")
def sublattice_from_mask(dataset_id: str, body: MaskBody, request: Request) -> dict:
    """A sublattice of the columns in a mask (an image of the same size: a
    pixel above 0.5), added next to the sublattices there are; columns
    without a fitted position are left out. Returns its ``file`` and ``id``."""
    from ..processing.masks import inside

    session = request.app.state.session
    ds = _columns_of(request, dataset_id)
    try:
        mask = session.get_dataset(body.mask)
    except KeyError:
        raise HTTPException(404, f"Unknown dataset id: {body.mask}")
    image = session.get_dataset(ds.parent_id)
    if mask.kind is not DatasetKind.IMAGE or tuple(mask.signal.data.shape) != tuple(image.signal.data.shape):
        raise HTTPException(422, f"{mask.title} is not an image the size of {image.title}")
    mask.materialise()
    table = np.asarray(ds.signal.data, dtype=np.float64)
    fields = list(atomic.COLUMN_FIELDS)
    pos = table[:, [fields.index("y"), fields.index("x")]]
    rows = np.flatnonzero(inside(np.asarray(mask.signal.data, dtype=np.float64), pos))
    try:
        made = session.add_sublattice(ds, rows, {"mask": mask.id})
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"file": session.file_of(ds).summary(session), "id": made.id}


class CellBody(BaseModel):
    # The columns whose lattice makes the cell (a sublattice or the Columns).
    reference: str
    # Its two lattice vectors, [dy, dx] px.
    vectors: tuple[tuple[float, float], tuple[float, float]]


def _cell_of(request: Request, cols, body: CellBody) -> np.ndarray:
    session = request.app.state.session
    try:
        ref = session.get_dataset(body.reference)
    except KeyError:
        raise HTTPException(404, f"Unknown dataset id: {body.reference}")
    try:
        if session._columns_of(ref).id != cols.id:
            raise ValueError(f"{ref.title} is not of {cols.title}")
        _, r = session.column_positions(ref)
        table = np.asarray(cols.signal.data, dtype=np.float64)
        return atomic.cell_coordinates(table[:, :2], r, body.vectors)
    except ValueError as e:
        raise HTTPException(422, str(e))


@router.post("/api/datasets/{dataset_id}/atomic/columns/cell")
def cell_positions(dataset_id: str, body: CellBody, request: Request) -> dict:
    """Each column's place in the cell of the reference columns' lattice:
    ``u`` and ``v``, fractions of the two vectors from the nearest reference
    column, in [−½, ½) (null without a position)."""
    cols = _columns_of(request, dataset_id)
    f = np.round(_cell_of(request, cols, body), 4)
    clean = lambda a: [None if not np.isfinite(x) else float(x) for x in a]
    return {"u": clean(f[:, 0]), "v": clean(f[:, 1])}


class CellSitesBody(CellBody):
    # The sites, (u, v) in the cell.
    sites: list[tuple[float, float]] = Field(min_length=1, max_length=24)
    # Each site's sublattice (sites with the same name make one, e.g. both
    # spots of a site the cell doesn't repeat onto each other); default A,
    # B, … in order.
    names: list[str] | None = None
    # How far (px) a column may be from its site.
    radius: float = 5.0
    # "radius": each column to the nearest site within ``radius``;
    # "clusters": each site the dense, connected groups of columns that come
    # within ``radius`` of it (columns within ``link`` px of each other,
    # groups of ``least`` or more), however far they reach.
    mode: Literal["radius", "clusters"] = "radius"
    link: float = 4.0
    least: int = 5

    def site_names(self) -> list[str]:
        per_site = [x.strip() for x in self.names] if self.names else list(atomic.SUBLATTICE_NAMES[:len(self.sites)])
        if len(per_site) != len(self.sites) or not all(per_site):
            raise ValueError("Each site needs its sublattice's name")
        if len(dict.fromkeys(per_site)) > len(atomic.SUBLATTICE_NAMES):
            raise ValueError(f"At most {len(atomic.SUBLATTICE_NAMES)} sublattices")
        return per_site

    def assign(self, cell: np.ndarray) -> tuple[list[int | None], list[dict] | None]:
        """Each column's site (index into ``sites``), and the fitted clusters (clusters mode)."""
        if self.mode == "clusters":
            labels, fits, _ = atomic.cluster_sites(cell, self.sites, self.vectors, self.link, self.least, self.radius)
            return labels, fits
        return atomic.assign_sites(cell, self.sites, self.vectors, self.radius), None


@router.put("/api/datasets/{dataset_id}/atomic/columns/sublattices/cell")
def sublattices_by_cell(dataset_id: str, body: CellSitesBody, request: Request) -> dict:
    """Split the columns by their place in the reference columns' cell: each
    to the nearest site within ``radius`` px (the cell repeating), each site
    into its sublattice (``names``; replacing those there were). Returns its
    ``file`` and the sublattices' ``ids``."""
    session = request.app.state.session
    cols = _columns_of(request, dataset_id)
    cell = _cell_of(request, cols, body)
    try:
        per_site = body.site_names()
        names = list(dict.fromkeys(per_site))
        site_label = [names.index(x) for x in per_site]
        sites, _ = body.assign(cell)
        labels = [None if k is None else site_label[k] for k in sites]
        made = session.set_sublattices(cols, labels, names)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"file": session.file_of(cols).summary(session), "ids": [d.id for d in made]}


class PicksBody(BaseModel):
    # What was picked or set by hand for this dataset (the page's own
    # record: e.g. a neighbourhood's lattice vectors and cage, the Columns'
    # cell sites), kept with it and saved with the analysis.
    picks: dict


@router.put("/api/datasets/{dataset_id}/atomic/picks", status_code=204)
def set_picks(dataset_id: str, body: PicksBody, request: Request) -> None:
    """Keep the picks made for a dataset of the column analysis (Columns,
    sublattices, neighbourhoods …) with it; its summary's ``atomic.picks``."""
    import json

    session = request.app.state.session
    try:
        ds = session.get_dataset(dataset_id)
    except KeyError:
        raise HTTPException(404, f"Unknown dataset id: {dataset_id}")
    if not ds.atomic:
        raise HTTPException(400, f"{ds.title} is not part of a column analysis")
    if len(json.dumps(body.picks)) > 200_000:
        raise HTTPException(413, "Too much to keep")
    ds.atomic = {**ds.atomic, "picks": body.picks}


@router.post("/api/datasets/{dataset_id}/atomic/columns/cell/fit")
def fit_cell_sites(dataset_id: str, body: CellSitesBody, request: Request) -> dict:
    """The sites' clusters in the cell, as a split in clusters mode makes
    them but without splitting: each column's ``labels`` (an index into the
    sites, or null), per site ``u``, ``v``, ``count`` and ``empty``, the
    groups no site took (``unpicked``: ``u``, ``v``, ``count``) and how many
    columns are ``left_out``."""
    cols = _columns_of(request, dataset_id)
    cell = _cell_of(request, cols, body)
    try:
        body.site_names()
        labels, fits, unpicked = atomic.cluster_sites(cell, body.sites, body.vectors, body.link, body.least, body.radius)
    except ValueError as e:
        raise HTTPException(422, str(e))
    ok = np.isfinite(cell).all(axis=1)
    return {"labels": labels, "sites": fits, "unpicked": unpicked,
            "left_out": int(sum(1 for x, o in zip(labels, ok) if o and x is None))}


class KMeansBody(BaseModel):
    values: list[float | None] = Field(max_length=500_000)
    k: int = 2


@router.post("/api/atomic/kmeans")
def kmeans(body: KMeansBody) -> dict:
    """``centres`` of ``k`` clusters of a column value, largest first."""
    values = np.array([np.nan if v is None else v for v in body.values], dtype=np.float64)
    try:
        return {"centres": atomic.kmeans_centres(values, body.k)}
    except ValueError as e:
        raise HTTPException(422, str(e))


# ---------- M22e: neighbourhoods and bonds ----------

def _atomic_of(request: Request, dataset_id: str, *outputs: str):
    session = request.app.state.session
    try:
        ds = session.get_dataset(dataset_id)
    except KeyError:
        raise HTTPException(404, f"Unknown dataset id: {dataset_id}")
    if not (ds.atomic and ds.atomic.get("output") in outputs):
        raise HTTPException(400, f"{ds.title} is not a {' or '.join(outputs)} dataset")
    return ds


class NeighbourhoodBody(BaseModel):
    neighbours: str
    k: int = 20
    #: A neighbourhood to make again with these settings (else a new one).
    replace: str | None = None


@router.post("/api/datasets/{dataset_id}/atomic/neighbourhood")
def make_neighbourhood(dataset_id: str, body: NeighbourhoodBody, request: Request) -> dict:
    """The neighbourhood of columns ``dataset_id`` (Columns or a sublattice):
    the vectors to their ``k`` nearest columns of ``neighbours`` (Columns or
    a sublattice), a new dataset under them, or ``replace`` made again.
    Returns ``file`` and its ``id``."""
    session = request.app.state.session
    origins = _atomic_of(request, dataset_id, "columns", "sublattice")
    neighbours = _atomic_of(request, body.neighbours, "columns", "sublattice")
    replace = _atomic_of(request, body.replace, "neighbourhood") if body.replace else None
    try:
        made = session.make_neighbourhood(origins, neighbours, body.k, replace)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"file": session.file_of(made).summary(session), "id": made.id}


class ClustersBody(BaseModel):
    n: int = 30


@router.post("/api/datasets/{dataset_id}/atomic/neighbourhood/clusters")
def cluster_neighbourhood(dataset_id: str, body: ClustersBody, request: Request) -> dict:
    """k-means clusters of a neighbourhood's vectors, kept with it. Returns ``file``."""
    session = request.app.state.session
    nb = _atomic_of(request, dataset_id, "neighbourhood")
    try:
        session.cluster_neighbourhood(nb, body.n)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"file": session.file_of(nb).summary(session)}


@router.get("/api/datasets/{dataset_id}/atomic/neighbourhood")
def get_neighbourhood(dataset_id: str, request: Request) -> dict:
    """A neighbourhood's vectors (``dy``, ``dx``, px, y down) and clusters."""
    nb = _atomic_of(request, dataset_id, "neighbourhood")
    v = np.round(np.asarray(nb.signal.data, dtype=np.float64), 3)
    return {"dy": v[:, 0].tolist(), "dx": v[:, 1].tolist(), "clusters": nb.atomic.get("clusters") or []}


class BondsBody(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    vector: tuple[float, float]
    threshold: float = 3.0


@router.post("/api/datasets/{dataset_id}/atomic/bonds")
def make_bonds(dataset_id: str, body: BondsBody, request: Request) -> dict:
    """Bonds along ``vector`` ([dy, dx] px) from the origins of neighbourhood
    ``dataset_id``, a dataset under it. Returns ``file`` and its ``id``."""
    session = request.app.state.session
    nb = _atomic_of(request, dataset_id, "neighbourhood")
    try:
        made = session.make_bonds(nb, body.name, body.vector, body.threshold)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"file": session.file_of(made).summary(session), "id": made.id}


class CageBody(BaseModel):
    # The picked clusters ([dy, dx] px) whose nearest neighbours make the cage.
    vectors: list[tuple[float, float]] = Field(default=[], max_length=24)
    threshold: float = 3.0
    # The cage is all the k nearest neighbours instead (no clusters picked).
    nearest: bool = False


@router.post("/api/datasets/{dataset_id}/atomic/neighbourhood/centres")
def cage_centres(dataset_id: str, body: CageBody, request: Request) -> dict:
    """Column guesses at the centre of each origin's cage (the neighbours
    nearest the picked ``vectors``), after the Columns' fitted positions, to
    fit them all again: ``columns``, ``labels``, ``names`` (the new ones a
    sublattice of their own) and ``added``."""
    session = request.app.state.session
    nb = _atomic_of(request, dataset_id, "neighbourhood")
    try:
        if not body.nearest and not body.vectors:
            raise ValueError("Pick the cage's clusters first")
        return session.cage_centres(nb, body.vectors, body.threshold, body.nearest)
    except ValueError as e:
        raise HTTPException(422, str(e))


@router.post("/api/datasets/{dataset_id}/atomic/neighbourhood/displacement")
def cage_displacement(dataset_id: str, body: CageBody, request: Request) -> dict:
    """Each origin's displacement from the centre of its cage (the
    neighbours nearest the picked ``vectors``): "Displacement" under the
    neighbourhood (vectors, as ``…/atomic/displacements``). Returns ``file``
    and its ``id``."""
    session = request.app.state.session
    nb = _atomic_of(request, dataset_id, "neighbourhood")
    try:
        if not body.nearest and not body.vectors:
            raise ValueError("Pick the cage's clusters first")
        made = session.set_cage_displacement(nb, body.vectors, body.threshold, body.nearest)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"file": session.file_of(made).summary(session), "id": made.id}


@router.get("/api/datasets/{dataset_id}/atomic/bonds")
def get_bonds(dataset_id: str, request: Request) -> dict:
    """A bonds dataset's table: ``fields`` and ``rows`` (NaN as null)."""
    ds = _atomic_of(request, dataset_id, "bonds")
    table = np.asarray(ds.signal.data, dtype=np.float64)
    rows = np.where(np.isfinite(table), np.round(table, 4), None).tolist()
    return {"fields": list(atomic.BOND_FIELDS), "rows": rows}



# ---------- M22f: periodic lattice displacements ----------

class DetectBody(BaseModel):
    count: int = 120
    within: float = 0.25
    noise: float = 4.0
    edge: Literal["none", "hann", "pplus"] = "none"


@router.post("/api/datasets/{dataset_id}/atomic/pld/detect")
def detect_peaks(dataset_id: str, body: DetectBody, request: Request) -> dict:
    """The image's FFT peaks found and labelled: ``peaks`` (``fy``, ``fx``,
    ``height``, ``hk`` in ``basis``, ``label``: bragg or superlattice,
    ``family``: 0 Bragg, n for family n, null), the lattice ``basis`` (two
    (fy, fx)) and the ``families`` of off-lattice peaks (``id``, ``q``,
    ``count``, ``strength``)."""
    image = image_of(request, dataset_id)
    try:
        peaks = atomic.detect_fft_peaks(image, body.count, body.within, edge=body.edge, noise=body.noise)
    except ValueError as e:
        raise HTTPException(422, str(e))
    basis = atomic.lattice_basis(peaks)
    peaks, families = atomic.peak_families(atomic.label_peaks(peaks, basis))
    return {"peaks": peaks, "basis": basis, "families": families}


class ReferenceBody(BaseModel):
    peaks: list[tuple[float, float]] = Field(min_length=1, max_length=500)
    radius: float = 8.0
    window: int = 9
    blur: float = 1.0
    edge: Literal["none", "hann", "pplus"] = "none"


@router.post("/api/datasets/{dataset_id}/atomic/pld/reference")
def pld_reference(dataset_id: str, body: ReferenceBody, request: Request) -> dict:
    """The image with the superlattice ``peaks`` (fy, fx) damped (kemstem's
    PLD method): "PLD reference" and "PLD difference" under it. Returns
    ``file``, their ``ids`` and the refined ``peaks``."""
    session = request.app.state.session
    image = image_of(request, dataset_id)
    ds = session.get_dataset(dataset_id)
    try:
        made = atomic.damp_peaks(image, np.array(body.peaks), body.radius, body.window, body.blur, body.edge)
    except ValueError as e:
        raise HTTPException(422, str(e))
    settings = {"radius": body.radius, "window": body.window, "blur": body.blur, "edge": body.edge,
                "damped": [[round(p["fy"], 6), round(p["fx"], 6)] for p in made["peaks"] if not p["failed"]]}
    out = session.set_pld_images(ds, made, settings)
    return {"file": session.file_of(ds).summary(session), "ids": [d.id for d in out], "peaks": made["peaks"]}


class DisplacementsBody(BaseModel):
    threshold: float = 0.4


@router.post("/api/datasets/{dataset_id}/atomic/pld/displacements")
def pld_displacements(dataset_id: str, body: DisplacementsBody, request: Request) -> dict:
    """Periodic lattice displacements of Columns ``dataset_id`` (a job): its
    columns fitted again in the image's PLD reference (same window and
    blur, from the fitted positions), and kemstem's
    ``measure_displacements`` from those to the fitted ones (within
    ``threshold`` px). The result, "Displacements" under the Columns: the
    job's ``file`` and ``id``."""
    session = request.app.state.session
    columns = _columns_of(request, dataset_id)
    image_ds = session.get_dataset(columns.parent_id)
    reference = session.pld_reference_of(image_ds)
    if reference is None:
        raise HTTPException(422, "Make the PLD reference first")
    if not body.threshold > 0:
        raise HTTPException(422, "The threshold must be more than 0")
    a = columns.atomic
    try:
        opts = atomic.ColumnOptions(a.get("blur", 0.5), a.get("distance", 4), a.get("threshold", 0.1), a.get("window", 7)).check()
    except ValueError as e:
        raise HTTPException(422, str(e))
    table = np.asarray(columns.signal.data, dtype=np.float64)
    fitted = table[:, :2].copy()
    fitted[table[:, 8] > 0] = np.nan  # failed fits: no position
    ref_image = np.asarray(reference.signal.data, dtype=np.float64)

    def run(job) -> dict:
        ok = np.isfinite(fitted).all(axis=1)
        ref = np.full_like(fitted, np.nan)
        rows = atomic.refine_columns(ref_image, fitted[ok], opts,
                                     report=lambda f: job.report(0.95 * f, f"Fitting {int(ok.sum())} columns in the reference"))
        good = rows[:, 8] == 0
        ref_ok = rows[:, :2].copy()
        ref_ok[~good] = np.nan
        ref[ok] = ref_ok
        job.report(0.97, "Measuring the displacements")
        out = atomic.measure_displacements(ref, fitted, body.threshold)
        made = session.set_displacements(columns, out, {"threshold": body.threshold, "reference": reference.id})
        return {"file": session.file_of(columns).summary(session), "id": made.id}

    return request.app.state.jobs.submit("atomic-pld", f"PLD displacements of {image_ds.title}", run).summary()


@router.get("/api/datasets/{dataset_id}/atomic/displacements")
def get_displacements(dataset_id: str, request: Request) -> dict:
    """A Displacements dataset's table: ``fields`` and ``rows`` (NaN as null), a row per column."""
    ds = _atomic_of(request, dataset_id, "displacements")
    table = np.asarray(ds.signal.data, dtype=np.float64)
    rows = np.where(np.isfinite(table), np.round(table, 5), None).tolist()
    return {"fields": list(atomic.DISPLACEMENT_FIELDS), "rows": rows}
