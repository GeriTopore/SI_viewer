"""Saving and opening an analysis as a project file (M18b).

A project is a HyperSpy file (``.hspy``, HDF5): ``hs.load(path)`` gives the
list of the results stored in it (one ``Experiments/<title> [<id>]`` group
each, written with rosettasciio's own writer). Next to them, the group
``si_viewer`` holds what the app needs to rebuild the analysis, as JSON:

- the raw files, referenced (absolute path, path relative to the project and
  a fingerprint), and the experiments (folders) they were opened in;
- every dataset's settings (origin, parent, title, filters, roles, links,
  window, background options, …) and the operations that changed its loaded
  data (``Dataset.ops``), in the tree's order;
- the views (the frontend's state, opaque here).

Saved at level "link", datasets loaded from a raw file are not stored:
opening a project opens the raw files again, gives their datasets their saved
ids (so the views apply unchanged), replays the operations in the order they
were made and puts the saved settings back. Every other dataset (maps, fits,
background-removed spectra, LineScans, copies, …) is stored, and so is the
frame range an in-situ series held (a mean or sum would have to be read
again).

Saved at level "raw", the raw files are embedded too (``si_viewer/raw/<file
id>``, their bytes; not in-situ series, whose frames are separate files:
their results are stored instead): opening uses the embedded copy (written to
a cache folder) when the raw file can't be found or has changed. A file
opened from its embedded copy is always embedded again.

Saved at level "results", the datasets loaded from raw files are stored too,
as they are now (an in-situ series: the frame or range shown, never the
series), so the analysis opens without the raw data: a file whose raw data
can't be found, or has changed, is opened from what was stored
(``FileEntry.raw_missing``). Datasets of a file opened that way are always
stored again.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
GROUP = "si_viewer"
LEVELS = ("link", "results", "raw")
# Bytes copied at a time when embedding or extracting a raw file.
BLOCK = 64 << 20

# Dataset fields saved as they are (JSON).
FIELDS = (
    "id", "file_id", "is_element_map", "origin", "element", "parent_id", "window", "background",
    "thickness", "deconvolution", "quantity", "source_frame", "map_options", "rebinned", "zlp_aligned",
    "ops", "copied_at", "filters", "alignment", "technique_source", "eels_role", "eels_role_source",
    "eels_group", "eels_linked", "derived_from", "custom_title", "survey", "dpc", "pca", "microscope", "curve_model",
)
# Bytes of a raw file hashed for its fingerprint.
FINGERPRINT_BYTES = 1 << 20


class ProjectError(ValueError):
    """A project that can't be opened, in plain English."""


def is_project(path: str | Path) -> bool:
    """Whether ``path`` is an HDF5 file holding a saved analysis."""
    import h5py

    try:
        with h5py.File(path, "r") as f:
            return GROUP in f
    except OSError:
        return False


def fingerprint(path: Path) -> dict:
    st = path.stat()
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read(FINGERPRINT_BYTES))
    return {"size": st.st_size, "mtime": st.st_mtime, "sha1": h.hexdigest()}


def _jsonable(value):
    """``value`` with NumPy scalars and arrays made plain (for json.dumps)."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_signal(f, signal, name: str) -> str:
    """Write ``signal`` as ``Experiments/<name>`` with rosettasciio's hspy
    writer (what ``signal.save`` writes). Returns the group's name."""
    from hyperspy.misc.utils import get_object_package_info
    from rsciio.hspy._api import HyperspyWriter

    name = name.replace("/", "-")
    group = f.require_group("Experiments").require_group(name)
    d = signal._to_dictionary(add_models=True)
    d["package_info"] = get_object_package_info(signal)
    HyperspyWriter(f, d, group, compression="gzip", shuffle=True, show_progressbar=False).write()
    return name


def _read_signal(path: Path, name: str):
    """The signal stored as ``Experiments/<name>``."""
    import h5py
    from hyperspy.io import dict2signal
    from rsciio.hspy._api import HyperspyReader

    with h5py.File(path, "r") as f:
        reader = HyperspyReader(f)
        exp = reader.group2signaldict(f["Experiments"][name], lazy=False)
    return dict2signal(exp, lazy=False)


# ---------- saving ----------


def embeddable(session, entry) -> bool:
    """Whether ``entry``'s raw file can be embedded: one file holding all its
    data (not an in-situ series, whose frames are in files of their own)."""
    if entry.raw_missing or not entry.path.is_file():
        return False
    for i in entry.dataset_ids:
        ds = session.datasets[i]
        if ds.origin == "file" and (ds.insitu is not None or ds.loaded_signal.metadata.has_item("Insitu")):
            return False
    return True


def embeds(session, entry, level: str) -> bool:
    """Whether ``entry``'s raw file is embedded at ``level``."""
    return embeddable(session, entry) and (level == "raw" or entry.saved_ref is not None)


def stores_raw(session, entry, level: str) -> bool:
    """Whether the datasets loaded from ``entry``'s raw file are stored."""
    if entry.raw_missing or level == "results":
        return True
    return level == "raw" and not embeds(session, entry, level)


def saved_files(session, file_ids=None) -> list:
    """The files a save of ``file_ids`` holds (all open files if None): those,
    and the files of whatever their datasets are linked to or made from, in
    the session's order."""
    with session._lock:
        if file_ids is None:
            return list(session.files.values())
        unknown = [i for i in file_ids if i not in session.files]
        if unknown:
            raise ProjectError(f"Unknown file id: {unknown[0]}")
        keep = set(file_ids)
        todo = list(keep)
        while todo:
            for i in session.files[todo.pop()].dataset_ids:
                ds = session.datasets[i]
                for other in [*ds.eels_linked, ds.parent_id, ds.derived_from]:
                    o = session.datasets.get(other) if other else None
                    if o is not None and o.file_id not in keep:
                        keep.add(o.file_id)
                        todo.append(o.file_id)
        return [e for e in session.files.values() if e.id in keep]


def estimate(session, file_ids=None) -> dict:
    """Bytes stored (before compression) at each level, saving ``file_ids``
    (see ``saved_files``); and ``file_ids``, the files saved."""
    sizes = dict.fromkeys(LEVELS, 0)
    entries = saved_files(session, file_ids)
    with session._lock:
        for level in LEVELS:
            for entry in entries:
                raw = stores_raw(session, entry, level)
                if embeds(session, entry, level):
                    sizes[level] += entry.path.stat().st_size
                for i in entry.dataset_ids:
                    ds = session.datasets[i]
                    shown_range = ds.insitu is not None and ds.insitu.view.mode != "frame"
                    if ds.origin != "file" or raw or shown_range:
                        sizes[level] += _nbytes(ds)
    return {**sizes, "file_ids": [e.id for e in entries]}


def _nbytes(ds) -> int:
    signals = list(ds.quantities.values()) or [ds.signal]
    if ds.original_signal is not None:
        signals.append(ds.original_signal)
    total = 0
    for s in signals:
        data = s.data
        total += int(np.prod(data.shape)) * np.dtype(data.dtype).itemsize
    return total


def save_project(
    session, path: str | Path, views: dict | None = None, level: str = "link", file_ids=None,
) -> Path:
    """Save the session as a project at ``path`` (``.hspy`` added if it has
    no suffix): written to a temporary file next to it, then moved into
    place. ``views``: the frontend's state, saved as it is. ``level``: "link"
    (raw data referenced), "results" (the raw datasets' data stored too) or
    "raw" (the raw files embedded; in-situ series' results stored).
    ``file_ids``: the files to save (with what they link to, see
    ``saved_files``); all open files if None."""
    if level not in LEVELS:
        raise ProjectError(f"Unknown level: {level}")
    import h5py
    from rsciio.hspy._api import version

    path = Path(path).expanduser()
    if not path.suffix:
        path = path.with_suffix(".hspy")
    path = path.resolve()
    if not path.parent.is_dir():
        raise ProjectError(f"The folder {path.parent} does not exist")
    tmp = path.with_name(f".{path.name}.saving")
    base = path.parent

    def rel(p: Path) -> str:
        return os.path.relpath(p, base)

    entries = saved_files(session, file_ids)
    with session._lock:
        files, datasets = [], []
        store_raw, embed = set(), {}
        for entry in entries:
            stored = stores_raw(session, entry, level)
            if stored:
                store_raw.update(entry.dataset_ids)
            if embeds(session, entry, level):
                embed[entry.id] = entry.path
            if entry.saved_ref is not None:  # the raw file is where it was, not the copy
                where, fp = Path(entry.saved_ref["path"]), entry.saved_ref["fingerprint"]
            elif entry.raw_missing:
                where, fp = entry.path, entry.fingerprint
            else:
                where, fp = entry.path, fingerprint(entry.path) if entry.path.is_file() else None
            files.append({
                "id": entry.id, "path": str(where), "relative": rel(where), "label": entry.label,
                "experiment_id": entry.experiment_id, "dataset_ids": list(entry.dataset_ids),
                "fingerprint": fp, "stored": stored, "embedded": entry.id in embed,
            })
        kept = {e.id for e in entries}
        experiments = [
            {"id": e.id, "path": str(e.path), "relative": rel(e.path), "file_ids": [i for i in e.file_ids if i in kept]}
            for e in session.experiments.values() if any(i in kept for i in e.file_ids)
        ]
        order = [d for entry in entries for d in entry.dataset_ids]
        groups = {
            k: list(v) for k, v in session.eels_groups.items() if any(m in order for m in v)
        }

    try:
        with h5py.File(tmp, "w") as f:
            f.attrs["file_format"] = "HyperSpy"
            f.attrs["file_format_version"] = version
            for ds_id in order:
                ds = session.datasets[ds_id]
                rec = {k: getattr(ds, k) for k in FIELDS}
                rec["kind"] = ds.kind.value
                rec["title"] = ds.title
                rec["stored"] = {}
                # A cropped in-situ series: its crop, remade from the raw files on open.
                crop_series = ds.insitu is not None and ds.insitu.crop is not None
                if crop_series:
                    rec["insitu_crop"] = [list(c) for c in ds.insitu.crop]
                if (ds.origin != "file" and not crop_series) or ds_id in store_raw:
                    ds.materialise()
                    label = f"{ds.title} [{ds.id}]"
                    if ds.quantities:  # fitted maps: each quantity
                        for q, signal in ds.quantities.items():
                            rec["stored"][q] = _write_signal(f, signal, f"{label} {q}")
                    elif ds.alignment is not None and ds.original_signal is not None:
                        # An aligned stack: as loaded, and aligned.
                        rec["stored"]["data"] = _write_signal(f, ds.original_signal, label)
                        rec["stored"]["aligned"] = _write_signal(f, ds.aligned_signal, f"{label} aligned")
                        rec["showing_aligned"] = ds.signal is ds.aligned_signal
                    else:
                        rec["stored"]["data"] = _write_signal(f, ds.signal, label)
                if ds.insitu is not None:
                    view = ds.insitu.view
                    rec["insitu"] = {"view": view.as_dict(), "binning": list(ds.insitu.binning)}
                    if "data" not in rec["stored"] and view.mode != "frame" and view.start != view.end:
                        ds.materialise()
                        rec["stored"]["view"] = _write_signal(f, ds.signal, f"{ds.title} [{ds.id}] {view.label()}")
                datasets.append(_jsonable(rec))
            g = f.require_group(GROUP)
            for file_id, src in embed.items():
                _embed(g.require_group("raw"), file_id, src)
            g.attrs["schema_version"] = SCHEMA_VERSION
            g.attrs["project"] = json.dumps(_jsonable({
                "level": level, "files": files, "experiments": experiments, "eels_groups": groups,
                "datasets": datasets,
            }))
            g.attrs["views"] = json.dumps(views or {})
            g.attrs["versions"] = json.dumps(_versions())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def _embed(group, name: str, src: Path) -> None:
    """Copy the bytes of ``src`` into ``group[name]`` (a block at a time)."""
    size = src.stat().st_size
    out = group.create_dataset(name, shape=(size,), dtype="u1", chunks=(min(size, 1 << 20),) if size else None)
    out.attrs["name"] = src.name
    with open(src, "rb") as fh:
        offset = 0
        while offset < size:
            block = fh.read(BLOCK)
            out[offset:offset + len(block)] = np.frombuffer(block, np.uint8)
            offset += len(block)


def _extract(path: Path, fe: dict) -> Path:
    """The raw file embedded for ``fe`` written to a cache folder (reused if
    it is there already). Returns its path."""
    import tempfile

    import h5py

    fp = fe.get("fingerprint") or {}
    folder = Path(tempfile.gettempdir()) / "si-viewer-embedded" / (fp.get("sha1") or fe["id"])
    target = folder / Path(fe["path"]).name
    with h5py.File(path, "r") as f:
        src = f[GROUP]["raw"][fe["id"]]
        size = src.shape[0]
        if target.is_file() and target.stat().st_size == size:
            return target
        folder.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.part")
        with open(tmp, "wb") as fh:
            for offset in range(0, size, BLOCK):
                fh.write(src[offset:offset + BLOCK].tobytes())
        os.replace(tmp, target)
    return target


def _versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for pkg in ("si-viewer", "hyperspy", "exspy", "rosettasciio"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    return out


# ---------- opening ----------


def read_project(path: str | Path) -> tuple[dict, dict]:
    """(project, views) as saved in ``path``. ProjectError if it is not one."""
    import h5py

    path = Path(path)
    try:
        with h5py.File(path, "r") as f:
            if GROUP not in f:
                raise ProjectError(f"{path.name} is not a saved analysis")
            g = f[GROUP]
            if int(g.attrs.get("schema_version", 0)) > SCHEMA_VERSION:
                raise ProjectError(f"{path.name} was saved by a newer version of the app")
            return json.loads(g.attrs["project"]), json.loads(g.attrs.get("views", "{}"))
    except OSError as e:
        raise ProjectError(f"{path.name} can't be read: {e}") from None


def _find(saved: dict, base: Path, locations: dict | None = None) -> Path | None:
    """A saved file or folder: where it was located (``locations``: id ->
    path), else next to the project as it was, else where it was, else in a
    folder another file was located in."""
    locations = locations or {}
    name = Path(saved["path"]).name
    candidates = [Path(locations[saved["id"]])] if saved.get("id") in locations else []
    candidates += [base / saved["relative"], Path(saved["path"])]
    candidates += [Path(p).parent / name for p in locations.values()]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _changed(p: Path, fp: dict | None) -> bool:
    if not fp or not p.is_file():
        return False
    now = fingerprint(p)
    return now["size"] != fp["size"] or now["sha1"] != fp["sha1"]


def missing_raw(project: dict, base: Path, locations: dict | None = None) -> list[dict]:
    """The raw files that can't be found, and are neither embedded nor
    stored: {id, name, path} each."""
    return [
        {"id": fe["id"], "name": Path(fe["path"]).name, "path": fe["path"]}
        for fe in project["files"]
        if not fe.get("stored") and not fe.get("embedded") and _find(fe, base, locations) is None
    ]


def open_project(
    session, path: str | Path, locations: dict | None = None, defer: list | None = None,
) -> tuple[dict, list[str]]:
    """Open the project at ``path`` into ``session`` (which should be empty:
    ids are reused). Returns (views, warnings: raw files changed or not
    found). ``locations``: where raw files that moved are (file id -> path;
    the others are also looked for in their folders). A raw file that can't
    be found, or has changed, is opened from its embedded copy, else from the
    data stored. ProjectError if a raw file that is neither can't be found
    (nothing is opened). ``defer``: a list to put the slow operations to
    redo in (drift corrections, and what came after them on the same stack)
    instead of redoing them: ``finish_open`` redoes them later."""
    from .session import Dataset, DatasetKind, Experiment, FileEntry

    path = Path(path).expanduser().resolve()
    project, views = read_project(path)
    base = path.parent
    files = project["files"]
    datasets = {d["id"]: d for d in project["datasets"]}

    missing = missing_raw(project, base, locations)
    if missing:
        raise ProjectError("Raw data not found: " + "; ".join(m["path"] for m in missing))
    # Opened next to what is open: not the same files twice.
    with session._lock:
        open_paths = {e.path for e in session.files.values()}
        for fe in files:
            p = _find(fe, base, locations)
            clash = fe["id"] in session.files or any(i in session.datasets for i in fe["dataset_ids"])
            if clash or (p is not None and p in open_paths):
                name = fe.get("label") or Path(fe["path"]).name
                raise ProjectError(f"{name} is already open: close it first")

    warnings = []
    found, detached, extracted = {}, set(), set()
    for fe in files:
        p = _find(fe, base, locations)
        name = Path(fe["path"]).name
        changed = p is not None and _changed(p, fe.get("fingerprint"))
        if p is not None and not changed:
            pass
        elif fe.get("embedded"):
            p = _extract(path, fe)
            extracted.add(fe["id"])
            if changed:
                warnings.append(f"{name} has changed since the analysis was saved: its embedded copy is used")
        elif fe.get("stored"):
            detached.add(fe["id"])
            warnings.append(
                f"{name} has changed since the analysis was saved: its saved data is used" if changed
                else f"{name} not found: its saved data is used")
        else:
            warnings.append(f"{name} has changed since the analysis was saved")
        found[fe["id"]] = p

    project_file_ids = {fe["id"] for fe in files}
    # The experiments, then the raw files (in their order), with their saved ids.
    for e in project.get("experiments", []):
        ep = _find(e, base) or Path(e["path"])
        with session._lock:
            session.experiments.setdefault(e["id"], Experiment(id=e["id"], path=ep))
    for fe in files:
        if fe["id"] in detached:
            entry = FileEntry(
                id=fe["id"], path=Path(fe["path"]), label=fe.get("label"), raw_missing=True,
                fingerprint=fe.get("fingerprint"),
            )
            if fe.get("experiment_id") in session.experiments:
                entry.experiment_id = fe["experiment_id"]
                session.experiments[fe["experiment_id"]].file_ids.append(fe["id"])
            with session._lock:
                session.files[fe["id"]] = entry
            continue
        p = found[fe["id"]]
        entry = session.open_file(p)
        if entry.id in project_file_ids:
            continue  # already opened with another (e.g. an in-situ SI brings its ADF)
        if fe["id"] in extracted:
            entry.saved_ref = {"path": fe["path"], "fingerprint": fe.get("fingerprint")}
        if fe.get("experiment_id") in session.experiments:
            entry.experiment_id = fe["experiment_id"]
            session.experiments[fe["experiment_id"]].file_ids.append(fe["id"])
        entry.label = fe.get("label")
        saved_raw = [i for i in fe["dataset_ids"] if datasets[i]["origin"] == "file"]
        _adopt_ids(session, entry, fe["id"], saved_raw)

    # The operations on the raw datasets, in the order they were made.
    raw = [d for d in datasets.values() if d["origin"] == "file" and d["id"] in session.datasets]
    for rec in raw:
        ds = session.datasets[rec["id"]]
        if rec.get("eels_role"):
            ds.eels_role, ds.eels_role_source = rec["eels_role"], rec["eels_role_source"]
    _restore_groups(session, project.get("eels_groups", {}), datasets)
    steps = sorted(
        ((op.get("seq", 0), rec["id"], op) for rec in raw for op in rec.get("ops", [])),
        key=lambda t: (t[0], t[1]),
    )
    if defer is not None:
        # A stack's drift correction is slow to redo, and its average is
        # stored: redone later, with what followed it on that stack.
        first_align = {}
        for seq, ds_id, op in steps:
            if op["op"] == "align":
                first_align.setdefault(ds_id, seq)
        later = [t for t in steps if t[1] in first_align and t[0] >= first_align[t[1]]]
        steps = [t for t in steps if t not in later]
        defer.extend((ds_id, op) for _, ds_id, op in later)
    done_seq = set()
    for seq, ds_id, op in steps:
        ds = session.datasets[ds_id]
        _replay(session, ds, op, seq in done_seq)
        done_seq.add(seq)
    # Results the replays made again (an alignment's average, a ZLP shift map)
    # are stored in the project: theirs are used.
    made_again = [
        session.datasets[i] for fid in project_file_ids if fid in session.files
        for i in session.files[fid].dataset_ids if i not in datasets
    ]
    for ds in made_again:
        session._remove(ds)
    for rec in raw:
        ds = session.datasets[rec["id"]]
        for k in ("map_options", "rebinned", "zlp_aligned", "ops", "filters", "custom_title", "technique_source"):
            if k in rec:
                setattr(ds, k, rec[k])
        if rec.get("custom_title"):
            ds.signal.metadata.General.title = rec["custom_title"]
        if rec.get("microscope"):
            ds.set_microscope(**rec["microscope"])
        if rec.get("filters"):
            ds.filter_rev += 1
        if ds.insitu is not None and rec.get("insitu"):
            _restore_view(session, path, ds, rec)

    # The stored datasets, in the tree's order, with their saved ids.
    for fe in files:
        for i in fe["dataset_ids"]:
            rec = datasets[i]
            if rec["origin"] == "file" and fe["id"] not in detached:
                continue
            if rec.get("insitu_crop"):
                src = session.datasets.get(rec.get("parent_id"))
                if src is None or src.insitu is None:
                    warnings.append(f"{rec['title']}: its in-situ series is not open, so it can't be cropped again")
                    continue
                made = session.crop_series_raw(src, tuple(rec["insitu_crop"]), rec["title"], ds_id=i)
                binning = tuple((rec.get("insitu") or {}).get("binning") or (1, 1, 1, 1))
                if binning != (1, 1, 1, 1):
                    made.insitu.set_binning(binning)
                    made.signal = made.insitu.placeholder()
                    made.signal.metadata.General.title = rec["title"]
                for k in ("custom_title", "filters", "map_options", "eels_role", "eels_role_source", "derived_from"):
                    if k in rec:
                        setattr(made, k, rec[k])
                if rec.get("insitu"):
                    _restore_view(session, path, made, rec)
                continue
            stored = rec["stored"]
            quantities = {}
            aligned = None
            if "data" in stored:
                signal = _read_signal(path, stored["data"])
                if "aligned" in stored:
                    aligned = _read_signal(path, stored["aligned"])
            else:
                quantities = {q: _read_signal(path, name) for q, name in stored.items()}
                signal = quantities[rec.get("quantity") or next(iter(quantities))]
            ds = Dataset(id=i, file_id=fe["id"], signal=signal, kind=DatasetKind(rec["kind"]))
            for k in FIELDS:
                if k not in ("id", "file_id") and k in rec:  # fields added since keep their defaults
                    setattr(ds, k, rec[k])
            ds.quantities = quantities
            if aligned is not None:
                ds.original_signal, ds.aligned_signal = signal, aligned
                if rec.get("showing_aligned"):
                    ds.signal = aligned
            with session._lock:
                session.datasets[i] = ds
        # The file's order as saved.
        with session._lock:
            session.files[fe["id"]].dataset_ids = [i for i in fe["dataset_ids"] if i in session.datasets]
    _restore_groups(session, project.get("eels_groups", {}), datasets)
    return views, warnings


def finish_open(session, deferred: list, report=None) -> list[str]:
    """Redo the operations ``open_project`` deferred (``(dataset id, op)``
    in order), on the stacks still open. A drift correction keeps the
    average stored with the analysis. ``report(fraction, message)`` is called
    before each. Returns the ids of the stacks redone."""
    done = []
    for k, (ds_id, op) in enumerate(deferred):
        ds = session.datasets.get(ds_id)
        if ds is None:
            continue  # closed meanwhile
        if report is not None:
            report(k / len(deferred), f"Aligning {ds.title}")
        ops = list(ds.ops)  # as saved: not recorded again
        if op["op"] == "align":
            session.restore_alignment(ds, realigned(ds, op["info"]))
        else:
            _replay(session, ds, op, False)
        ds.ops = ops
        if ds_id not in done:
            done.append(ds_id)
    return done


def _adopt_ids(session, entry, file_id: str, dataset_ids: list[str]) -> None:
    """Give a freshly opened file and its datasets (in order) the ids they
    had when the project was saved."""
    with session._lock:
        new_ids = list(entry.dataset_ids)
        if len(new_ids) != len(dataset_ids):
            raise ProjectError(f"{entry.path.name} no longer holds the datasets the analysis was made from")
        mapping = dict(zip(new_ids, dataset_ids))
        del session.files[entry.id]
        entry.id = file_id
        session.files[file_id] = entry
        entry.dataset_ids = [mapping[i] for i in new_ids]
        for old, new in mapping.items():
            ds = session.datasets.pop(old)
            ds.id, ds.file_id = new, file_id
            session.datasets[new] = ds
        for ds in session.datasets.values():
            ds.parent_id = mapping.get(ds.parent_id, ds.parent_id)
            ds.derived_from = mapping.get(ds.derived_from, ds.derived_from)
            ds.eels_linked = [mapping.get(x, x) for x in ds.eels_linked]
        for g, members in session.eels_groups.items():
            session.eels_groups[g] = [mapping.get(x, x) for x in members]


def _restore_groups(session, groups: dict, datasets: dict) -> None:
    """The project's EELS link groups as saved (members in the session)."""
    with session._lock:
        for g, members in groups.items():
            session.eels_groups[g] = [m for m in members if m in session.datasets]
        for ds in session.datasets.values():
            rec = datasets.get(ds.id)
            if rec is not None:
                ds.eels_group = rec.get("eels_group")
                ds.eels_linked = [m for m in rec.get("eels_linked", []) if m in session.datasets]


def _replay(session, ds, op: dict, repeat: bool) -> None:
    """Redo operation ``op`` on the raw dataset ``ds``. ``repeat``: the same
    operation was already redone for another dataset (a zero-loss alignment
    of a group is one operation)."""
    from .session import _rebin_in_place

    kind = op["op"]
    if kind == "rebin":
        ds.materialise()
        _rebin_in_place(ds, op["scale"])
    elif kind == "rebin-series":
        ds.materialise()
        session._rebin_series(ds, op["scale"], op["time"])
    elif kind == "zlp-align" and not repeat:
        from .processing.zlp import ZlpOptions

        options = ZlpOptions(**op["options"])
        ll = ds if ds.eels_role == "low_loss" else next(
            (d for d in session.linked_of(ds) if d.eels_role == "low_loss"), None)
        if ll is None:
            raise ProjectError(f"{ds.title}: its ZLP alignment can't be redone without its low loss")
        group, originals, result = session.compute_zlp_alignment(ll, options)
        session.apply_zlp_alignment(group, originals, result, options)
    elif kind == "energy-shift":
        session.shift_energy(ds, op["shift"], record=False)
    elif kind == "align":
        session.apply_alignment(ds, realigned(ds, op["info"]))
    elif kind == "unalign":
        session.remove_alignment(ds)


def realigned(stack, info: dict, signed: bool = False) -> dict:
    """``RegistrationSession.applied()`` for ``stack`` remade from a saved
    alignment ``info``: each frame shifted by minus its drift (as the package's
    ``generateShiftedImage``), cropped to the saved square. ``signed``: data
    of either sign (see ``shifted``)."""
    stack.materialise()
    frames = np.asarray(stack.loaded_signal.data)
    d = info["drift"]
    aligned = shifted(frames, d["frames"], [-y for y in d["y"]], [-x for x in d["x"]], signed)
    average = aligned.mean(axis=0)
    x, y, w, h = info["square"]
    return {
        "stack": np.ascontiguousarray(aligned[:, y:y + h, x:x + w]),
        "average": np.ascontiguousarray(average[y:y + h, x:x + w]),
        "info": info,
    }


def shifted(frames, which, rows, cols, signed: bool = False) -> np.ndarray:
    """Frames ``which`` of ``frames`` shifted by (``rows``, ``cols``) pixels
    each (subpixel, in Fourier space), as float32: the package's
    ``generateShiftedImage(frame, rows, cols)`` (the modulus of the shifted
    image), in single precision and on every core. ``signed``: the real
    part instead (for data of either sign, e.g. a DPC deflection)."""
    import scipy.fft as sf

    n0, n1 = frames.shape[1:]
    k0, k1 = np.fft.fftfreq(n0), np.fft.fftfreq(n1)
    out = np.empty((len(which), n0, n1), np.float32)
    for i, (f, r, c) in enumerate(zip(which, rows, cols)):
        spectrum = sf.fft2(np.asarray(frames[f], np.float32), workers=-1)
        spectrum *= np.outer(np.exp(-2j * np.pi * r * k0), np.exp(-2j * np.pi * c * k1)).astype(np.complex64)
        image = sf.ifft2(spectrum, workers=-1)
        out[i] = image.real if signed else np.abs(image)
    return out


def _restore_view(session, path: Path, ds, rec: dict) -> None:
    """The frame (or the stored frame range) an in-situ series held."""
    from .insitu_series import ViewState

    view = rec["insitu"]["view"]
    if "view" in rec["stored"]:
        data = np.asarray(_read_signal(path, rec["stored"]["view"]).data)
        session._set_insitu_view(ds, data, ViewState(view["mode"], view["start"], view["end"]))
    elif view["start"] < ds.insitu.n:
        session.show_insitu_frame(ds, view["start"])
