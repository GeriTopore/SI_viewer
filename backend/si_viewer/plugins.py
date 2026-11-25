"""Optional plugins: discovered at startup, isolated from the base app.

A plugin is a module (or object) with a ``manifest`` (:class:`PluginManifest`)
and ``register(ctx)`` (given a :class:`PluginContext`), installed under the
entry-point group ``si_viewer.plugins``. For development, ``SI_VIEWER_PLUGINS``
names further modules, comma-separated.

A plugin that can't be loaded is reported by ``GET /api/plugins`` and skipped;
what it had registered so far is dropped, and the app starts without it.
"""

from __future__ import annotations

import importlib
import os
import re
import traceback
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path
from typing import Callable, Iterable, Sequence

from fastapi import APIRouter, FastAPI
from fastapi.staticfiles import StaticFiles

from .jobs import Job

API_VERSION = 1
ENTRY_POINT_GROUP = "si_viewer.plugins"
ENV_VAR = "SI_VIEWER_PLUGINS"
_ID = re.compile(r"^[a-z][a-z0-9-]*$")


@dataclass(frozen=True)
class PluginManifest:
    id: str
    name: str
    version: str
    # The plugin API it was written for.
    api_version: int = API_VERSION
    # Folder of its built frontend, holding ``index.js``.
    frontend: Path | None = None


@dataclass
class PluginRecord:
    """What ``GET /api/plugins`` says about one plugin."""

    id: str
    name: str
    version: str | None = None
    status: str = "active"  # active | failed | incompatible
    error: str | None = None
    frontend: str | None = None

    def summary(self) -> dict:
        return {
            "id": self.id, "name": self.name, "version": self.version, "status": self.status,
            "error": self.error, "frontend": self.frontend,
        }


@dataclass
class PluginContext:
    """What a plugin may use of the app."""

    app: FastAPI
    manifest: PluginManifest
    _routers: list[APIRouter] = field(default_factory=list)

    @property
    def session(self):
        return self.app.state.session

    def add_router(self, router: APIRouter) -> None:
        """Serve ``router`` under ``/api/plugins/<id>``."""
        self._routers.append(router)

    def add_dataset(
        self, parent, signal, title: str, settings: dict | None = None,
        label: str | None = None, stack: dict | None = None, hidden_processes: list[str] | None = None,
        entry=None, colormap: str | None = None,
    ):
        """A result of the plugin as a dataset under ``parent`` (or, with no
        parent, last in ``entry``). ``label`` makes it an object of its own
        type, which keeps its data kind's processes except ``hidden_processes``;
        ``colormap`` is the one it is first shown in (see
        ``Session.add_plugin_dataset``)."""
        return self.session.add_plugin_dataset(
            parent, signal, self.manifest.id, title, settings, label, stack, hidden_processes, entry, colormap,
        )

    def add_project(self, path: str | Path, project_type: str, label: str | None = None):
        """An empty project of this plugin: a top-level entry for the folder ``path``."""
        return self.session.add_plugin_project(path, self.manifest.id, project_type, label)

    def add_item(self, entry, item: str, label: str, title: str, settings=None, detail=None, parent=None):
        """A tree item that isn't data, e.g. a structure (see ``Session.add_plugin_item``)."""
        return self.session.add_plugin_item(entry, self.manifest.id, item, label, title, settings, detail, parent)

    def save_project(self, entry) -> Path:
        """Write ``entry``'s analysis file, ``<folder>/<folder>.hspy``, keeping
        the views last saved in it."""
        from .project import read_project, save_project

        target = entry.path / f"{entry.path.name}.hspy"
        views = read_project(target)[1] if target.exists() else {}
        return save_project(self.session, target, views, "link", [entry.id])

    def submit_process(
        self,
        kind: str,
        title: str,
        command: Sequence[str],
        on_result: Callable[[dict], dict] | None = None,
        cwd: str | Path | None = None,
    ) -> Job:
        """Run ``command`` as a cancellable job (see ``JobManager.submit_process``)."""
        return self.app.state.jobs.submit_process(
            f"{self.manifest.id}:{kind}", title, command, on_result, cwd
        )


def _discover() -> list[tuple[str, Callable[[], object]]]:
    found: list[tuple[str, Callable[[], object]]] = [
        (ep.name, ep.load) for ep in entry_points(group=ENTRY_POINT_GROUP)
    ]
    for name in filter(None, (m.strip() for m in os.environ.get(ENV_VAR, "").split(","))):
        found.append((name, lambda n=name: importlib.import_module(n)))
    return found


def _load_one(app: FastAPI, name: str, load: Callable[[], object], seen: set[str]) -> PluginRecord:
    try:
        plugin = load()
        manifest = plugin.manifest
        if not _ID.match(manifest.id):
            raise ValueError(f"Invalid plugin id: {manifest.id!r}")
        if manifest.id in seen:
            raise ValueError(f"Plugin id already in use: {manifest.id}")
    except Exception as e:  # a broken plugin must not stop the app
        traceback.print_exc()
        return PluginRecord(id=name, name=name, status="failed", error=str(e) or type(e).__name__)

    seen.add(manifest.id)
    record = PluginRecord(id=manifest.id, name=manifest.name, version=manifest.version)
    if manifest.api_version != API_VERSION:
        record.status = "incompatible"
        record.error = f"Needs plugin API {manifest.api_version}; this app has {API_VERSION}"
        return record
    try:
        ctx = PluginContext(app, manifest)
        plugin.register(ctx)
        index = manifest.frontend / "index.js" if manifest.frontend else None
        if index is not None and not index.is_file():
            raise FileNotFoundError(f"No frontend at {index}")
    except Exception as e:
        traceback.print_exc()
        record.status, record.error = "failed", str(e) or type(e).__name__
        return record

    # Added only now, so a failed plugin leaves nothing behind.
    for router in ctx._routers:
        app.include_router(router, prefix=f"/api/plugins/{manifest.id}", tags=[f"plugin:{manifest.id}"])
    if index is not None:
        app.mount(f"/plugins/{manifest.id}", _Revalidated(directory=manifest.frontend), name=f"plugin-{manifest.id}")
        record.frontend = f"/plugins/{manifest.id}/index.js"
    return record


class _Revalidated(StaticFiles):
    """A plugin's frontend: its entry keeps one name across builds, so browsers
    must check for a new one on each load (unchanged files are a 304)."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


def load_plugins(app: FastAPI, plugins: Iterable[object] | None = None) -> list[PluginRecord]:
    """Load ``plugins`` (default: those installed). Call before the catch-all
    frontend route is added, which would otherwise answer their paths."""
    sources = _discover() if plugins is None else [
        (getattr(getattr(p, "manifest", None), "id", repr(p)), (lambda x=p: x)) for p in plugins
    ]
    seen: set[str] = set()
    records = [_load_one(app, name, load, seen) for name, load in sources]
    app.state.plugins = records

    @app.get("/api/plugins", tags=["meta"])
    def list_plugins() -> list[dict]:
        return [r.summary() for r in app.state.plugins]

    return records
