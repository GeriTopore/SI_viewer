"""FastAPI application and the ``si-viewer`` command-line entry point."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
import webbrowser
from importlib.metadata import version
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api import (
    atomic, background, crop, curve_fit, datasets, decomposition, deconvolution, dpc, eds, eels, eels_fit, examples, export, files, fourd, fs, orientation, insitu, jobs, masks, plot_export, process,
    project, registration, thickness, zlp,
)
from .auth import TokenAuth, new_token
from .jobs import JobManager
from .session import Session

STATIC_DIR = Path(__file__).parent / "static"
# Running servers, one file each (``<pid>.json``: host, port, token, url),
# for a launcher to find the one it started.
SESSIONS_DIR = Path.home() / ".si-viewer" / "sessions"
TOKEN_ENV = "SI_VIEWER_TOKEN"


def create_app(session: Session | None = None, token: str | None = None) -> FastAPI:
    """The app; with ``token``, every request needs it (``auth.TokenAuth``)."""
    app = FastAPI(title="SI Viewer", version=version("si-viewer"))
    if token:
        app.add_middleware(TokenAuth, token=token)
    app.state.session = session or Session()
    app.state.jobs = JobManager()
    # Drift-correction sessions in progress, by stack id.
    app.state.registrations = {}
    app.include_router(files.router)
    app.include_router(datasets.router)
    app.include_router(fs.router)
    app.include_router(eds.router)
    app.include_router(eels.router)
    app.include_router(background.router)
    app.include_router(thickness.router)
    app.include_router(deconvolution.router)
    app.include_router(zlp.router)
    app.include_router(eels_fit.router)
    app.include_router(curve_fit.router)
    app.include_router(examples.router)
    app.include_router(atomic.router)
    app.include_router(masks.router)
    app.include_router(fourd.router)
    app.include_router(orientation.router)
    app.include_router(jobs.router)
    app.include_router(process.router)
    app.include_router(export.router)
    app.include_router(registration.router)
    app.include_router(insitu.router)
    app.include_router(plot_export.router)
    app.include_router(project.router)
    app.include_router(dpc.router)
    app.include_router(decomposition.router)
    app.include_router(crop.router)

    @app.get("/api/health", tags=["meta"])
    def health() -> dict:
        return {
            "status": "ok",
            "versions": {
                pkg: version(pkg)
                for pkg in ("si-viewer", "hyperspy", "exspy", "rosettasciio")
            },
        }

    # The built frontend (``npm run build``) is served from the same origin.
    if (STATIC_DIR / "index.html").is_file():
        app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

        @app.api_route("/{path:path}", methods=["GET", "HEAD"], include_in_schema=False)
        def spa(path: str) -> FileResponse:
            candidate = STATIC_DIR / path
            if path and candidate.is_file():
                return FileResponse(candidate)
            # Revalidated on every load, so a rebuilt frontend (new asset
            # names) is picked up on reload.
            return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    return app


def _warm_up() -> None:
    """Import HyperSpy/exSpy (a few seconds) before the first file is opened."""
    import exspy  # noqa: F401
    import hyperspy.api  # noqa: F401

    from .io.emd import ensure_velox_lite_support

    ensure_velox_lite_support()


def set_in_memory_limit(nbytes: int) -> None:
    """Data up to ``nbytes`` (spectrum images, 4D-STEM, py4DSTEM cubes) is
    read into RAM when opened, larger stays lazy; stacks up to a quarter."""
    from . import session as session_mod
    from .io import py4dstem
    from .processing import frames

    session_mod.SI_IN_MEMORY_LIMIT = session_mod.DIFFRACTION4D_IN_MEMORY_LIMIT = int(nbytes)
    py4dstem.IN_MEMORY_LIMIT = int(nbytes)
    frames.STACK_IN_MEMORY_LIMIT = int(nbytes) // 4


def _bind(host: str, port: int) -> socket.socket:
    """A listening socket on ``host``:``port`` (0: a free port)."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(128)
    return sock


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _prune_session_files() -> None:
    """Delete the files of servers no longer running (killed, so their
    ``finally`` never ran)."""
    if sys.platform == "win32":  # os.kill(pid, 0) there sends Ctrl-C
        return
    for path in SESSIONS_DIR.glob("*.json"):
        if path.stem.isdigit() and not _alive(int(path.stem)):
            path.unlink(missing_ok=True)


def _write_session_file(info: dict) -> Path:
    """``SESSIONS_DIR/<pid>.json``, readable by its user only (it holds the token)."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(SESSIONS_DIR, 0o700)
    _prune_session_files()
    path = SESSIONS_DIR / f"{info['pid']}.json"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(info, f)
    return path


def _stop_on_eof(server) -> None:
    """Stop ``server`` when standard input closes: started over SSH, when
    the connection ends (without a terminal there is no hang-up signal)."""
    def watch() -> None:
        try:
            while sys.stdin.buffer.read(4096):
                pass
        except (OSError, ValueError):
            pass
        server.should_exit = True

    threading.Thread(target=watch, daemon=True, name="stdin-eof").start()


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    # ``si-viewer remote HOST`` from an install whose command still points here.
    if (sys.argv[1:] if argv is None else argv)[:1] == ["remote"]:
        from .remote import main as remote_main

        return remote_main((sys.argv[1:] if argv is None else argv)[1:])

    parser = argparse.ArgumentParser(prog="si-viewer", description=__doc__)
    parser.add_argument("files", nargs="*", help="files to open at startup")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765, help="0: any free port (reported, and in the session file)")
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    parser.add_argument("--token", default=os.environ.get(TOKEN_ENV),
                        help=f"the token every request needs (default: ${TOKEN_ENV}, else a new random one)")
    parser.add_argument("--no-token", action="store_true",
                        help="no token: anyone who can reach the port can use the app and read your files")
    parser.add_argument("--exit-on-eof", action="store_true",
                        help="stop when standard input closes (e.g. the SSH connection that started it)")
    parser.add_argument("--in-memory", type=float, metavar="GB",
                        help="read data up to this size into RAM when opened (default 4); larger stays on disk")
    args = parser.parse_args(argv)

    if args.in_memory is not None:
        set_in_memory_limit(int(args.in_memory * (1 << 30)))
    token = None if args.no_token else (args.token or new_token())
    if token is None and args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"WARNING: no token on {args.host}: anyone who can reach it can use the app.", file=sys.stderr)

    threading.Thread(target=_warm_up, daemon=True).start()
    session = Session()
    for path in args.files:
        session.open_file(path)
    app = create_app(session, token=token)

    sock = _bind(args.host, args.port)
    port = sock.getsockname()[1]
    shown = f"[{args.host}]" if ":" in args.host else args.host
    url = f"http://{shown}:{port}/" + (f"?token={token}" if token else "")
    info = {"pid": os.getpid(), "host": args.host, "port": port, "token": token, "url": url,
            "started": time.time(), "version": version("si-viewer"), "cwd": os.getcwd()}
    session_file = _write_session_file(info)
    # One line a launcher can wait for (and a person can open).
    print(f"SI Viewer running at {url}", flush=True)

    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    server = uvicorn.Server(uvicorn.Config(app, log_level="info"))
    if args.exit_on_eof:
        _stop_on_eof(server)
    try:
        server.run(sockets=[sock])
    finally:
        session_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
