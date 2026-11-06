"""FastAPI application and the ``si-viewer`` command-line entry point."""

from __future__ import annotations

import argparse
import threading
import webbrowser
from importlib.metadata import version
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api import datasets, eds, files, fs
from .session import Session

STATIC_DIR = Path(__file__).parent / "static"


def create_app(session: Session | None = None) -> FastAPI:
    app = FastAPI(title="SI Viewer", version=version("si-viewer"))
    app.state.session = session or Session()
    app.include_router(files.router)
    app.include_router(datasets.router)
    app.include_router(fs.router)
    app.include_router(eds.router)

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
            return FileResponse(STATIC_DIR / "index.html")

    return app


def _warm_up() -> None:
    """Import HyperSpy/exSpy (a few seconds) before the first file is opened."""
    import exspy  # noqa: F401
    import hyperspy.api  # noqa: F401

    from .io.emd import ensure_velox_lite_support

    ensure_velox_lite_support()


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    parser = argparse.ArgumentParser(prog="si-viewer", description=__doc__)
    parser.add_argument("files", nargs="*", help="files to open at startup")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    args = parser.parse_args(argv)

    threading.Thread(target=_warm_up, daemon=True).start()
    session = Session()
    for path in args.files:
        session.open_file(path)
    app = create_app(session)

    if not args.no_browser:
        url = f"http://{args.host}:{args.port}/"
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
