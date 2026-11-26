"""The ``si-viewer`` command: ``si-viewer [files…]`` runs the app here;
``si-viewer remote HOST`` runs it on another machine over SSH and opens it
here (``remote.py``). Kept light, so the launcher starts at once (the app's
modules import HyperSpy)."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["remote"]:
        from .remote import main as remote_main

        return remote_main(argv[1:])
    from .app import main as app_main

    return app_main(argv)


if __name__ == "__main__":
    main()
