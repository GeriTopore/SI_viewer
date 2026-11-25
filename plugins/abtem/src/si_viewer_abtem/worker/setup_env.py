"""Create abTEM's environment: ``python -I setup_env.py <folder> <spec> <worker>``.

Standard library only. Writes progress as JSON lines. The folder is marked as
ours when made, so a folder that isn't is never replaced, and ``.ready`` is
written last, so an interrupted install doesn't count as installed.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import venv
from pathlib import Path

MARKER = ".si-viewer-abtem-env"
READY = ".ready"
# Share of the bar reached when pip starts installing what it downloaded.
DOWNLOADED = 0.75
INSTALLED = 0.92
# Packages abTEM brings in, roughly, for the bar while they are collected.
EXPECTED_PACKAGES = 60


def emit(**message) -> None:
    print(json.dumps(message), flush=True)


def python_in(folder: Path) -> Path:
    return folder / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def prepare(folder: Path) -> None:
    """An empty folder of ours at ``folder``."""
    if folder.exists() and any(folder.iterdir()):
        if not (folder / MARKER).is_file():
            raise SystemExit(f"{folder} is in use: not replacing it")
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / MARKER).write_text("abTEM environment made by SI Viewer\n")


def install(python: Path, spec: str) -> None:
    process = subprocess.Popen(
        [str(python), "-m", "pip", "install", "--no-input", "--disable-pip-version-check", spec],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
    )
    collected = 0
    problem = None
    for line in process.stdout:
        text = line.strip()
        if text.startswith("Collecting "):
            collected += 1
            fraction = 0.1 + (DOWNLOADED - 0.1) * min(1.0, collected / EXPECTED_PACKAGES)
            emit(progress=fraction, message=f"Downloading {re.split(r'[<>=!~;\[ ]', text.split()[1])[0]}")
        elif text.startswith("Installing collected packages"):
            emit(progress=DOWNLOADED, message="Installing packages")
        elif text.startswith("ERROR"):
            problem = problem or text
    if process.wait() != 0:
        raise SystemExit(problem or f"pip exited with code {process.returncode}")


def verify(python: Path, worker: str) -> dict:
    done = subprocess.run([str(python), "-I", worker, "ping"], capture_output=True, text=True)
    if done.returncode != 0:
        lines = [x for x in done.stderr.splitlines() if x.strip()]
        raise SystemExit(f"abTEM doesn't start: {lines[-1] if lines else done.returncode}")
    for line in reversed(done.stdout.splitlines()):
        try:
            return json.loads(line)["result"]
        except (ValueError, KeyError, TypeError):
            continue
    raise SystemExit("abTEM doesn't start: no reply")


def main(argv: list[str]) -> int:
    folder, spec, worker = Path(argv[0]), argv[1], argv[2]
    emit(progress=0.01, message="Creating the environment")
    prepare(folder)
    try:
        venv.create(folder, with_pip=True, symlinks=os.name != "nt")
        python = python_in(folder)
        emit(progress=0.1, message="Downloading abTEM")
        install(python, spec)
        emit(progress=INSTALLED, message="Starting abTEM")
        versions = verify(python, worker)
        (folder / READY).write_text(json.dumps(versions))
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    emit(progress=1.0, message="Ready", result=versions)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
