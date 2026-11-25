"""abTEM's own environment: where it is, whether it works, and installing it."""

from __future__ import annotations

import collections
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

ABTEM_VERSION = "1.0.11"
ENV_VAR = "SI_VIEWER_ABTEM_ENV"
DEFAULT_ENV = Path.home() / ".si-viewer" / "abtem-env"
READY = ".ready"
PING_TIMEOUT = 120
WORKER_DIR = Path(__file__).parent / "worker"
WORKER = WORKER_DIR / "main.py"
SETUP = WORKER_DIR / "setup_env.py"
INSTALL_KIND = "engine-install"

# The last check of the environment: (.ready's modification time, result).
_checked: dict[Path, tuple[float, dict]] = {}


class ServiceError(RuntimeError):
    """The long-running worker failed to answer, or answered with an error."""


class Service:
    """A long-running worker (``main.py serve``) answering quick requests one
    at a time, for previews that must not wait for abTEM to import. Started on
    the first request, again if it stops or the environment changes."""

    START_TIMEOUT = 120
    CALL_TIMEOUT = 30

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._python: Path | None = None
        self._lines: queue.Queue | None = None
        self._errors: collections.deque = collections.deque(maxlen=20)
        self._lock = threading.Lock()

    @staticmethod
    def _read(stream, sink, ends: bool) -> None:
        for line in stream:
            sink(line)
        if ends:
            sink(None)  # the worker stopped

    def _next(self, timeout: float) -> dict:
        """The next JSON message from the worker."""
        while True:
            try:
                line = self._lines.get(timeout=timeout)
            except queue.Empty:
                self.close()
                raise ServiceError("abTEM didn't answer in time") from None
            if line is None:
                errors = [x for x in self._errors if x.strip()]
                self.close()
                raise ServiceError(f"abTEM stopped: {errors[-1].strip() if errors else 'no reason given'}")
            try:
                return json.loads(line)
            except ValueError:
                continue  # not a message (e.g. a library printing)

    def _start(self) -> None:
        python = python_path()
        self._proc = subprocess.Popen(
            [str(python), "-I", str(WORKER), "serve"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        self._python = python
        self._lines = queue.Queue()
        self._errors.clear()
        threading.Thread(target=self._read, args=(self._proc.stdout, self._lines.put, True), daemon=True).start()
        threading.Thread(target=self._read, args=(self._proc.stderr, self._errors.append, False), daemon=True).start()
        if not self._next(self.START_TIMEOUT).get("ready"):
            self.close()
            raise ServiceError("abTEM didn't start")

    def call(self, command: str, request: dict, timeout: float | None = None) -> dict:
        """The worker's answer to ``command`` (given ``timeout`` seconds, else
        ``CALL_TIMEOUT``). ServiceError if it fails."""
        with self._lock:
            if self._proc is None or self._proc.poll() is not None or self._python != python_path():
                self.close()
                try:
                    self._start()
                except OSError as e:
                    raise ServiceError(f"abTEM doesn't start: {e}") from None
            try:
                self._proc.stdin.write(json.dumps({"command": command, "request": request}) + "\n")
                self._proc.stdin.flush()
            except OSError:
                self.close()
                raise ServiceError("abTEM stopped") from None
            message = self._next(timeout or self.CALL_TIMEOUT)
            if "error" in message:
                raise ServiceError(message["error"])
            return message["result"]

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()


def env_path() -> Path:
    return Path(os.environ.get(ENV_VAR) or DEFAULT_ENV).expanduser()


def python_path(env: Path | None = None) -> Path:
    env = env or env_path()
    return env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def setup_command() -> list[str]:
    return [sys.executable, "-I", str(SETUP), str(env_path()), f"abtem=={ABTEM_VERSION}", str(WORKER)]


def _ping(python: Path) -> dict:
    """The versions the worker reports. RuntimeError if it can't start."""
    try:
        done = subprocess.run(
            [str(python), "-I", str(WORKER), "ping"], capture_output=True, text=True, timeout=PING_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"abTEM doesn't start: {e}") from None
    if done.returncode != 0:
        lines = [x for x in done.stderr.splitlines() if x.strip()]
        raise RuntimeError(f"abTEM doesn't start: {lines[-1] if lines else done.returncode}")
    for line in reversed(done.stdout.splitlines()):
        try:
            return json.loads(line)["result"]
        except (ValueError, KeyError, TypeError):
            continue
    raise RuntimeError("abTEM doesn't start: no reply")


def status() -> dict:
    """``installed``, ``versions``, ``error`` (installed but not working),
    ``path`` and ``version`` (the one that would be installed). The worker is
    started once per install, as importing abTEM takes seconds."""
    env = env_path()
    out = {"installed": False, "versions": None, "error": None, "path": str(env), "version": ABTEM_VERSION}
    ready = env / READY
    python = python_path(env)
    if not ready.is_file() or not python.exists():
        return out
    stamp = ready.stat().st_mtime
    cached = _checked.get(env)
    if cached is None or cached[0] != stamp:
        try:
            result = {"versions": _ping(python), "error": None}
        except RuntimeError as e:
            result = {"versions": None, "error": str(e)}
        cached = (stamp, result)
        _checked[env] = cached
    return {**out, "installed": cached[1]["error"] is None, **cached[1]}
