"""``si-viewer remote HOST``: SI Viewer on another machine, used here.

Over the system's ``ssh`` (so ``~/.ssh/config``, keys, jump hosts and
password prompts work as they do in a terminal):

1. One SSH connection runs the server there on a free port, stopping when
   the connection ends (``--port 0 --exit-on-eof``), and is the master of
   a shared connection (``ControlMaster``), so nothing else asks to log in.
2. The server's address (``SI Viewer running at http://127.0.0.1:PORT/?token=…``)
   is read from it; a local port is forwarded to that port through the
   same connection (``ssh -O forward``).
3. The browser opens at the local address with the token.

Ctrl-C (or the connection dropping) ends it all: the server's standard
input closes and it stops; the shared connection and its forward go with
the SSH process. The server's output is kept in
``~/.si-viewer/remote-<host>.log``.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from pathlib import Path
from urllib.parse import urlsplit

# The server's command there, unless ``--command``: the conda environment's
# entry point, as an install with Miniforge in the home folder makes it.
DEFAULT_COMMAND = "~/miniforge3/envs/si-viewer/bin/si-viewer"
READY = re.compile(r"SI Viewer running at (http://\S+)")
LOG_DIR = Path.home() / ".si-viewer"


def server_command(command: str, in_memory: float | None, files: list[str]) -> str:
    """The shell line run there: the server on a free port, stopping with the
    connection (``exec``: the SSH session's process is the server's)."""
    extra = ["--no-browser", "--port", "0", "--exit-on-eof"]
    if in_memory is not None:
        extra += ["--in-memory", f"{in_memory:g}"]
    # ``command`` is the user's (it may hold ``~`` or ``VAR=…``, hence ``env``): not quoted.
    return f"exec env {command} " + " ".join(shlex.quote(a) for a in extra + files)


def parse_ready(line: str) -> tuple[str, int, str | None] | None:
    """(host, port, token) from the server's address line, else None."""
    m = READY.search(line)
    if not m:
        return None
    url = urlsplit(m.group(1))
    token = next((v for k, _, v in (p.partition("=") for p in url.query.split("&")) if k == "token"), None)
    return url.hostname or "127.0.0.1", url.port or 80, token


def free_local_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def local_url(port: int, token: str | None) -> str:
    return f"http://localhost:{port}/" + (f"?token={token}" if token else "")


class Remote:
    """One remote server and its forward, for as long as ``run`` runs."""

    def __init__(self, host: str, command: str, *, local_port: int = 0, in_memory: float | None = None,
                 files: list[str] | None = None, ssh: str = "ssh", open_browser: bool = True,
                 timeout: float = 120.0, log=None, out=sys.stdout):
        self.host = host
        self.ssh = ssh
        self.line = server_command(command, in_memory, files or [])
        self.local_port = local_port
        self.open_browser = open_browser
        self.timeout = timeout
        self.out = out
        # Short (a socket path has ~100 characters at most) and ours alone.
        self.control = os.path.join(tempfile.gettempdir() if sys.platform == "win32" else "/tmp", f"siv-{os.getpid()}")
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = LOG_DIR / f"remote-{re.sub(r'[^A-Za-z0-9._-]', '_', host)}.log"
        self.log = log or open(self.log_path, "a", buffering=1)
        self.proc: subprocess.Popen | None = None
        self.url: str | None = None
        self.remote_port: int | None = None

    def _ssh(self, *args: str) -> list[str]:
        return [self.ssh, "-o", f"ControlPath={self.control}", *args]

    def say(self, text: str) -> None:
        print(text, file=self.out, flush=True)

    def start(self) -> str:
        """Start the server there, forward a port to it; the local address."""
        self.say(f"Starting SI Viewer on {self.host}…")
        self.proc = subprocess.Popen(
            self._ssh("-o", "ControlMaster=auto", "-o", "ControlPersist=no", "-o", "ServerAliveInterval=30",
                      "-T", self.host, self.line),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        found: list[tuple[str, int, str | None]] = []
        tail: list[str] = []

        def read() -> None:
            # The server's output: to the log; its address line found.
            for line in self.proc.stdout:
                self.log.write(line)
                tail.append(line.rstrip())
                del tail[:-20]
                if not found and (hit := parse_ready(line)):
                    found.append(hit)

        threading.Thread(target=read, daemon=True, name="remote-output").start()
        deadline = time.time() + self.timeout
        while not found:
            if self.proc.poll() is not None:
                time.sleep(0.2)  # the last lines
                raise RuntimeError(f"The server on {self.host} stopped before starting:\n" + "\n".join(tail[-10:]))
            if time.time() > deadline:
                raise RuntimeError(f"The server on {self.host} didn't start within {self.timeout:.0f} s "
                                   f"(output in {self.log_path})")
            time.sleep(0.1)
        rhost, rport, token = found[0]
        self.remote_port = rport
        lport = self.local_port or free_local_port()
        forward = subprocess.run(
            self._ssh("-o", "ExitOnForwardFailure=yes", "-O", "forward", "-L", f"{lport}:{rhost}:{rport}", self.host),
            capture_output=True, text=True, timeout=30,
        )
        if forward.returncode != 0:
            raise RuntimeError(f"Couldn't forward local port {lport}: {forward.stderr.strip()}")
        self.url = local_url(lport, token)
        self.say(f"SI Viewer on {self.host} (port {rport} there) at {self.url}")
        self.say("Ctrl-C stops it.")
        if self.open_browser:
            webbrowser.open(self.url)
        return self.url

    def wait(self) -> int:
        """Until the server or its connection ends; its exit code."""
        assert self.proc
        return self.proc.wait()

    def stop(self) -> None:
        """Close the server's standard input (it stops), then the connection."""
        if not self.proc or self.proc.poll() is not None:
            return
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="si-viewer remote",
        description="Run SI Viewer on another machine over SSH and open it here. Ctrl-C stops it.",
    )
    parser.add_argument("host", help="the SSH host (as for ssh: an alias from ~/.ssh/config, or user@host)")
    parser.add_argument("files", nargs="*", help="files there to open at startup")
    parser.add_argument("--command", default=os.environ.get("SI_VIEWER_REMOTE_COMMAND"),
                        help=f"the si-viewer command there (default: the conda prefix's, {DEFAULT_COMMAND})")
    parser.add_argument("--install", action="store_true",
                        help="install SI Viewer there from this checkout, or update it, then start it")
    parser.add_argument("--install-only", action="store_true", help="install or update, then stop")
    parser.add_argument("--conda-prefix", default="~/miniforge3",
                        help="where Miniforge is (or goes) there (default: ~/miniforge3)")
    parser.add_argument("--local-port", type=int, default=0, help="the port here (default: any free one)")
    parser.add_argument("--in-memory", type=float, metavar="GB", help="passed to the server: data up to this size in RAM")
    parser.add_argument("--no-browser", action="store_true", help="print the address instead of opening it")
    parser.add_argument("--ssh", default="ssh", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    from .remote_install import install, remote_command

    if args.install or args.install_only:
        try:
            install(args.host, [args.ssh], args.conda_prefix)
        except (RuntimeError, OSError) as e:
            print(e, file=sys.stderr)
            sys.exit(1)
        if args.install_only:
            return
    command = args.command or remote_command(args.conda_prefix)
    remote = Remote(args.host, command, local_port=args.local_port, in_memory=args.in_memory,
                    files=args.files, ssh=args.ssh, open_browser=not args.no_browser)
    # Ctrl-C (or a terminal closing) stops it cleanly.
    stopping = threading.Event()

    def on_signal(*_):
        stopping.set()
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_signal)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, on_signal)
    code = 0
    try:
        remote.start()
        code = remote.wait()
        if not stopping.is_set():
            remote.say(f"The server on {remote.host} ended (exit {code}); output in {remote.log_path}")
    except KeyboardInterrupt:
        remote.say(f"\nStopping SI Viewer on {remote.host}…")
    except RuntimeError as e:
        remote.say(str(e))
        code = 1
    finally:
        remote.stop()
    sys.exit(code)


if __name__ == "__main__":
    main()
