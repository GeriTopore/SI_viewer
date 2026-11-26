"""``si-viewer remote HOST --install``: install SI Viewer there, or update it.

The code here is what is installed, so both ends run the same version:

- here, wheels are built of the app (with its built frontend), and of the
  local packages it uses (kemstem, rigidRegistration) from their
  checkouts; a conda environment file is made from ``environment.yml``'s
  conda packages (its ``pip: -e ../…`` lines are this machine's);
- one SSH call sends them (a tar stream) with ``install.sh``, which there:
  installs Miniforge if there is none (batch, shell start-up files left
  alone), creates the environment or updates it when its packages
  changed, installs the wheels (their dependencies first, then the wheels
  themselves again, as the version number doesn't change between builds),
  and records what was installed in ``~/.si-viewer/installed.json``.
"""

from __future__ import annotations

import importlib.util
import io
import json
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

DEFAULT_PREFIX = "~/miniforge3"
ENV_NAME = "si-viewer"
INSTALL_DIR = "~/.si-viewer/install"
# Local packages the app imports, built from their checkouts when found.
LOCAL_PACKAGES = ("kemstem", "rigidregistration")


def remote_path(path: str) -> str:
    """``~/x`` as the remote shell expands it inside double quotes."""
    return "$HOME" + path[1:] if path.startswith("~") else path


def app_root() -> Path:
    """The checkout this app runs from (its ``pyproject.toml``)."""
    root = Path(__file__).resolve().parents[2]
    if not (root / "pyproject.toml").is_file():
        raise RuntimeError(f"Installing elsewhere needs the app's source checkout (not found at {root})")
    return root


def package_root(name: str) -> Path | None:
    """The source checkout of installed package ``name`` (where its
    ``pyproject.toml`` or ``setup.py`` is), if it is one."""
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.origin:
        return None
    for parent in Path(spec.origin).resolve().parents:
        if (parent / "pyproject.toml").is_file() or (parent / "setup.py").is_file():
            return parent
        if parent.name in ("site-packages", "dist-packages"):
            return None
    return None


def conda_environment(env_file: Path, name: str = ENV_NAME) -> str:
    """``env_file`` (a conda ``environment.yml``) with only its conda
    packages, named ``name``: the YAML text."""
    import yaml

    spec = yaml.safe_load(env_file.read_text())
    deps = [d for d in spec.get("dependencies", []) if isinstance(d, str)]
    return yaml.safe_dump({"name": name, "channels": spec.get("channels", ["conda-forge"]), "dependencies": deps},
                          sort_keys=False)


def git_info(root: Path) -> dict:
    def git(*args):
        r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None

    return {"commit": git("rev-parse", "--short", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "changed": bool(git("status", "--porcelain", "--untracked-files=no"))}


def _backend_here(src: Path) -> bool:
    """Whether ``src``'s build backend (``pyproject.toml``) can be imported here."""
    try:
        import tomllib

        backend = tomllib.loads((src / "pyproject.toml").read_text()).get("build-system", {}).get("build-backend")
    except (OSError, ValueError):
        return False
    if not backend:
        return False
    try:
        return importlib.util.find_spec(backend.split(".")[0]) is not None and importlib.util.find_spec(backend.split(":")[0]) is not None
    except ModuleNotFoundError:
        return False


def build_wheels(out_dir: Path, say=print) -> list[Path]:
    """Wheels of the app (its frontend must be built) and of the local
    packages found, in ``out_dir``."""
    root = app_root()
    if not (root / "backend" / "si_viewer" / "static" / "index.html").is_file():
        raise RuntimeError(f"Build the frontend first: npm --prefix {root / 'frontend'} run build")
    sources = [root] + [p for p in (package_root(n) for n in LOCAL_PACKAGES) if p is not None]
    for src in sources:
        say(f"Building a wheel of {src.name}…")
        # Without pip's fresh build environment (seconds each) when the
        # package's build backend is here already.
        no_iso = ["--no-build-isolation"] if _backend_here(src) else []
        r = subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "-q", *no_iso, "-w", str(out_dir), str(src)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"Couldn't build {src.name}:\n{(r.stderr or r.stdout)[-2000:]}")
    return sorted(out_dir.glob("*.whl"))


def install_script(prefix: str, env: str = ENV_NAME) -> str:
    """The script run there (``install.sh``), from the folder it was sent to."""
    return f"""#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
PREFIX="{remote_path(prefix)}"
ENV={shlex.quote(env)}
say() {{ echo "[si-viewer install] $*"; }}
sha() {{ if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }}

if [ ! -x "$PREFIX/bin/conda" ]; then
  os=$(uname); [ "$os" = Darwin ] && os=MacOSX
  url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$os-$(uname -m).sh"
  say "Installing Miniforge in $PREFIX"
  if command -v curl >/dev/null; then curl -fsSL -o "$DIR/miniforge.sh" "$url"; else wget -q -O "$DIR/miniforge.sh" "$url"; fi
  bash "$DIR/miniforge.sh" -b -p "$PREFIX" > "$DIR/miniforge.log"
  rm -f "$DIR/miniforge.sh"
fi
MAMBA="$PREFIX/bin/mamba"; [ -x "$MAMBA" ] || MAMBA="$PREFIX/bin/conda"
PY="$PREFIX/envs/$ENV/bin/python"
STAMP="$PREFIX/envs/$ENV/.si-viewer-environment"
WANT=$(sha "$DIR/environment.yml")
if [ ! -x "$PY" ]; then
  say "Creating the conda environment $ENV (several minutes)"
  "$MAMBA" env create -y -q -n "$ENV" -f "$DIR/environment.yml"
elif [ "$(cat "$STAMP" 2>/dev/null)" != "$WANT" ]; then
  say "Updating the conda environment $ENV"
  "$MAMBA" env update -q -n "$ENV" -f "$DIR/environment.yml"
else
  say "Conda environment $ENV up to date"
fi
echo "$WANT" > "$STAMP"
say "Installing SI Viewer and its packages"
"$PY" -m pip install -q --disable-pip-version-check "$DIR"/wheels/*.whl
"$PY" -m pip install -q --disable-pip-version-check --force-reinstall --no-deps "$DIR"/wheels/*.whl
cp "$DIR/build.json" "$HOME/.si-viewer/installed.json"
"$PY" -c "import si_viewer.app, importlib.metadata as m; print('[si-viewer install] Installed si-viewer', m.version('si-viewer'))"
say "Run it with: $PREFIX/envs/$ENV/bin/si-viewer"
"""


def remote_command(prefix: str, env: str = ENV_NAME) -> str:
    """The installed ``si-viewer`` there, for ``si-viewer remote --command``."""
    return f"{prefix.rstrip('/')}/envs/{env}/bin/si-viewer"


def bundle(prefix: str, say=print) -> bytes:
    """The gzipped tar sent there: ``wheels/``, ``environment.yml``,
    ``install.sh``, ``build.json``."""
    root = app_root()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        wheels = build_wheels(tmp / "wheels", say)
        files = {
            "environment.yml": conda_environment(root / "environment.yml").encode(),
            "install.sh": install_script(prefix).encode(),
            "build.json": json.dumps({**git_info(root), "built": time.strftime("%Y-%m-%d %H:%M:%S"),
                                      "wheels": [w.name for w in wheels]}, indent=1).encode(),
        }
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for name, data in files.items():
                info = tarfile.TarInfo(name)
                info.size, info.mode = len(data), 0o755 if name.endswith(".sh") else 0o644
                tar.addfile(info, io.BytesIO(data))
            for w in wheels:
                tar.add(w, arcname=f"wheels/{w.name}")
        return buf.getvalue()


def install(host: str, ssh: list[str], prefix: str = DEFAULT_PREFIX, say=print, stdout=None) -> None:
    """Install or update SI Viewer on ``host`` (``ssh``: the ssh command and
    its options), its output shown as it comes (or sent to ``stdout``)."""
    data = bundle(prefix, say)
    say(f"Sending {len(data) / 1e6:.1f} MB to {host} and installing there…")
    d = remote_path(INSTALL_DIR)
    line = f'rm -rf "{d}" && mkdir -p "{d}" && tar xzf - -C "{d}" && bash "{d}/install.sh"'
    r = subprocess.run([*ssh, "-T", host, line], input=data, stdout=stdout, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"Installing on {host} failed (exit {r.returncode})")
