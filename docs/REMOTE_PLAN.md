# SI Viewer remotely: a workstation over SSH, or a cluster (CX3)

Plan, 2026-11-26. Nothing built yet.

## 1. Goal

Run the backend where the data and the compute are (a workstation, or a
CX3 compute node at Imperial) and use the app in the laptop's browser, as
if local: open files by their remote paths, process there, view here.

## 2. Why it mostly works already

- The app is a web server (FastAPI + uvicorn) serving the built frontend
  and a REST API on one port (`si-viewer --host --port --no-browser`). A
  browser anywhere that reaches that port is the whole client.
- Files are opened by path through a server-side file browser (`api/fs.py`),
  so remote paths are the normal case.
- The frontend is prebuilt into the package (`backend/si_viewer/static`),
  so the remote machine needs Python only, not Node.
- Images and spectra travel as binary `Float32Array`s; frames can be
  binned server-side (`bin`), and drags already keep one request in flight
  (the newest wanted next), so latency doesn't pile up.
- Jobs run in the server, with progress polled: a closed laptop doesn't
  stop them, and a reload finds the session as it was.

So the first remote use is just: `ssh -L 8765:localhost:8765 host`, then
`si-viewer --no-browser` there, then `http://localhost:8765` here. The work
is making that safe, easy, and fast enough, and fitting CX3's rules.

## 3. What CX3 imposes (from the RCS user guide, 2026-11)

- **Access:** `ssh user@login.cx3.hpc.imperial.ac.uk` (also `login-b`,
  `login-c`, `login-ai`, `login-bi`). Only from the college network or
  Unified Access (Zscaler). **Password only: SSH keys are disabled** on the
  login nodes.
- **Login nodes:** no compute-, memory- or I/O-heavy work. So the server
  runs in a **job** on a compute node, not on a login node.
- **Scheduler: PBS Pro.** `qsub -I -l select=1:ncpus=8:mem=64gb -l
  walltime=08:00:00` for interactive jobs (no longer than 8 h); batch jobs
  up to 72 h; the queue is chosen from the size (small ≤ 16 cores / 128 GB,
  medium ≤ 64 / 450 GB, large ≤ 128 / 920 GB, largemem to 4 TB, gpu72 with
  L40S or A100: `ngpus=1:gpu_type=L40S`). Up to 500 CPUs and 12 GPUs per
  user at once.
- **Compute nodes:** reachable by SSH from a login node while a job of
  yours runs there (`qstat -nu $USER -1` gives the node, e.g. `cx3-13-24`).
  Nodes are shared between jobs (and users).
- **Web services:** JupyterHub at `jupyter.cx3.rcs.ic.ac.uk` runs sessions
  as jobs (1, 4 or 8 cores; 8, 32 or 64 GB; 2–8 h; a GPU queue with A40 or
  RTX6000, 4 cores, 32 GB, 8 h; one Jupyter or OnDemand session at a time).
  Open OnDemand (`openondemand.rcs.ic.ac.uk`) offers RStudio only.
- **Storage:** `$HOME` 930 GB / 10 M files (backed up); `$EPHEMERAL` 10 TB,
  deleted after 30 days, not backed up; `$TMPDIR` on the node's own disk,
  200–900 GB, gone at the job's end; project data on RDS / RDF-Active.
- **Software:** `module load miniforge/3; miniforge-setup` once, then
  `eval "$(~/miniforge3/bin/conda shell.bash hook)"`; conda-forge from the
  login nodes. Internet from compute nodes: not stated (assume none).

## 4. Ways to connect

| | Workstation (SSH) | CX3: job + SSH tunnel | CX3: JupyterHub proxy |
|---|---|---|---|
| Where the server runs | the workstation | a compute node, in a PBS job | the Jupyter session's job |
| How the browser reaches it | `ssh -L` to the workstation | `ssh -L` through the login node to the node | `https://jupyter.cx3…/user/<you>/proxy/<port>/` |
| Size | the machine | up to a whole node, GPUs, 72 h | ≤ 8 cores, 64 GB, 8 h |
| Needs | SSH | SSH with a password (no keys), tunnel through login allowed | `jupyter-server-proxy` in the hub's server; the app under a URL prefix |
| Verdict | first | main CX3 route | lightweight CX3 route, if the proxy exists |

## 5. Changes to the app

### 5.1 Safety (needed before anything remote)
- **Token auth**, as Jupyter: the server makes a random token (or takes
  `--token`), prints `http://localhost:PORT/?token=…`; the first visit sets
  a cookie; every API call and the static files need it (403 otherwise).
  On a shared machine any user can reach `127.0.0.1:PORT`, and CX3's nodes
  are shared, so this is not optional there. Local use: still on by default,
  the browser opened with the token.
- Bind to `127.0.0.1` unless told otherwise; on a cluster node, bind to the
  node's own address (reached from the login node) only with a token.
- `--port 0`: pick a free port and report it (stdout and a session file).

### 5.2 Running under a URL prefix
- `--base-url /user/x/proxy/8765/`: the frontend uses relative URLs
  (Vite `base: "./"`, `"api/…"` instead of `"/api/…"`: about 170 places in
  10 files) and FastAPI `root_path`. Needed for the JupyterHub proxy (and
  any reverse proxy); harmless locally.

### 5.3 Fitting the machine
- Threads from the job: `NCPUS` / `OMP_NUM_THREADS` / the cgroup's CPU
  quota → numpy/numba/dask thread counts (today they see the whole node).
- Memory from the job (`PBS` mem or the cgroup limit) → the in-memory
  limits (`SI_IN_MEMORY_LIMIT`, `DIFFRACTION4D_IN_MEMORY_LIMIT`, now fixed at
  4 GB), so a 450 GB node keeps big 4D data in RAM and a 32 GB one doesn't.
- Scratch: big intermediates in `$TMPDIR` when it exists.
- GPU later (cupy where pyxem/abTEM use it).

### 5.4 Files
- File browser roots: `$HOME`, `$EPHEMERAL`, `$TMPDIR`, configured project
  paths (RDS), recent folders; Linux mounts instead of macOS volumes.
- Drag and drop from the laptop can't give a remote path (today it is
  located on the server by name, with Spotlight on macOS). Remote: offer
  an **upload** for small files (to a chosen remote folder), and say so for
  big ones (use the file browser, or rsync/Globus).
- Saving: analyses and exports are written on the server (as now) and can
  also be **downloaded** to the laptop.

### 5.5 Lasting through a job's end
- Show the job's time left in the status bar (from `PBS_WALLTIME` /
  `qstat`); warn at 15 min.
- Autosave the analysis (to `$HOME/.si-viewer/autosave/`) periodically and
  on SIGTERM (PBS sends it before the kill); reopen from there next time.

### 5.6 Speed over the network
- Measure first: the frames and requests of typical use (4D drags, SI
  cursor, stacks) with the browser's throttling (e.g. 20 Mbit/s, 30 ms).
- Then, as needed: compression (gzip/zstd) of responses; frames sent at
  the screen's resolution (server-side `bin`, already there) and refined
  when still; float16 or 16-bit quantised display frames with exact values
  on demand; fewer round trips per drag.

### 5.7 Linux and headless
- Install and run on Linux (CX3 is Rocky/RHEL-like): the conda env
  (`environment.yml`), no macOS-only calls on the main paths (Spotlight is
  already guarded), no browser opened (`--no-browser`).
- Examples that download data: cache them in `$HOME` from a login node, or
  hide them offline. The abTEM plugin's own env (`~/.si-viewer/abtem-env`)
  installed from a login node.

## 6. The launcher (laptop side)

`si-viewer remote …` (later a *Connect…* dialog in the desktop app) does
the SSH steps with the system's `ssh`, so `~/.ssh/config`, ProxyJump and
password prompts work as usual; one SSH connection reused
(`ControlMaster`) so a password is typed once.

- **Workstation:** `si-viewer remote user@host [--env si-viewer]`: start
  the server there in the background (`--port 0 --token …`, in `tmux` or
  `nohup`), read its session file (port, token), forward a local port, open
  the browser. `--list` / `--stop` / reconnect to a running one.
- **CX3:** `si-viewer remote cx3 --cpus 8 --mem 64gb --hours 8 [--gpu L40S]`:
  submit a PBS job (a script that activates the env and starts the server
  on the node), wait until it runs (`qstat`), read the session file from
  `$HOME/.si-viewer/sessions/<jobid>.json` (node, port, token), forward
  `localhost:LOCAL → node:PORT` through the login node, open the browser.
  `--stop` ends the job (`qdel`).

## 7. Phases

- **R0, by hand (small):** install on CX3 and on a Linux workstation, run
  `si-viewer --no-browser` in an interactive job, tunnel by hand; write the
  steps in the README; note what breaks on Linux. Settles §8's questions.
- **R1, safe and configurable:** token auth, `--port 0` with a session
  file, threads and memory from the job, Linux file-browser roots.
- **R2, workstation launcher:** `si-viewer remote user@host`.
- **R3, CX3 launcher:** the PBS job script, the tunnel through the login
  node, time left, autosave on SIGTERM.
- **R4, speed:** measure, then compression and display-resolution frames.
- **R5, files:** upload for small files, download of saved analyses and
  exports.
- **R6, JupyterHub route:** the URL prefix and the proxy, if CX3's hub has
  `jupyter-server-proxy`.
- **R7, desktop app:** *Connect…* (with the pywebview wrapper).

## 7b. R0 on a workstation: done by hand (2026-11-26)

kspace (`kspace-through-cx3`: SSH to `127.0.0.1:62680`, a port the
`tunnel-cx3` connection to the CX3 login node forwards to kspace:22).
Ubuntu 22.04, 32 cores, 503 GB RAM, RTX 6000 Ada 48 GB, 940 GB free on
/home; shared (another user logged in); internet reaches GitHub,
conda-forge and PyPI.

- Install: Miniforge in `~/miniforge3` (batch, `.bashrc` untouched); the
  public repos cloned to `~/si-viewer/` (`SI_viewer` on `4dstem`, `kemstem`,
  `rigidRegistration`, side by side as `environment.yml` expects); the
  built frontend copied in by rsync (`static/` is not in git); `mamba env
  create -f environment.yml`: 3.3 GB, all imports fine. No Linux problems
  on the paths tried.
- Run: `ssh -L 8791:127.0.0.1:47865 kspace-through-cx3
  '~/miniforge3/envs/si-viewer/bin/si-viewer --no-browser --port 47865'`,
  then `http://localhost:8791`. The whole app works: examples, the 4D
  view, virtual images, the file browser (at `/home/geri`).
- Speed through the CX3 hop: ~30 ms per request; a 256 × 256 float32
  pattern (256 KB) takes 100–240 ms (~2 MB/s), so ~5–8 patterns/s while
  dragging. Responses are uncompressed; gzip shrinks the simulated
  patterns 38× (93 % zeros); real data still to measure.
- **Found:** closing the SSH connection leaves the server running (no
  terminal, no hang-up signal), still listening on the shared machine. The
  launcher must stop it (or run `ssh -t`, or the server exits when its
  parent goes); until token auth, never leave one running.
- **Found:** `api/fs.py`'s note that binding to localhost "exposes nothing"
  is wrong on a shared machine: any user there can reach the port.
- The first key login was refused, then worked once `ssh-copy-id` had run.

## 7c. R1: built (2026-11-26, branch `remote`)

- **Token** (`auth.py`, ASGI middleware `TokenAuth`): every request needs
  the server's token, as a cookie, `Authorization: token …`, or `?token=`.
  A page opened with `?token=` gets the cookie (HttpOnly, SameSite=Strict,
  named from the token's hash, so servers tunnelled to one `localhost`
  keep their own) and a 303 to the same address without it; an API call
  with `?token=` just passes (scripts keep no cookie). Without it: 403, a
  "token needed" page for a browser, JSON for the API. `/favicon.ico` is
  open (browsers fetch it without the cookie). On by default: a random
  token, or `--token` / `$SI_VIEWER_TOKEN`; `--no-token` turns it off (with
  a warning when not on loopback).
- **Free port and session file:** `--port 0` binds any free port; the
  server prints one line, `SI Viewer running at http://host:port/?token=…`,
  and writes `~/.si-viewer/sessions/<pid>.json` (pid, host, port, token,
  url, started, version, cwd; the folder 0700, the file 0600), removed
  when it stops.
- **`--exit-on-eof`:** stops when standard input closes, i.e. when the SSH
  connection that started it ends.
- **`--in-memory GB`:** data up to this size is read into RAM (SI, 4D-STEM,
  py4DSTEM cubes; stacks up to a quarter); default 4 as before.
- Threads per job: left for R3 (a workstation's server should use all its
  cores).
- Tests (`tests/test_remote.py`): the token rules, the cookie and redirect,
  separate cookies per token, the limit, and a real server (`--port 0
  --exit-on-eof`): its line, session file and permissions, 200 with the
  token and 403 without, and closing stdin stops it and removes the file.
  Full suite: 439 passed, 140 skipped.
- On kspace (the backend copied to `~/si-viewer/remote-test`): another
  user's request without the token gets 403 (and a page, "token needed");
  the session file is 0600 in a 0700 folder; in the browser the token goes
  from the address, the cookie isn't readable by the page, the whole app
  works; closing the SSH session stopped the server and removed its file.
- **When merged:** a local server needs its printed address (with the
  token) or `--no-token`; the dev servers in `.claude/launch.json` (fixed
  ports, opened at the bare address) will need `--no-token` or
  `SI_VIEWER_TOKEN` and the token in the tab's address.
- A server killed outright (SIGKILL, e.g. stopped from the preview pane)
  never removes its session file: a new server deletes the files of
  servers no longer running before writing its own (not on Windows, where
  `os.kill(pid, 0)` sends Ctrl-C). (2026-11-27)

## 7d. R2: built (2026-11-26, branch `remote`)

`si-viewer remote HOST [files…]` (`remote.py`; the `si-viewer` command is
now `cli.py`, light, so the launcher starts in 0.05 s without importing
HyperSpy; `app.main` also passes `remote` on, for installs still pointing
there):

1. `ssh -o ControlMaster=auto -o ControlPath=/tmp/siv-<pid> -T HOST 'exec
   env <command> --no-browser --port 0 --exit-on-eof …'`: the server there,
   and the shared connection's master (password prompts, if any, in the
   terminal as usual). `<command>`: `--command`, `$SI_VIEWER_REMOTE_COMMAND`,
   else `~/miniforge3/envs/si-viewer/bin/si-viewer`; run through `env` so it
   may start with `VAR=…`.
2. The address line read from its output (the rest to
   `~/.si-viewer/remote-<host>.log`); a free local port (or `--local-port`)
   forwarded through the same connection (`ssh -O forward -L`), no second
   login.
3. The browser opened at `http://localhost:<local>/?token=…` (or printed,
   `--no-browser`).
4. Ctrl-C, SIGTERM, SIGHUP or the connection dropping: the server's stdin
   closes, it stops; the master, its forward and socket go with it. A
   server that fails to start is reported with its last lines.

Options: `--in-memory GB` (passed on), `--local-port`, `--no-browser`.
Tests: the command line and address parsing, a full start / request /
stop through a stand-in `ssh`, a failing start. On kspace: one command
started it (port chosen there, forwarded to a free local port), the app
worked in the browser, Ctrl-C left no server, session file, forward or
socket.

**Installing and updating there (built 2026-11-26):** `si-viewer remote
HOST --install` (install or update, then start) or `--install-only`;
`--conda-prefix` (default `~/miniforge3`; the server command follows it).
`remote_install.py`:

- Here: wheels of the app (with its built frontend; refused if not built)
  and of the local packages it uses, kemstem and rigidRegistration, from
  their checkouts (found by import), so both ends run the same code; built
  without pip's fresh build environment where the backend is here already
  (5 s for all three, from ~45 s). A conda file with only
  `environment.yml`'s conda packages (its `pip: -e ../…` lines are this
  machine's), and `build.json` (commit, branch, local changes, time).
- One SSH call: a tar stream (2.3 MB) into `~/.si-viewer/install/`, then
  `install.sh` there: Miniforge if missing (batch; shell start-up files
  left alone; Linux or macOS, x86_64 or arm64), the environment created,
  or updated only when its packages changed (a hash kept in the env), the
  wheels installed (with their dependencies, then forced, as the version
  number doesn't change between builds), `~/.si-viewer/installed.json`
  written.
- Tests: the environment file, `install.sh` against stand-in conda, mamba
  and python (create, then "up to date", then update when packages change),
  and a real bundle (the frontend inside the app's wheel).
- On kspace, into a fresh prefix (`~/siv-test-mf`): Miniforge, the
  environment and the app in 46 s; again: "up to date", the wheels
  reinstalled, the server started from the installed command, the app
  worked. kemstem's two examples are hidden (their images are in kemstem's
  git checkout, not its wheel).

Still to do for the workstation:
- Per-host settings (command, in-memory, local port) in
  `~/.si-viewer/hosts.json`, so `si-viewer remote kspace` needs no flags.
- Keep running after a disconnect and reconnect later (in `tmux`, found
  again by its session file), for long jobs.
- A *Connect…* dialog in the app (R7).

## 8. To find out (on CX3, in R0)

1. Does `ssh -L` through a login node to a compute node's port work (TCP
   forwarding allowed), and does SSH to the node itself need a password?
2. Any MFA prompt on SSH from Unified Access?
3. Do compute nodes reach the internet (pip, example data)?
4. Is `jupyter-server-proxy` in the JupyterHub server's environment, and
   can a user add it?
5. The project storage path (RDS / RDF-Active) the data will be on, and
   typical data sizes (they set the default job size).
6. The workstation: its OS, and whether it's shared with other users.

## 9. Decisions for the user

1. Which first: a workstation over SSH (simplest), or CX3?
2. On CX3: the job + SSH tunnel (full size, needs a password per connect),
   the JupyterHub proxy (no SSH, ≤ 8 cores / 64 GB / 8 h), or both?
3. Token auth on for local use too (one less mode to test), or only when
   not on `127.0.0.1`?
