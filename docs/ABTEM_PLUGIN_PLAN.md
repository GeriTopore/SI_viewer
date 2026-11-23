# abTEM plugin: organisation and rollout (draft, 2026-11-23)

Status: proposal. Nothing here is built. abTEM has not been installed or run
yet; its API below comes from its docs and PyPI metadata, and Phase 0 tests the
assumptions marked **check**.

## 1. Findings

### abTEM 1.0.11 (released 2026-10-22)

- GPL-3.0-or-later, Python >= 3.11, on PyPI and conda-forge.
- Workflow: ASE `Atoms` → `Potential` (optionally `FrozenPhonons`) →
  `PlaneWave` or `Probe` → `multislice` or `scan` with detectors → measurements
  (`Images`, `DiffractionPatterns`, line profiles, polar measurements).
- Modes: HRTEM, SAED, CBED, PACBED, PED, STEM, 4D-STEM, PRISM, core loss,
  partial coherence, Bloch waves. DFT potentials need GPAW.
- Computation is Dask-based, in batches by probe position and phonon
  configuration. Results save to Zarr.
- GPU: NVIDIA through CuPy. On Apple silicon, an experimental Metal path through
  PyTorch, with fallback to NumPy for unsupported operations.

### What this means for SI Viewer

| Finding | Consequence |
|---|---|
| abTEM requires `zarr>=3.1`. The `si-viewer` env has zarr 2.18.7, and rosettasciio's `zspy` extra requires `zarr<3`. | Installing abTEM into the main env upgrades zarr to 3.x and breaks that extra. SI Viewer's own code doesn't use zarr or `.zspy`, but HyperSpy's reader would be affected. |
| `JobManager` runs one job at a time (`jobs.py`). | A long simulation in it would block every other process in the app. |
| Jobs cancel only when they next call `job.report`. | A thread inside a Dask chunk can't be stopped promptly. A process can be killed. |
| abTEM adds about 14 packages, including distributed, pandas and ASE. | Base installs should not carry them. |
| Routers, right-hand tabs and processes are hard-coded (`app.py`, `RightPanel.tsx`, `ProcessPanel.tsx`). | The core needs a small plugin host first. |
| The SPA route in `app.py` catches every remaining path. | Plugin routes and assets must be mounted before it. |
| The dataset pane already groups files under an **experiment**. | A simulation project can be the same kind of top-level entry. |
| `FileEntry` assumes a raw file: projects fingerprint and reference `entry.path`. | A simulation project needs its own entry type, with no raw file. |
| Project save keeps a fixed list of dataset fields (`FIELDS` in `project.py`). | Plugin data needs a field of its own, or a save would silently drop it. |
| `Dataset` requires a HyperSpy signal, and `MainView` picks its view by `kind`. | Structures need a non-signal item kind and a registered view. |
| kemstem and mask processes accept any calibrated `image` dataset. | Simulated images get them with no changes, if stored as ordinary images in Å. |
| HyperSpy, exSpy, rosettasciio, kemstem and abTEM are all GPL-3. | The plugin split doesn't avoid the GPL question. The open licence decision covers both. |

## 2. Recommended organisation

### One repository, two installable packages, two environments

```
SI_viewer/
  backend/si_viewer/
    plugins.py              # discovery, manifest, isolation of failures
  frontend/src/plugins/     # plugin host and the API given to plugins
  plugins/abtem/
    pyproject.toml          # distribution: si-viewer-abtem
    src/si_viewer_abtem/
      plugin.py             # runs in the SI Viewer env: routes, jobs, import
      project.py            # simulation projects
      protocol.py           # request/result schema (standard library only)
      worker/               # runs in the abTEM env; never imports si_viewer
    frontend/               # builds to an ES module inside the package
    tests/
```

- `pip install si-viewer` stays free of abTEM.
- `pip install si-viewer-abtem` adds the thin UI and API part, plus ASE for
  structures, with no abTEM dependency.
- `si-viewer plugin setup abtem` creates a dedicated env with a pinned abTEM and
  records its path.

**Why not a separate repository yet.** The plugin API will change often while
the first plugin is built. In one repo, a core change and its plugin change are
one commit and share one CI run. The package boundary is enforced from day one
(the plugin imports only the public plugin API), so `git subtree split` can move
it to `si-viewer-abtem` with its history once the API is stable or if someone
else starts maintaining it.

### Run abTEM in a worker process

```
Browser ── REST ── SI Viewer process ── JSON lines (stdin/stdout) ── abTEM worker
                        ▲                                                │
                        └────────── npy + JSON manifest in the run folder ┘
```

- The dependency clash disappears, and each side can upgrade alone.
- Cancel kills the process. Crashes and out-of-memory stay out of the viewer.
- Simulations run beside normal jobs without blocking them.
- The same protocol can later run over SSH, which fits the planned remote work.
- Results are `.npy` plus axes and settings in JSON, so 4D data can be
  memory-mapped and neither side needs the other's Zarr version.

### Plugin API added to the core

Backend, through the entry-point group `si_viewer.plugins`:

- `manifest`: id, name, version, required core API version, frontend entry.
- `register(ctx)`: adds a router under `/api/plugins/<id>`.
- `ctx.add_project_type(...)`: a top-level entry in the dataset pane, with its
  own create, open, save and close.
- `ctx.add_item_kind(...)`: tree items that aren't HyperSpy signals.
- `ctx.add_file_type(".cif", handler)`: file types a plugin opens.
- `ctx.add_dataset(signal, origin, parent, settings)`: results as normal datasets.
- `ctx.submit_external_job(...)`: a job lane for worker processes, shown in the
  existing jobs flap with progress and cancel.
- `Dataset.plugin`: plugin-owned settings, saved in analysis files.
- `GET /api/plugins`: what is installed, and its status.
- A plugin that fails to load is logged and skipped. The base app still starts.

Frontend:

- The plugin's ES module is loaded with a dynamic `import()`.
- React and Zustand are shared through an import map, so there is one copy.
- It registers contributions: a *New simulation project* command, views for its
  item kinds, bottom tabs and processes.
- It reuses exported viewer parts and CSS classes, per the UI conventions.

## 3. Simulation projects

With the plugin installed, the user can create a **simulation project**. It works
like any other data in the app: same three panes, nothing separate.

### On disk

```
STO simulation/
  STO simulation.hspy       # the normal analysis file: tree, views, results
  structures/SrTiO3.cif     # as loaded
  runs/<id>/
    settings.json           # everything needed to repeat the run
    manifest.json           # outputs, axes, abTEM and library versions
    *.npy
```

- The folder is needed because runs can be large and are written while they
  compute. The analysis file is rewritten whole on each save.
- A finished run is on disk at once, so it survives a server restart even if the
  project wasn't saved.
- The `.hspy` still opens in HyperSpy.
- **Without the plugin**, the base app opens the project's images and analyses.
  Structures and runs show as unavailable, and are kept unchanged on save.

### Dataset pane (left)

```
▾ STO simulation
  ▾ SrTiO3                       structure
    ▾ [001] 10×10×20             oriented model
      ▾ HAADF, 300 kV            run
          HAADF                  image
          ▾ Columns              existing kemstem items
              Sublattice A
      ▸ HRTEM defocus series     run
```

Click, shift-click, rename, delete and undo behave as for other datasets.

### Middle pane

- A new project shows nothing.
- Loading a structure brings up the **Structure** tab. With no image in view, it
  fills the middle column.
- With a simulated image in view, Structure is a normal bottom tab beside
  Histogram and Line profile, so the image and the model can be compared or split.

### Process tab (right)

No separate Simulate tab. The plugin adds a **Simulation** section to the Process
tab, and the highlighted item decides what is offered, as it does now:

| Highlighted | Processes |
|---|---|
| Empty project | Load structure |
| Structure | Orientation and supercell |
| Oriented model | Potential, then HRTEM, diffraction, STEM, 4D-STEM |
| Run | Its settings, re-run, change one setting as a new run |
| Simulated image | Source size, noise and other post-processing, plus **all existing image processes**: filters, masks, Fourier filtering, atomic columns, sublattices, neighbourhoods, lattice displacements |
| 4D result | Virtual detectors, DPC/iDPC |

Tunable steps stay guided: settings, a live preview in the viewer, and the heavy
part as a cancellable job with its estimated memory and size shown first.

## 4. Structure viewer (.cif)

A **Structure** tab in the bottom pane of the middle column, with the usual
split, drag and focus behaviour.

### Display

- Three.js with instanced spheres, loaded only when the tab first opens, so
  large supercells stay interactive. It is not added to the base app.
- **Orthographic by default**, so atomic columns line up down a zone axis.
  Perspective is a toggle.
- Atoms by element colour and radius, unit-cell box, a/b/c axis triad, element
  legend with show/hide, optional bonds.
- Repeats n × m × k, atom size, background following the app theme.
- Hover: element, fractional and Cartesian position, occupancy.
- Mixed sites (for example A-site solid solutions) drawn as split-colour atoms.
- PNG export.

### Zone axes

- Buttons for [100], [010], [001], [110], [101], [011], [111].
- A typed [uvw] field. Hexagonal cells also accept four-index [uvtw].
- The view direction is computed through the real lattice, u**a** + v**b** +
  w**c**, so non-orthogonal cells are correct.
- The in-plane direction is settable, with a readout of the current nearest
  low-index zone axis and the angle from it.
- Shortcuts work only while the tab has focus, because keys such as `1`, `C` and
  the arrows are already global: `Z` jumps to the [uvw] field, `A`/`B`/`C` look
  down each axis, arrows tilt by 1°, Shift+arrows by 5°, `R` resets.

### Core and data

- `BottomTab` is a closed type today, so it gains a `plugin:<id>:<tab>` form, as
  `eels:<id>` already does for linked spectra.
- Parsing uses ASE's CIF reader, including symmetry expansion. No custom parser.
- The original CIF is kept in the project's `structures/` folder.

### Link to simulation

- *Create model* builds the oriented, orthogonal supercell for the chosen zone
  axis with ASE/abTEM (`orthogonalize_cell`) and shows its strain, since some
  orientations need a large cell or a small distortion to be periodic. The model
  appears under the structure in the dataset pane.
- **Check:** how mixed occupancy reaches the potential. The viewer can show it,
  but a simulation needs an explicit choice, such as a random supercell.

## 5. Rollout

Each phase ends with something usable.

### Phase 0: feasibility and plugin host

- Install abTEM in a throwaway env. **Check:** a small run on this Mac, Dask
  progress reporting, kill-based cancel, thread and memory settings, Metal.
- Build the plugin host, then a minimal test plugin (one item kind, one process,
  one bottom tab, one worker job).
- **Done when:** the full test suite passes with no plugin installed, and the
  test plugin's job reports progress, cancels and returns a dataset.

### Phase 1: simulation projects

- *New simulation project*, the folder format, the entry in the dataset pane.
- Save, reopen, close, and opening without the plugin.
- Load a structure as a tree item.
- **Done when:** a project with a structure survives a restart, and a save from
  the base app loses nothing.

### Phase 2: structure viewer

- 2a: the Structure tab, orthographic 3D view, cell, legend, hover.
- 2b: zone-axis buttons, [uvw] and [uvtw], tilt and shortcuts.
- 2c: repeats, bonds, mixed sites, PNG export.
- 2d: *Create model* with the oriented supercell.
- **Done when:** zone axes are checked against known projections on a cubic, a
  hexagonal and a low-symmetry CIF.

### Phase 3: worker and potential

- Worker env setup from the app, potential settings, projected potential
  preview, cost estimate, runs written to the project.

### Phase 4: HRTEM

- Plane wave, exit wave, CTF, image. Defocus and thickness series as stacks.
- First validation against a published or abTEM tutorial result.

### Phase 5: diffraction

- SAED and CBED, with calibrated reciprocal axes, then PACBED.

### Phase 6: STEM imaging and quantification

- Probe, grid scan, annular detectors (BF, ABF, ADF, HAADF), frozen phonons,
  source size and noise as post-processing.
- Run the existing kemstem processes on the results, and check them against the
  model's known atom positions.

### Phase 7: 4D-STEM and DPC

- Pixelated and segmented detectors, memory-mapped 4D results, virtual detectors.
- Segmented output feeds the existing DPC/iDPC process.

### Phase 8: compare with experiment

- Match calibration, compare a simulated and a measured image across workspaces,
  fit thickness and defocus from a simulated series.

### Phase 9: advanced, as needed

- PRISM, core-loss EELS maps (feeding the EELS tools), remote runs over SSH,
  NVIDIA GPU, DFT potentials via GPAW.

## 6. Decisions for the user

1. **Project on disk (decided 2026-11-23):** a folder holding the analysis file,
   structures and runs.
2. **Repository:** one repo with a separate package (recommended), or a separate
   repo now.
3. **Worker env:** a dedicated env (recommended), or abTEM in the main env with
   zarr upgraded.
4. **Structure viewer without abTEM:** the plugin's thin part depends on ASE, so
   CIF viewing works without the abTEM env or a worker (recommended). ASE is not
   part of the zarr clash. The alternative is parsing in the worker, which makes
   every CIF view depend on the full abTEM install.
5. **First simulation mode:** HRTEM is the simplest path. STEM could come first
   since it is closest to the current atomic-resolution work.
6. **Before core changes:** the working tree holds a large uncommitted change
   touching the files the plugin host edits (`app.py`, `ProcessPanel.tsx`), so
   it should be committed first.
7. **Licence:** still open, and it now also covers the plugin.
8. **Test structures:** `examples/` has no `.cif` yet. Phase 2 needs a cubic, a
   hexagonal and a low-symmetry file, ideally including BTFMO or STO.
