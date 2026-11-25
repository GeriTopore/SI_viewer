# abTEM plugin: organisation and rollout (draft, 2026-11-23)

Status: Phases 0 to 2 built (plugin host, simulation projects, structure viewer
and models). Phases 3 onwards are proposals. Remaining unknowns are marked
**check**.

Interactive session: the `si-viewer-abtem` configuration in `.claude/launch.json`
(http://localhost:8767). It uses `plugins/abtem/.venv`, which adds the plugin and
ASE on top of the `si-viewer` environment and leaves that environment unchanged.
Test project (from 2026-11-24): `examples/abtem/Si simulation/`, from
`examples/abtem/Si.cif` (diamond, a = 5.431 Å), with a [110] 2×2×5 model whose
dumbbells (1.36 Å) show in the projected potential. Tested on 8767 and left open.
The earlier STO project is `examples/abtem/STO simulation/`.
`examples/abtem` stays out of commits.

Plugin tests: `plugins/abtem/.venv/bin/python -m pytest -c plugins/abtem/pyproject.toml --rootdir plugins/abtem plugins/abtem/tests`.

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
- Built in Phase 0:
  - `ctx.add_router(router)`, served under `/api/plugins/<id>`.
  - `ctx.add_dataset(parent, signal, title, settings)`: results as normal
    datasets.
  - `ctx.submit_process(...)`: a job lane for worker processes, shown in the
    existing jobs flap with progress and cancel.
  - `Dataset.plugin`: plugin-owned settings, saved in analysis files.
  - `GET /api/plugins`: what is installed, and its status.
  - A plugin that fails to load is logged and skipped. The base app still starts.
  - `SI_VIEWER_PLUGINS`: extra plugin modules, for development.
- Built in Phase 1:
  - `ctx.add_project(path, type, label)`: a top-level entry in the dataset pane
    for a folder, saved and reopened by the normal analysis code.
  - `ctx.add_item(entry, item, label, title, ...)`: tree items that aren't data
    (`DatasetKind.ITEM`, with a 1×1 placeholder signal so the base app keeps them).
  - `ctx.save_project(entry)`: writes `<folder>/<folder>.hspy`, keeping its views.

Frontend:

- The plugin's ES module is loaded with a dynamic `import()`, and its
  `activate(host)` receives the app's React and the plugin API, so there is one
  copy of React.
- Built in Phase 0: processes and bottom tabs.
- Built in Phase 1: top-bar commands, a project's Process panel, item views, and
  picking a file or new folder in the file browser.
- The plugin's frontend is one plain ES module so far. Phase 2 adds a build step
  for Three.js.
- It reuses exported viewer parts and CSS classes, per the UI conventions.

## 3. Simulation projects

With the plugin installed, the user can create a **simulation project**. It works
like any other data in the app: same three panes, nothing separate.

### On disk

```
STO simulation/
  STO simulation.hspy       # the normal analysis file: tree, views, results
  structures/SrTiO3.cif     # as loaded
  models/*.extxyz           # oriented supercells
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

### Phase 0: feasibility and plugin host (built)

- Plugin host, and a test plugin in `backend/tests/plugin_fixture` with a process,
  a bottom tab and a worker job.
- abTEM 1.0.11 tested in a throwaway env on the M3 Max, with SrTiO3 (3×3×5,
  235² grid, 16² probe positions, 4 threads): 4.4 s, 0.44 GB peak.
- What the test showed, for Phase 3 onwards:
  - Detector angles beyond the simulated maximum fail only after the run starts,
    so the app must check them first.
  - Progress is too coarse by default (4 tasks). `max_batch=8` gave 99.
  - A hard kill of the process group stopped a run in about 4 ms.
  - `.npy` output reads correctly in the main env.
  - abTEM's axes are (x, y). SI Viewer's images are (y, x), so results need
    transposing.
- **Check:** the Metal path was not tested (it needs PyTorch).

### Phase 1: simulation projects (built)

- *New simulation…* in the top bar makes the project folder and its entry.
- *Load structure…* reads a `.cif` with ASE, copies it into `structures/`, adds it
  to the tree and saves. The Structure tab shows its formula, space group, cell
  and composition, including mixed sites. XYZ and POSCAR are not read yet.
- Save analysis writes into the project's folder without asking.
- Checked in the app: a restart, opening and saving in the base app (structures
  show "Needs the abtem plugin"), and reopening with the plugin.
- A moved project folder still opens.

### Phase 2: structure viewer

- 2a (built): the Structure tab, orthographic 3D view (Three.js, built with
  `npm --prefix plugins/abtem/frontend run build`), cell, a/b/c axes, element
  legend with show/hide, hover readout, reset.
- 2b (built): zone-axis buttons, typed [uvw] and [uvtw], nearest zone axis and
  its angle, tilt, in-plane rotation and shortcuts while the view has focus: `A`,
  `B`, `C`, `Z`, `R`, arrows (1°, Shift 5°), `Q`/`E` (rotate). Checked on STO [001],
  [110] and [111]. The [uvtw] maths is checked numerically, not yet on a real
  hexagonal CIF.
- 2c (built): repeats up to 20 per axis (50,000 atoms), atom size, bonds (closer
  than a tolerance × the sum of covalent radii, default 1.05; no metal–metal bonds
  when a non-metal is present), mixed sites as viewer-facing wedges with vacancies
  in grey and clashing colours separated, transparent PNG export.
- 2d (built): *Create model* in the Process tab. The form doesn't follow the view.
  z is [uvw]; x and y are the shortest perpendicular lattice directions, with y
  nearest the viewer's up. Built with ASE in the plugin's environment, because
  abTEM's `orthogonalize_cell` isn't available there. Up to 1% strain, reported.
  Mixed sites are filled at random from a seed with exact counts per site, and
  vacancies removed. Models are stored as `models/*.extxyz`, up to 50,000 atoms.
  STO [001], [110] and [111] models ran through abTEM 1.0.11 multislice.
  *From view*, beside the zone axis field, imports only the view's zone axis.
  Repeats default to 1 × 1 × 1 of
  the orthogonal cell (for STO [112]: 5.52 × 6.76 × 9.57 Å, 6 unit cells, 30 atoms).
- Built one change at a time, each left running at localhost:8767 for the user
  to try before the next.
- **Done when:** zone axes are checked against known projections on STO. Low
  symmetry is set aside. [uvtw] needs a hexagonal CIF to check.

### Phase 3: worker and potential

- 3a (built): an *abTEM* process on models shows the install's status and
  installs abTEM 1.0.11 into its own environment (`~/.si-viewer/abtem-env`, about
  600 MB; `SI_VIEWER_ABTEM_ENV` overrides it) as a cancellable job. An interrupted
  install doesn't count, and a folder the app didn't make is never replaced. The
  worker (`worker/main.py`) runs in that environment.
- 3b (built): a *Potential* process on models: sampling, slice thickness,
  Lobato or Kirkland, infinite or finite projection, with the grid and slice
  count. *Build potential* runs in the worker and adds a **Potential** object
  under the model: a stack of slices along z (each integrated through its slice,
  V·Å), oriented as the beam view (x right, y up; the image's y axis counts from
  the top). It opens on *Projected*, the stack's Sum, which equals abTEM's
  projected potential exactly; there is no separate projected object. Masks and
  kemstem need a single image, so they aren't offered on a Potential.
  New settings add a Potential; the same settings reuse it. Limits: 4096 pixels
  per side, 2 GB of slices.
- Frozen phonons (built): a Potential setting (abTEM `FrozenPhonons`: 0 or 1
  configurations is off, up to 64; σ in Å per direction, default 0.1; seed). The
  Potential stack is the mean over configurations (built one at a time). Runs on
  it keep one wave per configuration, never an averaged wave: `exit_wave.npy` is
  [config, x, y] (and `wave.npy` [config, plane, x, y]); without phonons they stay
  [x, y] and [plane, x, y]. Intensities are averaged: the worker's `image` forms an
  image per leading-axis configuration and averages them (inside each, a focal
  series averages over defocus). The exit wave's Amplitude and Phase datasets are
  the first configuration's (titled "(config 1)"), a Wavefunction's Intensity the
  mean over configurations and its Phase the first's. A tiled wavefunction tiles
  the displaced atoms, so every tile moves independently.
- Object hierarchy: Structure → Model → Potential → run (HRTEM) → its images.
  Objects have their own type (badge) and keep their data kind's processes, except
  those they hide (a Potential hides *Align frames* and *Rebin*). Potentials and
  HRTEM runs made before this change stay in their old place as plain images.
- 3c (deferred, decided 2026-11-23): cost estimate of memory, result size and
  time, shown before a run. To be built after the simulation modes, with the time
  calibrated by a short benchmark on the machine.
- 3d: runs written to the project (built with HRTEM, Phase 4).

### Phase 4: HRTEM

- 4a (built): an *HRTEM* process on Potentials: energy, defocus (positive is
  underfocus, abTEM's convention), Cs, a Scherzer button and an optional objective
  aperture. A run appears under its Potential with the image (Intensity) and the
  exit wave's Amplitude and Phase. Runs are written to
  `runs/<id>/` (`exit_wave.npy`, `image.npy`, `settings.json`, `manifest.json`)
  only when finished. The lens was checked against an independent Fourier
  calculation (difference under 1e-5) and the exit phase against σV.
- 4b (built): CTFs applied to runs, following the HRTEM quickstart and the
  CTF and partial coherence walkthroughs. An *HRTEM* run takes one of the
  project's CTFs (and so its energy) or none; it keeps the exit wave
  (Amplitude, Phase) and adds the **Image** the CTF forms
  (`exit_wave.apply_ctf(ctf).intensity()`). *Apply CTF* on a run adds the
  image of another CTF at the same energy without repeating multislice; each
  image keeps a snapshot of its CTF. The CTF form gained *Crossover* (aperture
  at the first zero) and *From Cc, ΔE* (focal spread √2·Cc·ΔE/E, Cc 1.2 mm and
  ΔE 0.35 eV by default). Runs made before keep their lens in the run.
- 4c (built for HRTEM): focal spread as abTEM's quasi-coherent envelope, or as
  an incoherent focal series (Gaussian, σ = δ/√2, 7 defocus values within
  ±1.5σ, `reduce_ensemble`), from the partial coherence tutorial. The angular
  spread stays an envelope. Checked on Si [110] at Scherzer with the aperture
  at the crossover: dark dumbbells; mean intensity 0.977 (envelope) and 0.998
  (focal series). Still to do: probes, and source size for STEM.
- 4d (built): *Thickness series* on a Potential (job): a plane wave kept every
  *n* slices (`exit_planes`; plane 0 the incident wave), `runs/<id>/exit_waves.npy`,
  and a **Thickness series** item with stacks through thickness (axis t, Å):
  Image (with a CTF), exit-wave Amplitude and Phase. *Defocus series* on an
  HRTEM run (long-running worker, no multislice): one CTF at a range of
  defocus values, a **Defocus series** stack (axis Δf, Å), up to 100 images.
  Both handle frozen-phonon configurations (intensities averaged). Code in
  `series.py`, `worker/series.py`, `frontend/src/series.js`. Checked: last
  plane = plain HRTEM exit wave (diff 0); the defocus-series image at the CTF's
  own defocus = Apply CTF (≤ 2.4e-7).
- Validation against a published or abTEM tutorial result.
- Wavefunction (built): a *Wavefunction* process on Potentials sends a plane wave
  or probe (convergence, defocus, Cs, position) through the potential, tiled
  laterally with `CrystalPotential`, with `exit_planes=1`. The result is a
  **Wavefunction** under the Potential with Intensity and Phase stacks of
  slices + 1 planes (plane 1 is the incident wave) and the complex wave in
  `runs/<id>/wave.npy`. A plane wave's incident pixels are 1; a probe's incident
  total is 1. Phase has no Mean or Sum. Probes default to a cell of at least 20 Å.
- CTF (built: creation and visualisation): **CTF** items in the project (*New
  CTF* on any simulation object, or with nothing selected; *Copy as new CTF*).
  Settings: energy, defocus (Scherzer), Cs, C5, astigmatism C12 and angle,
  aperture, focal spread and angular spread (abTEM's quasi-coherent envelopes,
  1/e widths), range and, with astigmatism, the profiles' direction. A CTF is
  an image dataset at the project's top level: the 2D CTF (256², kx right and
  ky down in 1/Å from the centre, so readouts are exact; red–blue colormap)
  shown in the viewer like any image, with processes that mean nothing for it
  hidden. Its *CTF profile* bottom tab shows abTEM's profiles (CTF with its
  envelopes and aperture, and each of those) in 1/Å or mrad, with point
  resolution and information limit (envelopes at exp(-2)) marked, and Scherzer
  defocus, crossover angle, point resolution and information limit. Edits
  redraw the profile at once, the image after 0.2 s (`PUT …?save=false`,
  same revision so contrast stays), and are saved 1.5 s after the last. The
  core gained plugin datasets with no parent and a plugin-chosen first
  colormap. Computed by a long-running worker (`main.py serve`,
  `engine.Service`): abTEM imports once (~2 s), then each CTF takes ~10 ms; it
  restarts if it stops. Works round abTEM 1.0.11's `to_diffraction_patterns`
  ignoring `gpts` by setting the grid first. Checked: the profile equals
  −sin χ to 1e-4. Next: apply a CTF to HRTEM runs and exit waves.
- Cross-section (built): a *Cross-section* bottom tab on the plugin's stacks
  (Potential, wave Intensity and Phase) cuts every plane along the selected
  profile line (`POST /stacks/{id}/section`, linear interpolation that wraps
  at the cell edges, the line's width averaged, the stack's filter applied).
  While the tab is shown, the image has an x–z and a y–z line (fixed profile
  lines: through the centre at first, full length, moving only across, never
  deleted; hidden when the tab is left, kept with their position). *x–z* and
  *y–z* pick which one the section follows; another drawn line can be picked
  from the list. Depth runs down from the entrance surface; shown at true
  aspect or stretched (*Fill*), in the view's colormap (a phase in the
  circular one). The wheel zooms (up to ×64), dragging pans, a double-click
  resets. The line tool stays on a plugin tab while lines are dragged. The core
  gained fixed profile lines (`ProfileLine.fixed`) and, in the plugin host,
  `useLines`, `getLines`, `addLine`, `updateLine`, `selectLine`,
  `useColormap` and `colormapLut`.

### Phase 5: diffraction

- SAED and CBED, with calibrated reciprocal axes, then PACBED.

### Phase 6: STEM imaging and quantification

- Built (following the STEM quickstart and the scan and detect walkthrough):
  *STEM* on a Potential (job): probe (energy, convergence, defocus, Cs),
  lateral tiling (cell ≥ 20 Å by default, scan stays over the model's cell),
  scan region as fractions of the cell, scan sampling (Nyquist by default),
  radial bin, detectors (BF 0–α, ABF α/2–α, MAADF 50–120, HAADF 90–200 mrad,
  clipped to the simulated maximum and refused before running if beyond it).
  A `FlexibleAnnularDetector` keeps the polar data (`runs/<id>/polar.npy`);
  the **STEM** run holds one image per detector. *Add detector* integrates
  another annulus from the saved data (`integrate_radial`), *Source size and
  noise* on an image interpolates, blurs (Gaussian σ, Å) and adds Poisson
  noise (dose e⁻/Å²) as a new image. Frozen phonons averaged. Code in
  `stem.py`, `worker/stem.py`, `frontend/src/stem.js`. Checked: integration
  from the polar data = `AnnularDetector` (≤ 2.4e-7); Si [110] HAADF peaks
  within 0.11 Å of the projected columns.
- Still to do: run kemstem on simulated STEM images and check against the
  model's atom positions in the app.

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
8. **Test structures (decided 2026-11-23):** STO in `examples/abtem`. Low-symmetry
   structures are set aside.
