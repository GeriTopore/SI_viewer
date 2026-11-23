# Handover (2026-11-23)

Read this first, then `docs/PLAN.md` (the design and milestones; §4.11 EELS,
§4.13 in-situ, §4.14 experiments, §4.25–4.27 atomic-resolution work). This file covers what PLAN.md doesn't: how
the work is done, the state of the working tree, and what is open.

## The project

SI Viewer: a local web app for STEM spectrum images (EDS and EELS), images,
stacks and in-situ series. FastAPI + HyperSpy/exSpy backend
(`backend/si_viewer`), React + TypeScript + Zustand + uPlot frontend
(`frontend/src`). Repo: GitHub `GeriTopore/SI_viewer`, branch `main`.

- Run: `/Users/gt622/miniforge3/envs/si-viewer/bin/si-viewer --no-browser`
  (port 8765). In the desktop app: `preview_start` with the `si-viewer`
  configuration in `.claude/launch.json`; `si-viewer-test` is the same on
  port 8766, for testing.
- Build the frontend: `npm --prefix frontend run build` (runs `tsc -b`, writes
  into `backend/si_viewer/static`, which the server serves). Type check only:
  `npm --prefix frontend run typecheck`.
- Tests: `conda run -n si-viewer --cwd backend python -m pytest -q` (about
  2.5 min; 507 pass, 6 skip). Run only the relevant files while iterating.
- Example data: `examples/` (git-ignored). `InSitu (3)`: an in-situ EELS
  series (28 GB of `.raw`) with its survey and ADF. `InSitu (10)`: a dual EELS
  pair (HL/LL summed, 96×415) with survey and ADF. `LSMO_linescan*.hspy`: a
  low/core-loss line-scan pair (no DM role tags: mark the low loss by hand).
  `iDPC_BTFMO_test.tif`: atomic columns with three sublattices (A, B and an
  oxygen site C that moves up and down with the octahedral tilt: one site, not
  two); its saved analysis is `iDPC_BTFMO_test analysis.hspy`. The user's
  kemstem notebook for it: `~/Documents/scripts/BTFMO_polarisation_mapping.ipynb`.
  The test fixtures (`backend/tests/conftest.py`) look in subfolders too.

## How the user works (also in the memory files)

Planning
- The Obsidian note `/Users/gt622/Documents/Obsidian Vault/Codes/Si-viewer.md`
  is the user's todo list and decides what is planned. Read it before "what's
  next". When it changes, bring `docs/PLAN.md` in line; when a feature is
  built, tick it there and record it in PLAN.md.

Implementation
- Small fixes and features: do them directly. Subagents (`model: "sonnet"`)
  only for large or parallel work, with a self-contained brief; they start
  cold and are slow. Never two writers on one file. Review a subagent's report
  before calling the work done.
- Use the dependencies' implementations (HyperSpy, exSpy, kemstem,
  scikit-learn, scikit-image, rosettasciio) and write only glue. A custom
  algorithm only when none of them does it.
- Commit and push only when asked, and don't offer it after each step. Never
  commit `.claude/` or `examples/STO`.

UI
- UI text is terse: labels, values, data-loss warnings; short tooltips are
  fine. No explanatory paragraphs, provenance notes or hints in the UI.
- New UI reuses existing classes: dialogs `modal rebin-dialog` with
  `rebin-subtitle` and `rebin-actions`; editable values the `axes` table with
  `cell-input`; Info-tab entries `props` rows with a `link-button`; toolbar
  toggles `label.field`. Compare a screenshot with a neighbouring element
  before reporting.
- Tunable processing is guided: the user decides each step, with settings in
  the Process tab, a live preview and diagnostics drawn in the viewer, and the
  heavy part as a cancellable job when the user asks (see background removal
  and deconvolution). No one-click pipelines.
- Processes run on the highlighted dataset, including derived images such as
  crops and gratings, never silently on a parent.
- Dataset pane: click shows only that dataset, shift-click adds or removes it
  from the view, ⌘/Ctrl-click toggles selection.
- General tools stay general: masks are made from any image and aren't tied to
  sublattices.

Testing
- Test in the built-in browser on the test server (8766), in a separate tab
  brought to the front so the user can watch. Never use or navigate the user's
  tab ("seed") for a test. Afterwards stop the test server, close its tab,
  switch back to "seed" and reload it. Reload "seed" after every rebuild or
  restart too.
- When driving a page, take refs from a fresh `find`. Stale refs have clicked
  the wrong control in the user's session.
- Don't run tests of long processes. Use kemstem's example images for atomic
  work where possible. Test curve fitting on
  `examples/003_EELS-SI_1.7nm_2mmAp_TiO_Stack_Averaged_x_1_38_y_1_235_445.05_488.05_eV_PCA_4.hspy`.
  No new synthetic datasets unless asked; backend tests may use small arrays.

Server and shell
- Backend changes need the user's server restarted, which loses the session
  (everything is in memory). Ask each time: an earlier yes doesn't carry over.
  Prefer saving the session to a file and reopening it afterwards. Otherwise,
  before restarting, record the open files, in-situ views (`insitu.view`),
  window maps (`window: {lo, hi}`) and renamed titles from `GET /api/files`;
  after, reopen them (`POST /api/files/folder` + `POST /api/files` with
  `experiment_id` for an experiment; `POST …/insitu/reduce` or `PUT
  …/insitu/frame`; `POST …/window-maps {lo, hi, label}`), then reload the tab.
  Display settings (contrast, zoom, cursor) live in the browser.
- Shell: never `cd X && …` (a hook blocks it). Use absolute paths, `git -C`,
  `npm --prefix`, `conda run -n si-viewer --cwd <dir>`. Subagents: no `sed -i`
  or heredoc rewrites, short single commands, temporary files in the
  scratchpad. Free a port with `lsof -sTCP:LISTEN`, never plain `lsof` (it
  once killed an app process).
- The user's own EELS code is at `~/Documents/Dev/EELS` (outside the working
  folders: read it with Bash).

## State of the working tree

Last commit: `8eff709` (2026-11-16), level with `origin/main`. Everything in
"Built since the last commit" is **uncommitted** (33 tracked files changed, plus
new files). On 2026-11-23 the full test suite passed (507, 6 skipped) and the
type check was clean. `.claude/` and `examples/STO` must stay out of commits.

Neither server is running, so the next start of `si-viewer` picks up the current
code. The in-memory session is gone, and the crop session saved to the
scratchpad (`idpc_crop_session.h5`) no longer exists: the crop analysis has to
be redone unless `examples/iDPC_BTFMO_test analysis.hspy` (saved 2026-11-17
13:10) holds what is needed. Its contents have not been checked.

## Built since the last commit

Details are in PLAN.md §4.25–4.27.

Atomic columns
- Masks from any image or an image made from it: `processing/masks.py`,
  `api/masks.py`, `state/masks.ts`, `components/MaskSection.tsx`. A mask adds
  one sublattice of the columns inside it, next to the existing ones.
- Find: template cross-correlation and a background σ. Seeded guesses are
  appended, and only they are fitted (`append_columns`).
- Cages: sites at the centre of a cage, and each origin's displacement from it
  drawn as vectors. The cage is picked clusters or the k nearest (shape check
  plus origin inside the cage).
- Sublattices by position in cell: `state/cell.ts`, `components/CellPlot.tsx`,
  the Cell bottom tab. Sites are circles or clusters. Clusters use seeded
  DBSCAN (`cluster_sites`), because Gaussian ellipses took in strays near
  (½, 0).
- Picks (lattice vectors, cage, cell sites) are kept with the analysis:
  `state/picks.ts`, `PUT /api/datasets/{id}/atomic/picks`.

Workflow
- Process pane split by dragging a process icon, with links between processes
  (`components/ProcessPanel.tsx`, `components/ProcessLink.tsx`). Dataset titles
  removed from process headers.
- Processes run on the highlighted dataset (fixes crops running on the full
  image).
- Dataset pane: new click behaviour and a stronger focused row.

Results on `iDPC_BTFMO_test.tif`
- Full image: 4094 columns (template σ 3.5, blur 1.5, distance 8, threshold
  0.4), split A 1404, B 1142, C 1299. C displacement from its 4 nearest A:
  median 58.7 pm, largest 119.7 pm, 362 of 1299 left out.
- Crop (737 columns): circles at 16 px give A 262, B 207, C 197, 71 left out.
  Clusters give A 262, B 207, C 248, 20 left out, mostly unclaimed groups near
  (½, 0).

## Open items

The Obsidian note holds the full todo list. Nearest to the current work:
- Keep the reference's neighbourhoods when splitting by position in cell.
- Cages: polarisation (−displacement), smoothed vectors, calibrated units.
- PLD results: no hover readout for displacements; grid panels are black where
  an image is implied.
- Decide the licence before packaging (kemstem is GPL-3.0).

Carried over from 2026-11-10, not rechecked since:
- **Waiting on the user:** should a spectrum whose energy axis covers 0 eV be
  marked low loss automatically? (Roles come only from DM's dual-EELS tag;
  everything else is core loss until set in the Info tab.) A test in
  `test_eels_links.py` asserts that nothing is guessed; change it with the rule.
- Known gaps: thickness and ZLP results aren't titled with the in-situ frame;
  the ZLP section shows its "Overwrites…" warning even when disabled; no
  non-square in-situ test data; FFT-comparison panels don't draw the
  bandpass/mask rings; PNG export doesn't include survey regions or the FFT
  comparison; opening a single SI file doesn't bring its survey (only the
  folder does); survey rectangle rotation is ignored; deconvolution previews
  use the raw (unfiltered) data.
