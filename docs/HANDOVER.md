# Handover (2026-11-10)

Read this first, then `docs/PLAN.md` (the design and milestones; §4.11 EELS,
§4.13 in-situ, §4.14 experiments). This file covers what PLAN.md doesn't: how
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
  75 s, 370 tests; 2 skip because `examples/STEM SI_EELS Spectrum
  Image_summed.dm4` is gone). Run only the relevant files while iterating.
- Example data: `examples/` (git-ignored). `InSitu (3)`: an in-situ EELS
  series (28 GB of `.raw`) with its survey and ADF. `InSitu (10)`: a dual EELS
  pair (HL/LL summed, 96×415) with survey and ADF. `LSMO_linescan*.hspy`: a
  low/core-loss line-scan pair (no DM role tags: mark the low loss by hand).
  The test fixtures (`backend/tests/conftest.py`) look in subfolders too.

## How the user works (also in the memory files)

- Small fixes and features: do them directly. Subagents (`model: "sonnet"`)
  only for large or parallel work, with a self-contained brief; they start
  cold and are slow.
- Commit and push only when asked. Never commit `.claude/`.
- UI text is terse: labels, values, data-loss warnings; short tooltips are
  fine. No explanatory paragraphs in the UI.
- Tunable processing is guided: settings in the Process tab, a live preview
  drawn in the viewer, the heavy part as a cancellable job when the user asks
  (see background removal and deconvolution).
- Test in the built-in browser in a separate tab on the test server (8766),
  then close it, switch back to the user's tab ("seed") and reload it after
  every rebuild.
- Backend changes need the user's server restarted, which loses the session
  (everything is in memory). Ask before restarting. Before restarting, record
  the open files, in-situ views (`insitu.view`), window maps
  (`window: {lo, hi}`) and renamed titles from `GET /api/files`; after,
  reopen them (`POST /api/files/folder` + `POST /api/files` with
  `experiment_id` for an experiment; `POST …/insitu/reduce` or `PUT
  …/insitu/frame`; `POST …/window-maps {lo, hi, label}`), then reload the tab.
  Display settings (contrast, zoom, cursor) live in the browser.
- Shell: a hook blocks `cd X && …`. Use absolute paths, `git -C`,
  `npm --prefix`, `conda run -n si-viewer --cwd <dir>`. No `sed -i` or
  heredoc rewrites in subagents. Free a port with `lsof -sTCP:LISTEN`, never
  plain `lsof` (it once killed an app process).
- The user's own EELS code is at `~/Documents/Dev/EELS` (outside the working
  folders: read it with Bash).

## State of the working tree

Everything below is committed on `main` in the commit after `ee24d92`
("Experiments, survey regions, deconvolution, …"; not pushed). The tests
pass and the frontend builds. `.claude/` is never committed.

## Built in the last session

Images and export
- PNG export dialog (font size, scale bar length and background, grid
  spacing): `components/PngExportDialog.tsx`, `viewer/snapshot.ts`.
- Matplotlib plot export of spectra and profiles (cascade with spacing, a
  second y axis, fonts, axis limits): `processing/plot_export.py`,
  `api/plot_export.py`, `components/PlotExportDialog.tsx`.
- Export dialog (browse a folder, name, file type; overwrite asks) replacing
  the "Save as…" items, and *Batch export…* on a type group's header:
  `components/ExportDialog.tsx`, `POST /api/datasets/{id}/save`,
  `POST /api/export/batch` (`api/export.py`).
- Gamma as a number; colormap as an icon with a popover list
  (`components/ContrastControls.tsx`).

FFT
- FFT pane: Linear/Log, contrast controls (foldable), Hann toggle, bandpass
  and alignment-mask rings drawn and draggable (`components/FftPanel.tsx`,
  `viewer/fft.worker.ts`, `viewer/fourierMask.ts`).
- FFT boxes: an *FFT* tool on images, stacks and SI navigators draws square
  regions; the FFT pane lists View / FFT 1 / …; *View in main window* shows
  all their FFTs beside the image (`state/fftBoxes.ts`,
  `views/useFftBoxTool.ts`, `components/FftBoxOverlay.tsx`,
  `components/FftCompare.tsx`). Boxes are kept per view, in memory only.

Processing
- Bandpass spatial filter (`processing/filters.py`, `state/filters.ts`).
- Deconvolution (§4.11 of PLAN.md): Fourier-log, Fourier-ratio (on
  background-removed SIs only), Richardson–Lucy (PSF low loss or its ZLP),
  live preview on the spectrum plots, result as a new SI
  (`processing/deconvolution.py`, `api/deconvolution.py`,
  `components/DeconvolutionSection.tsx`, `state/deconvolution.ts`,
  `views/si/useDeconvPreview.ts`).

Datasets pane
- Same-type datasets fold into a group with tree lines; rename (F2,
  double-click, right-click, or the Info tab title; `PUT
  /api/datasets/{id}/title`, kept through recomputation).
- Experiments (§4.14): *Open all in folder* opens a folder as one experiment
  listed by role, EELS HL/LL linked on opening; the DM survey is a *Survey*
  dataset with the SI and drift rectangles drawn on it
  (`io/dm_experiment.py`, `Session.open_folder`, `components/SurveyRegions.tsx`).

In-situ (§4.13)
- M12a reader (`io/insitu.py`), M12b frame browsing with Frame/Mean/Sum of a
  range (`insitu_series.py`, `api/insitu.py`,
  `components/InsituFrameBar.tsx`), sub-scan navigators kept at full ADF
  resolution with a k×k cursor (`subscan.py`, `views/si/scan.ts`).

## Open items

- **Waiting on the user:** should a spectrum whose energy axis covers 0 eV be
  marked low loss automatically? (Roles come only from DM's dual-EELS tag;
  everything else is core loss until set in the Info tab, which is why
  zero-loss alignment is greyed out for the LSMO pair.) A test in
  `test_eels_links.py` asserts that nothing is guessed; change it with the rule.
- In-situ next: M12c (time-series job), M12d (Time tab), M12e (sum frames
  with drift correction). See PLAN.md §4.13; recommended order c+d, then e.
- Known gaps: thickness and ZLP results aren't titled with the in-situ frame;
  the ZLP section shows its "Overwrites…" warning even when disabled; no
  non-square in-situ test data; FFT-comparison panels don't draw the
  bandpass/mask rings; PNG export doesn't include survey regions or the FFT
  comparison; opening a single SI file doesn't bring its survey (only the
  folder does); survey rectangle rotation is ignored; deconvolution previews
  use the raw (unfiltered) data.
- `examples/STEM SI_EELS Spectrum Image_summed.dm4` was removed by the user:
  2 tests skip.
