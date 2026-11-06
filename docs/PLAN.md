# SI Viewer — architecture and first-draft plan

A desktop-feeling app for browsing STEM data (images, image stacks, EDX/EELS
spectrum images) and processing EDX spectra with HyperSpy + exSpy.
EDX on ThermoFisher Velox `.emd` comes first; EELS later.

## 1. What the example data tells us

`0016-20261105_1651_13.0_Mx_HAADF.emd` (60 MB, Spectra 300 kV, Velox 3.27):

| Item | Type | Shape |
|---|---|---|
| EDS spectrum image | `EDSTEMSpectrum` | 256 × 256 × 4096 (uint16, 512 MB dense), 122 frames |
| Sum spectrum | `EDSTEMSpectrum` | 4096 |
| HAADF | `Signal2D` | 256 × 256 (+ a 122-frame stack acquired with the SI) |
| Velox element maps (Bi, C, Mn, O, Ti, Cu, Fe) | `Signal2D` | 256 × 256 |

Elements stored in the file: C, O, Ti, Mn, Fe, Cu, Bi. Pixel size: 0.0306 nm.
Energy axis: 5 eV/channel, offset -0.48 keV.

**Load blocker found:** `hs.load()` fails with "not a supported EMD file".
This Velox version writes `{"format": "VeloxLite", "version": "13"}`, and
RosettaSciIO 0.14 only accepts `Velox` / `DevelopersKit`
(`rsciio/emd/_api.py::is_EMD_Velox`). If the check also accepts `VeloxLite`,
the existing `FeiEMDReader` loads everything correctly. The app ships a small
loader shim that does this. It also tests whether rsciio 0.15 already fixes it,
and the issue should be reported upstream.

Measured timings (hyperspy env, this Mac):

| Operation | Time |
|---|---|
| Lazy open, all signals | 3.8 s |
| Full SI load into RAM | 2.1 s |
| Sum spectrum over a 30 × 30 ROI | 1.6 ms |
| Energy-window map (40 channels) | 3 ms |
| Whole-SI sum | 108 ms |
| `create_model` + `fit` on the sum spectrum (7 elements) | 1.7 s |
| Line-intensity maps for the full SI (no background) | 0.17 s |

Takeaway: keep the SI in RAM on the backend and send only small results
(spectra, maps, profiles) to the frontend. Cursor and ROI updates then fit
easily inside one animation frame. Only per-pixel model fitting needs a
background job.

## 2. Architecture (recommended)

```
┌──────────── Frontend (React + TypeScript, Vite) ────────────┐
│ Dataset list │ Image/Navigator view │ Spectrum plot │ Panels │
│  WebGL2 image renderer + SVG overlay (cursor, ROIs, lines)   │
│  uPlot spectrum/profile plots, Zustand state                 │
└──────────────┬───────────────────────────┬───────────────────┘
       REST (JSON commands,         WebSocket (cursor/ROI stream,
       binary array payloads)       job progress)
┌──────────────┴───────────────────────────┴───────────────────┐
│ Backend (Python, FastAPI + uvicorn, same process as HyperSpy) │
│  Session: open files → signals (lazy until selected)          │
│  io/emd.py (VeloxLite shim) · processing/ (rois, maps,        │
│  eds_fit, quant, profiles) · jobs (thread pool + cancel)      │
└───────────────────────────────────────────────────────────────┘
```

Why this stack:
- **Python backend in-process with HyperSpy.** All exSpy features (models,
  background windows, calibration, quantification, PCA) are available directly.
  No reimplementation.
- **Web frontend.** A modern, responsive UI. WebGL keeps image contrast and
  colormap changes on the GPU, with no round trip. uPlot draws 4096-point
  spectra at 60 fps, where Plotly feels sluggish.
- **Binary transport.** Images and spectra travel as raw `Float32Array`
  buffers with a small JSON header (shape, axes, units), not as JSON lists.
- **Desktop wrapper later.** The app first runs as `si-viewer` and opens the
  browser at `localhost`. Then it is wrapped with **pywebview**: one Python
  process, a native window, and native file and folder dialogs. Tauri or
  Electron can be considered if it is ever distributed widely.

Alternatives considered: PyQt + pyqtgraph is fast and single-language, but
harder to make look modern and polished. Panel/Bokeh and Jupyter widgets feel
clunky for this kind of interaction. napari is image-centric and awkward for
spectra.

## 3. Repository layout

```
SI_viewer/
  pyproject.toml            # backend package `si_viewer`, entry point `si-viewer`
  environment.yml           # conda env: hyperspy, exspy, fastapi, uvicorn, pytest
  backend/si_viewer/
    app.py                  # FastAPI app, static frontend serving, CLI entry
    session.py              # Dataset registry: id → (file, signal, kind, lazy/loaded)
    io/emd.py               # load_emd(): VeloxLite shim, split SI / stack / images
    io/generic.py           # fallback via hs.load for other formats
    api/datasets.py         # list/open/close, thumbnails, metadata
    api/data.py             # image frames, spectra, maps (binary responses)
    api/eds.py              # elements, lines, background, fit, quant
    api/profiles.py         # line profiles
    api/ws.py               # cursor/ROI stream + job progress
    processing/rois.py      # point/rect/ellipse/polygon masks → sum/mean spectrum
    processing/maps.py      # energy-window maps, line-intensity maps
    processing/eds_fit.py   # wraps EDSTEMModel: build, fit, calibrate, results
    processing/quant.py     # CL / zeta / cross-section, absorption correction
    processing/profiles.py  # Line2DROI-based profiles for images and SIs
    jobs.py                 # background jobs with progress + cancellation
  backend/tests/            # pytest, uses the example .emd (path via env var)
  frontend/
    src/api/                # typed REST + WS client, binary decoding
    src/state/              # Zustand stores (datasets, view, rois, eds)
    src/components/
      DatasetList/          # grouped by file, type badges, thumbnails
      ImageView/            # WebGL renderer, pan/zoom, colormaps, scale bar
      Overlay/              # cursor, ROI shapes, line tool (SVG)
      SpectrumPlot/         # uPlot, X-ray line markers, span selector
      PeriodicTable/
      EdsPanel/             # elements, background, fit, quant, results
      ProfilePanel/
  docs/PLAN.md
```

## 4. Features — first draft

### 4.1 Loading and dataset list
- Open files or a whole folder: a server-side file browser at first, native
  dialogs once in pywebview. Drag and drop works where the path is available.
- Each file expands into its signals, with a badge for each kind: **Image**,
  **Stack**, **Spectrum**, **SI**. Velox element maps sit under an "Element
  maps" subgroup.
- Files open lazily; a signal is materialised only when selected. Memory use is
  shown, and datasets can be closed to free RAM.
- Up/down arrow keys switch datasets. Each dataset keeps its own view state
  (zoom, contrast, ROIs).
- Metadata inspector: microscope, beam energy, detector, dwell time, frames.

### 4.2 Visualisation
- **Image view:** pan/zoom (wheel, drag), fit-to-window, pixel readout in
  calibrated units, colormaps (gray, inferno, viridis, …), contrast from a
  histogram with percentile auto-scaling, gamma, and a scale bar.
- **Stacks:** a slider bar under the image, dragged to scrub through the frames,
  with a frame number readout (no playback). Neighbouring frames are prefetched
  so scrubbing stays smooth. Sum or mean projection is available as an option.
- **SI layout:** navigator image on the left, spectrum on the right. The
  navigator can be the simultaneous HAADF, the SI total counts, or any
  energy-window or element map.
- **Spectrum plot:** log/linear y, zoom/pan, auto-range, an energy readout
  under the mouse, and overlay of several spectra with a colour per ROI.
- **Composite maps:** RGB/CMY overlay of up to 3 to 6 element maps on the HAADF.

### 4.3 Cursor and area spectra (SI)
- **Point cursor:** a draggable crosshair, nudged with arrow keys. Optional
  n × n binning around the cursor helps with noisy data. The spectrum updates
  live over the WebSocket, coalesced to one in-flight request at a time.
- **Area ROIs:** rectangle, ellipse and polygon. Each gives the sum or mean
  spectrum. Several ROIs can be on screen at once, each colour-coded on both the
  image and the spectrum, and each can be renamed, hidden or deleted.
- **Reverse direction:** drag an energy span on the spectrum and the navigator
  shows the map of that window live.
- Any ROI spectrum can be sent to the EDX panel as the active spectrum for
  fitting.

### 4.4 EDX fitting and quantification (exSpy)
- **Element selection:** a periodic table picker. Elements found in the file
  metadata are pre-selected. Hovering an element previews its X-ray lines on
  the spectrum (from `exspy.misc.elements.elements_db`). Individual lines
  (`Ka`, `La`, `Ma`, …) can be turned on or off.
- **Microscope parameters:** beam energy, energy resolution (Mn Kα), tilt,
  azimuth/elevation and live time. They are pre-filled from metadata and
  editable.
- **Background:**
  - Window method: `estimate_background_windows` and
    `estimate_integration_windows` show the windows as draggable bands, then
    `get_lines_intensity(background_windows=…)` is run.
  - Model method: `create_model(auto_background=True)` with a selectable
    polynomial order (`add_polynomial_background`), then `fit_background`.
- **Model fit:** `create_model` → `fit` → optionally
  `calibrate_energy_axis('offset' | 'resolution' | 'scale')` and
  `calibrate_xray_lines('energy' | 'width' | 'sub_weight')`. Show the fit, the
  separate components, the background and the residual. Show line intensities
  in a table.
- **Quantification:** `quantification(method='CL' | 'zeta' | 'cross_section')`
  with an editable k-factor table saved as presets, and optional absorption
  correction (thickness/density). The result is at% / wt% for the spectrum.
- **Mapping (SI):**
  - Line-intensity maps by windows, which is fast.
  - Per-pixel model fit as a background job with progress and cancel, with an
    optional rebin first.
  - Quantified composition maps, with a vacuum mask (`vacuum_mask`).
- Results export: CSV for spectra and tables, TIFF/PNG for maps, and `.hspy`
  for processed signals.

### 4.5 Line profiles
- A line tool on any image or navigator, with draggable endpoints, an
  adjustable width (integration perpendicular to the line) and the length shown
  in calibrated units.
- **Image profile:** intensity vs distance, updated live while dragging.
- **SI profile:** uses `hs.roi.Line2DROI(linewidth=…)` to give a
  distance × energy signal. From that:
  - Element line-intensity profiles (window method at first, model fit later).
  - Composition profile after quantification.
  - The spectrum at any position along the line, via a cursor on the profile
    plot.
- Several profiles at once, overlaid. Export to CSV.

### 4.6 Later (not in the first draft)
EELS SIs (`exspy.signals.EELSSpectrum`, background power law, edges), PCA/NMF
denoising (`decomposition`), drift correction from the SI HAADF stack, batch
processing recipes across files, and saved sessions.

## 5. Responsiveness rules
- The backend holds signals as NumPy arrays in RAM after selection and uses
  lazy (dask) only for files too large for memory.
- Cursor and ROI requests go over the WebSocket. The frontend sends the latest
  position only (it drops intermediate positions), and the backend answers in
  under 10 ms for typical SIs.
- Anything over about 200 ms runs as a job: it does not block the UI, shows
  progress, and can be cancelled. HyperSpy calls run in a worker thread so the
  event loop stays free.
- Contrast, colormap and zoom are entirely client-side (WebGL), so they need
  no network call.
- Responses are cached by key (dataset, ROI geometry, energy window).

## 6. Milestones

| # | Deliverable | Status |
|---|---|---|
| M0 | Repo scaffold, `environment.yml`, `load_emd` with the VeloxLite shim and pytest on the example file, FastAPI skeleton, Vite app shell, `si-viewer` command | Done |
| M1 | Dataset list and loading; image and stack viewer; SI navigator + sum spectrum; manual axis calibration | Done |
| M2 | Point cursor, area ROIs, energy-span maps, X-ray line markers, periodic table | |
| M3 | Background (windows + model), model fit + calibration, quantification, SI maps job | |
| M4 | Line profiles for images and SIs (intensity and element profiles) | |
| M5 | Export, keyboard shortcuts, pywebview desktop window, polish | |

## 7. Verification
- **Backend:** pytest against the example `.emd`: signal kinds and shapes,
  that the ROI sum equals a NumPy reference, that fit intensities are within
  tolerance of `get_lines_intensity`, and that the profile length matches the
  geometry.
- **Performance tests:** cursor spectrum under 10 ms; window maps for the
  256 × 256 × 4096 SI under 50 ms.
- **Frontend:** component tests (Vitest) plus a scripted run in a browser:
  load the example, move the cursor, draw an ROI, fit, draw a profile.
