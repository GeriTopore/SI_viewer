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
- **SI layout:** navigator image on top, with a tabbed pane under it:
  **Spectrum** and **Line profile**. The navigator can be the simultaneous
  HAADF, the SI total counts, or any energy-window or element map.
  - Drawing or dragging a profile line switches to the profile tab.
    Picking the cursor or region tools brings back the spectrum.
  - Image and stack views get the same pane for line profiles only, once the
    line tool is picked or a line exists.
- **Resizable panes:** drag the gaps beside the side panes to set their
  widths, and the bar between the image and the pane below to set its
  height. Double-click a handle to reset it. Sizes and the active tab are
  remembered in the browser.
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

### 4.4 Element maps from the periodic table (M3, prioritised)

Adding an element in the periodic table creates its map from the spectrum
image, and that map appears in the dataset pane like any other.

- **Computing a map:** a map is the net counts of the element's analysis line
  at every pixel. The analysis line is exSpy's choice, as already used for the
  marker label.
  - The counts are integrated over a window of the line ± 1 FWHM. The FWHM is
    taken from the detector's Mn Kα resolution.
  - The background is subtracted using exSpy's background windows on either
    side of the peak (`estimate_background_windows`, then
    `get_lines_intensity(background_windows=…)`).
  - This is fast enough (tens of ms for the example scan) that a map appears as
    soon as the element is ticked.
- **Map options** (per spectrum image, at the foot of the periodic table):
  - background subtraction on/off;
  - integration width in FWHM units;
  - spatial binning (1×, 2×, 4×), because raw EDS maps are sparse.

  Changing an option recomputes all of that spectrum image's computed maps.
- **The table and the maps stay in sync:**
  - Ticking an element adds a computed map, listed with the maps, in the
    element's colour, and marked as computed to tell it apart from a map
    loaded from the file.
  - Unticking an element removes its computed map.
  - **Raw data only:** when a file has an EDS spectrum image, its maps are
    computed from the raw spectrum stream as the file opens, one for each
    element recorded in the file.
    - Velox's stored maps are not listed: they are processed (quantified to
      weight fractions and usually blurred, e.g. σ = 1.7).
    - Files with maps but no spectrum image still show their maps.
- **Deleting maps from the dataset pane:** each map gets a × button, and the
  Delete key removes the focused map.
  - Deleting a map also unticks its element, which removes that element's
    computed map and its line markers.
  - Maps loaded from the file are only removed from the app; the file on disk
    is never changed.
- **Backend:**
  - Computed maps are registered in the session as datasets of the spectrum
    image's file, flagged with `origin: "computed"` and their line and windows.
  - Setting a spectrum image's elements creates or removes its computed maps
    and returns the updated file listing.
  - A delete endpoint removes any map dataset.

### 4.5 Pre-processing tab and FFT (M5, done)

The **Process** tab on the right (before Quantify) processes the data in
view. Filters are non-destructive: they are stored with the dataset and
applied whenever its data is read, so the loaded data is never changed and a
filter can be changed or removed at any time
(`PUT /api/datasets/{id}/filters`, `processing/filters.py`).

- **What the tab works on:**
  - With a spectrum image in the viewer, always that spectrum image,
    whichever maps are overlaid or focused.
  - With a map of a spectrum image shown alone, its spectrum image.
  - Otherwise, the focused dataset.
- **Spectrum images:** Gaussian or mean spatial filter, and Gaussian or
  Savitzky–Golay smoothing along energy. The processing is global: everything
  made from the spectrum image uses the processed data.
  - It reaches cursor and region spectra, the total-counts and energy-window
    navigators, and every element map, including maps added later from the
    periodic table.
  - It also reaches single-spectrum fits, per-pixel fitted maps and
    composition maps.
  - Computed maps are remade when the processing changes. Per-pixel fit
    results are removed and need fitting again.
  - Maps of a spectrum image have no filters of their own; the backend
    refuses them.
- **How the processed data is computed** (no processed copy of the cube):
  - Only linear spatial filters are offered. Summing the filtered image over
    any pixels equals a weighted sum of the raw spectra; the weights are the
    filter's transpose, with scipy's reflecting edges (`spatial_weights`).
    Only pixels near a region's edge need extra work.
  - A weighted sum of the channels of smoothed spectra is a weighted sum of
    the raw channels (`spectral_adjoint`). Window maps and the per-pixel
    fit's solve matrix absorb the smoothing that way.
  - Tests check pixel and region spectra and window maps, edges included,
    against a brute-force processed cube.
- **Images and stacks** (not maps of a spectrum image): Gaussian (σ), median
  or mean (box) filter.
  - NaN pixels (masked composition) stay NaN and don't spread.
  - Stacks are filtered frame by frame.
  - "Apply to all maps" copies a filter between maps loaded from a file.
- **Spectra:** spectral smoothing, also used by their fits.
- **Rebin** (destructive) is in the tab and the dataset context menu. It is
  offered for spectrum images only, and always takes the whole scan. The
  spectrum image is binned by the chosen factors. The file's images and stacks
  on the same scan grid (e.g. the HAADF) get the same x/y binning, so they
  still line up as navigators. Computed maps are recomputed and per-pixel fit
  results are removed. Single maps, images, stacks and spectra can't be
  rebinned on their own.
- Each dataset has a `filter_rev`, bumped on every change. The frontend keys
  its frame, spectrum and thumbnail caches on `revision.filter_rev`, and redoes
  auto contrast when the version changes. A filter change doesn't reset the
  view, cursor or regions.
- Later: Wiener / FFT band-pass filters, and PCA denoising.

**FFT panel.** The top of the right column shows the power spectrum of the
viewer's active layer.
- It transforms the visible part of the image, so zooming in gives a local
  FFT: the centred square, up to 2048 px. Mean removal and a Hann window come
  first, then zero-padding to a power of two, then a radix-2 FFT in a Web
  Worker.
- The display is log power, max-pooled to the canvas so isolated spots stay
  visible, with ×1–×8 zoom on the centre.
- Hovering gives the spatial frequency |k| (Å⁻¹ for length calibrations,
  else the image's units⁻¹ or px⁻¹), the spacing d = 1/|k| (Å) and the
  direction. A ring marks |k|. The
  idle readout gives the Nyquist spacing (2 × pixel size). Checked against
  NumPy on the synthetic atomic image: d = 1.969 Å.

### 4.6 EDX fitting and quantification (exSpy)

**Decisions for M4:**
- **Two phases.**
  - Phase 1: fit one spectrum (all pixels, cursor, a region, or a standalone
    spectrum) and quantify it.
  - Phase 2: per-pixel fitted and quantified maps, as a background job.
- **Placement:** the right-hand column becomes tabs, **Info | Quantify**, with
  **Pre-process** joining in M5. Fit curves are drawn on the existing spectrum
  plot.
- **k-factors:** an editable k-factor per line, defaulting to 1.0 and flagged
  "uncalibrated" until real values are entered. Named presets are saved in the
  browser. Velox files don't store k-factors (Velox uses an ionisation
  cross-section model).
- **Fitting defaults**, tested on exSpy's `EDS_TEM_FePt_nanoparticles`:
  - ignore everything below 0.5 keV (the noise peak);
  - `fit_background()` first, then `fit(bounded=True)` so intensities can't
    go negative;
  - `calibrate_energy_axis("resolution")` and
    `calibrate_xray_lines("energy", bound=10)`, then refit.

  On FePt (with Cu from the grid added) this gives positive intensities for
  every line, with a residual of about 9% of the signal. The unbounded default
  fit returned negative Fe intensities.
- **Test data:** exSpy's FePt spectrum (good counts) for fitting tests; the
  example `.emd` for spectrum-image plumbing.
- **Phase 2 (per-pixel maps):** the summed spectrum is fitted once to fix the
  line positions, widths and minor-line ratios. Each pixel is then solved for
  the line amplitudes and a Legendre-polynomial background by linear least
  squares, as chunked matrix products: 0.2 s for 256×256 × 4096 channels.
  Being linear, the per-pixel intensities add up to the summed fit.
  - It runs as a background job with progress and cancel.
  - The output is a fitted map per element, plus optional at% composition maps
    using the k-factors, with a minimum-counts mask (default 5) so vacuum and
    film pixels aren't read as 100% of one element.

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
- Results export (M7, done): CSV for spectra and tables, TIFF/PNG for maps,
  and `.hspy` for signals (see 4.8).

### 4.7 Line profiles (M6, done)
- **Line tool** on every image view: the ╱ button beside Pan on images and
  stacks, or in the spectrum-image navigator's tools.
  - Drag to draw a line; Shift snaps the angle to 15°.
  - Drag an endpoint or the line itself to change it. Delete removes the
    selected line.
  - Each line shows its length in calibrated units, and its width as a band
    when the width is over 1 px.
  - Lines belong to their view and are cleared when the data is rebinned.
- **Profiles** are computed in the browser from the frames already shown, so
  they update live while dragging, with no network calls (`viewer/profile.ts`).
  - About one sample per pixel along the line, bilinear between pixel
    centres. Each sample is the mean across the line's width.
  - Masked (NaN) pixels and samples off the image are left out.
- **Profile panel** (the Line profile tab under the image), one series per
  visible layer and line.
  - Overlaying element maps (computed, fitted or composition) gives element
    or composition profiles beside the HAADF. No distance × energy cube is
    needed: maps and window sums are linear in the counts.
  - Maps use their element colour. The base image is grey, on its own
    right-hand axis.
  - Further lines are dashed.
  - Options: per-line width and visibility, Normalise (max = 1) and CSV
    export.
  - Hovering the plot marks the position on the lines.
- **Spectrum along the line** (spectrum images): click the profile for the
  spectrum at that point, or drag for a stretch of it.
  - The spectrum is summed over the line's band (a point covers 1 px along
    the line) and drawn in the line's colour on the spectrum plot.
  - The pick is kept as fractions of the line, so it follows the line when it
    moves.

### 4.8 Saving and shortcuts (M7, done)
- **Datasets** (right-click → Save as…, or Save in the Info tab;
  `processing/export.py`):
  - Images and maps: TIFF (32-bit float, calibrated), `.hspy`, CSV.
  - Stacks: TIFF stack, `.hspy`.
  - Spectra: EMSA/MSA, CSV, `.hspy`.
  - Spectrum images: `.hspy`.
  - Data is saved as shown, filters applied. A spectrum image is saved as
    loaded (rebinned, if it was); its processing settings go into the
    metadata (`Processing.si_viewer`), as applying them would need a
    processed copy of the cube.
  - "All maps (.zip)" saves every element map of a file as TIFFs.
  - Velox/exSpy dates like 14.10.2014 are made ISO for the EMSA writer.
- **In the browser:**
  - **Save PNG** under the image saves the whole image as shown (layers,
    colours, contrast) at full resolution with a scale bar. Small scans are
    enlarged by a whole factor to about 1024 px.
  - The spectrum plot saves CSV (every series shown) and PNG.
  - The quantification table and line profiles save as CSV.
- **Keyboard shortcuts** (`?` or the ? button lists them):
  - Tools: C, R, E, P, L, H.
  - View: F to fit, 1 for 1:1, ⌘/Ctrl S to save the PNG.
  - Panes: [ and ] fold the side panes; T switches the bottom tab.
  - Stacks: Home and End for the first and last frame.
  - Existing keys: arrows, Delete, Esc, Space-drag and ⌘/Ctrl O.
- The window title names the dataset in focus.

### 4.9 Later (not in the first draft)
PCA/NMF denoising (`decomposition`), drift correction of EDS spectrum images (applying
the HAADF stack's shifts to the EDS frames, see 4.10), batch processing
recipes across files, and saved sessions.

### 4.10 Drift correction of image stacks (M9, planned)

Registering and averaging HAADF (and other STEM) image stacks with the
**rigidregistration** package (Savitzky et al., Ultramicroscopy 2018; local
copy at `~/Documents/Dev/rigidRegistration`). Only image stacks for now; EDS
spectrum images come later.

**How the package works:**
1. The FFT of every frame is taken (Hann-windowed, mean removed). A Fourier
   mask weights the trustworthy frequencies; `n` is the cutoff (features
   smaller than ~n px are ignored).
2. Every pair of frames is cross-correlated (N(N−1)/2 pairs), giving the
   shift matrices Xᵢⱼ, Yᵢⱼ. Maxima are found by brightest pixel, a Gaussian
   fit (subpixel, robust to unit-cell jumps) or centre of mass.
3. Wrong shifts are found from transitivity (shift i→j + j→k must equal
   i→k) or from nearest-neighbour jumps in the matrix, then corrected along
   good paths.
4. Each frame's shift is the mean of its row. Frames are shifted in Fourier
   space (subpixel), averaged, and cropped to the area every frame covers.

Measured here without pyfftw (NumPy FFT):
- the Nb₃Cl₈ sample (27 × 512²) takes 2 s;
- the CdSe sample (21 × 1024²) takes 5 s;
- the HAADF stack of the example `.emd` (122 × 256²) takes 7 s. Its drift is
  about ±1.5 px.

Pairs grow as N², so large stacks (100 × 1024²) take minutes. pyfftw (up to
20× faster) is optional.

**Integration:**
- Install the package editable from the local copy
  (`pip install -e ~/Documents/Dev/rigidRegistration`, listed in
  `environment.yml`), with pyfftw from conda-forge for speed.
- `processing/registration.py` wraps `imstack`.
  - It converts the frame axis: the package uses (x, y, frame), with frames
    last.
  - It runs headless: matplotlib uses the Agg backend, and nothing is
    printed.
  - It reports progress and handles cancel by swapping the package's `tqdm`
    for an adapter that reports to the job (and raises `Cancelled`).
- Registration runs as a job (like per-pixel fitting): the FFTs, then the pair
  correlations (progress), outliers, correction, then shifting and
  averaging.
- **Non-destructive, like the filters.** The stack keeps its frames, and the
  alignment is stored with it: per-frame shifts, bad frames, parameters and
  crop.
  - Frames are shifted (Fourier shift) and cropped when read, so the slider,
    projections, FFT and line profiles all see aligned frames.
  - An *Original / Aligned* switch shows either. Removing the alignment
    restores the original.
  - The cropped average is added to the file as a new image dataset
    ("HAADF (aligned average)").

**Guided workflow: the user decides at every step.** Nothing runs the whole
pipeline in one go. Each step shows its diagnostics in the viewer, the user
sets that step's parameters while watching them update, and moves on only
when satisfied. These are the notebook's steps (QuickWalkThrough /
DetailedWalkThrough), drawn by the app's own viewers instead of matplotlib:

1. **Frames.** (Removed as a step: the stack viewer already shows the frames
   and their FFT. The frame range is set in the Fourier mask step; bad frames
   in the Shift matrices step, or with *Mark bad* in the Result step.)
2. **Fourier mask.** Three panels for a chosen pair of frames (i, j), as
   `show_Fourier_mask` draws them, updating live as the mask changes:
   - the FFT of frame i with the mask overlaid in colour;
   - the masked FFT;
   - the cross-correlation of i and j.

   Choose the mask type (bandpass, Hann/lowpass, Gaussian, Hamming,
   Blackman, none; elliptical Gaussian later) and the cutoff n. The cutoff
   is drawn as a ring on the FFT, with its spacing in Å.
   - *Compare n* shows the cross-correlation for a row of n values side by
     side, as the notebook's loop over n does.
3. **Peak finding.** The cross-correlation of the pair, zoomed on its
   maximum, with the peaks found marked and the chosen one crossed.
   - Choose: brightest pixel, Gaussian fit or centre of mass, and the
     correlation type (cross, mutual or phase).
   - Gaussian fit: number of peaks, σ guess and window radius, with the
     data/fit cut-outs side by side, as `show_Gaussian_fit` shows them.
   - The pair can be changed at any time; mismatched pairs are the cases to
     check.
4. **Shift matrices.** *Compute shifts* runs all N(N−1)/2 correlations as a
   job, with progress and cancel. It shows Xᵢⱼ and Yᵢⱼ as heatmaps:
   - a diverging colormap with a colour bar;
   - hovering shows i, j and the shift;
   - clicking a cell opens that pair's cross-correlation and peak (step 3's
     view), so a suspicious entry can be checked directly.
5. **Outliers and correction.**
   - Choose the outlier method: transitivity (threshold, max paths) or
     nearest neighbours (max shift).
   - Choose how far off the diagonal to use, the frame range and the bad
     frames.
   - Masked entries are overlaid on the matrices live.
   - *Correct* fills them along good paths and shows the corrected
     matrices beside the originals.
6. **Result.**
   - The drift trajectory (x and y shift against frame, bad frames marked).
   - The cropped average beside its FFT; compare it with a single frame's
     FFT to check information was gained, not blurred.
   - The crop region drawn on a frame.
   - *Back* to any step to change something; *Apply* stores the alignment on
     the stack (non-destructive) and adds the average as a new image dataset.

**UI arrangement:**
- **Alignment workspace in the viewer.**
  - *Align frames…* (Process tab, or right-click a stack) switches the main
    viewer to a workspace for that stack.
  - A stepper at the top: Fourier mask → Peak finding → Shift matrices (with
    outliers) → Result. It has Back / Next and shows which steps
    are done; later steps unlock once earlier ones are done.
  - The panels of the current step fill the viewer: images in the app's
    WebGL image view (zoom, contrast, hover readout), plots in uPlot, and
    matrices as heatmaps.
  - *Close* returns to the normal viewer; the workspace's state is kept until
    the stack is closed.
- **Parameters in the Process tab.** Its *Align frames* section shows the
  controls of the current step, simple ones first and the rest under
  *Advanced*.
  - The pair (i, j) used for previews is chosen there, or by clicking a
    matrix cell.
- **Processing order in the Process tab:** ① *Align frames* (stacks),
  ② *Rebin* (spectrum images), ③ *Filters*, each shown only when it applies.
- **After Apply:**
  - The stack shows aligned frames, with an *Original / Aligned* switch.
  - The *Alignment* tab under the image keeps the trajectory and summary.
  - *Align frames…* reopens the workspace with the stored parameters, to
    refine them.

**Backend:**
- A registration session per stack, held in memory: the `imstack` object
  with its FFTs, shift matrices and masks.
- Each step's parameters are applied to the session, and the previews come
  back as float32 arrays like every other frame.
- Preview requests (mask, one pair's correlation, peak fit, outlier mask,
  correction) are fast; only *Compute shifts* and the final shift-and-average
  run as jobs.
- Memory: the package keeps complex128 FFTs of every frame (122 × 256² is
  128 MB; 100 × 1024² would be 1.6 GB). Large stacks warn first, and may use
  complex64 or a cropped region.

**Rollout:**
1. **M9a: workspace, Frames and Fourier mask** (done).
   - Done: `processing/registration.py`, `api/registration.py`, and
     `AlignmentWorkspace` + `AlignSection` in the frontend.
   - Cross-correlation previews come from the session's own FFTs
     (`imstack.fftstack`), so they match what the registration will use.
   - Layout after review:
     - Steps 1–2 show the frame in the viewer, which feeds the FFT panel
       (log power) instead of extra FFT views.
     - The mask is drawn on the FFT panel (`viewer/fourierMask.ts`, the
       package's formulas in terms of frequency), with its cutoff ring and a
       *Mask* show/hide toggle.
     - The masked-FFT view and the per-frame intensity plot were dropped, and
       the diagnostic panels have no readout bar.
   - FFTW's wisdom is kept in `~/.cache/si-viewer`.
   - Install the package (editable) and write the headless wrapper
     (progress, cancel, Agg backend).
   - The registration session API.
   - The workspace shell with the stepper, and steps 1–2 with live mask
     previews and *Compare n*.
2. **M9b: Peak finding and Shift matrices** (done).
   - Previews run the package's own methods on a stand-in object with its
     own FFTW plan. They never touch the `imstack` that the shift job uses,
     and they give the job's numbers exactly (tested to 1e-5 px for pixel,
     Gaussian fit and centre of mass).
   - For i > j the preview analyses (j, i) and mirrors it, as the package
     fills the lower triangle by negation.
   - The Gaussian fit's candidates are the package's n brightest pixels
     (often neighbours on one ridge), each shown with its cut-out and fit.
   - Matrices use a diverging RdBu colormap with a symmetric range.
   - After review, the outlier masking moved into the Shift matrices step
     (steps 4 and 5 merged).
     - The mask (outliers, pairs too far off the diagonal, bad frames) is
       drawn over both matrices as dark cells, with a *Mask* toggle to hide
       it.
     - Its settings are in the Process tab: transitivity threshold and
       paths, or the neighbour check's maximum shift, and the off-diagonal
       limit.
     - It uses the package's `get_outliers` / `get_outliers_NN`,
       `mask_off_diagonal`, `set_bad_images` and `refine_mask`, and updates
       live (0.2 s for 122 frames).
     - The steps are now Frames, Fourier mask, Peak finding, Shift matrices
       and Result.
   - Correction also happens in step 4, and re-runs live when the mask
     changes.
     - Masked shifts are filled from paths of good pairs (the package's
       `make_corrected_Rij`), and each frame's shift is the mean of its row
       (`get_imshifts`).
     - The step shows the measured Δx/Δy (mask overlaid) above the corrected
       ones (same colour range), with a 2D drift plot centred below.
     - The drift plot shows each frame's position relative to the mean,
       shaded from the first frame to the last, with equal axes (Å when
       calibrated) and a net drift arrow.
     - The correction loop is the package's, with two safety changes. Rounds
       that fill nothing double the paths at once, where the package waits
       until its 1000th round. Paths through frames that don't exist (its
       generator overruns small stacks) are skipped. Past 640 paths per pair
       it stops with an error naming the frames, where the package would
       loop forever (e.g. a group of frames with no good pair to the rest).
   - Reproducing the DetailedWalkThrough notebook on Nb₃Cl₈ led to these
     changes. With them the app gives the notebook's shift matrices (to
     4e-6 px), outlier mask and drift exactly; this is tested.
     - The elliptical Gaussian mask (`makeFourierMask_eg`: n1, n2, θ in
       degrees) was added. Its cutoff is drawn as a rotated ellipse on the
       FFT panel.
     - Frames are normalised to 0–1 before registration, as the notebooks
       do (uint16 / 2¹⁶; floats by their maximum). The Gaussian peak fit
       depends on the data scale, and raw counts changed some pair shifts by
       tens of pixels.
     - The correction wraps path indices that run past the stack's ends, as
       NumPy does in the package, instead of dropping those paths. They are
       still valid chains of pairs, and the package relies on them near the
       first frames.
     - The notebook's hand edits of single mask entries aren't supported
       yet; on this data they don't change the drift.
   - The Result step (M9c) shifts, averages and applies.
   - Step 3 (pair cross-correlation, peaks, Gaussian-fit cut-outs).
   - Step 4 (the *Compute shifts* job, interactive Xᵢⱼ/Yᵢⱼ heatmaps, click
     a cell to inspect the pair).
3. **M9c: Result and Apply.** (Outliers were merged into step 4 in M9b.)
   - Step 5, Result (done):
     - The good frames shifted into register as the package's
       `get_averaged_image` does (Fourier shift of the loaded data), with a
       slider (← / →) to scroll through them.
     - By default they are cropped to a square: the largest one centred in
       the area every frame covers (the package's `crop_image` rectangle).
     - *Original* shows the frames as loaded, with that square drawn where
       it falls on each frame (square + the frame's drift).
     - Below, the average of the aligned frames, cropped the same way; it
       feeds the FFT panel.
     - Tested against the package on the notebook's Nb3Cl8 run (average
       and crop) and on a synthetic stack (crop size, the square's place
       on each frame).
   - Apply (done):
     - Stores the alignment with the stack and shows it: while *Aligned* is
       on, the stack's data is the aligned stack (kept frames, cropped to
       the square, image offsets moved to its corner), so the slider,
       projections, line profiles, FFT, export and thumbnails all use it.
       The loaded frames are kept; *Original / Aligned* under the image
       switches (bumping `filter_rev` so views refetch).
     - The average is added after the stack as an image
       ("… (aligned average)"); applying again replaces it.
     - The *Alignment* tab under the stack shows the drift plot and how
       the correction was made; the Process tab summarises it, with
       *Remove alignment* (restores the loaded frames, removes the average).
     - The registration keeps working on the loaded frames, so
       *Align frames…* refines an applied alignment. Rebinning the scan
       removes the alignment first (its shifts are in the old pixels).
   - Tests: a synthetic stack with known shifts (under 0.2 px), and the
     sample stacks.
4. **M9d: extras.**
   - The dual mask (two-pass registration) and the clipping dimension
     (elliptical masks are done).
   - Editing single entries or blocks of the outlier mask by hand, as the
     notebook does (done: *Edit mask* in the matrices; edits are stored per
     pair, applied to both triangles, and kept with the applied alignment).
   - Apply the shifts to other stacks of the same scan.
   - Save the shifts (CSV) and a report (PNG/PDF of the steps): done. CSV
     of shifts and drift in px and Å; 2-page PDF: settings, drift, average
     + FFT; matrices measured/corrected.

### 4.11 EELS spectrum images (M10, in progress)

**Example data** (`examples/`):

| File | Navigation | Energy | Notes |
|---|---|---|---|
| `STEM SI_EELS Spectrum Image_summed.dm4` | 120 × 120 | 3456 ch, 362–673 eV (0.09 eV) | Core loss only. GIF Continuum K3 HR, 300 kV, α 2.68 / β 55.41 mrad. A "Summed" image of a parent SI's frames (drift-aligned in GMS). `SI.Acquisition` has the date, start/end time and the survey image's ID and rectangle. |
| `CuZn_EELS_mapping_tutorial.hspy` | 40 × 50 | 162 ch, 700–1988 eV (8 eV) | Core loss only (Cu, Zn L edges), 200 kV. |
| `LSMO_linescan.hspy` + `LSMO_linescan_low_loss.hspy` | line scan, 40 points (3.2 nm) | 586 ch, 428.5–721 eV / 1024 ch, −50–461.5 eV | A core-loss / low-loss pair. Titles end in "(high-loss)" / "(low-loss)" (from the original .dm3 names). No date or time, no role marker. |
| `STEM SI_EELS HL SI_summed.dm4` + `STEM SI_EELS LL SI_summed.dm4` | 96 × 415 | 3456 ch each (550 MB each) | A dual-EELS pair, 300 kV. The low loss's zero-loss peak is at −3.15 eV. The tags `EELS.Acquisition.Is dual acquire low-loss` (1 / 0) and `Dual acquire sibling.Unique Image ID` (the other file's `Processing…Parent.Unique Image ID`) tie them together. |

All of them have `signal_type = "EELS"` in their metadata, so HyperSpy loads
them as `EELSSpectrum`.

**Where we are.**
- `.dm3`, `.dm4` and `.hspy` already open.
- The 2D EELS SIs get the generic SI viewer (navigator, spectra, ROIs,
  profiles). EDS features stay off because they need `signal_type` EDS.
- Line scans (1D navigation) are classed as "other" and can't be viewed.
- `.dm5` isn't read by RosettaSciIO 0.15 (open issue
  hyperspy/rosettasciio#315).

**M10a: reading.**
- **Line-scan SIs:** a spectrum image with one navigation axis. The
  navigator is an image of energy (x) against position (y), as GMS shows
  them. Click a row for its spectrum, or drag over rows to sum them.
- **`.dm5`:** a reader of our own in `si_viewer/io/dm5.py` (DM5 is HDF5),
  like the VeloxLite shim. It sets the calibrations, puts the DM tags in
  `original_metadata` and maps them to `metadata` the way RosettaSciIO's DM
  reader does. Needs a sample file. Add `.dm5` to the supported suffixes and
  `.gitignore`.
- **EELS SIs in memory:** load them into RAM under `SI_IN_MEMORY_LIMIT`, as
  EDS SIs are. The .dm4 opens lazily today.
- **Tests:** `conftest.py` finds files in `examples/`, with one fixture per
  file (skipped when missing). Check each EELS example's kind, shape, energy
  calibration and signal type.

**M10b: EDS or EELS.** Each SI and spectrum gets a `technique` (EDS, EELS or
unknown), from, in order:
1. `metadata.Signal.signal_type` (EDS_TEM / EDS_SEM / EELS), set by the
   Velox, DM and hspy readers;
2. the DM tag `Meta Data.Signal`;
3. the energy axis: named "Energy loss" → EELS; in keV from about 0 → EDS.

When it's unknown, or the guess is wrong, a *Signal type* choice in the
Info tab sets it (`set_signal_type`, so exSpy's class follows).

The technique decides what is offered:
- **EDS only:** X-ray lines, periodic-table maps, the Quantify tab, the
  maps computed on opening.
- **EELS only:** the EELS tools (M10d).
- **Both:** navigator, ROIs, spectra, profiles, filters, rebinning, export.

**M10c: low loss and core loss.** No role guessing: a role comes from the
file's metadata when it has a marker, else the user says.
- **Role:** every EELS SI is *low loss* or *core loss*, with a `role_source`:
  - `metadata`: DM's `EELS.Acquisition.Is dual acquire low-loss` tag (1: low
    loss, 0: core loss);
  - `user`: the LL / HL switch on the *Loss* row of the Info tab, beside
    where the role came from;
  - `default`: no marker, so core loss.

  The dataset list tags them LL / HL. Switching the signal type to EELS reads
  the metadata role again; leaving EELS clears it. Only tags are read, never
  spectra. A low loss is what unlocks the low-loss tools (M10d).
- **Link groups:** linked SIs form a group of two or more. A group has at
  most one low loss, and any number of core losses (a dual acquisition plus
  more core-loss ranges). Roles don't decide whether SIs can be linked, only
  the scan. They need:
  - the same navigation shape and scale (within 1 %, same units);
  - the same beam energy, where both give one.
- **Linking is the user's call:** nothing is linked automatically. Opening a
  file only reads each EELS SI's role (no names, DM tags or IDs are matched),
  however well two SIs seem to fit. Right-click → *Link with…* lists every
  compatible SI (its role, and whether it's in another group); picking one
  moves this SI into that SI's group (or makes a group of the two).
  *Unlink* leaves the group (a group of one dissolves). A group that would
  get a second low loss is refused. Switching a member to LL is refused too
  when the group has one.
- **Viewer:**
  - The pane under the image gets one tab per group member, including the
    shown one: *Low loss*, and *Core loss* (or its energy range, e.g.
    "430–1640 eV", when there are several core losses). Each has its own
    energy axis, at the same cursor and ROIs. The tabs keep their ids when
    another member is shown, so the split stays. They show one at a time
    or two side by side (the split
    pane). A line scan's image follows the focused tab.
  - Unlinked SIs keep the single *Spectrum* tab. Any member can be viewed alone.
- **Processing:** rebinning and spatial filters apply to every member of a
  group, so their pixels keep matching. Rebinning uses the same factors for
  all, energy included; a factor too large for any member rebins none.
- **Backend:** `Session.eels_groups` maps a group id to its members, and
  `Dataset.eels_group` says which group a dataset is in. The summary gets
  `eels: {role, role_source, linked, beam_energy, convergence_angle,
  collection_angle}`: `linked` has the other members' ids (low loss first,
  then core losses by energy offset), and the angles come from the metadata
  and are editable in the Info tab, since thickness and deconvolution use
  them. API: `GET …/eels-partners` (link candidates: `{id, title, file_id,
  role, group}`), `PUT …/eels-partner` `{partner_id}` (join its group; null:
  leave), `PUT …/eels-role` `{role}`. Each returns `files`. Rebin, filters and
  signal-type changes return the other members as `partners` and
  `partner_files`.

**M10d: EELS analysis, in progress** (an EELS section in the Process tab). Each step runs
as a job with progress, and its results are new datasets under the SI, like
the fitted maps.
All of these use the low loss of the SI's link group (the one marked or
read as low loss, M10c); they're disabled, with the reason, when the group
has none.
- **Zero-loss alignment** (built): a *Zero-loss alignment* section in the
  Process tab, after Rebin (disabled, "No low loss in its group", otherwise).
  exSpy's `align_zero_loss_peak` on the group's low loss with every linked core
  loss of the same scan as `also_align`, with `calibrate` always on (the peak
  ends up at 0 eV; the core losses' axes move by the same amount, and each is
  cropped on its own axis). Run as a job (progress by phase, cancel); the
  steps are exSpy's, taken apart to keep the shifts.
  - **Options:** *Sub-pixel* (default; cross-correlation within ±3 eV of the
    peak) and *Crop* (default; else the emptied channels are 0).
  - **Overwrites, like Rebin:** the loaded data of the low loss and its core
    losses is replaced (the file is unchanged), with the warning "Overwrites
    the loaded data; the file is unchanged." Not reversible in the app. Each
    gets a `zlp_aligned` entry ("sub-pixel, cropped"), shown as *ZLP aligned*
    in the Info tab. Running it again is allowed (shifts ~0).
  - **Remade:** window maps, thickness and background-removed spectra follow
    the axis (their windows and a fixed threshold move by the mean shift, so
    they stay on the same features; one that leaves the energy range is
    removed); a ZLP shift is removed on rebinning.
  - **Result:** a dataset under the low loss (`origin "zlp-shift"`, "ZLP
    shift", eV): an image for a 2D scan, a profile for a line scan. Replaced
    by a later alignment; deletable.
  - API: `POST …/zlp-alignment {options}` (job; any member of the group;
    400 not EELS / no low loss, 422 bad options). Result `{file,
    partner_files, dataset_id, aligned, shift}`.
- **Thickness** (built): a *Thickness* section in the Process tab, for an
  EELS spectrum image or line scan that is a low loss or linked to one
  (disabled, "No low loss in its group", otherwise). It always works on the
  group's low loss. exSpy's `estimate_thickness`, run on chunks as a job
  (progress, cancel).
  - **Threshold:** *Auto* (default; `estimate_elastic_scattering_threshold` at
    every pixel, with a *Search window*, 10 eV; the `tol` it takes over all
    pixels is found first, so chunking changes nothing; pixels with no
    inflexion get NaN) or *Fixed* (one energy, typed or dragged, snapped to a
    channel). With *Show on spectrum* it is a vertical line on the low-loss
    plot: its own plot or the *Low loss* tab. In Auto it sits at the estimate
    for the summed spectrum; dragging it, in either mode, switches to Fixed.
  - **Output:** t/λ, or *Absolute (nm)* from a typed mean free path or from a
    density (exSpy's Iakoubovskii λ and angular correction, from the beam
    energy and angles in the metadata; Apply is disabled, with the missing
    ones named, if any is absent).
  - **Result:** a dataset under the low loss (`origin "thickness"`, titled "t/λ"
    or "Thickness (nm)"): an image for a 2D scan, a profile along the scan
    (a 1D spectrum dataset on the position axis) for a line scan. Its summary
    has `thickness: {source_id, threshold: "auto" | number, window, output,
    mean_free_path, density, units}`. One per low loss: running again replaces
    it (same id, new revision), and the section shows its settings. It is
    deletable, recomputed when the low loss is rebinned or filtered (removed
    if the settings no longer fit), and removed with its source, when the
    source stops being a low loss or EELS, or when the file closes.
  - **Processed data:** like the maps, it uses the filtered data: spectral
    smoothing on the spectra, the spatial filter on the finished map.
  - API: `POST …/thickness/threshold {window}` (the Auto estimate for the
    summed spectrum), `POST …/thickness {options}` (job; any member of the
    group).
- **Plural-scattering deconvolution** (built): a *Deconvolution* section in
  the Process tab for EELS spectrum images and line scans, always with the
  group's low loss (for a background-removed spectrum image, its source's).
  - **Methods:** *Fourier-log* (the low loss only; optionally keeping the
    zero-loss peak), *Fourier-ratio* (background-removed spectra only; FWHM
    typed or that of the summed low loss' zero-loss peak; power-law
    extrapolation on or off) and *Richardson–Lucy* (iterations; PSF the low
    loss, or its zero-loss peak to sharpen; not on background-removed
    spectra, nor the low loss by itself). A method that doesn't fit the
    dataset is disabled, with the reason.
  - **Zero-loss peak:** the low loss up to a threshold, Auto (per pixel, as
    for the thickness; the summed spectrum's where a pixel has none) or Fixed.
  - **Preview:** with *Show on spectrum*, the deconvolved spectrum of the sum,
    the cursor and each region is drawn dashed over the original (its own
    plot or the linked tab), with the threshold and FWHM used.
  - **Result:** a new spectrum image under the source (`origin
    "deconvolved"`, "<source> (Fourier-ratio)"), with its settings to change
    and *Re-run*. A snapshot: later changes to the source leave it as it is.
    Deletable.
  - Fourier-log and Fourier-ratio are exSpy's (equal to its own calls); the
    power-law extrapolation is done per spectrum, so one with no counts at
    its end doesn't stop the rest. Richardson–Lucy is our own (FFT, every
    pixel at once, PSF normalised so the counts keep their scale).
  - API: `POST …/deconvolution/preview {options, selections}`, `POST
    …/deconvolution {options}` (job), `PUT …/deconvolution` (re-run).

**M10e: background removal** (EELS only; spectrum images, line scans and
single spectra). A *Background removal* section in the Process tab.
- **Two windows on the spectrum:** a *Background* window (the fit range,
  before the edge) and a *Signal* window (the range kept). Both are drawn as
  labelled bands, moved and resized by dragging like energy windows, and
  typed in the section. The background window must end at or before the
  signal window starts.
- **Model:** power law (default), exponential, polynomial (with its order)
  or offset; *Fast* (the closed-form estimate, instant) or *Full*
  (least-squares fit at every pixel).
- **Preview:** each spectrum on the plot (sum, cursor, regions) gets its
  fitted background as a dashed curve over both windows; a *Subtracted*
  toggle shows the remainders over the signal window instead.
- **Apply** runs as a job and saves a new SI under the source: just the
  signal window, background subtracted, titled by its signal window
  ("635–680 eV, bg removed"; it can be renamed). Several removals (one per edge) give several SIs.
  - For the power law its exponent *r* is saved as a map under the new SI,
    to show where the fit fails.
  - Selecting the new SI shows its settings; changing them and *Re-run*
    replaces it in place.
  - Rebinning the source recomputes its background-removed SIs.
- **Log fit** (power law and exponential): *Fast | Log | Full*; Log is a
  linear least-squares fit of ln I to ln E (or E), every pixel at once;
  channels without counts are left out.
- **LBA** (spectrum images and line scans): each background-window channel
  is smoothed across the scan by a Gaussian (FWHM in px), each pixel's
  window scaled back to its own mean, then fitted; the fit is subtracted from
  the raw spectrum. The cursor's preview uses its neighbourhood's LBA.

**M10e: later.**
- Edge maps from the periodic table using exSpy's EELS edges (the EELS
  version of M3).
- `EELSModel` fitting with the low loss for quantification.
- Spike removal.

**Verification on the examples.**
- **Detection:** all the EELS files are classed EELS; the Velox `.emd` SI
  is classed EDS.
- **Dual-EELS .dm4 pair:** the roles come out low loss / core loss from the
  metadata, without reading a spectrum. Opening them links nothing; *Link
  with…* does.
- **LSMO pair:** opening links nothing; link it by hand. Both are core loss
  until the low loss is marked by hand. Its thickness map and Fourier-ratio result
  equal exSpy's own calls on the same signals.
- **LSMO line scan:** it is viewable, with spectra at single points and
  over ranges of rows.
- **CuZn and the older .dm4:** core loss (by default) with nothing linked;
  *Link with…* offers nothing; the EELS tools that need a low loss are
  disabled, with the reason given.

**Open questions.**
- EDS tests: the only `.emd` in `examples/` is a HAADF stack. Which EDS SI
  `.emd` should the tests use?
- A `.dm5` sample.

### 4.12 Grid and merged views, split bottom pane (M11, done)

For every kind of data: the images in the viewer can be shown **merged**
(one composite, as overlays work now) or as a **grid** (one panel each).
The pane under the image can be split to show two of its tabs side by side.

**M11a: Grid / Merged switch.**
- **The switch:** a *Merged | Grid* toggle in the viewer toolbar, plus a
  shortcut (G).
  - The default follows the content: **merged** for element maps over a
    spectrum image (as now), **grid** for several plain images (ADF, BF,
    DPC… from one `.emd`).
  - Once the user picks a mode, it is remembered for that kind of view
    (spectrum image, images).
- **What can be shown together:**
  - The selection rules relax. Several plain images can now be selected
    together (⌘-click), which today is refused ("Images can only be
    overlaid on a spectrum image").
  - Merged needs the same pixel size, as overlays do now. Grid accepts mixed
    sizes.
  - Stacks and line scans stay on their own, as now.
- **Grid layout:**
  - One panel per layer: the SI navigator (ADF or total counts) first, then
    the maps or images in selection order.
  - Columns are chosen to make the panels as large as possible for their
    aspect ratio.
  - Each panel shows its name and value readout; only the first shows a
    scale bar.
- **Linked panels:**
  - Panels of the same pixel size share pan and zoom (one view transform),
    so zooming one zooms all.
  - The SI cursor, ROIs and profile lines are drawn on every panel, and can
    be moved or drawn from any of them.
  - Mixed sizes pan and zoom independently.
- **Contrast:** each panel keeps its own display settings (contrast,
  colormap or element colour, as in merged). Clicking a panel makes it the
  one the toolbar edits (the "editing" chip, as for layers now).
- **Profiles:** the Line profile tab plots every panel's profile, as it does
  for layers.
- **Save PNG:** saves the grid as shown, with a label per panel, or the
  composite in merged mode.
- **Implementation:**
  - A `ViewerGrid` built from the same `ViewLayer` list that
    `LayeredViewer` builds, so both modes share loading, contrast and
    profiles.
  - `ImageView` gains an optional shared view transform from the display
    store, so linked panels stay in step.
- **Maps for EELS (and EDS) from energy windows:**
  - Shift-drag on the spectrum makes an energy window: a map dataset under
    the SI (e.g. "640–660 eV") that joins the view and can join the grid or
    the composite like an element map. Windows are edited by dragging their
    edges or inside, and deleted with Delete or the right-click menu.
  - The grey *main window* (the whole range by default) is the *Virtual
    image* navigator: the SI summed over it. It can be resized and moved but
    not deleted; double-click it to get the whole range back.
  - Background-subtracted EELS edge maps come with M10e.

**M11b: split bottom pane.**
- Any two tabs of the pane under the image (Spectrum, Line profile, and
  later Low loss / Core loss) can be shown side by side or stacked, with a
  draggable divider.
- Two ways to split:
  - a *Split* button in the tab bar, which puts the next tab beside (or
    below, in the last used orientation) the current one;
  - dragging a tab onto the pane, with the target half highlighted while
    dragging, as editors do: the left or right third makes two columns with
    the tab on that side; on the middle third, the upper or lower half makes
    two rows.
- Dragging a tab back onto the tab bar, or *Unsplit*, returns to one tab.
- The split, its orientation and each orientation's divider position are
  remembered.
- Each half keeps its own controls (axis zoom, Linear/Log, legend).
- The *focused* half (the one last clicked) is the one the T key cycles and
  the line-scan image follows (low/core-loss tab).

**Order:** after M10c. Linked low/core-loss pairs then gain the split pane.

### 4.13 In-situ EELS series (M12, in progress)

Gatan in-situ acquisitions (GMS 3.61 and later) save, per detector, a `.dm4`
plus the frames in `.raw` chunks (`<name>.raw`, `<name>.raw_1`, …). A
`.dm5` is only written if saved by hand, so the `.dm4` must be enough.

**What the `.dm4` holds** (`ImageTags.Insitu`):
- `Raw File Format Info`: `Dimensions/Data0…` (Data0 = x, Data1 = y,
  Data2 = energy), `Data Type` (DM type code), `Data Size (bytes)`,
  `Frame Size (bytes)`, `Calibrations/dim-N` (origin, scale, unit).
- `Recorded`: `# Frames`, `Duration (s)`, `Experiment Start Time`, and
  `Sync/Per Frame/Time Since Start (ns)`, a 64-bit time per frame stored as
  pairs of 32-bit words (low, high). HyperSpy gives it as a list (a dict with
  "[i]" keys, sorted as text, is also handled). The times are when each frame
  was finished: 18.4 s for the first, exactly `Duration` for the last.
- The tag group is `In-situ` in the raw tag dict (`Insitu` as an attribute).
- Type codes are DM's (2 float32, 6 uint8, 10 uint16, …; as RosettaSciIO).
- Its own image is one frame and is not used.

**`.raw` layout:** frames back to back across the chunks (a frame can
straddle two files); within a frame the DM order, Data0 fastest, i.e. a
NumPy array of shape (Data2, Data1, Data0) = (E, y, x) for a spectrum image
and (y, x) for an image. Checked on `InSitu (3)` (100 × 100 × 3456, float32,
208 frames, 28.75 GB in 3 chunks): the energy-summed frame matches the
simultaneous ADF (400 × 400, 4× finer sampling of the same field).
Unconfirmed on a non-square scan.

**Read costs** (USB exFAT drive, about 40 MB/s): a whole frame 3.6 s; one
pixel's spectrum in a frame 3.9 s (one read per energy plane); a 100-channel
window map 0.1 s; one pass over the series about 12 min. So the unit of
reading is the whole frame, and anything over all frames is one pass.

**M12a — Reader.** `io/insitu.py`: recognise an in-situ `.dm4` (the
`Insitu` tag plus `.raw*` beside it); a lazy dask array with one chunk per
frame (one sequential read per frame, across chunk boundaries); the time
axis from the per-frame times (else duration ÷ frames); fewer complete
frames than recorded (an aborted run) are used, with a note. Opening the
SI's `.dm4` also opens the simultaneous ADF as a stack in the same file
entry. Tests on small synthetic `.raw` sets (non-square, a straddling frame,
a truncated last frame) and on `examples/InSitu (3)` when present.
Done. Layout: (Time, y, x, E) for the SI, (Time, y, x) for an image series;
`metadata.Insitu` holds `frames`, `recorded_frames`, exact `frame_times_s`,
`time_source`, `note`. The Time axis is uniform (first time, mean spacing).
The SI was first its own dataset kind (`insitu_si`) that could not be loaded
whole; M12b made it a dataset holding one frame (see below). The ADF series is
an ordinary stack. A chunk file
shorter than the first, followed by another, ends the stream (a copy in
progress), unless the files already cover every recorded frame.

**M12b — Frame browsing.** Done. The series is shown one frame at a time:
the current frame is read whole into memory and is an ordinary spectrum image
to every existing tool; neighbouring frames are read ahead into a small cache.
A frame slider (frame, time in s) with Frame / Mean / Sum over a frame range;
the ADF frame at the same time as navigator.

- *Design.* The series is ONE dataset (a row in the list, kind
  `spectrum_image`, badge "IS EELS"/"IS LL"/"IS HL") whose signal is the
  frame held, in memory, with the frame's axes and metadata (no `Insitu`
  tags). `Dataset.insitu` is an `InsituSeries` (`insitu_series.py`): the lazy
  series signal, the frame cache, the read-ahead and the `ViewState` (frame, or
  mean/sum of frames a…b). `DatasetKind.INSITU_SI` is gone; summaries carry
  `insitu` (frames, times, sizes, `view`). Before the first use the signal is
  a lazy placeholder of frame 0; `materialise()` reads that frame (one read,
  never the series). A new frame swaps the signal in with `Dataset.refresh_data`:
  same id and revision (so filters, windows, cursor, regions, contrast and zoom
  stay), `data_rev` bumped (clients refetch: `dataVersion` includes it,
  `contrastVersion` does not), caches replaced. Energy-window maps (and
  computed element maps) are remade for the new data, also without a new
  revision; fits are removed. Results that are snapshots of a frame stay as
  made: background removal is titled "… (frame 37)" and redoing it takes the
  frame shown now; thickness and ZLP shift are not retitled.
- *Reading.* The unit is the whole frame, via the dask graph of M12a, in the
  calling thread (request threadpool or job). `FrameCache`: LRU bounded by
  bytes (`FRAME_CACHE_BYTES` = 1.5 GB), cached frames read-only (shared with
  the view). After a frame is shown, frames n+1 and n−1 are read on a single
  background thread; tasks of an earlier frame not yet started are dropped; a
  frame being read is waited for, not read twice. A range reduction reads the
  frames (cache hits are used, the others are not cached) and is the only
  multi-frame read.
- *API.* `PUT /api/datasets/{id}/insitu/frame {frame}` → `{dataset, file}`
  (416 out of range); `POST /api/datasets/{id}/insitu/reduce {mode, start, end}`
  → job "insitu-reduce", progress "Frame 12 of 40", cancellable, result
  `{file, dataset_id}`; `GET /api/datasets/{id}/insitu/adf` → the ADF image of
  the view. `/navigators` lists the ADF (source id `insitu-adf`) first when the
  file has an ADF stack with the series' frame count whose frame is the scan
  or an integer multiple of it with the same field of view (here 4×); a
  Mean/Sum view gets the ADF mean/sum over the same range (computed in the job
  and memoised).
- *Sub-scan navigators.* Gatan's ADF is acquired k× finer than the SI scan
  (400² against 100²). Any image k× finer over the same field (±1 %; `subscan.py`,
  `/navigators` `factor` and `dataset_id`) is a navigator at its own resolution
  and calibration, not binned. The view works in navigator pixels but cursor,
  regions and spectra stay in SI pixels: `views/si/scan.ts` divides pointer
  positions by k and multiplies the overlay transform by k (a k × k box per SI
  pixel), and maps drawn over it are stretched k× (`Frame.bin = k`).
- *UI.* Frame bar under the navigator (`InsituFrameBar`): slider 1…N with
  "37 / 208 · 128.4 s", Frame / Mean / Sum. The slider reads on release (a
  click or key: after 150 ms), one request at a time, the latest frame wins;
  the last frame stays until the new one arrives, with "Reading frame 37…".
  Mean/Sum: From / To and Apply, with a progress bar and Cancel; the result
  becomes the view. A chip on the navigator names what is held. ←/→ (⇧ ten) and
  Home/End step the frames; in a series the cursor is nudged with ⌥←/⌥→.
  Contrast is kept from frame to frame. The Info tab has an "In-situ series"
  section (frames and the missing-frames note, time span, frame time, duration,
  start, frame size, raw files and size). The list's shape column is short
  ("208×100×100|3456") and ellipsizes before the title does.
- *Rebin.* A series is rebinned as a series: the dialog adds a *Time* row
  (frames per new frame). Every frame read from then on is binned as it is
  read (y, x, energy in summed blocks, remainder cropped; time: the sum of
  that many recorded frames, at their mean time); the binning stacks. The
  view becomes the binned frame where it started (a range becomes one frame).
  The ADF navigator keeps its resolution (the sub-scan factor grows) and its
  frames are binned in time the same way. Linked series get the same factors.
- *Disabled on a frame.* Zero-loss alignment (the next frame would undo it).
  Everything else works on the frame: spectra, regions, line profiles, energy
  windows and maps, spatial/spectral filters, background removal, thickness
  (set the role to low loss), FFT, Save (file named with the frame), plot export.
- *Not done.* Thickness / ZLP-shift titles don't name their frame; a
  background-removed result is not remade when the frame changes.

**M12c — Time scans.** Done. Right-click the image of a series: *Time scan
at the cursor (x, y[, n×n])* and *Time scan of <region>* for each region under
the pointer. A job (progress in the frame bar's ring, cancel by clicking it)
reads every frame (of the current time binning) and makes a line scan under
the series: row k is the mean spectrum of the cursor's block or the region in
frame k, on a "Time" axis (s; first frame's time, median interval; the frame
times are kept in the metadata). Linked series on the same scan get one too,
linked in turn. `POST …/insitu/time-scan {pixel | region, name}`.
Line scans (from profiles or over time) can be rebinned along both axes.

(The earlier M12c time-series job and M12d Time tab were dropped.)

**M12e — Aligned sum.** Done for series with a simultaneous ADF (and for
imported shifts). Process tab *Aligned sum* on an in-situ SI:
- *Shifts:* the alignment applied to the ADF stack (Align frames), taken
  automatically: each frame's shift is minus its drift, divided by the
  sub-scan factor k (ADF px per scan px); or a `.npy` file (frames, 2): the
  (x, y) shift that aligns each frame, in ADF or scan pixels, NaN leaving a
  frame out (shape, frame count, finiteness and size are checked; the check
  shows frames used, the largest shift and the area kept). Aligned stacks
  export their shifts in this format (*Shifts (.npy)*).
- *Applying them:* rigidregistration's `generateShiftedImage` is 2D only
  and keeps the modulus, so `processing/insitu_align.py` does the same
  Fourier shift for a whole frame (every energy channel), keeping the real
  part (integer shifts give the package's result exactly). The package is
  not changed.
- *The job:* every recorded frame read once (binned in y, x, E as the series
  is; not in time), shifted, **summed** (counts, not a mean), cropped to the
  area every frame covers (the package's `crop_image`). Progress in the
  section and the frame bar's ring; the result is a new SI under the series
  (`origin "aligned-sum"`, deletable), calibrated and offset to the crop.
- *Next:* series without an ADF: align a stack of virtual images instead.

**Later:** in-situ low-/core-loss pairs; an optional local copy or
spectrum-major conversion for fast spectra over time; `.dm5`; in-situ
4D-STEM.

### 4.14 Experiments: a folder opened together (done)

GMS saves one acquisition as several files in one folder (survey, ADF, the
EELS spectrum images). *Open all in folder* opens them as one experiment,
listed under the folder's name by role (Survey, scan images, EELS LL, EELS
HL, EDS, others), with the experiment's EELS spectrum images linked on opening
(a dual EELS pair by DM's sibling ids, else ones of the same scan and DM
spectrum imaging run). An in-situ spectrum image's ADF series opens with it,
so it is not opened again. Opening files one by one links nothing.

The survey image (DM keyword "Survey") is a **Survey** dataset showing the
rectangles GMS drew on it: where the spectrum image was taken and the spatial
drift reference (`io/dm_experiment.py`, from the survey's
`DocumentObjectList`).

### 4.15 EELS element maps by model fitting (M13, in progress)

Element maps of an EELS spectrum image from a fit of its edges with computed
cross-sections (exSpy's `EELSModel`), not from window integration. The
workflow is the one of the HyperSpy tutorials
(`~/Documents/Dev/Hyperspy_Workshop_2025/EELS`): `add_elements` → microscope
parameters → `create_model(low_loss=…, GOS=…)` → `smart_fit` on a spectrum,
check it → (optionally the onset, the fine structure) →
`multifit(kind="smart")` → the edges' `intensity` as maps.

**The model.** A power-law background plus one `EELSCLEdge` per subshell of
each element inside the energy range; an element's subshells are tied to its
first (L2 and L1 follow L3), so each element has one intensity. It needs the
beam energy and the convergence and collection angles (Info tab). With a low
loss in the link group, the model is convolved with it (plural scattering).
The intensities are already divided by the cross-sections, so their ratios
are atomic ratios.

- **Cross-sections:** *DFT* (default; Segger, Guzzinati & Kohl), *Dirac* or
  *Hydrogenic* (K and L only). The DFT and Dirac tables are files exSpy
  downloads once from Zenodo into pooch's cache.
- **Fine structure:** not fitted by default. *Fine structure (spline)* turns
  on exSpy's spline over a typed width (30 eV) after each onset.
- **Onset:** the database's, or typed per element (as the tutorial does for
  O K at 524 eV).
- **Fit range:** the whole spectrum, or typed limits.
- **Background window:** off by default (exSpy's `smart_fit`: the background
  over the range up to the first edge, refitted with each edge). On, a window
  of the user's (default: the range's start to 3 eV before the first edge;
  typed, or dragged as a hatched band on the spectrum while the Quantify
  panel is open): the background components are fitted over it alone with
  the edges switched off (`disable_edges`, `set_signal_range`, `fit`), then
  fixed while the edges are fitted over the whole range, for one spectrum and
  for every pixel (`multifit`). Options `bg_lo`, `bg_hi`; tested equal to
  those exSpy steps.
- **Shown with the Quantify tab only:** the fits (EDS and EELS, also the fit
  at the cursor), the background window and the elements' markers (X-ray
  lines, edge onsets) are drawn only while the Quantify tab is open; the
  element hovered in the periodic table is still previewed anywhere.

**M13a — Edges in the periodic table.** The *Elements* button on EELS
spectra, spectrum images and line scans. Elements without an edge inside the
energy range are disabled; hovering previews the onsets on the spectrum;
ticked elements keep labelled markers ("Mn L₃"). Each dataset has its own
elements (`Sample.elements`; summary `edges: {elements, subshells}`).
`PUT …/eels-elements {elements}`.

**M13b — Fit one spectrum.** Quantify tab, for EELS: the sum, the cursor or a
region. *Fit* draws the model, the background and each element's edges on the
spectrum (as the EDS fit) and lists each element's intensity, its error, at%
and onset, with the residual (its size relative to the signal) and the
background's exponent. Typing an onset refits. `POST …/eels-fit {source,
options}`. Several spectra (the whole, the cursor, regions) can be ticked
and are fitted together; each fit is drawn over its spectrum while that is
shown, the table lists one (picked by name) and the CSV all of them.
Built (`processing/eels_model.py`, `api/eels_fit.py`,
`components/EelsFitPanel.tsx`, `state/eelsFit.ts`); both paths are tested
equal to exSpy's own calls.

**M13c — Maps.** *Fit every pixel* runs `multifit(kind="smart")` as a job
(about 10 ms a pixel without a low loss, 30 ms with one; progress, cancel),
with the settings of the single fit. Result: a fitted map per element with
a Counts / at% switch, as the EDS fitted maps (`origin "fitted"`; removed when the spectrum image is rebinned, filtered or
re-aligned). 2D scans only. `POST …/eels-fit-maps {options}`. The fit of every
pixel is kept on the spectrum image (`Dataset.pixel_fits`, while its data is
unchanged; summary `pixel_fit`): with *Show fit* the spectrum view draws the
fit at the cursor (the cursor's block added up) instead of the single fit.
`GET …/eels-pixel-fit?x&y&size`.

**Later.** A linear per-pixel solve (onsets and exponent from the summed fit;
40 spectra in 0.05 s) and an LCPL background; LBA; profiles for line scans;
elements shared by a link group and quantification across core-loss ranges;
Gaussians in the fine structure; the per-pixel fit uses the loaded data
(filters are not applied; the single fit uses the processed spectrum); a
fit is not cleared when an in-situ frame changes.

### 4.16 Dataset pane as a tree; one list of fitted maps (M14, done)

**M14a — The dataset pane is a tree** (built). Everything made from a dataset is
listed under it, indented with tree lines, to any depth: a background-removed
spectrum image under its source, and the window maps, exponent map, fitted
maps and deconvolved spectra made from *it* under it, not beside the source's
own. Today these are flat groups by type per file, so the maps of a
background-removed subset can't be told from the source's.
- **Applies to every result that becomes a dataset:** computed, window,
  fitted and composition maps, background-removed and deconvolved spectra
  (and their maps), thickness, ZLP shift, aligned averages.
- **Parent:** `parent_id` (and `derived_from`), which the backend already
  sets; the tree is built from it in the frontend. Check that every result
  has one (window maps of a background-removed spectrum image point at it,
  not at its source).
- **Behaviour:** a node folds with its children; same-type siblings still
  fold into a type group inside their parent; deleting or closing a node
  takes its subtree (as `_remove_tree` does); ↑/↓ walk the visible rows.
- **Experiments** stay the top level (experiment → files by role → datasets
  → results).
- **Built:** `layoutTree` in `state/datasets.ts` (per file: the children of
  each dataset by `parent_id`, else `derived_from`, within the file; each
  level gets link boxes and type groups); `DatasetList.tsx` renders a
  dataset's results right after it, one level in, with tree lines (large
  icons: a full-width row of narrower cards). A chevron before the title
  folds a dataset's results (`foldedNodes` in the ui store, this session
  only); a type group nested in a dataset saves its folding under the
  parent's id. ↑/↓ skip rows hidden by either. Every result the backend
  makes already has its parent (window maps on a background-removed SI
  point at it).

**M14b — One list of fitted maps, with an at% switch** (built). A per-pixel fit (EELS
model, and EDS) gives one map per element, not a *Map* list plus an *at%*
list. A *Counts | at%* switch (on the group's header, and in the viewer
toolbar when such a map is shown) says which quantity the maps show; it
applies to all the maps of that fit at once.
- **Backend:** each fitted map keeps both arrays (intensity and at%) and
  serves the one asked for; `origin "composition"` goes. The summary says
  which quantities a map has and which is shown; switching bumps the
  revision so contrast is redone.
- **Display, profiles and export** use the quantity shown; the title and
  units follow ("Cu (fit)" / "Cu at%").
- **Built:** `Dataset.quantities` / `quantity` (`FIT_QUANTITIES`),
  `Session.set_fit_quantity`, `PUT …/fit-quantity {quantity}` on the
  spectrum image; a refit keeps showing at% if it was shown. Fitted maps get
  their own type group, *Fit*; `FitQuantitySwitch.tsx` sits on its header
  and in the viewer toolbar. Maps without at% (excluded elements) stay on
  counts.

### Line scans cut from a spectrum image (built)

*Generate LineScan*, from the right-click menu of a line (on the image, with
any tool, or on its name in the Line profile tab), makes a line scan along it, listed under the spectrum image (`origin "line-scan"`):
positions sampled as the profile is (about one per pixel along the line,
bilinear between pixel centres, so at any angle), each spectrum the mean
across the line's width; a "Distance" axis in the scan's units. The
spectrum images linked to it are cut along the same line and their line
scans linked. Made once from the loaded data (filters not applied).
`processing/line_scan.py`, `Session.add_line_scans`, `POST
…/line-scan {line: {x0, y0, x1, y1, width}, name, with_linked}`.

### 4.17 Grid panels arranged by hand (M15, done)

In the Grid view, a panel is moved by dragging the strip along its top edge
(a grip shows on hover). A bar shows where it lands: left or right of another
panel (into that row), or above or below it (a new row). Panels in a row share
its width and the rows share the height; images fit their panels. The
arrangement is kept per view for the session (`gridLayouts` in the ui
store); panels added later join the last row, removed ones leave it. *Reset
layout* goes back to the automatic grid, whose short last row keeps the
column width. *Save PNG* lays the figure out in the same rows (a shorter row
centred). `components/ViewerGrid.tsx`, `figureLayout` in `viewer/snapshot.ts`.

### 4.18 One workspace per loaded dataset (M16, done)

Each top-level entry of the dataset pane (a file, an experiment, or the files
of a linked EELS box) is a workspace. The one being viewed is tinted in the
pane; the others are dimmed. A click on anything in another workspace (a
row, or its name) only enters it: what it showed when it was left comes back
(the datasets shown, the one being edited, the tab under the image and its
split, the right-hand tab); one never entered shows its first dataset. Then
clicks act as usual. Nothing else in another workspace responds (folding,
×, renaming, right-click menus, colours, dragging, keys): any click on it,
wherever, only enters it. Hovering over it lights up its name only. ↑/↓ stay
in the workspace. Showing a dataset from code (a file
just opened, a new result) moves to its workspace, saving the one left.
Cursors, regions, lines, FFT boxes, contrast and zoom were already kept per
dataset. `active` and `workspaces` in `state/datasets.ts`; `DatasetList.tsx`.

### 4.19 Undo a deletion (M17, done)

⌘/Ctrl Z brings back the last thing deleted; each deletion says it can be
undone ("Deleted Fe Kα · ⌘/Ctrl Z to undo"), and undoing says what came back.
- **Datasets** (window maps, thickness, ZLP shift, fitted maps, background-
  removed and deconvolved spectra with everything made from them): the
  backend keeps the last 20 deletions (`Session.undoable`) and puts the
  datasets back in their places (`POST /api/undo/{token}`; the token comes
  with `DELETE /api/datasets/{id}`). A computed element map comes back by
  re-ticking its element (it is listed after the others). Closing the file
  drops its deletions.
- **Regions, line profiles, FFT boxes** (deleted with × or the Delete key; a
  click that draws nothing is not a deletion): kept in the browser and put
  back in their places.
- Not undone: other changes (rebinning, alignment, closing a file). No redo.
- `state/undo.ts`, `deleteRois` / `deleteLine` / `deleteBox`; tests in
  `tests/test_undo.py`.

### 4.20 Saving an analysis as a project (M18)

An analysis (the dataset tree, how each result was made, and what the views
showed) saved as one HDF5 project file that the app reopens and that HyperSpy
reads in a notebook.

**Decisions.**
- *Raw data is referenced by default; embedding is an option* (Save as…:
  "Embed raw data", for files that fit; in-situ series are always
  referenced). A reference keeps the path absolute and relative to the
  project, and a fingerprint (size, modification time, hash of the first
  MBs).
- *Opens in HyperSpy:* every stored signal is an `.hspy`-style group (written
  with rosettasciio), so `h5py`/`hs.load` can read them; the app's own
  information is in attributes and JSON next to them.
- *Changed raw data* (fingerprint mismatch: expected only after an
  accidental overwrite): the project opens with a warning naming the files;
  stored arrays are kept as saved; results rebuilt from settings are rebuilt
  from the current file and marked "made from changed data".

**What is saved.**
- *Recipes for everything:* each node of the tree with its parent, origin and
  settings (window, filters, background options, line spec, thickness
  options, …). Cheap ones are rebuilt on load: window and element maps,
  filters, LineScans, background removal, thickness, time scans.
- *Arrays where rebuilding is slow or impossible:* per-pixel fits, aligned
  sums, deconvolution, drift correction (shifts and average), anything made
  from imported shifts; stored compressed and used while the fingerprint
  matches.
- *Operations that change loaded data:* an ordered, structured log per
  dataset (rebin factors, ZLP alignment options, in-situ binning, stack
  rebin, applied alignment), replayed on the raw data at load. Today these
  are labels (`rebinned`); they become records.
- *The views:* regions, profile lines, energy marks, FFT boxes, contrast and
  colour maps, layers, each dataset's workspace (M16), the selection. Not
  undo history.
- *Format details:* `schema_version` and app/library versions on the root;
  written to a temporary file and renamed (HDF5 does not reclaim rewritten
  space); no pickles.

```
analysis.siv (HDF5)
├─ attrs: schema_version, versions, saved
├─ raw/<file_id>     attrs: paths, fingerprint     [data only if embedded]
├─ datasets/<id>     attrs: origin, parent_id, title, recipe (JSON), ops (JSON)
│   └─ signal/       .hspy group (stored arrays only)
└─ ui/               JSON: regions, lines, marks, display, layers, workspaces
```

**Sharing (Save as… levels).**

| Save as… | Contains | For |
|---|---|---|
| Link raw data | recipes, operation log, views; small arrays | yourself, with the data |
| Include results | the above + every dataset's current data stored | sharing without the raw data |
| Include raw data | the above + the raw files embedded | a full handover |

"Include results" lets the datasets to store be ticked (all by default) and
shows the size; an in-situ series stores what was made from it (the frame or
range shown, aligned sums, time scans), never the series. Opened without the
raw data, everything stored works and can be built on; what needs the raw
files (undoing an in-place step, frame browsing) is marked "raw data not
included" and its actions disabled, until the raw data is found next to the
project (then relinked).

**Opening.** Raw files are found by the relative path, then the absolute
one, then a *Locate…* dialog (finding one also searches its folder for the
others); fingerprints are checked; operations replayed; recipes rebuilt;
stored arrays loaded; the views restored.

**In place or as a new dataset (M18a, done).** Built as below: an *In
place | New dataset* switch (remembered per operation in the browser) in the
Rebin dialog (not for in-situ series, which are binned as they are read),
Zero-loss alignment and Align frames' Apply. New datasets: origins
"rebinned", "zlp-aligned" (linked as their sources are; the ZLP shift map
under the new low loss), "aligned" (its average under it); deletable.
Operations are recorded on ``Dataset.ops`` ({"op": "rebin" | "rebin-series"
| "zlp-align" | "align" | "unalign", …}); copies carry ``copied_at``.

Originally planned: Every operation that changes
loaded data gets the choice: *Rebin*, *Zero-loss alignment*, *Align frames*
(Apply), stack and in-situ *Rebin*. "In place" works as now; "New dataset"
leaves the source as it is and adds the result under it (`origin` e.g.
"rebinned", "zlp-aligned", "aligned", with its recipe), so it is an ordinary
node of the tree, saved and deleted like the others. The default is in place
(as now); the choice is remembered per operation.

**Save and open (M18b, done).** *Save analysis…* (⇧⌘S) and *Open
analysis…* (⇧⌘O) in the toolbar, through the file browser (a name to save
as; an existing file asks to be replaced; opening closes what is open).
`POST /api/project/save {path, views, overwrite}` and `POST
/api/project/open {path}` (`si_viewer/project.py`). The project is a `.hspy`
(HDF5; `hs.load` gives the stored results): raw files referenced (absolute
path, path relative to the project, fingerprint: size, mtime, SHA-1 of the
first MB); every dataset's settings and `ops` (with a global `seq`) as JSON
in the `si_viewer` group, with the views (the frontend stores' state about
the open data). Opening opens the raw files again with their saved ids,
replays the operations in order, puts the settings back and loads the
stored datasets.

Deviation from the plan: every dataset not loaded from a raw file (maps,
fits, background-removed spectra, LineScans, copies, …) is stored as an
array rather than rebuilt from its recipe (its settings are saved with it);
so is the frame range an in-situ series showed. Opening is all or nothing:
raw data not found (next to the project, then where it was) is an error
before anything is closed; a changed fingerprint is a warning. Not yet:
saving and opening as jobs (with progress), per-pixel fit models, the
LineScan exploded view's energy marks (component state).

**Include results (M18c, done).** The save dialog chooses *Link raw data*
or *Include results*, each with the size it would store (before
compression; `GET /api/project/estimate`). Include results also stores the
datasets loaded from raw files as they are now (an aligned stack: as loaded
and aligned; an in-situ series: the frame or range shown, never the
series). Opening: a stored file whose raw data can't be found, or whose
fingerprint changed, is opened from the stored data (`raw_missing`, tagged
"no raw data" in the tree, with a warning); it can be worked on, and its data
is stored again at any level. Its in-situ series becomes a plain spectrum
image (no frames to browse). Not done: ticking which datasets to store.

**Include raw data and Locate (M18d, done).** A third level, *Include raw
data*, embeds each raw file's bytes (`si_viewer/raw/<file id>`); in-situ
series are never embedded (their frames are separate files): their results
are stored instead. Opening prefers the raw file on disk when it is
unchanged, else the embedded copy (written to a cache folder in the system's
temporary folder, reused while it is there; the file is tagged "embedded"),
else the stored data. A file opened from its embedded copy keeps its
original path as its reference and is embedded again whenever it is saved.
Raw files that are neither found, embedded nor stored are asked for before
anything opens (`POST /api/project/check`): the file browser shows *Locate
<name>*, the file picked is used (`locations`, file id -> path), and the
others are also looked for in its folder.

**One workspace per analysis.** Save analysis saves the workspace shown
(an experiment, or a file with the files linked to it; the only one if
nothing is shown), with the files its datasets are linked to or made from
(`file_ids` on save and estimate); the views saved are that workspace's.
The name offered is "<workspace> analysis.hspy". Opening adds the analysis
next to what is open and shows it (per-dataset view state merged in); it is
refused while any of its files is open ("… is already open: close it
first").

Not done (set aside): ticking which datasets to store; autosave and export
as a folder of `.hspy` files (later).

**Phases.**
1. M18a: in place / new dataset; operations recorded as structured data.
2. M18b: save and open: referenced raw data, recipes, the operation log, the
   views; HyperSpy-readable groups.
3. M18c: the "Include results" level (opening without the raw data);
   fingerprint checks and the changed-data warning.
4. M18d: embedding raw data; the *Locate…* dialog.
5. Later: autosave / recovery; export as a folder of `.hspy` files.

### 4.21 Decomposition of spectrum images: PCA (M19), then BSS and diffusion maps

For EELS and EDS spectrum images (raw or background-removed). Guided, step by
step: decompose, look at the components, then denoise with as many as needed.

1. **Decompose** (Process tab of the SI): an energy window (the whole range by
   default; drawn on the spectrum, editable as numbers), Poisson weighting (on:
   counts, Keenan & Kotula), and how many components to keep (32). A job:
   HyperSpy's `decomposition` (SVD, randomized solver,
   `normalize_poissonian_noise`) of the window, its loadings and factors
   stored as datasets; the explained variance is HyperSpy's
   `explained_variance_ratio` (relative to the components kept). No
   processing of our own: HyperSpy (and its dependencies) do the work.
2. **The PCA object** under the SI (origin `pca`): a stack whose frames are the
   components' maps (frame k = component k + 1), with **PCA factors** under it:
   the components' spectra (a line scan, components × channels). In the viewer:
   the frame slider runs through the components; bottom tabs **Component** (the
   spectrum of the component shown) and **Scree** (explained variance on a log
   scale; a click sets how many components to keep).
3. **Denoise**: the SI rebuilt from its first N components (HyperSpy's
   `get_decomposition_model` of the stored loadings and factors), a
   new spectrum image under the SI over the window (origin `pca-denoised`).
   Decompose again replaces the PCA object in place.
4. Saved with an analysis as derived datasets (no replay).

Later in M19: blind source separation (HyperSpy's `blind_source_separation`
on the kept components) and diffusion maps (pydiffmap, as in the user's own
`dm.py`).

### 4.22 Focus mode (M20, planned)

As panes and tabs multiply, a way to work on one of them without the rest of
the interface around it: e.g. the spectrum tab while adjusting the
components of a curve fit one by one.

**The idea (the user's):** a button or shortcut that lifts the current pane
or tab out of the layout and shows it at about 80–90 % of the window, on top
of the interface, with everything behind it blurred and dimmed.

**Suggested shape (to settle when it is built):**

- **Independent of any process.** Any pane can be focused: the viewer (single
  or grid), a bottom tab (spectrum, profile, histogram, scree, exploded…),
  the FFT, a Process section. Processes with many decisions (curve fitting,
  drift correction, background removal) can *offer* it, e.g. a "Focus"
  button in their section or opening in focus by choice, but nothing
  requires it.
- **The same component, moved, not a copy.** The focused pane is the live one
  (same state, zoom, selection), rendered in an overlay; leaving focus puts
  it back where it was. Edits made while focused are just edits.
- **What comes with it.** A pane alone is often not enough: fitting a
  spectrum needs its controls. A focused pane can bring a slim side strip
  with the controls it depends on (the Process section that drives it, the
  legend), rather than the whole right-hand pane.
- **In and out.** A focus button in each pane's header (an expand icon, next
  to Split / Save PNG), a shortcut (F is taken by "fit"; e.g. Shift F or `),
  and Esc, a click on the blurred background, or the same button to leave.
  One pane focused at a time; switching focus to another pane is allowed.
- **Look.** Rounded panel with the app's border and shadow, the backdrop
  blurred (CSS `backdrop-filter`) and dimmed; animated open/close kept short.
  The focused size is remembered per pane.

**Open questions:** whether focus is per pane or a saved layout (several
panes side by side in focus, e.g. the spectrum and its map); whether
processes like curve fitting open in focus by default; how it interacts with
the split bottom pane (focus the split as a whole, or one side).

### 4.23 Curve fitting (M21, planned)

General model fitting of spectra, spectrum images and line scans with
HyperSpy's `Model1D`, covering what its user guide's *Model fitting* chapter
offers (creating a model, components, adding and switching them, indexing,
parameters, fitting, storing, big data, SAMFire). As everywhere: HyperSpy does
the fitting; our code builds the model from the user's choices, runs it as a
job, and turns its results into datasets. The EELS edge fit (§4.15) stays as
it is; it is an exSpy `EELSModel`, which is a `Model1D`, so the two share the
same plumbing (jobs, per-pixel results, drawing on the spectrum) and an EELS
model can later be opened here to add components to it.

**Where it lives.** A **Curve fitting** process in the Process tab of
spectra, spectrum images and line scans (any signal type, not only
EELS/EDS), driving the spectrum plot: its model and handles are
drawn while the section is open. (First built in the Quantify tab; moved.)
**The whole spectrum is always fitted**: there is no fit range to set; to fit
part of it, crop the data to an energy window first (Crop to this window) and
fit the new spectrum (image). The components are listed in a panel beside the plot (a bottom
tab, *Model*), with their curves drawn in their colours on the spectrum. It
is the first process to offer **focus mode** (§4.22): the spectrum, the
component list and the fit controls together, the rest blurred.

#### The model

- **Creating it** (`Signal1D.create_model()`): on a chosen spectrum (the sum,
  the cursor, a region, as the EELS fit) for building and checking it, then
  on every pixel, over the whole energy axis.
  Before creating it, the energy axis's `is_binned` is checked and shown
  (counts per channel vs. intensity); data from DM that is stored as an
  average is flagged (HyperSpy's warning), with a switch to treat it as
  binned or not.
- **One model per spectrum (image)**, kept on the dataset while it is
  unchanged, and saved with an analysis (`store()` into the signal's
  `models`; restored on open, see *Storing*). Several named models can be
  kept and switched between (`models.restore(name)`), e.g. "3 Gaussians" and
  "2 Voigts" on the same data.

#### Components

- **Built-in** (`hs.model.components1D`): Gaussian, GaussianHF, Lorentzian,
  Voigt, SplitVoigt, SkewNormal, Doniach, PowerLaw, Exponential, Polynomial
  (order chosen), Offset, Arctan, Erf, HeavisideStep, Logistic, Bleasdale. An
  **Add** menu, grouped (*peaks*, *backgrounds*, *steps*, *other*), each with
  a small icon of its shape and its formula on hover.
- **From an expression** (`Expression`): a formula typed with its parameter
  names, e.g. `A * exp(-(x - x0)**2 / (2 * s**2))`, checked by SymPy as typed
  (errors shown under the field); substitutions with `;` for long ones.
  Saved expressions become user components offered in the Add menu.
- **From a reference spectrum** (`ScalableFixedPattern`): any spectrum in the
  session (a dataset, or the sum/cursor/region of a spectrum image), scaled
  and shifted (`yscale`, `xscale`, `shift`) — e.g. fitting a known edge shape
  or a standard.
- **Custom Python components** (subclassing `Component`): not from the UI;
  later, a folder of user components loaded at start-up.
- **Placing peaks by hand.** *Add* then click on the spectrum puts the peak
  there (centre and height from the click, width from the zoom); dragging
  its handles on the plot (centre, height, half-width) sets the starting
  values, as HyperSpy's interactive widgets do in Jupyter. Other components
  start from `estimate_parameters` over the spectrum (or, since 2.5, over
  several intervals for PowerLaw, Polynomial and Offset: drawn as windows on
  the spectrum, as the background removal's).
- **Each component** in the list: its name (editable; unique, as HyperSpy
  requires), colour, an on/off switch (`active`), and its parameters below
  it. Switching off at every pixel or at the current one only
  (`active_is_multidimensional`, `set_component_active_value(only_current=…)`)
  — e.g. a peak present only in part of a map. Duplicate, delete, reorder.

#### Parameters

Shown as a table per component, as the axes table of the Info tab: value,
± error after a fit, free/fixed, min and max, and the twin.

- **Values**: typed, dragged on the plot, or `estimate_parameters`. At the
  current pixel or every pixel (`set_parameters_value(only_current=…)`).
- **Free / fixed** (`free`; `set_parameters_free` / `set_parameters_not_free`,
  also for a whole component or the whole model, by parameter name, e.g.
  "fix every centre").
- **Bounds** (`bmin`, `bmax`), used when fitting bounded.
- **Twins** (`twin`, `twin_function_expr`, `twin_inverse_function_expr`): tie a
  parameter to another, optionally through an expression, e.g. a spin–orbit
  doublet (second centre = first + 5.4 eV, area ratio 1:2) or equal widths.
  Picked from a menu of the model's parameters; twinned parameters show what
  they follow and are not editable.
- **Per pixel**: after a fit of every pixel, the cursor shows that pixel's
  values (`Parameter.map`), and any parameter can be opened as a map.

#### Background window (built)

As in the EELS edge fit: *Background window* (Process tab), typed or
dragged as a hatched band. The background components (PowerLaw,
Exponential, Polynomial, Offset) are fitted over the window alone with the
others switched off, then fixed while the others are fitted over the whole
spectrum (their free state restored after). `ModelSpec.bg_lo/bg_hi`; tested
equal to HyperSpy's fit over the window. Edge steps go on top as `Logistic`
components (0 below the step, unlike Arctan and Erf); two steps fitted freely
trade off against each other, so their 2 : 1 ratio is set with twins.

#### Excluded data

The whole spectrum is fitted. Windows can still be *excluded*
(`remove_signal_range`, e.g. a spike or an unmodelled feature; drawn
hatched on the spectrum, M21b) and *Reset* (`reset_signal_range`); a mask
(`set_signal_range_from_mask`): later. A narrower range is a crop, not a
setting.

#### Fitting

- **Fit one spectrum** (`fit()`): the curve, each component and the residual
  drawn on the plot, live; the parameter table updated with the errors.
- **Optimizer**: a menu with HyperSpy's choices, the default first —
  local: `lm` (default), `trf`, `dogbox`, `odr`; linear: `lstsq`, `ols`,
  `nnls`, `ridge`; global: Differential Evolution, Dual Annealing, SHGO; and
  scipy's `minimize` methods (Nelder-Mead, Powell, L-BFGS-B, …). Options the
  chosen one does not support are disabled (bounds, gradients, lazy data),
  following the guide's table.
- **Loss function**: least squares (default; errors and χ²), Poisson maximum
  likelihood (`ML-poisson`, for very low counts) and Huber (outliers); the
  last two switch to a suitable optimizer and note that errors are not
  available.
- **Bounded** (`bounded=True`): a checkbox, available for the optimizers that
  support it.
- **Gradients** (`grad="analytical"` / `"fd"` / none): advanced.
- **Linear fitting**: when every free parameter is linear (HyperSpy says
  which are not), a *Linear* option: no starting values needed, fast,
  vectorised over every pixel; bounds not available.
- **Goodness of fit**: χ², reduced χ² and degrees of freedom (`chisq`,
  `red_chisq`, `dof`) shown after each fit, with a note that they need the
  noise variance (Poisson by default for counts: `estimate_poissonian_noise_variance`,
  or set). The optimisation result (converged, iterations, message) on hover.

#### Fitting every pixel

- **Fit every pixel** (`multifit()`) as a job with progress and cancel: the
  path (`iterpath`: serpentine (default) or flyback), starting each pixel from
  the last, or from the values set (*same start for every pixel*). With the
  same options as the single fit (optimizer, loss, bounded, linear).
- **SAMFire** (`create_samfire()`, `start(...)`): for maps with poor starting
  values or local minima. Seed pixels are fitted first (picked on the
  navigator: the cursor and regions, or several clicked points), then
  SAMFire's strategies (local: reduced χ², global: histogram) spread the fit.
  Its tolerance (goodness test) set in the dialog; workers from the machine's
  cores.
- **Big data**: lazy spectrum images and in-situ series are fitted chunk by
  chunk (HyperSpy's lazy model support; linear `lstsq` only for lazy data),
  progress per chunk; the results are written as they come.
- **Results**: each free parameter of each component as a map under the
  dataset (origin `fitted`, as the EELS maps, e.g. "Gaussian 1 · centre"),
  plus the reduced χ² map (where the model fits badly). Line scans: as
  profiles. The fitted model at the cursor is drawn on its spectrum (as the
  EELS *Show fit*). The residual as a new spectrum image on request
  (`s - m.as_signal()`), and the model itself (`m.as_signal()`), or one
  component's contribution (`as_signal(component_list=…)`), e.g. a
  background-free spectrum image.

#### Indexing

A fitted model of a spectrum image can be sliced like the data (`m.inav`,
`m.isig`): the fit of a region alone, or of a narrower energy range, as a new
model (goodness of fit recomputed). Used for "refit this region with another
model" and when cropping a dataset that has a model (its model is cropped
with it).

#### Storing and exporting

- **Stored with the data** (`m.store(name)`, `s.models`), so saving an
  analysis keeps every model and its fitted values; opening restores them
  (`models.restore`). The old `save_parameters2file` route is not used.
- **Export**: the parameter table as CSV (single spectrum: values ± errors;
  every pixel: one row per pixel), each map as image/HSPY, and HyperSpy's own
  `export_results` / `Component.export` / `Parameter.export` for the formats
  it supports. The model plot as PNG/CSV like other plots.

#### Phases

| Phase | Content |
| --- | --- |
| M21a | Model on one spectrum: built-in components from the Add menu, parameter table (value, free, bounds), `estimate_parameters`, `fit()` of the whole spectrum with the default optimizer, curves drawn on the plot, goodness of fit |
| M21b | Peaks placed and dragged on the plot; component on/off, rename, duplicate; twins with expressions; batch free/fixed; excluded windows |
| M21c | Fit every pixel as a job: parameter maps, reduced χ² map, fit at the cursor; per-pixel active components |
| M21d | Optimizer, loss function, bounded, gradients and linear fitting; residual, model and component signals |
| M21e | Expression and fixed-pattern components; several stored models, saved with an analysis; export |
| M21f | SAMFire with seed pixels; lazy/big data; slicing a model (regions, cropping); focus mode for fitting (§4.22) |

**M21a — built.** Process tab → *Curve fitting* (any spectrum, spectrum
image or line scan). The spectrum (all pixels, the cursor
or a region), an *Add* menu of HyperSpy's components in groups (Peaks, Backgrounds, Steps,
Other; not `Expression` and `ScalableFixedPattern`, M21e), each estimated over
the spectrum when it can (`estimate_parameters`; *Estimate* again any time).
Each component: on/off, Polynomial order, a parameter table (value, ± after a
fit, free, min, max). The model and its components are drawn on the spectrum
with their current values as they are edited (evaluate without fitting), and
*Fit* runs `fit()` (bounded when a free parameter has a limit) and shows the
reduced χ², χ², the number of free parameters and the residual. The spectrum
is turned into a plain `Signal1D` (so an EELS or EDS spectrum doesn't get
exSpy's model) with Poisson noise variance for counts. The model is a spec
(`processing/curve_fit.py`) rebuilt for every request (`api/curve_fit.py`:
`GET /api/curve-fit/components`, `POST …/curve-fit/estimate`,
`POST …/curve-fit`); tested equal to HyperSpy's own fit. The component list
sits in the Process section for now (the *Model* bottom tab and focus mode
come with M21b/M21f). A channel with no counts gets a variance of 1 (Poisson's
estimate is 0 there, which would give it an infinite weight). Data stored as averages (e.g. DM's "Stack_Averaged") gives
a large reduced χ²: the Poisson variance assumes counts.

**Interacting with components — built** (the first half of M21b):
- **Selecting:** click a component's curve on the plot or its row in the list;
  the selected one is drawn solid and thicker, the others faded and dashed;
  its parameter table unfolds (the others fold to their header).
- **Handles** on the selected component, placed from its drawn curve (so any
  peak shape works): peaks — the top (moves the centre, scales the
  amplitude) and the half-height points (scale the widths, the area with
  them so the height stays; SplitVoigt one side each); Offset — its level;
  Polynomial — its level and slope; PowerLaw and Exponential — two points
  (the curve through both); steps — the position and the height.
- **Click to place:** choosing a peak in *Add* waits for a click on the
  spectrum (crosshair; Esc or *Cancel* stops): the centre there, a width of
  1.5 % of the spectrum's range, the amplitude so the model reaches the click.
  Backgrounds and steps still start from their estimate. (The fit range
  that was dragged here has gone: the whole spectrum is always fitted.)
- **Live:** dragging redraws the model and components as it goes (one
  request in flight; the latest position sent when it returns).
- **List:** double-click a name to rename it (names stay unique), duplicate
  (⧉), remove (×); for the selected one *Estimate*, *Fix all*, *Free all*.
- **Right-click a curve:** Estimate, Duplicate, Fix all, Free all, Delete.
- **Parameter limits on the spectrum:** each parameter is *Free*, *Limit*
  (fitted within Min–Max: HyperSpy's `bmin`/`bmax`, the fit then bounded) or
  *Fixed* (`free = False`). Choosing *Limit* without limits gives some around
  the value (a position ± 5 % of the spectrum; others half to twice it, an
  amplitude from 0). Clicking a parameter's row selects it, and while it is
  limited, a position's limits (centre, x0, origin) are drawn on the spectrum
  as a band on the energy axis in the component's colour, dragged there as
  well as typed. A band edge dragged to the end of the axis means no limit on
  that side; the value is kept within its limits, also while dragging the
  component's handles. Other parameters (widths, amplitudes, exponents) are
  limited by typing only: their limits don't map cleanly onto the spectrum.
- **Kept (built):** each dataset's model (components, settings, last result)
  is kept in the browser, so a reload keeps it, and saved with an analysis
  (with the normalisation settings).
- **Twins (built):** a fourth choice, *Twin*: the parameter follows another
  (any component's parameter, from a menu) through a formula of `x`, its
  value (`x + 5.4`, `0.5*x`; `x` alone: the same), HyperSpy's `twin` and
  `twin_function_expr`; it is not fitted and its value is shown greyed as
  last drawn. A missing target, a loop or a formula SymPy can't read is
  refused; renaming a component keeps the twins following it, removing one
  unlinks them. E.g. the Ti L₂ step at half the L₃ step's height, the same
  steepness, 5.2 eV higher.

#### Testing

Fitting is tried on real EELS data:
`examples/003_EELS-SI_1.7nm_2mmAp_TiO_Stack_Averaged_x_1_38_y_1_235_445.05_488.05_eV_PCA_4.hspy`
(TiO, 19 × 117 pixels, 445–488 eV, PCA-denoised with 4 components), whose Ti
L₂,₃ edge (L₃ and L₂, each split by the crystal field) gives overlapping
peaks on a background. HyperSpy's two-Gaussians simulation is also in the
Examples menu (*Fitting test: two Gaussians*, with its true maps), for a
quick check where the answer is known; no further simulated datasets are
planned.

**Later:** 2D models of images (`Signal2D.create_model()`, e.g. Gaussian2D for
atomic columns), user Python components, opening an EELS edge model here to
add peaks (fine structure) to it.

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
| M2 | Point cursor, area ROIs, energy-span maps, X-ray line markers, periodic table | Done |
| M3 | Element maps from the periodic table: ticking an element computes its map (window + background) into the dataset pane, unticking removes it, maps can be deleted from the pane (unticking the element); map options (background, width, binning) | Done |
| M4 | Background (windows + model), model fit + calibration, quantification, per-pixel fitted maps as a job | Done |
| M5 | Pre-processing tab (right side): non-destructive filters (Gaussian/median/box blur) for images and maps; binning and spectral smoothing for spectrum images; live FFT of the viewer with d-spacing readout | Done |
| M6 | Line profiles for images and SIs (intensity and element profiles) | Done |
| M7 | Export, keyboard shortcuts, polish | Done |
| M8 | pywebview desktop window and installers (deferred until the features are complete) | |
| M9 | Drift correction of image stacks with rigidregistration, as a guided workflow in the viewer: Frames + Fourier mask (M9a, done), Peak finding + Shift matrices (M9b, done), Result + Apply (M9c, done), extras (M9d) | In progress |
| M10 | EELS: line scans and `.dm5` (M10a), EDS or EELS detection (M10b), low-/core-loss pairs (M10c), ZLP alignment, thickness and deconvolution (M10d), background removal and edge maps (M10e) | In progress (M10a–c done; M10d: thickness, ZLP alignment and deconvolution built) |
| M11 | Grid / Merged views for images and maps (M11a, done), split pane under the image (M11b, done) | Done |
| M12 | In-situ EELS series from `.dm4` + `.raw`: reader (M12a), frame browsing (M12b), time scans (M12c), sum frames with drift correction (M12e) | In progress (M12a–c done) |
| M13 | EELS element maps by fitting edges with cross-sections: edges in the periodic table (M13a), fit one spectrum (M13b), per-pixel maps (M13c) | In progress |
| M14 | Dataset pane as a tree of datasets and their results (M14a); one list of fitted maps with a Counts / at% switch (M14b) | Done |
| M15 | Grid panels arranged by hand: drag by the top edge into a row or a new row; PNG export follows | Done |
| M16 | One workspace per loaded dataset: entering one brings back what it showed | Done |
| M17 | Undo a deletion (⌘/Ctrl Z): datasets, regions, line profiles, FFT boxes | Done |
| M18 | Save and open an analysis (HDF5 project, HyperSpy-readable): in place / new dataset for operations (M18a), save and open (M18b), stored arrays and checks (M18c), embedding and relinking (M18d) | Done (choosing which datasets to store set aside) |
| M19 | Decomposition of spectrum images: PCA with denoising (M19a), blind source separation (M19b), diffusion maps (M19c) | In progress |
| M20 | Focus mode: any pane or tab lifted to most of the window over a blurred interface, with the controls it needs (§4.22) | Planned |
| M21 | Curve fitting with HyperSpy's `Model1D`: model on one spectrum (M21a), placing peaks and twins (M21b), every pixel (M21c), optimizers, losses and linear fitting (M21d), expression and pattern components, stored models (M21e), SAMFire and big data (M21f) (§4.23) | In progress (M21a and M21b built but excluded windows) |

## 7. Verification
- **Backend:** pytest against the example `.emd`: signal kinds and shapes,
  that the ROI sum equals a NumPy reference, that fit intensities are within
  tolerance of `get_lines_intensity`, and that the profile length matches the
  geometry.
- **Performance tests:** cursor spectrum under 10 ms; window maps for the
  256 × 256 × 4096 SI under 50 ms.
- **Frontend:** component tests (Vitest) plus a scripted run in a browser:
  load the example, move the cursor, draw an ROI, fit, draw a profile.
