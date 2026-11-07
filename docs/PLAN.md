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
EELS SIs (`exspy.signals.EELSSpectrum`, background power law, edges), PCA/NMF
denoising (`decomposition`), drift correction of EDS spectrum images (applying
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

1. **Frames.** The stack (slider) beside the FFT of the current frame, and a
   plot of each frame's mean intensity, to spot blank or damaged frames.
   - Choose the frame range, and mark bad frames to leave out.
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
  - A stepper at the top: Frames → Fourier mask → Peak finding → Shift
    matrices → Outliers → Result. It has Back / Next and shows which steps
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
3. **M9c: Outliers, Result and Apply.**
   - Step 5 (outlier methods, masks overlaid, corrected matrices).
   - Step 6 (trajectory, average and FFT, crop).
   - Apply: aligned frames on read, the *Original / Aligned* switch, the
     average as a new dataset, and the *Alignment* tab.
   - Tests: a synthetic stack with known shifts (under 0.2 px), and the
     sample stacks.
4. **M9d: extras.**
   - The dual mask (two-pass registration) and the clipping dimension
     (elliptical masks are done).
   - Editing single entries or blocks of the outlier mask by hand, as the
     notebook does.
   - Apply the shifts to other stacks of the same scan.
   - Save the shifts (CSV) and a report (PNG/PDF of the steps).

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
| M9 | Drift correction of image stacks with rigidregistration, as a guided workflow in the viewer: Frames + Fourier mask (M9a, done), Peak finding + Shift matrices (M9b, done), Outliers + Result + Apply (M9c), extras (M9d) | In progress |

## 7. Verification
- **Backend:** pytest against the example `.emd`: signal kinds and shapes,
  that the ROI sum equals a NumPy reference, that fit intensities are within
  tolerance of `get_lines_intensity`, and that the profile length matches the
  geometry.
- **Performance tests:** cursor spectrum under 10 ms; window maps for the
  256 × 256 × 4096 SI under 50 ms.
- **Frontend:** component tests (Vitest) plus a scripted run in a browser:
  load the example, move the cursor, draw an ROI, fit, draw a profile.
