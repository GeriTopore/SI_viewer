# 4D-STEM: dependencies and rollout (draft, 2026-11-25)

Status: plan only. Nothing is built yet. Open questions are marked **check**.

Goal: view and analyse 4D-STEM data (a diffraction pattern at every scan
position) in SI Viewer, with the same panes and conventions as the rest of the
app, before simulating diffraction with the abTEM plugin. Simulated 4D-STEM
from abTEM should later open in the same viewer.

## 1. Findings

### pyxem 0.21.0 (PyPI, 2025-06-04)

- GPL-3.0-or-later. Built on HyperSpy 2 (`hyperspy>=2.0`): its signals are
  HyperSpy signals (`Diffraction2D`, `ElectronDiffraction2D`, `BeamShift`,
  `PolarDiffraction2D`, `DiffractionVectors`, `VirtualDarkFieldImage`, …), with
  lazy (dask) versions of each.
- Requires: dask, diffsims ≥ 0.7, h5py, lmfit, matplotlib, numba, numpy, orix
  ≥ 0.12.1, pooch, psutil, scikit-image, scikit-learn, scipy, shapely, tqdm,
  traits, transforms3d, **zarr < 3** (the same pin as rosettasciio's zspy, so
  no clash with the `si-viewer` env). Optional: cupy (`gpu`), dask-image and
  distributed (`dask`).
- In the `si-viewer` env, `pip install pyxem` adds 14 packages (pyxem,
  diffsims, orix, diffpy.structure, diffpy.utils, lmfit, asteval, dill,
  shapely, uncertainties, PyCifRW, transforms3d, xraydb, SQLAlchemy) and
  **changes nothing installed** (numpy 2.5, scipy 1.18, numba 0.68, dask,
  scikit-image and hyperspy stay).
- What it does (method names as in the API):
  - Calibration: `set_diffraction_calibration`, `set_scan_calibration`,
    `set_experimental_parameters`, `apply_affine_transformation` (ellipticity,
    distortion), `rotate_diffraction`, `apply_gain_normalisation`,
    `CalibrationGenerator` (standards).
  - Centring: `get_direct_beam_position(method)`, `center_direct_beam`,
    `shift_diffraction`, `get_direct_beam_mask`, `template_match_disk` /
    `template_match_ring`.
  - Virtual imaging: `VirtualImageGenerator`, `VirtualDarkFieldGenerator`,
    `VirtualDarkFieldImage` (+ `get_vdf_segments`), masks.
  - Radial and azimuthal: `get_azimuthal_integral1d` / `2d`,
    `PolarDiffraction2D`, radial profiles, `subtract_diffraction_background`.
  - COM and DPC: `center_of_mass`, `BeamShift` (`get_magnitude_signal`,
    `get_phase_signal`, `phase_retrieval` for iDPC, `get_linear_plane` for
    descan, `rotate_beam_shifts`, `calibrate_electric_shifts`, magnetic DPC).
  - Peaks and vectors: `get_diffraction_vectors`, `template_match`,
    `SubpixelrefinementGenerator`, `DiffractionVectors2D`, clustering.
  - Strain: `get_DisplacementGradientMap`, `StrainMap`.
  - Orientation and phase mapping: `PolarDiffraction2D.get_orientation`
    (template matching with diffsims simulations, orix orientations),
    `OrientationMap`.
  - Variance / FEM: `get_variance`, `VarianceGenerator`.
  - Symmetry and correlations: `get_angular_correlation`, `get_angular_power`,
    Pearson correlations.
  - Amorphous: reduced intensity, `PDFGenerator1D` (pair distribution function).
  - Detector cleanup: `find_hot_pixels`, `find_dead_pixels`,
    `correct_bad_pixels`.
  - In-situ: `InSituDiffraction2D`.
  - No ptychography, no tilt-corrected BF / parallax.

### py4DSTEM 0.14.19 (the user's fork, `~/Documents/Dev/py4DSTEM`)

- GPL-3.0. Its own data model (`DataCube`, `BraggVectors`, `Calibration`,
  emdfile `.h5` files), not HyperSpy.
- Requires numpy, scipy, h5py, hdf5plugin, ncempy, matplotlib, scikit-image,
  scikit-learn, scikit-optimize, tqdm, dill, gdown, dask, distributed, emdfile,
  mpire, threadpoolctl, pylops, colorspacious, pymatgen; extras for cupy,
  tensorflow (AI/ML disk detection), numba.
- In the `si-viewer` env, `pip install py4DSTEM` **downgrades numpy 2.5 →
  1.26.4, scipy 1.18 → 1.17.1 and scikit-learn 1.9 → 1.4.2**. It can't share
  the app's environment.
- What it does that pyxem doesn't, or does better:
  - Phase retrieval: DPC, parallax / tilt-corrected BF, direct and iterative
    ptychography (single-slice, multislice, mixed-state, tomographic,
    magnetic), with GPU.
  - Bragg disk detection by probe-kernel cross-correlation (`get_vacuum_probe`,
    `get_probe_kernel`, `find_Bragg_disks`, CUDA and AI/ML variants), then
    `BraggVectors` calibration (origin, ellipse, rotation, pixel size from a
    known crystal).
  - Crystal tools: kinematic and Bloch-wave diffraction, ACOM (automated
    crystal orientation mapping), crystal phase mapping, flowlines.
  - Whole-pattern fitting, polar analysis (`PolarDatacube`), RDF/amorphous,
    digital dark field, classification.
  - DataCube operations: `bin_Q`, `bin_R`, `crop`, `filter_hot_pixels`,
    `median_filter_masked_pixels`, beamstop masks, local average patterns.

### py4D-browser and the user's plugins

- py4D-browser (PyQt): a virtual-image pane and a diffraction pane. Virtual
  detector shapes point, rectangle, circle, annulus; responses integrating,
  maximum, COM x/y, iCOM; real-space point or rectangle selector (mean or max
  pattern); FFT pane of the virtual image or exit-wave power cepstrum; raw
  TIFF export; keyboard moves (WASD detector, IJKL selector). Plugins:
  calibration, tcBF, disk detection, kernel, metadata.
- `better_virtual_imaging_plugin` (the user's): apertures placed by click (BF
  radius from the mean pattern), g-vector aperture arrays (origin, g1, g2,
  repeats, shifts, half orders, include origin, snap to Bragg peaks), several
  previews to compare, merged colour composite with a key, pattern shown at
  the cursor / mean / maximum, log, linear or power display, figure export,
  virtual image saved as a thresholded mask (Otsu), Bin Q / Bin R, hot and
  dead pixel repair with a before/after check.
- `disk_detection_plugin` (the user's, not yet read in detail): disk detection
  saved as Bragg vectors in the file.
- Test data: `better_virtual_imaging_plugin/sim_Au_data_binned.h5`
  (py4DSTEM emdfile 1.0: 100 × 84 scan, 125 × 125 patterns, uint16, with
  Bragg vectors). `hs.load` can't read it (four HDF5 readers claim it).

### SI Viewer today

- A 4D signal (2 navigation, 2 signal axes) is `DatasetKind.OTHER` and has no
  view.
- rosettasciio reads common 4D-STEM formats: Quantum Detectors `.mib`,
  Blockfile `.blo`, EMPAD `.xml` + raw, Arina `.h5`, DigitalMicrograph 4D `.dm4`,
  TVIPS, HSPY / ZSPY, and some EMD. Large files load lazily (dask).
- Already there, to reuse: the spectrum-image layout (navigator + signal
  pane, cursors and regions), layered image views, FFT panel, masks,
  decomposition (PCA etc.), DPC / iDPC from segmented images, kemstem,
  in-situ series, background jobs with progress, the plugin host (on the
  `abtem-plugin` branch), and abTEM's long-running worker pattern.

## 2. Dependency decision

**Recommended: pyxem as the 4D-STEM dependency of SI Viewer; py4DSTEM not a
dependency.**

- pyxem is HyperSpy-native, like the rest of the app: its results are HyperSpy
  signals the app already stores, views and saves, and lazy data works the
  same way.
- It installs cleanly into the app's environment, with no downgrades, and its
  zarr pin matches rosettasciio's.
- diffsims and orix come with it. They are what diffraction simulation and
  orientation mapping need next, so they serve the abTEM diffraction work too.
- py4DSTEM pins numpy below 2, so it can only run in its own environment. Its
  unique strengths (ptychography, parallax, probe-kernel disk detection, ACOM)
  can come later as an optional plugin with its own environment and worker,
  exactly like abTEM, if they are wanted.
- py4DSTEM's emdfile `.h5` files are read with a small h5py reader in SI
  Viewer, without py4DSTEM.

**How pyxem is included (decided 2026-11-25): a core dependency** of
`si-viewer`, so its processes appear for every user. The viewer (phases 0–2)
needs only numpy and dask; pyxem is added to the dependencies when the first
process using it is built (phase 2's pixel repair or phase 3).

## 3. Design

- **Data kind:** a new `DatasetKind.DIFFRACTION4D` ("4D-STEM") for 2
  navigation + 2 signal axes, with calibrations in real space (Å/nm) and
  reciprocal space (1/Å or mrad). Its own badge colour.
- **Loading:** in memory up to a limit (as spectrum images, 4 GB), lazy beyond
  it; optional binning in Q and R on open for big data. Readers: rosettasciio,
  plus the emdfile reader for py4DSTEM `.h5` (data, calibration, later its
  Bragg vectors).
- **View:** two panes side by side, like the spectrum-image view: the virtual
  image (navigator) and the diffraction pattern, both layered image views
  (zoom, contrast, colormaps, magnifier, scale bars), resizable.
  - Real space: a cursor (point) or region (rectangle and the other ROI
    shapes); the pattern shown is the one at the cursor, or the mean or
    maximum over the region, or of the whole scan.
  - Diffraction: virtual detectors drawn on the pattern: circle (BF), annulus
    (ADF), rectangle, point, several apertures by click, g-vector arrays.
    Response: sum, mean, maximum, COM x/y, iCOM.
  - The virtual image updates live (computed in the backend from the in-memory
    array, cached; for lazy data a job with progress).
  - Pattern display linear, log or power; mean and maximum pattern always one
    click away.
- **Results are datasets:** a virtual image saved from the view is an image
  dataset under the 4D dataset, with its detector in its settings, so every
  image process applies (masks with Otsu, kemstem, FFT, filters, layers and
  colour composites, grids to compare). This replaces the plugin's previews,
  merged view and mask dialog with the app's own grid, merge, layers and Masks.
- **Processing:** processes in the Process tab of a 4D dataset, sections
  General, Calibration, Virtual imaging, DPC, Radial, Peaks and strain,
  Orientation. Tunable steps are guided: settings, a preview in the viewer,
  then the heavy part as a cancellable job, as elsewhere in the app.
- **Simulation link:** abTEM's pixelated-detector STEM output (4D-STEM phase in
  the abTEM plan) becomes a 4D dataset in a simulation project, viewed and
  processed the same way, and comparable with experiment.

## 4. Rollout

Each phase ends with something usable; one change at a time, tested on the
user's data.

### Phase 0: data (built, 2026-11-25)
- `DatasetKind.DIFFRACTION4D` ("4dstem", badge "4D-STEM") for 2 scan + 2
  pattern axes; axes named x, y (scan) and kx, ky (pattern) whatever the
  reader called them (a DM 4D file names its scan axes "<undefined>" and "z").
- Readers: rosettasciio for `.dm4` (checked: `examples/4dstem/Au_NP_experiment.dm4`,
  60 × 60 × 128 × 128 uint8, nm and 1/nm), `.mib`, `.blo`, `.hspy`; a new
  emdfile reader (`io/py4dstem.py`, h5py only) for py4DSTEM `.h5` (checked:
  `sim_Au_data_binned.h5`, 100 × 84 × 125 × 125 uint16): its DataCubes, and
  its 2D arrays (virtual images, diffraction slices) as images, with
  py4DSTEM's calibration (R/Q pixel sizes and units; the whole calibration kept
  in metadata). `[Rx, Ry, Qx, Qy]` keeps py4DSTEM's orientation (Rx the scan's
  rows). Bragg vectors and other groups are skipped for now. Other `.h5` files
  are refused with a clear message.
- In memory up to 4 GB on open, lazy beyond (the emdfile reader keeps the file
  open and reads per scan row).
- Frames: the scan's total intensity (`projection=scan`, also the thumbnail),
  and the mean and maximum pattern (`mean-pattern`, `max-pattern`).
- Bin Q / Bin R: the Rebin dialog and process on 4D data (pairs y–x and
  ky–kx linked), in place or as a copy; sizes cropped to whole bins first;
  integer data kept integer, the smallest type that holds the sums.
- A first view (`FourDView`): the scan beside the mean or max pattern,
  linear or log (log by default), each a normal image view.
- Tests: `backend/tests/test_4dstem.py` (synthetic emdfiles).
- **Check:** EMD Velox 4D, Arina and EMPAD with real files when available.

### Phase 1: the 4D viewer (built, 2026-11-25)
- `FourDView`: the virtual image (left) and the diffraction pattern (right),
  both normal image views (zoom, contrast, colormaps, FFT panel, PNG).
- Scan: *Cursor* (click or drag; ← → ↑ ↓ with the scan focused, Shift ×5) or
  *Region* (drag a rectangle); the pattern shown is the one at the cursor, or
  the mean or max over the region or the whole scan (*Cursor / Region /
  Scan*, *Mean / Max*); *Linear / Log* (log by default).
- Virtual detector on the pattern: *Circle*, *Annulus*, *Rectangle*,
  *Point*; drag a handle to resize, inside to move (a press elsewhere pans:
  it doesn't move the detector, so a double-click fits the view); W/A/S/D (and arrows with the pattern focused) move it, Shift ×5;
  responses *Sum*, *Mean*, *Max*; size shown in the pattern's calibration.
  The virtual image follows live (one request at a time, the newest next).
- *Save image*: the virtual image as an image dataset under the 4D data
  (origin "virtual", its detector and response kept in `virtual`, saved with
  analyses), titled e.g. "Virtual annulus 5.35–29.7 1/nm".
- Backend: `processing/fourd.py` (detector masks by pixel centres, virtual
  images by gathering the detector's pixels, patterns at a position or over a
  region; numpy or dask), `api/fourd.py` (`/4d/pattern`, `/4d/virtual`,
  `/4d/virtual/save`).
- Per dataset, the detector, cursor and region are kept while the app runs.
- Not yet: COM / iCOM responses (phase 4); virtual images of lazy (> 4 GB)
  data are computed in the request, not as jobs with progress; power scale.

### Phase 2: better virtual imaging (built, 2026-11-25)
- pyxem 0.21 is now a dependency (`pyproject.toml`) and installed in the
  `si-viewer` env (14 packages added, none changed).
- *Apertures* detector: a ⌘/Ctrl-press on the pattern places a disc (drag
  to move; a plain press off the discs pans),
  the handle beside the selected one sizes them all, right-click or Delete
  removes one; *BF radius* sizes them as the bright-field disk of the mean
  pattern; *Clear*. Backend: `Detector.shape = "apertures"` (union of discs).
- *g-vector array*: apertures at O + (i + s1) g1 + (j + s2) g2, i in −n1…n1,
  j in −n2…n2 (shifts 0.5 for half orders), with or without O; O starts at
  the bright-field disk's centre; O, g1 and g2 dragged on the pattern. Sent as
  apertures, the array kept with saved images (`detector.lattice`).
- Bright-field disk (`/4d/bf-disk`): pixels above halfway between the median
  and the maximum, connected to the maximum of a 3 × 3 median-filtered pattern
  (so a hot pixel isn't taken for the beam); centre and equal-area radius.
  Checked: dm4 (64.9, 64.2) r 5.3 px; py4DSTEM file (63.8, 64.0) r 5.1 px.
- *Bad pixels* (Process tab of 4D data): hot (pyxem `find_hot_pixels` on the
  mean pattern, threshold = × the mean intensity, default 5, outside 1.3 ×
  the BF disk whose edge is steep) and dead (below × the median of the 8
  around, default 0.2) found and ringed on the pattern; *Repair* in place or
  as a copy (pyxem `correct_bad_pixels`: the mean of the 8 around), then found
  again so the rings show what is left. Border pixels are left as they are
  (pyxem's repair has no neighbours there) and counted. Repaired data becomes
  float32 (rounded means of sparse counts left the pixels dead-looking).
  Checked on the dm4: 2 hot at (9, 11), (20, 22); none after.
- Comparing detector sets: saved virtual images in the existing merge / grid
  and layers (checked).
- Layout: the detector's settings live in the *Virtual image* process (Process
  tab of 4D data, first): shape, response and the detector readout,
  apertures (*BF radius*, *Clear*, *g-vector array* with n1/s1/n2/s2 and
  *Origin*) and *Save image*. The detector is still drawn and dragged on the
  pattern.
- Scan pane: any number of cursors (⌘/Ctrl-click adds one, a press on one
  grabs it, elsewhere moves the active one; right-click or Delete removes
  one; arrows move the active one), each coloured and numbered. Regions are
  the viewer's own ROIs (rectangle, ellipse, polygon tools), kept per view;
  a click inside one, or drawing it, shows its mean pattern (backend: the
  shared `regions.Region` mask, numpy or dask). *Image*: the live virtual
  image or any saved virtual image of this dataset.
- Pattern pane: a list of the cursors and regions (each with its colour) to
  pick the pattern shown, the pattern framed in that colour; *Mean / Max*
  over the whole scan (no frame); *Linear / Log*.
- One set of tool buttons (line, FFT, rectangle, ellipse, polygon) on the
  scan pane; the tool chosen works on both panes. The pattern pane keeps its
  hide-annotations and loupe buttons (`LayeredViewer` `toolButtons={false}`).

### Phase 2b: detector groups and multi-panel views (proposed, 2026-11-26)
Several virtual images at once (one per *detector group*) and several
patterns at once (one per cursor or region), shown side by side.

**Detector groups (Virtual image process).**
- A group is one detector, of any shape (circle, annulus, rectangle, point,
  apertures with or without a g-vector array), with its response, a name
  ("BF", "ADF", "Group 3") and a colour. Today's single detector becomes the
  first group.
- The process lists the groups (colour dot, name, eye to hide, ⋯ menu:
  rename, colour, duplicate, delete) with *+ Group* below; the controls
  under the list edit the selected group. *Save image* saves the selected
  group's image; *Save all* saves one image per visible group.
- Presets in the + menu: *BF* (disc at the BF radius), *ADF* (annulus from
  the BF radius to the pattern edge), *BF + ADF*, and *Duplicate*.
- Every group's detector is drawn on every pattern in its colour; the
  selected group's has handles, the others are thinner and are selected by
  a click on their outline.
- Backend: one request returns the virtual images of all groups: the masks
  stacked as (n, ky, kx) and contracted with the data in one pass (numpy or
  dask), so dragging one group recomputes only that group's image.

**Layouts (a *Single / Grid / List* switch on the 4D view's toolbar).**
- *Single* (today): one virtual image (the selected group's, or a saved
  one) beside one pattern (the chosen cursor or region, or the mean or max)
  with every group drawn on it.
- *Grid*: two areas side by side, the virtual images (one panel per visible
  group, framed and titled in its colour) and the patterns (one panel per
  cursor and region, framed in its colour, plus the mean or max if chosen),
  each laid out in rows and columns like the existing image grid. All
  virtual-image panels share pan and zoom, and so do all pattern panels.
  Cursors and regions show on every virtual image; every group shows on
  every pattern. A click on a panel makes that group or source the active
  one; double-click opens it in *Single*.
- *List*: one row per visible group: its pattern (the active source, with
  only that group's detector, editable) next to its virtual image, like the
  plugin's preview panes. Rows have a fixed height (a row-size slider in
  the toolbar), the pane scrolls, and a scroll bar is always shown on its
  right (styled, so macOS's hidden overlay scroll bars don't apply).

**Scrolling without zooming (Grid when it overflows, List).**
- In a scrolling layout, the wheel or a two-finger swipe scrolls the pane
  wherever the pointer is; ⌘/Ctrl + wheel zooms the panel under the pointer,
  and so does a trackpad pinch (browsers send it as a wheel with Ctrl).
  Single keeps plain-wheel zoom.
- Whether a gesture zooms or scrolls is decided when it starts and kept
  until 0.4 s pass without a wheel event, so momentum or a key let go
  partway doesn't switch it (as the plugin's `WheelRouter`).
- Fields in the rows (numbers, menus) take the wheel only once clicked into.
- Keyboard: Page Up / Page Down move a row; with a panel focused, the
  arrow keys keep moving the cursor or detector as now.

**Built (2026-11-26): groups.** `FourDState.groups` (`DetectorGroup`: id,
name, colour, visible, detector, response) and `selectedGroup`. The process
lists them in Layers-panel rows (eye, colour swatch, double-click to
rename, ×); a group's shape is chosen when it is added (no shape switch):
*+ Group* adds Circle, Annulus, Rectangle, Point, Apertures (empty), BF, ADF (annulus from
the BF radius to the nearest pattern edge), BF + ADF, or a duplicate of the
selected group; *Save image* (selected) and *Save all* (visible); a named
group's image is titled e.g. "BF: r 2.355 1/nm". The pattern shows every
visible group's outline in its colour (the selected one with handles); a
press near another group's outline selects it. The scan pane's list picks
the group shown live (or a saved image) and frames it in the group's
colour. Single view computes only the shown group's image, so the batched
request moves to Grid.

**Built (2026-11-26): Grid.** *Single / Grid* in the scan pane's footer.
Grid: the scan pane is a grid of the visible groups' virtual images, the
pattern pane a grid of the cursors' and regions' patterns (plus the scan's
mean or max when chosen), each panel framed in its colour; both are
`LayeredViewer`s with `gridOnly` and computed `stacks` panels (each with its
own geometry), so the toolbar, contrast, tools and region drawing are the
usual ones and the panels of each side share pan and zoom. Cursors and
regions show on every virtual image, every group on every pattern. A press
on another panel only makes it active (selects that group or pattern
source); the next press acts. Virtual images come from
`POST /4d/virtuals` (`virtual_images`: all at once, lazy data read once);
only the groups that changed are fetched again (`useFrameSet`). Switching
layout re-fits both panes. Contrast is redone per frame shown (its key),
in Single too. Not yet: double-click a panel to open it in Single.

**Pane order (2026-11-26).** The pattern is always on the left and the
virtual images on the right, in every layout (`.fourd-pattern { order: -1 }`);
the pattern pane's toolbar is left-aligned again (at the window's edge).
A top bar over both panes holds *Single / Grid / List* and the image tools
(line, FFT, rectangle, ellipse, polygon: `ImageToolButtons`, shared by both
panes); each pane's toolbar keeps hide-annotations, loupe and contrast.
List has the same top bar: its rows' images take the same tools
(`useViewTools`: lines, FFT boxes and regions kept under the Single panes'
keys, so what is drawn in one layout shows in the others; line profiles are
plotted in Single).

**Built (2026-11-26): Merge.** *Single / Grid / Merge / List*. Merge: the
visible groups' virtual images as layers of one view, each in its group's
colour, added up (the viewer's merged mode: the Layers panel shows, hides
and fades each; `StackLayer.colormap` gives the starting colour, and a colour
changed in the process follows); the label names the groups in their
colours. The pattern pane is as in Single (every group drawn on it).
*Dominant / Max / Add* (`mergeBlend`, Dominant by default) under the image:
Dominant keeps at each pixel only the group brightest there, judged within
each layer's contrast limits and among the layers shown, the others set to
their lower limit (done on the frames, then drawn added: one colour per
pixel, never white); Max keeps the brighter colour per channel (WebGL
`MAX` blending, `blend` prop down to the renderer); Add sums the colours.

**Built (2026-11-26): List.** *Single / Grid / List*. List: a row per
visible group (its colour down the left edge, name and detector in the
header): the pattern shown (the chosen cursor, region, mean or max, framed
in its colour) with only that group's detector, editable once its row is
selected, next to the group's virtual image with the cursors and regions.
The patterns of all rows share pan and zoom, as do the images. Rows are
`--row-h` high (*Rows* slider, 160–640 px); the list scrolls with a styled
scroll bar that stays visible. `ImageView` `wheelZoom="modifier"`: over an
image the wheel, a swipe or a pinch zooms, and ⌘ + wheel (Ctrl + wheel off
macOS) scrolls the list (the view scrolls its `data-wheel-scroll` ancestor
itself); the scroll bar, or the wheel over a row's header or contrast strip,
also scrolls; decided per gesture and kept until the wheel rests 0.4 s (one
latch for all views). A
press in another row selects its group only. The bottom bar holds the
layout switch, the pattern picker, *Mean / Max*, *Linear / Log*, *Rows* and
the cursor. Each image of a row has its own contrast strip above it (the
usual `ContrastControls`; each row's pattern its own contrast too); the top
bar has one eye (hides regions, lines and boxes on every row:
`AnnotationsToggle` over both views' keys) and the magnifier toggle (linked
along each side).

**Order of work.** (1) groups in the process and all groups drawn on the
pattern (built); (2) Grid, with the batched virtual images (built); (3) List (built); (3) List with the scroll bar and
wheel handling; (4) optional colour composite of the groups (the image
grid's *Merged* mode with each group's colour), and presets.

**Questions.**
- Grid: one shared pattern area beside the virtual images (proposed), or
  each virtual image paired with its own small pattern?
- List with several cursors or regions: each row shows the active source's
  pattern (proposed), or one pattern per source in each row?
- Groups of any detector shape (proposed), or apertures only?

### Phase 3: calibration and centring (pyxem)
- Direct beam position and centring (methods: blur, centre of mass,
  cross-correlation), descan correction (linear plane), shown in the viewer
  before applying.
- Reciprocal pixel size from a known standard or entered camera length;
  ellipticity / affine distortion from a ring; scan–diffraction rotation.
- Calibrations kept with the dataset and saved with the analysis.

### Phase 4: COM and DPC
- Centre of mass → beam shift (x, y) → magnitude, direction (colour wheel),
  iDPC / phase retrieval, magnetic DPC; feeding the existing DPC display.
- Rotation calibration by minimising curl, as the guided steps in the viewer.

### Phase 5: radial, azimuthal and variance
- Azimuthal integrals (1D and polar 2D), radial profiles per pixel (a
  spectrum image of q, using the existing spectrum-image tools), virtual
  annular images from profiles.
- Variance / FEM, background subtraction, reduced intensity and PDF for
  amorphous samples.

### Phase 6: Bragg peaks and strain
- **Built (2026-11-26): Disk detection process** (`processing/disks.py`,
  `/4d/disks/*`, `DiskDetectionSection`). Template: the probe (the mean
  pattern around its bright-field disk, shifted so the disk is centred,
  zero beyond 1.4 r) or a flat disc. Per pattern: pyxem's normalised
  cross-correlation (`normalize_template_match`, edge-padded so the border
  gives no false disks), HyperSpy's `find_local_max` (threshold, spacing,
  edge, max disks), sub-pixel centre of mass of the correlation (3 × 3).
  Tuned live on the patterns shown (every cursor/region in Grid) while the
  process is open; *Find in every pattern* is a job (scan rows in parallel
  threads; the Au NP dm4, 14 400 patterns of 128², ≈ 12 s on the M3 Max CPU)
  making a **Bragg vectors** table under the data (kind "other", origin
  "disks": one row per disk, scan_y, scan_x, ky, kx, score; selecting it
  shows its data) and a **Bragg vector map** under that (bilinear
  histogram, 2 × finer than the pattern). Found again in place; saved with
  the analysis. Disks are drawn (circles of the template radius) only while
  the process is shown: live with *Show on the patterns*, else the stored
  ones at the cursors.
  No GPU: CUDA (cupy) isn't available on Apple silicon; the correlation step
  could later get an optional Apple-GPU (PyTorch MPS / MLX) backend.
- Still to do: diffraction vectors as pyxem `DiffractionVectors` for its
  clustering/strain/indexing; calibrations applied to stored vectors (keep
  raw positions, py4DSTEM-style); Bragg vectors from py4DSTEM files.
- Lattice vectors, strain maps (exx, eyy, exy, rotation) with the same display
  and conventions as kemstem's strain; snap g-vector arrays to peaks.

### Phase 7: orientation and phase mapping
- Template matching against diffsims simulations from CIF structures (the
  abTEM plugin's structures), orix orientation maps (IPF colours), phase maps.
- Shares structures and diffraction simulation with the abTEM diffraction
  work.

### Phase 8: decomposition and segmentation
- The existing decomposition (PCA, NMF, …) on 4D data, VDF segmentation,
  clustering of patterns and vectors.

### Phase 9: py4DSTEM extras, as needed
- An optional plugin with py4DSTEM in its own environment and worker (as
  abTEM): parallax / tilt-corrected BF, ptychography (GPU where available),
  probe-kernel disk detection, ACOM, whole-pattern fitting.

### Phase 10: simulation and in-situ
- abTEM 4D-STEM (pixelated detector) opened in the same viewer; comparison
  with experiment.
- In-situ 4D-STEM series (frame slider, as the in-situ spectrum images).

## 5. Decisions for the user

1. **Dependency (decided 2026-11-25):** pyxem only; py4DSTEM not a dependency.
2. **pyxem in core (decided 2026-11-25):** a core dependency, not an extra.
3. **Branch (decided 2026-11-25):** `4dstem`, from `main` (`f9f43bf`), so
   4D-STEM can reach `main` without abTEM. It doesn't have the plugin host,
   the fixed profile lines, grab cursors and badge changes, or the PLD and FFT
   panel changes, all on `abtem-plugin`; 4D-STEM is core code, so it doesn't
   need the plugin host. The abTEM link (phase 10) comes when the branches
   meet. Work on it in a separate worktree (e.g. `../SI_viewer-4dstem`) so the
   abTEM checkout and its session keep running.
4. **py4DSTEM extras plugin (phase 9):** wanted, and which features first.
5. **Test data:** the simulated Au file for now; which experimental datasets
   (format, size) to test phases 0–3 on.
6. **Licence:** pyxem and py4DSTEM are GPL-3, like HyperSpy and kemstem; the
   open licence decision covers them too.
