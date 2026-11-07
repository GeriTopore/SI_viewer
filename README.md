# SI Viewer

A GUI for viewing and processing STEM data: images, image stacks and
spectrum images (EDX first, EELS later) from ThermoFisher Velox `.emd` files.
Processing uses [HyperSpy](https://hyperspy.org) and
[exSpy](https://exspy.readthedocs.io).

- **Backend:** Python (FastAPI) running HyperSpy/exSpy in-process — `backend/si_viewer`
- **Frontend:** React + TypeScript (Vite) — `frontend/`

See [docs/PLAN.md](docs/PLAN.md) for the architecture, feature list and milestones.

## Setup

```bash
conda env create -f environment.yml   # creates the `si-viewer` env, installs this package in editable mode
conda activate si-viewer
cd frontend && npm install && npm run build && cd ..
```

`npm run build` writes the frontend into `backend/si_viewer/static/`, which the
backend serves.

## Run

```bash
si-viewer                      # opens http://127.0.0.1:8765 in the browser
si-viewer path/to/file.emd     # open files at startup
si-viewer --no-browser --port 8765
```

### Example data

**Examples ▾** in the top bar opens bundled datasets that work offline:
exSpy's FePt (TEM) and TM002 (SEM) EDS spectra, HyperSpy's synthetic
atomic-resolution image and two-Gaussian spectrum image, and a synthetic
64×64 EDS spectrum image of core-shell FePt particles (built from the FePt
spectrum, with Poisson noise). They are defined in
`backend/si_viewer/examples.py`, ready to back a tutorial later.

### Processing and FFT

The right-hand column shows a live FFT of the visible part of the image in the
viewer. Hover over it to read the spatial frequency and the matching spacing
(d = 1/|k|) in ångström, e.g. to check resolution or lattice
spacings. Below it, the **Process** tab rebins and filters the data in view
without changing the loaded data:
- **Spectrum images:** Gaussian or mean spatial filters and spectral
  smoothing. They apply to the whole spectrum image, whatever maps are
  overlaid: its spectra, all its element maps, and the fits and
  quantification in the Quantify tab.
- **Images and stacks:** Gaussian, median or mean filters.

### Layout

Drag the gaps beside the side panes to resize them, and the bar between the
image and the Spectrum / Line profile tabs to change their heights.
Double-click a handle to reset it.

### Line profiles

Pick the ╱ tool (beside Pan, or among the spectrum-image navigator's tools)
and drag across the image:
- The **Line profile** tab under the image plots the profile of every image
  layer shown (spectrum images also have a **Spectrum** tab there). Overlay
  element maps to get element or composition profiles.
- Set the width to average across the line, normalise, or export the profiles
  as CSV.
- On a spectrum image, click or drag on the profile to see the spectrum of
  that part of the line.

### Saving

- Right-click a dataset (or use Save in the Info tab) to save it:
  - images and maps as TIFF, `.hspy` or CSV;
  - spectra as EMSA/MSA, CSV or `.hspy`;
  - stacks and spectrum images as `.hspy` (stacks also as TIFF);
  - all of a file's maps as a zip of TIFFs.
- **Save PNG** under the image saves the composite as shown, with a scale
  bar.
- The spectrum plot, the quantification table and line profiles save as CSV
  (the plot also as PNG).

### Keyboard shortcuts

Press `?` (or the ? button) for the list. Highlights:
- tools C/R/E/P/L/H;
- F to fit, 1 for 1:1, ⌘/Ctrl S to save the image;
- [ and ] to fold the side panes, T to switch the tab under the image.

### Frontend development (hot reload)

Run the backend and the Vite dev server side by side; Vite proxies `/api` to
the backend:

```bash
si-viewer --no-browser
cd frontend && npm run dev     # http://localhost:5173
```

## Tests

```bash
pytest
```

Tests that need data use the first `.emd` in the repository root (ignored by
git), or the file named by `SI_VIEWER_TEST_EMD`; they are skipped otherwise.

## Notes

**VeloxLite files.** Velox version 13 files declare `"format": "VeloxLite"`,
which RosettaSciIO (0.14 and 0.15) does not recognise, so `hs.load` fails
with "The file is not a supported EMD file". `si_viewer.io.emd` makes the
existing Velox reader accept them. The patch can be removed once RosettaSciIO
supports the format.
