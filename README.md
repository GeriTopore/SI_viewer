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
