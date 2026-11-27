# Handover (2026-11-27, late evening)

Read this first. Then:
- `docs/4DSTEM_PLAN.md`: the 4D-STEM design and what is built; phase 7 is the
  current orientation-mapping work.
- `docs/PLAN.md`: the rest of the app.
- `docs/REMOTE_PLAN.md`, on the `remote` branch: the SSH and CX3 work.
- The user's todo list, which decides what is planned:
  `/Users/gt622/Documents/Obsidian Vault/Codes/Si-viewer.md`.

This file covers what those don't: the checkouts and branches, the servers,
the state of the work, how the user works, and what is open.

## The project

SI Viewer is a local web app for STEM spectrum images (EDS, EELS), images,
stacks, in-situ series and 4D-STEM.
- **Backend:** FastAPI, HyperSpy, exSpy, pyxem, diffsims and orix
  (`backend/si_viewer`).
- **Frontend:** React, TypeScript, Zustand, uPlot and three.js (`frontend/src`).
- **Repo:** GitHub `GeriTopore/SI_viewer`.

## Checkouts and branches (git worktrees of one repo)

| Folder | Branch | State |
|---|---|---|
| `/Users/gt622/Documents/Dev/SI_viewer` | `abtem-plugin` (43ae9c9) | The abTEM plugin's work, separate. Holds `.claude/launch.json` and `examples/` (git-ignored). |
| `/Users/gt622/Documents/Dev/SI_viewer-4dstem` | `4dstem` | **Current work.** Everything committed and pushed (the phase map and the Match tab's "All matches" list are the last commit). |
| `/Users/gt622/Documents/Dev/SI_viewer-remote` | `remote` (a773a1d, pushed, clean) | `si-viewer remote HOST` (SSH launcher, token auth, install). |

- `main` and `origin/main` are at `d652fa6`, the merge of the remote work and
  4D-STEM's COM and calibration changes.
- Since `d652fa6`, `4dstem` and `remote` have moved apart:
  - `4dstem` has the orientation work (`780b69d`, `f7b2d37`, `3f2513c`,
    `590b050`, then the phase map).
  - `remote` has one fix: a server deletes the session files of servers no
    longer running.
- Merge them when the user asks.

`frontend/node_modules` in the 4dstem and remote worktrees is a symlink to the
main checkout's. Never commit it, `.claude/` or `examples/`.

## Run, build, test (4dstem worktree)

- **Build:** `npm --prefix /Users/gt622/Documents/Dev/SI_viewer-4dstem/frontend run build`.
  It writes `backend/si_viewer/static`, which the server serves. After a
  frontend change, reload the page; the server can stay up.
- **Test:**
  `PYTHONPATH=/Users/gt622/Documents/Dev/SI_viewer-4dstem/backend /Users/gt622/miniforge3/envs/si-viewer/bin/python -m pytest -q -p no:cacheprovider /Users/gt622/Documents/Dev/SI_viewer-4dstem/backend/tests`
  - It takes about 2.5 min; run it in the background.
  - Last run: **465 passed, 140 skipped**.
  - While iterating, run only `tests/test_orientation.py` (12 tests, about 30 s).
- **Token auth:** since the remote merge, a server needs its token unless
  started with `--no-token`. Every 4dstem configuration in `.claude/launch.json`
  has `--no-token`.

### Servers (launch configurations in the main checkout's `.claude/launch.json`)

| Port | Configuration | What |
|---|---|---|
| 8768 | `si-viewer-4dstem` | **The user's own server.** 4dstem code from 26 Nov, so it predates all the orientation work. **Ask before restarting it.** |
| 8769, 8770 | `si-viewer-4dstem-test`, `-claude` | Old test servers. |
| 8771 | `si-viewer-4dstem-orientation` | **The orientation test server**: 4dstem code with the latest backend and frontend build (restarted after the All matches endpoint), `--no-token`. The user also uses it in the browser pane. |

**Restarting 8771** (needed after any backend change). Everything is held in
memory, so save first:
1. Save:
   `curl -s -X POST localhost:8771/api/project/save -H 'Content-Type: application/json' -d '{"path": "<file>.hspy", "level": "link", "overwrite": true}'`
2. Stop and start it: `preview_stop`, then `preview_start` with
   `si-viewer-4dstem-orientation`.
3. Reopen: `POST /api/project/open {"path": …}`, then reload the tab.

The latest save is
`/Users/gt622/Documents/Dev/SI_viewer/examples/4dstem/SPED-Ag orientation analysis.hspy`.
It holds:
- SPED-Ag, its polar patterns, the Bragg vectors and BVM, and the Orientation
  map with its images (made from the polar patterns and Ag · 1°);
- Ag and its "Templates · 1°".

Orientation choices (Templates, Polar patterns, Match orientation, Scale,
structure views) are kept in the browser's localStorage, so a reload keeps
them. Match results aren't kept.

### Test data

- **SPED-Ag:** Examples → "Ag, precession (pyxem)", `example:pyxem-sped-ag`.
  - Its pattern pixel size must be 0.02673 1/Å: the tutorial's ×2 fix of the
    raw 0.01336.
  - In the saved analysis it is already set, and saved analyses now keep
    edited calibrations.
  - The Ag structure is `examples/4dstem/Ag.cif` in the main checkout.
- **Fe grains (phase map):** Examples → pyxem's simulated bcc/fcc Fe grains
  (`pyxem-fe-grains`, opens centred). Structures `examples/4dstem/Fe-bcc.cif`
  and `Fe-fcc.cif` in the main checkout (as pyxem's phases). Test options:
  resolution 2°, no precession, excitation error 0.1 1/Å, reciprocal radius
  2.5 1/Å.
- **The tutorial followed:**
  `/Users/gt622/Documents/Dev/Hyperspy_Workshop_2025/pyxem/Orientation_mapping_SPED-Ag.ipynb`.

## Orientation mapping: what is built (details in 4DSTEM_PLAN.md phase 7)

- **Structures and templates:**
  - Structures from `.cif`, with a 3D view (Structure tab; main zone axes in a
    dropdown).
  - Templates (diffsims) as a process of a Structure.
  - Polar transform, which makes **polar patterns**: their own object, badge
    Polar, with only Filters, Virtual imaging and Orientation mapping.
- **Orientation mapping** (Process tab of the 4D data or its polar patterns):
  - **Templates** dropdown: drawn as a layer, following the Structure tab's
    view.
  - **Scale**: log slider ¼× to 4×, plus "Use as pixel size".
  - **Polar patterns** choice: warns when its source's calibration changed.
  - **Match current position**: matches at every cursor. Matches are cached
    by polar patterns, templates and position in `useOrientation.matchCache`,
    so the data and its polar patterns share them.
  - **Map**: the whole scan as a job, 12 s on SPED-Ag. It makes an
    "Orientation map" object with Correlation and IPF-x, IPF-y, IPF-z images.
- **Display:**
  - The viewer draws **RGB layers**: `LayerData.rgb`, colours darkened by the
    value through the contrast. That is the IPF maps' correlation overlay.
  - The IPF images have an "IPF key" bottom tab.
  - On the patterns, at every cursor:
    - Best match (orange): from Match current position.
    - Orientation map (cyan): from the map, with hkl labels and an IPF inset,
      as the tutorial's `to_markers(annotate=True)` and `to_ipf_markers`.
    - Template layer (pink).
    All three are drawn at angle and radius on polar patterns.
- **Shared state:**
  - Cursors are shared by a file's 4D-STEM data with the same scan
    (`useFourD.scans`, keyed by `scanKey`).
  - Matches are shared too, so switching between the data and its polar
    patterns shows the same match.
- **Match orientation** (Structure tab checkbox) turns the 3D view to the
  match at the active cursor; its template then lands exactly on the best
  match.
- **Conventions that took work to get right** (each has a test):
  - `processing/orientation.py: view_of`: towards the viewer is orientation ×
    z, the screen's up is orientation × −y, both as [uvw].
    - A pattern exactly on a zone axis can't check this: a half turn about
      [101] is a crystal symmetry. The test uses a pattern 3° off.
    - When a convention matters, check it numerically on the real data
      through the API first.
  - pyxem's `to_vectors` hkl are the turned template's, rounded. The app
    labels reflections with the crystal's hkl (`_crystal_hkl`).
  - Marks on patterns are drawn at pixel centres (+0.5).
  - diffsims 0.7 crashes when an orientation has no reflection (for example,
    no precession with s = 0.001 1/Å). The app turns that into a message
    saying what to change.

### Commits

- `3f2513c` added the whole-scan **Map** and what it needs:
  - `match_scan`, `orientation_map`, `ipf_colours`, `map_at` and `describe`
    in `processing/orientation.py`;
  - `/4d/orientation/map`, `/orientation/ipf` and `/orientation/at`;
  - `session.set_orientation_map`;
  - the RGB renderer;
  - the IPF key tab;
  - the Orientation map layer.
- `590b050`:
  - the map's markers are fetched one request at a time, newest cursor
    positions first;
  - Template, Best match, Orientation map and Bragg disk marks share one
    Mark choice (circle or "+");
  - diffraction patterns open in Linear contrast.

### The last commit: the phase map and All matches

The phase map was built in another session. Details in `4DSTEM_PLAN.md`
phase 7, "Phase map" and "All matches".

- **Choosing phases** (`OrientationSection.tsx`): the Templates section is a
  list, one row per phase with its colour, and a "+ Template ▾" menu (one set
  per Structure; others disabled). The first row is the one drawn on the
  patterns and in the Structure tab; clicking another row moves it first; ×
  removes. Other phases are `useOrientation.others` (persisted);
  `phasesFor` joins the ids by commas.
- **Backend:** the `templates` field of match and map takes ids joined by
  commas (`api/orientation.py: _phase_set`, refusing two of one Structure).
  pyxem's `get_orientation` runs once per phase; each position keeps the
  phase whose best template correlates best (`best_phase`,
  `combine_matches`). pyxem's own multi-phase output was avoided: its
  `to_phase_index` fails with one result kept, misplaces each phase's first
  template, and its single-phase helpers refuse it.
- **The map** keeps `[phases, y, x, n_best, 4]` (`phase_blocks` reads a
  single-phase map as one phase, so the saved SPED-Ag analysis reads as
  before). Its record has `phases` [{name, templates, colour}] (tab10,
  `PHASE_COLOURS`, the same list in backend and frontend).
- **Images:** Correlation (the best phase's); **Phase** (each position's
  index, drawn with the colour map `stops:<colours>`; "Phase key" tab);
  IPF-x/y/z, each position in its own phase's key and shaded by its
  correlation as a fraction of its phase's best (IPF key tab shows a key per
  phase). RGB images' contrast now starts at 0.
- **Matches** carry `phase`, `phase_index`, `templates` and every phase's
  best (`phases`). The Match tab has a Phase column; its heat map and the
  map's IPF inset use the matched phase's templates.
- **Examples:** pyxem's orientation examples open with the pattern centre in
  the middle (`calibration.center = None`, `examples.py: _pyxem(centre=…)`).
- **Test:** `test_phase_map_of_fe_grains`: > 95 % of the Fe grains' positions
  get the right phase, at the cursors and over the scan.

- **All matches** (Match tab, collapsed): every template matched at the
  active cursor, all phases', best first; clicking a row draws that match on
  the patterns (`POST /templates/describe`, `useOrientation.matchPick`).

The Fe grains session is saved as
`/Users/gt622/Documents/Dev/SI_viewer/examples/4dstem/Fe grains phase analysis.hspy`.
The saved SPED-Ag analysis still reads after the phase map (checked in the
browser).

**Since then:** the Process tab is tiles only (the list view and the
numbers removed), every process has an icon, and the Match tab keeps its
IPF centred above the tables.

## Open items (nearest first)

1. **Phase map:** built and committed (above).
   - **On hold: overlapping grains.** pyxem keeps one best orientation per
     pattern (its runners-up are mostly the same grain). Options: a second
     pass (mask the best template's spots, via `to_vectors` → `to_mask` or
     the Bragg disks, match the rest, keep it if well misoriented by orix),
     NMF components matched one by one (phase 8), or py4DSTEM's vector
     orientation mapping (phase 9).
   - The Ti550 file (MacLaren et al., α/β Ti) was set aside: it holds only
     Bragg vectors.
2. **The vector-masked copy:** in the saved analysis it was deleted, but if it
   is made again, check its pixel size. It inherits whatever the data had when
   it was made, and its polar patterns would map at the wrong scale.
3. **Absolute strain** (todo list): strain against a reference lattice rather
   than the scan's own.
4. **Remote** (`docs/REMOTE_PLAN.md`):
   - R2 next: per-host settings, tmux reconnect, kemstem examples in remote
     installs;
   - R3 CX3 (PBS);
   - R4 speed;
   - R5 upload and download;
   - R6 JupyterHub;
   - R7 Connect… dialog.
   - kspace has three test installs; tidying them was offered, not done.
5. The rest of the todo list: decomposition of 4D data, the py4DSTEM plugin,
   abTEM 4D, licence (GPL).

## How the user works (also in the memory files)

- **Planning:** the Obsidian todo decides what is planned. Keep
  `4DSTEM_PLAN.md` and `PLAN.md` in line, and tick items when built.
- **Code:** use the dependencies (pyxem, diffsims, orix, HyperSpy, kemstem…)
  and write only glue. Small work directly; subagents only for large or
  parallel work.
- **Commits:** commit and push only when asked. Don't offer it after each
  step.
- **UI text:** terse. Labels, values and data-loss warnings only; no
  explanatory paragraphs.
- **UI elements:** new ones reuse existing classes (`q-form`, `field`,
  `layer-choice`, `cell-input`, `process-actions`…).
- **Views:** grid and single views show the same tabs under a viewer; never
  gate a tab on `!grid`.
- **Objects:** a process belongs on the object it acts on. Strain mapping and
  vector mask live on the Bragg vectors; polar patterns are their own object.
- **Guided processing:** settings in the Process tab, a live preview, and the
  heavy part as a cancellable job.
- **Testing:**
  - Test in the browser pane and check numerically, through the API or the
    DOM.
  - The user is often looking at the same tab (8771), so tell them before a
    reload resets their view.
  - Don't run long processes as tests.
- **Shell:**
  - Never `cd X && …` (a hook blocks it). Use absolute paths, `git -C` and
    `npm --prefix`.
  - Use `lsof -sTCP:LISTEN`, never plain `lsof`.
- **Speed:** the user noticed slow turns.
  - Group backend changes so the server restarts once.
  - Check a convention on real data before writing tests.
  - Don't add work that wasn't asked for; propose it instead.
