// The abTEM plugin's frontend: simulation projects and their structures.

import { MAX_ATOMS, MAX_REPEAT, countAtoms, distinctColors, expandAtoms, findBonds } from "./structure.js";
import { ON_AXIS, formatZone, isHexagonal, parseZone, toFourIndex, zoneFinder, zoneVector } from "./zone.js";
import { registerSeries } from "./series.js";
import { STEM_BADGES, activateStem } from "./stem.js";

const API = "/api/plugins/abtem";
const ZONES = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]];
const KEY_STEP = 1;
const SHIFT_STEP = 5;
const ROLL_BUTTON_STEP = 15;
// Atoms are drawn at this fraction of their covalent radius.
const ATOM_SCALE = { initial: 0.45, min: 0.1, max: 1, step: 0.01 };
const BOND_TOLERANCE = { initial: 1.05, min: 0.8, max: 1.5, step: 0.01 };
// A model's axes, beam first.
const MODEL_VIEWS = [
  { uvw: [0, 0, 1], label: "Beam (z)" },
  { uvw: [1, 0, 0], label: "x" },
  { uvw: [0, 1, 0], label: "y" },
];
const PREVIEW_DELAY_MS = 200;
const MODEL_MAX_REPEAT = 200;
const ENGINE_POLL_MS = 400;
const ENGINE_SIZE = "600 MB";
const POTENTIAL = {
  sampling: 0.05, slice: 1, minSampling: 0.01, maxSampling: 0.5, minSlice: 0.1, maxSlice: 10, maxBytes: 2 * 1024 ** 3,
  // Frozen phonons: 0 or 1 configurations is off. σ (Å) is per direction.
  maxConfigs: 64, defaultConfigs: 8, sigma: 0.1, maxSigma: 1,
};
const HRTEM = { energy: 300, defocus: 54.3, cs: 10, minEnergy: 20, maxEnergy: 1000 };
// Each object type's badge, chosen to stay apart from the app's own badges.
const BADGES = {
  structure: "#19432e",
  model: "#e9380c",
  potential: "#b16f43",
  hrtem: "#570536",
  wavefunction: "#1c5c9c",
  intensity: "#192a43",
  amplitude: "#9c8f1c",
  phase: "#910808",
  ctf: "#b14352",
  image: "#e90cae",
  ...STEM_BADGES,
  "thickness-series": "#067418",
  "defocus-series": "#1c3a9c",
};
const SECTION_DELAY_MS = 120;
// CTF: drawn soon after a change, saved a little after the last one.
const CTF_DRAW_MS = 60;
const CTF_IMAGE_MS = 200;
const CTF_SAVE_MS = 1500;
const CTF_DEFAULTS = {
  energy_kv: 300, defocus: 54.3, cs_um: 10, c5_mm: 0, astigmatism: 0, astigmatism_angle: 0,
  semiangle: null, focal_spread: 0, angular_spread: 0, max_angle: 40, profile_angle: 0,
  // The CTF walkthrough's chromatic aberration and energy spread (σ).
  cc_mm: 1.2, energy_spread: 0.35,
};
// The CTF's curves, as abTEM names them.
const CTF_CURVES = [
  { key: "ctf", label: "CTF", color: "var(--accent)", dash: null },
  { key: "temporal envelope", label: "Temporal", color: "#e9a23b", dash: "5 3" },
  { key: "spatial envelope", label: "Spatial", color: "#3bb273", dash: "5 3" },
  { key: "aperture", label: "Aperture", color: "var(--muted)", dash: "2 3" },
];
const ZOOM = { max: 64, speed: 0.003 };
const NO_ZOOM = { k: 1, x: 0, y: 0 };
// Room left round a cross-section in its pane (px).
const SECTION_MARGIN = 16;
// probeExtent: the cell a probe is given by default, so it doesn't wrap into itself.
const WAVE = { semiangle: 20, probeExtent: 20, maxTile: 64 };
const ANGSTROM_PER_UM = 1e4;
const ANGSTROM_PER_MM = 1e7;
// Share of the Nyquist frequency abTEM keeps, to avoid aliasing.
const ANTIALIAS = 2 / 3;
const PARAMETRIZATION_NAMES = { lobato: "Lobato", kirkland: "Kirkland" };
const PROJECTION_NAMES = { infinite: "Infinite", finite: "Finite" };

/** The relativistic electron wavelength (Å). */
const wavelength = (kv) => 12.39842 / Math.sqrt(kv * (2 * 510.99895 + kv));

const potentialLabel = (p) =>
  `${p.sampling} Å · ${p.slice_thickness} Å slices · ${PARAMETRIZATION_NAMES[p.parametrization]} · ${PROJECTION_NAMES[p.projection]}`
  + (p.phonon_configs ? ` · ${p.phonon_configs} phonon configs, σ ${p.phonon_sigma} Å` : "");

// The zone axis each structure's view is on, for its Create model process.
/** A round step (1, 2 or 5 × 10ⁿ) near ``rough``, for axis ticks. */
const niceStep = (rough) => {
  if (!(rough > 0)) return 1;
  const p = 10 ** Math.floor(Math.log10(rough));
  const f = rough / p;
  return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * p;
};

// A CTF's settings as they are being edited (before they are saved), so its
// view follows the form at once.
const ctfDrafts = new Map();
const ctfDraftListeners = new Set();
const setCtfDraft = (id, settings) => {
  ctfDrafts.set(id, settings);
  ctfDraftListeners.forEach((listener) => listener());
};
const subscribeCtfDraft = (listener) => {
  ctfDraftListeners.add(listener);
  return () => ctfDraftListeners.delete(listener);
};

const viewZones = new Map();
const viewZoneListeners = new Set();
const setViewZone = (id, key) => {
  if (viewZones.get(id) === key) return;
  viewZones.set(id, key);
  viewZoneListeners.forEach((listener) => listener());
};
const subscribeViewZone = (listener) => {
  viewZoneListeners.add(listener);
  return () => viewZoneListeners.delete(listener);
};

const STYLE = `
.structure-view { position: relative; flex: 1; display: flex; flex-direction: column; min-width: 0; min-height: 0; outline: none; }
.structure-view:focus-within .structure-canvas { box-shadow: inset 0 0 0 1px var(--accent); }
.structure-zones { flex: none; display: flex; flex-wrap: wrap; align-items: center; gap: 3px; padding: 4px 8px; border-bottom: 1px solid var(--border); }
.structure-zones button { padding: 2px 6px; font-size: 11px; font-variant-numeric: tabular-nums; color: var(--text); background: none; border: 1px solid var(--border); border-radius: 4px; }
.structure-zones button.active { color: var(--accent-text); background: var(--accent); border-color: var(--accent); }
.structure-zones input { width: 76px; padding: 2px 5px; font-size: 11px; color: var(--text); background: var(--bg); border: 1px solid var(--border); border-radius: 4px; }
.structure-zones input.invalid { border-color: #e5484d; }
.structure-zone-error { font-size: 11px; color: #e5484d; }
.structure-options { flex: none; display: flex; flex-wrap: wrap; align-items: center; gap: 4px 12px; padding: 4px 8px; font-size: 11px; border-bottom: 1px solid var(--border); }
.structure-options .cell-input { width: 46px; }
.structure-options input[type="range"] { width: 72px; }
.structure-options .spacer { flex: 1; }
.structure-options button { padding: 2px 7px; font-size: 11px; color: var(--text); background: none; border: 1px solid var(--border); border-radius: 4px; }
.structure-canvas { position: relative; flex: 1; min-height: 0; overflow: hidden; cursor: grab; }
.structure-canvas:active { cursor: grabbing; }
.structure-canvas canvas { position: absolute; inset: 0; display: block; }
.structure-legend { position: absolute; top: 6px; left: 8px; right: 8px; z-index: 1; cursor: default; display: flex; flex-wrap: wrap; align-items: center; gap: 4px; pointer-events: none; }
.structure-legend button { pointer-events: auto; display: inline-flex; align-items: center; gap: 4px; padding: 2px 7px; font-size: 11px; color: var(--text); background: var(--panel); border: 1px solid var(--border); border-radius: 10px; }
.structure-legend button.off { color: var(--muted); opacity: 0.6; }
.structure-legend .spacer { flex: 1; }
.structure-dot { width: 9px; height: 9px; border-radius: 50%; }
.structure-axes { position: absolute; left: 6px; bottom: 4px; pointer-events: none; }
.structure-axes text { font-size: 11px; font-style: italic; font-weight: 600; text-anchor: middle; dominant-baseline: central; }
.structure-readout { flex: none; display: flex; gap: 12px; padding: 3px 10px; font-size: 11px; color: var(--muted); border-top: 1px solid var(--border); white-space: nowrap; font-variant-numeric: tabular-nums; }
.structure-readout-text { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; }
.structure-orientation { flex: none; color: var(--text); }
.structure-message { margin: auto; color: var(--muted); }
.run-info { padding: 10px 12px; overflow: auto; }
.model-form .cell-input { width: 48px; }
.model-form .model-repeats { display: inline-flex; gap: 3px; }
.model-form .model-zone { display: inline-flex; align-items: center; gap: 4px; }
.model-form .model-zone button { padding: 1px 6px; font-size: 11px; white-space: nowrap; }
.model-form .primary { margin-top: 6px; }
.model-form .model-warning, .engine-form .model-warning { color: var(--danger, #e5484d); }
.project-actions { display: flex; flex-direction: column; align-items: flex-start; gap: 8px; }
.ctf-view { flex: 1; display: flex; flex-direction: column; min-width: 0; min-height: 0; }
.ctf-bar { flex: none; display: flex; flex-wrap: wrap; align-items: center; gap: 12px; padding: 4px 8px; border-bottom: 1px solid var(--border); }
.ctf-legend { display: inline-flex; gap: 10px; color: var(--muted); }
.ctf-legend span { display: inline-flex; align-items: center; gap: 4px; }
.ctf-facts { margin-left: auto; display: inline-flex; flex-wrap: wrap; gap: 12px; font-variant-numeric: tabular-nums; }
.ctf-error { padding: 4px 8px; }
.ctf-loading { color: var(--muted); }
.ctf-view { height: 100%; }
.ctf-body { flex: 1; display: flex; min-height: 0; padding: 4px 8px 8px; }
.ctf-plot { position: relative; flex: 1; min-width: 0; min-height: 0; }
.ctf-plot-svg { display: block; }
.ctf-axis { stroke: var(--muted); }
.ctf-grid { stroke: var(--border); }
.ctf-marker { stroke: var(--muted); stroke-dasharray: 2 3; }
.ctf-tick { fill: var(--muted); font-size: 10px; font-variant-numeric: tabular-nums; }
.ctf-readout { position: absolute; top: 2px; left: 40px; right: 8px; color: var(--muted); font-variant-numeric: tabular-nums; pointer-events: none; }
.cross-section { display: flex; flex-direction: column; height: 100%; min-height: 0; }
.cross-section-bar { flex: none; display: flex; align-items: center; gap: 8px; padding: 4px 8px; }
.cross-section-swatch { padding-left: 6px; border-left: 3px solid; }
.cross-section-readout { margin-left: auto; color: var(--muted); font-variant-numeric: tabular-nums; }
.cross-section-plot { position: relative; flex: 1; min-height: 0; overflow: hidden; }
.cross-section-canvas { position: absolute; transform-origin: 0 0; image-rendering: pixelated; touch-action: none; }
.cross-section-zoom { color: var(--muted); font-variant-numeric: tabular-nums; }
.cross-section .model-warning { color: var(--danger, #e5484d); }
`;

const AXES = [
  { name: "a", color: "#e5484d" },
  { name: "b", color: "#30a46c" },
  { name: "c", color: "#3e8bff" },
];
const MODEL_AXIS_NAMES = ["x", "y", "z"];
const AXES_SIZE = 64;
const AXIS_LENGTH = 20;
const LABEL_AT = 27;

const num = (x, digits = 4) => String(+Number(x).toFixed(digits));
const cssColor = ([r, g, b]) => `rgb(${Math.round(r * 255)} ${Math.round(g * 255)} ${Math.round(b * 255)})`;

export function activate(host) {
  const { React } = host;
  const h = React.createElement;
  if (!document.getElementById("si-viewer-abtem-style")) {
    const style = document.createElement("style");
    style.id = "si-viewer-abtem-style";
    style.textContent = STYLE + Object.entries(BADGES)
      .map(([label, colour]) => `.badge.badge-label-${label} { background: ${colour}; }`)
      .join("\n");
    document.head.appendChild(style);
  }

  const loadStructure = async (fileId) => {
    const path = await host.pickPath({ kind: "file", title: "Load structure", ext: "cif", action: "Load" });
    if (!path) return;
    const { file, dataset_id } = await host.request(`${API}/projects/${fileId}/structures`, {
      method: "POST",
      body: JSON.stringify({ path }),
    });
    host.showFile(file, dataset_id);
  };

  function LoadStructure({ fileId }) {
    const [busy, setBusy] = React.useState(false);
    const [error, setError] = React.useState(null);
    const run = async () => {
      setBusy(true);
      setError(null);
      try {
        await loadStructure(fileId);
      } catch (e) {
        setError(e.message);
      } finally {
        setBusy(false);
      }
    };
    return h("div", null,
      h("button", { className: "primary", onClick: run, disabled: busy }, busy ? "Loading…" : "Load structure…"),
      error && h("p", { className: "error small" }, error));
  }

  /** The a, b, c directions of the view, in the corner. */
  function AxesGizmo({ lines, names }) {
    const c = AXES_SIZE / 2;
    return h("svg", { className: "structure-axes", width: AXES_SIZE, height: AXES_SIZE, "aria-hidden": true },
      AXES.map((axis, i) => h("g", { key: axis.name, ref: (el) => { lines.current[i] = el; } },
        h("line", { x1: c, y1: c, x2: c, y2: c, stroke: axis.color, strokeWidth: 2, strokeLinecap: "round" }),
        h("text", { x: c, y: c, fill: axis.color }, names ? names[i] : axis.name))));
  }

  const summaryOf = (s) => {
    const [a, b, c, alpha, beta, gamma] = s.cell;
    if (s.uvw) {
      return [
        s.formula,
        `along ${formatZone(s.uvw)}`,
        `${num(a, 2)} × ${num(b, 2)} × ${num(c, 2)} Å`,
        s.strain_percent > 0 && `strain ${num(s.strain_percent, 3)}%`,
      ].filter(Boolean).join(" · ");
    }
    return [
      s.formula,
      s.spacegroup && `${s.spacegroup.symbol} (${s.spacegroup.number})`,
      `a ${num(a)}, b ${num(b)}, c ${num(c)} Å`,
      `${num(alpha, 2)}, ${num(beta, 2)}, ${num(gamma, 2)}°`,
    ].filter(Boolean).join(" · ");
  };

  const detailsOf = (s) => [
    `${s.uvw ? "Atoms" : "Sites"}: ${s.atoms}`,
    `Per cell: ${Object.entries(s.composition).map(([el, n]) => `${el} ${num(n, 3)}`).join(", ")}`,
    s.mixed_sites > 0 && `Mixed sites: ${s.mixed_sites}`,
    `File: ${s.file}`,
  ].filter(Boolean).join("\n");

  const atomText = (data, atoms, i) => {
    const site = data.sites[atoms.site[i]];
    const occupancy = Object.entries(site).map(([el, x]) => `${el} ${num(x, 3)}`).join(", ");
    return [
      Object.keys(site).join("/"),
      `frac ${atoms.frac[i].map((x) => num(x)).join(", ")}`,
      `${atoms.xyz[i].map((x) => num(x, 3)).join(", ")} Å`,
      `occ ${occupancy}`,
    ].join(" · ");
  };

  function StructureView({ d }) {
    const settings = d.plugin.settings;
    const isModel = d.plugin.item === "model";
    const [data, setData] = React.useState(null);
    const [error, setError] = React.useState(null);
    const [hover, setHover] = React.useState(null);
    const [hidden, setHidden] = React.useState([]);
    const [showCell, setShowCell] = React.useState(true);
    // The nearest zone axis: its label, and its key while the view is on it.
    const [orientation, setOrientation] = React.useState("");
    const [onZone, setOnZone] = React.useState(null);
    const [zoneText, setZoneText] = React.useState("");
    const [zoneError, setZoneError] = React.useState(null);
    const [repeats, setRepeats] = React.useState([1, 1, 1]);
    const [repeatText, setRepeatText] = React.useState(["1", "1", "1"]);
    const [repeatError, setRepeatError] = React.useState(null);
    const [atomScale, setAtomScale] = React.useState(ATOM_SCALE.initial);
    const [showBonds, setShowBonds] = React.useState(false);
    const [tolerance, setTolerance] = React.useState(BOND_TOLERANCE.initial);
    const [toleranceText, setToleranceText] = React.useState(String(BOND_TOLERANCE.initial));
    const [sceneReady, setSceneReady] = React.useState(0);
    const rootRef = React.useRef(null);
    const zoneInputRef = React.useRef(null);
    const canvasRef = React.useRef(null);
    const sceneRef = React.useRef(null);
    const axisRefs = React.useRef([]);

    React.useEffect(() => {
      let stale = false;
      host.request(`${API}/structures/${d.id}/atoms`)
        .then((atoms) => { if (!stale) setData(atoms); })
        .catch((e) => { if (!stale) setError(e.message); });
      return () => { stale = true; };
    }, [d.id]);

    React.useEffect(() => {
      if (!data) return undefined;
      let stale = false;
      const c = AXES_SIZE / 2;
      const onAxes = (...axes) => axes.forEach((v, i) => {
        const g = axisRefs.current[i];
        if (!g) return;
        const [line, text] = g.children;
        line.setAttribute("x2", c + v.x * AXIS_LENGTH);
        line.setAttribute("y2", c - v.y * AXIS_LENGTH);
        text.setAttribute("x", c + v.x * LABEL_AT);
        text.setAttribute("y", c - v.y * LABEL_AT);
      });
      const nearest = zoneFinder(data.cell);
      const hexagonal = isHexagonal(settings.cell);
      const onView = (view) => {
        const { uvw, angle } = nearest(view);
        const on = angle < ON_AXIS;
        setOnZone(on ? uvw.join(",") : null);
        if (isModel) {
          const fromBeam = (Math.acos(view[2] / Math.hypot(...view)) * 180) / Math.PI;
          setOrientation(fromBeam < ON_AXIS ? "Beam" : `${fromBeam.toFixed(1)}° from beam`);
          return;
        }
        const name = formatZone(uvw) + (hexagonal ? ` = ${formatZone(toFourIndex(uvw))}` : "");
        setOrientation(on ? name : `≈ ${name} ${angle.toFixed(1)}°`);
        setViewZone(d.id, on ? uvw.join(" ") : null);
      };
      import("./scene.js")
        .then(({ createScene }) => {
          if (stale || !canvasRef.current) return;
          sceneRef.current = createScene(canvasRef.current, data.cell, { onHover: setHover, onAxes, onView, beamFirst: isModel });
          setSceneReady((n) => n + 1);
        })
        .catch((e) => { if (!stale) setError(e.message); });

      // Only while the view has focus: these keys mean other things elsewhere.
      const root = rootRef.current;
      const onKey = (e) => {
        if (e.target instanceof HTMLInputElement) {
          if (e.key === "Escape") {
            e.stopPropagation();
            root.focus();
          }
          return;
        }
        if (e.metaKey || e.ctrlKey || e.altKey) return;
        const scene = sceneRef.current;
        if (!scene) return;
        const step = e.shiftKey ? SHIFT_STEP : KEY_STEP;
        const look = (uvw) => scene.lookAlong(zoneVector(data.cell, uvw));
        const actions = {
          a: () => look([1, 0, 0]),
          b: () => look([0, 1, 0]),
          c: () => look([0, 0, 1]),
          r: () => scene.reset(),
          z: () => { zoneInputRef.current?.focus(); zoneInputRef.current?.select(); },
          q: () => scene.roll(-step),
          e: () => scene.roll(step),
          ArrowLeft: () => scene.tilt(-step, 0),
          ArrowRight: () => scene.tilt(step, 0),
          ArrowUp: () => scene.tilt(0, step),
          ArrowDown: () => scene.tilt(0, -step),
        };
        const action = actions[e.key.length === 1 ? e.key.toLowerCase() : e.key];
        if (!action) return;
        e.preventDefault();
        e.stopPropagation();
        action();
      };
      root?.addEventListener("keydown", onKey);
      return () => {
        stale = true;
        root?.removeEventListener("keydown", onKey);
        sceneRef.current?.dispose();
        sceneRef.current = null;
      };
    }, [data]);

    const atoms = React.useMemo(() => (data ? expandAtoms(data, repeats) : null), [data, repeats]);
    const bonds = React.useMemo(
      () => (atoms && showBonds ? findBonds(atoms, data, tolerance) : []),
      [atoms, data, showBonds, tolerance],
    );
    const drawnElements = React.useMemo(() => (data ? distinctColors(data) : null), [data]);
    React.useEffect(() => {
      if (!atoms || !sceneRef.current) return;
      setHover(null);
      sceneRef.current.setModel({
        atoms, bonds, sites: data.sites, elements: drawnElements, repeats, scale: atomScale, hidden,
      });
    }, [sceneReady, atoms, bonds, data, drawnElements, repeats, atomScale, hidden]);

    if (error) return h("p", { className: "structure-message" }, error);
    if (!data) return h("p", { className: "structure-message" }, "Loading…");

    const elements = [...new Set(data.sites.flatMap((site) => Object.keys(site)))];
    const toggle = (symbol) => {
      setHidden(hidden.includes(symbol) ? hidden.filter((x) => x !== symbol) : [...hidden, symbol]);
    };
    const changeRepeat = (axis, text) => {
      const nextText = repeatText.map((x, i) => (i === axis ? text : x));
      setRepeatText(nextText);
      const next = nextText.map(Number);
      if (!next.every((n) => Number.isInteger(n) && n >= 1 && n <= MAX_REPEAT)) {
        setRepeatError(`1 to ${MAX_REPEAT}`);
        return;
      }
      const count = countAtoms(data, next);
      if (count > MAX_ATOMS) {
        setRepeatError(`${count.toLocaleString()} atoms: limit ${MAX_ATOMS.toLocaleString()}`);
        return;
      }
      setRepeatError(null);
      setRepeats(next);
    };
    const changeTolerance = (text) => {
      setToleranceText(text);
      const value = Number(text);
      if (text.trim() && value >= BOND_TOLERANCE.min && value <= BOND_TOLERANCE.max) setTolerance(value);
    };
    const savePng = () => {
      const url = sceneRef.current?.toPng();
      if (!url) return;
      const link = document.createElement("a");
      link.href = url;
      link.download = `${d.title}${onZone ? `_${onZone.replaceAll(",", "")}` : ""}.png`;
      link.click();
    };
    const hexagonal = isHexagonal(settings.cell);
    const look = (uvw) => {
      setZoneError(null);
      sceneRef.current?.lookAlong(zoneVector(data.cell, uvw));
    };
    const applyZone = () => {
      if (!zoneText.trim()) return setZoneError(null);
      try {
        look(parseZone(zoneText, hexagonal));
        rootRef.current?.focus();
      } catch (e) {
        setZoneError(e.message);
      }
    };
    return h("div", { className: "structure-view", ref: rootRef, tabIndex: 0 },
      h("div", { className: "structure-zones" },
        (isModel ? MODEL_VIEWS : ZONES.map((uvw) => ({ uvw, label: formatZone(uvw) }))).map(({ uvw, label }) => h("button", {
          key: uvw.join(""),
          className: onZone === uvw.join(",") ? "active" : undefined,
          "aria-pressed": onZone === uvw.join(","),
          title: `Look down ${label}`,
          onClick: () => look(uvw),
        }, label)),
        !isModel && h("input", {
          ref: zoneInputRef,
          className: zoneError ? "invalid" : undefined,
          value: zoneText,
          placeholder: hexagonal ? "[uvw], [uvtw]" : "[uvw]",
          "aria-label": "Zone axis",
          title: "Zone axis (Z), Enter to go",
          spellCheck: false,
          onChange: (e) => { setZoneText(e.target.value); setZoneError(null); },
          onKeyDown: (e) => { if (e.key === "Enter") applyZone(); },
        }),
        h("button", { title: "Rotate the view 15° anticlockwise (Q: 1°, Shift: 5°)", "aria-label": "Rotate anticlockwise", onClick: () => sceneRef.current?.roll(-ROLL_BUTTON_STEP) }, "⟲"),
        h("button", { title: "Rotate the view 15° clockwise (E: 1°, Shift: 5°)", "aria-label": "Rotate clockwise", onClick: () => sceneRef.current?.roll(ROLL_BUTTON_STEP) }, "⟳"),
        zoneError && h("span", { className: "structure-zone-error", role: "alert" }, zoneError)),
      h("div", { className: "structure-options" },
        !isModel && h("span", { className: "field", title: "Unit cells along a, b and c" },
          "Repeat",
          ["a", "b", "c"].map((axis, i) => h("input", {
            key: axis,
            className: "cell-input",
            type: "number",
            min: 1,
            max: MAX_REPEAT,
            value: repeatText[i],
            "aria-label": `Repeats along ${axis}`,
            title: `Unit cells along ${axis}`,
            onChange: (e) => changeRepeat(i, e.target.value),
          }))),
        h("label", { className: "field", title: "Atom radius, as a fraction of the covalent radius" },
          "Size",
          h("input", {
            type: "range",
            min: ATOM_SCALE.min,
            max: ATOM_SCALE.max,
            step: ATOM_SCALE.step,
            value: atomScale,
            "aria-label": "Atom size",
            onChange: (e) => setAtomScale(Number(e.target.value)),
          }),
          num(atomScale, 2)),
        h("label", { className: "field", title: "Draw bonds" },
          h("input", { type: "checkbox", checked: showBonds, onChange: (e) => setShowBonds(e.target.checked) }),
          "Bonds"),
        showBonds && h("label", { className: "field", title: "Bonded when closer than this × the sum of the covalent radii" },
          "×",
          h("input", {
            className: "cell-input",
            type: "number",
            min: BOND_TOLERANCE.min,
            max: BOND_TOLERANCE.max,
            step: BOND_TOLERANCE.step,
            value: toleranceText,
            "aria-label": "Bond tolerance",
            onChange: (e) => changeTolerance(e.target.value),
          })),
        repeatError && h("span", { className: "structure-zone-error", role: "alert" }, repeatError),
        h("span", { className: "spacer" }),
        h("span", { className: "muted" }, `${atoms.xyz.length.toLocaleString()} atoms${showBonds ? ` · ${bonds.length.toLocaleString()} bonds` : ""}`),
        h("button", { title: "Save the view as PNG, with a transparent background", onClick: savePng }, "Save PNG")),
      h("div", { className: "structure-canvas", ref: canvasRef },
      h("div", { className: "structure-legend" },
        elements.map((symbol) => h("button", {
          key: symbol,
          className: hidden.includes(symbol) ? "off" : undefined,
          "aria-pressed": !hidden.includes(symbol),
          title: `Show or hide ${symbol}`,
          onClick: () => toggle(symbol),
        },
          h("span", { className: "structure-dot", style: { background: cssColor(drawnElements[symbol].color) } }),
          symbol)),
        h("button", {
          className: showCell ? undefined : "off",
          "aria-pressed": showCell,
          title: "Show or hide the unit cell",
          onClick: () => {
            sceneRef.current?.setCellVisible(!showCell);
            setShowCell(!showCell);
          },
        }, "Cell"),
        h("span", { className: "spacer" }),
        h("button", { title: "Back to the first view (R)", onClick: () => sceneRef.current?.reset() }, "Reset")),
      h(AxesGizmo, { lines: axisRefs, names: isModel ? MODEL_AXIS_NAMES : null })),
      h("div", { className: "structure-readout", title: detailsOf(settings) },
        h("span", { className: "structure-readout-text" }, hover === null || !atoms.site[hover] && atoms.site[hover] !== 0 ? summaryOf(settings) : atomText(data, atoms, hover)),
        h("span", { className: "structure-orientation" }, orientation)));
  }

  /** Settings of a model along a zone axis (*From view* copies the view's),
   * what it would be, and the button that makes it. */
  function CreateModel({ d }) {
    const hexagonal = isHexagonal(d.plugin.settings.cell);
    const viewZone = React.useSyncExternalStore(subscribeViewZone, () => viewZones.get(d.id) ?? null);
    const [zoneText, setZoneText] = React.useState("0 0 1");
    const [repeatText, setRepeatText] = React.useState(null);
    const [seed, setSeed] = React.useState("0");
    const [info, setInfo] = React.useState(null);
    const [error, setError] = React.useState(null);
    const [busy, setBusy] = React.useState(false);

    let uvw = null;
    let zoneProblem = null;
    try {
      uvw = parseZone(zoneText, hexagonal);
    } catch (e) {
      zoneProblem = e.message;
    }
    const repeats = repeatText?.map(Number) ?? null;
    const repeatsValid = !repeats || repeats.every((n) => Number.isInteger(n) && n >= 1 && n <= MODEL_MAX_REPEAT);
    const requestKey = JSON.stringify({ uvw, repeats });

    React.useEffect(() => {
      if (!uvw || !repeatsValid) {
        setInfo(null);
        return undefined;
      }
      let stale = false;
      const timer = setTimeout(() => {
        host.request(`${API}/structures/${d.id}/model/preview`, { method: "POST", body: requestKey })
          .then((p) => {
            if (stale) return;
            setInfo(p);
            setError(null);
            setRepeatText((text) => text ?? p.repeats.map(String));
          })
          .catch((e) => {
            if (stale) return;
            setInfo(null);
            setError(e.message);
          });
      }, PREVIEW_DELAY_MS);
      return () => {
        stale = true;
        clearTimeout(timer);
      };
    }, [d.id, requestKey]);

    const tooMany = info && info.atoms > info.max_atoms;
    const seedValue = Number(seed);
    const seedValid = Number.isInteger(seedValue) && seedValue >= 0;
    const canCreate = info && uvw && repeatsValid && !tooMany && !busy && (!info.mixed || seedValid);
    const create = async () => {
      setBusy(true);
      setError(null);
      try {
        const { file, dataset_id } = await host.request(`${API}/structures/${d.id}/models`, {
          method: "POST",
          body: JSON.stringify({ uvw, repeats: info.repeats, seed: info.mixed ? seedValue : 0 }),
        });
        host.showFile(file, dataset_id);
      } catch (e) {
        setError(e.message);
      } finally {
        setBusy(false);
      }
    };
    const row = (label, value, title) => [
      h("dt", { key: `${label}-t`, title }, label),
      h("dd", { key: `${label}-d` }, value),
    ];
    return h("div", { className: "model-form" },
      h("dl", { className: "props" },
        row("Zone axis", h("span", { className: "model-zone" },
          h("input", {
            className: "cell-input",
            style: { width: 92 },
            value: zoneText,
            "aria-label": "Model zone axis",
            spellCheck: false,
            onChange: (e) => { setZoneText(e.target.value); setRepeatText(null); },
          }),
          h("button", {
            className: "secondary",
            disabled: !viewZone,
            title: viewZone ? "Use the view's zone axis" : "Turn the view onto a zone axis first",
            onClick: () => setZoneText(viewZone),
          }, "From view")), "The beam direction"),
        row("Repeat", h("span", { className: "model-repeats" },
          MODEL_AXIS_NAMES.map((axis, i) => h("input", {
            key: axis,
            className: "cell-input",
            type: "number",
            min: 1,
            max: MODEL_MAX_REPEAT,
            value: repeatText?.[i] ?? "",
            "aria-label": `Model repeats along ${axis}`,
            title: axis,
            onChange: (e) => setRepeatText((text) => (text ?? ["1", "1", "1"]).map((x, k) => (k === i ? e.target.value : x))),
          }))), "Orthogonal cells along x, y and z"),
        info && row("Size", `${info.size.map((x) => num(x, 1)).join(" × ")} Å`),
        info && row("x, y, z", info.axes.map(formatZone).join(" "), "Lattice directions of the model's axes"),
        info && row("Atoms", info.atoms.toLocaleString()),
        info && info.strain_percent > 0 && row("Strain", `${num(info.strain_percent, 3)}%`),
        info?.mixed && row("Seed", h("input", {
          className: "cell-input",
          type: "number",
          min: 0,
          value: seed,
          "aria-label": "Random seed for mixed sites",
          onChange: (e) => setSeed(e.target.value),
        }), "Mixed sites are filled at random from this seed")),
      zoneProblem && h("p", { className: "model-warning small" }, zoneProblem),
      !repeatsValid && h("p", { className: "model-warning small" }, `Repeats: 1 to ${MODEL_MAX_REPEAT}`),
      tooMany && h("p", { className: "model-warning small" }, `Limit: ${info.max_atoms.toLocaleString()} atoms`),
      error && h("p", { className: "model-warning small" }, error),
      h("button", { className: "primary", disabled: !canCreate, onClick: create }, busy ? "Creating…" : "Create model"));
  }

  /** Whether abTEM is installed, and installing it (a job that can be
   * cancelled, picked up again after a reload). */
  function EngineSection() {
    const [engine, setEngine] = React.useState(null);
    const [job, setJob] = React.useState(null);
    const [error, setError] = React.useState(null);
    const active = job?.status === "queued" || job?.status === "running";

    const refresh = React.useCallback(async () => {
      try {
        const e = await host.request(`${API}/engine`);
        setEngine(e);
        if (e.job) setJob(e.job);
      } catch (e) {
        setError(e.message);
      }
    }, []);
    React.useEffect(() => { refresh(); }, [refresh]);

    const jobId = active ? job.id : null;
    React.useEffect(() => {
      if (!jobId) return undefined;
      let stale = false;
      const timer = setInterval(async () => {
        try {
          const next = await host.request(`/api/jobs/${jobId}`);
          if (stale) return;
          setJob(next);
          if (next.status === "failed") setError(next.error);
          if (next.status !== "queued" && next.status !== "running") refresh();
        } catch {
          /* the backend is busy: try again */
        }
      }, ENGINE_POLL_MS);
      return () => {
        stale = true;
        clearInterval(timer);
      };
    }, [jobId, refresh]);

    const install = async () => {
      setError(null);
      try {
        setJob(await host.request(`${API}/engine/install`, { method: "POST" }));
      } catch (e) {
        setError(e.message);
      }
    };
    if (!engine) return h("p", { className: "muted small" }, error ?? "Checking abTEM…");
    const row = (label, value, title) => [
      h("dt", { key: `${label}-t`, title }, label),
      h("dd", { key: `${label}-d`, title }, value),
    ];
    return h("div", { className: "engine-form" },
      h("dl", { className: "props" },
        row("Status", active ? "Installing" : engine.installed ? "Ready" : engine.error ? "Not working" : "Not installed"),
        row("abTEM", engine.versions?.abtem ?? engine.version),
        engine.versions && row("Python", engine.versions.python),
        row("Location", engine.path.replace(/^.*\/(\.si-viewer\/)/, "~/$1"), engine.path),
        active && row("Progress", `${Math.round(job.progress * 100)}%${job.message ? ` · ${job.message}` : ""}`)),
      (error || engine.error) && h("p", { className: "model-warning small" }, error ?? engine.error),
      active
        ? h("button", { className: "secondary", onClick: () => host.cancelJob(job.id) }, "Cancel")
        : !engine.installed && h("button", {
          className: "primary",
          title: `Downloads abTEM ${engine.version} and its dependencies (about ${ENGINE_SIZE})`,
          onClick: install,
        }, engine.error ? "Reinstall abTEM" : `Install abTEM (${ENGINE_SIZE})`));
  }

  /** A model's potential: its settings, the grid they give, and computing the
   * projected potential as an image under the model. */
  function PotentialSection({ d }) {
    const [engine, setEngine] = React.useState(null);
    const [saved, setSaved] = React.useState(null);
    const [sampling, setSampling] = React.useState(String(POTENTIAL.sampling));
    const [slice, setSlice] = React.useState(String(POTENTIAL.slice));
    const [parametrization, setParametrization] = React.useState("lobato");
    const [projection, setProjection] = React.useState("infinite");
    const [phonons, setPhonons] = React.useState("0");
    const [sigma, setSigma] = React.useState(String(POTENTIAL.sigma));
    const [seed, setSeed] = React.useState("0");
    const [job, setJob] = React.useState(null);
    const [error, setError] = React.useState(null);
    const mounted = React.useRef(true);
    const active = job?.status === "queued" || job?.status === "running";

    const finish = React.useCallback((done) => {
      if (!mounted.current) return;
      setJob(done);
      if (done.status === "failed") setError(done.error);
      if (done.status === "done" && done.result?.file) host.showFile(done.result.file, done.result.dataset_id);
    }, []);

    React.useEffect(() => {
      mounted.current = true;
      let stale = false;
      host.request(`${API}/engine`).then((e) => { if (!stale) setEngine(e); }).catch((e) => { if (!stale) setError(e.message); });
      host.request(`${API}/models/${d.id}/potential`)
        .then((p) => {
          if (stale) return;
          setSaved(p);
          const last = p.results.at(-1)?.settings;
          if (last) {
            setSampling(String(last.sampling));
            setSlice(String(last.slice_thickness));
            setParametrization(last.parametrization);
            setProjection(last.projection);
            setPhonons(String(last.phonon_configs ?? 0));
            setSigma(String(last.phonon_sigma ?? POTENTIAL.sigma));
            setSeed(String(last.phonon_seed ?? 0));
          }
          if (p.job) host.followJob(p.job, (j) => { if (mounted.current) setJob(j); }).then(finish).catch((e) => setError(e.message));
        })
        .catch((e) => { if (!stale) setError(e.message); });
      return () => {
        stale = true;
        mounted.current = false;
      };
    }, [d.id, finish]);

    const s = Number(sampling);
    const t = Number(slice);
    const configs = Number(phonons);
    const sig = Number(sigma);
    const sd = Number(seed);
    const phononsOn = configs >= 2;
    const phononsOk = Number.isInteger(configs) && configs >= 0 && configs <= POTENTIAL.maxConfigs
      && (!phononsOn || (sig > 0 && sig <= POTENTIAL.maxSigma && Number.isInteger(sd) && sd >= 0));
    const valid = s >= POTENTIAL.minSampling && s <= POTENTIAL.maxSampling && t >= POTENTIAL.minSlice && t <= POTENTIAL.maxSlice;
    const [lx, ly, lz] = d.plugin.settings.cell;
    const nx = valid ? Math.ceil(lx / s) : null;
    const ny = valid ? Math.ceil(ly / s) : null;
    const slices = valid ? Math.ceil(lz / t) : null;
    const maxGpts = saved?.max_gpts ?? 4096;
    const tooLarge = valid && Math.max(nx, ny) > maxGpts;
    const tooMuchMemory = valid && nx * ny * slices * 4 > POTENTIAL.maxBytes;
    const compute = async () => {
      setError(null);
      try {
        finish(await host.runJob(`${API}/models/${d.id}/potential`, {
          sampling: s, slice_thickness: t, parametrization, projection,
          phonon_configs: phononsOn ? configs : 0,
          ...(phononsOn ? { phonon_sigma: sig, phonon_seed: sd } : {}),
        }, (j) => { if (mounted.current) setJob(j); }));
      } catch (e) {
        if (mounted.current) setError(e.message);
      }
    };
    const row = (label, value, title) => [
      h("dt", { key: `${label}-t`, title }, label),
      h("dd", { key: `${label}-d` }, value),
    ];
    const number = (label, value, set, min, max, step) => h("input", {
      className: "cell-input",
      type: "number",
      min,
      max,
      step,
      value,
      disabled: active,
      "aria-label": label,
      onChange: (e) => set(e.target.value),
    });
    const select = (label, value, set, options) => h("select", {
      className: "cell-input",
      style: { width: 96 },
      value,
      disabled: active,
      "aria-label": label,
      onChange: (e) => set(e.target.value),
    }, options.map(([id, name]) => h("option", { key: id, value: id }, name)));
    return h("div", { className: "engine-form model-form" },
      h("dl", { className: "props" },
        row("Sampling (Å)", number("Potential sampling", sampling, setSampling, POTENTIAL.minSampling, POTENTIAL.maxSampling, 0.01), "Pixel size of the potential"),
        row("Slice (Å)", number("Slice thickness", slice, setSlice, POTENTIAL.minSlice, POTENTIAL.maxSlice, 0.1), "Thickness of each multislice slice"),
        row("Parametrisation", select("Parametrisation", parametrization, setParametrization, [["lobato", "Lobato"], ["kirkland", "Kirkland"]]), "Parametrisation of the atomic potentials"),
        row("Projection", select("Projection", projection, setProjection, [["infinite", "Infinite"], ["finite", "Finite"]]), "Infinite: each atom in one slice. Finite: integrated across slices"),
        row("Frozen phonons", number("Frozen phonon configurations", phonons, setPhonons, 0, POTENTIAL.maxConfigs, 1), "Configurations of randomly displaced atoms; 0 or 1: off. The potential shown is their mean; simulations average their intensities"),
        phononsOn && row("σ (Å)", number("Phonon displacement σ", sigma, setSigma, 0.01, POTENTIAL.maxSigma, 0.01), "Standard deviation of each atom's displacement along each direction"),
        phononsOn && row("Seed", number("Phonon seed", seed, setSeed, 0, 2147483647, 1), "Random seed of the displacements"),
        valid && row("Grid", `${nx} × ${ny}`),
        valid && row("Slices", slices),
        active && row("Progress", `${Math.round((job.progress ?? 0) * 100)}%${job.message ? ` · ${job.message}` : ""}`)),
      !valid && h("p", { className: "model-warning small" }, `Sampling ${POTENTIAL.minSampling}–${POTENTIAL.maxSampling} Å, slice ${POTENTIAL.minSlice}–${POTENTIAL.maxSlice} Å`),
      !phononsOk && h("p", { className: "model-warning small" }, `Phonons: 0–${POTENTIAL.maxConfigs} configurations, σ up to ${POTENTIAL.maxSigma} Å`),
      tooLarge && h("p", { className: "model-warning small" }, `Limit: ${maxGpts} pixels per side`),
      !tooLarge && tooMuchMemory && h("p", { className: "model-warning small" }, "Limit: 2 GB of slices"),
      engine && !engine.installed && h("p", { className: "model-warning small" }, "Install abTEM first"),
      error && h("p", { className: "model-warning small" }, error),
      active
        ? h("button", { className: "secondary", disabled: !job.id, onClick: () => host.cancelJob(job.id) }, "Cancel")
        : h("button", {
          className: "primary",
          disabled: !valid || !phononsOk || tooLarge || tooMuchMemory || !engine?.installed,
          onClick: compute,
        }, "Build potential"));
  }

  /** The project's CTFs, for choosing one (refetched when ``version`` changes). */
  function useProjectCtfs(fileId, version) {
    const [ctfs, setCtfs] = React.useState(null);
    React.useEffect(() => {
      let stale = false;
      host.request(`${API}/projects/${fileId}/ctfs`).then((c) => { if (!stale) setCtfs(c); }).catch(() => { if (!stale) setCtfs([]); });
      return () => { stale = true; };
    }, [fileId, version]);
    return ctfs;
  }

  const ctfLabel = (c) => `${c.title} · Δf ${num(c.settings.defocus, 1)} Å`;

  /** A CTF and how its focal spread is treated: the choice shared by HRTEM
   * runs and Apply CTF. ``energy``: only CTFs at it can be chosen. */
  function CtfChoice({ ctfs, ctfId, setCtfId, coherence, setCoherence, energy, disabled, allowNone }) {
    const row = (label, value, title) => [
      h("dt", { key: `${label}-t`, title }, label),
      h("dd", { key: `${label}-d` }, value),
    ];
    const chosen = ctfs?.find((c) => c.id === ctfId);
    return [
      row("CTF", h("select", {
        className: "cell-input", style: { width: 170 }, value: ctfId ?? "", disabled, "aria-label": "CTF",
        onChange: (e) => setCtfId(e.target.value || null),
      },
      allowNone && h("option", { value: "" }, "None (exit wave only)"),
      (ctfs ?? []).map((c) => {
        const other = energy !== undefined && Math.abs(c.settings.energy_kv - energy) > 1e-6;
        return h("option", { key: c.id, value: c.id, disabled: other }, other ? `${ctfLabel(c)} (${c.settings.energy_kv} kV)` : ctfLabel(c));
      })), "The objective lens that forms the image"),
      chosen?.settings.focal_spread > 0 && row("Focal spread", h("select", {
        className: "cell-input", style: { width: 170 }, value: coherence, disabled, "aria-label": "Focal spread",
        onChange: (e) => setCoherence(e.target.value),
      },
      h("option", { value: "envelope" }, "Envelope"),
      h("option", { value: "focal-series" }, "Focal series (7)")),
      "Envelope: abTEM's quasi-coherent damping. Focal series: images averaged over 7 defocus values within ±1.5σ (σ = spread/√2)"),
    ];
  }

  /** An HRTEM run of a Potential: a plane wave through it, and the image a
   * CTF of the project forms of its exit wave. */
  function HrtemSection({ d }) {
    const [engine, setEngine] = React.useState(null);
    const ctfs = useProjectCtfs(d.file_id, d.id);
    const [ctfId, setCtfId] = React.useState(undefined);
    const [coherence, setCoherence] = React.useState("envelope");
    const [energy, setEnergy] = React.useState(String(HRTEM.energy));
    const [job, setJob] = React.useState(null);
    const [error, setError] = React.useState(null);
    const mounted = React.useRef(true);
    const active = job?.status === "queued" || job?.status === "running";

    React.useEffect(() => {
      if (ctfId === undefined && ctfs) setCtfId(ctfs[0]?.id ?? null);
    }, [ctfs, ctfId]);

    const finish = React.useCallback((done) => {
      if (!mounted.current) return;
      setJob(done);
      if (done.status === "failed") setError(done.error);
      if (done.status === "done" && done.result?.file) host.showFile(done.result.file, done.result.dataset_id);
    }, []);

    React.useEffect(() => {
      mounted.current = true;
      let stale = false;
      host.request(`${API}/engine`).then((e) => { if (!stale) setEngine(e); }).catch((e) => { if (!stale) setError(e.message); });
      host.request(`${API}/potentials/${d.id}/hrtem`)
        .then((r) => {
          if (!stale && r.job) host.followJob(r.job, (j) => { if (mounted.current) setJob(j); }).then(finish).catch((e) => setError(e.message));
        })
        .catch((e) => { if (!stale) setError(e.message); });
      return () => {
        stale = true;
        mounted.current = false;
      };
    }, [d.id, finish]);

    const potential = d.plugin.settings.potential;
    const chosen = ctfs?.find((c) => c.id === ctfId);
    const kv = chosen ? chosen.settings.energy_kv : Number(energy);
    const energyOk = kv >= HRTEM.minEnergy && kv <= HRTEM.maxEnergy;
    const limit = energyOk ? (wavelength(kv) / (2 * potential.sampling)) * ANTIALIAS * 1000 : null;
    const apertureOk = !chosen || chosen.settings.semiangle === null || (limit !== null && chosen.settings.semiangle <= limit);
    const valid = energyOk && apertureOk && ctfId !== undefined;
    const run = async () => {
      setError(null);
      try {
        finish(await host.runJob(`${API}/potentials/${d.id}/hrtem`, {
          energy_kv: kv, ctf: ctfId ?? null, coherence,
        }, (j) => { if (mounted.current) setJob(j); }));
      } catch (e) {
        if (mounted.current) setError(e.message);
      }
    };
    const row = (label, value, title) => [
      h("dt", { key: `${label}-t`, title }, label),
      h("dd", { key: `${label}-d` }, value),
    ];
    return h("div", { className: "engine-form model-form" },
      h("dl", { className: "props" },
        CtfChoice({ ctfs, ctfId, setCtfId, coherence, setCoherence, disabled: active, allowNone: true }),
        chosen
          ? row("Energy", `${kv} kV`, "The CTF's")
          : row("Energy (kV)", h("input", {
            className: "cell-input", type: "number", value: energy, disabled: active, "aria-label": "HRTEM energy",
            min: HRTEM.minEnergy, max: HRTEM.maxEnergy, onChange: (e) => setEnergy(e.target.value),
          })),
        row("Potential", potentialLabel(potential), "The potential this run is made from"),
        active && row("Progress", `${Math.round((job.progress ?? 0) * 100)}%${job.message ? ` · ${job.message}` : ""}`)),
      !energyOk && h("p", { className: "model-warning small" }, `Energy: ${HRTEM.minEnergy}–${HRTEM.maxEnergy} kV`),
      energyOk && !apertureOk && h("p", { className: "model-warning small" }, `The CTF's aperture: up to ${limit.toFixed(0)} mrad at this sampling`),
      ctfs && !ctfs.length && h("p", { className: "small", style: { color: "var(--muted)" } }, "No CTF in the project: the run keeps the exit wave only"),
      engine && !engine.installed && h("p", { className: "model-warning small" }, "Install abTEM first"),
      error && h("p", { className: "model-warning small" }, error),
      active
        ? h("button", { className: "secondary", disabled: !job.id, onClick: () => host.cancelJob(job.id) }, "Cancel")
        : h("button", { className: "primary", disabled: !valid || !engine?.installed, onClick: run }, "Run HRTEM"));
  }

  /** A CTF applied to a finished HRTEM run's exit wave: a new image under the run. */
  function ApplyCtf({ d }) {
    const s = d.plugin.settings;
    const ctfs = useProjectCtfs(d.file_id, d.id);
    const usable = (ctfs ?? []).filter((c) => Math.abs(c.settings.energy_kv - s.energy_kv) < 1e-6);
    const [ctfId, setCtfId] = React.useState(undefined);
    const [coherence, setCoherence] = React.useState("envelope");
    const [busy, setBusy] = React.useState(false);
    const [error, setError] = React.useState(null);
    React.useEffect(() => {
      if (ctfId === undefined && ctfs) setCtfId(usable[0]?.id ?? null);
    }, [ctfs]); // eslint-disable-line react-hooks/exhaustive-deps
    const apply = async () => {
      setBusy(true);
      setError(null);
      try {
        const { file, dataset_id } = await host.request(`${API}/runs/${d.id}/images`, {
          method: "POST", body: JSON.stringify({ ctf: ctfId, coherence }),
        });
        host.showFile(file, dataset_id);
      } catch (e) {
        setError(e.message);
      } finally {
        setBusy(false);
      }
    };
    return h("div", { className: "engine-form model-form" },
      h("dl", { className: "props" },
        CtfChoice({ ctfs, ctfId, setCtfId, coherence, setCoherence, energy: s.energy_kv, disabled: busy, allowNone: false })),
      ctfs && !usable.length && h("p", { className: "model-warning small" }, `No CTF at ${s.energy_kv} kV in the project`),
      error && h("p", { className: "model-warning small" }, error),
      h("button", { className: "primary", disabled: busy || !ctfId, onClick: apply }, busy ? "Applying…" : "Apply CTF"));
  }

  /** A plane wave or probe sent through a Potential, kept at every depth. */
  function WaveSection({ d }) {
    const s = d.plugin.settings;
    const [cellX, cellY] = s.gpts.map((n, i) => n * s.sampling_xy[i]);
    const probeTile = [Math.ceil(WAVE.probeExtent / cellX), Math.ceil(WAVE.probeExtent / cellY)];
    const [engine, setEngine] = React.useState(null);
    const [beam, setBeam] = React.useState("plane");
    const [energy, setEnergy] = React.useState(String(HRTEM.energy));
    const [semiangle, setSemiangle] = React.useState(String(WAVE.semiangle));
    const [defocus, setDefocus] = React.useState("0");
    const [cs, setCs] = React.useState("0");
    const [tileX, setTileX] = React.useState("1");
    const [tileY, setTileY] = React.useState("1");
    const [posX, setPosX] = React.useState("");
    const [posY, setPosY] = React.useState("");
    const [job, setJob] = React.useState(null);
    const [error, setError] = React.useState(null);
    const mounted = React.useRef(true);
    const active = job?.status === "queued" || job?.status === "running";

    const finish = React.useCallback((done) => {
      if (!mounted.current) return;
      setJob(done);
      if (done.status === "failed") setError(done.error);
      if (done.status === "done" && done.result?.file) host.showFile(done.result.file, done.result.dataset_id);
    }, []);

    React.useEffect(() => {
      mounted.current = true;
      let stale = false;
      host.request(`${API}/engine`).then((e) => { if (!stale) setEngine(e); }).catch((e) => { if (!stale) setError(e.message); });
      host.request(`${API}/potentials/${d.id}/wavefunction`)
        .then((r) => {
          if (!stale && r.job) host.followJob(r.job, (j) => { if (mounted.current) setJob(j); }).then(finish).catch((e) => setError(e.message));
        })
        .catch((e) => { if (!stale) setError(e.message); });
      return () => {
        stale = true;
        mounted.current = false;
      };
    }, [d.id, finish]);

    const chooseBeam = (next) => {
      setBeam(next);
      const [x, y] = next === "probe" ? probeTile : [1, 1];
      setTileX(String(x));
      setTileY(String(y));
      setPosX("");
      setPosY("");
    };
    const probe = beam === "probe";
    const kv = Number(energy);
    const alpha = Number(semiangle);
    const df = Number(defocus);
    const csUm = Number(cs);
    const tx = Number(tileX);
    const ty = Number(tileY);
    const tileOk = [tx, ty].every((n) => Number.isInteger(n) && n >= 1 && n <= WAVE.maxTile);
    const energyOk = kv >= HRTEM.minEnergy && kv <= HRTEM.maxEnergy;
    const limit = energyOk ? (wavelength(kv) / (2 * s.potential.sampling)) * ANTIALIAS * 1000 : null;
    const alphaOk = !probe || (alpha > 0 && limit !== null && alpha <= limit);
    const lensOk = !probe || (defocus.trim() !== "" && Number.isFinite(df) && cs.trim() !== "" && Number.isFinite(csUm));
    const extentX = cellX * tx;
    const extentY = cellY * ty;
    const placed = posX.trim() !== "" || posY.trim() !== "";
    const px = Number(posX);
    const py = Number(posY);
    const positionOk = !probe || !placed || (posX.trim() !== "" && posY.trim() !== "" && px >= 0 && px <= extentX && py >= 0 && py <= extentY);
    const nx = s.gpts[0] * tx;
    const ny = s.gpts[1] * ty;
    const planes = s.slices + 1;
    const tooLarge = tileOk && Math.max(nx, ny) > 4096;
    const phononConfigs = s.potential.phonon_configs || 1; // a wave per frozen phonon configuration
    const tooMuchMemory = tileOk && nx * ny * planes * 8 * phononConfigs > POTENTIAL.maxBytes;
    const small = probe && tileOk && Math.min(extentX, extentY) < WAVE.probeExtent;
    const valid = energyOk && alphaOk && lensOk && tileOk && positionOk && !tooLarge && !tooMuchMemory;
    const run = async () => {
      setError(null);
      try {
        finish(await host.runJob(`${API}/potentials/${d.id}/wavefunction`, {
          beam,
          energy_kv: kv,
          semiangle: probe ? alpha : WAVE.semiangle,
          defocus: probe ? df : 0,
          cs_um: probe ? csUm : 0,
          tile: [tx, ty],
          position: probe && placed ? [px, py] : null,
        }, (j) => { if (mounted.current) setJob(j); }));
      } catch (e) {
        if (mounted.current) setError(e.message);
      }
    };
    const row = (label, value, title) => [
      h("dt", { key: `${label}-t`, title }, label),
      h("dd", { key: `${label}-d` }, value),
    ];
    const number = (label, value, set, extra = {}) => h("input", {
      className: "cell-input",
      type: "number",
      value,
      disabled: active,
      "aria-label": label,
      onChange: (e) => set(e.target.value),
      ...extra,
    });
    return h("div", { className: "engine-form model-form" },
      h("dl", { className: "props" },
        row("Beam", h("select", {
          className: "cell-input",
          style: { width: 96 },
          value: beam,
          disabled: active,
          "aria-label": "Beam",
          onChange: (e) => chooseBeam(e.target.value),
        }, h("option", { value: "plane" }, "Plane wave"), h("option", { value: "probe" }, "Probe"))),
        row("Energy (kV)", number("Wave energy", energy, setEnergy, { min: HRTEM.minEnergy, max: HRTEM.maxEnergy })),
        probe && row("Convergence (mrad)", number("Probe convergence", semiangle, setSemiangle, { min: 0, max: limit ? Math.floor(limit) : undefined }), limit ? `Up to ${limit.toFixed(0)} mrad at this sampling` : undefined),
        probe && row("Defocus (Å)", number("Probe defocus", defocus, setDefocus, { step: 1 }), "Positive focuses into the specimen"),
        probe && row("Cs (µm)", number("Probe Cs", cs, setCs)),
        row("Tiling", h("span", { className: "model-zone" },
          number("Tiling x", tileX, setTileX, { min: 1, max: WAVE.maxTile, step: 1 }),
          number("Tiling y", tileY, setTileY, { min: 1, max: WAVE.maxTile, step: 1 })), "Repeats of the potential along x and y"),
        probe && row("Position (Å)", h("span", { className: "model-zone" },
          number("Probe x", posX, setPosX, { min: 0, max: extentX, step: 0.1, placeholder: "Centre" }),
          number("Probe y", posY, setPosY, { min: 0, max: extentY, step: 0.1, placeholder: "Centre" })), "Measured from the lower left of the tiled cell"),
        tileOk && row("Cell", `${extentX.toFixed(1)} × ${extentY.toFixed(1)} Å`),
        tileOk && row("Grid", `${nx} × ${ny} · ${planes} planes`),
        row("Potential", potentialLabel(s.potential)),
        active && row("Progress", `${Math.round((job.progress ?? 0) * 100)}%${job.message ? ` · ${job.message}` : ""}`)),
      !energyOk && h("p", { className: "model-warning small" }, `Energy: ${HRTEM.minEnergy}–${HRTEM.maxEnergy} kV`),
      energyOk && !alphaOk && h("p", { className: "model-warning small" }, `Convergence: up to ${limit.toFixed(0)} mrad`),
      !tileOk && h("p", { className: "model-warning small" }, `Tiling: 1–${WAVE.maxTile}`),
      !positionOk && h("p", { className: "model-warning small" }, `Position: within ${extentX.toFixed(1)} × ${extentY.toFixed(1)} Å`),
      tooLarge && h("p", { className: "model-warning small" }, "Limit: 4096 pixels per side"),
      !tooLarge && tooMuchMemory && h("p", { className: "model-warning small" }, "Limit: 2 GB of planes"),
      small && h("p", { className: "model-warning small" }, "Small cell: the probe can wrap into itself"),
      engine && !engine.installed && h("p", { className: "model-warning small" }, "Install abTEM first"),
      error && h("p", { className: "model-warning small" }, error),
      active
        ? h("button", { className: "secondary", disabled: !job.id, onClick: () => host.cancelJob(job.id) }, "Cancel")
        : h("button", { className: "primary", disabled: !valid || !engine?.installed, onClick: run }, "Propagate"));
  }

  /** A stack cut through its depth along a profile line on its image. While
   * the tab is shown, the image has an x–z and a y–z line (through its centre
   * at first; they move only across and can't be deleted); the section follows
   * the selected line, or x–z. Depth runs down, from where the beam enters.
   * The wheel zooms, dragging pans, a double-click resets. */
  function CrossSection({ d }) {
    const { lines, selected } = host.useLines(d.id);
    const colormap = host.useColormap(d);
    const visible = lines.filter((l) => l.visible);
    const line = visible.find((l) => l.id === selected) ?? visible.find((l) => l.fixed === "x");
    const [section, setSection] = React.useState(null);
    const [error, setError] = React.useState(null);
    const [hover, setHover] = React.useState(null);
    const [box, setBox] = React.useState({ w: 0, h: 0 });
    const [fill, setFill] = React.useState(false);
    const [zoom, setZoom] = React.useState(NO_ZOOM);
    const canvas = React.useRef(null);
    const holder = React.useRef(null);
    const pan = React.useRef(null);
    // The image's columns and rows (signal_shape is HyperSpy's: x first).
    const [width, height] = d.signal_shape;
    const phase = d.plugin?.settings?.result === "wavefunction-phase";

    // The two axis lines, shown while the tab is.
    React.useEffect(() => {
      const before = host.getLines(d.id);
      const ids = {};
      for (const axis of ["x", "y"]) {
        const found = before.lines.find((l) => l.fixed === axis);
        if (found) {
          host.updateLine(d.id, found.id, { visible: true });
          ids[axis] = found.id;
        } else {
          const geometry = axis === "x"
            ? { x0: 0, y0: height / 2, x1: width, y1: height / 2 }
            : { x0: width / 2, y0: 0, x1: width / 2, y1: height };
          ids[axis] = host.addLine(d.id, geometry, { fixed: axis, name: `${axis}–z` });
        }
      }
      const kept = before.lines.find((l) => l.id === before.selected && l.visible);
      host.selectLine(d.id, kept ? kept.id : ids.x);
      return () => {
        for (const id of Object.values(ids)) host.updateLine(d.id, id, { visible: false });
      };
    }, [d.id]); // eslint-disable-line react-hooks/exhaustive-deps

    React.useEffect(() => setZoom(NO_ZOOM), [d.id, fill]);

    const key = line ? [line.x0, line.y0, line.x1, line.y1, line.width].map((v) => v.toFixed(2)).join(",") : null;
    React.useEffect(() => {
      if (!line) {
        setSection(null);
        return;
      }
      let stale = false;
      const timer = setTimeout(() => {
        host.request(`${API}/stacks/${d.id}/section`, {
          method: "POST",
          body: JSON.stringify({ x0: line.x0, y0: line.y0, x1: line.x1, y1: line.y1, width: line.width }),
        })
          .then((r) => {
            if (stale) return;
            const bytes = Uint8Array.from(atob(r.data), (c) => c.charCodeAt(0));
            setSection({ ...r, values: new Float32Array(bytes.buffer) });
            setError(null);
          })
          .catch((e) => { if (!stale) setError(e.message); });
      }, SECTION_DELAY_MS);
      return () => {
        stale = true;
        clearTimeout(timer);
      };
    }, [d.id, d.filter_rev, key]); // eslint-disable-line react-hooks/exhaustive-deps

    React.useEffect(() => {
      const el = holder.current;
      if (!el) return undefined;
      const observer = new ResizeObserver(([e]) => setBox({ w: e.contentRect.width, h: e.contentRect.height }));
      observer.observe(el);
      return () => observer.disconnect();
    }, []);

    // Contrast: the 0.5–99.5 percentiles; a phase over its whole circle.
    const range = React.useMemo(() => {
      if (!section) return null;
      if (phase) return [-Math.PI, Math.PI];
      const sorted = Float32Array.from(section.values).sort();
      const at = (q) => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
      const lo = at(0.005);
      const hi = at(0.995);
      return hi > lo ? [lo, hi] : [lo, lo + 1];
    }, [section, phase]);

    React.useEffect(() => {
      const c = canvas.current;
      if (!c || !section || !range) return;
      const [planes, n] = section.shape;
      c.width = n;
      c.height = planes;
      const ctx = c.getContext("2d");
      const image = ctx.createImageData(n, planes);
      const lut = host.colormapLut(phase ? "phase" : colormap);
      const [lo, hi] = range;
      for (let i = 0; i < section.values.length; i++) {
        const v = Math.max(0, Math.min(255, Math.round(((section.values[i] - lo) / (hi - lo)) * 255)));
        image.data.set(lut.subarray(v * 4, v * 4 + 4), i * 4);
      }
      ctx.putImageData(image, 0, 0);
    }, [section, range, colormap, phase]);

    const centre = () => {
      for (const l of lines) {
        if (l.fixed === "x") host.updateLine(d.id, l.id, { y0: height / 2, y1: height / 2 });
        if (l.fixed === "y") host.updateLine(d.id, l.id, { x0: width / 2, x1: width / 2 });
      }
    };
    const choose = (axis) => {
      const fixed = lines.find((l) => l.fixed === axis);
      if (fixed) host.selectLine(d.id, fixed.id);
    };
    // Shown at its true aspect, as large as the pane allows.
    let shown = null;
    if (section) {
      const [planes, n] = section.shape;
      const across = n * section.step;
      const deep = planes * section.z_step;
      const roomW = box.w - SECTION_MARGIN;
      const roomH = box.h - SECTION_MARGIN;
      const scale = Math.min(roomW / across, roomH / deep);
      if (scale > 0) {
        const size = fill ? { w: roomW, h: roomH } : { w: across * scale, h: deep * scale };
        shown = { ...size, left: (box.w - size.w) / 2, top: (box.h - size.h) / 2, planes, n };
      }
    }
    const shownRef = React.useRef(null);
    shownRef.current = shown;

    // The wheel zooms about the pointer (it must not scroll the page).
    React.useEffect(() => {
      const el = holder.current;
      if (!el) return undefined;
      const onWheel = (e) => {
        const s = shownRef.current;
        if (!s) return;
        e.preventDefault();
        const r = el.getBoundingClientRect();
        const mx = e.clientX - r.left - s.left;
        const my = e.clientY - r.top - s.top;
        setZoom((z) => {
          const k = Math.min(ZOOM.max, Math.max(1, z.k * Math.exp(-e.deltaY * ZOOM.speed)));
          if (k === 1) return NO_ZOOM;
          return { k, x: mx - ((mx - z.x) / z.k) * k, y: my - ((my - z.y) / z.k) * k };
        });
      };
      el.addEventListener("wheel", onWheel, { passive: false });
      return () => el.removeEventListener("wheel", onWheel);
    }, []);
    const onDown = (e) => {
      if (e.button !== 0) return;
      e.currentTarget.setPointerCapture(e.pointerId);
      pan.current = { x: e.clientX - zoom.x, y: e.clientY - zoom.y };
    };
    const onDrag = (e) => {
      if (pan.current && zoom.k > 1) setZoom((z) => ({ ...z, x: e.clientX - pan.current.x, y: e.clientY - pan.current.y }));
    };
    const onUp = () => { pan.current = null; };
    const onMove = (e) => {
      if (!shown) return;
      const r = e.currentTarget.getBoundingClientRect();
      const i = Math.floor(((e.clientX - r.left) / r.width) * shown.n);
      const p = Math.floor(((e.clientY - r.top) / r.height) * shown.planes);
      if (i < 0 || i >= shown.n || p < 0 || p >= shown.planes) return setHover(null);
      setHover({ i, p, value: section.values[p * shown.n + i] });
    };
    const zOf = (p) => (d.plugin?.stack?.planes ? p * section.z_step : (p + 0.5) * section.z_step);
    return h("div", { className: "cross-section" },
      h("div", { className: "cross-section-bar" },
        h("div", { className: "segmented", role: "group", "aria-label": "Cross-section line" },
          ["x", "y"].map((axis) => h("button", {
            key: axis,
            className: line?.fixed === axis ? "active" : undefined,
            "aria-pressed": line?.fixed === axis,
            title: `Along the ${axis}–z line`,
            onClick: () => choose(axis),
          }, `${axis}–z`))),
        h("button", { className: "secondary", onClick: centre, title: "Move the x–z and y–z lines back to the centre of the image" }, "Centre"),
        h("div", { className: "segmented", role: "group", "aria-label": "Aspect" },
          [[false, "1:1"], [true, "Fill"]].map(([on, label]) => h("button", {
            key: label,
            className: fill === on ? "active" : undefined,
            "aria-pressed": fill === on,
            title: on ? "Stretched to the pane" : "Distances and depths to the same scale",
            onClick: () => setFill(on),
          }, label))),
        visible.some((l) => !l.fixed) && h("select", {
          className: "cell-input",
          style: { width: 96 },
          value: line?.id ?? "",
          "aria-label": "Line",
          onChange: (e) => host.selectLine(d.id, e.target.value),
        }, visible.map((l) => h("option", { key: l.id, value: l.id }, l.name))),
        line && h("span", { className: "small cross-section-swatch", style: { borderColor: line.color } }, line.name),
        zoom.k > 1 && h("span", { className: "small cross-section-zoom" }, `×${num(zoom.k, 1)}`),
        h("span", { className: "small cross-section-readout" },
          hover && section
            ? `${num(hover.i * section.step + section.step / 2, 3)} Å · z ${num(zOf(hover.p), 3)} Å · ${num(hover.value, 4)}`
            : section ? `${num(section.shape[1] * section.step, 3)} × ${num(section.shape[0] * section.z_step, 3)} Å` : "")),
      h("div", { className: "cross-section-plot", ref: holder },
        error && h("p", { className: "model-warning small" }, error),
        line && h("canvas", {
          ref: canvas,
          className: "cross-section-canvas",
          style: shown
            ? {
              width: shown.w,
              height: shown.h,
              left: shown.left,
              top: shown.top,
              transform: `translate(${zoom.x}px, ${zoom.y}px) scale(${zoom.k})`,
              cursor: zoom.k > 1 ? (pan.current ? "grabbing" : "grab") : "crosshair",
            }
            : { display: "none" },
          onMouseMove: onMove,
          onMouseLeave: () => setHover(null),
          onPointerDown: onDown,
          onPointerMove: onDrag,
          onPointerUp: onUp,
          onPointerCancel: onUp,
          onDoubleClick: () => setZoom(NO_ZOOM),
        })));
  }

  /** Makes a CTF in the project and shows it. */
  function NewCtf({ fileId, settings }) {
    const [busy, setBusy] = React.useState(false);
    const [error, setError] = React.useState(null);
    const run = async () => {
      setBusy(true);
      setError(null);
      try {
        const { file, dataset_id } = await host.request(`${API}/projects/${fileId}/ctfs`, {
          method: "POST",
          body: JSON.stringify(settings ?? CTF_DEFAULTS),
        });
        host.showFile(file, dataset_id);
      } catch (e) {
        setError(e.message);
      } finally {
        setBusy(false);
      }
    };
    return h("div", null,
      h("button", { className: settings ? "secondary" : "primary", onClick: run, disabled: busy }, settings ? "Copy as new CTF" : "New CTF"),
      error && h("p", { className: "error small" }, error));
  }

  const isCtf = (d) => d.plugin?.id === "abtem" && d.plugin.label === "CTF" && d.kind === "image";

  /** A CTF's saved settings (its dataset's settings also hold what it is). */
  const savedCtf = (d) => Object.fromEntries(
    Object.entries(CTF_DEFAULTS).map(([k, v]) => [k, d.plugin.settings[k] === undefined ? v : d.plugin.settings[k]]),
  );

  /** A CTF's settings as shown: the form's while it is edited, else the saved ones. */
  function useCtfSettings(d) {
    const draft = React.useSyncExternalStore(subscribeCtfDraft, () => ctfDrafts.get(d.id));
    return draft ?? savedCtf(d);
  }

  /** A CTF's settings: its profile follows at once, its image a moment later,
   * and they are saved with the project after the last change. */
  function CtfForm({ d }) {
    const saved = savedCtf(d);
    const [fields, setFields] = React.useState(() => Object.fromEntries(
      Object.entries(saved).map(([k, v]) => [k, v === null ? "" : String(v)]),
    ));
    const [error, setError] = React.useState(null);
    const drawTimer = React.useRef(null);
    const saveTimer = React.useRef(null);

    const parsed = {};
    let valid = true;
    for (const [k, v] of Object.entries(fields)) {
      if (k === "semiangle" && v.trim() === "") {
        parsed[k] = null;
        continue;
      }
      const x = Number(v);
      if (v.trim() === "" || !Number.isFinite(x)) valid = false;
      parsed[k] = x;
    }
    const inRange = valid && parsed.energy_kv >= 20 && parsed.energy_kv <= 1000 && parsed.max_angle > 0 &&
      parsed.max_angle <= 500 && parsed.focal_spread >= 0 && parsed.angular_spread >= 0 && parsed.astigmatism >= 0 &&
      (parsed.semiangle === null || parsed.semiangle > 0);
    const key = JSON.stringify(parsed);

    React.useEffect(() => {
      if (!inRange) return undefined;
      setCtfDraft(d.id, parsed);
      const put = (save) => host.request(`${API}/ctfs/${d.id}?save=${save}`, { method: "PUT", body: JSON.stringify(parsed) })
        .then(({ file }) => {
          host.updateFile(file);
          setError(null);
        })
        .catch((e) => setError(e.message));
      clearTimeout(drawTimer.current);
      clearTimeout(saveTimer.current);
      drawTimer.current = setTimeout(() => put(false), CTF_IMAGE_MS);
      saveTimer.current = setTimeout(() => put(true), CTF_SAVE_MS);
      return undefined;
    }, [key]); // eslint-disable-line react-hooks/exhaustive-deps
    // Leaving the form: the saved settings are shown again (the last change is still saved).
    React.useEffect(() => () => {
      ctfDrafts.delete(d.id);
      ctfDraftListeners.forEach((listener) => listener());
    }, [d.id]);

    const set = (k) => (e) => setFields((f) => ({ ...f, [k]: e.target.value }));
    const number = (k, label, extra = {}) => h("input", {
      className: "cell-input", type: "number", value: fields[k], "aria-label": label, onChange: set(k), ...extra,
    });
    const row = (label, value, title) => [
      h("dt", { key: `${label}-t`, title }, label),
      h("dd", { key: `${label}-d` }, value),
    ];
    const kv = parsed.energy_kv;
    const canScherzer = inRange && parsed.cs_um > 0;
    const scherzer = () => setFields((f) => ({
      ...f, defocus: String(+Math.sqrt(1.5 * parsed.cs_um * ANGSTROM_PER_UM * wavelength(kv)).toFixed(1)),
    }));
    // The walkthroughs' aperture: at the crossover angle (abTEM's, for these settings).
    const crossover = async () => {
      try {
        const r = await host.request(`${API}/ctfs/evaluate?image=false`, { method: "POST", body: JSON.stringify(parsed) });
        if (r.crossover_angle) setFields((f) => ({ ...f, semiangle: String(+r.crossover_angle.toFixed(1)) }));
      } catch (e) {
        setError(e.message);
      }
    };
    // The CTF walkthrough's focal spread: δ = √2 Cc ΔE / E (1/e width, ΔE its standard deviation).
    const canSpread = inRange && parsed.cc_mm > 0 && parsed.energy_spread > 0;
    const fromCc = () => setFields((f) => ({
      ...f,
      focal_spread: String(+((Math.SQRT2 * parsed.cc_mm * ANGSTROM_PER_MM * parsed.energy_spread) / (kv * 1e3)).toFixed(2)),
    }));
    return h("div", { className: "engine-form model-form" },
      h("dl", { className: "props" },
        row("Energy (kV)", number("energy_kv", "CTF energy", { min: 20, max: 1000 })),
        row("Defocus (Å)", h("span", { className: "model-zone" },
          number("defocus", "CTF defocus", { step: 1 }),
          h("button", {
            className: "secondary",
            disabled: !canScherzer,
            title: canScherzer ? "Set the Scherzer defocus for this Cs" : "Needs a positive Cs",
            onClick: scherzer,
          }, "Scherzer")), "Positive is underfocus"),
        row("Cs (µm)", number("cs_um", "CTF Cs")),
        row("C5 (mm)", number("c5_mm", "CTF C5", { step: 0.1 })),
        row("Astigmatism (Å)", h("span", { className: "model-zone" },
          number("astigmatism", "CTF astigmatism", { min: 0 }),
          number("astigmatism_angle", "CTF astigmatism angle", { step: 5, title: "Angle (°)" })), "Twofold (C12), and its angle in degrees"),
        row("Aperture (mrad)", h("span", { className: "model-zone" },
          number("semiangle", "CTF aperture", { min: 0, placeholder: "None" }),
          h("button", {
            className: "secondary",
            disabled: !canScherzer,
            title: canScherzer ? "Set the aperture at the CTF's first zero (crossover)" : "Needs a positive Cs",
            onClick: crossover,
          }, "Crossover")), "Objective aperture"),
        row("Focal spread (Å)", h("span", { className: "model-zone" },
          number("focal_spread", "CTF focal spread", { min: 0 }),
          h("button", {
            className: "secondary",
            disabled: !canSpread,
            title: "√2 · Cc · ΔE / E",
            onClick: fromCc,
          }, "From Cc, ΔE")), "Temporal coherence: 1/e width of the defocus spread"),
        row("Cc (mm)", number("cc_mm", "CTF Cc", { min: 0, step: 0.1 }), "Chromatic aberration"),
        row("ΔE (eV)", number("energy_spread", "CTF energy spread", { min: 0, step: 0.05 }), "Energy spread (standard deviation)"),
        row("Angular spread (mrad)", number("angular_spread", "CTF angular spread", { min: 0, step: 0.05 }), "Spatial coherence: 1/e width of the illumination angles"),
        row("Range (mrad)", number("max_angle", "CTF range", { min: 1, max: 500 }), "Largest angle shown"),
        parsed.astigmatism > 0 && row("Profile at (°)", number("profile_angle", "CTF profile angle", { step: 5 }), "Direction of the profiles")),
      !inRange && h("p", { className: "model-warning small" }, "Check the values"),
      error && h("p", { className: "model-warning small" }, error),
      h(NewCtf, { fileId: d.file_id, settings: inRange ? parsed : null }));
  }

  /** A CTF's radial profiles (the CTF, its envelopes and aperture) and landmarks:
   * the bottom tab under its 2D image. */
  function CtfProfile({ d }) {
    const settings = useCtfSettings(d);
    const [data, setData] = React.useState(null);
    const [error, setError] = React.useState(null);
    const [units, setUnits] = React.useState("k");
    const [hover, setHover] = React.useState(null);
    const [size, setSize] = React.useState({ w: 0, h: 0 });
    const plot = React.useRef(null);
    const key = JSON.stringify(settings);

    React.useEffect(() => {
      let stale = false;
      const timer = setTimeout(() => {
        host.request(`${API}/ctfs/evaluate?image=false`, { method: "POST", body: key })
          .then((r) => {
            if (stale) return;
            setData(r);
            setError(null);
          })
          .catch((e) => { if (!stale) setError(e.message); });
      }, CTF_DRAW_MS);
      return () => {
        stale = true;
        clearTimeout(timer);
      };
    }, [key]);

    React.useEffect(() => {
      const el = plot.current;
      if (!el) return undefined;
      const observer = new ResizeObserver(([e]) => setSize({ w: e.contentRect.width, h: e.contentRect.height }));
      observer.observe(el);
      return () => observer.disconnect();
    }, []);

    const lambda = data?.wavelength;
    const toX = (k) => (units === "k" ? k : k * lambda * 1000);
    const unitLabel = units === "k" ? "1/Å" : "mrad";
    const facts = data && [
      data.scherzer_defocus !== null && ["Scherzer", `${num(data.scherzer_defocus, 1)} Å`],
      data.point_resolution !== null && ["Point resolution", `${num(data.point_resolution, 3)} Å`],
      data.crossover_angle !== null && ["Crossover", `${num(data.crossover_angle, 1)} mrad`],
      data.information_limit !== null && ["Information limit", `${num(data.information_limit, 3)} Å`],
    ].filter(Boolean);

    let chart = null;
    if (data && size.w > 0) {
      const pad = { l: 36, r: 10, t: 8, b: 26 };
      const w = Math.max(10, size.w - pad.l - pad.r);
      const hgt = Math.max(10, size.h - pad.t - pad.b);
      const n = data.curves.ctf.length;
      const xMax = toX((n - 1) * data.k_step);
      const sx = (x) => pad.l + (x / xMax) * w;
      const sy = (y) => pad.t + ((1 - y) / 2) * hgt;
      const path = (values) => values.map((v, i) => `${i ? "L" : "M"}${sx(toX(i * data.k_step)).toFixed(1)},${sy(v).toFixed(1)}`).join("");
      const step = niceStep(xMax / 6);
      const ticks = [];
      for (let x = 0; x <= xMax + 1e-9; x += step) ticks.push(x);
      const markers = [
        data.point_resolution && { k: 1 / data.point_resolution, label: "Point res." },
        data.information_limit && { k: 1 / data.information_limit, label: "Info limit" },
      ].filter((m) => m && toX(m.k) <= xMax);
      const onMove = (e) => {
        const r = e.currentTarget.getBoundingClientRect();
        const x = ((e.clientX - r.left - pad.l) / w) * xMax;
        if (x < 0 || x > xMax) return setHover(null);
        const i = Math.round((units === "k" ? x : x / (lambda * 1000)) / data.k_step);
        setHover({ i: Math.max(0, Math.min(n - 1, i)) });
      };
      const hk = hover ? hover.i * data.k_step : null;
      chart = h("svg", {
        className: "ctf-plot-svg", width: size.w, height: size.h, onMouseMove: onMove, onMouseLeave: () => setHover(null),
      },
      h("line", { x1: pad.l, x2: pad.l + w, y1: sy(0), y2: sy(0), className: "ctf-axis" }),
      [-1, -0.5, 0.5, 1].map((y) => h("line", { key: y, x1: pad.l, x2: pad.l + w, y1: sy(y), y2: sy(y), className: "ctf-grid" })),
      [-1, 0, 1].map((y) => h("text", { key: y, x: pad.l - 6, y: sy(y) + 3, className: "ctf-tick", textAnchor: "end" }, y)),
      ticks.map((x) => h("g", { key: x },
        h("line", { x1: sx(x), x2: sx(x), y1: pad.t + hgt, y2: pad.t + hgt + 4, className: "ctf-axis" }),
        h("text", { x: sx(x), y: pad.t + hgt + 15, className: "ctf-tick", textAnchor: "middle" }, num(x, 3)))),
      h("text", { x: pad.l + w, y: pad.t + hgt + 24, className: "ctf-tick", textAnchor: "end" }, unitLabel),
      markers.map((m) => h("g", { key: m.label },
        h("line", { x1: sx(toX(m.k)), x2: sx(toX(m.k)), y1: pad.t, y2: pad.t + hgt, className: "ctf-marker" }),
        h("text", { x: sx(toX(m.k)) + 3, y: pad.t + hgt - 4, className: "ctf-tick" }, m.label))),
      CTF_CURVES.filter((c) => data.curves[c.key]).map((c) => h("path", {
        key: c.key, d: path(data.curves[c.key]), fill: "none", stroke: c.color,
        strokeWidth: c.key === "ctf" ? 1.6 : 1.2, strokeDasharray: c.dash ?? undefined,
      })),
      hover && h("line", { x1: sx(toX(hk)), x2: sx(toX(hk)), y1: pad.t, y2: pad.t + hgt, className: "ctf-marker" }));
      if (hover) {
        chart = h(React.Fragment, null, chart, h("div", { className: "ctf-readout small" },
          `${num(hk, 3)} 1/Å · ${hk > 0 ? `${num(1 / hk, 3)} Å · ` : ""}${num(hk * lambda * 1000, 3)} mrad · ` +
          CTF_CURVES.filter((c) => data.curves[c.key]).map((c) => `${c.label} ${num(data.curves[c.key][hover.i], 3)}`).join(" · ")));
      }
    }

    return h("div", { className: "ctf-view" },
      h("div", { className: "ctf-bar" },
        h("div", { className: "segmented", role: "group", "aria-label": "Units" },
          [["k", "1/Å"], ["mrad", "mrad"]].map(([u, label]) => h("button", {
            key: u, className: units === u ? "active" : undefined, "aria-pressed": units === u, onClick: () => setUnits(u),
          }, label))),
        data && h("span", { className: "ctf-legend small" },
          CTF_CURVES.filter((c) => data.curves[c.key]).map((c) => h("span", { key: c.key },
            h("svg", { width: 18, height: 8 }, h("line", { x1: 0, x2: 18, y1: 4, y2: 4, stroke: c.color, strokeWidth: 2, strokeDasharray: c.dash ?? undefined })),
            c.label))),
        facts && h("span", { className: "ctf-facts small" }, facts.map(([k, v]) => h("span", { key: k }, `${k} ${v}`)))),
      error && h("p", { className: "model-warning small ctf-error" }, error),
      !data && !error && h("p", { className: "small ctf-error ctf-loading" }, "Starting abTEM…"),
      h("div", { className: "ctf-body" },
        h("div", { className: "ctf-plot", ref: plot }, chart)));
  }

  /** What a wavefunction was made with. */
  function WaveInfo({ d }) {
    const s = d.plugin.settings;
    const row = (label, value) => [h("dt", { key: `${label}-t` }, label), h("dd", { key: `${label}-d` }, value)];
    const probe = s.beam === "probe";
    return h("div", { className: "run-info" },
      h("dl", { className: "props" },
        row("Beam", probe ? "Probe" : "Plane wave"),
        row("Energy", `${s.energy_kv} kV`),
        probe && row("Convergence", `${s.semiangle} mrad`),
        probe && row("Defocus", `${s.defocus} Å`),
        probe && row("Cs", `${s.cs_um} µm`),
        probe && row("Position", s.position.map((x) => `${num(x, 3)} Å`).join(", ")),
        row("Tiling", s.tile.join(" × ")),
        row("Cell", s.extent.map((x) => x.toFixed(1)).join(" × ") + " Å"),
        row("Potential", potentialLabel(s.potential)),
        row("Grid", s.gpts.join(" × ")),
        row("Planes", s.planes),
        s.configs > 0 && row("Configs", s.configs),
        row("abTEM", s.abtem),
        row("Folder", s.run)));
  }

  /** What a run was made with. */
  function RunInfo({ d }) {
    const s = d.plugin.settings;
    const row = (label, value) => [h("dt", { key: `${label}-t` }, label), h("dd", { key: `${label}-d` }, value)];
    return h("div", { className: "run-info" },
      h("dl", { className: "props" },
        row("Energy", `${s.energy_kv} kV`),
        // Runs made before CTFs had their lens in the run.
        s.defocus !== undefined && row("Defocus", `${s.defocus} Å`),
        s.cs_um !== undefined && row("Cs", `${s.cs_um} µm`),
        s.defocus !== undefined && row("Aperture", s.semiangle ? `${s.semiangle} mrad` : "None"),
        row("Potential", potentialLabel(s.potential)),
        row("Grid", s.gpts.join(" × ")),
        row("Slices", s.slices),
        s.configs > 0 && row("Configs", s.configs),
        row("Exit intensity", num(s.exit_intensity, 4)),
        row("abTEM", s.abtem),
        row("Folder", s.run)));
  }

  host.addCommand({
    id: "new-simulation",
    label: "New simulation…",
    title: "Create a simulation project in a new folder",
    run: async () => {
      const path = await host.pickPath({ kind: "new-folder", title: "New simulation", name: "Simulation", action: "Create" });
      if (!path) return;
      const { file } = await host.request(`${API}/projects`, { method: "POST", body: JSON.stringify({ path }) });
      host.showFile(file);
    },
  });

  host.addProjectPanel({
    type: "simulation",
    render: (file) => h("div", { className: "quantify process project-actions" },
      h(LoadStructure, { fileId: file.id }),
      h(NewCtf, { fileId: file.id })),
  });

  host.addProcess({
    id: "apply-ctf", title: "Apply CTF", tip: "The image a CTF forms of this run's exit wave, without repeating multislice", section: "Simulation",
    applies: (d) => d.plugin?.id === "abtem" && d.plugin.item === "run" && d.plugin.settings?.result === "hrtem",
    render: (d) => h(ApplyCtf, { key: d.id, d }),
  });

  host.addProcess({
    id: "new-ctf", title: "New CTF", tip: "A contrast transfer function in this project", section: "Simulation",
    applies: (d) => d.plugin?.id === "abtem" && !isCtf(d),
    render: (d) => h(NewCtf, { fileId: d.file_id }),
  });

  host.addProcess({
    id: "ctf", title: "CTF", tip: "The objective lens: aberrations, aperture and partial coherence", section: "Simulation",
    applies: isCtf,
    render: (d) => h(CtfForm, { key: d.id, d }),
  });

  host.addBottomTab({
    id: "ctf-profile",
    label: "CTF profile",
    applies: isCtf,
    render: (d) => h(CtfProfile, { key: d.id, d }),
  });

  host.addProcess({
    id: "structure", title: "Structures", tip: "Load another structure into this project", section: "Simulation",
    applies: (d) => d.plugin?.id === "abtem" && d.plugin.item === "structure",
    render: (d) => h(LoadStructure, { fileId: d.file_id }),
  });

  host.addProcess({
    id: "model", title: "Create model", tip: "An oriented, orthogonal supercell along a zone axis", section: "Simulation",
    applies: (d) => d.plugin?.id === "abtem" && d.plugin.item === "structure",
    render: (d) => h(CreateModel, { key: d.id, d }),
  });

  host.addProcess({
    id: "engine", title: "abTEM", tip: "The abTEM install that runs simulations", section: "Simulation",
    applies: (d) => d.plugin?.id === "abtem" && d.plugin.item === "model",
    render: () => h(EngineSection),
  });

  host.addProcess({
    id: "potential", title: "Potential", tip: "Build the model's potential, slice by slice", section: "Simulation",
    applies: (d) => d.plugin?.id === "abtem" && d.plugin.item === "model",
    render: (d) => h(PotentialSection, { key: d.id, d }),
  });

  host.addProcess({
    id: "hrtem", title: "HRTEM", tip: "A plane wave through this potential, then the objective lens", section: "Simulation",
    applies: (d) => d.plugin?.id === "abtem" && d.plugin.label === "Potential",
    render: (d) => h(HrtemSection, { key: d.id, d }),
  });

  host.addProcess({
    id: "wavefunction", title: "Wavefunction", tip: "A plane wave or probe through this potential, kept at every depth", section: "Simulation",
    applies: (d) => d.plugin?.id === "abtem" && d.plugin.label === "Potential",
    render: (d) => h(WaveSection, { key: d.id, d }),
  });

  registerSeries(host, { API, CtfChoice, useProjectCtfs, wavelength, ANTIALIAS, HRTEM, num, potentialLabel });

  activateStem(host);

  host.addBottomTab({
    id: "cross-section",
    label: "Cross-section",
    applies: (d) => d.plugin?.id === "abtem" && d.kind === "stack",
    render: (d) => h(CrossSection, { key: d.id, d }),
  });

  host.addItemView({
    item: "wavefunction",
    label: "Wavefunction",
    render: (d) => h(WaveInfo, { key: d.id, d }),
  });

  host.addItemView({
    item: "run",
    label: "Run",
    render: (d) => h(RunInfo, { key: d.id, d }),
  });

  host.addItemView({
    item: "structure",
    label: "Structure",
    render: (d) => h(StructureView, { key: d.id, d }),
  });

  host.addItemView({
    item: "model",
    label: "Model",
    render: (d) => h(StructureView, { key: d.id, d }),
  });
}
