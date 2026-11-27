// A crystal structure in 3D: zone axes, repeats, bonds, element legend, hover
// readout and the nearest zone axis (ported from the abTEM plugin).

import { useEffect, useMemo, useRef, useState } from "react";
import type { DatasetSummary } from "../api/client";
import type { Scene } from "../structure3d/scene";
import {
  MAX_ATOMS,
  MAX_REPEAT,
  countAtoms,
  distinctColors,
  expandAtoms,
  findBonds,
  type DrawnAtoms,
  type Rgb,
  type StructureData,
} from "../structure3d/structure";
import {
  ON_AXIS,
  cellParameters,
  formatZone,
  isHexagonal,
  parseZone,
  toFourIndex,
  zoneFinder,
  zoneVector,
} from "../structure3d/zone";

const ZONES = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]];
const KEY_STEP = 1;
const SHIFT_STEP = 5;
const ROLL_BUTTON_STEP = 15;
// Atoms are drawn at this fraction of their covalent radius.
const ATOM_SCALE = { initial: 0.45, min: 0.1, max: 1, step: 0.01 };
const BOND_TOLERANCE = { initial: 1.05, min: 0.8, max: 1.5, step: 0.01 };

const AXES = [
  { name: "a", color: "#e5484d" },
  { name: "b", color: "#30a46c" },
  { name: "c", color: "#3e8bff" },
];
const AXES_SIZE = 64;
const AXIS_LENGTH = 20;
const LABEL_AT = 27;

const num = (x: number, digits = 4) => String(+Number(x).toFixed(digits));
const cssColor = ([r, g, b]: Rgb) => `rgb(${Math.round(r * 255)} ${Math.round(g * 255)} ${Math.round(b * 255)})`;

/** The a, b, c directions of the view, in the corner. */
function AxesGizmo({ lines }: { lines: React.MutableRefObject<(SVGGElement | null)[]> }) {
  const c = AXES_SIZE / 2;
  return (
    <svg className="structure3d-axes" width={AXES_SIZE} height={AXES_SIZE} aria-hidden>
      {AXES.map((axis, i) => (
        <g key={axis.name} ref={(el) => { lines.current[i] = el; }}>
          <line x1={c} y1={c} x2={c} y2={c} stroke={axis.color} strokeWidth={2} strokeLinecap="round" />
          <text x={c} y={c} fill={axis.color}>{axis.name}</text>
        </g>
      ))}
    </svg>
  );
}

/** Atoms per cell, by element (a mixed site counts its occupancies). */
function compositionOf(data: StructureData): Record<string, number> {
  const out: Record<string, number> = {};
  data.atoms.site.forEach((s) => {
    for (const [el, x] of Object.entries(data.sites[s])) out[el] = (out[el] ?? 0) + x;
  });
  return out;
}

function summaryOf(d: DatasetSummary, data: StructureData): string {
  const [a, b, c, alpha, beta, gamma] = cellParameters(data.cell);
  const s = d.structure;
  const formula = s?.formula
    ?? Object.entries(compositionOf(data)).map(([el, n]) => `${el}${n === 1 ? "" : num(n, 2)}`).join("");
  return [
    formula,
    s?.spacegroup && `${s.spacegroup.symbol} (${s.spacegroup.number})`,
    `a ${num(a)}, b ${num(b)}, c ${num(c)} Å`,
    `${num(alpha, 2)}, ${num(beta, 2)}, ${num(gamma, 2)}°`,
  ].filter(Boolean).join(" · ");
}

function detailsOf(data: StructureData): string {
  return [
    `Sites: ${data.atoms.symbol.length}`,
    `Per cell: ${Object.entries(compositionOf(data)).map(([el, n]) => `${el} ${num(n, 3)}`).join(", ")}`,
    data.sites.some((site) => Object.keys(site).length > 1 || Object.values(site)[0] < 1)
      && `Mixed sites: ${data.sites.filter((site) => Object.keys(site).length > 1 || Object.values(site)[0] < 1).length}`,
  ].filter(Boolean).join("\n");
}

function atomText(data: StructureData, atoms: DrawnAtoms, i: number): string {
  const site = data.sites[atoms.site[i]];
  const occupancy = Object.entries(site).map(([el, x]) => `${el} ${num(x, 3)}`).join(", ");
  return [
    Object.keys(site).join("/"),
    `frac ${atoms.frac[i].map((x) => num(x)).join(", ")}`,
    `${atoms.xyz[i].map((x) => num(x, 3)).join(", ")} Å`,
    `occ ${occupancy}`,
  ].join(" · ");
}

/** How a structure is seen: the direction to the viewer (Cartesian, Å
 * frame of ``cell``), the cell's a, b, c directions in view coordinates (x
 * right, y up, z to the viewer) and the cell (rows a, b, c). */
export interface SceneOrientation {
  view: number[];
  /** The screen's up direction (Cartesian). */
  up: number[];
  axes: number[][];
  cell: number[][];
}

/** ``initial``: the view to start from (the structure's last, wherever shown). */
export function StructureScene({ d, onOrientation, initial }: {
  d: DatasetSummary;
  onOrientation?: (o: SceneOrientation) => void;
  initial?: SceneOrientation;
}) {
  const initialRef = useRef(initial);
  const orientationRef = useRef(onOrientation);
  orientationRef.current = onOrientation;
  const axesNow = useRef<number[][]>([]);
  const [data, setData] = useState<StructureData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hover, setHover] = useState<number | null>(null);
  const [hidden, setHidden] = useState<string[]>([]);
  const [showCell, setShowCell] = useState(true);
  // The nearest zone axis: its label, and its key while the view is on it.
  const [orientation, setOrientation] = useState("");
  const [onZone, setOnZone] = useState<string | null>(null);
  const [zoneText, setZoneText] = useState("");
  const [zoneError, setZoneError] = useState<string | null>(null);
  const [repeats, setRepeats] = useState([1, 1, 1]);
  const [repeatText, setRepeatText] = useState(["1", "1", "1"]);
  const [repeatError, setRepeatError] = useState<string | null>(null);
  const [atomScale, setAtomScale] = useState(ATOM_SCALE.initial);
  const [showBonds, setShowBonds] = useState(false);
  const [tolerance, setTolerance] = useState(BOND_TOLERANCE.initial);
  const [toleranceText, setToleranceText] = useState(String(BOND_TOLERANCE.initial));
  const [sceneReady, setSceneReady] = useState(0);
  const rootRef = useRef<HTMLDivElement>(null);
  const zoneInputRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<Scene | null>(null);
  const axisRefs = useRef<(SVGGElement | null)[]>([]);

  useEffect(() => {
    let stale = false;
    setData(null);
    setError(null);
    fetch(`/api/datasets/${d.id}/structure/atoms`)
      .then(async (res) => {
        if (!res.ok) {
          let detail = res.statusText;
          try {
            detail = (await res.json()).detail ?? detail;
          } catch {
            // not JSON: keep the status text
          }
          throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
        }
        return res.json() as Promise<StructureData>;
      })
      .then((atoms) => { if (!stale) setData(atoms); })
      .catch((e: Error) => { if (!stale) setError(e.message); });
    return () => { stale = true; };
  }, [d.id]);

  useEffect(() => {
    if (!data) return undefined;
    let stale = false;
    const c = AXES_SIZE / 2;
    const onAxes = (...axes: { x: number; y: number; z: number }[]) => {
      axesNow.current = axes.map((v) => [v.x, v.y, v.z]);
      drawAxes(...axes);
    };
    const drawAxes = (...axes: { x: number; y: number }[]) => axes.forEach((v, i) => {
      const g = axisRefs.current[i];
      if (!g) return;
      const [line, text] = Array.from(g.children);
      line.setAttribute("x2", String(c + v.x * AXIS_LENGTH));
      line.setAttribute("y2", String(c - v.y * AXIS_LENGTH));
      text.setAttribute("x", String(c + v.x * LABEL_AT));
      text.setAttribute("y", String(c - v.y * LABEL_AT));
    });
    const nearest = zoneFinder(data.cell);
    const hexagonal = isHexagonal(cellParameters(data.cell));
    const onView = (view: number[], up: number[]) => {
      orientationRef.current?.({ view, up, axes: axesNow.current, cell: data.cell });
      const { uvw, angle } = nearest(view);
      const on = angle < ON_AXIS;
      setOnZone(on ? uvw.join(",") : null);
      const name = formatZone(uvw) + (hexagonal ? ` = ${formatZone(toFourIndex(uvw))}` : "");
      setOrientation(on ? name : `≈ ${name} ${angle.toFixed(1)}°`);
    };
    import("../structure3d/scene")
      .then(({ createScene }) => {
        if (stale || !canvasRef.current) return;
        sceneRef.current = createScene(canvasRef.current, data.cell, { onHover: setHover, onAxes, onView });
        setSceneReady((n) => n + 1);
      })
      .catch((e: Error) => { if (!stale) setError(e.message); });

    // Only while the view has focus: these keys mean other things elsewhere.
    const root = rootRef.current;
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) {
        if (e.key === "Escape") {
          e.stopPropagation();
          root?.focus();
        }
        return;
      }
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const scene = sceneRef.current;
      if (!scene) return;
      const step = e.shiftKey ? SHIFT_STEP : KEY_STEP;
      const look = (uvw: number[]) => scene.lookAlong(zoneVector(data.cell, uvw));
      const actions: Record<string, () => void> = {
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

  const atoms = useMemo(() => (data ? expandAtoms(data, repeats) : null), [data, repeats]);
  const bonds = useMemo(
    () => (atoms && data && showBonds ? findBonds(atoms, data, tolerance) : []),
    [atoms, data, showBonds, tolerance],
  );
  const drawnElements = useMemo(() => (data ? distinctColors(data) : null), [data]);
  useEffect(() => {
    if (!atoms || !data || !drawnElements || !sceneRef.current) return;
    setHover(null);
    sceneRef.current.setModel({
      atoms, bonds, sites: data.sites, elements: drawnElements, repeats, scale: atomScale, hidden,
    });
    // The first model frames the view: then turn to the view to start from, once.
    const start = initialRef.current;
    if (start?.up) {
      initialRef.current = undefined;
      sceneRef.current.setOrientation(start.view, start.up);
    }
  }, [sceneReady, atoms, bonds, data, drawnElements, repeats, atomScale, hidden]);

  if (error) return <div className="structure3d-view"><p className="structure3d-message">{error}</p></div>;
  if (!data || !atoms || !drawnElements) {
    return <div className="structure3d-view"><p className="structure3d-message">Loading…</p></div>;
  }

  const elements = [...new Set(data.sites.flatMap((site) => Object.keys(site)))];
  const toggle = (symbol: string) => {
    setHidden(hidden.includes(symbol) ? hidden.filter((x) => x !== symbol) : [...hidden, symbol]);
  };
  const changeRepeat = (axis: number, text: string) => {
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
  const changeTolerance = (text: string) => {
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
  const hexagonal = isHexagonal(cellParameters(data.cell));
  const look = (uvw: number[]) => {
    setZoneError(null);
    sceneRef.current?.lookAlong(zoneVector(data.cell, uvw));
  };
  const applyZone = () => {
    if (!zoneText.trim()) {
      setZoneError(null);
      return;
    }
    try {
      look(parseZone(zoneText, hexagonal));
      rootRef.current?.focus();
    } catch (e) {
      setZoneError((e as Error).message);
    }
  };

  return (
    <div className="structure3d-view" ref={rootRef} tabIndex={0}>
      <div className="structure3d-zones">
        {ZONES.map((uvw) => {
          const label = formatZone(uvw);
          const active = onZone === uvw.join(",");
          return (
            <button
              key={uvw.join("")}
              className={active ? "active" : undefined}
              aria-pressed={active}
              title={`Look down ${label}`}
              onClick={() => look(uvw)}
            >
              {label}
            </button>
          );
        })}
        <input
          ref={zoneInputRef}
          className={zoneError ? "invalid" : undefined}
          value={zoneText}
          placeholder={hexagonal ? "[uvw], [uvtw]" : "[uvw]"}
          aria-label="Zone axis"
          title="Zone axis (Z), Enter to go"
          spellCheck={false}
          onChange={(e) => { setZoneText(e.target.value); setZoneError(null); }}
          onKeyDown={(e) => { if (e.key === "Enter") applyZone(); }}
        />
        <button
          title="Rotate the view 15° anticlockwise (Q: 1°, Shift: 5°)"
          aria-label="Rotate anticlockwise"
          onClick={() => sceneRef.current?.roll(-ROLL_BUTTON_STEP)}
        >
          ⟲
        </button>
        <button
          title="Rotate the view 15° clockwise (E: 1°, Shift: 5°)"
          aria-label="Rotate clockwise"
          onClick={() => sceneRef.current?.roll(ROLL_BUTTON_STEP)}
        >
          ⟳
        </button>
        {zoneError && <span className="structure3d-zone-error" role="alert">{zoneError}</span>}
      </div>
      <div className="structure3d-options">
        <span className="field" title="Unit cells along a, b and c">
          Repeat
          {["a", "b", "c"].map((axis, i) => (
            <input
              key={axis}
              className="cell-input"
              type="number"
              min={1}
              max={MAX_REPEAT}
              value={repeatText[i]}
              aria-label={`Repeats along ${axis}`}
              title={`Unit cells along ${axis}`}
              onChange={(e) => changeRepeat(i, e.target.value)}
            />
          ))}
        </span>
        <label className="field" title="Atom radius, as a fraction of the covalent radius">
          Size
          <input
            type="range"
            min={ATOM_SCALE.min}
            max={ATOM_SCALE.max}
            step={ATOM_SCALE.step}
            value={atomScale}
            aria-label="Atom size"
            onChange={(e) => setAtomScale(Number(e.target.value))}
          />
          {num(atomScale, 2)}
        </label>
        <label className="field" title="Draw bonds">
          <input type="checkbox" checked={showBonds} onChange={(e) => setShowBonds(e.target.checked)} />
          Bonds
        </label>
        {showBonds && (
          <label className="field" title="Bonded when closer than this × the sum of the covalent radii">
            ×
            <input
              className="cell-input"
              type="number"
              min={BOND_TOLERANCE.min}
              max={BOND_TOLERANCE.max}
              step={BOND_TOLERANCE.step}
              value={toleranceText}
              aria-label="Bond tolerance"
              onChange={(e) => changeTolerance(e.target.value)}
            />
          </label>
        )}
        {repeatError && <span className="structure3d-zone-error" role="alert">{repeatError}</span>}
        <span className="spacer" />
        <span className="muted">
          {`${atoms.xyz.length.toLocaleString()} atoms${showBonds ? ` · ${bonds.length.toLocaleString()} bonds` : ""}`}
        </span>
        <button title="Save the view as PNG, with a transparent background" onClick={savePng}>Save PNG</button>
      </div>
      <div className="structure3d-canvas" ref={canvasRef}>
        <div className="structure3d-legend">
          {elements.map((symbol) => (
            <button
              key={symbol}
              className={hidden.includes(symbol) ? "off" : undefined}
              aria-pressed={!hidden.includes(symbol)}
              title={`Show or hide ${symbol}`}
              onClick={() => toggle(symbol)}
            >
              <span className="structure3d-dot" style={{ background: cssColor(drawnElements[symbol].color) }} />
              {symbol}
            </button>
          ))}
          <button
            className={showCell ? undefined : "off"}
            aria-pressed={showCell}
            title="Show or hide the unit cell"
            onClick={() => {
              sceneRef.current?.setCellVisible(!showCell);
              setShowCell(!showCell);
            }}
          >
            Cell
          </button>
          <span className="spacer" />
          <button title="Back to the first view (R)" onClick={() => sceneRef.current?.reset()}>Reset</button>
        </div>
        <AxesGizmo lines={axisRefs} />
      </div>
      <div className="structure3d-readout" title={detailsOf(data)}>
        <span className="structure3d-readout-text">
          {hover === null ? summaryOf(d, data) : atomText(data, atoms, hover)}
        </span>
        <span className="structure3d-orientation">{orientation}</span>
      </div>
    </div>
  );
}
