import { useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import type { Frame } from "../api/arrays";
import { useUi } from "../state/ui";
import { formatValue } from "../format";
import { percentile, sortedSample } from "../viewer/contrast";
import type { ScanAxis } from "./LineScanImage";
import { Plot3D, type Plot3DMode } from "./Plot3D";
import { LINE_MAPS, lineColor, lineGradient, lineMap } from "../viewer/diverging";
import { ContextMenu, type MenuItem } from "./ContextMenu";
import { AnnotationsToggle } from "./AnnotationsToggle";
import { normaliseData, type NormSettings } from "../state/normalise";

/** Key of a LineScan's energy marks in ``ui.annotationsHidden`` (both views follow it). */
export const marksKey = (lineScanId: string) => `${lineScanId}:energy-marks`;

/** A mark on the energy axis: a line (lo = hi; shift-click) or a window
 * (shift-drag). Shown on the LineScan image too, in the same colour. */
export interface EnergyMark {
  id: string;
  lo: number;
  hi: number;
  color: string;
}

/** Colours of the marks, in the order they are made (clear of blue–red). */
const MARK_COLOURS = ["#22c55e", "#f59e0b", "#a855f7", "#06b6d4", "#ec4899", "#84cc16", "#eab308", "#14b8a6"];

/** The first colour no mark has (cycling once all are used). */
export const nextMarkColour = (marks: EnergyMark[]) =>
  MARK_COLOURS.find((c) => !marks.some((m) => m.color === c)) ?? MARK_COLOURS[marks.length % MARK_COLOURS.length];

/** What part of a mark is at screen position ``px`` (``toPx``: energy to
 * screen): a line, a window's edge, or its inside; lines and edges first. */
export type MarkPart = { id: string; part: "line" | "lo" | "hi" | "in" };
export function markPartAt(px: number, marks: EnergyMark[], toPx: (e: number) => number, grab = 5): MarkPart | null {
  let best: (MarkPart & { d: number }) | null = null;
  for (const m of marks) {
    const a = toPx(m.lo);
    const b = toPx(m.hi);
    const parts: [MarkPart["part"], number][] = m.lo === m.hi ? [["line", a]] : [["lo", a], ["hi", b]];
    for (const [part, at] of parts) {
      const d = Math.abs(px - at);
      if (d <= grab && (!best || d < best.d)) best = { id: m.id, part, d };
    }
  }
  if (best) return { id: best.id, part: best.part };
  const inside = marks
    .filter((m) => m.lo !== m.hi && px > toPx(m.lo) && px < toPx(m.hi))
    .sort((a, b) => a.hi - a.lo - (b.hi - b.lo));
  return inside[0] ? { id: inside[0].id, part: "in" } : null;
}

/** A mark dragged by ``part`` from where it was (``m0``) by ``d`` in energy;
 * a window keeps lo below hi. */
export function movedMark(m0: EnergyMark, part: MarkPart["part"], d: number): EnergyMark {
  if (part === "line" || part === "in") return { ...m0, lo: m0.lo + d, hi: m0.hi + d };
  if (part === "lo") return { ...m0, lo: Math.min(m0.lo + d, m0.hi), hi: m0.hi };
  return { ...m0, lo: m0.lo, hi: Math.max(m0.hi + d, m0.lo) };
}

/** The pointer over a part of a mark. */
export const markCursor = (part: MarkPart["part"] | undefined, dragging = false) =>
  part === "in" ? (dragging ? "grabbing" : "grab") : part ? "ew-resize" : "";




// The colour map chosen last (kept in this browser).
const CMAP_KEY = "si-viewer:exploded-cmap";
const storedCmap = () => {
  try {
    return localStorage.getItem(CMAP_KEY) ?? LINE_MAPS[0].id;
  } catch {
    return LINE_MAPS[0].id;
  }
};

// About this many spectra are drawn at first (every n-th row).
const MAX_LINES = 40;
// At full spacing the whole cascade is raised by this many spectrum heights
// (shared between the spectra drawn, however many).
const MAX_STACK = 3;
// The 3D view averages channels down to at most this many.
const MAX_3D_COLS = 400;

const cssVar = (name: string) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();


/** Bottom-pane tab of a LineScan: its spectra as a cascade, each raised
 * above the last by an adjustable spacing (the intensity axis is arbitrary),
 * coloured blue to red from the first row to the last, normalised as the
 * dataset's spectra are (``norm``: Process tab, Normalisation). Drag to zoom the
 * energy, double-click resets. Shift-click marks a line, shift-drag a window;
 * right-click one to delete it. *3D* shows the same spectra (the energy range zoomed to
 * in 2D) over energy and the row axis, as a surface or as lines. */
export function ExplodedPanel({ frame: raw, rows, cols, label, marksView, marks, onMarksChange, selected, onSelect, onClose, norm, normToggle }: {
  /** The LineScan's values: rows × channels. */
  frame: Frame | null;
  rows: ScanAxis;
  cols: ScanAxis;
  label: string;
  /** Key the marks are hidden under (``marksKey``). */
  marksView: string;
  marks: EnergyMark[];
  onMarksChange: (marks: EnergyMark[]) => void;
  /** The mark selected (highlighted here and on the image). */
  selected: string | null;
  onSelect: (id: string | null) => void;
  onClose: () => void;
  /** The dataset's normalisation (applied to each spectrum while on). */
  norm?: NormSettings;
  /** The dataset's "Normalised" checkbox (shared with the spectrum plot and the Process tab). */
  normToggle?: ReactNode;
}) {
  const [menu, setMenu] = useState<{ x: number; y: number; id: string } | null>(null);
  const [view, setView] = useState<"2d" | "3d">("2d");
  const [mode3d, setMode3d] = useState<Plot3DMode>("surface");
  const [cmapId, setCmapId] = useState(storedCmap);
  const cmap = lineMap(cmapId);
  const [picking, setPicking] = useState(false);
  const chooseCmap = (id: string) => {
    setCmapId(id);
    setPicking(false);
    try {
      localStorage.setItem(CMAP_KEY, id);
    } catch {
      /* not kept: fine */
    }
  };
  // The energy range of the 2D zoom, taken when switching to 3D (null: all).
  const [range3d, setRange3d] = useState<[number, number] | null>(null);
  const frame = useMemo(() => {
    if (!raw || !norm?.on) return raw;
    const x = Array.from({ length: raw.width }, (_, i) => cols.offset + cols.scale * i);
    const data = new Float32Array(raw.data.length);
    for (let r = 0; r < raw.height; r++) {
      const row = normaliseData(raw.data.subarray(r * raw.width, (r + 1) * raw.width), x, norm);
      row.forEach((v, i) => (data[r * raw.width + i] = v ?? NaN));
    }
    return { ...raw, data };
  }, [raw, norm, cols.offset, cols.scale]);
  const n = frame?.height ?? 0;
  const [step, setStep] = useState(() => Math.max(1, Math.ceil(n / MAX_LINES)));
  useEffect(() => setStep(Math.max(1, Math.ceil(n / MAX_LINES))), [n]);
  const [spacing, setSpacing] = useState(35); // % of the most the cascade is raised (MAX_STACK)
  const theme = useUi((s) => s.theme);
  const wrapRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | null>(null);
  const [hoverE, setHoverE] = useState<number | null>(null);

  // The rows drawn, and the typical range of a spectrum (spacing unit).
  const picked = useMemo(() => {
    const out: number[] = [];
    for (let r = 0; r < n; r += step) out.push(r);
    return out;
  }, [n, step]);
  const amplitude = useMemo(() => {
    if (!frame) return 1;
    const s = sortedSample(frame.data);
    return Math.max(1e-12, percentile(s, 99) - percentile(s, 1));
  }, [frame]);

  const x = useMemo(() => Array.from({ length: cols.size }, (_, i) => cols.offset + i * cols.scale), [cols]);
  const data = (): uPlot.AlignedData => {
    if (!frame) return [[]];
    const w = frame.width;
    // Spread over the spectra drawn: the slider does the same whatever their number.
    const lift = ((spacing / 100) * MAX_STACK * amplitude) / Math.max(1, picked.length - 1);
    return [
      x,
      ...picked.map((r, k) => {
        const row = frame.data.subarray(r * w, (r + 1) * w);
        return Array.from(row, (v) => (Number.isFinite(v) ? v + k * lift : null));
      }),
    ];
  };

  const rowValue = (r: number) => rows.offset + r * rows.scale;

  const show3d = () => {
    const z = zoomX.current;
    setRange3d(z ? [z.min, z.max] : null);
    setView("3d");
  };
  // The picked spectra over the energy range, averaged down to MAX_3D_COLS channels.
  const surface = useMemo(() => {
    if (view !== "3d" || !frame) return null;
    const w = frame.width;
    let c0 = 0;
    let c1 = w;
    if (range3d) {
      c0 = Math.max(0, Math.floor((range3d[0] - cols.offset) / cols.scale));
      c1 = Math.min(w, Math.ceil((range3d[1] - cols.offset) / cols.scale) + 1);
    }
    const bin = Math.max(1, Math.ceil((c1 - c0) / MAX_3D_COLS));
    const nx = Math.floor((c1 - c0) / bin);
    const xs = Array.from({ length: nx }, (_, j) => cols.offset + (c0 + (j + 0.5) * bin - 0.5) * cols.scale);
    const zs = picked.map((r) => {
      const row = frame.data.subarray(r * w, (r + 1) * w);
      return Array.from({ length: nx }, (_, j) => {
        let sum = 0;
        let n = 0;
        for (let c = c0 + j * bin; c < c0 + (j + 1) * bin; c++) {
          if (Number.isFinite(row[c])) {
            sum += row[c];
            n++;
          }
        }
        return n ? sum / n : NaN;
      });
    });
    return { x: xs, y: picked.map(rowValue), z: zs };
  }, [view, frame, picked, range3d, cols]); // eslint-disable-line react-hooks/exhaustive-deps
  const layoutKey = `${theme}|${picked.length}|${n}|${step}`;
  // Read by the lines' colours, so a new colour map is a redraw, not a new plot.
  const liveCmap = useRef(cmap);
  liveCmap.current = cmap;
  // The energy zoom the user set (drag; double-click clears it), kept when the
  // plot is made again (e.g. a new "Every") or its data changes.
  const zoomX = useRef<{ min: number; max: number } | null>(null);
  // Read by the plot's hooks.
  // Hidden marks are not drawn, hit or dragged.
  const hidden = useUi((s) => !!s.annotationsHidden[marksView]);
  const live = useRef({ marks, all: marks, onMarksChange, selected, onSelect, show: () => {} });
  live.current = {
    marks: hidden ? [] : marks,
    all: marks,
    onMarksChange,
    selected,
    onSelect,
    show: () => useUi.getState().setAnnotationsHidden(marksView, false),
  };
  const newId = () => Math.random().toString(36).slice(2, 10);

  useLayoutEffect(() => {
    const el = wrapRef.current!;
    const text = cssVar("--muted");
    const grid = cssVar("--border");
    const last = Math.max(1, n - 1);
    const plot = new uPlot(
      {
        width: el.clientWidth,
        height: el.clientHeight,
        scales: { x: { time: false } },
        axes: [
          {
            stroke: text,
            grid: { stroke: grid, width: 1 },
            ticks: { stroke: grid, width: 1 },
            label: cols.units ? `${cols.label} (${cols.units})` : cols.label,
            labelSize: 16,
            font: "11px -apple-system, sans-serif",
            labelFont: "11px -apple-system, sans-serif",
          },
          // Arbitrary intensity: no values.
          { stroke: text, grid: { show: false }, ticks: { show: false }, values: () => [], size: 8 },
        ],
        series: [
          {},
          ...picked.map((r) => ({
            label: String(r),
            stroke: () => lineColor(liveCmap.current, r / last),
            width: 1,
            points: { show: false },
          })),
        ],
        cursor: { drag: { x: true, y: false, setScale: false }, points: { show: false } },
        legend: { show: false },
        hooks: {
          setCursor: [
            (u) => {
              const i = u.cursor.idx;
              setHoverE(i == null ? null : (u.data[0][i] as number));
            },
          ],
          // Release of a drag: a window (with Shift), else zoom to it.
          setSelect: [
            (u) => {
              if (u.select.width >= 2) {
                const lo = u.posToVal(u.select.left, "x");
                const hi = u.posToVal(u.select.left + u.select.width, "x");
                if (shiftDown) {
                  const l = live.current;
                  const m = { id: newId(), lo, hi, color: nextMarkColour(l.all) };
                  l.onMarksChange([...l.all, m]);
                  l.onSelect(m.id);
                  l.show();
                } else {
                  zoomX.current = { min: lo, max: hi };
                  u.setScale("x", zoomX.current);
                }
              }
              u.setSelect({ left: 0, top: 0, width: 0, height: 0 }, false);
            },
          ],
          // The marks in their colours: lines dashed,
          // windows as a band between two edges; the selected one bolder.
          draw: [
            (u) => {
              const { marks: ms, selected: sel } = live.current;
              if (!ms.length) return;
              const ctx = u.ctx;
              const dpr = devicePixelRatio || 1;
              const { top, height } = u.bbox;
              const vline = (at: number) => {
                ctx.beginPath();
                ctx.moveTo(Math.round(at) + 0.5, top);
                ctx.lineTo(Math.round(at) + 0.5, top + height);
                ctx.stroke();
              };
              ctx.save();
              for (const m of ms) {
                const a = u.valToPos(m.lo, "x", true);
                const b = u.valToPos(m.hi, "x", true);
                const bold = m.id === sel;
                ctx.strokeStyle = m.color;
                ctx.lineWidth = (bold ? 2.5 : 1.25) * dpr;
                ctx.setLineDash(m.lo === m.hi ? [4 * dpr, 3 * dpr] : []);
                if (m.lo === m.hi) vline(a);
                else {
                  ctx.globalAlpha = bold ? 0.24 : 0.12;
                  ctx.fillStyle = m.color;
                  ctx.fillRect(Math.min(a, b), top, Math.abs(b - a), height);
                  ctx.globalAlpha = 1;
                  vline(a);
                  vline(b);
                }
              }
              ctx.restore();
            },
          ],
        },
      },
      data(),
      el,
    );
    plotRef.current = plot;
    if (zoomX.current) plot.setScale("x", zoomX.current);
    const onDblClick = () => {
      zoomX.current = null;
      plot.setData(plot.data);
    };
    plot.over.addEventListener("dblclick", onDblClick);
    // Shift held at the press: a drag makes a window, a click a line.
    let shiftDown = false;
    let downX = 0;
    const onDown = (e: MouseEvent) => {
      shiftDown = e.shiftKey;
      downX = e.clientX;
    };
    const posOf = (e: MouseEvent) => e.clientX - plot.over.getBoundingClientRect().left;
    const onUp = (e: MouseEvent) => {
      if (!shiftDown || Math.abs(e.clientX - downX) > 2) return;
      const at = plot.posToVal(posOf(e), "x");
      const l = live.current;
      const m = { id: newId(), lo: at, hi: at, color: nextMarkColour(l.all) };
      l.onMarksChange([...l.all, m]);
      l.onSelect(m.id);
      l.show();
    };
    // A press on a mark (no Shift) selects it and drags it: a line or an edge
    // moves, a window's inside moves the window. This runs before uPlot's own
    // drag-to-zoom, which it replaces for such a press; elsewhere it deselects.
    const hitAt = (px: number) => markPartAt(px, live.current.marks, (v) => plot.valToPos(v, "x"));
    let drag: { hit: MarkPart; m0: EnergyMark; e0: number } | null = null;
    const onMarkMove = (e: MouseEvent) => {
      if (!drag) return;
      const d = plot.posToVal(posOf(e), "x") - drag.e0;
      const l = live.current;
      const moved = movedMark(drag.m0, drag.hit.part, d);
      l.onMarksChange(l.all.map((m) => (m.id === moved.id ? moved : m)));
    };
    const onMarkUp = () => {
      drag = null;
      plot.over.style.cursor = "";
      document.removeEventListener("mousemove", onMarkMove);
      document.removeEventListener("mouseup", onMarkUp);
    };
    const onMarkDown = (e: MouseEvent) => {
      if (e.button !== 0 || e.shiftKey || !plot.over.contains(e.target as Node)) return;
      const hit = hitAt(posOf(e));
      const l = live.current;
      if (!hit) {
        if (l.selected) l.onSelect(null);
        return;
      }
      e.stopPropagation(); // not a zoom drag
      e.preventDefault();
      l.onSelect(hit.id);
      drag = { hit, m0: l.marks.find((m) => m.id === hit.id)!, e0: plot.posToVal(posOf(e), "x") };
      plot.over.style.cursor = markCursor(hit.part, true);
      document.addEventListener("mousemove", onMarkMove);
      document.addEventListener("mouseup", onMarkUp);
    };
    const onHover = (e: MouseEvent) => {
      if (drag) return;
      plot.over.style.cursor = e.shiftKey ? "" : markCursor(hitAt(posOf(e))?.part);
    };
    plot.root.addEventListener("mousedown", onMarkDown, true);
    plot.over.addEventListener("mousemove", onHover);
    // Right-click on a mark: its menu.
    const onMenu = (e: MouseEvent) => {
      const hit = hitAt(posOf(e));
      if (!hit) return;
      e.preventDefault();
      live.current.onSelect(hit.id);
      setMenu({ x: e.clientX, y: e.clientY, id: hit.id });
    };
    plot.over.addEventListener("mousedown", onDown);
    plot.over.addEventListener("mouseup", onUp);
    plot.over.addEventListener("contextmenu", onMenu);
    const ro = new ResizeObserver(() => plot.setSize({ width: el.clientWidth, height: el.clientHeight }));
    ro.observe(el);
    return () => {
      onMarkUp();
      ro.disconnect();
      plot.destroy();
      plotRef.current = null;
    };
  }, [layoutKey]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    plotRef.current?.redraw(false);
  }, [cmapId, marks, selected, hidden]);

  const menuMark = menu ? marks.find((m) => m.id === menu.id) : undefined;
  const menuItems: MenuItem[] = menuMark
    ? [
        {
          label: menuMark.lo === menuMark.hi ? "Delete line" : "Delete window",
          danger: true,
          onSelect: () => onMarksChange(marks.filter((m) => m.id !== menuMark.id)),
        },
      ]
    : [];

  // New spacing or data: redraw (keeping the energy zoom).
  useEffect(() => {
    const plot = plotRef.current;
    if (!plot) return;
    plot.setData(data());
    if (zoomX.current) plot.setScale("x", zoomX.current);
  }, [frame, spacing, amplitude, picked.length]); // eslint-disable-line react-hooks/exhaustive-deps

  const steps = [1, 2, 5, 10, 20, 50, 100].filter((s) => s === 1 || s < n);
  if (!steps.includes(step)) steps.push(step);
  steps.sort((a, b) => a - b);
  const axisName = rows.units ? `${rows.label} (${rows.units})` : rows.label;

  return (
    <div className="profile-panel exploded-panel">
      <div className="profile-header">
        <span className="profile-line-name" title={label}>
          {label}
        </span>
        <span className="profile-spacer" />
        <AnnotationsToggle view={marksView} what="the lines and windows (here and on the image)" />
        <div className="segmented" role="group" aria-label="2D or 3D">
          <button className={view === "2d" ? "active" : undefined} aria-pressed={view === "2d"} onClick={() => setView("2d")}>
            2D
          </button>
          <button
            className={view === "3d" ? "active" : undefined}
            aria-pressed={view === "3d"}
            onClick={show3d}
            title="3D plot: drag rotates, scroll zooms, right-drag pans"
          >
            3D
          </button>
        </div>
        {view === "3d" && (
          <div className="segmented" role="group" aria-label="Surface or lines">
            {(["surface", "lines"] as Plot3DMode[]).map((m) => (
              <button key={m} className={mode3d === m ? "active" : undefined} aria-pressed={mode3d === m} onClick={() => setMode3d(m)}>
                {m === "surface" ? "Surface" : "Lines"}
              </button>
            ))}
          </div>
        )}
        <label
          className="profile-width"
          title="Space between successive spectra"
          style={view === "3d" ? { display: "none" } : undefined}
        >
          Spacing
          <input
            type="range"
            min={0}
            max={100}
            step={1}
            value={spacing}
            onChange={(e) => setSpacing(Number(e.target.value))}
            aria-label="Spacing between spectra"
          />
        </label>
        {normToggle}
        <label className="profile-width" title="Draw every n-th spectrum">
          Every
          <select value={step} onChange={(e) => setStep(Number(e.target.value))}>
            {steps.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>
        <span className="exploded-scale">
          {formatValue(rowValue(0))}
          <button
            className="exploded-bar"
            style={{ background: lineGradient(cmap) }}
            onClick={() => setPicking((p) => !p)}
            title={`Colour along ${axisName}: ${cmap.label} (click to change)`}
            aria-label="Choose the colour map"
            aria-expanded={picking}
          />
          {formatValue(rowValue(Math.max(0, n - 1)))}
          {rows.units && ` ${rows.units}`}
          {picking && <CmapMenu current={cmapId} onChoose={chooseCmap} onClose={() => setPicking(false)} />}
        </span>
        <button className="link-button" onClick={onClose} title="Close the exploded view">
          Close
        </button>
      </div>
      {view === "3d" && surface && (
        <Plot3D
          x={surface.x}
          y={surface.y}
          z={surface.z}
          mode={mode3d}
          xTitle={cols.units ? `${cols.label} (${cols.units})` : cols.label}
          yTitle={axisName}
          zTitle={norm?.on ? "Intensity (normalised)" : "Intensity"}
          cmap={cmap}
        />
      )}
      <div className="profile-plot" style={view === "3d" ? { display: "none" } : undefined}>
        <div className="profile-readout">
          {hoverE !== null ? (
            <>
              {formatValue(hoverE)} {cols.units}
            </>
          ) : null}
        </div>
        <div ref={wrapRef} className="profile-canvas" />
      </div>
      {menu && menuItems.length > 0 && (
        <ContextMenu x={menu.x} y={menu.y} onClose={() => setMenu(null)} items={menuItems} />
      )}
    </div>
  );
}

/** The colour maps to choose from, under the colour bar. Esc or a click
 * elsewhere closes it. */
function CmapMenu({ current, onChoose, onClose }: { current: string; onChoose: (id: string) => void; onClose: () => void }) {
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const away = (e: PointerEvent) => {
      const t = e.target as Element;
      if (!box.current?.contains(t) && !t.closest?.(".exploded-bar")) onClose();
    };
    const esc = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("pointerdown", away);
    window.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("pointerdown", away);
      window.removeEventListener("keydown", esc);
    };
  }, [onClose]);
  return (
    <div className="cmap-menu" ref={box} role="listbox" aria-label="Colour map">
      {LINE_MAPS.map((m) => (
        <button
          key={m.id}
          role="option"
          aria-selected={m.id === current}
          className={m.id === current ? "active" : undefined}
          onClick={() => onChoose(m.id)}
        >
          <i style={{ background: lineGradient(m) }} />
          {m.label}
        </button>
      ))}
    </div>
  );
}
