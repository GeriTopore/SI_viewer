import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { useDisplay } from "../state/display";
import { useSi } from "../state/si";
import { useUi } from "../state/ui";
import { formatValue } from "../format";
import { downloadCsv, safeFileName } from "../download";
import { isTyping } from "../keyboard";
import { PlotExportDialog } from "./PlotExportDialog";

export interface SpectrumSeries {
  label: string;
  /** One value per channel; null where the series has no value (e.g. a fit
   * that only covers part of the spectrum). */
  data: ArrayLike<number | null>;
  color: string;
  /** Line style: dashes, width, and whether to fill under the curve. */
  dash?: number[];
  width?: number;
  fill?: boolean;
}

/** A vertical X-ray line marker. */
export interface Marker {
  energy: number;
  /** Shown for major lines, e.g. "Fe Kα". */
  label?: string;
  color: string;
  /** Relative line intensity (0–1): sets the marker height. */
  weight: number;
  /** Preview (hovered element): drawn dashed. */
  dashed?: boolean;
  element: string;
}

export interface EnergyRange {
  lo: number;
  hi: number;
}

/** An energy window drawn as a band on the plot. */
export interface PlotWindow {
  id: string;
  lo: number;
  hi: number;
  /** Colour of the band (a CSS colour). */
  color: string;
  label: string;
  /** Being drawn (provisional): dashed edges, not selectable. */
  draft?: boolean;
  /** A band of the Background removal tool: the fit range ("background",
   * hatched) or the range kept ("signal"). Always shown, edited by dragging
   * its edges or inside without selecting it; not selectable, no menu.
   * "threshold" is the Thickness tool's zero-loss threshold: a single
   * vertical line (lo = hi, labelled) that is dragged, not resized. */
  kind?: "background" | "signal" | "threshold";
  /** Limits that its edges and moves stay within. */
  min?: number;
  max?: number;
}

interface Props {
  /** Channel centres along the spectral axis. */
  x: Float64Array;
  series: SpectrumSeries[];
  xLabel: string;
  xUnits: string;
  loading?: string | null;
  error?: string | null;
  markers?: Marker[];
  /** Energy windows drawn as bands (toggled by the "Windows" checkbox). */
  windows?: PlotWindow[];
  /** Offers the "Main" checkbox (hiding the main window's band; the caller
   * leaves that band out of `windows` when the setting is off). */
  mainToggle?: boolean;
  /** Id of the selected window: highlighted and annotated, its edges and
   * inside can be dragged. */
  selectedId?: string | null;
  /** A window was clicked (its id) or the empty plot (null; also Esc). */
  onSelectWindow?: (id: string | null) => void;
  /** Called while Shift-dragging a new window (final = false) and on release.
   * Enables drawing windows (a Shift-click without a drag does nothing). */
  onWindow?: (range: EnergyRange, final: boolean) => void;
  /** The selected window's edge or inside is being dragged (snapped to
   * channels): the new range, final = true on release. */
  onEditWindow?: (id: string, range: EnergyRange, final: boolean) => void;
  /** A window was right-clicked (client coordinates); return true to handle it
   * (else the browser's menu shows). */
  onWindowMenu?: (id: string, x: number, y: number) => boolean;
  /** A window was double-clicked; return true to handle it (else the plot's
   * zoom is reset). */
  onWindowDoubleClick?: (id: string) => boolean;
  onMarkerClick?: (m: Marker) => void;
  /** Extra controls at the start of the toolbar. */
  toolbar?: ReactNode;
  /** Shown under the toolbar (e.g. a legend of series). */
  legend?: ReactNode;
  /** Base name of saved files (CSV, PNG). */
  name?: string;
  /** Linear/Log of this plot; without them the shared setting is used. */
  log?: boolean;
  onLog?: (log: boolean) => void;
  /** Remembers this plot's energy zoom while it is unmounted (e.g. on
   * another tab) and across rebuilds. */
  zoomKey?: string;
}

const zoomMemory = new Map<string, { min: number; max: number }>();

/** The windows drawn: the tool's bands always, energy windows when switched on. */
const visibleWindows = (l: { windows: PlotWindow[]; showWindows: boolean }) =>
  l.windows.filter((w) => w.kind || l.showWindows);

const cssVar = (name: string) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/** Spectrum plot (uPlot): drag to zoom energy, Shift-drag to draw an energy
 * window, wheel to zoom, double-click to reset. X-ray lines are drawn as
 * markers; windows as bands, the selected one annotated and editable. */
export function SpectrumPlot({
  x, series, xLabel, xUnits, loading, error, markers = [], windows = [], mainToggle, selectedId = null, onSelectWindow, onWindow,
  onEditWindow, onWindowMenu, onWindowDoubleClick, onMarkerClick, toolbar, legend, name = "spectrum", log: logProp, onLog, zoomKey,
}: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | null>(null);
  const sharedLog = useDisplay((s) => s.spectrumLog);
  const log = logProp ?? sharedLog;
  const theme = useUi((s) => s.theme); // axis colours are read from CSS on build
  const setSharedLog = useDisplay((s) => s.setSpectrumLog);
  const setLog = onLog ?? setSharedLog;
  const showWindows = useSi((s) => s.showWindows);
  const setShowWindows = useSi((s) => s.setShowWindows);
  const showMain = useSi((s) => s.showMain);
  const setShowMain = useSi((s) => s.setShowMain);
  const [hover, setHover] = useState<{ x: number; values: (number | null)[] } | null>(null);

  // Read inside uPlot hooks, which are created once per layout.
  const labelBoxes = useRef<LabelBox[]>([]);
  // Counts-axis range set by dragging the axis; null = auto-range to the data.
  const manualY = useRef<{ min: number; max: number } | null>(null);
  const live = useRef({
    markers, windows, selectedId, showWindows, onSelectWindow, onWindow, onEditWindow, onWindowMenu,
    onWindowDoubleClick, onMarkerClick, series, xUnits,
  });
  live.current = {
    markers, windows, selectedId, showWindows, onSelectWindow, onWindow, onEditWindow, onWindowMenu,
    onWindowDoubleClick, onMarkerClick, series, xUnits,
  };

  const toData = (): uPlot.AlignedData => [
    x,
    ...series.map((s) =>
      log ? Array.from(s.data, (v) => (v !== null && v > 0 ? v : null)) : (s.data as uPlot.AlignedData[1]),
    ),
  ];

  // (Re)build when the series layout or scale type changes; otherwise setData.
  const layoutKey = `${log}|${theme}|${series.map((s) => `${s.label}:${s.color}`).join(",")}|${xLabel}`;

  useLayoutEffect(() => {
    const el = wrapRef.current!;
    manualY.current = null;
    const saved = zoomKey ? zoomMemory.get(zoomKey) : undefined;
    const yRange: uPlot.Range.Function = (_u, lo, hi) => {
      if (manualY.current) return [manualY.current.min, manualY.current.max];
      if (lo == null || hi == null) return log ? [1, 10] : [0, 1];
      if (log) return uPlot.rangeLog(lo, hi, 10, true);
      // Counts start at 0: negatives (e.g. after background removal) only show
      // when the axis is dragged.
      return hi > 0 ? uPlot.rangeNum(Math.max(0, lo), hi, 0.1, true) : uPlot.rangeNum(lo, hi, 0.1, true);
    };
    const text = cssVar("--muted");
    const grid = cssVar("--border");
    const axis = (label?: string): uPlot.Axis => ({
      stroke: text,
      grid: { stroke: grid, width: 1 },
      ticks: { stroke: grid, width: 1 },
      label,
      labelSize: label ? 18 : 0,
      font: "11px -apple-system, sans-serif",
      labelFont: "11px -apple-system, sans-serif",
    });
    const opts: uPlot.Options = {
      width: el.clientWidth,
      height: el.clientHeight,
      scales: { x: { time: false }, y: log ? { distr: 3, log: 10, range: yRange } : { range: yRange } },
      axes: [
        axis(xUnits ? `${xLabel} (${xUnits})` : xLabel),
        { ...axis(), size: 56, values: (_u, vals) => vals.map((v) => (v == null ? "" : formatValue(v))) },
      ],
      series: [
        {},
        ...series.map((s) => ({
          label: s.label,
          stroke: s.color,
          width: s.width ?? 1.25,
          dash: s.dash,
          fill: s.fill === false ? undefined : `${s.color}1a`,
          points: { show: false },
        })),
      ],
      cursor: {
        drag: { x: true, y: false, setScale: false },
        points: { show: false },
      },
      legend: { show: false },
      hooks: {
        setCursor: [
          (u) => {
            const i = u.cursor.idx;
            const current = live.current.series;
            setHover(
              i == null
                ? null
                : { x: x[i], values: current.map((s) => s.data[i] ?? null) },
            );
          },
        ],
        // Drag released: zoom, or (Shift) set the energy window.
        setSelect: [
          (u) => {
            if (u.select.width < 2) return;
            const lo = u.posToVal(u.select.left, "x");
            const hi = u.posToVal(u.select.left + u.select.width, "x");
            if (shiftDrag && live.current.onWindow) live.current.onWindow({ lo, hi }, true);
            else u.setScale("x", { min: lo, max: hi });
            u.setSelect({ left: 0, top: 0, width: 0, height: 0 }, false);
          },
        ],
        setScale: [
          (u, key) => {
            if (key !== "x" || !zoomKey) return;
            const { min, max } = u.scales.x;
            if (min == null || max == null) return;
            const eps = (x[1] - x[0]) * 0.5;
            if (min <= x[0] + eps && max >= x[x.length - 1] - eps) zoomMemory.delete(zoomKey);
            else zoomMemory.set(zoomKey, { min, max });
          },
        ],
        draw: [
          (u) => {
            const l = live.current;
            labelBoxes.current = drawOverlays(u, l.markers, visibleWindows(l), l.selectedId, l.xUnits);
          },
        ],
      },
    };
    const plot = new uPlot(opts, toData(), el);
    plotRef.current = plot;
    if (saved && x.length > 1 && saved.max > x[0] && saved.min < x[x.length - 1]) {
      plot.setScale("x", { min: Math.max(saved.min, x[0]), max: Math.min(saved.max, x[x.length - 1]) });
    }

    const step = x.length > 1 ? x[1] - x[0] : 1;
    const [xFirst, xLast] = [x[0], x[x.length - 1]];
    /** The pointer's x in plot pixels (css, from the plot area's left edge). */
    const plotX = (e: MouseEvent) => e.clientX - plot.over.getBoundingClientRect().left;
    const snapTo = (v: number) => xFirst + Math.round((v - xFirst) / step) * step;
    const shown = () => visibleWindows(live.current).filter((w) => !w.draft && !w.kind);
    const selected = () => shown().find((w) => w.id === live.current.selectedId);
    /** The windows whose edges and inside can be dragged: the selected one and the tool's bands. */
    const editable = () => {
      const w = selected();
      const bands = live.current.windows.filter((b) => b.kind);
      return w ? [w, ...bands] : bands;
    };
    /** Where the pointer is on an editable window: an edge (the nearest) or inside. */
    const partAt = (px: number): { w: PlotWindow; part: "lo" | "hi" | "in" } | null => {
      let best: { w: PlotWindow; part: "lo" | "hi"; d: number } | null = null;
      for (const w of editable()) {
        const da = Math.abs(px - plot.valToPos(w.lo, "x"));
        const db = Math.abs(px - plot.valToPos(w.hi, "x"));
        const d = Math.min(da, db);
        if (d <= EDGE_PX && (!best || d < best.d)) best = { w, part: da <= db ? "lo" : "hi", d };
      }
      if (best) return { w: best.w, part: best.part };
      for (const w of editable()) {
        // A window spanning the whole spectrum can't move: its inside zooms.
        if (w.lo <= xFirst + step / 2 && w.hi >= xLast - step / 2) continue;
        if (px > plot.valToPos(w.lo, "x") && px < plot.valToPos(w.hi, "x")) return { w, part: "in" };
      }
      return null;
    };
    /** The narrowest window under the pointer. */
    const windowAt = (px: number) => {
      const under = shown().filter((w) => px >= plot.valToPos(w.lo, "x") && px <= plot.valToPos(w.hi, "x"));
      under.sort((a, b) => a.hi - a.lo - (b.hi - b.lo));
      return under[0]?.id ?? null;
    };
    /** Click (no drag): select the narrowest window under the pointer, else none. */
    const clickWindow = (px: number) => live.current.onSelectWindow?.(windowAt(px));

    // Shift-drag draws a window (live while dragging); other drags zoom.
    let down = false;
    let downX = 0;
    let shiftDrag = false;
    const onDown = (e: MouseEvent) => {
      down = true;
      downX = e.clientX;
      shiftDrag = e.shiftKey && !!live.current.onWindow;
      if (shiftDrag && !live.current.showWindows) useSi.getState().setShowWindows(true);
    };
    const onMove = (e: MouseEvent) => {
      if (!down) {
        // Pointer over an editable window: edges resize, inside moves.
        const part = e.shiftKey ? null : partAt(plotX(e))?.part;
        plot.over.style.cursor = part === "in" ? "grab" : part ? "ew-resize" : "";
        return;
      }
      if (!shiftDrag) return;
      const s = plot.select;
      if (s.width < 2) return;
      live.current.onWindow?.({ lo: plot.posToVal(s.left, "x"), hi: plot.posToVal(s.left + s.width, "x") }, false);
    };
    // A click (no drag) on a labelled line marker maps it; else it selects a window.
    const onUp = (e: MouseEvent) => {
      down = false;
      if (Math.abs(e.clientX - downX) > 3 || e.shiftKey) return;
      if (live.current.onMarkerClick) {
        // Hit-test label boxes (canvas pixels, which include the plot's offset).
        const rect = plot.root.querySelector("canvas")!.getBoundingClientRect();
        const dpr = devicePixelRatio || 1;
        const px = (e.clientX - rect.left) * dpr;
        const py = (e.clientY - rect.top) * dpr;
        const hit = labelBoxes.current.find((b) => px >= b.x0 - 4 * dpr && px <= b.x1 && py >= b.y0 && py <= b.y1);
        if (hit) {
          live.current.onMarkerClick(hit.marker);
          return;
        }
      }
      clickWindow(plotX(e));
    };

    // Dragging the selected window's edge, or its inside, edits it. This runs
    // before uPlot's own drag-to-zoom, which it replaces for such a press.
    interface Edit {
      id: string;
      part: "lo" | "hi" | "in";
      /** A single line (the threshold): dragging moves it. */
      point?: boolean;
      lo: number;
      hi: number;
      min?: number; // limits of the window
      max?: number;
      at: number; // energy where the press was
      x0: number; // screen x of the press
      moved: boolean;
    }
    let edit: Edit | null = null;
    const rangeOf = (edit: Edit, e: MouseEvent): EnergyRange => {
      const v = plot.posToVal(plotX(e), "x");
      const [first, last] = [Math.max(xFirst, edit.min ?? xFirst), Math.min(xLast, edit.max ?? xLast)];
      if (edit.point) {
        const at = Math.min(Math.max(snapTo(v), first), last);
        return { lo: at, hi: at };
      }
      // An edge drag moves that edge only; moving snaps both.
      if (edit.part === "lo") return { lo: Math.min(Math.max(snapTo(v), first), edit.hi - step), hi: edit.hi };
      if (edit.part === "hi") return { lo: edit.lo, hi: Math.max(Math.min(snapTo(v), last), edit.lo + step) };
      const lo0 = snapTo(edit.lo);
      const hi0 = Math.max(snapTo(edit.hi), lo0 + step);
      const shift = Math.round((v - edit.at) / step) * step;
      const d = Math.min(Math.max(shift, first - lo0), last - hi0);
      return { lo: lo0 + d, hi: hi0 + d };
    };
    const onEditMove = (e: MouseEvent) => {
      if (!edit) return;
      if (!edit.moved && Math.abs(e.clientX - edit.x0) <= 3) return;
      edit.moved = true;
      live.current.onEditWindow?.(edit.id, rangeOf(edit, e), false);
    };
    const onEditUp = (e: MouseEvent) => {
      document.removeEventListener("mousemove", onEditMove);
      document.removeEventListener("mouseup", onEditUp);
      const done = edit;
      edit = null;
      plot.over.style.cursor = "";
      if (!done) return;
      if (done.moved) live.current.onEditWindow?.(done.id, rangeOf(done, e), true);
      else clickWindow(plotX(e));
    };
    const onEditDown = (e: MouseEvent) => {
      if (!live.current.onEditWindow || e.button !== 0 || e.shiftKey || !plot.over.contains(e.target as Node)) return;
      const px = plotX(e);
      const hit = partAt(px);
      if (!hit) return;
      const { w, part } = hit;
      e.stopPropagation(); // not a zoom drag
      e.preventDefault();
      edit = {
        id: w.id, part, point: w.kind === "threshold", lo: w.lo, hi: w.hi, min: w.min, max: w.max,
        at: plot.posToVal(px, "x"), x0: e.clientX, moved: false,
      };
      plot.over.style.cursor = part === "in" ? "grabbing" : "ew-resize";
      document.addEventListener("mousemove", onEditMove);
      document.addEventListener("mouseup", onEditUp);
    };
    plot.root.addEventListener("mousedown", onEditDown, true);
    // Wheel zooms the energy axis around the cursor.
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const { min, max } = plot.scales.x;
      if (min == null || max == null) return;
      const rect = plot.over.getBoundingClientRect();
      const at = plot.posToVal(e.clientX - rect.left, "x");
      const f = Math.exp(e.deltaY * 0.0015);
      const lo = Math.max(x[0], at - (at - min) * f);
      const hi = Math.min(x[x.length - 1], at + (max - at) * f);
      if (hi - lo > (x[1] - x[0]) * 4) plot.setScale("x", { min: lo, max: hi });
    };
    const onDblClick = (e: MouseEvent) => {
      const id = windowAt(plotX(e));
      if (id && live.current.onWindowDoubleClick?.(id)) return;
      resetScales(plot);
    };
    const onContextMenu = (e: MouseEvent) => {
      const id = windowAt(plotX(e));
      if (id && live.current.onWindowMenu?.(id, e.clientX, e.clientY)) e.preventDefault();
    };

    // Drag on an axis (where the tick labels are): zoom that axis around the
    // point where the drag started (right/up zooms in); Shift-drag pans it.
    // Log axes zoom and pan in log space.
    let axisDrag: {
      axis: "x" | "y"; pan: boolean; start: number; min: number; max: number; anchor: number; size: number;
    } | null = null;
    const isLog = (axis: "x" | "y") => axis === "y" && log;
    const fwd = (axis: "x" | "y", v: number) => (isLog(axis) ? Math.log10(v) : v);
    const inv = (axis: "x" | "y", v: number) => (isLog(axis) ? 10 ** v : v);
    const axisAt = (e: PointerEvent): "x" | "y" | null => {
      const o = plot.over.getBoundingClientRect();
      if (e.clientX >= o.left && e.clientX <= o.right && e.clientY > o.bottom) return "x";
      if (e.clientY >= o.top && e.clientY <= o.bottom && e.clientX < o.left) return "y";
      return null;
    };
    const onAxisDown = (e: PointerEvent) => {
      const axis = axisAt(e);
      if (!axis || e.button !== 0) return;
      const { min, max } = plot.scales[axis];
      if (min == null || max == null) return;
      const o = plot.over.getBoundingClientRect();
      const at = axis === "x" ? plot.posToVal(e.clientX - o.left, "x") : plot.posToVal(e.clientY - o.top, "y");
      axisDrag = {
        axis,
        pan: e.shiftKey,
        start: axis === "x" ? e.clientX : e.clientY,
        min: fwd(axis, min),
        max: fwd(axis, max),
        anchor: fwd(axis, at),
        size: axis === "x" ? o.width : o.height,
      };
      try {
        plot.root.setPointerCapture(e.pointerId); // keep the drag when leaving the axis
      } catch {
        /* pointer already released */
      }
      e.preventDefault();
    };
    const onAxisMove = (e: PointerEvent) => {
      if (!axisDrag) {
        const axis = axisAt(e);
        plot.root.style.cursor = !axis ? "" : e.shiftKey ? "grab" : axis === "x" ? "ew-resize" : "ns-resize";
        return;
      }
      const { axis, pan, start, min, max, anchor, size } = axisDrag;
      // Pointer travel along the axis, positive rightwards / upwards.
      const d = axis === "x" ? e.clientX - start : start - e.clientY;
      let lo: number;
      let hi: number;
      if (pan) {
        const shift = (-d / size) * (max - min);
        [lo, hi] = [min + shift, max + shift];
      } else {
        const f = Math.exp(-d * 0.01);
        [lo, hi] = [anchor - (anchor - min) * f, anchor + (max - anchor) * f];
      }
      if (axis === "x") {
        // Stay within the spectrum (panning keeps the width).
        const [x0, x1] = [x[0], x[x.length - 1]];
        if (pan) {
          const w = Math.min(hi - lo, x1 - x0);
          lo = Math.min(Math.max(lo, x0), x1 - w);
          hi = lo + w;
        } else [lo, hi] = [Math.max(lo, x0), Math.min(hi, x1)];
        if (hi - lo > (x[1] - x[0]) * 4) plot.setScale("x", { min: lo, max: hi });
      } else {
        manualY.current = { min: inv("y", lo), max: inv("y", hi) };
        plot.setScale("y", manualY.current);
      }
    };
    const onAxisUp = () => {
      axisDrag = null;
    };
    plot.root.addEventListener("pointerdown", onAxisDown);
    plot.root.addEventListener("pointermove", onAxisMove);
    plot.root.addEventListener("pointerup", onAxisUp);
    plot.root.addEventListener("pointercancel", onAxisUp);
    plot.over.addEventListener("mousedown", onDown);
    plot.over.addEventListener("mousemove", onMove);
    plot.over.addEventListener("mouseup", onUp);
    plot.over.addEventListener("wheel", onWheel, { passive: false });
    plot.over.addEventListener("dblclick", onDblClick);
    plot.over.addEventListener("contextmenu", onContextMenu);

    const ro = new ResizeObserver(() => plot.setSize({ width: el.clientWidth, height: el.clientHeight }));
    ro.observe(el);
    return () => {
      ro.disconnect();
      document.removeEventListener("mousemove", onEditMove);
      document.removeEventListener("mouseup", onEditUp);
      plot.destroy();
      plotRef.current = null;
    };
  }, [layoutKey, x]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    // Keep the current energy zoom when only the data changes.
    const p = plotRef.current;
    if (!p) return;
    const { min, max } = p.scales.x;
    p.setData(toData(), false);
    p.setScale("x", { min: min ?? x[0], max: max ?? x[x.length - 1] });
  }, [series]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    plotRef.current?.redraw(false);
  }, [markers, windows, selectedId, showWindows]);
  const hasBands = windows.some((w) => w.kind === "background" || w.kind === "signal");

  // Esc deselects the window.
  useEffect(() => {
    if (!selectedId || !onSelectWindow) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !isTyping(e) && !document.querySelector(".modal-backdrop")) onSelectWindow(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selectedId, onSelectWindow]);

  /** Full energy range and an automatic counts range. */
  const resetScales = (plot: uPlot) => {
    manualY.current = null;
    plot.setData(plot.data); // resets x to the data and re-ranges y
  };
  const resetZoom = () => plotRef.current && resetScales(plotRef.current);

  /** Every series shown, one column each, against energy. */
  const saveCsv = () =>
    downloadCsv(
      [
        [xUnits ? `${xLabel} (${xUnits})` : xLabel, ...series.map((s) => s.label)],
        ...Array.from(x, (v, i) => [v, ...series.map((s) => s.data[i])]),
      ],
      `${safeFileName(name)}.csv`,
    );

  /** The plot, redrawn by the server (current zoom as the x limits). */
  const [exportZoom, setExportZoom] = useState<{ xlim: [number, number] | null } | null>(null);
  const openExport = () => {
    const { min, max } = plotRef.current?.scales.x ?? {};
    setExportZoom({ xlim: min != null && max != null ? [min, max] : null });
  };

  return (
    <div className="spectrum-view">
      <div className="spectrum-toolbar">
        {toolbar}
        <div className="segmented" role="group" aria-label="Intensity scale">
          <button className={!log ? "active" : undefined} aria-pressed={!log} onClick={() => setLog(false)}>
            Linear
          </button>
          <button className={log ? "active" : undefined} aria-pressed={log} onClick={() => setLog(true)}>
            Log
          </button>
        </div>
        {onWindow && (
          <label className="field" title="Show the energy windows on the plot (Shift-drag makes a new one)">
            <input type="checkbox" checked={showWindows} onChange={(e) => setShowWindows(e.target.checked)} />
            Windows
          </label>
        )}
        {onWindow && mainToggle && (
          <label
            className="field"
            title={showWindows ? "Show the main window (the virtual image's range)" : "Show the main window (needs Windows on)"}
          >
            <input
              type="checkbox"
              checked={showMain}
              disabled={!showWindows}
              onChange={(e) => setShowMain(e.target.checked)}
            />
            Main
          </label>
        )}
        <button
          className="secondary small-button"
          onClick={resetZoom}
          title="Full energy range and automatic counts range (or double-click the plot)"
        >
          Reset zoom
        </button>
        <span className="toolbar-spacer" />
        <button
          className="link-button"
          onClick={saveCsv}
          disabled={!series.length}
          title="Save every spectrum shown as CSV (one column each)"
        >
          CSV
        </button>
        <button
          className="link-button"
          onClick={openExport}
          disabled={!series.length}
          title="Save the plot (PNG, SVG or PDF)"
        >
          PNG
        </button>
      </div>
      {exportZoom && (
        <PlotExportDialog
          series={series.map((s) => ({ label: s.label, x, y: s.data, color: s.color, dash: s.dash, width: s.width }))}
          // Not the hovered element's preview lines, nor windows hidden on the plot.
          markers={markers.filter((m) => !m.dashed)}
          markersOn
          windows={visibleWindows({ windows, showWindows }).filter((w) => !w.draft)}
          windowsOn
          xLabel={xUnits ? `${xLabel} (${xUnits})` : xLabel}
          yLabel="Counts"
          xlim={exportZoom.xlim}
          log={log}
          name={name}
          onClose={() => setExportZoom(null)}
        />
      )}
      {legend}
      <div className="spectrum-plot-wrap">
        <div ref={wrapRef} className="spectrum-plot" />
        {(loading || error) && (
          <div className={`image-message${error ? " error" : ""}`}>
            {loading && !error && <span className="spinner" aria-hidden />}
            {error ?? loading}
          </div>
        )}
      </div>
      <div className="readout">
        <span>
          {hover ? (
            <>
              {formatValue(hover.x)} {xUnits}
              {hover.values.map((v, i) => (
                <span key={i}>
                  {" · "}
                  <span style={{ color: series[i]?.color }}>{series[i]?.label}</span>{" "}
                  {v === null ? "–" : formatValue(v)}
                </span>
              ))}
            </>
          ) : (
            <span className="muted">
              {x.length} channels · {formatValue(x[0] ?? 0)}–{formatValue(x[x.length - 1] ?? 0)} {xUnits}
              {" · drag to zoom"}
              {onWindow && " · Shift-drag for a new energy window"}
              {" · drag an axis to zoom it, Shift-drag to pan"}
              {onWindow && showWindows && selectedId && " · drag a window's edges or inside to edit · right-click a window for more · Esc"}
              {hasBands && " · drag the Background and Signal bands' edges or inside to move them"}
              {onMarkerClick && markers.length > 0 && " · click a line label to map it"}
            </span>
          )}
        </span>
      </div>
    </div>
  );
}

interface LabelBox {
  marker: Marker;
  x0: number;
  x1: number;
  y0: number;
  y1: number;
}

const EDGE_PX = 4;

const hatches = new Map<string, CanvasPattern>();

/** Diagonal hatching in ``color`` (the Background band's fill). */
function hatch(color: string, dpr: number): CanvasPattern {
  const key = `${color}|${dpr}`;
  let pattern = hatches.get(key);
  if (!pattern) {
    const size = Math.round(8 * dpr);
    const tile = document.createElement("canvas");
    tile.width = tile.height = size;
    const g = tile.getContext("2d")!;
    g.strokeStyle = color;
    g.globalAlpha = 0.55;
    g.lineWidth = Math.max(1, dpr);
    // Three segments so the lines continue across tile borders.
    g.beginPath();
    g.moveTo(0, size);
    g.lineTo(size, 0);
    g.moveTo(-size / 2, size / 2);
    g.lineTo(size / 2, -size / 2);
    g.moveTo(size / 2, size * 1.5);
    g.lineTo(size * 1.5, size / 2);
    g.stroke();
    pattern = g.createPattern(tile, "repeat")!;
    hatches.set(key, pattern);
  }
  return pattern;
}

/** Energy window bands, X-ray line markers (labels staggered to avoid
 * overlap) and the selected window's annotation. Returns the marker label
 * boxes, in canvas pixels, for click hit-testing. */
function drawOverlays(
  u: uPlot, markers: Marker[], windows: PlotWindow[], selectedId: string | null, units: string,
): LabelBox[] {
  const boxes: LabelBox[] = [];
  const ctx = u.ctx;
  const { left, top, width, height } = u.bbox;
  const dpr = devicePixelRatio || 1;
  const xmin = u.scales.x.min ?? 0;
  const xmax = u.scales.x.max ?? 0;
  ctx.save();
  ctx.beginPath();
  ctx.rect(left, top, width, height);
  ctx.clip();

  // Translucent bands with solid edges: wider ones underneath, the selected one on top.
  const ordered = [...windows].sort(
    (a, b) => Number(a.id === selectedId) - Number(b.id === selectedId) || Math.abs(b.hi - b.lo) - Math.abs(a.hi - a.lo),
  );
  for (const w of ordered) {
    if (w.kind === "threshold") {
      // One labelled vertical line; its label goes to the side with room.
      const p = u.valToPos(w.lo, "x", true);
      const lw = Math.max(1, Math.round(2 * dpr));
      ctx.globalAlpha = 1;
      ctx.fillStyle = w.color;
      ctx.fillRect(Math.round(p - lw / 2), top, lw, height);
      ctx.font = `600 ${11 * dpr}px -apple-system, sans-serif`;
      ctx.textBaseline = "top";
      const left_ = p > left + width * 0.8;
      ctx.textAlign = left_ ? "right" : "left";
      ctx.fillText(w.label, p + (left_ ? -5 : 5) * dpr, top + 4 * dpr);
      ctx.textAlign = "left";
      continue;
    }
    const sel = w.id === selectedId;
    const a = u.valToPos(Math.min(w.lo, w.hi), "x", true);
    const b = u.valToPos(Math.max(w.lo, w.hi), "x", true);
    ctx.fillStyle = w.color;
    ctx.globalAlpha = sel ? 0.3 : w.kind === "background" ? 0.08 : 0.14;
    ctx.fillRect(a, top, b - a, height);
    if (w.kind === "background") {
      ctx.globalAlpha = 1;
      ctx.fillStyle = hatch(w.color, dpr);
      ctx.fillRect(a, top, b - a, height);
    }
    ctx.globalAlpha = sel || w.kind ? 1 : 0.85;
    ctx.fillStyle = w.color;
    const lw = Math.max(1, Math.round((sel || w.kind ? 2.5 : 1.25) * dpr));
    if (w.draft) {
      ctx.strokeStyle = w.color;
      ctx.lineWidth = lw;
      ctx.setLineDash([5 * dpr, 4 * dpr]);
      for (const e of [a, b]) {
        ctx.beginPath();
        ctx.moveTo(Math.round(e) + 0.5, top);
        ctx.lineTo(Math.round(e) + 0.5, top + height);
        ctx.stroke();
      }
      ctx.setLineDash([]);
    } else for (const e of [a, b]) ctx.fillRect(Math.round(e - lw / 2), top, lw, height);
    ctx.globalAlpha = 1;
    if (w.kind) {
      // Labelled at the top, inside the band's left edge.
      ctx.font = `600 ${11 * dpr}px -apple-system, sans-serif`;
      ctx.textBaseline = "top";
      ctx.fillStyle = w.color;
      ctx.fillText(w.label, Math.max(a, left) + 5 * dpr, top + 4 * dpr);
    }
  }

  const visible = markers.filter((m) => m.energy >= xmin && m.energy <= xmax);
  ctx.font = `${11 * dpr}px -apple-system, sans-serif`;
  ctx.textBaseline = "top";
  const rowEnds: number[] = []; // right edge of the last label in each row
  const rowH = 14 * dpr;
  for (const m of [...visible].sort((a, b) => a.energy - b.energy)) {
    const px = Math.round(u.valToPos(m.energy, "x", true)) + 0.5;
    const h = height * (0.25 + 0.6 * Math.min(1, m.weight));
    ctx.strokeStyle = m.color;
    ctx.globalAlpha = m.dashed ? 0.8 : m.label ? 0.6 : 0.35;
    ctx.lineWidth = dpr;
    ctx.setLineDash(m.dashed ? [4 * dpr, 3 * dpr] : []);
    ctx.beginPath();
    ctx.moveTo(px, top + height);
    ctx.lineTo(px, top + height - h);
    ctx.stroke();
    if (m.label) {
      const w = ctx.measureText(m.label).width;
      let row = rowEnds.findIndex((end) => end < px - 2);
      if (row < 0) row = Math.min(rowEnds.length, 3);
      rowEnds[row] = px + 3 * dpr + w;
      ctx.globalAlpha = 1;
      ctx.fillStyle = m.color;
      const ty = top + 4 * dpr + row * rowH;
      ctx.fillText(m.label, px + 3 * dpr, ty);
      boxes.push({ marker: m, x0: px, x1: px + 3 * dpr + w, y0: ty - 2 * dpr, y1: ty + rowH });
    }
  }
  ctx.setLineDash([]);
  ctx.globalAlpha = 1;

  const chosen = windows.find((w) => w.id === selectedId);
  if (chosen) drawAnnotation(u, chosen, boxes, units);
  ctx.restore();
  return boxes;
}

/** The selected window's width above a dotted double arrow across
 * it, and its two limits: an arrow pointing in to a tick on each edge, with
 * the limit's energy beside the arrow (inside the window when there is no room
 * outside it). */
function drawAnnotation(u: uPlot, w: PlotWindow, boxes: LabelBox[], units: string) {
  const ctx = u.ctx;
  const { left, top, width } = u.bbox;
  const right = left + width;
  const dpr = devicePixelRatio || 1;
  const lo = Math.min(w.lo, w.hi);
  const hi = Math.max(w.lo, w.hi);
  const pa = u.valToPos(lo, "x", true);
  const pb = u.valToPos(hi, "x", true);
  if (pb < left || pa > right) return;

  // Energies to the channel width.
  const xs = u.data[0];
  const step = Math.abs((xs[1] ?? 0) - (xs[0] ?? 0));
  const digits = step > 0 ? Math.min(6, Math.max(0, Math.ceil(-Math.log10(step)))) : 3;
  const fmt = (v: number) => String(Number(v.toFixed(digits)));
  const unit = units ? ` ${units}` : "";
  const text = `Δ ${fmt(hi - lo)}${unit}`;

  const px = 11 * dpr; // text height
  ctx.font = `${px}px -apple-system, sans-serif`;
  ctx.lineCap = "butt";
  const panel = cssVar("--panel") || "#fff";
  const clamp = (v: number) => Math.min(right, Math.max(left, v));
  const ea = clamp(pa);
  const eb = clamp(pb);
  const tw = ctx.measureText(text).width;
  const cx = Math.min(Math.max((ea + eb) / 2, left + tw / 2 + 3 * dpr), Math.max(right - tw / 2 - 3 * dpr, left));

  // Below any line label in the way.
  let ty = top + 4 * dpr;
  for (const b of boxes) if (b.x1 > cx - tw / 2 - 6 * dpr && b.x0 < cx + tw / 2 + 6 * dpr) ty = Math.max(ty, b.y1 + 2 * dpr);
  const ay = Math.round(ty + px + 10 * dpr);

  const label = (s: string, x: number, y: number, align: "left" | "right" | "center") => {
    const tw2 = ctx.measureText(s).width;
    const x0 = align === "left" ? x : align === "right" ? x - tw2 : x - tw2 / 2;
    ctx.globalAlpha = 0.8;
    ctx.fillStyle = panel;
    ctx.fillRect(x0 - 2 * dpr, y - px / 2 - 2 * dpr, tw2 + 4 * dpr, px + 4 * dpr);
    ctx.globalAlpha = 1;
    ctx.fillStyle = w.color;
    ctx.textAlign = align;
    ctx.textBaseline = "middle";
    ctx.fillText(s, x, y);
  };
  ctx.strokeStyle = w.color;
  ctx.fillStyle = w.color;

  label(text, cx, ty + px / 2, "center");

  const head = Math.min(5 * dpr, (eb - ea) / 2);
  const hw = 3 * dpr;
  const triangle = (tip: number, dir: 1 | -1) => {
    ctx.beginPath();
    ctx.moveTo(tip, ay);
    ctx.lineTo(tip - dir * head, ay - hw);
    ctx.lineTo(tip - dir * head, ay + hw);
    ctx.closePath();
    ctx.fill();
  };

  // Dotted double-headed arrow across the window (no head where it is cut off).
  const lineW = Math.max(1, Math.round(1.5 * dpr));
  ctx.lineWidth = lineW;
  ctx.lineCap = "round";
  ctx.setLineDash([0.01, 4 * dpr]);
  ctx.beginPath();
  ctx.moveTo(ea + (pa >= left ? head : 0), ay);
  ctx.lineTo(eb - (pb <= right ? head : 0), ay);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.lineCap = "butt";
  if (pa >= left) triangle(ea, -1);
  if (pb <= right) triangle(eb, 1);

  // The limits: tick on the edge, arrow pointing in, energy beside it.
  const L = 24 * dpr;
  const gap = 5 * dpr;
  const arrowH = 5 * dpr;
  const inside: { s: string; align: "left" | "right" }[] = [];
  for (const side of [-1, 1] as const) {
    const edge = side < 0 ? pa : pb;
    const s = `${fmt(side < 0 ? lo : hi)}${unit}`;
    const room = side < 0 ? edge - left : right - edge; // outside the window
    if (edge < left || edge > right) {
      inside.push({ s, align: side < 0 ? "left" : "right" });
      continue;
    }
    // Tick on the limit.
    const tickW = Math.max(1, Math.round(2 * dpr));
    ctx.fillRect(Math.round(edge - tickW / 2), ay - 6 * dpr, tickW, 12 * dpr);
    // Arrow: shaft and head from outside, the tip on the tick.
    const len = Math.min(L, room);
    if (len >= arrowH + 4 * dpr) {
      const tip = edge + (side * tickW) / 2; // touches the tick's outer side
      const base = tip + side * arrowH;
      ctx.lineWidth = Math.max(1, Math.round(dpr));
      ctx.beginPath();
      ctx.moveTo(edge + side * len, ay);
      ctx.lineTo(base, ay);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(tip, ay);
      ctx.lineTo(base, ay - 3 * dpr);
      ctx.lineTo(base, ay + 3 * dpr);
      ctx.closePath();
      ctx.fill();
    }
    if (room >= L + gap + ctx.measureText(s).width + 2 * dpr) {
      label(s, edge + side * (L + gap), ay, side < 0 ? "right" : "left");
    } else inside.push({ s, align: side < 0 ? "left" : "right" });
  }
  // Labels that don't fit outside go inside, under the arrow row.
  let row = 0;
  for (const { s, align } of inside) {
    const x = align === "left" ? ea + gap : eb - gap;
    const overlap = row > 0 || (inside.length > 1 && eb - ea < 2 * ctx.measureText(s).width + 4 * gap);
    label(s, x, ay + (px + 6 * dpr) * (overlap ? ++row : 1), align);
  }
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
}
