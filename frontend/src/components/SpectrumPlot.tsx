import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { useDisplay } from "../state/display";
import { useSi } from "../state/si";
import { useUi } from "../state/ui";
import { formatValue } from "../format";

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

interface Props {
  /** Channel centres along the spectral axis. */
  x: Float64Array;
  series: SpectrumSeries[];
  xLabel: string;
  xUnits: string;
  loading?: string | null;
  error?: string | null;
  markers?: Marker[];
  /** Highlighted energy window. */
  window?: EnergyRange | null;
  /** Called while dragging a window (final = false) and on release. Enables
   * the "Map window" drag mode. */
  onWindow?: (range: EnergyRange, final: boolean) => void;
  onMarkerClick?: (m: Marker) => void;
  /** Extra controls at the start of the toolbar. */
  toolbar?: ReactNode;
  /** Shown under the toolbar (e.g. a legend of series). */
  legend?: ReactNode;
}

const cssVar = (name: string) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/** Spectrum plot (uPlot): drag to zoom energy (or select a window), wheel to
 * zoom, double-click to reset. X-ray lines are drawn as markers. */
export function SpectrumPlot({
  x, series, xLabel, xUnits, loading, error, markers = [], window, onWindow, onMarkerClick, toolbar, legend,
}: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | null>(null);
  const log = useDisplay((s) => s.spectrumLog);
  const theme = useUi((s) => s.theme); // axis colours are read from CSS on build
  const setLog = useDisplay((s) => s.setSpectrumLog);
  const dragMode = useSi((s) => (onWindow ? s.spectrumDrag : "zoom"));
  const setDragMode = useSi((s) => s.setSpectrumDrag);
  const [hover, setHover] = useState<{ x: number; values: (number | null)[] } | null>(null);

  // Read inside uPlot hooks, which are created once per layout.
  const labelBoxes = useRef<LabelBox[]>([]);
  // Counts-axis range set by dragging the axis; null = auto-range to the data.
  const manualY = useRef<{ min: number; max: number } | null>(null);
  const live = useRef({ markers, window, onWindow, onMarkerClick, dragMode, series });
  live.current = { markers, window, onWindow, onMarkerClick, dragMode, series };

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
    const yRange: uPlot.Range.Function = (_u, lo, hi) => {
      if (manualY.current) return [manualY.current.min, manualY.current.max];
      if (lo == null || hi == null) return log ? [1, 10] : [0, 1];
      return log ? uPlot.rangeLog(lo, hi, 10, true) : uPlot.rangeNum(lo, hi, 0.1, true);
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
        // Drag released: zoom (default mode) or set the energy window.
        setSelect: [
          (u) => {
            if (u.select.width < 2) return;
            const lo = u.posToVal(u.select.left, "x");
            const hi = u.posToVal(u.select.left + u.select.width, "x");
            if (live.current.dragMode === "window" && live.current.onWindow) live.current.onWindow({ lo, hi }, true);
            else u.setScale("x", { min: lo, max: hi });
            u.setSelect({ left: 0, top: 0, width: 0, height: 0 }, false);
          },
        ],
        draw: [(u) => (labelBoxes.current = drawOverlays(u, live.current.markers, live.current.window ?? null))],
      },
    };
    const plot = new uPlot(opts, toData(), el);
    plotRef.current = plot;

    // Live window updates while dragging in window mode.
    let down = false;
    let downX = 0;
    const onDown = (e: MouseEvent) => {
      down = true;
      downX = e.clientX;
    };
    const onMove = () => {
      if (!down || live.current.dragMode !== "window" || !live.current.onWindow) return;
      const s = plot.select;
      if (s.width < 2) return;
      live.current.onWindow(
        { lo: plot.posToVal(s.left, "x"), hi: plot.posToVal(s.left + s.width, "x") },
        false,
      );
    };
    // A click (no drag) near a labelled line marker selects it.
    const onUp = (e: MouseEvent) => {
      down = false;
      if (Math.abs(e.clientX - downX) > 3 || !live.current.onMarkerClick) return;
      // Hit-test label boxes (canvas pixels, which include the plot's offset).
      const rect = plot.root.querySelector("canvas")!.getBoundingClientRect();
      const dpr = devicePixelRatio || 1;
      const px = (e.clientX - rect.left) * dpr;
      const py = (e.clientY - rect.top) * dpr;
      const hit = labelBoxes.current.find((b) => px >= b.x0 - 4 * dpr && px <= b.x1 && py >= b.y0 && py <= b.y1);
      if (hit) live.current.onMarkerClick(hit.marker);
    };
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
    const onDblClick = () => resetScales(plot);

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

    const ro = new ResizeObserver(() => plot.setSize({ width: el.clientWidth, height: el.clientHeight }));
    ro.observe(el);
    return () => {
      ro.disconnect();
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
  }, [markers, window]);

  /** Full energy range and an automatic counts range. */
  const resetScales = (plot: uPlot) => {
    manualY.current = null;
    plot.setData(plot.data); // resets x to the data and re-ranges y
  };
  const resetZoom = () => plotRef.current && resetScales(plotRef.current);

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
          <div className="segmented" role="group" aria-label="Drag action">
            <button
              className={dragMode === "zoom" ? "active" : undefined}
              aria-pressed={dragMode === "zoom"}
              onClick={() => setDragMode("zoom")}
              title="Drag on the plot to zoom the energy axis"
            >
              Zoom
            </button>
            <button
              className={dragMode === "window" ? "active" : undefined}
              aria-pressed={dragMode === "window"}
              onClick={() => setDragMode("window")}
              title="Drag on the plot to map that energy window in the navigator"
            >
              Map window
            </button>
          </div>
        )}
        <button
          className="secondary small-button"
          onClick={resetZoom}
          title="Full energy range and automatic counts range (or double-click the plot)"
        >
          Reset zoom
        </button>
      </div>
      {legend}
      <div className="spectrum-plot-wrap">
        <div ref={wrapRef} className={`spectrum-plot${dragMode === "window" ? " window-mode" : ""}`} />
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
              {" · drag an axis to zoom it, Shift-drag to pan"}
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

/** Energy window band and X-ray line markers (labels staggered to avoid
 * overlap). Returns the label boxes, in canvas pixels, for click hit-testing. */
function drawOverlays(u: uPlot, markers: Marker[], window: EnergyRange | null): LabelBox[] {
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

  if (window) {
    const a = u.valToPos(window.lo, "x", true);
    const b = u.valToPos(window.hi, "x", true);
    ctx.fillStyle = "rgba(76, 141, 255, 0.16)";
    ctx.fillRect(Math.min(a, b), top, Math.abs(b - a), height);
    ctx.strokeStyle = "rgba(76, 141, 255, 0.7)";
    ctx.lineWidth = dpr;
    ctx.strokeRect(Math.min(a, b), top, Math.abs(b - a), height);
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
  ctx.restore();
  return boxes;
}
