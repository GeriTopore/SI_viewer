import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { Frame } from "../api/arrays";
import { ContextMenu } from "./ContextMenu";
import { formatValue } from "../format";
import { colormapLut } from "../viewer/colormaps";
import type { DisplayParams } from "../viewer/renderer";

/** One axis of the line-scan image, calibrated: value of index i = offset + i·scale. */
export interface ScanAxis {
  label: string;
  units: string;
  size: number;
  scale: number;
  offset: number;
}

/** A range of positions (rows, inclusive) to average, drawn as a band. */
export interface ScanRange {
  id: string;
  lo: number;
  hi: number;
  color: string;
  selected: boolean;
}

/** What is under the pointer, in hit-test priority. */
type Hit =
  | { kind: "cursor" }
  | { kind: "edge"; id: string; edge: "lo" | "hi" }
  | { kind: "range"; id: string };

/** What a drag in progress is doing (edges are in row-boundary coordinates). */
type Drag =
  | { kind: "cursor" }
  | { kind: "create"; id: string; anchor: number }
  | { kind: "edge"; id: string; anchor: number }
  | { kind: "move"; id: string; lo: number; hi: number; startF: number };

/** Screen distance within which the cursor line or a range edge is grabbed. */
const GRAB_PX = 5;

interface Props {
  /** The image as drawn (possibly log-scaled); rows are positions, columns energy channels. */
  frame: Frame | null;
  /** The same image in data values, for the readout. */
  values: Frame | null;
  display: DisplayParams | undefined;
  rows: ScanAxis;
  cols: ScanAxis;
  /** Row of the spectrum cursor. */
  cursor: number | null;
  cursorColor: string;
  /** The visible ranges, in the order they were made. */
  ranges: ScanRange[];
  loading?: string | null;
  error?: string | null;
  /** Dragging the cursor line (or an arrow key) moves it to this row. */
  onCursor: (row: number) => void;
  /** Shift+drag starts a range; returns its id. */
  onRangeAdd: (lo: number, hi: number) => string;
  /** Dragging a range or one of its edges. */
  onRangeChange: (id: string, lo: number, hi: number) => void;
  /** A press on a range selects it. */
  onSelect: (id: string) => void;
  onRangeRemove: (id: string) => void;
  onRangesClear: () => void;
}

/** 1/2/5 × 10ⁿ tick values inside [lo, hi], about ``count`` of them. */
function niceTicks(lo: number, hi: number, count: number): number[] {
  const span = hi - lo;
  if (!(span > 0) || count < 1) return [];
  const raw = span / count;
  const pow = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * pow).find((s) => s >= raw) ?? pow;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) out.push(Number(v.toPrecision(12)));
  return out;
}

/** Ticks of an axis whose ``size`` pixels span the full length of the plot
 * (pixel i is centred on its calibrated value): each with its position in %. */
function axisTicks(a: ScanAxis, lengthPx: number, pxPerTick: number) {
  const first = a.offset - 0.5 * a.scale;
  const last = a.offset + (a.size - 0.5) * a.scale;
  return niceTicks(Math.min(first, last), Math.max(first, last), Math.floor(lengthPx / pxPerTick)).map((v) => ({
    v,
    pct: (((v - a.offset) / a.scale + 0.5) / a.size) * 100,
  }));
}

/** Paint ``frame`` into ``canvas`` with the colormap and contrast, box-averaged
 * down to at most ``w`` × ``h`` pixels (it is stretched to the panel by CSS). */
function paint(canvas: HTMLCanvasElement, frame: Frame, d: DisplayParams, w: number, h: number) {
  const ow = Math.max(1, Math.min(frame.width, w));
  const oh = Math.max(1, Math.min(frame.height, h));
  canvas.width = ow;
  canvas.height = oh;
  const ctx = canvas.getContext("2d")!;
  const img = ctx.createImageData(ow, oh);
  const lut = colormapLut(d.colormap);
  const scale = 1 / Math.max(d.hi - d.lo, 1e-30);
  for (let oy = 0; oy < oh; oy++) {
    const y0 = Math.floor((oy * frame.height) / oh);
    const y1 = Math.max(y0 + 1, Math.floor(((oy + 1) * frame.height) / oh));
    for (let ox = 0; ox < ow; ox++) {
      const x0 = Math.floor((ox * frame.width) / ow);
      const x1 = Math.max(x0 + 1, Math.floor(((ox + 1) * frame.width) / ow));
      let sum = 0;
      let n = 0;
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          const v = frame.data[y * frame.width + x];
          if (Number.isFinite(v)) {
            sum += v;
            n++;
          }
        }
      }
      const o = (oy * ow + ox) * 4;
      if (!n) {
        img.data[o + 3] = 255;
        continue;
      }
      let t = Math.min(1, Math.max(0, (sum / n - d.lo) * scale));
      if (d.gamma !== 1) t **= d.gamma;
      const k = Math.round(t * 255) * 4;
      img.data[o] = lut[k];
      img.data[o + 1] = lut[k + 1];
      img.data[o + 2] = lut[k + 2];
      img.data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

const TICK_ROW_PX = 34;
const TICK_COL_PX = 84;

/**
 * A line scan as an image of energy (x) against position (y), stretched to
 * fill its panel (the pixels are not square). Hovering only updates the
 * readout. Dragging the cursor line moves the spectrum cursor (clicking
 * elsewhere does nothing; the arrow keys nudge it). Shift+drag makes a range
 * of positions to average; dragging a range's edge changes that limit,
 * dragging its inside moves it, and pressing on it selects it. Right-click a
 * range to delete it.
 */
export function LineScanImage({
  frame, values, display, rows, cols, cursor, cursorColor, ranges, loading, error,
  onCursor, onRangeAdd, onRangeChange, onSelect, onRangeRemove, onRangesClear,
}: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [hover, setHover] = useState<{ row: number; col: number } | null>(null);
  const [hit, setHit] = useState<Hit | null>(null);
  const [dragging, setDragging] = useState<Drag["kind"] | null>(null);
  const [shift, setShift] = useState(false);
  const [menu, setMenu] = useState<{ x: number; y: number; id: string } | null>(null);
  const drag = useRef<Drag | null>(null);

  useLayoutEffect(() => {
    const el = wrapRef.current!;
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !frame || !display || !size.w) return;
    const dpr = window.devicePixelRatio || 1;
    paint(canvas, frame, display, Math.round(size.w * dpr), Math.round(size.h * dpr));
  }, [frame, display, size]);

  // Track Shift so the crosshair pointer follows it without moving the mouse.
  useEffect(() => {
    const on = (e: KeyboardEvent) => e.key === "Shift" && setShift(e.type === "keydown");
    const off = () => setShift(false);
    window.addEventListener("keydown", on);
    window.addEventListener("keyup", on);
    window.addEventListener("blur", off);
    return () => {
      window.removeEventListener("keydown", on);
      window.removeEventListener("keyup", on);
      window.removeEventListener("blur", off);
    };
  }, []);

  /** Row and column of the data under a pointer (clamped to the image). */
  const cellAt = (e: React.PointerEvent | React.MouseEvent) => {
    const r = wrapRef.current!.getBoundingClientRect();
    const clamp = (v: number, n: number) => Math.min(n - 1, Math.max(0, Math.floor(v)));
    return {
      row: clamp(((e.clientY - r.top) / r.height) * rows.size, rows.size),
      col: clamp(((e.clientX - r.left) / r.width) * cols.size, cols.size),
    };
  };

  /** Position of a pointer in row units, 0…rows.size (fractional). */
  const rowAt = (e: React.PointerEvent | React.MouseEvent) => {
    const r = wrapRef.current!.getBoundingClientRect();
    return Math.min(rows.size, Math.max(0, ((e.clientY - r.top) / r.height) * rows.size));
  };

  // Selected range last: drawn on top, and the first to be hit.
  const drawn = [...ranges.filter((r) => !r.selected), ...ranges.filter((r) => r.selected)];

  /** Cursor line, then range edges, then range insides (topmost first). */
  const hitAt = (e: React.PointerEvent | React.MouseEvent, withCursor = true): Hit | null => {
    const rect = wrapRef.current!.getBoundingClientRect();
    const y = e.clientY - rect.top;
    const px = (row: number) => (row / rows.size) * rect.height;
    if (withCursor && cursor !== null && Math.abs(y - px(cursor + 0.5)) <= GRAB_PX) return { kind: "cursor" };
    const top = [...drawn].reverse();
    for (const r of top) {
      const yLo = px(r.lo);
      const yHi = px(r.hi + 1);
      const tol = Math.min(GRAB_PX, (yHi - yLo) / 3); // thin bands keep an inside to grab
      const dLo = Math.abs(y - yLo);
      const dHi = Math.abs(y - yHi);
      if (Math.min(dLo, dHi) <= tol) return { kind: "edge", id: r.id, edge: dLo <= dHi ? "lo" : "hi" };
    }
    for (const r of top) if (y >= px(r.lo) && y <= px(r.hi + 1)) return { kind: "range", id: r.id };
    return null;
  };

  const setHitIfChanged = (h: Hit | null) =>
    setHit((cur) => (JSON.stringify(cur) === JSON.stringify(h) ? cur : h));

  /** Range from an edge held at ``anchor`` and a limit dragged to ``f`` (min one position). */
  const fromEdge = (anchor: number, f: number): [number, number] => {
    let e = Math.round(f);
    if (e === anchor) e = f >= anchor ? anchor + 1 : anchor - 1;
    if (e > rows.size) e = anchor - 1;
    if (e < 0) e = anchor + 1;
    return [Math.min(e, anchor), Math.max(e, anchor) - 1];
  };

  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    wrapRef.current!.focus({ preventScroll: true });
    const row = cellAt(e).row;
    let d: Drag | null = null;
    if (e.shiftKey) {
      d = { kind: "create", id: onRangeAdd(row, row), anchor: row };
    } else {
      const h = hitAt(e);
      if (h?.kind === "cursor") d = { kind: "cursor" };
      else if (h?.kind === "edge") {
        const r = ranges.find((x) => x.id === h.id)!;
        onSelect(h.id);
        d = { kind: "edge", id: h.id, anchor: h.edge === "lo" ? r.hi + 1 : r.lo };
      } else if (h?.kind === "range") {
        const r = ranges.find((x) => x.id === h.id)!;
        onSelect(h.id);
        d = { kind: "move", id: h.id, lo: r.lo, hi: r.hi, startF: rowAt(e) };
      }
    }
    if (!d) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    drag.current = d;
    setDragging(d.kind);
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const c = cellAt(e);
    setHover(c);
    setShift(e.shiftKey);
    const d = drag.current;
    if (!d) return setHitIfChanged(hitAt(e));
    if (d.kind === "cursor") onCursor(c.row);
    else if (d.kind === "create") onRangeChange(d.id, Math.min(d.anchor, c.row), Math.max(d.anchor, c.row));
    else if (d.kind === "edge") onRangeChange(d.id, ...fromEdge(d.anchor, rowAt(e)));
    else {
      const w = d.hi - d.lo;
      const lo = Math.min(rows.size - 1 - w, Math.max(0, d.lo + Math.round(rowAt(e) - d.startF)));
      onRangeChange(d.id, lo, lo + w);
    }
  };

  const endDrag = () => {
    drag.current = null;
    setDragging(null);
  };

  const onContextMenu = (e: React.MouseEvent) => {
    const h = hitAt(e, false);
    if (!h || h.kind === "cursor") return;
    e.preventDefault();
    onSelect(h.id);
    setMenu({ x: e.clientX, y: e.clientY, id: h.id });
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    const step = e.key === "ArrowUp" ? -1 : e.key === "ArrowDown" ? 1 : 0;
    if (!step) return;
    e.preventDefault();
    onCursor(Math.min(rows.size - 1, Math.max(0, (cursor ?? 0) + step * (e.shiftKey ? 10 : 1))));
  };

  const pointer = dragging
    ? { cursor: "row-resize", edge: "ns-resize", move: "grabbing", create: "crosshair" }[dragging]
    : shift
      ? "crosshair"
      : hit?.kind === "cursor"
        ? "row-resize"
        : hit?.kind === "edge"
          ? "ns-resize"
          : hit?.kind === "range"
            ? "grab"
            : "default";

  const yTicks = axisTicks(rows, size.h, TICK_ROW_PX);
  const xTicks = axisTicks(cols, size.w, TICK_COL_PX);
  const axisTitle = (a: ScanAxis) => (a.units ? `${a.label} (${a.units})` : a.label);
  const rowPct = (r: number) => (r / rows.size) * 100;

  return (
    <div className="linescan">
      <div className="linescan-grid">
        <div className="linescan-yaxis">
          <span className="linescan-title">{axisTitle(rows)}</span>
          {yTicks.map((t) => (
            <span key={t.v} className="linescan-tick" style={{ top: `${t.pct}%` }}>
              {formatValue(t.v)}
            </span>
          ))}
        </div>
        <div
          ref={wrapRef}
          className="linescan-wrap"
          style={{ cursor: pointer }}
          tabIndex={0}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={endDrag}
          onPointerCancel={endDrag}
          onPointerLeave={() => {
            setHover(null);
            setHit(null);
          }}
          onContextMenu={onContextMenu}
          onKeyDown={onKeyDown}
        >
          <canvas ref={canvasRef} />
          {drawn.map((r) => (
            <div
              key={r.id}
              className={`linescan-band${r.selected ? " selected" : ""}`}
              style={{
                top: `${rowPct(r.lo)}%`,
                height: `${rowPct(r.hi + 1 - r.lo)}%`,
                ["--range-color" as string]: r.color,
              }}
            />
          ))}
          {cursor !== null && (
            <div
              className="linescan-cursor"
              style={{ top: `${rowPct(cursor + 0.5)}%`, borderColor: cursorColor }}
            />
          )}
          {(loading || error) && (
            <div className={`image-message${error ? " error" : ""}`}>
              {loading && !error && <span className="spinner" aria-hidden />}
              {error ?? loading}
            </div>
          )}
        </div>
        <div />
        <div className="linescan-xaxis">
          {xTicks.map((t) => (
            <span key={t.v} className="linescan-tick" style={{ left: `${t.pct}%` }}>
              {formatValue(t.v)}
            </span>
          ))}
          <span className="linescan-title">{axisTitle(cols)}</span>
        </div>
      </div>
      <div className="readout">
        <span className="readout-pos">
          {hover ? (
            <>
              Position {hover.row}
              <span className="muted">
                {" "}
                ({formatValue(rows.offset + hover.row * rows.scale)}
                {rows.units && ` ${rows.units}`})
              </span>
              {" · "}
              {cols.label} {formatValue(cols.offset + hover.col * cols.scale)}
              {cols.units && ` ${cols.units}`}
              {values && <> · {formatValue(values.data[hover.row * values.width + hover.col])}</>}
            </>
          ) : (
            <span className="muted">
              {rows.size} positions × {cols.size} channels · drag the cursor line · Shift+drag to select positions to average
            </span>
          )}
        </span>
        <span className="readout-zoom">
          {ranges.length > 0 && (
            <button className="link-button" onClick={onRangesClear} title="Delete every range">
              Clear ranges
            </button>
          )}
        </span>
      </div>
      {menu && (
        <ContextMenu
          x={menu.x}
          y={menu.y}
          onClose={() => setMenu(null)}
          items={[
            { label: "Delete range", danger: true, onSelect: () => onRangeRemove(menu.id) },
            ...(ranges.length > 1 ? [{ label: "Delete all ranges", danger: true, onSelect: onRangesClear }] : []),
          ]}
        />
      )}
    </div>
  );
}
