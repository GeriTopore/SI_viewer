import { create } from "zustand";
import { putBack, useUndo } from "./undo";

export type Tool = "cursor" | "rect" | "ellipse" | "polygon" | "line" | "fft";
export type Reduce = "mean" | "sum";

/** Region geometry in data pixel coordinates (pixel i spans [i, i+1)). */
export type Shape =
  | { kind: "rect" | "ellipse"; x0: number; y0: number; x1: number; y1: number }
  | { kind: "polygon"; points: [number, number][] };

export interface Roi {
  id: string;
  name: string;
  color: string;
  visible: boolean;
  shape: Shape;
}

/** Interaction state of one spectrum image. */
export interface SiState {
  cursor: { x: number; y: number } | null;
  showCursor: boolean;
  /** A second cursor (⌘-click), compared with the first. */
  cursor2: { x: number; y: number } | null;
  showCursor2: boolean;
  showWhole: boolean;
  rois: Roi[];
  selected: string | null;
  /** The main window: the energy range summed into the virtual image
   * (null: the whole range, i.e. the total counts). */
  main: { lo: number; hi: number } | null;
  /** The energy window selected on the spectrum: "main" or the id of a window map. */
  selectedWindow: string | null;
}

export const CURSOR_COLOR = "#ff9f0a";
export const CURSOR2_COLOR = "#ff453a";
/** Cursor block sizes (n × n pixels averaged). */
export const CURSOR_SIZES = [1, 3, 5, 7, 9, 15];
const ROI_COLORS = ["#00e5ff", "#ff6bd6", "#8cff5a", "#b18cff", "#ffd60a", "#ff7a59"];

const EMPTY: SiState = {
  cursor: null, showCursor: true, cursor2: null, showCursor2: true, showWhole: false, rois: [], selected: null, main: null, selectedWindow: null,
};

interface SiStore {
  bySi: Record<string, SiState>;
  tool: Tool;
  cursorSize: number;
  reduce: Reduce;
  /** Energy windows are drawn on the spectrum plot. */
  showWindows: boolean;
  /** The main window's band is drawn (the virtual image's range; needs showWindows). */
  showMain: boolean;
  /** The view whose region is selected (shown in the Info tab), with its image size. */
  regionView: { key: string; width: number; height: number } | null;
  /** Units of region positions and sizes in the Info tab: "px" or a length/axis unit. */
  regionUnits: string;
  setRegionUnits: (u: string) => void;
  setRegionView: (v: { key: string; width: number; height: number } | null) => void;
  patch: (si: string, patch: Partial<SiState>) => void;
  addRoi: (si: string, shape: Shape) => string;
  updateRoi: (si: string, id: string, patch: Partial<Roi>) => void;
  removeRoi: (si: string, id: string) => void;
  /** The user deletes regions: they can be undone. */
  deleteRois: (si: string, ids: string[]) => void;
  /** Forget a spectrum image's cursor, regions and main window (e.g. after rebinning). */
  reset: (si: string) => void;
  setTool: (tool: Tool) => void;
  setCursorSize: (n: number) => void;
  setReduce: (r: Reduce) => void;
  setShowWindows: (show: boolean) => void;
  setShowMain: (show: boolean) => void;
}

let nextId = 1;

export const useSi = create<SiStore>((set, get) => {
  const update = (si: string, fn: (s: SiState) => Partial<SiState>) =>
    set((st) => {
      const cur = st.bySi[si] ?? EMPTY;
      return { bySi: { ...st.bySi, [si]: { ...cur, ...fn(cur) } } };
    });

  return {
    bySi: {},
    tool: "cursor",
    cursorSize: 1,
    reduce: "mean",
    showWindows: true,
    showMain: false,
    regionView: null,
    regionUnits: "px",
    setRegionUnits: (regionUnits) => set({ regionUnits }),
    setRegionView: (regionView) => set({ regionView }),
    patch: (si, patch) => update(si, () => patch),
    addRoi: (si, shape) => {
      const id = `roi${nextId++}`;
      update(si, (s) => {
        const used = new Set(s.rois.map((r) => r.color));
        const color = ROI_COLORS.find((c) => !used.has(c)) ?? ROI_COLORS[s.rois.length % ROI_COLORS.length];
        const n = s.rois.reduce((m, r) => Math.max(m, Number(r.name.match(/\d+$/)?.[0] ?? 0)), 0) + 1;
        return { rois: [...s.rois, { id, name: `Region ${n}`, color, visible: true, shape }], selected: id };
      });
      return id;
    },
    updateRoi: (si, id, patch) =>
      update(si, (s) => ({ rois: s.rois.map((r) => (r.id === id ? { ...r, ...patch } : r)) })),
    removeRoi: (si, id) =>
      update(si, (s) => ({
        rois: s.rois.filter((r) => r.id !== id),
        selected: s.selected === id ? null : s.selected,
      })),
    deleteRois: (si, ids) => {
      const rois = get().bySi[si]?.rois ?? [];
      const gone = rois.map((r, i) => ({ r, i })).filter(({ r }) => ids.includes(r.id));
      if (!gone.length) return;
      update(si, (s) => ({
        rois: s.rois.filter((r) => !ids.includes(r.id)),
        selected: s.selected && ids.includes(s.selected) ? null : s.selected,
      }));
      useUndo.getState().push(gone.length > 1 ? `${gone.length} regions` : gone[0].r.name, () =>
        update(si, (s) => ({ rois: gone.reduce((out, { r, i }) => putBack(out, r, i), s.rois) })),
      );
    },
    reset: (si) =>
      set((st) => {
        const bySi = { ...st.bySi };
        delete bySi[si];
        return { bySi };
      }),
    setTool: (tool) => set({ tool }),
    setCursorSize: (cursorSize) => set({ cursorSize }),
    setReduce: (reduce) => set({ reduce }),
    setShowWindows: (showWindows) => set({ showWindows }),
    setShowMain: (showMain) => set({ showMain }),
  };
});

export const useSiState = (si: string): SiState => useSi((s) => s.bySi[si]) ?? EMPTY;
export const getSiState = (si: string): SiState => useSi.getState().bySi[si] ?? EMPTY;

// ---------- geometry ----------

export function shapeBounds(shape: Shape): [number, number, number, number] {
  if (shape.kind === "polygon") {
    const xs = shape.points.map((p) => p[0]);
    const ys = shape.points.map((p) => p[1]);
    return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
  }
  return [
    Math.min(shape.x0, shape.x1), Math.min(shape.y0, shape.y1),
    Math.max(shape.x0, shape.x1), Math.max(shape.y0, shape.y1),
  ];
}

export function containsPoint(shape: Shape, x: number, y: number): boolean {
  const [x0, y0, x1, y1] = shapeBounds(shape);
  if (shape.kind !== "polygon") {
    if (shape.kind === "rect") return x >= x0 && x <= x1 && y >= y0 && y <= y1;
    const rx = (x1 - x0) / 2 || 1e-9;
    const ry = (y1 - y0) / 2 || 1e-9;
    return ((x - (x0 + x1) / 2) / rx) ** 2 + ((y - (y0 + y1) / 2) / ry) ** 2 <= 1;
  }
  // Ray casting.
  let inside = false;
  const pts = shape.points;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [xi, yi] = pts[i];
    const [xj, yj] = pts[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

export function translateShape(shape: Shape, dx: number, dy: number): Shape {
  if (shape.kind === "polygon") return { ...shape, points: shape.points.map(([x, y]) => [x + dx, y + dy]) };
  return { ...shape, x0: shape.x0 + dx, x1: shape.x1 + dx, y0: shape.y0 + dy, y1: shape.y1 + dy };
}

/** ``shape`` grown (``step`` 1) or shrunk (-1) about its centre, its
 * proportions kept: the shorter side of its bounds changes by 2 pixels and
 * the other by as much in proportion. Each side changes by an even number of
 * pixels, so the centre stays exactly where it is and the edges on pixel
 * edges. Null when it would get smaller than a pixel. */
export function scaleShape(shape: Shape, step: 1 | -1): Shape | null {
  const [x0, y0, x1, y1] = shapeBounds(shape);
  const w = x1 - x0;
  const h = y1 - y0;
  const short = Math.min(w, h);
  if (short + 2 * step < 1) return null;
  const s = (short + 2 * step) / short;
  // New side: the scaled one, rounded to keep its parity (and at least 1).
  const side = (n: number) => {
    const m = n + 2 * Math.round((n * s - n) / 2);
    return m >= 1 ? m : n;
  };
  const [w2, h2] = [side(w), side(h)];
  const cx = (x0 + x1) / 2;
  const cy = (y0 + y1) / 2;
  if (shape.kind === "polygon") {
    // The bounds' corners land exactly on pixel edges; inner vertices are rounded.
    const [sx, sy] = [w ? w2 / w : 1, h ? h2 / h : 1];
    return {
      ...shape,
      points: shape.points.map(([x, y]) => [
        x === x0 ? cx - w2 / 2 : x === x1 ? cx + w2 / 2 : Math.round(cx + (x - cx) * sx),
        y === y0 ? cy - h2 / 2 : y === y1 ? cy + h2 / 2 : Math.round(cy + (y - cy) * sy),
      ]),
    };
  }
  return { ...shape, x0: cx - w2 / 2, x1: cx + w2 / 2, y0: cy - h2 / 2, y1: cy + h2 / 2 };
}

/** Request body for the backend's region-spectrum endpoint. */
export const shapeToRegion = (shape: Shape) =>
  shape.kind === "polygon" ? { kind: "polygon", points: shape.points } : { ...shape };
