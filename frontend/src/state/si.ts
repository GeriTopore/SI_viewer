import { create } from "zustand";

export type Tool = "cursor" | "rect" | "ellipse" | "polygon" | "pan";
export type Reduce = "mean" | "sum";
export type SpectrumDrag = "zoom" | "window";

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

export interface EnergyWindow {
  lo: number;
  hi: number;
  /** Set when the window was taken from an element's X-ray line. */
  element?: string;
  label?: string;
}

/** Interaction state of one spectrum image. */
export interface SiState {
  cursor: { x: number; y: number } | null;
  showCursor: boolean;
  showWhole: boolean;
  rois: Roi[];
  selected: string | null;
  window: EnergyWindow | null;
}

export const CURSOR_COLOR = "#ff9f0a";
const ROI_COLORS = ["#00e5ff", "#ff6bd6", "#8cff5a", "#b18cff", "#ffd60a", "#ff7a59"];

const EMPTY: SiState = {
  cursor: null, showCursor: true, showWhole: true, rois: [], selected: null, window: null,
};

interface SiStore {
  bySi: Record<string, SiState>;
  tool: Tool;
  cursorSize: number;
  reduce: Reduce;
  spectrumDrag: SpectrumDrag;
  patch: (si: string, patch: Partial<SiState>) => void;
  addRoi: (si: string, shape: Shape) => string;
  updateRoi: (si: string, id: string, patch: Partial<Roi>) => void;
  removeRoi: (si: string, id: string) => void;
  /** Forget a spectrum image's cursor, regions and window (e.g. after rebinning). */
  reset: (si: string) => void;
  setTool: (tool: Tool) => void;
  setCursorSize: (n: number) => void;
  setReduce: (r: Reduce) => void;
  setSpectrumDrag: (d: SpectrumDrag) => void;
}

let nextId = 1;

export const useSi = create<SiStore>((set) => {
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
    spectrumDrag: "zoom",
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
    reset: (si) =>
      set((st) => {
        const bySi = { ...st.bySi };
        delete bySi[si];
        return { bySi };
      }),
    setTool: (tool) => set({ tool }),
    setCursorSize: (cursorSize) => set({ cursorSize }),
    setReduce: (reduce) => set({ reduce }),
    setSpectrumDrag: (spectrumDrag) => set({ spectrumDrag }),
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

/** Request body for the backend's region-spectrum endpoint. */
export const shapeToRegion = (shape: Shape) =>
  shape.kind === "polygon" ? { kind: "polygon", points: shape.points } : { ...shape };
