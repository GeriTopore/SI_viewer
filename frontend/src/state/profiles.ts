import { create } from "zustand";
import { putBack, useUndo } from "./undo";

/** A profile line in data pixel coordinates (pixel i spans [i, i+1)). */
export interface ProfileLine {
  id: string;
  name: string;
  color: string;
  visible: boolean;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  /** Width integrated across the line, in pixels. */
  width: number;
  /** Line scans generated along it (dataset ids): deleted with it. */
  lineScans?: string[];
}

export type LineGeometry = Pick<ProfileLine, "x0" | "y0" | "x1" | "y1">;

/** Part of a line picked on the profile plot, as fractions of its length
 * (t0 = t1 for a single point). On a spectrum image its spectrum is shown. */
export interface ProfilePick {
  line: string;
  t0: number;
  t1: number;
}

/** Profile lines and plot state of one image view. */
export interface ViewProfiles {
  lines: ProfileLine[];
  selected: string | null;
  /** Each profile scaled to a maximum of 1. */
  normalise: boolean;
  pick: ProfilePick | null;
  /** Distance (calibrated) hovered on the profile plot, marked on the lines. */
  hover: number | null;
}

/** Pan, profile lines, or FFT boxes. */
export type ImageTool = "pan" | "line" | "fft" | "rect" | "ellipse" | "polygon";

const LINE_COLORS = ["#ffd60a", "#00e5ff", "#ff6bd6", "#8cff5a", "#ff7a59", "#b18cff"];

const EMPTY: ViewProfiles = { lines: [], selected: null, normalise: false, pick: null, hover: null };

interface ProfilesStore {
  byView: Record<string, ViewProfiles>;
  /** Tool of image and stack views (spectrum images use their own tool bar). */
  imageTool: ImageTool;
  setImageTool: (tool: ImageTool) => void;
  patch: (view: string, patch: Partial<ViewProfiles>) => void;
  addLine: (view: string, geometry: LineGeometry) => string;
  updateLine: (view: string, id: string, patch: Partial<ProfileLine>) => void;
  removeLine: (view: string, id: string) => void;
  /** The user deletes a line: it can be undone. */
  deleteLine: (view: string, id: string) => void;
  /** Forget the lines of views whose key starts with one of ``ids`` (e.g. rebinned data). */
  reset: (ids: string[]) => void;
  /** A line whose deletion waits for the user to confirm (it has line scans). */
  confirming: { view: string; id: string } | null;
  setConfirming: (c: { view: string; id: string } | null) => void;
}

let nextId = 1;

export const useProfiles = create<ProfilesStore>((set, get) => {
  const update = (view: string, fn: (s: ViewProfiles) => Partial<ViewProfiles>) =>
    set((st) => {
      const cur = st.byView[view] ?? EMPTY;
      return { byView: { ...st.byView, [view]: { ...cur, ...fn(cur) } } };
    });

  return {
    byView: {},
    imageTool: "pan",
    setImageTool: (imageTool) => set({ imageTool }),
    patch: (view, patch) => update(view, () => patch),
    addLine: (view, geometry) => {
      const id = `line${nextId++}`;
      update(view, (s) => {
        const used = new Set(s.lines.map((l) => l.color));
        const color = LINE_COLORS.find((c) => !used.has(c)) ?? LINE_COLORS[s.lines.length % LINE_COLORS.length];
        const n = s.lines.reduce((m, l) => Math.max(m, Number(l.name.match(/\d+$/)?.[0] ?? 0)), 0) + 1;
        const line: ProfileLine = { id, name: `Line ${n}`, color, visible: true, width: 1, ...geometry };
        return { lines: [...s.lines, line], selected: id };
      });
      return id;
    },
    updateLine: (view, id, patch) =>
      update(view, (s) => ({ lines: s.lines.map((l) => (l.id === id ? { ...l, ...patch } : l)) })),
    removeLine: (view, id) =>
      update(view, (s) => ({
        lines: s.lines.filter((l) => l.id !== id),
        selected: s.selected === id ? null : s.selected,
        pick: s.pick?.line === id ? null : s.pick,
      })),
    confirming: null,
    setConfirming: (confirming) => set({ confirming }),
    deleteLine: (view, id) => {
      const lines = get().byView[view]?.lines ?? [];
      const i = lines.findIndex((l) => l.id === id);
      if (i < 0) return;
      const line = lines[i];
      get().removeLine(view, id);
      useUndo.getState().push(line.name, () => update(view, (s) => ({ lines: putBack(s.lines, line, i) })));
    },
    reset: (ids) =>
      set((st) => ({
        byView: Object.fromEntries(
          Object.entries(st.byView).filter(([k]) => !ids.some((id) => k === id || k.startsWith(`${id}:`))),
        ),
      })),
  };
});

export const useViewProfiles = (view: string): ViewProfiles => useProfiles((s) => s.byView[view]) ?? EMPTY;
export const getViewProfiles = (view: string): ViewProfiles => useProfiles.getState().byView[view] ?? EMPTY;
