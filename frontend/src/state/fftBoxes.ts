import { create } from "zustand";

/** A square region of an image whose FFT the FFT panel can show, in
 * full-resolution pixels (top-left corner and side). */
export interface FftBox {
  id: string;
  name: string;
  color: string;
  x: number;
  y: number;
  size: number;
}

export type BoxGeometry = Pick<FftBox, "x" | "y" | "size">;

/** The FFT boxes of one image view. */
export interface ViewFftBoxes {
  boxes: FftBox[];
  /** The box being edited (Delete removes it). */
  selected: string | null;
  /** The box the FFT panel transforms (null: the part of the image in view). */
  source: string | null;
}

const BOX_COLORS = ["#4ade80", "#38bdf8", "#f472b6", "#facc15", "#fb923c", "#a78bfa"];

const EMPTY: ViewFftBoxes = { boxes: [], selected: null, source: null };

interface FftBoxesStore {
  byView: Record<string, ViewFftBoxes>;
  patch: (view: string, patch: Partial<ViewFftBoxes>) => void;
  /** Add a box; it becomes the selected one and the FFT's source. */
  addBox: (view: string, geometry: BoxGeometry) => string;
  updateBox: (view: string, id: string, patch: Partial<FftBox>) => void;
  removeBox: (view: string, id: string) => void;
  /** Forget the boxes of views whose key starts with one of ``ids`` (e.g. rebinned data). */
  reset: (ids: string[]) => void;
}

let nextId = 1;

export const useFftBoxes = create<FftBoxesStore>((set) => {
  const update = (view: string, fn: (s: ViewFftBoxes) => Partial<ViewFftBoxes>) =>
    set((st) => {
      const cur = st.byView[view] ?? EMPTY;
      return { byView: { ...st.byView, [view]: { ...cur, ...fn(cur) } } };
    });

  return {
    byView: {},
    patch: (view, patch) => update(view, () => patch),
    addBox: (view, geometry) => {
      const id = `fft${nextId++}`;
      update(view, (s) => {
        const used = new Set(s.boxes.map((b) => b.color));
        const color = BOX_COLORS.find((c) => !used.has(c)) ?? BOX_COLORS[s.boxes.length % BOX_COLORS.length];
        const n = s.boxes.reduce((m, b) => Math.max(m, Number(b.name.match(/\d+$/)?.[0] ?? 0)), 0) + 1;
        return { boxes: [...s.boxes, { id, name: `FFT ${n}`, color, ...geometry }], selected: id, source: id };
      });
      return id;
    },
    updateBox: (view, id, patch) =>
      update(view, (s) => ({ boxes: s.boxes.map((b) => (b.id === id ? { ...b, ...patch } : b)) })),
    removeBox: (view, id) =>
      update(view, (s) => ({
        boxes: s.boxes.filter((b) => b.id !== id),
        selected: s.selected === id ? null : s.selected,
        source: s.source === id ? null : s.source,
      })),
    reset: (ids) =>
      set((st) => ({
        byView: Object.fromEntries(
          Object.entries(st.byView).filter(([k]) => !ids.some((id) => k === id || k.startsWith(`${id}:`))),
        ),
      })),
  };
});

export const useViewFftBoxes = (view: string | null | undefined): ViewFftBoxes =>
  useFftBoxes((s) => (view ? s.byView[view] : undefined)) ?? EMPTY;
export const getViewFftBoxes = (view: string): ViewFftBoxes => useFftBoxes.getState().byView[view] ?? EMPTY;
