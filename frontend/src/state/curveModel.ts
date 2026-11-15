import { useEffect } from "react";
import { create } from "zustand";
import { api, dataVersion, type CurveFitResult, type DatasetSummary } from "../api/client";
import { COLOURS, useCurveFit } from "./curveFit";
import { useSi, useSiState } from "./si";

/** A curve model's view (origin "curve-model", M21c): which components the
 * plot shows, which sections are folded, and the model at the cursor. */
interface ModelView {
  hidden: string[];
  folded: string[];
  at: { key: string; result: CurveFitResult & { data: number[] } } | null;
}

interface CurveModelStore {
  byId: Record<string, ModelView>;
  patch: (id: string, patch: Partial<ModelView>) => void;
}

const EMPTY: ModelView = { hidden: [], folded: [], at: null };

export const useCurveModel = create<CurveModelStore>((set) => ({
  byId: {},
  patch: (id, patch) => set((s) => ({ byId: { ...s.byId, [id]: { ...(s.byId[id] ?? EMPTY), ...patch } } })),
}));

export const useModelView = (d: DatasetSummary) => useCurveModel((s) => s.byId[d.id]) ?? EMPTY;

/** Show or hide a component on the plot. */
export function toggleShown(d: DatasetSummary, name: string, show: boolean) {
  const v = useCurveModel.getState().byId[d.id] ?? EMPTY;
  useCurveModel.getState().patch(d.id, { hidden: show ? v.hidden.filter((n) => n !== name) : [...v.hidden, name] });
}

/** Fold or unfold a component's section. */
export function toggleFolded(d: DatasetSummary, name: string) {
  const v = useCurveModel.getState().byId[d.id] ?? EMPTY;
  useCurveModel.getState().patch(d.id, {
    folded: v.folded.includes(name) ? v.folded.filter((n) => n !== name) : [...v.folded, name],
  });
}

/** Each component's colour: as in the Curve fitting section of the data it
 * was fitted to, where the component is still there; else in order. */
export function useModelColours(d: DatasetSummary): Record<string, string> {
  const source = d.curve_model?.source_id;
  const live = useCurveFit((s) => (source ? s.byId[source]?.components : undefined));
  const out: Record<string, string> = {};
  (d.curve_model?.model.components ?? []).forEach((c, i) => {
    out[c.name] = live?.find((x) => x.name === c.name)?.color ?? COLOURS[i % COLOURS.length];
  });
  return out;
}

const fetching = new Set<string>();
/** The model at the cursor's block (the last one stays while the next is fetched). */
export function useModelAtCursor(d: DatasetSummary) {
  const st = useSiState(d.id);
  const size = useSi((s) => s.cursorSize);
  const view = useModelView(d);
  const cursor = st.cursor;
  const key = d.curve_model && cursor ? `${d.id}|${dataVersion(d)}|${cursor.x}|${cursor.y}|${size}` : null;
  useEffect(() => {
    if (!key || !cursor || fetching.has(key) || useCurveModel.getState().byId[d.id]?.at?.key === key) return;
    fetching.add(key);
    api
      .curveModelPixel(d.id, cursor.x, cursor.y, size)
      .then((result) => useCurveModel.getState().patch(d.id, { at: { key, result } }))
      .catch(() => useCurveModel.getState().patch(d.id, { at: null })) // e.g. the fit failed there
      .finally(() => fetching.delete(key));
  }, [key]); // eslint-disable-line react-hooks/exhaustive-deps
  return { result: key ? (view.at?.result ?? null) : null, cursor: cursor ? { ...cursor, size } : null };
}
