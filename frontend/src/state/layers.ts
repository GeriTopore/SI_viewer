import { create } from "zustand";

/** Visibility and opacity of a layer in the viewer. */
export interface LayerProps {
  visible: boolean;
  opacity: number;
  /** Drawing colour of a shape layer (e.g. atomic columns). */
  colour?: string;
  /** A shape layer coloured by this value (its display: contrast, colormap). */
  colourBy?: string;
  /** A shape layer's mark size (× its default). */
  size?: number;
  /** A vector layer's marks, and how their length follows the vector's
   * magnitude (√: area ∝ magnitude, as kemstem's triangles; equal: all
   * alike). */
  shape?: "triangle" | "arrow" | "line";
  length?: "sqrt" | "linear" | "equal";
}

export const DEFAULT_LAYER: LayerProps = { visible: true, opacity: 1 };

/** Layer-props key of a view's base image (images use their dataset id). */
export const baseLayerKey = (viewKey: string) => `base:${viewKey}`;

interface LayersState {
  props: Record<string, LayerProps>;
  /** Short message for the toast, e.g. why a map can't be added. */
  notice: string | null;
  /** A background task's progress bar (0–1), shown until it is cleared. */
  progress: { label: string; fraction: number } | null;
  setProps: (key: string, patch: Partial<LayerProps>) => void;
  notify: (notice: string | null) => void;
  setProgress: (progress: LayersState["progress"]) => void;
}

export const useLayers = create<LayersState>((set) => ({
  props: {},
  notice: null,
  progress: null,
  setProps: (key, patch) =>
    set((s) => ({ props: { ...s.props, [key]: { ...(s.props[key] ?? DEFAULT_LAYER), ...patch } } })),
  notify: (notice) => set({ notice }),
  setProgress: (progress) => set({ progress }),
}));
