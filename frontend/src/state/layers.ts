import { create } from "zustand";

/** Visibility and opacity of a layer in the viewer. */
export interface LayerProps {
  visible: boolean;
  opacity: number;
}

export const DEFAULT_LAYER: LayerProps = { visible: true, opacity: 1 };

/** Layer-props key of a view's base image (images use their dataset id). */
export const baseLayerKey = (viewKey: string) => `base:${viewKey}`;

interface LayersState {
  props: Record<string, LayerProps>;
  /** Short message for the toast, e.g. why a map can't be added. */
  notice: string | null;
  setProps: (key: string, patch: Partial<LayerProps>) => void;
  notify: (notice: string | null) => void;
}

export const useLayers = create<LayersState>((set) => ({
  props: {},
  notice: null,
  setProps: (key, patch) =>
    set((s) => ({ props: { ...s.props, [key]: { ...(s.props[key] ?? DEFAULT_LAYER), ...patch } } })),
  notify: (notice) => set({ notice }),
}));
