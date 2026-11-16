import { create } from "zustand";
import type { LayersPanelProps } from "../components/LayersPanel";

/** The main viewer's layer list, published for the Layers tab (the same
 * list and actions as the panel floating over the image). */
interface LayersTabState {
  owner: string | null;
  panel: LayersPanelProps | null;
  publish: (owner: string, panel: LayersPanelProps) => void;
  clear: (owner: string) => void;
}

export const useLayersTab = create<LayersTabState>((set, get) => ({
  owner: null,
  panel: null,
  publish: (owner, panel) => set({ owner, panel }),
  clear: (owner) => get().owner === owner && set({ owner: null, panel: null }),
}));
