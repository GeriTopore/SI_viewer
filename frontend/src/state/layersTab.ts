import { create } from "zustand";
import type { LayersPanelProps } from "../components/LayersPanel";

/** The sections of a view whose panes each publish their layers (4D-STEM:
 * the diffraction pattern's, the virtual image's), in the tab's order. */
export const SECTION_ORDER = ["Diffraction pattern", "Virtual image"];

/** The main viewer's layer list, published for the Layers tab (the same
 * list and actions as the panel floating over the image); or, for a view of
 * several panes, a list per pane, each under its section's title. */
interface LayersTabState {
  owner: string | null;
  panel: LayersPanelProps | null;
  publish: (owner: string, panel: LayersPanelProps) => void;
  clear: (owner: string) => void;
  sections: Record<string, { owner: string; panel: LayersPanelProps }>;
  publishSection: (section: string, owner: string, panel: LayersPanelProps) => void;
  clearSection: (section: string, owner: string) => void;
}

export const useLayersTab = create<LayersTabState>((set, get) => ({
  owner: null,
  panel: null,
  publish: (owner, panel) => set({ owner, panel }),
  clear: (owner) => get().owner === owner && set({ owner: null, panel: null }),
  sections: {},
  publishSection: (section, owner, panel) => set((s) => ({ sections: { ...s.sections, [section]: { owner, panel } } })),
  clearSection: (section, owner) =>
    get().sections[section]?.owner === owner &&
    set((s) => {
      const sections = { ...s.sections };
      delete sections[section];
      return { sections };
    }),
}));
