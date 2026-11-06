import { create } from "zustand";
import type { DisplayParams, ViewTransform } from "../viewer/renderer";

// Display settings (contrast, colormap) and pan/zoom per view key, so each
// dataset keeps its look when switching back and forth.
interface DisplayState {
  display: Record<string, DisplayParams>;
  views: Record<string, ViewTransform>;
  spectrumLog: boolean;
  setDisplay: (key: string, patch: Partial<DisplayParams>) => void;
  initDisplay: (key: string, params: DisplayParams) => void;
  setView: (key: string, view: ViewTransform) => void;
  setSpectrumLog: (log: boolean) => void;
}

export const DEFAULT_COLORMAP = "gray";

export const useDisplay = create<DisplayState>((set) => ({
  display: {},
  views: {},
  spectrumLog: false,
  setDisplay: (key, patch) =>
    set((s) => ({ display: { ...s.display, [key]: { ...s.display[key], ...patch } } })),
  initDisplay: (key, params) =>
    set((s) => (s.display[key] ? s : { display: { ...s.display, [key]: params } })),
  setView: (key, view) => set((s) => ({ views: { ...s.views, [key]: view } })),
  setSpectrumLog: (spectrumLog) => set({ spectrumLog }),
}));
