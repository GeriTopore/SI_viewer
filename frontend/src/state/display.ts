import { create } from "zustand";
import { OVERLAY_COLOURS } from "../viewer/colormaps";
import type { DisplayParams, ViewTransform } from "../viewer/renderer";

// Display settings (contrast, colormap) and pan/zoom per view key, so each
// dataset keeps its look when switching back and forth. Image datasets use
// their dataset id as the key, so a map looks the same alone or overlaid.
interface DisplayState {
  display: Record<string, DisplayParams>;
  views: Record<string, ViewTransform>;
  /** Colour of each element's maps (by element symbol, e.g. "Fe"). */
  elementColours: Record<string, string>;
  spectrumLog: boolean;
  setDisplay: (key: string, patch: Partial<DisplayParams>) => void;
  initDisplay: (key: string, params: DisplayParams) => void;
  setView: (key: string, view: ViewTransform) => void;
  /** Give ``element`` the next unused palette colour if it has none. */
  ensureElementColour: (element: string) => string;
  /** Recolour an element and the display settings of its maps. */
  setElementColour: (element: string, colour: string, datasetIds: string[]) => void;
  setSpectrumLog: (log: boolean) => void;
}

export const DEFAULT_COLORMAP = "gray";

// Element colours are remembered in this browser across sessions.
const COLOURS_KEY = "si-viewer.elementColours";

function loadColours(): Record<string, string> {
  try {
    return JSON.parse(localStorage.getItem(COLOURS_KEY) ?? "{}");
  } catch {
    return {};
  }
}

function saveColours(colours: Record<string, string>) {
  try {
    localStorage.setItem(COLOURS_KEY, JSON.stringify(colours));
  } catch {
    /* storage unavailable */
  }
}

export const useDisplay = create<DisplayState>((set, get) => ({
  display: {},
  views: {},
  elementColours: loadColours(),
  spectrumLog: false,
  setDisplay: (key, patch) =>
    set((s) => ({ display: { ...s.display, [key]: { ...s.display[key], ...patch } } })),
  initDisplay: (key, params) =>
    set((s) => (s.display[key] ? s : { display: { ...s.display, [key]: params } })),
  setView: (key, view) => set((s) => ({ views: { ...s.views, [key]: view } })),
  ensureElementColour: (element) => {
    const existing = get().elementColours[element];
    if (existing) return existing;
    const used = new Set(Object.values(get().elementColours));
    const n = Object.keys(get().elementColours).length;
    const colour = OVERLAY_COLOURS.find((c) => !used.has(c)) ?? OVERLAY_COLOURS[n % OVERLAY_COLOURS.length];
    const elementColours = { ...get().elementColours, [element]: colour };
    set({ elementColours });
    saveColours(elementColours);
    return colour;
  },
  setElementColour: (element, colour, datasetIds) => {
    const display = { ...get().display };
    for (const id of datasetIds) if (display[id]) display[id] = { ...display[id], colormap: colour };
    const elementColours = { ...get().elementColours, [element]: colour };
    set({ elementColours, display });
    saveColours(elementColours);
  },
  setSpectrumLog: (spectrumLog) => set({ spectrumLog }),
}));
