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
  /** Linear/Log of the low-loss spectra of a linked group (linear by default). */
  lowLossLog: boolean;
  /** Linear/Log of a line-scan image, by dataset id (unset: the view's default). */
  imageLog: Record<string, boolean>;
  setDisplay: (key: string, patch: Partial<DisplayParams>) => void;
  /** Forget the display settings and pan/zoom of datasets whose data changed
   * (keys equal to an id or starting with "id:"), so they start afresh. */
  resetDisplay: (ids: string[]) => void;
  initDisplay: (key: string, params: DisplayParams) => void;
  setView: (key: string, view: ViewTransform) => void;
  /** Colour of each window map (by dataset id), kept for the session. */
  windowColours: Record<string, string>;
  /** Give a window map its own solid colour if it has none. */
  ensureWindowColour: (id: string) => string;
  /** The colour the next window map will get. */
  nextWindowColour: () => string;
  /** Give ``element`` the next unused palette colour if it has none. */
  ensureElementColour: (element: string) => string;
  /** Recolour an element and the display settings of its maps. */
  setElementColour: (element: string, colour: string, datasetIds: string[]) => void;
  setSpectrumLog: (log: boolean) => void;
  setLowLossLog: (log: boolean) => void;
  setImageLog: (id: string, log: boolean) => void;
}

export const DEFAULT_COLORMAP = "gray";

// Window maps get these, none of them an element colour.
const WINDOW_COLOURS = ["#2ec4b6", "#e0a96d", "#b8b8ff", "#e07a5f", "#81b29a", "#d4c26a", "#c77dff", "#7ae582"];

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
  windowColours: {},
  nextWindowColour: () => {
    const used = new Set(Object.values(get().windowColours));
    const n = Object.keys(get().windowColours).length;
    return WINDOW_COLOURS.find((c) => !used.has(c)) ?? WINDOW_COLOURS[n % WINDOW_COLOURS.length];
  },
  ensureWindowColour: (id) => {
    const existing = get().windowColours[id];
    if (existing) return existing;
    const colour = get().nextWindowColour();
    set({ windowColours: { ...get().windowColours, [id]: colour } });
    return colour;
  },
  spectrumLog: false,
  lowLossLog: false,
  imageLog: {},
  setDisplay: (key, patch) =>
    set((s) => ({ display: { ...s.display, [key]: { ...s.display[key], ...patch } } })),
  resetDisplay: (ids) =>
    set((s) => {
      const stale = (k: string) => ids.some((id) => k === id || k.startsWith(`${id}:`));
      const display = Object.fromEntries(Object.entries(s.display).filter(([k]) => !stale(k)));
      const views = Object.fromEntries(Object.entries(s.views).filter(([k]) => !stale(k)));
      return { display, views };
    }),
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
  setLowLossLog: (lowLossLog) => set({ lowLossLog }),
  setImageLog: (id, log) => set((s) => ({ imageLog: { ...s.imageLog, [id]: log } })),
}));
