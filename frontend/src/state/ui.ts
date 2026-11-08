import { create } from "zustand";

export type ListMode = "list" | "grid";
export type Theme = "light" | "dark";
export type RightTab = "info" | "quantify" | "process";
/** How several images are shown: one composite, or a panel each. */
export type GridMode = "merged" | "grid";
/** Kinds of view with their own mode: a spectrum image (with maps), plain images. */
export type GridContext = "si" | "images";
export type BottomTab = "spectrum" | "low-loss" | "core-loss" | "profile" | "alignment";

// Pane sizes: default and limits (side panes in px, the viewer's bottom pane
// as a fraction of the viewer's height).
export const PANE_SIZES = {
  left: { initial: 280, min: 180, max: 640 },
  right: { initial: 330, min: 260, max: 720 },
  bottom: { initial: 0.42, min: 0.15, max: 0.85 },
  // The alignment's row of compared cross-correlations, as a fraction of the step.
  compare: { initial: 0.38, min: 0.12, max: 0.8 },
};

const clamp = (v: number, { min, max }: { min: number; max: number }) => Math.min(max, Math.max(min, v));

// Per-browser conveniences, remembered across reloads when storage is available.
function load<T extends string>(key: string, fallback: T): T {
  try {
    return (localStorage.getItem(key) as T | null) ?? fallback;
  } catch {
    return fallback;
  }
}

function loadNumber(key: string, limits: { initial: number; min: number; max: number }): number {
  const v = Number(load(key, ""));
  return Number.isFinite(v) && v > 0 ? clamp(v, limits) : limits.initial;
}

function save(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* storage unavailable */
  }
}

function loadGridMode(context: GridContext): GridMode | null {
  const v = load<string>(`si-viewer.gridMode.${context}`, "auto");
  return v === "merged" || v === "grid" ? v : null;
}

interface UiState {
  /** Set on <html data-theme> by index.html before first paint. */
  theme: Theme;
  setTheme: (theme: Theme) => void;
  listMode: ListMode;
  /** Tab shown in the right-hand column. */
  rightTab: RightTab;
  setRightTab: (tab: RightTab) => void;
  /** Side panes folded to their edges, leaving the viewer. */
  leftCollapsed: boolean;
  rightCollapsed: boolean;
  setCollapsed: (side: "left" | "right", collapsed: boolean) => void;
  /** Widths of the side panes (px). */
  leftWidth: number;
  rightWidth: number;
  setPaneWidth: (side: "left" | "right", width: number) => void;
  /** Height of the viewer's bottom pane (spectrum / line profile), as a
   * fraction of the viewer. */
  bottomFraction: number;
  setBottomFraction: (f: number) => void;
  compareFraction: number;
  setCompareFraction: (f: number) => void;
  bottomTab: BottomTab;
  setBottomTab: (tab: BottomTab) => void;
  /** FFT of the viewer's image, above the tabs. */
  fftOpen: boolean;
  setFftOpen: (open: boolean) => void;
  /** Clicking the FFT marks a spatial frequency (and shows the mark). */
  fftMeasure: boolean;
  setFftMeasure: (on: boolean) => void;
  /** Merged/Grid choice per kind of view; null follows the content
   * (spectrum image: merged, images: grid). */
  gridMode: Record<GridContext, GridMode | null>;
  setGridMode: (context: GridContext, mode: GridMode | null) => void;
  lastDir: string;
  browserOpen: boolean;
  setListMode: (mode: ListMode) => void;
  setLastDir: (dir: string) => void;
  setBrowserOpen: (open: boolean) => void;
}

export const useUi = create<UiState>((set) => ({
  theme: document.documentElement.dataset.theme === "dark" ? "dark" : "light",
  setTheme: (theme) => {
    document.documentElement.dataset.theme = theme;
    save("si-viewer.theme", theme);
    set({ theme });
  },
  listMode: load<ListMode>("si-viewer.listMode", "list"),
  rightTab: load<RightTab>("si-viewer.rightTab", "info"),
  leftCollapsed: load<string>("si-viewer.leftCollapsed", "false") === "true",
  rightCollapsed: load<string>("si-viewer.rightCollapsed", "false") === "true",
  setCollapsed: (side, collapsed) => {
    save(`si-viewer.${side}Collapsed`, String(collapsed));
    set(side === "left" ? { leftCollapsed: collapsed } : { rightCollapsed: collapsed });
  },
  leftWidth: loadNumber("si-viewer.leftWidth", PANE_SIZES.left),
  rightWidth: loadNumber("si-viewer.rightWidth", PANE_SIZES.right),
  setPaneWidth: (side, width) => {
    const w = Math.round(clamp(width, PANE_SIZES[side]));
    save(`si-viewer.${side}Width`, String(w));
    set(side === "left" ? { leftWidth: w } : { rightWidth: w });
  },
  bottomFraction: loadNumber("si-viewer.bottomFraction", PANE_SIZES.bottom),
  setBottomFraction: (f) => {
    const bottomFraction = clamp(f, PANE_SIZES.bottom);
    save("si-viewer.bottomFraction", bottomFraction.toFixed(3));
    set({ bottomFraction });
  },
  compareFraction: loadNumber("si-viewer.compareFraction", PANE_SIZES.compare),
  setCompareFraction: (f) => {
    const compareFraction = clamp(f, PANE_SIZES.compare);
    save("si-viewer.compareFraction", compareFraction.toFixed(3));
    set({ compareFraction });
  },
  bottomTab: load<BottomTab>("si-viewer.bottomTab", "spectrum"),
  setBottomTab: (bottomTab) => {
    save("si-viewer.bottomTab", bottomTab);
    set({ bottomTab });
  },
  fftOpen: load<string>("si-viewer.fftOpen", "true") === "true",
  fftMeasure: load<string>("si-viewer.fftMeasure", "true") === "true",
  setFftMeasure: (fftMeasure) => {
    save("si-viewer.fftMeasure", String(fftMeasure));
    set({ fftMeasure });
  },
  setFftOpen: (fftOpen) => {
    save("si-viewer.fftOpen", String(fftOpen));
    set({ fftOpen });
  },
  setRightTab: (rightTab) => {
    save("si-viewer.rightTab", rightTab);
    set({ rightTab });
  },
  gridMode: { si: loadGridMode("si"), images: loadGridMode("images") },
  setGridMode: (context, mode) => {
    save(`si-viewer.gridMode.${context}`, mode ?? "auto");
    set((s) => ({ gridMode: { ...s.gridMode, [context]: mode } }));
  },
  lastDir: load("si-viewer.lastDir", ""),
  browserOpen: false,
  setListMode: (listMode) => {
    save("si-viewer.listMode", listMode);
    set({ listMode });
  },
  setLastDir: (lastDir) => {
    save("si-viewer.lastDir", lastDir);
    set({ lastDir });
  },
  setBrowserOpen: (browserOpen) => set({ browserOpen }),
}));
