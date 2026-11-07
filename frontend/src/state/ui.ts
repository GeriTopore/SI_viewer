import { create } from "zustand";

export type ListMode = "list" | "grid";
export type Theme = "light" | "dark";
export type RightTab = "info" | "quantify" | "process";

// Per-browser conveniences, remembered across reloads when storage is available.
function load<T extends string>(key: string, fallback: T): T {
  try {
    return (localStorage.getItem(key) as T | null) ?? fallback;
  } catch {
    return fallback;
  }
}

function save(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* storage unavailable */
  }
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
  /** FFT of the viewer's image, above the tabs. */
  fftOpen: boolean;
  setFftOpen: (open: boolean) => void;
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
  fftOpen: load<string>("si-viewer.fftOpen", "true") === "true",
  setFftOpen: (fftOpen) => {
    save("si-viewer.fftOpen", String(fftOpen));
    set({ fftOpen });
  },
  setRightTab: (rightTab) => {
    save("si-viewer.rightTab", rightTab);
    set({ rightTab });
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
