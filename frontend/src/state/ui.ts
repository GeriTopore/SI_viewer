import { create } from "zustand";

export type ListMode = "list" | "grid";

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
  listMode: ListMode;
  lastDir: string;
  browserOpen: boolean;
  setListMode: (mode: ListMode) => void;
  setLastDir: (dir: string) => void;
  setBrowserOpen: (open: boolean) => void;
}

export const useUi = create<UiState>((set) => ({
  listMode: load<ListMode>("si-viewer.listMode", "list"),
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
